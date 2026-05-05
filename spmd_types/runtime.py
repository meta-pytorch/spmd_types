# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Lightweight runtime API for SPMD type annotations.

This module provides the core functions that model code (non-test,
non-checker) needs to annotate tensors with SPMD types:

- ``assert_type`` / ``assert_local_type`` / ``mutate_type`` -- annotation APIs
- ``register_autograd_function`` / ``register_local_autograd_function`` /
  ``register_decomposition`` -- autograd.Function registration decorators
- ``has_local_type`` / ``get_partition_spec`` -- queries
- ``trace`` -- logging

Crucially, this module does NOT depend on ``_checker.py`` (the type
inference engine and TorchFunctionMode).  This means changes to the
heavy checker logic do not invalidate downstream targets that only
need the annotation APIs.
"""

from __future__ import annotations

import builtins
import logging
import os
from contextlib import contextmanager
from typing import Any, TypeAlias

import torch
from spmd_types._frame import _get_user_frame
from spmd_types._mesh_axis import MeshAxis
from spmd_types._scalar_sentinel import _Scalar
from spmd_types._state import current_mesh, is_type_checking, no_typecheck
from spmd_types._traceback import api_boundary
from spmd_types._type_attr import (
    _LOCAL_TYPE_ATTR,
    get_local_type,
    set_local_type as _set_local_type_raw,
)
from spmd_types.types import (
    _canonicalize_shard,
    _check_orthogonality,
    DeviceMeshAxis,
    format_axis,
    LocalSpmdType,
    normalize_axis,
    normalize_local_type,
    P,
    PartitionSpec,
    PerMeshAxisLocalSpmdType,
    PerMeshAxisSpmdType,
    PerMeshAxisSpmdTypes,
    Shard,
    shard_types_to_partition_spec,
    SpmdTypeError,
    to_local_type,
    V,
)
from torch.distributed.tensor import DTensor

# =============================================================================
# Tensor type queries
# =============================================================================


def has_local_type(tensor: torch.Tensor) -> bool:
    """Return True if the tensor has an SPMD type annotation.

    Distinguishes between truly untyped tensors (no attribute) and factory
    tensors (attribute set to ``{}``).
    """
    return hasattr(tensor, _LOCAL_TYPE_ATTR)


# =============================================================================
# Trace mode: set SPMD_TYPES_TRACE=1 to log every non-trivial operator.
# =============================================================================

_trace_logger = logging.getLogger(__name__ + ".trace")
_TRACE = os.environ.get("SPMD_TYPES_TRACE", "") == "1"


@contextmanager
def trace(enabled: bool = True):
    """Context manager to enable or disable SPMD type trace logging.

    When enabled, every non-trivial tensor operator logs its name, input
    SPMD types, output SPMD type, and the user-code callsite to the
    ``spmd_types._checker.trace`` logger at INFO level.

    Example::

        import logging
        logging.basicConfig()
        with typecheck(), trace():
            z = x + y  # logs: my_file.py:42  add({dp: R}, {dp: V}) -> {dp: V}

    Can also be used to temporarily suppress tracing when the envvar
    ``SPMD_TYPES_TRACE=1`` is set::

        with trace(enabled=False):
            ...  # no trace output
    """
    global _TRACE
    old = _TRACE
    _TRACE = enabled
    try:
        yield
    finally:
        _TRACE = old


def _format_type(t: object) -> str:
    """Format a per-mesh-axis type dict for trace output."""
    if isinstance(t, dict):
        if not t:
            return "{}"
        parts = []
        for k, v in t.items():
            # Use the short name (R, I, V, P) instead of the full enum repr.
            v_str = v.name if isinstance(v, PerMeshAxisLocalSpmdType) else repr(v)
            parts.append(f"{k}: {v_str}")
        return "{" + ", ".join(parts) + "}"
    return repr(t)


def _trace_op(
    func: object,
    input_types: list[LocalSpmdType],
    output_type: LocalSpmdType | None,
) -> None:
    """Log a trace line for a non-trivial tensor operator.

    Omits the log line when all inputs and the output are empty local types
    (``{}``), since these carry no SPMD information and would just be noise.
    """
    # Skip when every type is an empty dict -- no mesh axes involved.
    if (
        all(isinstance(t, dict) and not t for t in input_types)
        and isinstance(output_type, dict)
        and not output_type
    ):
        return
    name = getattr(func, "__name__", None) or getattr(func, "__qualname__", repr(func))
    inputs_str = ", ".join(_format_type(t) for t in input_types)
    out_str = _format_type(output_type)
    loc = _get_user_frame()
    _trace_logger.info("%s  %s(%s) -> %s", loc, name, inputs_str, out_str)


# =============================================================================
# Validation and type setting
# =============================================================================


def _validate(type: LocalSpmdType) -> LocalSpmdType:
    """Validate a LocalSpmdType and normalize axis keys.

    Extends ``normalize_local_type`` with additional checks for Shard
    sentinels that must not be stored on tensors.

    Args:
        type: A LocalSpmdType dict mapping mesh axes to per-axis SPMD types.

    Raises:
        TypeError: If any value is not a PerMeshAxisLocalSpmdType (R, I, V, or P),
            or is an internal sentinel type.
    """
    for axis, typ in type.items():
        if not isinstance(typ, PerMeshAxisLocalSpmdType):
            if typ is _Scalar:
                raise TypeError(
                    f"_Scalar sentinel on axis {format_axis(axis)} must not be stored "
                    f"on a tensor. _Scalar is internal to type inference; it should "
                    f"be filtered out by infer_output_type before reaching a tensor."
                )
            if isinstance(typ, Shard):
                raise TypeError(
                    f"Shard type {typ!r} on axis {format_axis(axis)} cannot be stored "
                    f"as a local SPMD type. Shard is only valid as src/dst in "
                    f"collective operations. Use V instead for local type tracking."
                )
            # Fall through to normalize_local_type for the generic error.
    return normalize_local_type(type)


def _set_local_type(tensor: torch.Tensor, type: LocalSpmdType) -> torch.Tensor:
    """Set SPMD type on a tensor (internal). Validates and returns tensor."""
    return _set_local_type_raw(tensor, _validate(type))


# =============================================================================
# Partition spec helpers
# =============================================================================

# Attribute name for storing the PartitionSpec on tensors (global SPMD only).
# This is the single source of truth for which mesh axes shard which tensor dims.
_PARTITION_SPEC_ATTR = "_partition_spec"


def get_partition_spec(tensor: torch.Tensor) -> PartitionSpec | None:
    """Get the PartitionSpec stored on a tensor (global SPMD only).

    Returns None if no global shard info is present. The PartitionSpec
    is the single source of truth for which mesh axes shard which tensor
    dims in global SPMD.

    Args:
        tensor: The tensor to retrieve the PartitionSpec from.
    """
    return getattr(tensor, _PARTITION_SPEC_ATTR, None)


def _set_partition_spec(tensor: torch.Tensor, spec: PartitionSpec | None) -> None:
    """Set or clear the partition spec on a single tensor."""
    if spec is not None:
        setattr(tensor, _PARTITION_SPEC_ATTR, spec)
    elif hasattr(tensor, _PARTITION_SPEC_ATTR):
        delattr(tensor, _PARTITION_SPEC_ATTR)


def _update_axis_in_partition_spec(  # noqa: C901
    spec: PartitionSpec | None,
    axis_to_remove: DeviceMeshAxis,
    axis_to_insert: Shard | None,
    ndim: int,
) -> PartitionSpec:
    """Replace a single axis's shard entry in a PartitionSpec.

    Removes ``axis_to_remove`` from the spec, then if ``axis_to_insert`` is a
    Shard, inserts it at the appropriate dim. For multi-axis entries like
    ``('tp', 'dp')``, only the target axis is stripped; remaining axes are kept.

    The caller must ensure ``axis_to_remove`` is the innermost (last) axis in
    any multi-axis group it belongs to; an assertion guards this invariant.
    """
    if spec is not None:
        for entry in spec:
            if isinstance(entry, tuple) and axis_to_remove in entry:
                assert entry[-1] == axis_to_remove, (
                    f"axis_to_remove {axis_to_remove} is not the innermost axis "
                    f"in its multi-axis group {entry}"
                )
    if spec is None:
        entries: list = [None] * ndim
        if isinstance(axis_to_insert, Shard):
            entries[axis_to_insert.dim] = axis_to_remove
        return PartitionSpec(*entries)
    new_entries: list = []
    for entry in spec:
        if entry is None:
            new_entries.append(None)
        elif isinstance(entry, tuple):
            filtered = tuple(a for a in entry if a != axis_to_remove)
            if not filtered:
                new_entries.append(None)
            elif len(filtered) == 1:
                new_entries.append(filtered[0])
            else:
                new_entries.append(filtered)
        elif entry == axis_to_remove:
            new_entries.append(None)
        else:
            new_entries.append(entry)
    if isinstance(axis_to_insert, Shard):
        assert axis_to_insert.dim < len(new_entries)
        existing = new_entries[axis_to_insert.dim]
        if existing is None:
            new_entries[axis_to_insert.dim] = axis_to_remove
        elif isinstance(existing, tuple):
            new_entries[axis_to_insert.dim] = (*existing, axis_to_remove)
        else:
            new_entries[axis_to_insert.dim] = (existing, axis_to_remove)
    return PartitionSpec(*new_entries)


# =============================================================================
# assert_type
# =============================================================================


_TensorOrSequence = torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]


@api_boundary
def assert_type(  # noqa: C901
    tensor: _TensorOrSequence,
    type: PerMeshAxisSpmdTypes | PerMeshAxisLocalSpmdType,
    partition_spec: PartitionSpec | None = None,
) -> _TensorOrSequence:
    """Assert or set the SPMD type on a tensor or sequence of tensors.

    If the tensor has no SPMD type, sets it. If the tensor already has an SPMD
    type, checks compatibility using refinement semantics (see below).

    When ``tensor`` is a list or tuple, applies ``assert_type`` to each element
    and returns a collection of the same type.

    Three calling conventions (see overloads):

    1. ``assert_type(tensor, {axis: R/I/V/P, ...}, partition_spec=...)``
       Explicit local types with optional PartitionSpec for shard metadata.

    2. ``assert_type(tensor, {axis: S(i), ...})`` S(i) entries are automatically
       converted to V + PartitionSpec. Cannot be combined with an explicit
       ``partition_spec``.

    3. ``assert_type(tensor, R)`` (bare PerMeshAxisLocalSpmdType)
       Expands to ``{axis: R for axis in current_mesh()}``. Raises
       ``SpmdTypeError`` if no current mesh is set.

    S(i) always stores a PartitionSpec regardless of whether the axis is in
    global SPMD mode. Global axes only affect whether S(i) propagates through
    ops (via DTensor); storage is unconditional.

    Refinement semantics (re-check on already-typed tensors):

    When called on a tensor that already has SPMD types, ``assert_type`` checks
    consistency rather than overwriting. For local types (R/I/P), the existing
    and new values must match exactly. For shard metadata, S(i) is a refinement
    of V -- it adds information about which tensor dimension is sharded without
    changing the local type (both are V locally). A re-check may add new shard
    info but must not contradict existing info.

    Worked examples:

    - V then S(i): OK, stores the shard info (refinement).
    - S(i) then V: OK, keeps existing shard info (V is less specific).
    - S(i) then S(i): OK (consistent).
    - S(i) then S(j) on same axis: SpmdTypeError (contradicts).
    - {tp: S(0)} then {dp: S(1)}: OK, merges to PartitionSpec(tp, dp).
    - {tp: S(0)} then {dp: S(0)}: SpmdTypeError (multi-axis same dim requires
      explicit PartitionSpec, e.g. PartitionSpec((tp, dp), None)).
    - {dp: S(0)} then PartitionSpec(dp, None): OK (consistent).
    - {dp: S(0)} then PartitionSpec(dp, tp): OK, adds new shard info.
    - {dp: S(0)} then PartitionSpec((dp, tp), None): SpmdTypeError, touching
      existing S(0) ordering info.
    - PartitionSpec(dp, None) then PartitionSpec(dp, tp): OK, adds new
      shard info for axis tp.
    - PartitionSpec(dp, None) then PartitionSpec((dp, tp), None): SpmdTypeError,
      touching existing S(0) ordering info.
    - PartitionSpec((dp, tp), None) then {dp: S(0)}: SpmdTypeError,
      unresolved order on S(0).
    - PartitionSpec ordering matters: (tp, dp) != (dp, tp).

    Args:
        tensor: The tensor to assert or set SPMD type on. type: A dict mapping
        mesh axes to per-axis SPMD types.
            Accepts R, I, V, P, or S(i). S(i) entries are syntax sugar for
            setting V on the axis and storing a PartitionSpec that maps tensor
            dim ``i`` to that mesh axis.
        partition_spec: Optional PartitionSpec describing how tensor
            dimensions map to mesh axes for Varying dimensions. Mutually
            exclusive with S(i) entries in ``type``.

    Raises:
        SpmdTypeError: If S(i) and partition_spec are both provided,
            if partition_spec length doesn't match tensor ndim, if a type dict
            axis conflicts with partition_spec, or if a re-check has conflicting
            PartitionSpec info.
        SpmdTypeError: If existing local SPMD type doesn't match.
    """
    if isinstance(tensor, (list, tuple)):
        result = [assert_type(t, type, partition_spec) for t in tensor]
        return builtins.type(tensor)(result)

    if isinstance(tensor, DTensor):
        raise TypeError(
            "assert_type() does not support DTensor. SPMD type checking "
            "operates on local tensors only; DTensor tracks its own "
            "placement metadata."
        )

    ############ Expand bare PerMeshAxisLocalSpmdType ############
    if isinstance(type, PerMeshAxisLocalSpmdType):
        mesh = current_mesh()
        if mesh is None:
            raise SpmdTypeError(
                f"assert_type(tensor, {type}) requires an active mesh, "
                "but no current mesh is set. Use set_mesh() or pass an "
                "explicit per-axis dict instead."
            )
        type = {axis: type for axis in mesh}

    ############ Validate partition_spec length ############
    if partition_spec is not None:
        if len(partition_spec) != tensor.ndim:
            raise SpmdTypeError(
                f"PartitionSpec length {len(partition_spec)} doesn't match "
                f"tensor ndim {tensor.ndim}"
            )

    ############ Build canonical LocalSpmdType + auto PartitionSpec ############
    # Single pass: normalize S(i) dims, separate into local types vs shards,
    # and check for multi-axis-same-dim conflicts.
    local_type: LocalSpmdType = {}
    axis_to_dims: dict[MeshAxis, Shard] = {}
    dim_to_axes: dict[int, list[MeshAxis]] = {}
    for axis, typ in type.items():
        axis = normalize_axis(axis)
        if axis.size() == 1:
            continue  # singleton axes carry no sharding info; skip
        typ = _canonicalize_shard(typ, tensor.ndim)
        if isinstance(typ, Shard):
            dim_to_axes.setdefault(typ.dim, []).append(axis)
            axis_to_dims[axis] = typ
            local_type[axis] = V
        else:
            local_type[axis] = typ

    # Enforce overload contract: S(i) and partition_spec are mutually exclusive.
    if axis_to_dims and partition_spec is not None:
        raise SpmdTypeError(
            "Cannot use S(i) in type dict and partition_spec at the same time. "
            "Use either S(i) in the type dict or an explicit PartitionSpec."
        )

    # Convert S(i) entries to PartitionSpec.
    if axis_to_dims:
        # Forbid sharding the same tensor dim on multiple mesh axes without
        # an explicit PartitionSpec (which specifies the axis ordering).
        for dim, axes in dim_to_axes.items():
            if len(axes) > 1:
                names = ", ".join(format_axis(a) for a in axes)
                raise SpmdTypeError(
                    f"Tensor dim {dim} is sharded on multiple axes "
                    f"({names}). Use an explicit PartitionSpec "
                    f"to specify the axis ordering."
                )
        partition_spec = shard_types_to_partition_spec(axis_to_dims, tensor.ndim)

    ############ Fill V for axes in partition_spec ############
    if partition_spec is not None:
        for entry in partition_spec:
            if entry is None:
                continue
            axes = entry if isinstance(entry, tuple) else (entry,)
            for axis in axes:
                axis_type = local_type.get(axis)
                if axis_type is not None and axis_type is not V:
                    raise SpmdTypeError(
                        f"Mesh axis {format_axis(axis)} appears in "
                        f"partition_spec (implying Varying/Shard) but is "
                        f"specified as {axis_type} in type."
                    )
                local_type[axis] = V

    _validate(local_type)

    ############ Set or check ############
    if not has_local_type(tensor):
        _set_local_type(tensor, local_type)
        _set_partition_spec(tensor, partition_spec)
        if _TRACE:
            _trace_op(assert_type, [{}], local_type)
        return tensor

    # 1. Re-check: compare local types.
    # Only axes present in local_type are checked; extra axes in existing are
    # ignored (partial re-check). New axes are merged in.
    #
    # Validate before mutating so that failures are side-effect-free.
    existing_local = get_local_type(tensor)  # existing_local is mutable.
    old = dict(existing_local) if _TRACE else None
    for axis, typ in local_type.items():
        if axis in existing_local:
            existing_typ = existing_local[axis]
            if existing_typ != typ:
                # Allow implicit transmute: V -> P.  Both are "varying per
                # rank" locally; P just adds the semantic that the values are
                # partial sums awaiting reduction.  Silently overwrite.
                if existing_typ is V and typ is P:
                    existing_local[axis] = typ
                    continue
                raise SpmdTypeError(
                    f"SPMD type mismatch on axis {format_axis(axis)}: "
                    f"tensor has {existing_typ}, expected {typ}"
                )
    # 1a. Orthogonality: check the union of old and new keys before committing.
    _check_orthogonality(list(existing_local.keys() | local_type.keys()))
    # Commit: merge new axes into the live dict.
    existing_local.update(local_type)

    # 2. Re-check PartitionSpec: refinement semantics.
    # Only None -> sharded refinement is allowed; adding a new mesh axis to an
    # already-sharded dim requires an explicit PartitionSpec upfront.
    existing_spec = getattr(tensor, _PARTITION_SPEC_ATTR, None)
    if partition_spec is not None:
        if existing_spec is not None:
            # Build map from mesh axis to tensor dim for conflict detection.
            existing_axis_to_dim: dict[MeshAxis, int] = {}
            for dim, entry in enumerate(existing_spec):
                if entry is not None:
                    axes = entry if isinstance(entry, tuple) else (entry,)
                    for a in axes:
                        existing_axis_to_dim[a] = dim

            merged_entries = list(existing_spec)
            for dim, (ex_spec, new_spec) in enumerate(
                zip(existing_spec, partition_spec)
            ):
                if new_spec is None:
                    continue  # New spec doesn't constrain this dim.
                if ex_spec is None:
                    # Check new axis isn't already at another dim.
                    new_axes = new_spec if isinstance(new_spec, tuple) else (new_spec,)
                    for a in new_axes:
                        if a in existing_axis_to_dim:
                            raise SpmdTypeError(
                                f"PartitionSpec conflict: axis "
                                f"{format_axis(a)} already shards dim "
                                f"{existing_axis_to_dim[a]},"
                                f" cannot also shard dim {dim}"
                            )
                    merged_entries[dim] = new_spec
                    continue
                if ex_spec != new_spec:
                    raise SpmdTypeError(
                        f"PartitionSpec conflict at dim {dim}: "
                        f"tensor has {ex_spec!r}, new assert has {new_spec!r}"
                    )
            setattr(tensor, _PARTITION_SPEC_ATTR, PartitionSpec(*merged_entries))
        else:
            # Refinement: V -> S(i). Store the new spec.
            setattr(tensor, _PARTITION_SPEC_ATTR, partition_spec)
    if _TRACE:
        _trace_op(assert_type, [old], existing_local)
    return tensor


def assert_local_type(tensor: torch.Tensor, type: PerMeshAxisSpmdTypes) -> torch.Tensor:
    """Deprecated: use ``assert_type`` instead."""
    return assert_type(tensor, type)


def assert_type_like(
    tensor: torch.Tensor,
    source: torch.Tensor,
    overrides: dict[MeshAxis, LocalSpmdType] | None = None,
) -> None:
    """Assert that *tensor* has the same SPMD type as *source*, with overrides.

    Analogous to ``torch.ones_like``: copies the type from *source* then
    applies *overrides* on top.  Useful for collective outputs where only
    one axis changes::

        assert_type_like(out, x, {mesh.CP: R})
    """
    full_type = {**get_local_type(source), **(overrides or {})}
    assert_type(tensor, full_type)


# =============================================================================
# mutate_type
# =============================================================================


@api_boundary
def mutate_type(
    tensor: torch.Tensor,
    axis: DeviceMeshAxis,
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
) -> torch.Tensor:
    """Change the SPMD type of a single mesh axis on an already-annotated tensor.

    Unlike ``assert_type``, this function *overwrites* the existing type on
    ``axis``.  The caller must specify the expected current type (``src``) to
    prevent silent corruption; a ``SpmdTypeError`` is raised if the tensor's
    current type on ``axis`` does not match ``src``.

    This is intended for internal use in low-level parallelism primitives
    where a buffer legitimately changes its distribution semantics in place
    (e.g. an all-gather output buffer that transitions from S(0) to R).

    With D95084345 (raw dist type rules), most collectives can be type-checked
    automatically. mutate_type is still needed for updating spmd types on
    *views* into a buffer after an in-place collective has changed the
    buffer's semantics.

    Args:
        tensor: The tensor whose type to mutate.  Must already have a local
            type annotation.
        axis: The mesh axis (MeshAxis or ProcessGroup) to modify.
        src: The expected current type on ``axis``.  Raises if it does not
            match.  Accepts R, I, V, P, or S(i) (S(i) is compared as V).
        dst: The new type to set on ``axis``.  Accepts R, I, V, P, or S(i)
            (S(i) is stored as V).

    Returns:
        The tensor for chaining.

    Raises:
        SpmdTypeError: If the tensor has no type, the axis is missing,
            or the current type does not match ``src``.
    """
    axis = normalize_axis(axis)
    if axis.size() == 1:
        return tensor  # singleton axes are not tracked
    local_type = get_local_type(tensor)
    if axis not in local_type:
        raise SpmdTypeError(
            f"mutate_type: axis {format_axis(axis)} not found in tensor's "
            f"SPMD type. Tensor has axes: {set(local_type.keys())}"
        )

    # Normalize S(i) -> V for comparison and storage
    src_local = to_local_type(src)
    dst_local = to_local_type(dst)

    current = local_type[axis]
    if current is not src_local:
        raise SpmdTypeError(
            f"mutate_type: expected current type {src_local} on axis "
            f"{format_axis(axis)}, got {current}"
        )

    new_type = dict(local_type)
    new_type[axis] = dst_local
    return _set_local_type(tensor, new_type)


# =============================================================================
# Autograd function registration
# =============================================================================

# Sets of autograd.Function subclasses registered for type checking.
# _LOCAL_AUTOGRAD_FUNCTIONS: element-wise / local-only functions.
# _TYPECHECK_AUTOGRAD_FUNCTIONS: functions with custom typecheck_forward.
_LOCAL_AUTOGRAD_FUNCTIONS: set[type] = set()
_TYPECHECK_AUTOGRAD_FUNCTIONS: set[type] = set()


def register_local_autograd_function(cls: type) -> type:
    """Register an autograd.Function subclass as local-only for SPMD type checking.

    Local-only means the function's forward operates element-wise (or more
    generally, does not rearrange data across the tensor in a way that would
    change its sharding type).  It must NOT perform any collectives or
    cross-device communication.  For functions that do, use
    :func:`register_autograd_function` with a ``typecheck_forward`` method instead.

    Registered functions get the standard local type propagation rule when
    type checking is active:

    - Inputs may freely mix R and V types; the output is R unless any input
      is V, in which case it is V.
    - All-I inputs produce I outputs.
    - R/V and I cannot be mixed.
    - P is forbidden.

    Unregistered autograd functions that reach the type checker will leave
    their outputs untyped (or raise in strict mode), since the checker
    cannot know whether the function is safe for automatic type propagation.

    Can be used as a decorator::

        @register_local_autograd_function
        class MyOp(torch.autograd.Function):
            ...
    """
    _LOCAL_AUTOGRAD_FUNCTIONS.add(cls)
    return cls


def register_autograd_function(cls: type) -> type:
    """Register an autograd.Function subclass with a custom typecheck method.

    Use this for autograd functions that perform collectives or have
    non-trivial type transformations where the default local-only rule
    would produce incorrect output types.

    The class must define a ``typecheck_forward`` staticmethod that receives
    the same positional/keyword arguments as ``.apply()``.  Inside, it should
    call ``assert_type`` on inputs, call ``.apply()`` to run the function,
    then call ``assert_type`` on the output.  This is symmetric with
    ``typecheck_forward`` on ``nn.Module`` subclasses in llama4x::

        @register_autograd_function
        class MyCollectiveOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x + y

            @staticmethod
            def typecheck_forward(x, y):
                assert_type(x, {pg: S(-1)})
                out = MyCollectiveOp.apply(x, y)
                assert_type(out, {pg: R})
                return out

            @staticmethod
            def backward(ctx, g):
                return g, g
    """
    if not callable(getattr(cls, "typecheck_forward", None)):
        raise TypeError(
            f"{cls.__name__} must define a typecheck_forward staticmethod "
            f"when using @register_autograd_function"
        )
    _TYPECHECK_AUTOGRAD_FUNCTIONS.add(cls)
    return cls


def register_decomposition(
    cls: type,
    ref_impl: "Callable[..., object] | None" = None,  # noqa: F821
) -> "type | Callable[..., object]":  # noqa: F821
    """Register an autograd.Function whose SPMD types are derived from a
    pure-PyTorch reference implementation.

    Fused custom kernels (e.g. Triton) that are wrapped as a single
    ``torch.autograd.Function`` are opaque to the SPMD type checker.  Rather
    than hand-writing a ``typecheck_forward`` that re-validates every input
    and stamps every output, register the equivalent pure-PyTorch
    decomposition.  The checker traces the decomposition -- each primitive
    aten op contributes its own SPMD rule -- and the inferred types are
    copied onto the real fused output at runtime.

    The fused kernel still runs for the actual forward.  The decomposition
    runs only for its types; it is executed under the active ``typecheck()``
    mode but its computed tensor values are discarded.

    Args:
        cls: autograd.Function subclass (the fused kernel wrapper).
        ref_impl: callable with the same signature and output tree shape as
            ``cls.apply``.  Must use only ops that already have SPMD rules
            (aten primitives or other registered autograd functions).
            If omitted, returns a decorator that takes ``ref_impl``.

    Returns:
        ``ref_impl`` when both arguments are provided (so a direct call
        does not clobber the function name when used as a one-arg
        decorator); otherwise a decorator that registers ``ref_impl``
        against ``cls``.

    Example (function-call form)::

        from ops.interfaces.pos_emb.rope import _RoPE, native_rope

        register_decomposition(_RoPE, native_rope)

    Example (decorator form)::

        @register_decomposition(_RoPE)
        def native_rope(xq, xk, freqs_cis, ...):
            ...
    """
    if ref_impl is None:
        # Decorator-factory form: @register_decomposition(cls)
        def decorator(ref_impl):
            return register_decomposition(cls, ref_impl)

        return decorator

    from torch.utils._pytree import tree_flatten, tree_map

    def _to_meta(arg):
        """Convert a tensor to a meta tensor preserving SPMD annotations."""
        if not isinstance(arg, torch.Tensor):
            return arg
        meta = torch.empty_like(arg, device="meta", requires_grad=arg.requires_grad)
        if has_local_type(arg) or get_partition_spec(arg) is not None:
            assert_type(
                meta, get_local_type(arg), partition_spec=get_partition_spec(arg)
            )
        return meta

    # Note: ``handle_torch_function`` pops the active ``_SpmdTypeMode`` off
    # torch's TorchFunctionMode stack while ``typecheck_forward`` runs.  To
    # have the decomposition's primitive aten ops re-dispatch through the
    # checker, we re-enter the *same* mode via ``with current_mode:``.  The
    # mode is reentrant: setup (autograd patch, vmap, backward hooks) is
    # owned by ``typecheck()`` once per session, so re-pushing the mode
    # only updates torch's mode stack -- no double-install.
    def typecheck_forward(*args, **kwargs):
        # Lazy import: runtime.py must not import _checker at module load.
        from spmd_types._checker import _current_mode

        current_mode = _current_mode()
        assert current_mode is not None, (
            f"{cls.__name__}.apply with register_decomposition requires an "
            f"active typecheck() context.  (If a typecheck() context IS "
            f"active, this means _SpmdTypeMode is on torch's stack but "
            f"_current_mode is None -- typecheck() session bookkeeping is "
            f"broken.)"
        )

        # 1. Trace the decomposition on meta tensors under the active mode.
        #    Primitive aten ops propagate SPMD types / PartitionSpecs, but
        #    the decomp does not allocate any storage.
        meta_args = tree_map(_to_meta, args)
        meta_kwargs = tree_map(_to_meta, kwargs)
        with current_mode:
            ref_out = ref_impl(*meta_args, **meta_kwargs)

        # 2. Run the real computation path.
        real_out = cls.apply(*args, **kwargs)

        # 3. Copy the inferred SPMD types from the decomposition output onto the
        #    real computation path output, walking both trees in parallel.
        ref_flat, ref_spec = tree_flatten(ref_out)
        real_flat, real_spec = tree_flatten(real_out)
        if ref_spec != real_spec:
            raise SpmdTypeError(
                f"{cls.__name__}: decomposition output tree does not match "
                f"real computation path output tree. "
                f"decomposition: {ref_spec}, fused: {real_spec}"
            )
        for real_leaf, ref_leaf in zip(real_flat, ref_flat):
            if not (
                isinstance(real_leaf, torch.Tensor)
                and isinstance(ref_leaf, torch.Tensor)
            ):
                continue
            local_type = get_local_type(ref_leaf)
            spec = get_partition_spec(ref_leaf)
            if not local_type and spec is None:
                continue
            assert_type(real_leaf, local_type, partition_spec=spec)
        return real_out

    cls.typecheck_forward = staticmethod(typecheck_forward)
    _TYPECHECK_AUTOGRAD_FUNCTIONS.add(cls)
    return ref_impl


# =============================================================================
# local_map
# =============================================================================


class _InferType:
    """Type of the ``Infer`` sentinel.  See ``local_map`` for semantics."""

    def __repr__(self) -> str:
        return "Infer"


Infer = _InferType()


_LocalMapSpecLeaf: TypeAlias = (
    "PerMeshAxisSpmdTypes "
    "| PartitionSpec "
    "| tuple[PerMeshAxisSpmdTypes, PartitionSpec | None] "
    "| None"
)


def local_map(  # noqa: C901
    *,
    in_types: Any = Infer,
    out_types: Any,
):
    """Decorator that wraps a function whose body the SPMD type checker
    can't reason about -- data-dependent ops (e.g. ``searchsorted``), custom
    kernels, manual shard manipulations (e.g. ring attention). Inside ``fn``,
    type checking is suspended; ``local_map`` re-establishes SPMD types at the
    boundary:

    * On **entry**, each input is checked against ``in_types``.  The tensor's
      existing SPMD type must be consistent with what's declared
      (``assert_type``-style refinement -- compatible existing types are
      preserved, conflicts raise).  When ``in_types=Infer`` (the default),
      inputs flow through unchecked, trusting whatever the caller already
      annotated.

    * On **return**, each output is assigned the SPMD type declared by
      ``out_types``.  The body produces tensors with no SPMD types of their own
      (because type checking was suspended), so this is what makes them usable
      in the surrounding typed code.

    Forward-only: gradient types are not part of the contract.  Inspired by
    ``torch.distributed.local_map``.

    ``in_types`` is either the bare sentinel ``Infer`` (the default) or a pytree
    mirroring the structure of the function's positional arguments.
    ``out_types`` is always a pytree mirroring the return value.

    Each spec leaf is one of:

    * ``dict[axis, type]`` -- per-axis local SPMD type (``R``/``I``/``V``/ ``P``
      or ``S(i)``).  ``S(i)`` decays to ``V`` + ``PartitionSpec`` automatically
      (see ``assert_type``).
    * ``PartitionSpec`` -- only the partition spec; local types (V) are inferred
      from the spec's axes.  Use this when you only care about which dim is
      sharded and don't need to constrain replicate / invariant axes explicitly.
    * ``(dict[axis, type], PartitionSpec | None)`` -- the dict plus an explicit
      ``PartitionSpec``.  Cannot mix ``S(i)`` entries with an explicit
      ``PartitionSpec``.
    * ``None`` -- assert the leaf is non-tensor (int, float, ...).  Raises if a
      tensor reaches this position.

    ``Infer`` is **only** valid as the bare value of ``in_types`` (the default)
    -- it short-circuits the entire input pass with no boundary checks at all.
    ``Infer`` anywhere as a leaf in an ``in_types`` or ``out_types`` pytree
    raises ``SpmdTypeError``: the caller must always state the contract
    leaf-by-leaf.

    Mismatches in pytree structure, tensor/non-tensor alignment, local type, or
    ``PartitionSpec`` raise ``SpmdTypeError``.

    Example -- typical usage just states the output contract; ``in_types``
    defaults to ``Infer`` so the caller's existing input annotations flow
    through unchanged::

        @spmd.local_map(out_types=PartitionSpec(None, ddp))
        def local_fn(x, w):
            return x @ w

    When args are nested, ``in_types`` / ``out_types`` mirror that nesting.
    Spec leaves sit wherever args have leaves -- the alignment is purely
    structural::

        @spmd.local_map(
            in_types=({ddp: S(0)}, [{ddp: R}, {ddp: R}]),
            out_types=({ddp: S(0)}, {ddp: S(0)}),
        )
        def fn(x, weights):
            return x @ weights[0], x @ weights[1]
    """
    import functools

    from torch.utils._pytree import tree_map

    def decorator(fn):
        prefix = f"local_map[{fn.__qualname__}]"

        def _check_arg(arg: object, spec: _LocalMapSpecLeaf, kind: str) -> None:
            if spec is None:
                if isinstance(arg, torch.Tensor):
                    raise SpmdTypeError(
                        f"{prefix}: a {kind} is a tensor but its "
                        f"{kind}_types leaf is None (non-tensor expected)"
                    )
                return
            # Explicit dict / (dict, ps) / PartitionSpec specs require a tensor.
            if not isinstance(arg, torch.Tensor):
                raise SpmdTypeError(
                    f"{prefix}: a {kind} is not a tensor but its "
                    f"{kind}_types leaf = {spec!r}"
                )

        def _assert_spec(arg: torch.Tensor, spec: _LocalMapSpecLeaf) -> None:
            """Assert that ``arg`` satisfies ``spec`` via ``assert_type``.

            On input, the call is a refinement check -- ``assert_type``
            verifies the existing type/spec on the tensor is consistent
            with ``spec`` (and may enrich it with new axes, but does not
            change types that are already set).  On output, it stamps the
            declared type/spec onto the otherwise-untyped body result.

            Errors from ``assert_type`` propagate as-is; the call stack
            already shows which ``local_map``-wrapped function and which
            boundary the failure came from.  Leaves that don't match any
            of the recognized spec shapes raise ``SpmdTypeError`` here.
            """
            # Check ``PartitionSpec`` before generic ``tuple`` since
            # PartitionSpec is itself a tuple subclass.
            if isinstance(spec, PartitionSpec):
                # PartitionSpec-only leaf: assert_type derives V on each axis
                # mentioned in the spec.
                local_type, pspec = {}, spec
            elif isinstance(spec, tuple):
                local_type, pspec = spec
            elif isinstance(spec, dict):
                local_type, pspec = spec, None
            else:
                raise SpmdTypeError(
                    f"{prefix}: invalid spec leaf {spec!r}; expected "
                    f"a per-axis dict, a (dict, PartitionSpec | None) "
                    f"tuple, a PartitionSpec, or None."
                )
            assert_type(arg, local_type, partition_spec=pspec)

        def _walk_boundary(values: object, specs: Any, kind: str) -> None:
            """Walk ``values``'s pytree and align ``specs`` along its leaves.

            ``values``'s structure is the prototype: wherever it has a leaf
            (a tensor or any non-container value), the corresponding
            subtree of ``specs`` is treated as a spec leaf and handed to
            ``_check_arg`` / ``_assert_spec``.  No duck-typing on what
            "looks like" a spec dict -- alignment is purely structural.

            ``Infer`` reaching this point is always at a leaf (the bare
            ``in_types=Infer`` case is short-circuited above) and is
            misuse -- it raises here.
            """

            def _check(val: object, spec: object) -> object:
                if spec is Infer:
                    # Bare ``in_types=Infer`` is short-circuited before we
                    # reach here, so any ``Infer`` at this point is at a
                    # pytree leaf, which is misuse.
                    raise SpmdTypeError(
                        f"{prefix}: {kind}_types contains Infer at a leaf; "
                        f"only the bare value of in_types may be Infer."
                    )
                _check_arg(val, spec, kind)
                if spec is not None:
                    _assert_spec(val, spec)  # pyre-ignore[6]
                return val

            try:
                tree_map(_check, values, specs)
            except ValueError as e:
                raise SpmdTypeError(
                    f"{prefix}: {kind}_types pytree structure does not "
                    f"match {kind}: {e}"
                ) from None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Fast path: with no active typecheck (or inside ``no_typecheck``).
            if not is_type_checking():
                return fn(*args, **kwargs)

            # Boundary input checks run in the caller's typecheck environment
            # (whatever mode is active when the wrapped fn is called), so
            # ``assert_type`` sees the same global/local SPMD context as the
            # caller -- only the body below is scoped under ``no_typecheck()``.
            #
            # Bare ``in_types=Infer`` short-circuits the entire input pass;
            # any ``Infer`` reaching ``_walk_boundary`` is a leaf-level misuse
            # and raises there.
            if in_types is not Infer:
                _walk_boundary(args, in_types, "input")

            with no_typecheck():
                result = fn(*args, **kwargs)

            _walk_boundary(result, out_types, "output")
            return result

        return wrapper

    return decorator
