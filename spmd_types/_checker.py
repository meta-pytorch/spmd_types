# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SPMD type checking logic and tensor type tracking.

This module provides:
- Functions to track SPMD types on tensors
- Type inference/checking logic for operations
- TorchFunctionMode for automatic type propagation
- Global SPMD shard propagation via DTensor's ShardingPropagator
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Literal, NamedTuple, Optional, overload

import torch
import torch.overrides
from spmd_types import _dist
from spmd_types._collectives import (
    all_gather,
    all_reduce,
    all_to_all,
    redistribute,
    reduce_scatter,
)
from spmd_types._frame import _get_user_frame
from spmd_types._local import convert, invariant_to_replicate, reinterpret
from spmd_types._mesh_axis import MeshAxis
from spmd_types._scalar import _unwrap_args, Scalar
from spmd_types._state import _current_mode, _set_current_mode, current_mesh
from spmd_types._traceback import _filter_and_reraise, api_boundary
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
    I,
    LocalSpmdType,
    normalize_axis,
    normalize_local_type,
    P,
    partition_spec_get_shard,
    PartitionSpec,
    PerMeshAxisLocalSpmdType,
    PerMeshAxisSpmdType,
    PerMeshAxisSpmdTypes,
    R,
    RedistributeError,
    Shard,
    shard_types_to_partition_spec,
    SpmdTypeError,
    to_local_type,
    V,
)
from torch.distributed._local_tensor import maybe_disable_local_tensor_mode
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, placement_types as dtensor_type
from torch.distributed.tensor._utils import ExplicitRedistributionContext
from torch.overrides import handle_torch_function, has_torch_function


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
# Scalar Sentinel
# =============================================================================


class _ScalarType:
    """Sentinel for Python scalars in SPMD type inference.

    We assume a Python scalar is the same value on all ranks and carries no
    gradient.  This assumption can be wrong: a rank-dependent scalar (e.g.
    ``rank / world_size``) is really Varying, and a scalar that is the sum of
    per-rank contributions is really Partial.  We choose to assume Replicate
    anyway because (1) the vast majority of scalars in practice *are* the same
    on every rank, and (2) requiring users to annotate every literal ``2.0``
    or ``eps`` would be extremely noisy for little safety gain.  A future
    improvement could add an explicit wrapper (e.g. ``Varying(scalar)``) for
    the rare rank-dependent case; until then Dynamo tracing with SymInt offers
    partial safety.

    Given the assumption, _Scalar is compatible with both R and I -- analogous
    to how Python scalars are "weak" in dtype promotion and do not determine
    the specific dtype.

    _Scalar participates in linearity validation (P + scalar is affine, not
    linear) but does not influence the inferred output type (R vs I).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "Scalar"


_Scalar = _ScalarType()


# =============================================================================
# Fix Suggestion Engine
# =============================================================================

_TYPE_FULL_NAMES = {
    R: "Replicate",
    I: "Invariant",
    V: "Varying",
    P: "Partial",
}

# Each entry: (from_type, to_type_instance, operation_str, consequence_str)
# When a type error occurs, we try replacing one operand's type and re-running
# inference. If it succeeds, we report the fix.
_FIX_CANDIDATES: list[
    tuple[PerMeshAxisLocalSpmdType, PerMeshAxisLocalSpmdType, str, str]
] = [
    (
        I,
        R,
        "convert(tensor, {axis_arg}, src=I, dst=R)",
        "no-op forward, all-reduce in backward",
    ),
    (
        I,
        V,
        "reinterpret(tensor, {axis_arg}, src=I, dst=V)",
        "no-op forward, all-reduce in backward",
    ),
    (
        I,
        P,
        "convert(tensor, {axis_arg}, src=I, dst=P)",
        "zeros non-rank-0 in forward, no-op backward",
    ),
    (
        P,
        R,
        "all_reduce(tensor, {axis_arg}, src=P, dst=R)",
        "all-reduce in forward, all-reduce in backward",
    ),
    (
        P,
        I,
        "all_reduce(tensor, {axis_arg}, src=P, dst=I)",
        "all-reduce in forward, no-op backward",
    ),
    (
        R,
        P,
        "convert(tensor, {axis_arg}, src=R, dst=P)",
        "zeros non-rank-0 in forward, zeros non-rank-0 in backward",
    ),
]
# NB: We don't suggest R->I because compute typically happens on R, not I.

# NB: We don't suggest
# (Partial, V, "reduce_scatter(tensor, axis, src=P, dst=V)", ...)
# because this requires reasoning about the desired size of the output; if the
# original code is the correct size, the rewrite is incorrect.

# NB: We don't suggest
# (Varying, P, "reinterpret(tensor, axis, src=V, dst=P)", ...)
# because the operation requiring P should have just been fed V directly.


def _suggest_fixes(
    axis: MeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    infer_fn: Callable[
        [MeshAxis, list[PerMeshAxisLocalSpmdType]], PerMeshAxisLocalSpmdType
    ],
) -> list[tuple[str, str, PerMeshAxisLocalSpmdType]]:
    """Try each candidate fix and return suggestions for ones that work.

    For each candidate (from_type, to_type, ...):
    1. Check if from_type exists in axis_types
    2. Replace one occurrence with to_type
    3. Call infer_fn on the modified list
    4. If no exception, this is a valid fix -- include it

    The ``infer_fn`` must be a *raw* inference function that raises plain
    ``SpmdTypeError`` without calling ``_format_error_with_suggestions``.
    This makes recursion structurally impossible.

    Args:
        axis: The mesh axis where the type error occurred.
        axis_types: The list of per-axis SPMD types that caused the error.
        infer_fn: A raw inference function that takes (axis, axis_types) and
            returns the inferred output type, or raises ``SpmdTypeError``.

    Returns:
        List of (operation_str, consequence_str, from_type) tuples.
    """
    # Compute the axis argument text once
    axis_arg = format_axis(axis)

    suggestions: list[tuple[str, str, PerMeshAxisLocalSpmdType]] = []
    for from_type, to_type, operation_template, consequence in _FIX_CANDIDATES:
        # Find the first operand matching from_type
        idx = None
        for i, t in enumerate(axis_types):
            if t is from_type:
                idx = i
                break
        if idx is None:
            continue
        # Try replacing that operand
        modified = list(axis_types)
        modified[idx] = to_type
        try:
            fix_output = infer_fn(axis, modified)
        except SpmdTypeError:
            continue
        # Filter: does this fix preserve the natural output of the other operands?
        remaining = [t for i, t in enumerate(axis_types) if i != idx]
        if remaining:
            try:
                natural_output = infer_fn(axis, remaining)
                if fix_output != natural_output:
                    continue  # Fix changes the output type -- skip
            except (SpmdTypeError, ValueError):
                pass  # Can't determine natural output -- keep the suggestion
        operation = operation_template.format(axis_arg=axis_arg)
        suggestions.append((operation, consequence, from_type))
    return suggestions


def _format_error_with_suggestions(
    base_msg: str,
    axis: MeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    infer_fn: Callable[
        [MeshAxis, list[PerMeshAxisLocalSpmdType]], PerMeshAxisLocalSpmdType
    ],
) -> str:
    """Format an error message, appending fix suggestions if any exist.

    Args:
        base_msg: The base error message to display.
        axis: The mesh axis where the type error occurred.
        axis_types: The list of per-axis SPMD types that caused the error.
        infer_fn: A raw inference function used to discover valid fixes.
    """
    suggestions = _suggest_fixes(axis, axis_types, infer_fn)
    if not suggestions:
        return base_msg
    lines = [
        base_msg,
        "Are you missing a collective or a reinterpret/convert call? e.g.,",
    ]
    for operation, consequence, from_type in suggestions:
        lines.append(
            f"  {operation} on the {_TYPE_FULL_NAMES[from_type]} operand ({consequence})"
        )
    return "\n".join(lines)


# =============================================================================
# SPMD Type Tracking on Tensors
# =============================================================================


def _validate(type: LocalSpmdType) -> LocalSpmdType:
    """Validate a LocalSpmdType and normalize axis keys.

    Extends ``normalize_local_type`` with additional checks for internal
    sentinel types (_Scalar, Shard) that must not be stored on tensors.

    Args:
        type: A LocalSpmdType dict mapping mesh axes to per-axis SPMD types.

    Raises:
        TypeError: If any value is not a PerMeshAxisLocalSpmdType (R, I, V, or P),
            or is an internal sentinel type.
    """
    # Pre-check for internal sentinel types that normalize_local_type
    # does not know about (they live in _checker.py, not types.py).
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


def _update_axis_in_partition_spec(
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


@overload
def assert_type(
    tensor: torch.Tensor,
    type: LocalSpmdType,
    partition_spec: PartitionSpec | None = ...,
) -> torch.Tensor: ...


@overload
def assert_type(
    tensor: torch.Tensor,
    type: PerMeshAxisSpmdTypes,
) -> torch.Tensor: ...


@overload
def assert_type(
    tensor: torch.Tensor,
    type: PerMeshAxisLocalSpmdType,
) -> torch.Tensor: ...


@api_boundary
def assert_type(  # noqa: C901
    tensor: torch.Tensor,
    type: PerMeshAxisSpmdTypes | PerMeshAxisLocalSpmdType,
    partition_spec: PartitionSpec | None = None,
) -> torch.Tensor:
    """Assert or set the SPMD type on a tensor.

    If the tensor has no SPMD type, sets it. If the tensor already has an SPMD
    type, checks compatibility using refinement semantics (see below).

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


@api_boundary
def reinterpret_mesh(
    tensor: torch.Tensor,
    type: PerMeshAxisSpmdTypes,
) -> torch.Tensor:
    """Explicitly reinterpret a tensor onto a cross-mesh-compatible local type.

    This is a no-op on the underlying tensor data. It only retags the tensor
    with a new local SPMD type after verifying the source and destination mesh
    presentations are compatible under one-hop grouping.
    """
    if not has_local_type(tensor):
        raise SpmdTypeError(
            "reinterpret_mesh requires a tensor with an existing local SPMD type."
        )
    if get_partition_spec(tensor) is not None:
        raise SpmdTypeError(
            "reinterpret_mesh does not support tensors carrying PartitionSpec. "
            "Cross-mesh reinterpretation is currently local-SPMD-only."
        )

    from spmd_types._mesh_region import check_reinterpret_mesh_compatible

    src_type = get_local_type(tensor)
    dst_type = _validate(type)
    compat = check_reinterpret_mesh_compatible(src_type, dst_type)
    if isinstance(compat, str):
        context = _format_operator_context(
            reinterpret_mesh,
            [_RawArgEntry(None, tensor), _RawArgEntry(None, dst_type)],
            mesh=current_mesh(),
        )
        raise SpmdTypeError(compat, context=context)

    result = torch.ops.aten.alias.default(tensor)
    _set_local_type_raw(result, dst_type)
    if _TRACE:
        _trace_op(reinterpret_mesh, [src_type], dst_type)
    return result


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
# Type Inference Logic
# =============================================================================


class OpLinearity(Enum):
    """Classifies how a torch op interacts with Partial (P) types.

    - NONLINEAR: P cannot propagate (safe default for unclassified ops).
    - LINEAR: Linear map on direct sum; all-P -> P.
      Examples: addition, subtraction, concat, clone.
    - MULTILINEAR: Linear in each factor separately; P in one factor with R
      in others -> P, but P in multiple factors is forbidden.
      Examples: multiplication, matmul, einsum.
    """

    NONLINEAR = auto()
    LINEAR = auto()
    MULTILINEAR = auto()


def _infer_local_type_for_axis_raw(  # noqa: C901
    axis: MeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    out_partial: bool = False,
    linearity: OpLinearity = OpLinearity.NONLINEAR,
) -> PerMeshAxisLocalSpmdType:
    """Raw inference logic -- raises plain ``SpmdTypeError`` without suggestions.

    The public wrapper ``infer_local_type_for_axis`` catches these errors and
    enriches them with fix suggestions.

    Args:
        axis: The mesh axis name (used for error messages).
        axis_types: List of input SPMD types for this axis.
        out_partial: If True, reinterpret the inferred result as Partial.
        linearity: How the op interacts with Partial types.
    """
    if not axis_types:
        if out_partial:
            return P
        raise ValueError(f"No types provided for axis {format_axis(axis)}")

    type_set = set(axis_types)

    # Check type compatibility and infer output type.
    #
    # _Scalar is compatible with R, I, and V (adopts the tensor type).
    # With P it depends on linearity: P * scalar is valid (MULTILINEAR,
    # scaling preserves partial-sum), but P + scalar is affine (LINEAR,
    # the constant gets summed N times across ranks).
    #
    #   {R}, {R, _Scalar} -> R           {I}, {I, _Scalar} -> I
    #   {V}, {V, _Scalar}, {R, V, ...} -> V
    #   {P} -> P (linearity checks)
    #   {P, _Scalar} -> MULTILINEAR: P, LINEAR: error (affine)
    #   {P, R, ...} -> MULTILINEAR single-P: P, else: error

    if type_set <= {R, _Scalar}:
        inferred_type = R
    elif type_set <= {I, _Scalar}:
        inferred_type = I
    elif type_set <= {R, V, _Scalar}:
        inferred_type = V
    elif I in type_set:
        raise SpmdTypeError(
            f"Invariant type on axis {format_axis(axis)} cannot mix with other types. "
            f"Found types: {axis_types}"
        )
    elif P in type_set:
        # All P-related inference is handled here.
        non_p = type_set - {P, _Scalar}  # real non-P tensor types
        has_scalar = _Scalar in type_set
        p_count = sum(1 for t in axis_types if t is P)

        # P + V is always invalid regardless of linearity.
        if non_p - {R}:
            raise SpmdTypeError(
                f"Partial type on axis {format_axis(axis)} cannot combine with "
                f"Varying. Reduce Partial first (all_reduce -> R, or "
                f"reduce_scatter -> V). Found types: {axis_types}"
            )

        assert type_set <= {P, R, _Scalar}, type_set
        if linearity is OpLinearity.NONLINEAR:
            raise SpmdTypeError(
                f"Partial type on axis {format_axis(axis)} cannot propagate "
                f"through non-linear ops (non-linear of a partial sum != "
                f"partial sum of non-linear). Reduce first with all_reduce "
                f"or reduce_scatter. Found types: {axis_types}"
            )
        if linearity is OpLinearity.LINEAR and (non_p or has_scalar):
            # P + R is invalid: R contributes the same value N times.
            # P + scalar is affine: the constant gets summed N times.
            raise SpmdTypeError(
                f"Partial type on axis {format_axis(axis)} in a linear op "
                f"requires all operands to be Partial (sum of partial sums "
                f"is a partial sum, but adding a Replicate or scalar value "
                f"makes the result affine -- the non-partial term gets "
                f"summed N times across ranks). "
                f"Found types: {axis_types}"
            )
        if linearity is OpLinearity.MULTILINEAR and p_count > 1:
            raise SpmdTypeError(
                f"Partial in multiple factors of multilinear op on axis "
                f"{format_axis(axis)} is forbidden. Reduce all but one "
                f"factor first. Found types: {axis_types}"
            )
        # Valid cases that reach here:
        # - LINEAR, all-P: sum of partial sums is still partial
        # - MULTILINEAR, single P with R|scalar: scaling preserves partial
        inferred_type = P
    else:
        raise SpmdTypeError(
            f"Incompatible types on axis {format_axis(axis)}: {axis_types}"
        )

    # Apply out_partial: reinterpret as P
    if out_partial:
        if inferred_type is V or inferred_type is P:
            return P
        elif inferred_type is R:
            raise SpmdTypeError(
                f"out_partial_axes includes axis {format_axis(axis)} but inferred type "
                f"is R (Replicate). A replicated result cannot be partial -- this likely "
                f"indicates an unsharded contraction dimension. "
                f"Input types: {axis_types}"
            )
        else:
            raise SpmdTypeError(
                f"Cannot mark axis {format_axis(axis)} as partial with type {inferred_type}. "
                f"out_partial_axes is only valid for V or P types."
            )

    return inferred_type


def infer_local_type_for_axis(
    axis: DeviceMeshAxis,
    axis_types: list[PerMeshAxisLocalSpmdType],
    out_partial: bool = False,
    linearity: OpLinearity = OpLinearity.NONLINEAR,
) -> PerMeshAxisLocalSpmdType:
    """
    Infer the output SPMD type for a single mesh axis given input types.

    Args:
        axis: The mesh axis (for error messages)
        axis_types: List of input types for this axis
        out_partial: If True, reinterpret the result as Partial
        linearity: How the op interacts with Partial types

    Returns:
        The inferred output type

    Raises:
        SpmdTypeError: If the input types are incompatible
    """
    axis = normalize_axis(axis)
    try:
        return _infer_local_type_for_axis_raw(axis, axis_types, out_partial, linearity)
    except SpmdTypeError as e:
        raise SpmdTypeError(
            _format_error_with_suggestions(
                str(e),
                axis,
                axis_types,
                lambda a, t: _infer_local_type_for_axis_raw(
                    a, t, out_partial, linearity
                ),
            )
        ) from None


def _format_axis_set(axes: Iterable[DeviceMeshAxis]) -> str:
    """Format a set of mesh axes for display (stride-descending, comma-separated, in braces)."""
    normalized = [(normalize_axis(ax), format_axis(ax)) for ax in axes]
    # Sort by max stride descending (outermost first), then by name for ties.
    normalized.sort(
        key=lambda item: (-max(d for _, d in item[0].layout.sizes_and_strides), item[1])
    )
    return "{" + ", ".join(name for _, name in normalized) + "}"


def _format_rank_set(ranks: Iterable[int]) -> str:
    """Format a set of rank offsets for display (sorted, in braces)."""
    return "{" + ", ".join(str(r) for r in sorted(set(ranks))) + "}"


def _compute_rank_space(axes: frozenset[MeshAxis]) -> frozenset[int] | None:
    """Compute the set of relative rank offsets covered by a set of mesh axes.

    Returns None if the axes are not mutually orthogonal (cannot compute).
    """
    from spmd_types._mesh_axis import flatten_axes

    if not axes:
        return frozenset()
    axes_list = list(axes)
    if len(axes_list) == 1:
        return frozenset(axes_list[0].layout.all_ranks_from_zero())
    if not axes_list[0].isorthogonal(*axes_list[1:]):
        return None
    combined = flatten_axes(tuple(axes_list))
    return frozenset(combined.layout.all_ranks_from_zero())


def _cross_mesh_advice(input_types_list: list[LocalSpmdType]) -> str:  # noqa: C901
    """Check if operands with mismatched axes have structural relationships.

    Produces advice for:
    1. Cross-mesh compatible axes -> suggest reinterpret_mesh()
    2. Sub-axis relationships (partial rank overlap) -> identify missing axes
    3. Different rank spaces without sub-axis relationship -> note rank mismatch

    Returns an empty string if no advice can be given.
    """
    from spmd_types._mesh_region import check_reinterpret_mesh_compatible

    # Normalize all types to use MeshAxis keys (not raw ProcessGroup).
    normalized: list[LocalSpmdType] = [
        normalize_local_type(t) for t in input_types_list
    ]

    # Collect distinct axis sets (ignoring types -- we only care about
    # structural mesh compatibility here).
    seen_axis_sets: list[LocalSpmdType] = []
    for typ in normalized:
        axes = frozenset(typ.keys())
        if not any(frozenset(s.keys()) == axes for s in seen_axis_sets):
            seen_axis_sets.append(typ)
    if len(seen_axis_sets) < 2:
        return ""

    parts: list[str] = []

    for i in range(len(seen_axis_sets)):
        for j in range(i + 1, len(seen_axis_sets)):
            a, b = seen_axis_sets[i], seen_axis_sets[j]
            if a.keys() == b.keys():
                continue

            # 1. Check cross-mesh compatibility with uniform R types,
            # so that type mismatches don't mask axis compatibility.
            a_uniform: LocalSpmdType = {ax: R for ax in a}
            b_uniform: LocalSpmdType = {ax: R for ax in b}
            compat = check_reinterpret_mesh_compatible(a_uniform, b_uniform)
            if not isinstance(compat, str):
                # They are structurally compatible. Suggest converting to
                # the set with more axes (the finer-grained mesh).
                if len(a) >= len(b):
                    fine, coarse = a, b
                else:
                    fine, coarse = b, a
                parts.append(
                    f"The operand axes {_format_axis_set(coarse.keys())} and "
                    f"{_format_axis_set(fine.keys())} are cross-mesh compatible. "
                    f"Use reinterpret_mesh() to convert the operand "
                    f"with axes {_format_axis_set(coarse.keys())} to axes "
                    f"{_format_axis_set(fine.keys())} before this operation."
                )
                continue

            # 2. Not cross-mesh compatible. Check sub-axis relationships
            # among the non-shared axes.
            shared = a.keys() & b.keys()
            a_only = frozenset(ax for ax in a if ax not in shared)
            b_only = frozenset(ax for ax in b if ax not in shared)

            if not a_only or not b_only:
                continue

            # Restrict complement/sub-axis searches to axes related to
            # these operands (members or sub-axes of their axis sets),
            # so we don't suggest axes from unrelated meshes.
            all_operand_axes = frozenset(a.keys()) | frozenset(b.keys())
            universe = _relevant_axes(all_operand_axes)

            # Look for sub-axis relationships (one axis is a subgroup
            # of another, indicating a flatten relationship with missing
            # intermediate axes).
            found_sub = False
            for ax_a in sorted(a_only, key=lambda ax: format_axis(ax)):
                for ax_b in sorted(b_only, key=lambda ax: format_axis(ax)):
                    if ax_a < ax_b:
                        found_sub = True
                        _add_sub_axis_advice(parts, ax_a, ax_b, a, universe)
                    elif ax_b < ax_a:
                        found_sub = True
                        _add_sub_axis_advice(parts, ax_b, ax_a, b, universe)

            if found_sub:
                continue

            # 3. No sub-axis relationship. Check for partial rank overlap.
            a_ranks = _compute_rank_space(a_only)
            b_ranks = _compute_rank_space(b_only)

            if a_ranks is None or b_ranks is None:
                continue

            if a_ranks == b_ranks:
                continue

            # Exclude the trivial overlap at rank 0 (all_ranks_from_zero
            # always includes 0).
            overlap = (a_ranks & b_ranks) - {0}
            if overlap:
                # Try to find named axes that would complete each side.
                a_missing = _find_orthogonal_complements(frozenset(a.keys()), universe)
                b_missing = _find_orthogonal_complements(frozenset(b.keys()), universe)
                suggestions: list[str] = []
                if a_missing:
                    names = ", ".join(format_axis(ax) for ax in a_missing)
                    suggestions.append(
                        f"the operand with axes "
                        f"{_format_axis_set(a.keys())} is missing "
                        f"{names}"
                    )
                if b_missing:
                    names = ", ".join(format_axis(ax) for ax in b_missing)
                    suggestions.append(
                        f"the operand with axes "
                        f"{_format_axis_set(b.keys())} is missing "
                        f"{names}"
                    )
                if suggestions:
                    msg = (
                        f"The non-shared axes {_format_axis_set(a_only)} "
                        f"and {_format_axis_set(b_only)} partially "
                        f"overlap. "
                    )
                    combined = "; ".join(suggestions)
                    msg += combined[0].upper() + combined[1:] + "."
                    parts.append(msg)

    return " ".join(parts)


def _relevant_axes(
    all_operand_axes: frozenset[MeshAxis],
) -> frozenset[MeshAxis]:
    """Compute the set of named axes relevant to a set of operand axes.

    Returns axes that either appear in *all_operand_axes* or are sub-axes
    of a member.  This scopes complement/sub-axis searches to the mesh
    dimensions actually in play, avoiding spurious suggestions from
    unrelated meshes registered in the global ``_mesh_axis_names`` table.
    """
    from spmd_types._mesh_axis import _mesh_axis_names

    result: set[MeshAxis] = set()
    for candidate in _mesh_axis_names:
        if candidate in all_operand_axes:
            result.add(candidate)
            continue
        # Include if candidate is a sub-axis of any operand axis.
        for ax in all_operand_axes:
            if candidate < ax:
                result.add(candidate)
                break
    return frozenset(result)


def _find_orthogonal_complements(
    axes: frozenset[MeshAxis],
    universe: frozenset[MeshAxis],
) -> list[MeshAxis]:
    """Find mesh axes from *universe* orthogonal to all axes in the set.

    Returns axes from *universe* that are not already in *axes* and are
    mutually orthogonal with every member -- i.e., axes that could be
    added to the set without conflict.  Sorted by name for deterministic
    output.
    """
    result: list[MeshAxis] = []
    for candidate in universe:
        if candidate in axes:
            continue
        if all(candidate.isorthogonal(ax) for ax in axes):
            result.append(candidate)
    return sorted(result, key=lambda ax: format_axis(ax))


def _add_sub_axis_advice(
    parts: list[str],
    sub: MeshAxis,
    super_: MeshAxis,
    sub_type: LocalSpmdType,
    universe: frozenset[MeshAxis],
) -> None:
    """Add advice when one axis is a proper sub-axis of another."""
    from spmd_types._mesh_axis import flatten_axes

    sub_ranks = sub.layout.all_ranks_from_zero()
    super_ranks = super_.layout.all_ranks_from_zero()

    # Try to identify the complement axis -- a known axis that, combined
    # with sub, produces super_.
    complement: MeshAxis | None = None
    for candidate in universe:
        if candidate == sub:
            continue
        if candidate <= super_ and candidate.isorthogonal(sub):
            try:
                combined = flatten_axes((sub, candidate))
                if combined == super_:
                    complement = candidate
                    break
            except ValueError:
                pass

    if complement is not None:
        parts.append(
            f"{format_axis(sub)} is a sub-axis of {format_axis(super_)} "
            f"(ranks {_format_rank_set(sub_ranks)} vs "
            f"{_format_rank_set(super_ranks)}), "
            f"so the operand with axes {_format_axis_set(sub_type.keys())} "
            f"is missing axis {format_axis(complement)}."
        )
    else:
        parts.append(
            f"{format_axis(sub)} is a sub-axis of {format_axis(super_)} "
            f"(ranks {_format_rank_set(sub_ranks)} vs "
            f"{_format_rank_set(super_ranks)}), "
            f"so the operand with axes {_format_axis_set(sub_type.keys())} "
            f"is missing some mesh axes."
        )


def infer_output_type(  # noqa: C901
    input_types_list: list[LocalSpmdType],
    out_partial_axes: set[DeviceMeshAxis] | None = None,
    linearity: OpLinearity = OpLinearity.NONLINEAR,
) -> LocalSpmdType:
    """
    Infer output SPMD types from a list of input types.

    This implements the typing rules for operations like einsum:
    - If all operands are R -> output is R
    - If all operands are I -> output is I
    - If all operands are V -> output is V
    - If all operands are P -> output is P (linear ops only)
    - Mixed R/V -> output is V
    - I cannot mix with other types
    - P cannot mix with non-P types

    In strict mode, raises ``SpmdTypeError`` when an operand is missing an
    axis.  In permissive mode, operands missing an axis are skipped and the
    output type for that axis is inferred from the remaining operands.

    Args:
        input_types_list: List of LocalSpmdType dicts, one per operand
        out_partial_axes: Optional set of mesh axis names to mark as partial
        linearity: How the op interacts with Partial types

    Returns:
        LocalSpmdType dict for the output
    """
    if out_partial_axes is None:
        out_partial_axes = set()

    # Auto-reinterpret operands from foreign meshes to the current mesh
    # before collecting axes, so we don't need to recompute afterwards.
    mesh = current_mesh()
    if mesh is not None:
        from spmd_types._mesh_region import check_reinterpret_mesh_compatible

        for i, typ in enumerate(input_types_list):
            non_mesh = frozenset(typ.keys()) - mesh if typ else frozenset()
            if non_mesh:
                target = check_reinterpret_mesh_compatible(typ, mesh)
                if isinstance(target, str):
                    raise SpmdTypeError(
                        f"args[{i}] cannot be auto-reinterpreted from "
                        f"{_format_axis_set(typ.keys())} to the current mesh "
                        f"{_format_axis_set(mesh)}. {target}"
                    )
                input_types_list[i] = target

    # Collect all mesh axes mentioned (after reinterpretation).
    all_axes: set[DeviceMeshAxis] = set()
    for typ in input_types_list:
        all_axes.update(typ.keys())
    all_axes.update(out_partial_axes)
    if mesh is not None:
        all_axes.update(mesh)

    strict = _state.is_strict()

    # Check for cross-mesh compatibility and give advice on axis mismatches.
    if strict:
        axis_sets = [set(typ.keys()) for typ in input_types_list]
        if len({frozenset(s) for s in axis_sets}) > 1:
            advice = _cross_mesh_advice(input_types_list)
            if advice:
                advice = f"\n\nHint: {advice}"
            else:
                advice = ""
        else:
            advice = ""
    else:
        advice = ""

    # Infer output type for each axis.  Sort for deterministic error messages.
    output_type: LocalSpmdType = {}
    for axis in sorted(all_axes, key=lambda ax: format_axis(ax)):
        axis_types = []
        for operand_idx, typ in enumerate(input_types_list):
            if axis not in typ:
                if strict:
                    if mesh is not None and axis in mesh:
                        raise SpmdTypeError(
                            f"args[{operand_idx}] missing axis "
                            f"{format_axis(axis)} "
                            f"from the current mesh "
                            f"{_format_axis_set(mesh)}. "
                            f"args[{operand_idx}] has axes "
                            f"{_format_axis_set(typ.keys())}." + advice
                        )
                    raise SpmdTypeError(
                        f"args[{operand_idx}] missing axis "
                        f"{format_axis(axis)}. "
                        f"args[{operand_idx}] has axes "
                        f"{_format_axis_set(typ.keys())}, "
                        f"but the union of all operand axes is "
                        f"{_format_axis_set(all_axes)}. "
                        f"All operands must be annotated on the same set of "
                        f"mesh axes." + advice
                    )
                continue  # permissive: skip this operand for this axis
            axis_types.append(typ[axis])

        if not axis_types:
            continue  # all operands missing this axis

        output_type[axis] = infer_local_type_for_axis(
            axis, axis_types, out_partial=axis in out_partial_axes, linearity=linearity
        )

    return output_type


def _linear(*tys: LocalSpmdType) -> LocalSpmdType:
    """Type of a linear combination (addition)."""
    return infer_output_type(list(tys), linearity=OpLinearity.LINEAR)


def _multilinear(*tys: LocalSpmdType) -> LocalSpmdType:
    """Type of a multilinear product (matmul, elementwise mul)."""
    return infer_output_type(list(tys), linearity=OpLinearity.MULTILINEAR)


# =============================================================================
# SPMD Function Registry
# =============================================================================

# Default src/dst for each SPMD collective/local op.  When a kwarg is omitted
# by the caller the Python default from the function signature applies; we
# record those defaults here so that __torch_function__ can recover them
# (handle_torch_function does not forward defaults).
# A value of ``None`` means the parameter is required (no default).


@dataclass(frozen=True)
class _OpSpec:
    """Specification of how a torch op interacts with SPMD types.

    Attributes:
        linearity: How the op interacts with Partial (P) types.
        tensor_args: Positional arg indices (0-based) that are tensor inputs.
            Each position may hold a single tensor OR a list of tensors.
            Only needed for LINEAR ops (scalars at these positions become R).
        tensor_kwargs: Kwarg names that are tensor inputs.
        tensor_varargs_from: If set, all positional args from this index onward
            are tensor inputs (for ops with *args like einsum).
    """

    linearity: OpLinearity
    tensor_args: tuple[int, ...] = ()
    tensor_kwargs: tuple[str, ...] = ()
    tensor_varargs_from: int | None = None
    fixed_args: tuple[int, ...] = ()


_OP_REGISTRY: dict[Callable, _OpSpec] = {
    # =================================================================
    # LINEAR -- binary arithmetic (add / sub)
    # =================================================================
    torch.add: _OpSpec(OpLinearity.LINEAR, (0, 1), ("input", "other")),
    torch.Tensor.add: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.add_: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__add__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__radd__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__iadd__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.sub: _OpSpec(OpLinearity.LINEAR, (0, 1), ("input", "other")),
    torch.subtract: _OpSpec(OpLinearity.LINEAR, (0, 1), ("input", "other")),
    torch.Tensor.sub: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.subtract: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.sub_: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__sub__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__rsub__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    torch.Tensor.__isub__: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",)),
    # =================================================================
    # LINEAR -- negation / positive
    # =================================================================
    torch.neg: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.neg: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.neg_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.__neg__: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.negative: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.negative: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.negative_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.positive: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.positive: _OpSpec(OpLinearity.LINEAR, (0,)),
    # =================================================================
    # LINEAR -- clone / detach
    # =================================================================
    torch.clone: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.clone: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.detach: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.detach: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.detach_: _OpSpec(OpLinearity.LINEAR, (0,)),
    # =================================================================
    # LINEAR -- reductions (sum, mean)
    # =================================================================
    torch.sum: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.sum: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.mean: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.mean: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.nansum: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.nanmean: _OpSpec(OpLinearity.LINEAR, (0,)),
    # =================================================================
    # LINEAR -- concat / stack (tensor list at pos 0)
    # =================================================================
    torch.cat: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.concat: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.concatenate: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.stack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.hstack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.vstack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.dstack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.column_stack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.row_stack: _OpSpec(OpLinearity.LINEAR, (0,), ("tensors",)),
    torch.block_diag: _OpSpec(OpLinearity.LINEAR, (0,)),
    # =================================================================
    # LINEAR -- structural / shape ops (tensor at pos 0)
    # =================================================================
    torch.reshape: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.reshape: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.view: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.view_as: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.transpose: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.transpose: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.transpose_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.t: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.t: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.t_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.permute: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.permute: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.contiguous: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.flatten: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.flatten: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.unflatten: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.unflatten: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.ravel: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.ravel: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.squeeze: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.squeeze: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.squeeze_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.unsqueeze: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.unsqueeze: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.unsqueeze_: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.expand: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.expand_as: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.broadcast_to: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.narrow: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.narrow: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.select: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.select: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.index_select: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.index_select: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.gather: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.gather: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.split: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.split: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.chunk: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.chunk: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.unbind: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.unbind: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.flip: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.flip: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.fliplr: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.flipud: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.roll: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.roll: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.movedim: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.movedim: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.moveaxis: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.moveaxis: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.swapaxes: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.swapaxes: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.swapdims: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.swapdims: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.diagonal: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.diagonal: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.repeat: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.tile: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.tile: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.repeat_interleave: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.repeat_interleave: _OpSpec(OpLinearity.LINEAR, (0,)),
    torch.Tensor.unfold: _OpSpec(OpLinearity.LINEAR, (0,)),
    # =================================================================
    # LINEAR with fixed_args -- division (linear in numerator only)
    # =================================================================
    torch.div: _OpSpec(OpLinearity.LINEAR, (0, 1), ("input", "other"), fixed_args=(1,)),
    torch.divide: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("input", "other"), fixed_args=(1,)
    ),
    torch.true_divide: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("input", "other"), fixed_args=(1,)
    ),
    torch.Tensor.div: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)),
    torch.Tensor.divide: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)
    ),
    torch.Tensor.true_divide: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)
    ),
    torch.Tensor.div_: _OpSpec(OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)),
    torch.Tensor.__truediv__: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)
    ),
    torch.Tensor.__rtruediv__: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(0,)
    ),
    torch.Tensor.__itruediv__: _OpSpec(
        OpLinearity.LINEAR, (0, 1), ("other",), fixed_args=(1,)
    ),
    # =================================================================
    # MULTILINEAR -- multiplication
    # =================================================================
    torch.mul: _OpSpec(OpLinearity.MULTILINEAR),
    torch.multiply: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.mul: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.multiply: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.mul_: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.__mul__: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.__rmul__: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.__imul__: _OpSpec(OpLinearity.MULTILINEAR),
    # =================================================================
    # MULTILINEAR -- matmul / mm / bmm
    # =================================================================
    torch.matmul: _OpSpec(OpLinearity.MULTILINEAR),
    torch.mm: _OpSpec(OpLinearity.MULTILINEAR),
    torch.bmm: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.matmul: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.mm: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.bmm: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.__matmul__: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.__rmatmul__: _OpSpec(OpLinearity.MULTILINEAR),
    # =================================================================
    # MULTILINEAR -- einsum, dot, mv, etc.
    # =================================================================
    torch.einsum: _OpSpec(OpLinearity.MULTILINEAR),
    torch.dot: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.dot: _OpSpec(OpLinearity.MULTILINEAR),
    torch.mv: _OpSpec(OpLinearity.MULTILINEAR),
    torch.Tensor.mv: _OpSpec(OpLinearity.MULTILINEAR),
}

# Type-level decompositions for compound ops.
# Each function mirrors the original op's signature (taking one LocalSpmdType
# per tensor arg) and returns the output type, using _linear/_multilinear to
# describe the algebraic structure.  These mirror PyTorch decompositions in
# fbcode/caffe2/torch/_decomp/decompositions.py but operate purely on types.


def _addmm_types(
    self_t: LocalSpmdType, mat1_t: LocalSpmdType, mat2_t: LocalSpmdType
) -> LocalSpmdType:
    # addmm(self, mat1, mat2) = self + mm(mat1, mat2)
    return _linear(_multilinear(mat1_t, mat2_t), self_t)


def _addmv_types(
    self_t: LocalSpmdType, mat_t: LocalSpmdType, vec_t: LocalSpmdType
) -> LocalSpmdType:
    # addmv(self, mat, vec) = self + mv(mat, vec)
    return _linear(_multilinear(mat_t, vec_t), self_t)


def _addbmm_types(
    self_t: LocalSpmdType, batch1_t: LocalSpmdType, batch2_t: LocalSpmdType
) -> LocalSpmdType:
    # addbmm(self, batch1, batch2) = self + bmm(batch1, batch2).sum(0)
    # sum is LINEAR so the intermediate doesn't change the type
    return _linear(_multilinear(batch1_t, batch2_t), self_t)


def _baddbmm_types(
    self_t: LocalSpmdType, batch1_t: LocalSpmdType, batch2_t: LocalSpmdType
) -> LocalSpmdType:
    # baddbmm(self, batch1, batch2) = self + bmm(batch1, batch2)
    return _linear(_multilinear(batch1_t, batch2_t), self_t)


def _addr_types(
    self_t: LocalSpmdType, vec1_t: LocalSpmdType, vec2_t: LocalSpmdType
) -> LocalSpmdType:
    # addr(self, vec1, vec2) = self + outer(vec1, vec2)
    return _linear(_multilinear(vec1_t, vec2_t), self_t)


_DECOMP_TYPE_RULES: dict[Callable, Callable[..., LocalSpmdType]] = {
    torch.addmm: _addmm_types,
    torch.Tensor.addmm: _addmm_types,
    torch.addmv: _addmv_types,
    torch.Tensor.addmv: _addmv_types,
    torch.addbmm: _addbmm_types,
    torch.Tensor.addbmm: _addbmm_types,
    torch.baddbmm: _baddbmm_types,
    torch.Tensor.baddbmm: _baddbmm_types,
    torch.addr: _addr_types,
    torch.Tensor.addr: _addr_types,
}


_SCALAR_TYPES = (int, float, complex, bool, str, type(None))
_LEAF_TYPES = (torch.Tensor, *_SCALAR_TYPES)


def _iter_tensors_in(val: object):
    """Yield tensors from *val*, fast-pathing common leaf/flat cases.

    Direct tensors and flat list/tuple of tensors are handled without pytree.
    Anything else (nested containers, custom pytree-registered types) falls
    back to ``torch.utils._pytree.tree_flatten``.
    """
    if isinstance(val, torch.Tensor):
        yield val
        return
    if isinstance(val, (list, tuple)):
        if all(isinstance(item, _LEAF_TYPES) for item in val):
            # Fast path: flat list/tuple of tensors and scalar leaves.
            for item in val:
                if isinstance(item, torch.Tensor):
                    yield item
            return
        # Unknown or nested types present -- flatten via pytree to handle
        # nested containers and custom pytree-registered types.
        flat, _ = torch.utils._pytree.tree_flatten(val)
        for item in flat:
            if isinstance(item, torch.Tensor):
                yield item
        return
    if isinstance(val, _SCALAR_TYPES):
        return
    # Unknown type -- fall back to pytree.
    flat, _ = torch.utils._pytree.tree_flatten(val)
    for item in flat:
        if isinstance(item, torch.Tensor):
            yield item


class _RawArgEntry(NamedTuple):
    """One argument (positional or keyword) captured for error display."""

    location: str | None
    value: object


class _ArgInfo(NamedTuple):
    """Classification and collected types from all tensor arguments.

    Produced by ``_classify_args`` in a single pass over args/kwargs.

    Attributes:
        tensor_types: Types from all tensors (in iteration order).
            Untyped tensors contribute ``{}`` (unknown on all axes).
        raw_entries: All arguments (tensor and non-tensor), in call order,
            captured so error formatting can be deferred until a type error
            actually occurs.
        partition_specs: PartitionSpec for each tensor (same order as
            tensor_types). None for tensors without a PartitionSpec.
    """

    tensor_types: list[LocalSpmdType]
    raw_entries: list[_RawArgEntry]
    partition_specs: list[PartitionSpec | None]


def _format_tensor_for_context(t: torch.Tensor) -> str:
    """Format a single tensor for error context display."""
    from torch.utils._dtype_abbrs import dtype_abbrs

    lt = get_local_type(t)
    dtype_str = dtype_abbrs.get(t.dtype, str(t.dtype).removeprefix("torch."))
    shape_str = ", ".join(str(d) for d in t.shape)
    type_items = ", ".join(f"{format_axis(axis)}: {typ!r}" for axis, typ in lt.items())
    result = f"{dtype_str}[{shape_str}] {{{type_items}}}"
    ps = get_partition_spec(t)
    if ps is not None:
        result += f" {ps}"
    return result


def _format_non_tensor_for_context(val: object) -> str:
    """Format a non-tensor argument for error context display.

    Special-cases ProcessGroup to show the mesh axis name instead of the
    raw repr.
    """
    if isinstance(val, _dist.dist.ProcessGroup):
        return repr(normalize_axis(val))
    return repr(val)


def _format_arg_for_context(val: object) -> str:
    """Format a single argument value for error context display.

    Handles tensors, lists of tensors, ProcessGroups, and other values.
    """
    if isinstance(val, torch.Tensor):
        return _format_tensor_for_context(val)
    if isinstance(val, (list, tuple)):
        has_tensors = any(isinstance(item, torch.Tensor) for item in val)
        if has_tensors:
            bracket = "[" if isinstance(val, list) else "("
            close = "]" if isinstance(val, list) else ")"
            items = []
            for item in val:
                if isinstance(item, torch.Tensor):
                    items.append(_format_tensor_for_context(item))
                else:
                    items.append(_format_non_tensor_for_context(item))
            # If all items are identical, abbreviate: [item] * N
            if len(items) > 1 and all(it == items[0] for it in items):
                return f"{bracket}{items[0]}{close} * {len(items)}"
            # Multi-line format; items on separate lines.
            inner = ",\n".join(items)
            return bracket + "\n" + inner + ",\n" + close
    return _format_non_tensor_for_context(val)


def _get_param_names(func: Callable) -> list[str] | None:
    """Try to get positional parameter names from a function signature.

    Returns None if the signature cannot be inspected (e.g., C builtins).
    """
    import inspect

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return None
    return [
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]


def _classify_args(args: tuple, kwargs: dict) -> _ArgInfo:
    """Classify all tensor arguments in a single pass.

    Iterates every tensor in ``args`` and ``kwargs`` (skipping ``out=``),
    collecting ``get_local_type`` and ``get_partition_spec`` for each.
    Untyped tensors contribute ``{}`` (unknown on all axes).

    Also records the original call arguments so error formatting can be
    deferred until a type error actually occurs.

    Args:
        args: Positional arguments (post-Scalar-unwrapping).
        kwargs: Keyword arguments (post-Scalar-unwrapping).
    """
    tensor_types: list[LocalSpmdType] = []
    raw_entries: list[_RawArgEntry] = []
    partition_specs: list[PartitionSpec | None] = []

    for arg in args:
        for t in _iter_tensors_in(arg):
            tensor_types.append(get_local_type(t))
            partition_specs.append(get_partition_spec(t))
        raw_entries.append(_RawArgEntry(None, arg))
    for key, v in kwargs.items():
        if key == "out":
            continue
        for t in _iter_tensors_in(v):
            tensor_types.append(get_local_type(t))
            partition_specs.append(get_partition_spec(t))
        raw_entries.append(_RawArgEntry(key, v))

    return _ArgInfo(
        tensor_types,
        raw_entries,
        partition_specs,
    )


def _format_operator_context(
    func: Callable,
    raw_entries: list[_RawArgEntry],
    mesh: frozenset[MeshAxis] | None = None,
) -> str:
    """Format an operator context string for error messages.

    Produces a multi-line block showing all arguments (tensor and
    non-tensor) like::

        In all_gather_into_tensor(
          output_tensor: f32[64] {},
          input_tensor: f32[32] {},
          group: DP,
        )

    Positional args use parameter names when available from the function
    signature, otherwise they fall back to ``args[i]``. Keyword args use
    their key name directly. ProcessGroup arguments are shown by their
    mesh axis name. Lists of tensors are shown with each element
    formatted inline.
    """
    #   4 spaces = arg indent (items in the function call)
    #   6 spaces = nested indent (items inside a list/tuple arg)
    _ARG_INDENT = "    "
    _NESTED_INDENT = "      "

    mesh_suffix = ""
    if mesh is not None:
        mesh_suffix = f" under mesh {_format_axis_set(mesh)}"

    op_name = getattr(func, "__name__", repr(func))
    if not raw_entries:
        return f"  In {op_name}(){mesh_suffix}"
    param_names = _get_param_names(func)
    parts = []
    positional_index = 0
    for entry in raw_entries:
        formatted = _format_arg_for_context(entry.value)
        if entry.location is not None:
            raw = f"{entry.location}: {formatted}"
        else:
            if param_names is not None and positional_index < len(param_names):
                label = param_names[positional_index]
            else:
                label = f"args[{positional_index}]"
            positional_index += 1
            raw = f"{label}: {formatted}"
        if "\n" not in raw:
            parts.append(f"{_ARG_INDENT}{raw}")
        else:
            # Multi-line value (e.g., list of tensors).  First line stays
            # at arg indent, middle lines at nested indent, last line
            # (closing bracket) back at arg indent.
            lines = raw.split("\n")
            result = f"{_ARG_INDENT}{lines[0]}"
            for line in lines[1:-1]:
                result += f"\n{_NESTED_INDENT}{line}"
            result += f"\n{_ARG_INDENT}{lines[-1]}"
            parts.append(result)
    joined = ",\n".join(parts)
    return f"  In {op_name}(\n{joined},\n  ){mesh_suffix}"


# Deterministic factory ops whose output is identical on every rank
# (given the same arguments).  When a current mesh is set, these get
# automatic Replicate on every mesh axis.  Ops not in this set (e.g.
# torch.randn, torch.empty) stay unannotated ({}).
_DETERMINISTIC_FACTORIES: set[Callable] = {
    torch.zeros,
    torch.ones,
    torch.full,
    torch.zeros_like,
    torch.ones_like,
    torch.full_like,
    torch.arange,
    torch.linspace,
    torch.logspace,
    torch.eye,
}


def _deterministic_factory_type(func: Callable) -> LocalSpmdType:
    """Return Replicate type for deterministic factory ops under a current mesh.

    If ``func`` is a deterministic factory (torch.zeros, torch.ones, etc.)
    and a current mesh is set, returns R on every mesh axis.  Otherwise
    returns {} (typed but unknown on all axes).
    """
    mesh = current_mesh()
    if mesh is not None and func in _DETERMINISTIC_FACTORIES:
        return {axis: R for axis in mesh}
    return {}


def _is_numeric_scalar(val: object) -> bool:
    """Return True if val is a numeric scalar (int/float/complex, not bool)."""
    return isinstance(val, (int, float, complex)) and not isinstance(val, bool)


def _collect_scalar_types(  # noqa: C901
    tensor_types: list[LocalSpmdType],
    original_args: tuple,
    original_kwargs: dict,
    spec: _OpSpec | None,
) -> list[LocalSpmdType]:
    """Append scalar types to tensor types based on op spec positions.

    Numeric scalars at declared tensor-input positions are included with
    ``_Scalar`` on all known mesh axes.  ``Scalar`` wrapper objects use their
    exact SPMD type.

    Additionally, ``Scalar`` wrapper objects at *any* argument position
    (not just tensor-input positions) participate in type inference.  This
    handles cases like ``narrow(tensor, dim, Scalar(V), Scalar(V))`` where
    the start/length are not tensor args but still carry rank-dependent
    values.  Plain numeric scalars at non-tensor-arg positions are ignored
    (they are structural parameters like ``dim``).

    When ``spec`` is ``None`` (unregistered op), only ``Scalar`` wrappers
    are collected from all argument positions.

    Returns a new list with scalar types appended (or ``tensor_types``
    unchanged if no scalars are found).

    Args:
        tensor_types: Types already collected from typed tensors.
        original_args: Positional arguments (pre-Scalar-unwrapping, so
            ``Scalar`` objects are visible).
        original_kwargs: Keyword arguments (pre-Scalar-unwrapping).
        spec: Op specification declaring which positions are tensor inputs,
            or ``None`` for unregistered ops.
    """
    all_axes: set[DeviceMeshAxis] = set()
    for typ in tensor_types:
        all_axes.update(typ.keys())

    for a in (*original_args, *original_kwargs.values()):
        if isinstance(a, Scalar):
            all_axes.update(get_local_type(a).keys())
        elif isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, Scalar):
                    all_axes.update(get_local_type(item).keys())

    if not all_axes:
        return tensor_types

    scalar_type: LocalSpmdType = {axis: _Scalar for axis in all_axes}
    extra: list[LocalSpmdType] = []

    def _check(val: object) -> None:
        if isinstance(val, Scalar):
            extra.append(get_local_type(val))
        elif _is_numeric_scalar(val):
            extra.append(scalar_type)
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, Scalar):
                    extra.append(get_local_type(item))
                elif _is_numeric_scalar(item):
                    extra.append(scalar_type)

    def _check_scalar_only(val: object) -> None:
        """Check for Scalar wrappers only (ignore plain numeric scalars)."""
        if isinstance(val, Scalar):
            extra.append(get_local_type(val))
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, Scalar):
                    extra.append(get_local_type(item))

    if spec is not None:
        # Collect from declared tensor-input positions (Scalar + plain numeric).
        tensor_arg_positions: set[int] = set(spec.tensor_args)
        for i in spec.tensor_args:
            if i < len(original_args):
                _check(original_args[i])
        if spec.tensor_varargs_from is not None:
            for i in range(spec.tensor_varargs_from, len(original_args)):
                tensor_arg_positions.add(i)
                _check(original_args[i])
        for name in spec.tensor_kwargs:
            if name in original_kwargs:
                _check(original_kwargs[name])

        # Collect Scalar wrappers from non-tensor-arg positions.  Plain numeric
        # scalars here are structural (e.g. dim) and don't carry SPMD types.
        for i, arg in enumerate(original_args):
            if i not in tensor_arg_positions:
                _check_scalar_only(arg)
        for name, val in original_kwargs.items():
            if name not in spec.tensor_kwargs:
                _check_scalar_only(val)
    else:
        # No spec; scan all args for Scalar wrappers only.
        for a in (*original_args, *original_kwargs.values()):
            _check_scalar_only(a)

    if extra:
        return list(tensor_types) + extra
    return tensor_types


def _get_mutated_tensors(  # noqa: C901
    func: Callable,
    args: tuple,
    kwargs: dict,
    result: object,
) -> list[torch.Tensor]:
    """Detect tensors being mutated/written-to by this operation.

    Uses three heuristics (any match is sufficient):
    1. An input tensor is identical to the result (result is args[i]).
    2. The function name has a trailing underscore (PyTorch in-place convention),
       excluding dunder methods but including __iadd__ etc.
    3. An out= keyword argument was provided.

    Returns:
        List of mutated tensors (deduplicated by identity).
    """
    mutated: list[torch.Tensor] = []
    seen_ids: set[int] = set()

    def _add(t: torch.Tensor) -> None:
        tid = id(t)
        if tid not in seen_ids:
            seen_ids.add(tid)
            mutated.append(t)

    # (3) out= kwarg
    out = kwargs.get("out")
    if out is not None:
        if isinstance(out, torch.Tensor):
            _add(out)
        elif isinstance(out, (list, tuple)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    _add(item)

    # (1) Identity: result is one of the input tensors
    if isinstance(result, torch.Tensor):
        for arg in args:
            if isinstance(arg, torch.Tensor) and result is arg:
                _add(result)
                break

    # (2) Trailing underscore (fallback when identity/out didn't trigger).
    # Catches method-style in-place ops (add_, zero_) and dunder in-place
    # ops (__iadd__, __imul__). PASSTHROUGH functions (requires_grad_, etc.)
    # never reach this code path.
    if not mutated:
        func_name = getattr(func, "__name__", "")
        is_inplace_name = (
            func_name.endswith("_") and not func_name.startswith("__")
        ) or (
            func_name.startswith("__i")
            and func_name.endswith("__")
            and len(func_name) > 5
        )
        if is_inplace_name and args and isinstance(args[0], torch.Tensor):
            _add(args[0])

    return mutated


def _validate_mutation_types(
    func: Callable,
    mutated_tensors: list[torch.Tensor],
    output_type: LocalSpmdType,
) -> None:
    """Validate that mutated tensors' existing SPMD types match the output type.

    For mutating/out operations, the operation writes into an existing tensor.
    The output SPMD type must match the mutated tensor's existing type on every
    axis; otherwise the mutation would silently change the tensor's SPMD type,
    which is unsound (other references to the tensor still expect the old type).

    Args:
        func: The torch function (for error messages).
        mutated_tensors: Tensors being mutated by this operation.
        output_type: The inferred output SPMD type.

    Raises:
        SpmdTypeError: If any mutated tensor's type conflicts with output_type.
    """
    func_name = getattr(func, "__name__", repr(func))
    for t in mutated_tensors:
        existing = get_local_type(t)
        for axis, new_typ in output_type.items():
            old_typ = existing.get(axis)
            if old_typ is not None and old_typ is not new_typ:
                # Allow R->V and I->V for non-grad tensors: replicated/invariant
                # data becoming varying after in-place mutation with varying
                # inputs is sound -- the data was identical across ranks and
                # now differs, which is exactly what V means.  We restrict
                # this to tensors that do not require grad because in-place
                # type changes on autograd leaves would silently alter the
                # gradient type contract.
                if old_typ in (R, I) and new_typ is V and not t.requires_grad:
                    continue
                raise SpmdTypeError(
                    f"{func_name}: in-place/out operation would change "
                    f"SPMD type on axis {format_axis(axis)} from {old_typ} "
                    f"to {new_typ}. In-place and out= operations cannot "
                    f"change a tensor's SPMD type."
                )


def _set_result_type(result: object, output_type: LocalSpmdType) -> None:
    """Set SPMD types on the result tensor(s).

    Handles single tensors, flat list/tuple of tensors, and arbitrary nested
    structures (e.g., NamedTuples from ops like torch.linalg.lu_factor).
    The common cases (single tensor, flat sequence) are checked first to
    avoid the overhead of pytree flattening.
    """

    def _apply(t: torch.Tensor) -> None:
        _set_local_type(t, output_type)

    if isinstance(result, torch.Tensor):
        _apply(result)
        return
    if isinstance(result, (list, tuple)):
        all_flat = True
        for item in result:
            if isinstance(item, torch.Tensor):
                _apply(item)
            elif isinstance(item, (list, tuple, dict)):
                all_flat = False
        if all_flat:
            return
    # Fall back to pytree for nested structures.
    flat, _ = torch.utils._pytree.tree_flatten(result)
    for item in flat:
        if isinstance(item, torch.Tensor):
            _apply(item)


def _apply_fixed_args(  # noqa: C901
    func: Callable,
    args: tuple,
    kwargs: dict,
    spec: _OpSpec,
    input_types_list: list[LocalSpmdType],
) -> list[LocalSpmdType]:
    """Filter fixed_args for LINEAR ops when Partial is present.

    ``fixed_args`` lists positional arg indices that must be held fixed (not P)
    for the op to be linear in the remaining args.  For example, ``div(a, b)``
    is linear in ``a`` when ``b`` is fixed, so ``fixed_args=(1,)``.

    When P is present among the non-fixed tensor args:
    1. Validate that tensor args at fixed_args positions don't have P on any axis.
    2. Exclude their types from the returned list so LINEAR inference sees only
       the "free" args (and doesn't reject P + R mixing).

    When no P is present in the free args, return the original list unchanged
    so normal inference rules apply (e.g., R + V -> V).

    Args:
        func: The torch function being called.
        args: Positional arguments to the torch operation.
        kwargs: Keyword arguments to the torch operation.
        spec: The op specification (must have non-empty fixed_args).
        input_types_list: The already-collected input types list.
    """
    # Identify which input_types_list entries came from fixed_args positions.
    # We re-walk args to figure out which types are "fixed" vs "free".
    fixed_positions = set(spec.fixed_args)
    free_types: list[LocalSpmdType] = []
    fixed_types: list[LocalSpmdType] = []

    for i in spec.tensor_args:
        if i < len(args):
            val = args[i]
            if isinstance(val, torch.Tensor):
                if i in fixed_positions:
                    fixed_types.append(get_local_type(val))
                else:
                    free_types.append(get_local_type(val))
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, torch.Tensor):
                        if i in fixed_positions:
                            fixed_types.append(get_local_type(item))
                        else:
                            free_types.append(get_local_type(item))

    # Check if P is present in any free arg on any axis
    has_p_in_free = any(P in typ.values() for typ in free_types)
    # Also check default V (missing from dict means V, not P) -- V != P, so fine.

    if not has_p_in_free:
        # No P in free args -- normal inference, include all types.
        return input_types_list

    # P is present in free args -- validate fixed args don't have P
    for fixed_type in fixed_types:
        for axis, typ in fixed_type.items():
            if typ is P:
                raise SpmdTypeError(
                    f"{func.__name__}: Partial type in fixed argument "
                    f"(denominator/divisor) on axis {format_axis(axis)} is not allowed. "
                    f"Division is only linear in the numerator."
                )

    # Exclude fixed_types from input_types_list: return only free_types
    # plus any scalar types that were appended by _collect_scalar_types.
    # scalar_types count = len(input_types_list) - len(free_types) - len(fixed_types)
    n_tensor_types = len(free_types) + len(fixed_types)
    scalar_types = input_types_list[n_tensor_types:]
    return free_types + scalar_types


# Every function in this registry must accept (x: Tensor, axis, *, src=..., dst=...)
# as its leading arguments, since __torch_function__ recovers src/dst from
# args[0], args[1], and kwargs.
_SPMD_FUNCTION_DEFAULTS: dict[Callable, dict[str, PerMeshAxisSpmdType | None]] = {
    all_reduce: {"src": P, "dst": None},
    all_gather: {"src": V, "dst": None},
    reduce_scatter: {"src": P, "dst": V},
    all_to_all: {"src": V, "dst": V},
    redistribute: {"src": None, "dst": None},
    reinterpret: {"src": None, "dst": None},
    convert: {"src": None, "dst": None},
    invariant_to_replicate: {"src": I, "dst": R},
}


def _resolve_collective_axes(
    axis: MeshAxis,
    local_type: LocalSpmdType,
) -> list[MeshAxis] | None:
    """Resolve a collective's mesh axis to the matching axes on a tensor.

    Returns [axis] if axis is directly in local_type.
    Returns [sub1, sub2, ...] if axis is a flattened version of sub-axes
    that are all present in local_type.
    Returns None if no match is found.
    """
    if axis in local_type:
        return [axis]

    # Try flattened decomposition: check if sub-axes in local_type
    # flatten to exactly this axis.
    from spmd_types._mesh_axis import flatten_axes

    candidates = [a for a in local_type if a < axis]
    if not candidates:
        return None
    if flatten_axes(tuple(candidates)) == axis:
        return candidates
    return None


# =============================================================================
# Global SPMD Shard Propagation (via DTensor ShardingPropagator)
# =============================================================================


def _set_result_partition_spec(
    result: object,
    partition_specs: list[PartitionSpec | None],
) -> None:
    """Set partition specs on result tensor(s).

    Mirrors ``_set_result_type``: accepts the raw (unflattened) result and
    walks it to apply per-leaf partition specs.
    """
    flat, _ = torch.utils._pytree.tree_flatten(result)
    assert len(partition_specs) == len(flat), (
        f"PartitionSpec list length {len(partition_specs)} "
        f"does not match flattened result length {len(flat)}"
    )
    for item, spec in zip(flat, partition_specs):
        if isinstance(item, torch.Tensor):
            _set_partition_spec(item, spec)


def _collect_shard_axes(
    partition_specs: list[PartitionSpec | None],
    global_axes: set[MeshAxis],
) -> tuple[list[MeshAxis], dict[MeshAxis, set[MeshAxis]]]:
    """Collect global shard axes from input PartitionSpecs in topological order.

    Scans the pre-collected PartitionSpecs, extracts ordering constraints from
    adjacent pairs in multi-axis tuples (e.g., ``('tp', 'dp', 'ep')`` yields
    edges tp->dp, dp->ep), and returns a topologically sorted list of global
    axes that are sharded (i.e., appear in at least one input's PartitionSpec).
    The function also validates that ordering constraints are consistent across
    all inputs (no conflicts such as tp before dp in one input but dp before tp
    in another).

    Args:
        partition_specs: Pre-collected PartitionSpecs from ``_classify_args``
            (one per input tensor, None for tensors without a PartitionSpec).
        global_axes: The set of axes in global SPMD mode.

    Returns:
        A tuple of (sorted_axes, edges):
        - ``sorted_axes``: Topologically sorted list of global shard axes. When
          multiple valid orderings exist, axes are ordered by first appearance
          in the input PartitionSpecs.
        - ``edges``: Dict mapping each axis to the set of axes that must come
          after it (ordering constraints from input PartitionSpec tuples). Used
          by the caller to verify that axes sharing the same output dim have a
          defined ordering.

        **Why a full topsort is safe**: the returned order only affects tuple
        ordering within output PartitionSpec entries. Two axes A and B can only
        appear in the same output tuple (e.g., ``(A, B)``) if they are reachable
        from each other in the edge graph, which determines their relative
        order. If no path connects them, they land on different dims and their
        relative order is never observable. So even though the topsort is more
        constrained than a partial order, the extra constraint never changes the
        result. Callers must not rely on the specific linear order for other
        purposes.

    Raises:
        SpmdTypeError: If pairwise ordering conflicts are detected.
    """
    shard_axes_seen: list[MeshAxis] = []
    seen: set[MeshAxis] = set()
    # edges[a] = {b} means a must come before b.
    edges: dict[MeshAxis, set[MeshAxis]] = {}

    for ps in partition_specs:
        if ps is None:
            continue
        for entry in ps:
            if entry is None:
                continue
            if isinstance(entry, tuple):
                global_in_entry = [a for a in entry if a in global_axes]
                # Record adjacent ordering constraints.
                for i in range(len(global_in_entry) - 1):
                    a, b = global_in_entry[i], global_in_entry[i + 1]
                    # Check for conflicts: a before b here, but b before a
                    # in another input (direct or transitive).
                    if _is_reachable(b, a, edges):
                        raise SpmdTypeError(
                            f"PartitionSpec device ordering conflict: "
                            f"{b} before {a} vs {a} before {b}"
                        )
                    edges.setdefault(a, set()).add(b)
                axes = global_in_entry
            else:
                axes = [entry] if entry in global_axes else []
            for a in axes:
                if a not in seen:
                    shard_axes_seen.append(a)
                    seen.add(a)

    # Topological sort using Kahn's algorithm, with first-seen order as
    # tiebreaker so that axes without ordering constraints preserve their
    # natural appearance order.
    if not edges:
        return shard_axes_seen, edges

    # Precompute position for O(1) ordering lookups.
    pos: dict[MeshAxis, int] = {a: i for i, a in enumerate(shard_axes_seen)}

    # Build in-degree map over shard axes only.
    in_degree: dict[MeshAxis, int] = dict.fromkeys(shard_axes_seen, 0)
    for a in shard_axes_seen:
        for b in edges.get(a, set()):
            if b in in_degree:
                in_degree[b] += 1

    # Seed queue with zero-in-degree axes in first-seen order.
    # Use a heap keyed by first-seen position for O(log n) insert/pop.
    import heapq

    heap: list[tuple[int, MeshAxis]] = [
        (pos[a], a) for a in shard_axes_seen if in_degree[a] == 0
    ]
    heapq.heapify(heap)
    result: list[MeshAxis] = []
    while heap:
        _, node = heapq.heappop(heap)
        result.append(node)
        for neighbor in edges.get(node, set()):
            if neighbor not in in_degree:
                continue
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, (pos[neighbor], neighbor))

    if len(result) != len(shard_axes_seen):
        # Cycle detected -- should not happen if PartitionSpec entries are
        # consistent, but report clearly just in case.
        missing = set(shard_axes_seen) - set(result)
        raise SpmdTypeError(
            f"Cyclic ordering constraint among global shard axes: "
            f"{', '.join(format_axis(a) for a in missing)}"
        )

    return result, edges


def spmd_type_to_dtensor_placement(
    typ: PerMeshAxisSpmdType,
) -> dtensor_type.Placement:
    """Convert our per-axis SPMD type to a DTensor Placement.

    I (Invariant) maps to Replicate since DTensor has no Invariant concept.
    The caller is responsible for recovering I after calling the inverse
    operation ``dtensor_placement_to_spmd_type`` (which maps Replicate back
    to R, not I).
    """
    if isinstance(typ, Shard):
        return dtensor_type.Shard(typ.dim)
    if typ is R or typ is I:
        return dtensor_type.Replicate()
    if typ is P:
        # Only DTensor Partial('sum') is supported.
        return dtensor_type.Partial()
    raise ValueError(f"Cannot convert {typ} to DTensor placement")


def dtensor_placement_to_spmd_type(
    placement: dtensor_type.Placement,
) -> PerMeshAxisSpmdType:
    """Convert a DTensor Placement to our per-axis SPMD type.

    Replicate maps to R (not I); the caller must recover I if needed.
    """
    if isinstance(placement, dtensor_type.Shard):
        return Shard(placement.dim)
    if isinstance(placement, dtensor_type.Replicate):
        return R
    if isinstance(placement, dtensor_type.Partial):
        if placement.reduce_op != "sum":
            raise ValueError(
                f"Only sum-Partial is supported, got reduce_op={placement.reduce_op!r}"
            )
        return P
    raise ValueError(f"Cannot convert DTensor placement {placement}")


class _ShardPropagator:
    """Per-axis shard propagator using DTensor meta dispatch.

    Creates meta DTensors with the appropriate placements, runs the torch
    function, and reads the output placements. DTensor's own
    ``__torch_dispatch__`` handles sharding strategy selection and decomposition
    propagation naturally.

    Caches 1-D DeviceMeshes by axis dimension size.
    """

    def __init__(self):
        # Cache 1-D meshes by axis dimension size.
        self._meshes: dict[int, DeviceMesh] = {}

    def _get_mesh(self, axis_size: int) -> DeviceMesh:
        """Get or create a 1-D DeviceMesh of the given size."""
        cached = self._meshes.get(axis_size)
        if cached is not None:
            return cached
        mesh = DeviceMesh("cpu", torch.arange(axis_size), _init_backend=False, _rank=0)
        self._meshes[axis_size] = mesh
        return mesh

    def _to_meta_dtensor(
        self,
        arg: object,
        axis: MeshAxis,
        mesh: DeviceMesh,
    ) -> object:
        """Convert a tensor arg to a meta DTensor for shard propagation.

        Non-tensor args pass through unchanged. Untyped tensors become plain
        meta tensors (not DTensors, so DTensor dispatch ignores them).
        """
        if not isinstance(arg, torch.Tensor):
            return arg
        if not has_local_type(arg):
            return torch.empty(arg.shape, dtype=arg.dtype, device="meta")

        # If the tensor has a PartitionSpec on this axis, use that. Otherwise
        # fall back to the local SPMD type (R, I, or P).
        shard = partition_spec_get_shard(get_partition_spec(arg), axis)
        typ = shard if shard is not None else get_local_type(arg)[axis]
        # I (Invariant) is lost here -- it maps to Replicate since DTensor
        # has no Invariant concept. It will be recovered based on local
        # SPMD propagation in _validate_and_update_local_global_correspondence.
        placement = spmd_type_to_dtensor_placement(typ)
        local_meta = torch.empty(arg.shape, dtype=arg.dtype, device="meta")
        return DTensor.from_local(local_meta, mesh, [placement])

    def propagate(
        self,
        func: Callable,
        axis: MeshAxis,
        args: tuple,
        kwargs: dict | None = None,
    ) -> list[dtensor_type.Placement | None]:
        """Propagate shard types by running func on meta DTensors.

        Creates meta DTensors with the appropriate placements for the given
        axis, runs the torch function (DTensor dispatch handles propagation),
        and reads the output placements as raw DTensor Placements. Always
        returns a flat list with one entry per output leaf.

        Returns raw DTensor Placements (Shard, Replicate, Partial).
        """
        mesh = self._get_mesh(axis.size())

        # Disable LocalTensorMode during meta DTensor propagation.
        # DTensor.from_local calls mesh.get_coordinate() which, under
        # LocalTensorMode(N), tries to map all N simulated ranks into
        # the propagator's fake 1-D mesh.
        with maybe_disable_local_tensor_mode():
            meta_args = torch.utils._pytree.tree_map(
                lambda x: self._to_meta_dtensor(x, axis, mesh), args
            )
            meta_kwargs = torch.utils._pytree.tree_map(
                lambda x: self._to_meta_dtensor(x, axis, mesh),
                kwargs or {},
            )
            try:
                with ExplicitRedistributionContext(strict=True):
                    meta_result = func(*meta_args, **meta_kwargs)
            except RuntimeError as e:
                raise SpmdTypeError(
                    f"No valid sharding strategy for {func.__name__} "
                    f"on axis {format_axis(axis)}"
                ) from e

            # Extract per-leaf output placements.
            flat, _ = torch.utils._pytree.tree_flatten(meta_result)
            out: list[dtensor_type.Placement | None] = []
            for item in flat:
                if isinstance(item, DTensor):
                    assert len(item.placements) == 1, item.placements
                    out.append(item.placements[0])
                else:
                    out.append(None)
            return out


_shard_propagator = _ShardPropagator()


def _validate_and_update_local_global_correspondence(
    local_result: PerMeshAxisLocalSpmdType,
    per_tensor_output_on_axis: list[PerMeshAxisSpmdType | None],
    axis: MeshAxis,
) -> list[PerMeshAxisSpmdType | None]:
    """Validate that local SPMD and DTensor outputs correspond.

    Global SPMD propagation only runs for axes where PartitionSpec exists in the
    input args. Local SPMD handles R, I, V, and P propagation; DTensor is
    consulted for shard dimension tracking and Partial('sum') conversion.

    Expected DTensor placement type <-> spmd type correspondence: S(i)->V,
    P(sum)->P/V, R->R/I (refers to local spmd result to decide on R or I).

    Validates each output element in the flat list.
    """
    axis_str = format_axis(axis)

    def mismatch_msg(gr: PerMeshAxisSpmdType, idx: int, detail: str = "") -> str:
        return (
            f"Local SPMD produces {local_result} on axis {axis_str} but "
            f"DTensor produces {gr} for output {idx}{detail}. "
            f"This indicates a bug in the SPMD type checker or DTensor "
            f"shard propagation."
        )

    out: list[PerMeshAxisSpmdType | None] = []
    for idx, gr in enumerate(per_tensor_output_on_axis):
        if gr is None:
            out.append(None)
        elif local_result is I:
            assert gr is R, mismatch_msg(gr, idx)
            out.append(I)
        elif local_result is V:
            assert gr is V or gr is P or isinstance(gr, Shard), mismatch_msg(gr, idx)
            out.append(gr)
        else:
            mapped_local = to_local_type(gr)
            assert mapped_local == local_result, mismatch_msg(
                gr, idx, f" (maps to {mapped_local})"
            )
            out.append(gr)
    return out


def _is_reachable(
    src: MeshAxis,
    dst: MeshAxis,
    edges: dict[MeshAxis, set[MeshAxis]],
) -> bool:
    """Check if ``dst`` is reachable from ``src`` via directed edges (DFS)."""
    visited: set[MeshAxis] = set()

    def dfs(node: MeshAxis) -> bool:
        for neighbor in edges.get(node, set()):
            if neighbor == dst:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                if dfs(neighbor):
                    return True
        return False

    return dfs(src)


def _assert_shard_order_determined(
    shard_dict: dict[MeshAxis, PerMeshAxisSpmdType],
    shard_edges: dict[MeshAxis, set[MeshAxis]],
    func: Callable,
) -> None:
    """Assert that axes sharing the same output Shard dim have a determined order.

    If multiple axes produce Shard(d) for the same tensor dim d, their
    ordering must be determined by the edge graph from input PartitionSpec
    tuples. Each consecutive pair of axes on the same dim must be connected
    by a path in the edge graph (reachable in either direction). Since
    reachability is transitive, consecutive checks cover all pairs.

    For example, given input tuples ``(a, b, c)`` and ``(c, d)``, the edges
    are ``a->b, b->c, c->d``. If axes ``(a, d)`` share an output dim,
    ``a->d`` is reachable via ``a->b->c->d``, so the ordering is valid. But
    if axes ``(a, e)`` share an output dim with no path between them, the
    ordering is ambiguous and we raise an error.
    """
    # Group axes by their Shard dim.
    dim_to_axes: dict[int, list[MeshAxis]] = {}
    for axis, typ in shard_dict.items():
        assert isinstance(typ, Shard)
        dim_to_axes.setdefault(typ.dim, []).append(axis)

    # Check consecutive pairs; since reachability is transitive, this
    # covers all pairs. Check both directions since list order is arbitrary.
    for dim, axes in dim_to_axes.items():
        for i in range(len(axes) - 1):
            a, b = axes[i], axes[i + 1]
            if not _is_reachable(a, b, shard_edges) and not _is_reachable(
                b, a, shard_edges
            ):
                raise SpmdTypeError(
                    f"Op {func.__name__}: axes {format_axis(a)} and "
                    f"{format_axis(b)} both shard output dim {dim} but "
                    f"no input PartitionSpec provides an ordering "
                    f"constraint between them. This likely indicates a "
                    f"bug in combining per mesh axis shard propagation."
                )


def _infer_global_output_type(
    func: Callable,
    args: tuple,
    kwargs: dict | None,
    global_shard_axes: list[MeshAxis],
    flat_results: list[object],
    shard_edges: dict[MeshAxis, set[MeshAxis]] | None = None,
) -> tuple[
    list[dict[MeshAxis, dtensor_type.Placement] | None],
    list[PartitionSpec | None],
]:
    """Run DTensor shard propagation and produce output PartitionSpecs.

    For each axis in ``global_shard_axes``, runs DTensor propagation to
    determine how shard dimensions flow through the op. Only ``Shard(dim)``
    placements contribute to the output PartitionSpec; ``Replicate`` and
    ``Partial`` are recorded in the raw placements but do not affect the spec.

    ``global_shard_axes`` is the set of axes with S(i) on at least one input,
    containing only those that have S(i) on at least one input tensor
    (collected by ``_collect_shard_axes``). The list is ordered by tuple
    sequence from input PartitionSpecs: for ``('tp', 'dp')``, tp is first,
    dp second. Propagation follows this order and the output PartitionSpec
    preserves it, so if both axes land on the same dim the result is
    ``('tp', 'dp')``.

    Args:
        func: The torch function being propagated.
        args: Positional arguments to the torch function.
        kwargs: Keyword arguments to the torch function.
        global_shard_axes: Topologically sorted shard axes from
            ``_collect_shard_axes``.
        flat_results: Flattened output leaves from the op.
        shard_edges: Optional ordering constraints from ``_collect_shard_axes``.
            ``shard_edges[a]`` is the set of axes that must come after ``a``.
            When provided, verifies that axes sharing the same output dim
            have a defined ordering from the inputs. When ``None``, the
            check is skipped.

    Returns:
        A tuple of (placements, partition_specs), both flat lists with one
        entry per leaf of ``flat_results``:
        - ``placements``: per-output-leaf dict mapping axis to raw DTensor
          Placement (or None for non-tensor leaves). The caller should pass
          these to ``_validate_and_update_local_global_correspondence`` to
          check consistency with local SPMD results.
        - ``partition_specs``: ``PartitionSpec | None`` per output leaf.
    """
    n_outputs = len(flat_results)

    # Propagate global shard axes, collecting per-axis raw DTensor Placements.
    # Each entry is a flat list with one Placement per output leaf. Iteration
    # follows global_shard_axes (topologically sorted from input PartitionSpecs).
    # axis_order=global_shard_axes is passed to shard_types_to_partition_spec to
    # determine multi-axis tuple ordering.
    per_axis: dict[MeshAxis, list[dtensor_type.Placement | None]] = {}
    for axis in global_shard_axes:
        axis_out = _shard_propagator.propagate(func, axis, args, kwargs=kwargs)
        assert len(axis_out) == n_outputs, (
            f"Op {func.__name__} has {len(axis_out)} outputs on axis "
            f"{format_axis(axis)} but result has {n_outputs}."
        )
        per_axis[axis] = axis_out

    # --- Build flat output lists (one entry per output leaf) ---
    placements: list[dict[MeshAxis, dtensor_type.Placement] | None] = []
    partition_specs: list[PartitionSpec | None] = []
    for i, output_i in enumerate(flat_results):
        leaf_placements: dict[MeshAxis, dtensor_type.Placement] = {}
        shard_dict: dict[MeshAxis, PerMeshAxisSpmdType] = {}
        for axis, axis_pls in per_axis.items():
            p = axis_pls[i]
            if p is not None:
                leaf_placements[axis] = p
            if isinstance(p, dtensor_type.Shard):
                shard_dict[axis] = Shard(p.dim)
        if not isinstance(output_i, torch.Tensor):
            placements.append(None)
            partition_specs.append(None)
        else:
            placements.append(leaf_placements if leaf_placements else None)
            if shard_dict:
                # Verify that axes sharing the same output dim have an ordering
                # constraint from the input PartitionSpec tuples. If two axes A
                # and B both produce Shard(d) for the same dim d but neither is
                # reachable from the other in the edge graph, the output tuple
                # ordering would be arbitrary.
                if shard_edges is not None:
                    _assert_shard_order_determined(shard_dict, shard_edges, func)
                partition_specs.append(
                    shard_types_to_partition_spec(
                        shard_dict, output_i.ndim, axis_order=global_shard_axes
                    )
                )
            else:
                partition_specs.append(None)
    return placements, partition_specs


# =============================================================================
# TorchFunctionMode for SPMD Type Tracking
# =============================================================================

# Functions that are not tensor math -- autograd bookkeeping, metadata queries,
# etc.  These go through __torch_function__ but should not trigger type
# inference or strict-mode annotation checks.
#
# This set is intentionally exhaustive: any function that reaches
# __torch_function__ and is NOT in this set will be type-checked.  This
# ensures that unrecognized operations (e.g. raw torch.distributed
# collectives) are caught rather than silently bypassing the checker.
#
# Note: property accesses like .shape/.ndim are handled by the __get__
# check in __torch_function__, and .stride() is in PyTorch's
# get_ignored_functions() so it never reaches __torch_function__ at all.
_PASSTHROUGH = {
    # Autograd bookkeeping
    # TODO: Actually, backward probably can support type checking; in
    # particular we could put annotations on all of the resulting grad
    # tensors based on the typing of the leaf tensor.
    torch.Tensor.backward,
    torch.Tensor.requires_grad_,
    torch.Tensor.retain_grad,
    # Metadata queries (return non-tensor values)
    torch.Tensor.dim,
    torch.Tensor.element_size,
    torch.Tensor.get_device,
    torch.Tensor.is_complex,
    torch.Tensor.is_contiguous,
    torch.Tensor.is_floating_point,
    torch.Tensor.nelement,
    torch.Tensor.numel,
    torch.Tensor.size,
    torch.Tensor.untyped_storage,
    torch.numel,
}


# Import shared state from _state module to avoid circular dependencies
import spmd_types._state as _state  # noqa: E402
from spmd_types._raw_dist import RAW_DIST_RULES  # noqa: E402
from spmd_types._state import is_type_checking  # noqa: E402, F401

# =============================================================================
# Autograd Function.apply monkeypatch
# =============================================================================

# Set of autograd.Function subclasses whose apply is known to be local-only
# (i.e., each output element depends only on the corresponding input elements
# so the standard element-wise type propagation rule is safe).
_LOCAL_AUTOGRAD_FUNCTIONS: set[type] = set()

# Set of autograd.Function subclasses registered with a typecheck_forward
# staticmethod.  When .apply() is intercepted by __torch_function__,
# typecheck_forward is called INSTEAD of .apply().  It should assert_type()
# on inputs, call .apply() to execute, and assert_type() on the output.
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
    ``typecheck_forward`` on ``nn.Module`` subclasses::

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


# The original C++ descriptor for autograd.Function.apply, saved when the
# patch is installed and restored when the patch is removed.
_orig_autograd_apply = None


class _AutogradApplyDescriptor:
    """Wraps autograd.Function.apply to dispatch through __torch_function__.

    We wrap Function.apply (the Python classmethod that handles
    vmap/functorch dispatch), NOT _FunctionBase.apply (the raw C++
    apply).  This preserves the vmap-aware code path so that autograd
    functions with ``generate_vmap_rule=True`` continue to work under
    vmap while SpmdTypeMode is active.
    """

    def __init__(self, orig):
        self._orig = orig  # Original Function.apply classmethod

    def __get__(self, obj, cls=None):
        if cls is None and obj is not None:
            cls = type(obj)
        # Bind original apply to this specific Function subclass
        orig_method = self._orig.__get__(obj, cls)

        if not _state.is_type_checking():
            return orig_method

        def wrapper(*args, **kwargs):
            tensors = tuple(
                a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)
            )
            if tensors and has_torch_function(tensors):
                return handle_torch_function(orig_method, tensors, *args, **kwargs)
            return orig_method(*args, **kwargs)

        return wrapper


def _install_autograd_apply_patch():
    """Replace Function.apply with a descriptor that dispatches through __torch_function__."""
    global _orig_autograd_apply
    # Save the original classmethod from Function (NOT _FunctionBase).
    # Function.apply is a Python classmethod that handles vmap/functorch
    # dispatch; _FunctionBase.apply is the raw C++ apply that does not.
    _orig_autograd_apply = torch.autograd.Function.__dict__["apply"]
    torch.autograd.Function.apply = _AutogradApplyDescriptor(_orig_autograd_apply)


def _remove_autograd_apply_patch():
    """Restore the original Function.apply classmethod."""
    global _orig_autograd_apply
    if _orig_autograd_apply is not None:
        torch.autograd.Function.apply = _orig_autograd_apply
    _orig_autograd_apply = None


def _get_autograd_function_class(func) -> type | None:
    """Return the autograd.Function subclass if func is a bound apply method, else None."""
    if getattr(func, "__name__", None) != "apply":
        return None
    cls = getattr(func, "__self__", None)
    if cls is None or not isinstance(cls, type):
        return None
    if issubclass(cls, torch.autograd.Function):
        return cls
    return None


def _validate_partition_spec_for_global_spmd(
    local_type: LocalSpmdType,
    spec: PartitionSpec | None,
) -> None:
    """Validate PartitionSpec <-> local type consistency.

    Checks bidirectional consistency between a tensor's local SPMD types and
    its PartitionSpec:

    1. Every V-typed axis must have a PartitionSpec entry.
    2. Every PartitionSpec axis must have local type V.

    Args:
        local_type: The tensor's local SPMD type dict.
        spec: The tensor's PartitionSpec (None if absent).
    """
    spec_axes = spec.axes_with_partition_spec() if spec is not None else set()
    for axis, typ in local_type.items():
        if typ is V and axis not in spec_axes:
            ax = format_axis(axis)
            raise SpmdTypeError(
                f"Tensor has Varying type on axis {ax} but no PartitionSpec entry. "
                f"Use assert_type with S(i) or PartitionSpec to specify which "
                f"dimension is sharded. Otherwise, reinterpret the tensor from V to P."
            )
    for axis in spec_axes:
        typ = local_type.get(axis)
        if typ is not V:
            ax = format_axis(axis)
            raise SpmdTypeError(
                f"Tensor has PartitionSpec on axis {ax} but local type is "
                f"{typ!r}, not V. PartitionSpec (sharding) is only valid for Varying axes."
            )


class _SpmdTypeMode(torch.overrides.TorchFunctionMode):
    """
    TorchFunctionMode for tracking SPMD types on tensors.

    When active, this mode intercepts torch operations and propagates
    SPMD types from inputs to outputs according to the typing rules.

    For SPMD collectives and local ops (all_reduce, all_gather, etc.), the
    mode validates that the input tensor's type on the relevant mesh axis
    matches the declared ``src`` and sets the output type to ``dst``.  Types
    on all other mesh axes are copied through unchanged.

    Type checking runs *after* the function executes so that runtime errors
    (shape mismatches, invalid arguments) surface before type errors.

    Args:
        strict_mode: Controls the strictness of type checking.

            - ``"permissive"``: allows mixing annotated and unannotated
              tensor operands without error.  Useful during incremental
              annotation of an existing codebase.

            - ``"strict"`` (default): raises ``SpmdTypeError`` when a
              regular torch op mixes typed and untyped tensor operands.
              Ops with no typed tensor inputs (like ``torch.zeros``)
              produce tensors with ``{}`` type (typed but unknown on
              all axes) when no current mesh is set.  When a current
              mesh is set via ``set_current_mesh``, an allowlisted set
              of deterministic factory ops (``torch.zeros``,
              ``torch.ones``, ``torch.full``, ``torch.arange``, etc.)
              produce tensors annotated as Replicate on every mesh axis.
              All other factories (``torch.randn``, ``torch.empty``,
              etc.) still produce ``{}``.  If a factory receives a
              ``Scalar`` wrapper, the output type is inferred from the
              Scalar's annotations.  Without a mesh, combining a ``{}``
              tensor with a typed tensor that has axes raises
              ``SpmdTypeError`` because the ``{}`` tensor is missing
              those axes.  Call ``assert_type()`` to assign axes before
              combining.

        local: When True (default), only local SPMD types are tracked.
            When False, enables global SPMD mode where every axis with
            Varying type on a tensor must have a corresponding
            PartitionSpec entry specifying which tensor dimension is
            sharded. This ensures DTensor shard propagation can determine
            how shards flow through ops.
    """

    def __init__(
        self,
        *,
        strict_mode: Literal["permissive", "strict"] = "strict",
        local: bool = True,
    ):
        super().__init__()
        self._strict = strict_mode == "strict"
        self._local: bool = local

        self._disabled = False

    def __enter__(self):
        assert _current_mode() is None, (
            "_SpmdTypeMode must only be entered via typecheck()"
        )
        # FIXME: the autograd patch mutates a global class attribute
        # (torch.autograd.Function.apply).  If multiple threads enter/exit
        # concurrently, one thread may remove the patch while another is
        # still active.  This will need a lock + refcount when we support
        # multi-threaded type checking.
        _install_autograd_apply_patch()

        # Patch vmap so BatchedTensors carry SPMD type annotations.
        import spmd_types._vmap as _vmap

        _vmap.install()

        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = super().__exit__(exc_type, exc_val, exc_tb)
        _remove_autograd_apply_patch()

        import spmd_types._vmap as _vmap

        _vmap.uninstall()

        return result

    def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: C901
        kwargs = kwargs or {}
        # Paused via no_typecheck(): run without type checking.
        if self._disabled:
            return func(*args, **kwargs)

        # Property access (e.g. .grad, .data, .shape) -- not tensor math.
        if getattr(func, "__name__", None) == "__get__":
            return func(*args, **kwargs)

        # Autograd bookkeeping, metadata queries -- not tensor math.
        if func in _PASSTHROUGH:
            return func(*args, **kwargs)

        # Outer try/except: apply traceback filtering to SpmdTypeError.
        # _typecheck_core stamps e.context with operator info first,
        # then this except strips internal frames from the traceback.
        try:
            return self._typecheck_core(func, types, args, kwargs)
        except SpmdTypeError as e:
            _filter_and_reraise(e)
            raise

    def _typecheck_core(self, func, types, args=(), kwargs=None):  # noqa: C901
        kwargs = kwargs or {}
        # Unwrap Scalar objects to raw values for the actual function call,
        # but keep the original args for type collection.
        original_args, original_kwargs = args, kwargs
        args, kwargs = _unwrap_args(args, kwargs)

        # =============================================================
        # Classification phase: single pass over all tensor args.
        #
        # Hoisted before any dispatch so that _format_operator_context
        # is available in the except handler for ALL code paths
        # (typecheck_forward, DTensor, collectives, local ops).
        # =============================================================
        info = _classify_args(args, kwargs)

        try:
            # Global SPMD validation.
            if not self._local:
                for local_type, spec in zip(info.tensor_types, info.partition_specs):
                    _validate_partition_spec_for_global_spmd(local_type, spec)

            # Typecheck-registered autograd function: dispatch to typecheck_forward
            # INSTEAD of executing func.  The mode is popped off the
            # TorchFunctionMode stack during __torch_function__ dispatch (via
            # _pop_mode_temporarily in torch/overrides.py), so .apply() called
            # inside typecheck_forward executes normally without re-entering
            # this method.
            autograd_cls = _get_autograd_function_class(func)
            if (
                autograd_cls is not None
                and autograd_cls in _TYPECHECK_AUTOGRAD_FUNCTIONS
            ):
                return autograd_cls.typecheck_forward(*args, **kwargs)

            # DTensor tracks its own placement metadata; the SPMD type checker
            # should not interfere with DTensor operations.  This covers
            # DTensor arithmetic as well as DTensor-internal autograd functions
            # like _Redistribute (whose .apply() receives the DTensor directly).
            # Registered boundary functions (_ToTorchTensor, _FromTorchTensor)
            # are already handled above via _TYPECHECK_AUTOGRAD_FUNCTIONS.
            #
            # We check args (not types) because DTensor disables
            # __torch_function__ (_disabled_torch_function_impl), so
            # _get_overloaded_args never includes it in types.
            if any(
                isinstance(t, DTensor) for arg in args for t in _iter_tensors_in(arg)
            ):
                return func(*args, **kwargs)

            # Run the function, then type-check the result below.
            # Raw torch.distributed collectives are already type-checked above;
            # SPMD collectives and regular ops are checked after execution.
            result = func(*args, **kwargs)

            # =============================================================
            # Type checking phase.
            #
            # Every tensor's type is collected via get_local_type (untyped
            # tensors contribute {} -- unknown on all axes).  The per-axis
            # inference rules in infer_output_type will raise SpmdTypeError
            # when incompatible types are combined.
            # =============================================================
            if func in _SPMD_FUNCTION_DEFAULTS:
                # SPMD collective/reinterpret/convert: local first, then global.
                x = args[0]
                defaults = _SPMD_FUNCTION_DEFAULTS[func]
                axis = normalize_axis(args[1] if len(args) > 1 else kwargs["axis"])
                src = kwargs.get("src", defaults["src"])
                dst = kwargs.get("dst", defaults["dst"])

                # Decay Shard to Varying for local SPMD type checking.
                # S(i) is a global SPMD refinement; locally it behaves as V.
                local_src = to_local_type(src)
                local_dst = to_local_type(dst)

                # Validate input type on the axis matches src.
                # Special case: reductions (all_reduce, reduce_scatter) accept
                # V when src=P, implicitly reinterpreting V as P.  The runtime
                # in _collectives.py already handles this conversion.
                local_type = get_local_type(x)

                # Resolve axis to a list of matching axes on the tensor.
                # Direct match: [axis]. Flattened decomposition: [sub1, sub2, ...].
                resolved_axes = _resolve_collective_axes(axis, local_type)

                if resolved_axes is not None:
                    # Validate input types on all resolved axes
                    for ax in resolved_axes:
                        input_type = local_type[ax]
                        if local_src is not None and input_type != local_src:
                            # Allow V -> P implicit cast for reductions
                            if not (local_src is P and input_type is V):
                                raise SpmdTypeError(
                                    f"{func.__name__}: expected input type {local_src} on axis "
                                    f"{format_axis(ax)}, got {input_type}"
                                )
                elif axis.size() == 1:
                    pass  # singleton axes are never stored; skip
                elif _state.is_strict():
                    raise SpmdTypeError(
                        f"{func.__name__}: tensor has no type for axis "
                        f"{format_axis(axis)}. Use assert_type() to annotate "
                        f"the tensor on this axis before calling collectives."
                    )
                # else: permissive mode -- skip validation for this axis.

                # Build output types: copy all axes from input, override
                # resolved axes with local_dst.
                output_type = local_type.copy()
                if resolved_axes is not None and local_dst is not None:
                    for ax in resolved_axes:
                        output_type[ax] = local_dst
                _set_local_type(result, output_type)

                if _state._is_global():
                    input_spec = get_partition_spec(x)
                    # Find S(dim) for this axis in the input spec, validating
                    # it is innermost in its multi-axis group.
                    input_shard = None
                    if input_spec is not None:
                        for dim, entry in enumerate(input_spec):
                            if entry is None:
                                continue
                            if isinstance(entry, tuple):
                                if axis in entry:
                                    if axis != entry[-1]:
                                        group_str = (
                                            "("
                                            + ", ".join(format_axis(a) for a in entry)
                                            + ")"
                                        )
                                        # TODO: update error message to cover both
                                        # user error (suggest correct collective) and
                                        # expert mode (suggest local_map).
                                        raise RedistributeError(
                                            f"redistribute on axis {format_axis(axis)} is not allowed "
                                            f"because it is not the innermost axis in its PartitionSpec "
                                            f"group {group_str}. Only the innermost axis "
                                            f"({format_axis(entry[-1])}) can be directly redistributed."
                                        )
                                    input_shard = Shard(dim)
                                    break
                            elif entry == axis:
                                input_shard = Shard(dim)
                                break
                    # V on a global axis must be backed by S(i).
                    if input_type is V and input_shard is None:
                        raise SpmdTypeError(
                            f"{func.__name__}: axis {format_axis(axis)} is "
                            f"global but the input tensor has V without a "
                            f"corresponding S(i) in its PartitionSpec. Use "
                            f"assert_type() to set an S(i) type."
                        )
                    # Validate src S(i) matches input PartitionSpec.
                    if isinstance(src, Shard) and input_shard is not None:
                        if input_shard != src:
                            raise SpmdTypeError(
                                f"{func.__name__}: expected input shard {src} "
                                f"on axis {format_axis(axis)}, got "
                                f"{input_shard}"
                            )

                    # Build output PartitionSpec: replace this axis's shard.
                    new_output_spec = _update_axis_in_partition_spec(
                        input_spec,
                        axis,
                        dst if isinstance(dst, Shard) else None,
                        result.ndim,
                    )
                    _set_partition_spec(result, new_output_spec)

                if _TRACE:
                    _trace_op(
                        func,
                        info.tensor_types,
                        get_local_type(result) if has_local_type(result) else None,
                    )

            elif func in RAW_DIST_RULES:
                # Raw torch.distributed collective (e.g. all_gather_into_tensor).
                # The rule functions validate types internally.
                RAW_DIST_RULES[func](func, args, kwargs)

                if _TRACE:
                    _trace_op(func, info.tensor_types, None)

            elif (
                autograd_cls is not None
                and autograd_cls not in _LOCAL_AUTOGRAD_FUNCTIONS
            ):
                # Unregistered autograd Function: unknown semantics, cannot
                # safely assume element-wise behaviour.
                if _state.is_strict():
                    raise SpmdTypeError(
                        f"{autograd_cls.__name__}.apply: autograd Function "
                        f"is not registered for SPMD type "
                        f"checking. Use "
                        f"{register_local_autograd_function.__name__}("
                        f"{autograd_cls.__name__}) to mark it as safe "
                        f"for type propagation, and provide "
                        f"register_autograd_function("
                        f"{autograd_cls.__name__}) for a custom "
                        f"typecheck_forward."
                    )
                return result

            else:
                # Local op (regular op or registered local autograd Function):
                # infer output type from tensor types + scalars.
                spec = _OP_REGISTRY.get(func)
                input_types_list = list(info.tensor_types)
                input_types_list = _collect_scalar_types(
                    input_types_list, original_args, original_kwargs, spec
                )

                # Collect global shard axes in topologically sorted order from
                # input PartitionSpecs. This order is preserved through
                # propagation and into the output PartitionSpec.
                if self._local:
                    global_shard_axes, shard_edges = [], {}
                else:
                    all_axes: set[DeviceMeshAxis] = set()
                    for typ in input_types_list:
                        all_axes.update(typ.keys())
                    global_shard_axes, shard_edges = _collect_shard_axes(
                        info.partition_specs, all_axes
                    )

                # Local first: infer R/I/V/P output types for all axes.
                # Strict/permissive mode is enforced here (missing axis
                # annotations raise in strict, get skipped in permissive).
                # Global shard propagation runs after and operates only on axes
                # present in output_type (filtered below).
                if not input_types_list:
                    # No typed tensor inputs and no scalars -- factory op.
                    output_type = _deterministic_factory_type(func)
                elif (decomp_rule := _DECOMP_TYPE_RULES.get(func)) is not None:
                    output_type = decomp_rule(*input_types_list)
                else:
                    linearity = (
                        spec.linearity if spec is not None else OpLinearity.NONLINEAR
                    )
                    if spec is not None and spec.fixed_args:
                        input_types_list = _apply_fixed_args(
                            func, args, kwargs, spec, input_types_list
                        )
                    # NB: output_type may be further updated because global
                    # shard propagation below may promote local type V to P.
                    output_type = infer_output_type(
                        input_types_list, linearity=linearity
                    )

                # Validate mutation safety for in-place/out operations.
                mutated = _get_mutated_tensors(func, args, kwargs, result)
                if mutated:
                    _validate_mutation_types(func, mutated, output_type)

                _set_result_type(result, output_type)
                out = original_kwargs.get("out")
                if out is not None:
                    _set_result_type(out, output_type)

                # In permissive mode, local inference may skip axes that some
                # operands lack. Global shard propagation must also skip those
                # axes.
                global_shard_axes = [
                    ax for ax in global_shard_axes if ax in output_type
                ]

                if global_shard_axes:
                    # Global layer: overlay DTensor shard propagation for S(i)
                    # axes and set partition specs on the result.
                    flat_results, _ = torch.utils._pytree.tree_flatten(result)
                    raw_placements, output_specs = _infer_global_output_type(
                        func,
                        args,
                        kwargs,
                        global_shard_axes,
                        flat_results,
                        shard_edges=shard_edges,
                    )
                    # Validate local <-> global correspondence per axis.
                    for axis in global_shard_axes:
                        per_tensor_output_on_axis: list[PerMeshAxisSpmdType | None] = []
                        for per_leaf_placements in raw_placements:
                            if (
                                per_leaf_placements is None
                                or axis not in per_leaf_placements
                            ):
                                per_tensor_output_on_axis.append(None)
                            else:
                                per_tensor_output_on_axis.append(
                                    dtensor_placement_to_spmd_type(
                                        per_leaf_placements[axis]
                                    )
                                )
                        # NB: local spmd type assumes all output tensors have
                        # the same output_type[axis] on this axis. While global
                        # spmd type may have different types on different output
                        # tensors. This is needed to explicitly tell the S(?)
                        # from V.
                        validated_global_result = (
                            _validate_and_update_local_global_correspondence(
                                output_type[axis], per_tensor_output_on_axis, axis
                            )
                        )
                        # Upgrade V to P when global shard propagation
                        # determines the output is Partial.
                        if output_type[axis] is V and any(
                            vgr is P for vgr in validated_global_result
                        ):
                            assert all(vgr is P for vgr in validated_global_result), (
                                f"Inconsistent global results on axis {format_axis(axis)}: "
                                f"expected all P, got {validated_global_result}"
                            )
                            output_type[axis] = P
                            _set_result_type(result, output_type)
                            if out is not None:
                                _set_result_type(out, output_type)
                    _set_result_partition_spec(result, output_specs)

                if _TRACE:
                    _trace_op(func, info.tensor_types, output_type)

        except SpmdTypeError as e:
            if e.context is None:
                e.context = _format_operator_context(
                    func, info.raw_entries, mesh=current_mesh()
                )
            raise

        return result


@contextmanager
def typecheck(
    strict_mode: Optional[Literal["permissive", "strict"]] = None,
    local: bool | None = None,
):
    """Context manager that activates SPMD type checking.

    If type checking is already active, temporarily overrides ``strict_mode`` on
    the existing mode (reentrant no-op for the mode itself).  If not active,
    creates and enters a new ``_SpmdTypeMode``.

    Args:
        strict_mode: Controls the strictness of type checking.  See
            ``_SpmdTypeMode`` for details on each mode.  When ``None``
            (the default), inherits the current strict mode if reentrant,
            or uses ``"strict"`` when creating a fresh mode.
        local: When True (default), only local SPMD types are tracked.
            When False, enables global SPMD mode where every Varying axis
            must have a PartitionSpec entry. When None, inherits the
            current setting if reentrant, or uses True when creating a
            fresh mode.
    """
    existing = _current_mode()
    if existing is not None:
        old_strict = existing._strict
        old_disabled = existing._disabled
        old_local = existing._local
        if strict_mode is not None:
            existing._strict = strict_mode == "strict"
        existing._disabled = False
        if local is not None:
            existing._local = local
        try:
            yield
        finally:
            existing._strict = old_strict
            existing._disabled = old_disabled
            existing._local = old_local
    else:
        if strict_mode is None:
            strict_mode = "strict"
        if local is None:
            local = True
        mode = _SpmdTypeMode(
            strict_mode=strict_mode,
            local=local,
        )
        with mode:
            _set_current_mode(mode)
            try:
                yield
            finally:
                _set_current_mode(None)


@contextmanager
def no_typecheck():
    """Temporarily disable type checking on this thread (like ``no_grad``)."""
    mode = _current_mode()
    if mode is not None:
        old_disabled = mode._disabled
        mode._disabled = True
        try:
            yield
        finally:
            mode._disabled = old_disabled
    else:
        yield


class _SpmdTypeBackwardCompatibleMode:
    """Backwards-compatible wrapper around ``typecheck()``.

    Exported as ``SpmdTypeMode`` for existing callers that do::

        with SpmdTypeMode():
            ...

    New code should use ``typecheck()`` instead.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __enter__(self):
        self._cm = typecheck(**self._kwargs)
        self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._cm.__exit__(exc_type, exc_val, exc_tb)
