# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SPMD type definitions for distributed tensor expressions.

This module provides:
- Per-mesh-axis local SPMD types (R, I, V, P, S)
- PartitionSpec for global SPMD
- TensorSharding for module boundary contracts
- Type aliases
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence, TYPE_CHECKING, TypeAlias

from spmd_types._mesh_axis import MeshAxis

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

# =============================================================================
# Per-Mesh-Axis Local SPMD Type Enum
# =============================================================================


class PerMeshAxisLocalSpmdType(Enum):
    """
    Per-mesh-axis local SPMD type as an enum.

    Describes how a tensor is distributed across ranks on one axis of the
    device mesh, as well as how the gradients are distributed. The four
    values are: R (Replicate), I (Invariant), V (Varying), P (Partial).
    """

    R = "R"
    I = "I"  # noqa: E741
    V = "V"
    P = "P"

    def __repr__(self):
        return self.value

    def backward_type(self) -> PerMeshAxisLocalSpmdType:
        """Return the type that gradients have in the backward pass."""
        return _BACKWARD_TYPE[self]


_BACKWARD_TYPE = {
    PerMeshAxisLocalSpmdType.R: PerMeshAxisLocalSpmdType.P,
    PerMeshAxisLocalSpmdType.I: PerMeshAxisLocalSpmdType.I,
    PerMeshAxisLocalSpmdType.V: PerMeshAxisLocalSpmdType.V,
    PerMeshAxisLocalSpmdType.P: PerMeshAxisLocalSpmdType.R,
}


@dataclass(frozen=True)
class Shard:
    """
    A refinement of Varying that specifies the tensor is sharded on a particular
    dimension.

    While Varying (V) only says "each rank has different data" without
    specifying how ranks relate to a global tensor, Shard(dim) additionally
    says "the global tensor can be reconstructed by concatenating the local
    tensors along dimension ``dim``."  This is analogous to DTensor's
    ``Shard(dim)`` placement.

    This is not a true type (notice it doesn't inherit from PerMeshAxisLocalSpmdType)
    but it is accepted at any src/dst argument and, from a typing perspective,
    is equivalent to Varying.  However, it does change the semantics of collectives
    by switching from stack/unbind semantics (V) to concat/split semantics (S):

    - ``all_gather(src=V)``: stacks per-rank tensors on a new dim 0
      (output gains a dimension).
    - ``all_gather(src=S(i))``: concatenates per-rank tensors along dim ``i``
      (output has the same number of dimensions as each input).
    - ``reduce_scatter(dst=V)``: unbinds along dim 0 after reducing
      (output loses a dimension).
    - ``reduce_scatter(dst=S(i))``: splits along dim ``i`` after reducing
      (output has the same number of dimensions, but dim ``i`` shrinks).

    In practice, most users want concat/split (S) rather than stack/unbind (V),
    because tensors are typically sharded along an existing dimension.

    In global SPMD types, a per-mesh-axis Shard can also be used to manipulate
    the PartitionSpec in a mesh-oriented way, although the PartitionSpec is still
    the canonical way of representing this typing information.
    """

    dim: int

    def __repr__(self):
        return f"S({self.dim})"

    def backward_type(self) -> Shard:
        return self


# Single character aliases for ease of pattern matching
R = PerMeshAxisLocalSpmdType.R
I = PerMeshAxisLocalSpmdType.I  # noqa: E741
V = PerMeshAxisLocalSpmdType.V
P = PerMeshAxisLocalSpmdType.P
S = Shard  # S(i) creates a Shard with dim=i

# Backward compatibility aliases
Replicate = R
Invariant = I
Varying = V
Partial = P

# Type aliases
PerMeshAxisSpmdType = PerMeshAxisLocalSpmdType | Shard

# Axis identifier: a MeshAxis or a ProcessGroup.
# ProcessGroup is accepted for convenience but is normalized to MeshAxis
# internally via normalize_axis().
DeviceMeshAxis: TypeAlias = "MeshAxis | ProcessGroup"

# PerMeshAxisSpmdTypes is the permissive input variant that also accepts S(i).
# Used in user-facing APIs like assert_type, where S(i) is syntax sugar
# for setting the partition_spec.
PerMeshAxisSpmdTypes: TypeAlias = "dict[DeviceMeshAxis, PerMeshAxisSpmdType]"

# LocalSpmdType maps axis identifiers to per-axis SPMD types (R, I, V, P only).
# This is the type stored on tensors; Shard is never stored.
LocalSpmdType: TypeAlias = "dict[DeviceMeshAxis, PerMeshAxisLocalSpmdType]"

# Element type for PartitionSpec entries after normalization.
# Each entry is None (replicated), a single MeshAxis, or a tuple of MeshAxis
# (multi-axis sharding on one dim).
PartitionSpecEntry: TypeAlias = "MeshAxis | tuple[MeshAxis, ...] | None"

# =============================================================================
# PartitionSpec for Global SPMD
# =============================================================================


class PartitionSpec(tuple[PartitionSpecEntry, ...]):
    """
    A partition spec describes how tensor dimensions map to mesh axes.

    Each element corresponds to a tensor dimension and specifies zero, one, or
    multiple mesh axes that shard that dimension. For example:
        - PartitionSpec(tp, None) means dim 0 is sharded on tp, dim 1 is replicated
        - PartitionSpec((dp, tp), None) means dim 0 is sharded on both dp and tp
        - PartitionSpec() means fully replicated
    """

    def __new__(
        cls,
        *args: DeviceMeshAxis | tuple[DeviceMeshAxis, ...] | None,
    ):
        normalized: list[MeshAxis | tuple[MeshAxis, ...] | None] = []
        for i, entry in enumerate(args):
            if entry is None:
                normalized.append(None)
            elif isinstance(entry, tuple):
                if len(entry) == 0:
                    raise ValueError(
                        f"PartitionSpec entry at dim {i} is an empty tuple. "
                        f"Use None for replicated dimensions."
                    )
                normalized.append(tuple(normalize_axis(a) for a in entry))
            else:
                normalized.append(normalize_axis(entry))
        return super().__new__(cls, normalized)

    def __repr__(self):
        if not self:
            return "PartitionSpec()"
        parts = []
        for entry in self:
            if entry is None:
                parts.append("None")
            elif isinstance(entry, tuple):
                parts.append("(" + ", ".join(repr(a) for a in entry) + ")")
            else:
                parts.append(repr(entry))
        return f"PartitionSpec({', '.join(parts)})"

    def axes_with_partition_spec(self) -> set["MeshAxis"]:
        """Return the set of all mesh axes referenced by this PartitionSpec."""
        result: set[MeshAxis] = set()
        for entry in self:
            if entry is None:
                continue
            for a in entry if isinstance(entry, tuple) else (entry,):
                result.add(a)
        return result


# =============================================================================
# TensorSharding for Module Boundary Contracts
# =============================================================================

# Sharding for one tensor axis.
# * None: replicated;
# * str: the device mesh axis name, e.g., "fsdp" or "tp";
# * Sequence[str]: multiple device mesh axis names, e.g., ("fsdp", "cp").
#
# Note: DimSharding uses plain strings (mesh dim names) rather than MeshAxis
# because it describes DeviceMesh dimensions by name, not axis identifiers.
DimSharding = None | str | Sequence[str]


def normalize_dim_sharding(dim_sharding: DimSharding) -> Sequence[str]:
    """Normalizes a sharding to a sequence of dim names."""
    if dim_sharding is None:
        return ()
    if isinstance(dim_sharding, str):
        return (dim_sharding,)
    return dim_sharding


class _PytreeTuple:
    """Tuple-like values that are treated as leaves of a PyTree.

    The purpose of this class is to allow ``TensorSharding``s to be treated as
    tree leaves (e.g., for compat with ``tree.map``).
    """

    def __init__(self, *values) -> None:
        self._values = tuple(values)

    def __repr__(self) -> str:
        pr = repr(self._values)[1:-1]
        return f"{type(self).__name__}({pr})"

    def __getitem__(self, i):
        return self._values[i]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self._values == other._values
        elif isinstance(other, tuple):
            return self._values == other
        return False

    def __hash__(self) -> int:
        return hash(self._values)

    def __add__(self, other):
        if isinstance(other, (self.__class__, tuple)):
            return self.__class__(*self, *other)
        raise NotImplementedError(type(other))

    def __radd__(self, other):
        if isinstance(other, (self.__class__, tuple)):
            return self.__class__(*other, *self)
        raise NotImplementedError(type(other))

    def index(self, value):
        return self._values.index(value)

    def count(self, value):
        return self._values.count(value)


class TensorSharding(_PytreeTuple):
    """A tuple of ``DimSharding``s describing the sharding for a Tensor.

    Each element maps one tensor dimension to zero, one, or multiple mesh axis
    names.  For example::

        TensorSharding("fsdp", "tp")   # dim 0 on fsdp, dim 1 on tp
        TensorSharding(None, "tp")     # dim 0 replicated, dim 1 on tp
        TensorSharding(("fsdp", "cp")) # dim 0 on both fsdp and cp

    Its length must be less than or equal to the number of Tensor axes.
    """

    def __init__(self, *dim_shardings: DimSharding) -> None:
        super().__init__(*dim_shardings)


class SpmdTypeError(RuntimeError):
    """Error raised for SPMD type mismatches.

    Inherits from RuntimeError (not TypeError) so that it is not swallowed by
    Python's binary-operator dispatch machinery when raised inside
    ``__torch_function__``.  Python interprets a TypeError from an operator
    dunder as "this type doesn't support the operation" and silently falls
    through to reflected operations, masking the real error message.

    The optional ``context`` field carries a pre-formatted string describing
    the operator and its typed operands (shapes, dtypes, SPMD types).  When
    present, ``__str__`` appends it after the base message so that error
    output looks like::

        SpmdTypeError: Partial type on axis DP in a linear op ...
          In add(args[0]: f32[8, 16] {DP: P, TP: R}, args[1]: f32[16] {DP: R, TP: R})
    """

    context: str | None

    def __init__(self, msg: str, *, context: str | None = None):
        super().__init__(msg)
        self.context = context

    def __str__(self):
        base = super().__str__()
        if self.context:
            return base + "\n\n" + self.context
        return base


class RedistributeError(SpmdTypeError):
    """Error raised when a redistribute is not allowed.

    For example, redistributing a non-innermost axis in a multi-axis
    PartitionSpec group.
    """

    pass


def _canonicalize_shard(typ: PerMeshAxisSpmdType, ndim: int) -> PerMeshAxisSpmdType:
    """Resolve negative dims in Shard types. Returns typ unchanged if not Shard.

    Args:
        typ: The per-mesh-axis SPMD type, possibly a Shard with a negative dim.
        ndim: The number of dimensions of the tensor, used to resolve negative dims.

    Raises:
        SpmdTypeError: If the resolved dim is out of bounds.
    """
    if isinstance(typ, Shard):
        if typ.dim < 0:
            if typ.dim < -ndim:
                raise SpmdTypeError(
                    f"S({typ.dim}) is out of bounds for tensor with ndim={ndim}"
                )
            return Shard(typ.dim + ndim)
        if typ.dim >= ndim:
            raise SpmdTypeError(
                f"S({typ.dim}) is out of bounds for tensor with ndim={ndim}"
            )
    return typ


def normalize_axis(axis: DeviceMeshAxis) -> MeshAxis:
    """Normalize a DeviceMeshAxis to its canonical form (MeshAxis).

    ProcessGroup is converted to MeshAxis via MeshAxis.of().
    MeshAxis passes through unchanged.

    Args:
        axis: The mesh axis identifier (MeshAxis or ProcessGroup).
    """
    if isinstance(axis, MeshAxis):
        return axis
    # ProcessGroup -> MeshAxis
    return MeshAxis.of(axis)


def _check_orthogonality(axes: list[MeshAxis]) -> None:
    """Check that all mesh axes are mutually orthogonal.

    Axes must tile different dimensions of the device mesh without rank
    collisions. Overlapping axes (e.g., a flattened ``dp_cp`` axis alongside
    an individual ``dp`` axis) produce incorrect type inference results because
    type inference reasons about each axis independently.

    Args:
        axes: The mesh axes to check.

    Raises:
        SpmdTypeError: If any two axes overlap (share ranks).
    """
    if len(axes) < 2:
        return  # 0 or 1 axes are trivially orthogonal.
    # Fast path: check all axes at once.
    if axes[0].isorthogonal(*axes[1:]):
        return
    # Slow path: find the specific overlapping pair for a clear error message.
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            if not axes[i].isorthogonal(axes[j]):
                raise SpmdTypeError(
                    f"Mesh axes {format_axis(axes[i])} and "
                    f"{format_axis(axes[j])} are not orthogonal (they share "
                    f"ranks). All axes in a LocalSpmdType must be mutually "
                    f"orthogonal."
                )
    # If no pair is non-orthogonal but the group fails, it means a three-way
    # collision. This shouldn't happen with standard mesh layouts but handle
    # it for completeness.
    axis_names = ", ".join(format_axis(a) for a in axes)
    raise SpmdTypeError(f"Mesh axes [{axis_names}] are not mutually orthogonal.")


def normalize_mesh(axes: frozenset[MeshAxis]) -> frozenset[MeshAxis]:
    """Normalize a mesh: drop singleton axes and check orthogonality.

    This is the mesh-key analogue of ``normalize_local_type``.  It ensures:

    1. Size-1 (singleton) axes are dropped -- they carry no sharding
       information and keeping them causes spurious strict-mode errors.
    2. Remaining axes are mutually orthogonal (no shared ranks).

    Args:
        axes: The mesh axes to normalize.

    Raises:
        SpmdTypeError: If any two (non-singleton) axes overlap.
    """
    filtered = frozenset(a for a in axes if a.size() > 1)
    _check_orthogonality(list(filtered))
    return filtered


def normalize_local_type(spmd_type: LocalSpmdType) -> LocalSpmdType:
    """Normalize a LocalSpmdType dict: canonicalize axis keys and drop singletons.

    This is the canonical way to build a ``LocalSpmdType`` dict from
    user-supplied axis/type pairs.  It ensures that:

    1. All axis keys are normalized to ``MeshAxis`` (via ``normalize_axis``).
    2. All values are ``PerMeshAxisLocalSpmdType`` (R, I, V, or P).
    3. Size-1 (singleton) axes are dropped -- they carry no information since
       there is only one rank, and dropping them keeps the key set consistent
       across tensors and ``Scalar`` objects.
    4. Remaining axes are mutually orthogonal (no shared ranks).

    Higher-level validators (e.g., ``_validate`` in ``_checker.py``) may add
    further checks (rejection of internal sentinel types) on top of this
    function.

    Args:
        spmd_type: A mapping from mesh axes to per-axis SPMD types.

    Raises:
        TypeError: If any value is not a PerMeshAxisLocalSpmdType.
        SpmdTypeError: If any two axes overlap (share ranks).
    """
    result: LocalSpmdType = {}
    for axis, typ in spmd_type.items():
        if not isinstance(typ, PerMeshAxisLocalSpmdType):
            raise TypeError(
                f"Expected PerMeshAxisLocalSpmdType (R, I, V, or P) on axis "
                f"{format_axis(axis)}, got {typ!r}"
            )
        norm = normalize_axis(axis)
        if norm.size() == 1:
            continue  # singleton axes are not tracked
        result[norm] = typ
    _check_orthogonality(list(result.keys()))
    return result


def to_local_type(typ: PerMeshAxisSpmdType) -> PerMeshAxisLocalSpmdType:
    """Map a global per-axis type to its local counterpart.

    S(i) -> V, R -> R, I -> I, P -> P.
    """
    if isinstance(typ, Shard):
        return V
    return typ


def format_axis(axis: DeviceMeshAxis) -> str:
    """Format a mesh axis for display in error messages.

    For MeshAxis, uses the MeshAxis repr (e.g., ``tp``).
    For ProcessGroup axes, converts to MeshAxis first.

    Args:
        axis: The mesh axis identifier (MeshAxis or ProcessGroup).
    """
    if isinstance(axis, MeshAxis):
        return repr(axis)
    # ProcessGroup - convert to MeshAxis for display
    return repr(normalize_axis(axis))


def shard_types_to_partition_spec(
    types: dict[DeviceMeshAxis, PerMeshAxisSpmdType],
    ndim: int,
    axis_order: Sequence[DeviceMeshAxis] | None = None,
) -> PartitionSpec:
    """Convert Shard entries in a types dict to a PartitionSpec.

    Scans ``types`` for Shard entries, groups them by resolved dim, and
    returns a PartitionSpec of length ``ndim``.

    Axes listed in ``axis_order`` are placed first (in the given order)
    for each dim; remaining Shard axes are appended.  If a dim ends up
    with multiple axes and their relative order was not determined by
    ``axis_order``, a ``SpmdTypeError`` is raised.  When every dim has
    at most one axis, ``axis_order`` can be omitted.

    Args:
        types: Dict mapping mesh axes to per-axis SPMD types.
            Only Shard entries are considered; R/I/P entries are ignored.
        ndim: Number of tensor dimensions (length of the returned spec).
        axis_order: Optional sequence that pins the ordering of axes
            within multi-axis dims.  Need only cover axes that share a
            dim; axes on unique dims are handled automatically.

    Returns:
        A PartitionSpec of length ``ndim``.

    Raises:
        SpmdTypeError: If multiple axes shard the same dim and
            ``axis_order`` does not resolve their ordering.
    """
    dim_to_axes: dict[int, list[DeviceMeshAxis]] = {}
    mesh_axis_processed: set[DeviceMeshAxis] = set()
    if axis_order is not None:
        for axis in axis_order:
            typ = types.get(axis)
            if isinstance(typ, Shard):
                dim_to_axes.setdefault(typ.dim, []).append(axis)
                mesh_axis_processed.add(axis)
    # Scan remaining entries, but make sure no unresolved ordering.
    for axis, typ in types.items():
        if isinstance(typ, Shard) and axis not in mesh_axis_processed:
            dim_to_axes.setdefault(typ.dim, []).append(axis)
            if len(dim_to_axes[typ.dim]) > 1:
                names = ", ".join(format_axis(a) for a in dim_to_axes[typ.dim])
                raise SpmdTypeError(
                    f"Tensor dim {typ.dim} is sharded on multiple axes "
                    f"({names}) but axis_order does not resolve their "
                    f"ordering. Pass all conflicting axes in axis_order."
                )

    entries: list[DeviceMeshAxis | tuple[DeviceMeshAxis, ...] | None] = [None] * ndim
    for dim, axes in dim_to_axes.items():
        if len(axes) == 1:
            entries[dim] = axes[0]
        else:
            entries[dim] = tuple(axes)
    return PartitionSpec(*entries)


def partition_spec_to_shard_types(
    spec: PartitionSpec,
) -> dict[MeshAxis, Shard]:
    """Convert a PartitionSpec to a dict of Shard types.

    Walks PartitionSpec entries; for each axis at dim ``d``, emits
    ``axis: S(d)``.

    Args:
        spec: The PartitionSpec to convert.

    Returns:
        A dict mapping mesh axes to Shard types.

    Raises:
        SpmdTypeError: If an axis appears at two different dims.
    """
    result: dict[MeshAxis, Shard] = {}
    for dim, entry in enumerate(spec):
        if entry is None:
            continue
        axes = entry if isinstance(entry, tuple) else (entry,)
        for axis in axes:
            if axis in result:
                raise SpmdTypeError(
                    f"Axis {format_axis(axis)} appears at both dim "
                    f"{result[axis].dim} and dim {dim} in PartitionSpec"
                )
            result[axis] = Shard(dim)
    return result


def partition_spec_get_shard(
    spec: PartitionSpec | None,
    axis: MeshAxis,
) -> Shard | None:
    """Return the Shard type for ``axis`` in ``spec``, or None if not sharded.

    Args:
        spec: The PartitionSpec to query (None means no sharding).
        axis: The mesh axis to look up.

    Returns:
        ``Shard(dim)`` if ``axis`` appears at dimension ``dim`` in the spec,
        otherwise None.
    """
    if spec is None:
        return None
    for dim, entry in enumerate(spec):
        if entry is None:
            continue
        axes = entry if isinstance(entry, tuple) else (entry,)
        if axis in axes:
            return Shard(dim)
    return None
