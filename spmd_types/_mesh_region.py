# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for explicit cross-mesh compatibility on local SPMD types.

This module supports the two mesh-compatibility cases used by explicit mesh
reinterpretation:

* a flattened axis directly matching a tuple of orthogonal axes
* two tuples of orthogonal axes that flatten to the same larger region

Compatibility is checked by sorting axes by stride descending (outermost
first) on each side and walking both lists with two pointers.  Group
boundaries appear when the cumulative sizes align; at each boundary the
flattened regions and uniform local types must match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spmd_types._mesh_axis import flatten_axes, MeshAxis
from spmd_types.types import (
    format_axis,
    LocalSpmdType,
    PartitionSpec,
    PartitionSpecEntry,
    PerMeshAxisLocalSpmdType,
    SpmdTypeError,
)


def _max_stride(axis: MeshAxis) -> int:
    """Return the maximum stride across all atoms of an axis."""
    return max(d for _, d in axis.layout.sizes_and_strides)


def check_reinterpret_mesh_compatible(
    src: LocalSpmdType,
    dst: LocalSpmdType | frozenset[MeshAxis],
    src_partition_spec: PartitionSpec | None = None,
) -> tuple[LocalSpmdType, PartitionSpec | None]:
    """Check whether ``src`` can be reinterpreted onto ``dst``.

    ``dst`` may be a full ``LocalSpmdType`` (types are validated against src) or
    a ``frozenset[MeshAxis]`` (types are derived from src, used for
    auto-reinterpret).

    Returns ``(LocalSpmdType, PartitionSpec | None)`` on success.  The second
    element is the remapped PartitionSpec when ``src_partition_spec`` is provided,
    or ``None`` otherwise.

    Raises ``SpmdTypeError`` on incompatibility.

    Exactly matching axes must carry the same local type.  Remaining axes are
    sorted by stride descending and walked with two pointers; group boundaries
    appear when cumulative sizes align.  At each boundary the flattened regions
    and uniform local types must match.
    """
    axes_only = isinstance(dst, frozenset)
    src_axes = frozenset(src.keys())
    dst_axes: frozenset[MeshAxis] = dst if axes_only else frozenset(dst.keys())

    shared_axes = src_axes & dst_axes
    result: LocalSpmdType = {}
    for axis in shared_axes:
        if not axes_only:
            assert not isinstance(dst, frozenset)  # help the type checker
            if src[axis] is not dst[axis]:
                raise SpmdTypeError(
                    f"Shared mesh axis {axis} changes local type from {src[axis]} to "
                    f"{dst[axis]} under reinterpret_mesh."
                )
        result[axis] = src[axis]

    src_remaining = {axis: typ for axis, typ in src.items() if axis not in shared_axes}
    dst_remaining_axes = dst_axes - shared_axes
    # No foreign axes -- src is already on (a subset of) the target mesh.
    # Pass type and spec through unchanged.
    if not src_remaining:
        return result, src_partition_spec

    if not dst_remaining_axes:
        if axes_only:
            raise SpmdTypeError(
                f"Operand has axes {sorted(src_remaining.keys(), key=repr)} not in "
                f"the current mesh {sorted(dst_axes, key=repr)} and has no destination "
                f"to fold into."
            )
        raise SpmdTypeError(
            f"Cross-mesh reinterpretation is incomplete: {src_remaining or src} vs "
            f"{ {ax: dst[ax] for ax in dst_remaining_axes} or dst }."
        )

    # Sort axes by stride descending (outermost first).
    src_sorted = sorted(
        src_remaining.items(), key=lambda item: _max_stride(item[0]), reverse=True
    )
    if axes_only:
        dst_sorted: list[tuple[MeshAxis, PerMeshAxisLocalSpmdType | None]] = sorted(
            [(ax, None) for ax in dst_remaining_axes],
            key=lambda item: _max_stride(item[0]),
            reverse=True,
        )
    else:
        assert not isinstance(dst, frozenset)  # help the type checker
        dst_sorted = sorted(
            [(ax, dst[ax]) for ax in dst_remaining_axes],
            key=lambda item: _max_stride(item[0]),
            reverse=True,
        )

    walked = _walk_groups(src_sorted, dst_sorted, src, None if axes_only else dst)
    result.update(walked.types)

    if src_partition_spec is not None:
        all_group_pairs = [([ax], [ax]) for ax in shared_axes] + walked.group_pairs
        return result, _remap_partition_spec(src_partition_spec, all_group_pairs)
    return result, None


def _format_group(axes: list[MeshAxis]) -> str:
    """Format a group of axes as '{a, b}' (braces, comma-separated)."""
    return "{" + ", ".join(repr(a) for a in axes) + "}"


@dataclass
class _WalkResult:
    """Result of _walk_groups: type mapping and group pairs."""

    types: LocalSpmdType = field(default_factory=dict)
    group_pairs: list[tuple[list[MeshAxis], list[MeshAxis]]] = field(
        default_factory=list
    )


def _walk_groups(  # noqa: C901
    src_sorted: list[tuple[MeshAxis, PerMeshAxisLocalSpmdType]],
    dst_sorted: list[tuple[MeshAxis, PerMeshAxisLocalSpmdType | None]],
    src: LocalSpmdType,
    dst: LocalSpmdType | None,
) -> _WalkResult:
    """Two-pointer walk over stride-sorted axes, building a type mapping.

    When a dst axis has a type (not None), it is validated against the src
    group type.  When a dst axis has no type (None), the src group type is
    assigned.  Returns a ``_WalkResult`` (with the type mapping and the list
    of matched ``(src_group, dst_group)`` pairs); raises ``SpmdTypeError`` on
    incompatibility.
    """
    axes_only = dst is None
    i = j = 0
    src_cum = dst_cum = 1
    src_group: list[MeshAxis] = []
    dst_group: list[MeshAxis] = []
    src_group_type: PerMeshAxisLocalSpmdType | None = None
    dst_group_type: PerMeshAxisLocalSpmdType | None = None
    result = _WalkResult()

    while i < len(src_sorted) or j < len(dst_sorted):
        # Advance whichever side has the smaller cumulative size.
        if src_cum <= dst_cum and i < len(src_sorted):
            axis, typ = src_sorted[i]
            src_cum *= axis.size()
            src_group.append(axis)
            if src_group_type is None:
                src_group_type = typ
            elif src_group_type is not typ:
                prior = ", ".join(repr(a) for a in src_group[:-1])
                msg = (
                    f"Axes {_format_group(src_group)} would need to be "
                    f"flattened, but {prior} "
                    f"{'has' if len(src_group) <= 2 else 'have'} type "
                    f"{src_group_type!r} while {axis!r} has type {typ!r}."
                )
                if axes_only:
                    msg += (
                        " Check that the operand has the local SPMD types "
                        "you expect and that the current mesh is correct "
                        "for this region of code."
                    )
                raise SpmdTypeError(msg)
            i += 1
        elif j < len(dst_sorted):
            axis, typ = dst_sorted[j]
            dst_cum *= axis.size()
            dst_group.append(axis)
            if typ is not None:
                if dst_group_type is None:
                    dst_group_type = typ
                elif dst_group_type is not typ:
                    raise SpmdTypeError(
                        f"Axes {_format_group(dst_group)} would need to be "
                        f"flattened, but they have mixed local types in the "
                        f"destination."
                    )
            j += 1
        else:
            # src side exhausted but dst_cum < src_cum -- sizes don't align.
            break

        # Check for group boundary.
        if src_cum == dst_cum:
            assert src_group_type is not None
            # Validate dst group type against src group type if provided.
            if dst_group_type is not None and src_group_type is not dst_group_type:
                raise SpmdTypeError(
                    f"Source group {_format_group(src_group)} has type "
                    f"{src_group_type!r} but destination group "
                    f"{_format_group(dst_group)} has type {dst_group_type!r}."
                )
            if flatten_axes(tuple(src_group)) != flatten_axes(tuple(dst_group)):
                raise SpmdTypeError(
                    f"Source group {_format_group(src_group)} and destination "
                    f"group {_format_group(dst_group)} flatten to different "
                    f"rank regions. Set MESH_AXIS_DEBUG=1 to see underlying "
                    f"size/stride layouts."
                )
            # Assign types to dst axes.
            for ax in dst_group:
                result.types[ax] = src_group_type
            result.group_pairs.append((list(src_group), list(dst_group)))
            # Reset for next group.
            src_cum = dst_cum = 1
            src_group = []
            dst_group = []
            src_group_type = dst_group_type = None

    if src_cum != 1 or dst_cum != 1:
        raise SpmdTypeError(
            "The source and destination meshes do not appear "
            "to be related. Set MESH_AXIS_DEBUG=1 to see underlying "
            "size/stride layouts."
        )

    return result


def _remap_partition_spec(
    spec: PartitionSpec,
    group_pairs: list[tuple[list[MeshAxis], list[MeshAxis]]],
) -> PartitionSpec:
    """Remap a PartitionSpec from src mesh axes to dst mesh axes.

    Each spec entry is decomposed into contiguous segments that each match a
    ``src_group`` key in ``group_pairs`` (greedy).  Matched segments are
    replaced with the corresponding ``dst_group`` axes.  Axes shared between
    both meshes appear as identity pairs ``([ax], [ax])`` in ``group_pairs`` and
    pass through unchanged.

    Args:
        spec: The source PartitionSpec to remap.
        group_pairs: Matched ``(src_group, dst_group)`` pairs from
            ``_walk_groups``, plus identity pairs for shared axes.

    Returns:
        The remapped PartitionSpec (preserving rank).  Raises
        ``SpmdTypeError`` if a spec entry cannot be fully segmented into
        group pairs.

    Examples::

        # group_pairs = [([K], [K]), ([A, B], [C])]

        # Multi-axis entry segments into ([A, B],) -> replaced by C:
        _remap_partition_spec(PartitionSpec((A, B)), group_pairs)      # -> PartitionSpec(C)

        # Multi-axis entry segments into ([K],) + ([A, B],) -> K then C:
        _remap_partition_spec(PartitionSpec((K, A, B)), group_pairs)   # -> PartitionSpec((K, C))

        # Reordered match is rejected (no segment matches (B, A)):
        _remap_partition_spec(PartitionSpec((B, A)), group_pairs)      # -> raises SpmdTypeError

        # Partial match rejected (A alone is not a complete segment):
        _remap_partition_spec(PartitionSpec(A), group_pairs)           # -> raises SpmdTypeError

        # Shared axis passes through:
        _remap_partition_spec(PartitionSpec(K, None), group_pairs)     # -> PartitionSpec(K, None)
    """
    # Build a map from tuple(src_group) -> dst_group for segment matching.
    src_to_dst: dict[tuple[MeshAxis, ...], list[MeshAxis]] = {}
    for src_group, dest_group in group_pairs:
        src_to_dst[tuple(src_group)] = dest_group

    entries: list[PartitionSpecEntry] = []
    for dim, entry in enumerate(spec):
        if entry is None:
            entries.append(None)
            continue
        src_axes = entry if isinstance(entry, tuple) else (entry,)
        sub_entries: list[MeshAxis] = []
        i = 0
        while i < len(src_axes):
            for end in range(i + 1, len(src_axes) + 1):
                segment = tuple(src_axes[i:end])
                if segment in src_to_dst:
                    sub_entries.extend(src_to_dst[segment])
                    i = end
                    break
            else:
                raise SpmdTypeError(
                    f"Cannot remap PartitionSpec: dim {dim} is sharded on "
                    f"{format_axis(src_axes[0]) if len(src_axes) == 1 else src_axes} "
                    f"which does not exactly match any remapped group."
                )
        if len(sub_entries) == 1:
            entries.append(sub_entries[0])
        else:
            entries.append(tuple(sub_entries))
    return PartitionSpec(*entries)
