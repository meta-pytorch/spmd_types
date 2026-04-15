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

from spmd_types._mesh_axis import flatten_axes, MeshAxis
from spmd_types.types import LocalSpmdType, PerMeshAxisLocalSpmdType


def _max_stride(axis: MeshAxis) -> int:
    """Return the maximum stride across all atoms of an axis."""
    return max(d for _, d in axis.layout.sizes_and_strides)


def check_reinterpret_mesh_compatible(
    src: LocalSpmdType,
    dst: LocalSpmdType | frozenset[MeshAxis],
) -> LocalSpmdType | str:
    """Check whether ``src`` can be reinterpreted onto ``dst``.

    ``dst`` may be a full ``LocalSpmdType`` (types are validated against src)
    or a ``frozenset[MeshAxis]`` (types are derived from src, used for
    auto-reinterpret).

    Returns the constructed output ``LocalSpmdType`` on success, or an error
    message string on failure.

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
                return (
                    f"Shared mesh axis {axis} changes local type from {src[axis]} to "
                    f"{dst[axis]} under reinterpret_mesh."
                )
        result[axis] = src[axis]

    src_remaining = {axis: typ for axis, typ in src.items() if axis not in shared_axes}
    dst_remaining_axes = dst_axes - shared_axes
    if not src_remaining and not dst_remaining_axes:
        return result

    if axes_only:
        # For auto-reinterpret, both foreign and missing axes must exist.
        # If src only has mesh axes but is missing some, that is a
        # missing-annotation problem, not a mesh mismatch.
        if not src_remaining:
            return "No foreign axes found; not a cross-mesh reinterpretation."
        if not dst_remaining_axes:
            return "No missing mesh axes; not a cross-mesh reinterpretation."
    elif not src_remaining or not dst_remaining_axes:
        return (
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
    if isinstance(walked, str):
        return walked
    result.update(walked)
    return result


def _format_group(axes: list[MeshAxis]) -> str:
    """Format a group of axes as '{a, b}' (braces, comma-separated)."""
    return "{" + ", ".join(repr(a) for a in axes) + "}"


def _walk_groups(  # noqa: C901
    src_sorted: list[tuple[MeshAxis, PerMeshAxisLocalSpmdType]],
    dst_sorted: list[tuple[MeshAxis, PerMeshAxisLocalSpmdType | None]],
    src: LocalSpmdType,
    dst: LocalSpmdType | None,
) -> LocalSpmdType | str:
    """Two-pointer walk over stride-sorted axes, building a type mapping.

    When a dst axis has a type (not None), it is validated against the src
    group type.  When a dst axis has no type (None), the src group type is
    assigned.  Returns the constructed mapping on success, or an error string
    on failure.
    """
    axes_only = dst is None
    i = j = 0
    src_cum = dst_cum = 1
    src_group: list[MeshAxis] = []
    dst_group: list[MeshAxis] = []
    src_group_type: PerMeshAxisLocalSpmdType | None = None
    dst_group_type: PerMeshAxisLocalSpmdType | None = None
    result: LocalSpmdType = {}

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
                return msg
            i += 1
        elif j < len(dst_sorted):
            axis, typ = dst_sorted[j]
            dst_cum *= axis.size()
            dst_group.append(axis)
            if typ is not None:
                if dst_group_type is None:
                    dst_group_type = typ
                elif dst_group_type is not typ:
                    return (
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
                return (
                    f"Source group {_format_group(src_group)} has type "
                    f"{src_group_type!r} but destination group "
                    f"{_format_group(dst_group)} has type {dst_group_type!r}."
                )
            if flatten_axes(tuple(src_group)) != flatten_axes(tuple(dst_group)):
                return (
                    f"Source group {_format_group(src_group)} and destination "
                    f"group {_format_group(dst_group)} flatten to different "
                    f"rank regions. Set MESH_AXIS_DEBUG=1 to see underlying "
                    f"size/stride layouts."
                )
            # Assign types to dst axes.
            for ax in dst_group:
                result[ax] = src_group_type
            # Reset for next group.
            src_cum = dst_cum = 1
            src_group = []
            dst_group = []
            src_group_type = dst_group_type = None

    if src_cum != 1 or dst_cum != 1:
        return (
            "The source and destination meshes do not appear "
            "to be related. Set MESH_AXIS_DEBUG=1 to see underlying "
            "size/stride layouts."
        )

    return result
