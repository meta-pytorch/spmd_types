# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context manager for setting the current mesh axes."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from spmd_types._mesh_axis import MeshAxis
from spmd_types._state import _axes_to_pgs, _current_mesh_entry, _pop_mesh, _push_mesh
from spmd_types.types import DeviceMeshAxis, format_axis, normalize_axis, normalize_mesh
from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@contextmanager
def set_current_mesh(
    axes: (
        frozenset[MeshAxis]
        | DeviceMesh
        | Sequence[ProcessGroup]
        | dict[str, MeshAxis]
        | None
    ) = None,
    *,
    local_axes: tuple[DeviceMeshAxis, ...] = (),
):
    """Context manager that pushes a mesh onto the stack.

    Args:
        axes: The mesh to set. Accepts:
            - A ``dict[str, MeshAxis]`` mapping names to axes (preferred).
            - A ``DeviceMesh`` whose named dimensions are converted to MeshAxis
              (names are taken from ``mesh_dim_names``).
            - A frozenset of orthogonal MeshAxis objects (no string lookup).
            - A sequence of ProcessGroup objects, each converted via
              ``MeshAxis.of()`` (no string lookup).
            When omitted, reuses the current mesh.
        local_axes: An outer prefix of mesh axes that retains local SPMD
            semantics during global type checking. Each coordinate on these
            axes selects an independent local tensor; the remaining inner axes
            describe the global sharding of that tensor. For example, with
            ``dp`` local and ``tp`` global, each ``dp`` coordinate has a
            separate tensor globally sharded across ``tp``.

    Singleton (size-1) axes are dropped from the active axis set via
    ``normalize_mesh`` but remain in the name lookup table
    (``current_mesh_all_names``).
    """
    if axes is None:
        entry = _current_mesh_entry()
        if entry is None:
            raise RuntimeError(
                "set_current_mesh() without axes requires a current mesh"
            )
        resolved, names, pgs = entry.axes, entry.all_names, entry.pgs
    else:
        resolved, names = _resolve_axes(axes)
        pgs = _axes_to_pgs(axes)
    resolved_local_axes = _resolve_local_axes(local_axes, resolved, names)
    _push_mesh(
        resolved,
        names,
        pgs,
        local_axes=resolved_local_axes,
    )
    try:
        yield
    finally:
        _pop_mesh()


def _resolve_local_axes(
    local_axes: tuple[DeviceMeshAxis, ...],
    mesh: frozenset[MeshAxis],
    names: dict[str, MeshAxis],
) -> frozenset[MeshAxis]:
    """Resolve ``local_axes`` to members of ``mesh``.

    Rejects unknown or duplicate axes and requires the local axes to be the
    outermost axes and the global axes to be the innermost axes. For a
    ``(dp, tp)`` mesh, ``("dp",)`` is valid but ``("tp",)`` is not.

    Returns the resolved local axes as a ``frozenset[MeshAxis]``.
    """
    unknown = tuple(
        axis for axis in local_axes if isinstance(axis, str) and axis not in names
    )
    if unknown:
        raise ValueError(
            f"local_axes contains names not present in this mesh: {unknown!r}; "
            f"available names: {tuple(sorted(names))!r}"
        )
    resolved_list = tuple(
        names[axis] if isinstance(axis, str) else normalize_axis(axis)
        for axis in local_axes
    )
    if len(set(resolved_list)) != len(resolved_list):
        duplicates = {axis for axis in resolved_list if resolved_list.count(axis) > 1}
        duplicate_names = ", ".join(format_axis(axis) for axis in duplicates)
        raise ValueError(
            f"local_axes must not contain duplicate axes; got {local_axes!r}; "
            f"duplicate axes: ({duplicate_names})"
        )
    resolved = frozenset(resolved_list)
    if not resolved <= mesh:
        outside = resolved - mesh
        outside_names = ", ".join(format_axis(axis) for axis in outside)
        mesh_names = ", ".join(format_axis(axis) for axis in sorted(mesh, key=repr))
        raise ValueError(
            f"local_axes contains axes outside this mesh: ({outside_names}); "
            f"current mesh axes: ({mesh_names})"
        )

    ordered = sorted(
        mesh,
        key=lambda axis: max(stride for _, stride in axis.layout.sizes_and_strides),
        reverse=True,
    )
    prefix = set(ordered[: len(resolved)])
    if set(resolved) != prefix:
        local_names = ", ".join(format_axis(axis) for axis in resolved)
        mesh_names = ", ".join(format_axis(axis) for axis in ordered)
        raise ValueError(
            "local_axes must come before all global axes in outer-to-inner mesh "
            f"order; got local_axes=({local_names}) for mesh=({mesh_names})"
        )
    return resolved


def _pg_for_axis(axis: DeviceMeshAxis) -> ProcessGroup:
    """Resolve a mesh axis to its ``ProcessGroup`` using the current mesh.

    Accepts a ``ProcessGroup`` (returned unchanged), an axis name, or a
    ``MeshAxis``. Raises ``RuntimeError`` if no current mesh carries process
    groups, or if the axis is not part of it.
    """
    from torch.distributed import ProcessGroup

    if isinstance(axis, ProcessGroup):
        return axis

    entry = _current_mesh_entry()
    pgs = entry.pgs if entry is not None else None
    if not pgs:
        raise RuntimeError(
            "Passing an axis name to a redistribution or collective "
            "API requires an ambient DeviceMesh. Wrap the call in "
            "set_current_mesh(device_mesh), or pass a ProcessGroup directly."
        )

    if isinstance(axis, str):
        resolved = entry.all_names.get(axis) if entry is not None else None
    else:
        resolved = axis
    if resolved is None or resolved not in pgs:
        raise RuntimeError(f"Axis {axis!r} is not in the current mesh.")
    return pgs[resolved]


def _resolve_axes(
    axes: (
        frozenset[MeshAxis] | DeviceMesh | Sequence[ProcessGroup] | dict[str, MeshAxis]
    ),
) -> tuple[frozenset[MeshAxis], dict[str, MeshAxis]]:
    """Normalize the various input forms to a frozenset of MeshAxis and a name mapping.

    Returns:
        A tuple of (frozenset of normalized axes, name-to-axis dict).
        The name dict is empty for frozenset and Sequence[ProcessGroup] inputs.
    """
    if isinstance(axes, dict):
        raw = frozenset(axes.values())
        names = axes
    elif isinstance(axes, frozenset):
        raw = axes
        names = {}
    elif isinstance(axes, DeviceMesh):
        raw, names = _device_mesh_to_axes(axes)
    else:
        raw = frozenset(MeshAxis.of(pg) for pg in axes)
        names = {}
    normalized = normalize_mesh(raw)
    return normalized, names


def _device_mesh_to_axes(
    mesh: DeviceMesh,
) -> tuple[frozenset[MeshAxis], dict[str, MeshAxis]]:
    """Convert a DeviceMesh to a frozenset of MeshAxis objects and a name mapping.

    Each named dimension of the mesh becomes a MeshAxis via
    ``MeshAxis.of(mesh.get_group(name))``.  Singleton filtering is handled
    by the caller (``_resolve_axes`` -> ``normalize_mesh``).
    """
    if not mesh.mesh_dim_names:
        raise ValueError(
            "DeviceMesh must have mesh_dim_names set to be used with "
            "set_current_mesh. Use init_device_mesh(..., mesh_dim_names=...) "
            "or pass a frozenset of MeshAxis objects instead."
        )
    names = {name: MeshAxis.of(mesh.get_group(name)) for name in mesh.mesh_dim_names}
    return frozenset(names.values()), names
