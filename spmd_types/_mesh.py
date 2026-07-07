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
from spmd_types._state import _axes_to_pgs, _pop_mesh, _push_mesh
from spmd_types.types import DeviceMeshAxis, normalize_mesh
from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@contextmanager
def set_current_mesh(
    axes: (
        frozenset[MeshAxis] | DeviceMesh | Sequence[ProcessGroup] | dict[str, MeshAxis]
    ),
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

    Singleton (size-1) axes are dropped from the active axis set via
    ``normalize_mesh`` but remain in the name lookup table
    (``current_mesh_all_names``).
    """
    resolved, names = _resolve_axes(axes)
    _push_mesh(resolved, names, _axes_to_pgs(axes))
    try:
        yield
    finally:
        _pop_mesh()


def _pg_for_axis(axis: DeviceMeshAxis) -> ProcessGroup:
    """Resolve a mesh axis to its ``ProcessGroup`` using the current mesh.

    Accepts a ``ProcessGroup`` (returned unchanged), an axis name, or a
    ``MeshAxis``. Raises ``RuntimeError`` if no current mesh carries process
    groups, or if the axis is not part of it.
    """
    from spmd_types._state import _tls, current_mesh_all_names
    from torch.distributed import ProcessGroup

    if isinstance(axis, ProcessGroup):
        return axis

    stack = getattr(_tls, "mesh_stack", None)
    pgs = stack[-1].pgs if stack else None
    if not pgs:
        raise RuntimeError(
            "Passing an axis name to a redistribution or collective "
            "API requires an ambient DeviceMesh. Wrap the call in "
            "set_current_mesh(device_mesh), or pass a ProcessGroup directly."
        )

    if isinstance(axis, str):
        resolved = (current_mesh_all_names() or {}).get(axis)
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
