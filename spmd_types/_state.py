# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared state for SPMD type checking.

This module holds thread-local state that needs to be shared between _scalar
and _checker without creating circular dependencies.

The state is per-thread because PyTorch's TorchFunctionMode stack is backed by
C++ thread-local storage (PythonTorchFunctionTLS), so modes are per-thread.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spmd_types._mesh_axis import MeshAxis
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh


class MeshEntry(NamedTuple):
    axes: frozenset[MeshAxis]
    names: dict[str, MeshAxis]
    all_names: dict[str, MeshAxis]
    pgs: dict[MeshAxis, ProcessGroup]


_tls = threading.local()


def is_type_checking() -> bool:
    """Return True if type checking is active and not paused on this thread."""
    mode = _current_mode()
    return mode is not None and not mode._disabled


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


def _current_mode():
    """Return the currently active SpmdTypeMode, or None."""
    return getattr(_tls, "current_mode", None)


def _set_current_mode(mode) -> None:
    """Set the currently active SpmdTypeMode (or None to clear)."""
    _tls.current_mode = mode


def is_strict() -> bool:
    """Return True if type checking is active and in strict mode."""
    mode = _current_mode()
    return mode is not None and mode._strict


def current_mesh() -> frozenset[MeshAxis] | None:
    """Return the current mesh axes, or None if no mesh is set."""
    stack = getattr(_tls, "mesh_stack", None)
    if stack:
        return stack[-1].axes
    return None


def current_mesh_names() -> dict[str, MeshAxis] | None:
    """Return the name-to-axis mapping for the current mesh, or None.

    Only includes non-singleton axes, consistent with ``current_mesh()``.
    Use ``current_mesh_all_names()`` to include singleton axes.
    """
    stack = getattr(_tls, "mesh_stack", None)
    if stack:
        return stack[-1].names
    return None


def current_mesh_all_names() -> dict[str, MeshAxis] | None:
    """Return the full name-to-axis mapping including singleton axes, or None."""
    stack = getattr(_tls, "mesh_stack", None)
    if stack:
        return stack[-1].all_names
    return None


def _axes_to_pgs(
    axes: (
        frozenset[MeshAxis] | DeviceMesh | Sequence[ProcessGroup] | dict[str, MeshAxis]
    ),
) -> dict[MeshAxis, ProcessGroup]:
    """Capture the MeshAxis-to-ProcessGroup mapping for mesh inputs that carry PGs.

    Only a ``DeviceMesh`` currently populates this map; the other input forms
    (frozenset/dict of ``MeshAxis``) carry no retrievable PGs, so an empty
    mapping is returned.
    """
    from spmd_types._mesh_axis import MeshAxis
    from torch.distributed.device_mesh import DeviceMesh

    if isinstance(axes, DeviceMesh) and axes.mesh_dim_names:
        return {
            MeshAxis.of(g): g
            for name in axes.mesh_dim_names
            for g in (axes.get_group(name),)
        }
    return {}


def _push_mesh(
    mesh: frozenset[MeshAxis]
    | DeviceMesh
    | Sequence[ProcessGroup]
    | dict[str, MeshAxis],
    names: dict[str, MeshAxis] | None = None,
    pgs: dict[MeshAxis, ProcessGroup] | None = None,
) -> None:
    if isinstance(mesh, frozenset):
        resolved, resolved_names = mesh, names if names is not None else {}
        resolved_pgs = pgs if pgs is not None else {}
    else:
        from spmd_types._mesh import _resolve_axes

        resolved, resolved_names = _resolve_axes(mesh)
        resolved_pgs = pgs if pgs is not None else _axes_to_pgs(mesh)
    assert all(ax.size() > 1 for ax in resolved), (
        f"mesh must not contain size-1 axes, got {resolved}"
    )
    all_names = resolved_names
    filtered_names = {k: v for k, v in resolved_names.items() if v in resolved}
    if not hasattr(_tls, "mesh_stack"):
        _tls.mesh_stack = []
    _tls.mesh_stack.append(MeshEntry(resolved, filtered_names, all_names, resolved_pgs))


def _pop_mesh() -> MeshEntry:
    return _tls.mesh_stack.pop()


def _find_name_in_stack(name: str) -> bool:
    """Check whether *name* exists in any entry on the mesh stack (not just the top)."""
    stack: list[MeshEntry] = getattr(_tls, "mesh_stack", [])
    return any(name in entry.all_names for entry in stack)


def _clear_mesh_stack() -> None:
    _tls.mesh_stack = []


def _is_global() -> bool:
    """Return True if global SPMD mode is active (local=False)."""
    mode = _current_mode()
    return not mode._local if mode is not None else False
