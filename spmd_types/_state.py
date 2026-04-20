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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spmd_types._mesh_axis import MeshAxis

_tls = threading.local()


def is_type_checking() -> bool:
    """Return True if type checking is active and not paused on this thread."""
    mode = _current_mode()
    return mode is not None and not mode._disabled


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
        return stack[-1]
    return None


def _push_mesh(mesh: frozenset[MeshAxis]) -> None:
    assert all(ax.size() > 1 for ax in mesh), (
        f"mesh must not contain size-1 axes, got {mesh}"
    )
    if not hasattr(_tls, "mesh_stack"):
        _tls.mesh_stack = []
    _tls.mesh_stack.append(mesh)


def _pop_mesh() -> frozenset[MeshAxis]:
    return _tls.mesh_stack.pop()


def _clear_mesh_stack() -> None:
    _tls.mesh_stack = []


def _is_global() -> bool:
    """Return True if global SPMD mode is active (local=False)."""
    mode = _current_mode()
    return not mode._local if mode is not None else False
