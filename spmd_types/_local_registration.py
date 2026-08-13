# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightweight registries for local SPMD type propagation."""

from __future__ import annotations

from collections.abc import Callable

_LOCAL_AUTOGRAD_FUNCTIONS: set[type] = set()
_LOCAL_BACKWARD_HOOKS: set[Callable] = set()


def register_local_autograd_function(cls: type) -> type:
    """Register an autograd.Function subclass as local-only for SPMD type checking.

    Local-only means the function's forward operates element-wise (or more
    generally, does not rearrange data across the tensor in a way that would
    change its sharding type). It must not perform collectives or cross-device
    communication. For functions that do, use ``register_autograd_function``
    with custom typechecking instead.

    Registered functions get the standard local type propagation rule when
    type checking is active:

    - Inputs may freely mix R and V types; the output is R unless any input
      is V, in which case it is V.
    - All-I inputs produce I outputs.
    - R/V and I cannot be mixed.
    - P is forbidden.

    Unregistered autograd functions that reach the type checker leave their
    outputs untyped or raise in strict mode.
    """
    _LOCAL_AUTOGRAD_FUNCTIONS.add(cls)
    return cls


def register_local_backward_hook(fn: Callable) -> Callable:
    """Declare that ``fn`` does not alter SPMD types when it runs in backward.

    Analogous to ``register_local_autograd_function``, the hook's backward is
    treated as local with no collectives or type-changing effects on gradients.
    This covers both full backward post-hooks and pre-hooks.
    """
    _LOCAL_BACKWARD_HOOKS.add(fn)
    return fn
