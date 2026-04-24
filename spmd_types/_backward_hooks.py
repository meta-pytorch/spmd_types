# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-hook SPMD type-propagation registry for nn.Module backward hooks.

BackwardHookFunction (nn.Module.register_full_backward_hook) has a pure
pass-through forward. Each user hook callable must be explicitly registered
via ``register_local_backward_hook`` to declare that it does not alter SPMD
types; unregistered hooks raise SpmdTypeError when type checking is active.

TODO: a per-hook rule API (for hooks that do collectives on grads, e.g.
all-reduce flipping I -> R) can be added later when a real use case
motivates the exact shape.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.utils.hooks as _torch_hooks
from spmd_types._state import is_type_checking
from spmd_types.runtime import assert_type_like
from spmd_types.types import SpmdTypeError
from torch.nn.modules._functions import BackwardHookFunction

_LOCAL_BACKWARD_HOOKS: set[Callable] = set()


def register_local_backward_hook(fn: Callable) -> Callable:
    """Declare that ``fn`` does not alter SPMD types when it runs in backward.

    Analogous to :func:`register_local_autograd_function`: the hook's backward
    is treated as local (no collectives, no type-changing effects on grads).
    Side-agnostic: covers both ``register_full_backward_hook`` (post) and
    ``register_full_backward_pre_hook`` (pre) use.  Usable as a decorator.
    """
    _LOCAL_BACKWARD_HOOKS.add(fn)
    return fn


def _validate(hooks):
    if not is_type_checking():
        return
    for fn in hooks:
        if fn in _LOCAL_BACKWARD_HOOKS:
            continue
        raise SpmdTypeError(
            f"Backward hook {getattr(fn, '__qualname__', fn)!r} attached to "
            f"nn.Module is not registered for SPMD type propagation.  Call "
            f"register_local_backward_hook(fn) if the hook does not alter "
            f"gradient SPMD types."
        )


def _apply_types(hooks, inputs, outputs):
    if not is_type_checking():
        return
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    for fn in hooks:
        if fn in _LOCAL_BACKWARD_HOOKS:
            for inp, out in zip(inputs, outputs):
                if isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor):
                    assert_type_like(out, inp)


_orig_setup_input_hook = None
_orig_setup_output_hook = None


def _patched_setup_input_hook(self, input):
    _validate(self.user_hooks)
    result = _orig_setup_input_hook(self, input)
    _apply_types(self.user_hooks, input, result)
    return result


def _patched_setup_output_hook(self, output):
    _validate(self.user_pre_hooks)
    result = _orig_setup_output_hook(self, output)
    _apply_types(self.user_pre_hooks, output, result)
    return result


def install() -> None:
    """Install the BackwardHook monkey-patch.  Idempotent."""
    global _orig_setup_input_hook, _orig_setup_output_hook

    if _orig_setup_input_hook is not None:
        return

    from spmd_types._checker import register_autograd_function

    BackwardHookFunction.typecheck_forward = staticmethod(BackwardHookFunction.apply)
    register_autograd_function(BackwardHookFunction)

    _orig_setup_input_hook = _torch_hooks.BackwardHook.setup_input_hook
    _orig_setup_output_hook = _torch_hooks.BackwardHook.setup_output_hook
    _torch_hooks.BackwardHook.setup_input_hook = _patched_setup_input_hook
    _torch_hooks.BackwardHook.setup_output_hook = _patched_setup_output_hook


def uninstall() -> None:
    """Remove the BackwardHook monkey-patch.  Idempotent."""
    global _orig_setup_input_hook, _orig_setup_output_hook

    if _orig_setup_input_hook is None:
        return

    _torch_hooks.BackwardHook.setup_input_hook = _orig_setup_input_hook
    _torch_hooks.BackwardHook.setup_output_hook = _orig_setup_output_hook
    _orig_setup_input_hook = None
    _orig_setup_output_hook = None
