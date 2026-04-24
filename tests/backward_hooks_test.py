# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for per-hook SPMD type-propagation registration.

Covers: _backward_hooks.py
"""

import torch
import torch.nn as nn
from spmd_types import register_local_backward_hook, SpmdTypeError
from spmd_types._test_utils import SpmdTypeCheckedTestCase


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.0


def _test_hook_unregistered(module, grad_input, grad_output):
    return None


@register_local_backward_hook
def _test_hook_local(module, grad_input, grad_output):
    return None


class TestBackwardHookRegistry(SpmdTypeCheckedTestCase):
    def test_unregistered_hook_raises(self):
        module = _Identity()
        module.register_full_backward_hook(_test_hook_unregistered)
        x = torch.randn(4, requires_grad=True)

        with self.assertRaises(SpmdTypeError) as ctx:
            module(x)
        self.assertIn("not registered", str(ctx.exception))

    def test_local_hook_preserves_shape_and_grad(self):
        module = _Identity()
        module.register_full_backward_hook(_test_hook_local)
        x = torch.randn(4, requires_grad=True)
        y = module(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(y.requires_grad)
        self.assertIsNotNone(y.grad_fn)
