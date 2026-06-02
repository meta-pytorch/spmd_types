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
from spmd_types import R, register_local_backward_hook, S, SpmdTypeError, V
from spmd_types._checker import assert_type, get_partition_spec, typecheck
from spmd_types._test_utils import LocalTensorTestCase, SpmdTypeCheckedTestCase
from spmd_types._type_attr import get_axis_local_type
from spmd_types.types import normalize_axis, PartitionSpec


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.0


def _test_hook_unregistered(module, grad_input, grad_output):
    return None


@register_local_backward_hook
def _test_hook_local(module, grad_input, grad_output):
    return None


@register_local_backward_hook
def _test_pre_hook_local(module, grad_output):
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

    def test_unregistered_hook_does_not_propagate(self):
        module = _Identity()
        module.register_full_backward_hook(_test_hook_unregistered)
        x = self._generate_inputs((4,), self.pg, V)

        with self.assertRaises(SpmdTypeError) as ctx:
            module(x)
        self.assertIn("not registered", str(ctx.exception))

    def test_local_hook_preserves_spmd_type(self):
        module = _Identity()
        module.register_full_backward_hook(_test_hook_local)
        x = self._generate_inputs((4,), self.pg, V)
        y = module(x)

        self.assertIs(get_axis_local_type(y, self.pg), V)

    def test_local_pre_hook_preserves_spmd_type(self):
        module = _Identity()
        module.register_full_backward_pre_hook(_test_pre_hook_local)
        x = self._generate_inputs((4,), self.pg, R)
        y = module(x)

        self.assertIs(get_axis_local_type(y, self.pg), R)


class TestBackwardHookGlobalMetadata(LocalTensorTestCase):
    def test_local_hook_preserves_partition_spec(self):
        module = _Identity()
        module.register_full_backward_hook(_test_hook_local)
        x = self.rank_map(lambda r: torch.randn(4, 3))
        x.requires_grad_(True)
        assert_type(x, {self.pg: S(0)})

        with typecheck(local=False):
            y = module(x)

        self.assertIs(get_axis_local_type(y, self.pg), V)
        self.assertEqual(
            get_partition_spec(y),
            PartitionSpec(normalize_axis(self.pg), None),
        )
