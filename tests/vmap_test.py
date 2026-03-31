# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests: vmap propagates SPMD type checking."""

import unittest

import torch
import torch.distributed as dist
from spmd_types import R, V
from spmd_types._checker import assert_type, typecheck
from spmd_types._mesh_axis import _reset
from spmd_types._type_attr import get_local_type
from spmd_types.types import normalize_axis
from torch.func import vmap
from torch.testing._internal.distributed.fake_pg import FakeStore


class TestVmapTypeChecking(unittest.TestCase):
    """Verifies that vmap propagates SPMD type annotations."""

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        cls.pg = dist.distributed_c10d._get_default_group()
        cls.axis = normalize_axis(cls.pg)

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def test_vmap_preserves_type_on_mapped_input(self):
        """vmap-mapped input retains its SPMD type annotation."""
        x = torch.randn(4, 8)
        observed_types = []

        def fn(row):
            observed_types.append(get_local_type(row))
            return row * 2.0

        with typecheck():
            assert_type(x, {self.pg: R})
            vmap(fn)(x)

        self.assertEqual(observed_types[0], {self.axis: R})

    def test_vmap_output_carries_type(self):
        """Output of vmap carries the SPMD type from inside the vmapped fn."""
        x = torch.randn(4, 8)

        with typecheck():
            assert_type(x, {self.pg: R})
            out = vmap(lambda row: row * 2.0)(x)

        self.assertEqual(get_local_type(out), {self.axis: R})

    def test_vmap_captured_tensor_type_propagates(self):
        """Captured typed tensor mixes with vmapped input correctly."""
        x = torch.randn(4, 8)
        bias = torch.randn(8)

        with typecheck():
            assert_type(x, {self.pg: R})
            assert_type(bias, {self.pg: V})

            out = vmap(lambda row: row + bias)(x)

        # R + V -> V
        self.assertEqual(get_local_type(out), {self.axis: V})

    def test_vmap_pytree_input(self):
        """vmap with pytree (dict) input propagates types correctly."""
        x = torch.randn(4, 8)
        y = torch.randn(4, 3)

        with typecheck():
            assert_type(x, {self.pg: R})
            assert_type(y, {self.pg: V})

            out = vmap(lambda d: d["a"] * 2.0 + d["b"].sum(-1, keepdim=True))(
                {"a": x, "b": y}
            )

        # R * scalar -> R, V -> V, R + V -> V
        self.assertEqual(get_local_type(out), {self.axis: V})

    def test_vmap_multi_arg(self):
        """vmap with multiple positional args propagates types correctly."""
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        with typecheck():
            assert_type(x, {self.pg: R})
            assert_type(y, {self.pg: V})

            out = vmap(lambda a, b: a + b)(x, y)

        self.assertEqual(get_local_type(out), {self.axis: V})

    def test_without_vmap_types_propagate(self):
        """Sanity check: same operation without vmap propagates types correctly."""
        x = torch.randn(4, 8)
        bias = torch.randn(8)

        with typecheck():
            assert_type(x, {self.pg: R})
            assert_type(bias, {self.pg: V})

            out = x + bias

            self.assertEqual(get_local_type(out), {self.axis: V})


if __name__ == "__main__":
    unittest.main()
