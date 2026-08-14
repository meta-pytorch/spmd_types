# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from spmd_types import (
    assert_local_block,
    assert_type,
    I,
    P,
    PartitionSpec,
    R,
    V,
)
from spmd_types._mesh_axis import MeshAxis
from spmd_types.types import SpmdTypeError


class TestAssertLocalBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.tp = MeshAxis.of(4, stride=1)
        cls.dp = MeshAxis.of(4, stride=4)

    def test_replicated_and_invariant_tensor(self):
        tensor = torch.randn(4, 8)
        assert_type(tensor, {self.dp: R, self.tp: I})

        self.assertIs(assert_local_block(tensor), tensor)

    def test_qualified_multi_axis_leading_shard(self):
        tensor = torch.randn(2, 3, 4, 8)
        assert_type(
            tensor,
            {self.dp: V, self.tp: V},
            PartitionSpec((self.dp, self.tp), None, None, None),
        )

        self.assertIs(assert_local_block(tensor), tensor)

    def test_custom_trailing_dims(self):
        tensor = torch.randn(2, 3, 4)
        assert_type(tensor, {self.tp: V}, PartitionSpec(None, self.tp, None))

        self.assertIs(assert_local_block(tensor, trailing_dims=1), tensor)
        with self.assertRaisesRegex(SpmdTypeError, "final 2 dimensions"):
            assert_local_block(tensor)

    def test_rejects_sharded_trailing_dimension(self):
        tensor = torch.randn(2, 3, 4)
        assert_type(tensor, {self.tp: V}, PartitionSpec(None, None, self.tp))

        with self.assertRaisesRegex(SpmdTypeError, "dimension 2 is sharded"):
            assert_local_block(tensor)

    def test_rejects_partial(self):
        tensor = torch.randn(4, 8)
        assert_type(tensor, {self.tp: P})

        with self.assertRaisesRegex(SpmdTypeError, "Partial on axis"):
            assert_local_block(tensor)

    def test_rejects_local_only_varying(self):
        tensor = torch.randn(4, 8)
        assert_type(tensor, {self.tp: V})

        with self.assertRaisesRegex(SpmdTypeError, "without a PartitionSpec"):
            assert_local_block(tensor)

    def test_rejects_unannotated_tensor(self):
        with self.assertRaisesRegex(SpmdTypeError, "globally annotated"):
            assert_local_block(torch.randn(4, 8))

    def test_validates_trailing_dims(self):
        tensor = torch.randn(4, 8)
        assert_type(tensor, {self.tp: R})

        for trailing_dims in (0, -1):
            with self.subTest(trailing_dims=trailing_dims):
                with self.assertRaisesRegex(ValueError, "at least 1"):
                    assert_local_block(tensor, trailing_dims)
        with self.assertRaisesRegex(ValueError, "rank-2"):
            assert_local_block(tensor, 3)
        for trailing_dims in (True, 2.0):
            with self.subTest(trailing_dims=trailing_dims):
                with self.assertRaisesRegex(TypeError, "to be an integer"):
                    assert_local_block(tensor, trailing_dims)


if __name__ == "__main__":
    unittest.main()
