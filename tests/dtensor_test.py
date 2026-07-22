# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from spmd_types import (
    assert_local_block,
    dtensor_placements_to_spmd_type,
    dtensor_to_local,
    get_local_type,
    get_partition_spec,
    I,
    P,
    PartitionSpec,
    SpmdTypeError,
    V,
)
from spmd_types.checker import typecheck
from spmd_types._test_utils import FakeProcessGroupTestCase
from spmd_types.types import normalize_axis
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard


class TestDTensorLocalBridge(FakeProcessGroupTestCase):
    MESH_SHAPE = (2, 2)
    MESH_DIM_NAMES = ("dp", "tp")

    def test_placement_vector_preserves_partition_spec(self):
        spmd_type = dtensor_placements_to_spmd_type(
            self.mesh,
            [Shard(0), Replicate()],
            tensor_ndim=3,
        )
        dp = normalize_axis(self.mesh.get_group("dp"))
        tp = normalize_axis(self.mesh.get_group("tp"))

        self.assertEqual(spmd_type.local_type, {dp: V, tp: I})
        self.assertEqual(spmd_type.partition_spec, PartitionSpec(dp, None, None))

    def test_multi_axis_shard_order_is_preserved(self):
        spmd_type = dtensor_placements_to_spmd_type(
            self.mesh,
            [Shard(0), Shard(0)],
            tensor_ndim=3,
        )
        dp = normalize_axis(self.mesh.get_group("dp"))
        tp = normalize_axis(self.mesh.get_group("tp"))

        self.assertEqual(spmd_type.local_type, {dp: V, tp: V})
        self.assertEqual(
            spmd_type.partition_spec, PartitionSpec((dp, tp), None, None)
        )

    def test_partial_placement_is_preserved(self):
        spmd_type = dtensor_placements_to_spmd_type(
            self.mesh,
            [Partial(), Replicate()],
            tensor_ndim=3,
        )
        dp = normalize_axis(self.mesh.get_group("dp"))
        tp = normalize_axis(self.mesh.get_group("tp"))
        self.assertEqual(spmd_type.local_type, {dp: P, tp: I})
        self.assertIsNone(spmd_type.partition_spec)

    def test_no_grad_local_view_is_typed_and_shares_storage(self):
        local = torch.randn(2, 3, 4)
        tensor = DTensor.from_local(
            local,
            self.mesh,
            [Shard(0), Replicate()],
            run_check=False,
            shape=(4, 3, 4),
            stride=(12, 4, 1),
        )

        with torch.no_grad():
            compute = dtensor_to_local(tensor)

        dp = normalize_axis(self.mesh.get_group("dp"))
        tp = normalize_axis(self.mesh.get_group("tp"))
        self.assertEqual(get_local_type(compute), {dp: V, tp: I})
        self.assertEqual(
            get_partition_spec(compute), PartitionSpec(dp, None, None)
        )
        self.assertEqual(compute.data_ptr(), local.data_ptr())
        assert_local_block(compute, trailing_dims=2)

        compute.add_(1)
        torch.testing.assert_close(tensor.to_local(), local)

    def test_grad_enabled_to_local_preserves_partition_spec(self):
        local = torch.randn(2, 3, 4, requires_grad=True)
        tensor = DTensor.from_local(
            local,
            self.mesh,
            [Shard(0), Replicate()],
            run_check=False,
            shape=(4, 3, 4),
            stride=(12, 4, 1),
        )

        with typecheck():
            compute = tensor.to_local()

        dp = normalize_axis(self.mesh.get_group("dp"))
        self.assertEqual(
            get_partition_spec(compute), PartitionSpec(dp, None, None)
        )
        assert_local_block(compute, trailing_dims=2)

    def test_matrix_dimension_shard_is_rejected(self):
        local = torch.randn(4, 3, 2)
        tensor = DTensor.from_local(
            local,
            self.mesh,
            [Shard(2), Replicate()],
            run_check=False,
            shape=(4, 3, 4),
            stride=(12, 4, 1),
        )
        compute = dtensor_to_local(tensor)

        with self.assertRaisesRegex(SpmdTypeError, "final 2 dimensions"):
            assert_local_block(compute, trailing_dims=2)


if __name__ == "__main__":
    import unittest

    unittest.main()
