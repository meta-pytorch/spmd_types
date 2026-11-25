# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from spmd_types import (
    assert_local_block,
    dtensor_compute_view,
    dtensor_placements_to_spmd_type,
    dtensor_to_local,
    get_local_type,
    get_partition_spec,
    has_local_type,
    I,
    P,
    PartitionSpec,
    R,
    SpmdTypeError,
    V,
)
from spmd_types.checker import typecheck
from spmd_types._test_utils import FakeProcessGroupTestCase
from spmd_types.types import normalize_axis
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard


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

    def test_local_borrows_do_not_leave_sticky_annotations(self):
        local = torch.randn(3, 4)
        tensor = DTensor.from_local(
            local,
            self.mesh,
            [Replicate(), Replicate()],
            run_check=False,
            shape=(3, 4),
            stride=(4, 1),
        )
        dp = normalize_axis(self.mesh.get_group("dp"))

        invariant = dtensor_to_local(tensor)
        replicated = dtensor_to_local(
            tensor,
            grad_placements=[Partial(), Replicate()],
        )

        self.assertEqual(get_local_type(invariant)[dp], I)
        self.assertEqual(get_local_type(replicated)[dp], R)
        self.assertFalse(has_local_type(tensor.to_local()))

    def test_compute_view_redistributes_and_writes_back(self):
        local = torch.arange(6.0).reshape(2, 3)
        tensor = DTensor.from_local(
            local.clone(),
            self.mesh,
            [Shard(0), Replicate()],
            run_check=False,
            shape=(4, 3),
            stride=(3, 1),
        )
        version = tensor._version

        with dtensor_compute_view(
            tensor,
            placements=[Replicate(), Replicate()],
            writeback=True,
        ) as compute:
            self.assertEqual(tuple(compute.shape), (4, 3))
            self.assertIsNone(get_partition_spec(compute))
            compute.add_(10)

        torch.testing.assert_close(tensor.to_local(), local + 10)
        self.assertGreater(tensor._version, version)

    def test_compute_view_replicates_composed_strided_storage(self):
        local = torch.arange(6.0).reshape(2, 3)
        storage_placements = (
            _StridedShard(0, split_factor=self.mesh["tp"].size()),
            Shard(0),
        )
        tensor = DTensor.from_local(
            local.clone(),
            self.mesh,
            storage_placements,
            run_check=False,
            shape=(8, 3),
            stride=(3, 1),
        )

        with dtensor_compute_view(
            tensor,
            placements=[Replicate(), Shard(0)],
            writeback=True,
        ) as compute:
            self.assertEqual(tuple(compute.shape), (4, 3))
            self.assertEqual(
                get_partition_spec(compute),
                PartitionSpec(normalize_axis(self.mesh.get_group("tp")), None),
            )
            compute.add_(10)

        self.assertEqual(tensor.placements, storage_placements)
        torch.testing.assert_close(tensor.to_local(), local + 10)

    def test_compute_view_rejects_unconverted_strided_storage(self):
        tensor = DTensor.from_local(
            torch.arange(6.0).reshape(2, 3),
            self.mesh,
            [_StridedShard(0, split_factor=2), Shard(0)],
            run_check=False,
            shape=(8, 3),
            stride=(3, 1),
        )

        with self.assertRaisesRegex(NotImplementedError, "Strided DTensor shards"):
            with dtensor_compute_view(tensor, writeback=False):
                pass

    def test_read_only_compute_view_does_not_write_redistributed_value(self):
        local = torch.arange(6.0).reshape(2, 3)
        tensor = DTensor.from_local(
            local.clone(),
            self.mesh,
            [Shard(0), Replicate()],
            run_check=False,
            shape=(4, 3),
            stride=(3, 1),
        )

        with dtensor_compute_view(
            tensor,
            placements=[Replicate(), Replicate()],
            writeback=False,
        ) as compute:
            compute.add_(10)  # Deliberate contract violation for this test.

        torch.testing.assert_close(tensor.to_local(), local)

    def test_same_placement_mutation_updates_identity_version(self):
        local = torch.arange(6.0).reshape(2, 3)
        tensor = DTensor.from_local(
            local.clone(),
            self.mesh,
            [Shard(0), Replicate()],
            run_check=False,
            shape=(4, 3),
            stride=(3, 1),
        )
        version = tensor._version

        with dtensor_compute_view(tensor, writeback=True) as compute:
            compute.add_(5)

        torch.testing.assert_close(tensor.to_local(), local + 5)
        self.assertGreater(tensor._version, version)

    def test_mutable_compute_view_is_non_transactional(self):
        local = torch.arange(6.0).reshape(2, 3)
        tensor = DTensor.from_local(
            local.clone(),
            self.mesh,
            [Shard(0), Replicate()],
            run_check=False,
            shape=(4, 3),
            stride=(3, 1),
        )

        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with dtensor_compute_view(
                tensor,
                placements=[Replicate(), Replicate()],
                writeback=True,
            ) as compute:
                compute.add_(7)
                raise RuntimeError("body failed")

        torch.testing.assert_close(tensor.to_local(), local + 7)

    def test_mutable_compute_view_rejects_partial_storage(self):
        tensor = DTensor.from_local(
            torch.randn(3, 4),
            self.mesh,
            [Partial(), Replicate()],
            run_check=False,
            shape=(3, 4),
            stride=(4, 1),
        )

        with self.assertRaisesRegex(ValueError, "Partial storage"):
            with dtensor_compute_view(tensor, writeback=True):
                pass

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
