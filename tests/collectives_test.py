"""
Tests for collective operations: all_reduce, all_gather, reduce_scatter, all_to_all.

Covers: _collectives.py
"""

import unittest

import torch
import torch.distributed as dist
from spmd_types import all_gather, all_reduce, all_to_all, I, P, R, reduce_scatter, S, V
from spmd_types._checker import assert_type, typecheck
from spmd_types._test_utils import LocalTensorTestCase
from spmd_types._type_attr import get_axis_local_type
from spmd_types.types import MeshAxis, SpmdTypeError
from torch.distributed.device_mesh import init_device_mesh


class TestAllReduce(LocalTensorTestCase):
    """Test all_reduce operation: P -> R | I."""

    def test_all_reduce_p_to_r(self):
        """all_reduce(R): P -> R, sums across ranks. Tests forward and backward."""
        # Create "partial" input - different values per rank that need summing
        x = self._generate_inputs((4,), self.pg, P)

        # Get expected sum before all_reduce modifies x in-place
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        # Run all_reduce
        with typecheck():
            result = all_reduce(x, self.pg, src=P, dst=R)

        # Check all ranks have the same summed value
        self._assert_all_ranks_equal(
            result, "all_reduce result should be same on all ranks"
        )
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_all_reduce_p_to_i(self):
        """all_reduce(I): P -> I, sums across ranks. Tests forward and backward."""
        x = self._generate_inputs((4,), self.pg, P)
        # Get expected sum before all_reduce modifies x in-place
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        with typecheck():
            result = all_reduce(x, self.pg, src=P, dst=I)

        self._assert_all_ranks_equal(
            result, "all_reduce result should be same on all ranks"
        )
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_all_reduce_invalid_src(self):
        """all_reduce only accepts partial src."""
        x = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(ValueError) as ctx:
            all_reduce(x, self.pg, src=R, dst=R)
        self.assertIn("must be P", str(ctx.exception))

    def test_all_reduce_invalid_dst(self):
        """all_reduce only accepts replicate or invariant dst."""
        x = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(ValueError) as ctx:
            all_reduce(x, self.pg, src=P, dst=V)
        self.assertIn("must be R or I", str(ctx.exception))

    def test_all_reduce_p_to_r_inplace(self):
        """all_reduce(R, inplace=True): P -> R in-place. Result shares storage with input."""
        x = self._generate_inputs((4,), self.pg, P)
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        # Save data pointers before
        input_ptrs = {r: x._local_tensors[r].data_ptr() for r in range(self.WORLD_SIZE)}

        with typecheck():
            result = all_reduce(x, self.pg, src=P, dst=R, inplace=True)

        # Check correctness
        self._assert_all_ranks_equal(
            result, "all_reduce inplace result should be same on all ranks"
        )
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)
        self.assertIs(get_axis_local_type(result, self.pg), R)

        # Check in-place: returned tensor shares storage with input
        for r in range(self.WORLD_SIZE):
            self.assertEqual(
                result._local_tensors[r].data_ptr(),
                input_ptrs[r],
                f"rank {r}: inplace result should share storage with input",
            )

    def test_all_reduce_p_to_i_inplace(self):
        """all_reduce(I, inplace=True): P -> I in-place. Result shares storage with input."""
        x = self._generate_inputs((4,), self.pg, P)
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        # Save data pointers before
        input_ptrs = {r: x._local_tensors[r].data_ptr() for r in range(self.WORLD_SIZE)}

        with typecheck():
            result = all_reduce(x, self.pg, src=P, dst=I, inplace=True)

        # Check correctness
        self._assert_all_ranks_equal(
            result, "all_reduce inplace result should be same on all ranks"
        )
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)
        self.assertIs(get_axis_local_type(result, self.pg), I)

        # Check in-place: returned tensor shares storage with input
        for r in range(self.WORLD_SIZE):
            self.assertEqual(
                result._local_tensors[r].data_ptr(),
                input_ptrs[r],
                f"rank {r}: inplace result should share storage with input",
            )

    def test_all_reduce_type_mismatch(self):
        """all_reduce raises SpmdTypeError when input type doesn't match src."""
        x = self._generate_inputs((4,), self.pg, R)
        with typecheck():
            with self.assertRaises(SpmdTypeError):
                all_reduce(x, self.pg, src=P, dst=R)


class TestAllReduceMultiAxis(LocalTensorTestCase):
    """Test all_reduce with multiple mesh axes (requires WORLD_SIZE=6)."""

    WORLD_SIZE = 6

    def test_all_reduce_preserves_other_axes(self):
        """all_reduce preserves SPMD types on other mesh axes."""
        mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))
        tp = mesh.get_group("tp")
        dp = mesh.get_group("dp")
        x = self.mode.rank_map(lambda r: torch.randn(4) + r)
        assert_type(x, {tp: P, dp: R})
        with typecheck():
            result = all_reduce(x, tp, src=P, dst=R)
        self.assertIs(get_axis_local_type(result, tp), R)
        self.assertIs(get_axis_local_type(result, dp), R)


class TestAllGather(LocalTensorTestCase):
    """Test all_gather operation: V -> R | I."""

    def test_all_gather_v_to_r(self):
        """all_gather(R): V -> R, gathers shards from all ranks."""
        # Create varying input - different per rank
        x = self.mode.rank_map(lambda r: torch.tensor(float(r)))
        assert_type(x, {self.pg: V})

        with typecheck():
            result = all_gather(x, self.pg, src=V, dst=R)

        # Result should be [0, 1, 2] on all ranks (stack)
        expected = torch.tensor([0.0, 1.0, 2.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_all_gather_v_to_i(self):
        """all_gather(I): V -> I, gathers shards from all ranks."""
        x = self.mode.rank_map(lambda r: torch.tensor(float(r) * 2))
        assert_type(x, {self.pg: V})

        with typecheck():
            result = all_gather(x, self.pg, src=V, dst=I)

        expected = torch.tensor([0.0, 2.0, 4.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_all_gather_shard_to_r(self):
        """all_gather(R): S(i) -> R, gathers shards from all ranks."""
        # Create sharded input on dim 0
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {self.pg: V})

        with typecheck():
            result = all_gather(x, self.pg, src=S(0), dst=R)

        # Result should be concatenated shards
        expected = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_all_gather_invalid_src(self):
        """all_gather only accepts V or S(i) src."""
        x = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(ValueError) as ctx:
            all_gather(x, self.pg, src=R, dst=R)
        self.assertIn("must be V or S(i)", str(ctx.exception))

    def test_all_gather_invalid_dst(self):
        """all_gather only accepts replicate or invariant dst."""
        x = self._generate_inputs((4,), self.pg, V)
        with self.assertRaises(ValueError) as ctx:
            all_gather(x, self.pg, src=V, dst=P)
        self.assertIn("must be R or I", str(ctx.exception))


class TestReduceScatter(LocalTensorTestCase):
    """Test reduce_scatter operation: P -> V."""

    def test_reduce_scatter(self):
        """reduce_scatter: P -> V, reduces and scatters."""
        # Create input with world_size chunks per rank
        # Each rank has [A_r, B_r, C_r] where total length is world_size * chunk_size
        chunk_size = 2
        x = self.rank_map(
            lambda r: (
                torch.arange(self.WORLD_SIZE * chunk_size, dtype=torch.float).reshape(
                    self.WORLD_SIZE, chunk_size
                )
                + r
            )
        )
        assert_type(x, {self.pg: P})

        with typecheck():
            result = reduce_scatter(x, self.pg, src=P, dst=V, scatter_dim=0)

        # Each rank r gets the sum of chunk r from all ranks
        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(chunk_size)
            for src_rank in range(self.WORLD_SIZE):
                expected += x._local_tensors[src_rank][r]
            torch.testing.assert_close(
                result._local_tensors[r],
                expected,
                msg=f"rank {r}",
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_reduce_scatter_to_shard(self):
        """reduce_scatter: P -> S(i), reduces and scatters."""
        chunk_size = 2
        x = self.rank_map(
            lambda r: torch.arange(self.WORLD_SIZE * chunk_size, dtype=torch.float) + r
        )
        assert_type(x, {self.pg: P})

        with typecheck():
            result = reduce_scatter(x, self.pg, src=P, dst=S(0))

        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(chunk_size)
            for src_rank in range(self.WORLD_SIZE):
                chunk_start = r * chunk_size
                chunk_end = (r + 1) * chunk_size
                expected += x._local_tensors[src_rank][chunk_start:chunk_end]
            torch.testing.assert_close(
                result._local_tensors[r],
                expected,
                msg=f"rank {r}",
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_reduce_scatter_invalid_src(self):
        """reduce_scatter only accepts P (or V, which is auto-reinterpreted) src."""
        x = self._generate_inputs((6,), self.pg, R)
        with self.assertRaises(ValueError) as ctx:
            reduce_scatter(x, self.pg, src=R, dst=V)
        self.assertIn("must be P", str(ctx.exception))

    def test_reduce_scatter_invalid_dst(self):
        """reduce_scatter only accepts V or S(i) dst."""
        x = self._generate_inputs((6,), self.pg, P)
        with self.assertRaises(ValueError) as ctx:
            reduce_scatter(x, self.pg, src=P, dst=R)
        self.assertIn("must be V or S(i)", str(ctx.exception))


class TestAllToAll(LocalTensorTestCase):
    """Test all_to_all operation: V -> V."""

    def test_all_to_all(self):
        """all_to_all: V -> V, transposes mesh and tensor dims."""
        # Create input: rank r has [r*3, r*3+1, r*3+2]
        # After all_to_all, rank r should get [r, r+3, r+6]
        x = self.mode.rank_map(
            lambda r: torch.tensor([float(r * 3 + i) for i in range(self.WORLD_SIZE)])
        )
        assert_type(x, {self.pg: V})

        with typecheck():
            result = all_to_all(x, self.pg, src=V, dst=V, split_dim=0, concat_dim=0)

        # Check result
        for r in range(self.WORLD_SIZE):
            expected = torch.tensor([float(r + i * 3) for i in range(self.WORLD_SIZE)])
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_all_to_all_shard_to_shard(self):
        """all_to_all: S(i) -> S(j), resharding between dimensions."""
        # Each rank has a 2D tensor; dim 1 must be divisible by world_size
        # for S(0)->S(1) since split_dim=dst.dim=1.
        x = self.mode.rank_map(
            lambda r: torch.arange(6, dtype=torch.float).reshape(2, 3) + r * 10
        )
        assert_type(x, {self.pg: V})

        with typecheck():
            result = all_to_all(x, self.pg, src=S(0), dst=S(1))

        # Split on dim 1 (size 3/3=1), concat on dim 0 (2*3=6)
        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape[0], 6)
            self.assertEqual(result._local_tensors[r].shape[1], 1)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_all_to_all_invalid_src(self):
        """all_to_all only accepts V or S(i) src."""
        x = self._generate_inputs((6,), self.pg, R)
        with self.assertRaises(ValueError) as ctx:
            all_to_all(x, self.pg, src=R, dst=V)
        self.assertIn("must be V or S(i)", str(ctx.exception))

    def test_all_to_all_invalid_dst(self):
        """all_to_all only accepts V or S(i) dst."""
        x = self._generate_inputs((6,), self.pg, V)
        with self.assertRaises(ValueError) as ctx:
            all_to_all(x, self.pg, src=V, dst=R)
        self.assertIn("must be V or S(i)", str(ctx.exception))

    def test_all_to_all_v_v_backward(self):
        """all_to_all(V, V) backward satisfies the adjoint identity."""
        x = self._generate_inputs((self.WORLD_SIZE,), self.pg, V)
        self.spmd_gradcheck(
            lambda t: all_to_all(t, self.pg, src=V, dst=V, split_dim=0, concat_dim=0),
            x,
            self.pg,
            V,
            V,
        )

    def test_all_to_all_shard_backward(self):
        """all_to_all(S(0), S(1)) backward satisfies the adjoint identity."""
        # Each rank has a 2D tensor; both dims divisible by WORLD_SIZE.
        x = self._generate_inputs((self.WORLD_SIZE, self.WORLD_SIZE), self.pg, V)
        self.spmd_gradcheck(
            lambda t: all_to_all(t, self.pg, src=S(0), dst=S(1)),
            x,
            self.pg,
            V,  # S(0) is V at the base level
            V,  # S(1) is V at the base level
        )


class TestAllToAllUneven(LocalTensorTestCase):
    """Test all_to_all with explicit split sizes (_AllToAllUneven path)."""

    def _make_input(self, total, D):
        """Create a V-typed input of shape (total, D) with deterministic data."""
        x = self.mode.rank_map(
            lambda r: (
                torch.arange(total * D, dtype=torch.float).reshape(total, D) + r * 100
            )
        )
        assert_type(x, {self.pg: V})
        return x

    def test_all_to_all_uneven_both_splits(self):
        """all_to_all with both output_split_sizes and input_split_sizes matches even baseline."""
        D = 4
        total = 6  # 6 / world_size(3) = 2 per rank
        even_splits = [2, 2, 2]

        x_baseline = self._make_input(total, D)
        with typecheck():
            baseline = all_to_all(x_baseline, self.pg, src=S(0), dst=S(0))

        x = self._make_input(total, D)
        with typecheck():
            result = all_to_all(
                x,
                self.pg,
                src=S(0),
                dst=S(0),
                output_split_sizes=even_splits,
                input_split_sizes=even_splits,
            )

        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape, (total, D))
            torch.testing.assert_close(
                result._local_tensors[r],
                baseline._local_tensors[r],
                msg=f"rank {r}",
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_all_to_all_uneven_only_output_splits(self):
        """all_to_all with output_split_sizes only; input_split_sizes=None defaults to even."""
        D = 4
        total = 6
        even_splits = [2, 2, 2]

        x_baseline = self._make_input(total, D)
        with typecheck():
            baseline = all_to_all(x_baseline, self.pg, src=S(0), dst=S(0))

        x = self._make_input(total, D)
        with typecheck():
            result = all_to_all(
                x,
                self.pg,
                src=S(0),
                dst=S(0),
                output_split_sizes=even_splits,
            )

        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape, (total, D))
            torch.testing.assert_close(
                result._local_tensors[r],
                baseline._local_tensors[r],
                msg=f"rank {r}",
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_all_to_all_uneven_only_input_splits(self):
        """all_to_all with input_split_sizes only; output_split_sizes=None defaults to even."""
        D = 4
        total = 6
        even_splits = [2, 2, 2]

        x_baseline = self._make_input(total, D)
        with typecheck():
            baseline = all_to_all(x_baseline, self.pg, src=S(0), dst=S(0))

        x = self._make_input(total, D)
        with typecheck():
            result = all_to_all(
                x,
                self.pg,
                src=S(0),
                dst=S(0),
                input_split_sizes=even_splits,
            )

        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape, (total, D))
            torch.testing.assert_close(
                result._local_tensors[r],
                baseline._local_tensors[r],
                msg=f"rank {r}",
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_all_to_all_uneven_backward(self):
        """all_to_all with uneven splits backward satisfies the adjoint identity."""
        D = 4
        total = 6
        even_splits = [2, 2, 2]
        x = self._generate_inputs((total, D), self.pg, V)
        self.spmd_gradcheck(
            lambda t: all_to_all(
                t,
                self.pg,
                src=S(0),
                dst=S(0),
                output_split_sizes=even_splits,
                input_split_sizes=even_splits,
            ),
            x,
            self.pg,
            V,
            V,
        )


class TestAllGatherUnevenShard(LocalTensorTestCase):
    """Test all_gather with uneven split_sizes on S(i)."""

    def test_all_gather_uneven_shard_to_r(self):
        """all_gather(R) with split_sizes: S(0) -> R, uneven shards."""
        # World size is 3. Create per-rank shards of sizes [3, 2, 2].
        split_sizes = [3, 2, 2]
        x = self.rank_map(
            lambda r: torch.arange(split_sizes[r], dtype=torch.float) + r * 10
        )
        assert_type(x, {self.pg: V})

        with typecheck():
            result = all_gather(x, self.pg, src=S(0), dst=R, split_sizes=split_sizes)

        # Result should be concatenation of all shards on all ranks
        expected = torch.cat(
            [
                torch.arange(split_sizes[r], dtype=torch.float) + r * 10
                for r in range(self.WORLD_SIZE)
            ]
        )
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_all_gather_uneven_shard_to_i(self):
        """all_gather(I) with split_sizes: S(0) -> I, uneven shards."""
        split_sizes = [3, 2, 2]
        x = self.rank_map(
            lambda r: torch.arange(split_sizes[r], dtype=torch.float) + r * 10
        )
        assert_type(x, {self.pg: V})

        with typecheck():
            result = all_gather(x, self.pg, src=S(0), dst=I, split_sizes=split_sizes)

        expected = torch.cat(
            [
                torch.arange(split_sizes[r], dtype=torch.float) + r * 10
                for r in range(self.WORLD_SIZE)
            ]
        )
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_all_gather_uneven_shard_2d(self):
        """all_gather with split_sizes on 2D tensors: S(0) -> R."""
        split_sizes = [2, 3, 1]
        D = 4
        x = self.mode.rank_map(lambda r: torch.randn(split_sizes[r], D))
        assert_type(x, {self.pg: V})

        with typecheck():
            result = all_gather(x, self.pg, src=S(0), dst=R, split_sizes=split_sizes)

        # Result should have shape (sum(split_sizes), D) = (6, 4)
        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape, (6, D))
        self._assert_all_ranks_equal(result)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_all_gather_split_sizes_rejected_for_v(self):
        """split_sizes should be rejected when src=V."""
        x = self.mode.rank_map(lambda r: torch.tensor(float(r)))
        assert_type(x, {self.pg: V})
        with self.assertRaises(ValueError) as ctx:
            all_gather(x, self.pg, src=V, dst=R, split_sizes=[1, 1, 1])
        self.assertIn("only supported with src=S(i)", str(ctx.exception))


class TestReduceScatterUnevenShard(LocalTensorTestCase):
    """Test reduce_scatter with uneven split_sizes on S(i)."""

    @unittest.skip(
        "LocalTensorMode does not support c10d.reduce_scatter_ (list API). "
        "Covered by test_mlp_spmd_types_uneven_split with real GPUs."
    )
    def test_reduce_scatter_uneven_shard(self):
        """reduce_scatter with split_sizes: P -> S(0), uneven chunks."""
        split_sizes = [3, 2, 2]
        total = sum(split_sizes)
        x = self.mode.rank_map(lambda r: torch.arange(total, dtype=torch.float) + r)
        assert_type(x, {self.pg: P})

        result = reduce_scatter(x, self.pg, src=P, dst=S(0), split_sizes=split_sizes)

        # Each rank r gets the sum of chunk r from all ranks
        for r in range(self.WORLD_SIZE):
            start = sum(split_sizes[:r])
            end = start + split_sizes[r]
            expected = torch.zeros(split_sizes[r])
            for src_rank in range(self.WORLD_SIZE):
                expected += x._local_tensors[src_rank][start:end]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_reduce_scatter_split_sizes_rejected_for_v(self):
        """split_sizes should be rejected when dst=V."""
        x = self.mode.rank_map(
            lambda r: torch.arange(9, dtype=torch.float).reshape(3, 3) + r
        )
        assert_type(x, {self.pg: P})
        with self.assertRaises(ValueError) as ctx:
            reduce_scatter(x, self.pg, src=P, dst=V, split_sizes=[3, 3, 3])
        self.assertIn("only supported with dst=S(i)", str(ctx.exception))


class TestAllGatherMultiDim(LocalTensorTestCase):
    """Test all_gather with different gather dimensions."""

    def test_all_gather_dim_1(self):
        """all_gather with S(1) concatenates on dim 1."""
        # Each rank has shape (2, 1)
        x = self.mode.rank_map(lambda r: torch.tensor([[float(r)], [float(r + 10)]]))
        assert_type(x, {self.pg: V})

        with typecheck():
            result = all_gather(x, self.pg, src=S(1), dst=R)

        # Result should have shape (2, 3) on all ranks (cat along dim 1)
        expected = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_all_gather_2d_tensors(self):
        """all_gather with 2D varying tensors."""
        # Each rank has a 2x2 matrix
        x = self.mode.rank_map(lambda r: torch.full((2, 2), float(r)))
        assert_type(x, {self.pg: V})

        with typecheck():
            result = all_gather(x, self.pg, src=V, dst=R)

        # Result should be (3, 2, 2) - stack along dim 0
        expected = torch.stack(
            [torch.full((2, 2), float(r)) for r in range(self.WORLD_SIZE)],
            dim=0,
        )
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, self.pg), R)


class TestAllToAllMultiDim(LocalTensorTestCase):
    """Test all_to_all with different split/concat dimensions."""

    def test_all_to_all_2d_same_dims(self):
        """all_to_all with 2D tensors, split and concat on same dim."""
        # Each rank has shape (3, 2) - split on dim 0
        x = self.mode.rank_map(
            lambda r: torch.arange(6, dtype=torch.float).reshape(3, 2) + r * 10
        )
        assert_type(x, {self.pg: V})

        result = all_to_all(x, self.pg, split_dim=0, concat_dim=0)

        # After all_to_all:
        # Rank 0 gets [0th row from rank 0, 0th row from rank 1, 0th row from rank 2]
        # etc.
        for r in range(self.WORLD_SIZE):
            result_tensor = result._local_tensors[r]
            self.assertEqual(result_tensor.shape, (3, 2))


class TestRawDistCollective(LocalTensorTestCase):
    """Test type checking of raw torch.distributed collectives."""

    def test_raw_all_gather_into_tensor_v_to_r(self):
        """all_gather_into_tensor with V input and R output passes."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {self.pg: V})
        output = self.mode.rank_map(lambda r: torch.empty(self.WORLD_SIZE * 2))
        assert_type(output, {self.pg: R})

        with typecheck():
            dist.all_gather_into_tensor(output, x, group=pg)

        # Output retains its R annotation
        self.assertIs(get_axis_local_type(output, self.pg), R)

    def test_raw_all_gather_into_tensor_v_to_i(self):
        """all_gather_into_tensor with V input and I output passes."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {self.pg: V})
        output = self.mode.rank_map(lambda r: torch.empty(self.WORLD_SIZE * 2))
        assert_type(output, {self.pg: I})

        with typecheck():
            dist.all_gather_into_tensor(output, x, group=pg)

        self.assertIs(get_axis_local_type(output, self.pg), I)

    def test_raw_all_gather_into_tensor_wrong_input_type(self):
        """all_gather_into_tensor with R input raises SpmdTypeError."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([1.0, 2.0]))
        assert_type(x, {self.pg: R})
        output = self.mode.rank_map(lambda r: torch.empty(self.WORLD_SIZE * 2))
        assert_type(output, {self.pg: R})

        with typecheck():
            with self.assertRaises(SpmdTypeError):
                dist.all_gather_into_tensor(output, x, group=pg)

    def test_raw_all_gather_into_tensor_wrong_output_type(self):
        """all_gather_into_tensor with V output raises SpmdTypeError."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {self.pg: V})
        output = self.mode.rank_map(lambda r: torch.empty(self.WORLD_SIZE * 2))
        assert_type(output, {self.pg: V})

        with typecheck():
            with self.assertRaises(SpmdTypeError):
                dist.all_gather_into_tensor(output, x, group=pg)

    # -----------------------------------------------------------------
    # all_reduce: P -> R (in-place)
    # -----------------------------------------------------------------

    def test_raw_all_reduce_p_to_r(self):
        """all_reduce with P input mutates type to R."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {pg: P})

        with typecheck():
            dist.all_reduce(x, group=pg)

        # Type mutated in-place from P to R
        self.assertIs(get_axis_local_type(x, pg), R)

    def test_raw_all_reduce_wrong_input_type(self):
        """all_reduce with R input raises SpmdTypeError."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([1.0, 2.0]))
        assert_type(x, {pg: R})

        with typecheck():
            with self.assertRaises(SpmdTypeError):
                dist.all_reduce(x, group=pg)

    def test_raw_all_reduce_wrong_input_type_v(self):
        """all_reduce with V input raises SpmdTypeError."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)]))
        assert_type(x, {pg: V})

        with typecheck():
            with self.assertRaises(SpmdTypeError):
                dist.all_reduce(x, group=pg)

    # -----------------------------------------------------------------
    # reduce_scatter_tensor: P -> V
    # -----------------------------------------------------------------

    def test_raw_reduce_scatter_tensor_p_to_v(self):
        """reduce_scatter_tensor with P input and V output passes."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)] * self.WORLD_SIZE))
        assert_type(x, {pg: P})
        output = self.mode.rank_map(lambda r: torch.tensor([0.0]))
        assert_type(output, {pg: V})

        with typecheck():
            dist.reduce_scatter_tensor(output, x, group=pg)

        self.assertIs(get_axis_local_type(output, pg), V)

    def test_raw_reduce_scatter_tensor_wrong_input_type(self):
        """reduce_scatter_tensor with R input raises SpmdTypeError."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([1.0] * self.WORLD_SIZE))
        assert_type(x, {pg: R})
        output = self.mode.rank_map(lambda r: torch.tensor([0.0]))
        assert_type(output, {pg: V})

        with typecheck():
            with self.assertRaises(SpmdTypeError):
                dist.reduce_scatter_tensor(output, x, group=pg)

    def test_raw_reduce_scatter_tensor_wrong_output_type(self):
        """reduce_scatter_tensor with R output raises SpmdTypeError."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)] * self.WORLD_SIZE))
        assert_type(x, {pg: P})
        output = self.mode.rank_map(lambda r: torch.tensor([0.0]))
        assert_type(output, {pg: R})

        with typecheck():
            with self.assertRaises(SpmdTypeError):
                dist.reduce_scatter_tensor(output, x, group=pg)

    # -----------------------------------------------------------------
    # all_to_all_single: V -> V
    # -----------------------------------------------------------------

    def test_raw_all_to_all_single_v_to_v(self):
        """all_to_all_single with V input and V output passes."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)] * self.WORLD_SIZE))
        assert_type(x, {pg: V})
        output = self.mode.rank_map(lambda r: torch.empty(self.WORLD_SIZE))
        assert_type(output, {pg: V})

        with typecheck():
            dist.all_to_all_single(output, x, group=pg)

        self.assertIs(get_axis_local_type(output, pg), V)

    def test_raw_all_to_all_single_wrong_input_type(self):
        """all_to_all_single with R input raises SpmdTypeError."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([1.0] * self.WORLD_SIZE))
        assert_type(x, {pg: R})
        output = self.mode.rank_map(lambda r: torch.empty(self.WORLD_SIZE))
        assert_type(output, {pg: V})

        with typecheck():
            with self.assertRaises(SpmdTypeError):
                dist.all_to_all_single(output, x, group=pg)

    def test_raw_all_to_all_single_wrong_output_type(self):
        """all_to_all_single with R output raises SpmdTypeError."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)] * self.WORLD_SIZE))
        assert_type(x, {pg: V})
        output = self.mode.rank_map(lambda r: torch.empty(self.WORLD_SIZE))
        assert_type(output, {pg: R})

        with typecheck():
            with self.assertRaises(SpmdTypeError):
                dist.all_to_all_single(output, x, group=pg)

    # -----------------------------------------------------------------
    # all_gather (list-based): V -> R|I
    # -----------------------------------------------------------------

    def test_raw_all_gather_v_to_r(self):
        """all_gather with V input and R output tensors passes."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {pg: V})
        tensor_list = [
            self.mode.rank_map(lambda r: torch.empty(2)) for _ in range(self.WORLD_SIZE)
        ]
        for t in tensor_list:
            assert_type(t, {pg: R})

        with typecheck():
            dist.all_gather(tensor_list, x, group=pg)

        for t in tensor_list:
            self.assertIs(get_axis_local_type(t, pg), R)

    def test_raw_all_gather_v_to_i(self):
        """all_gather with V input and I output tensors passes."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {pg: V})
        tensor_list = [
            self.mode.rank_map(lambda r: torch.empty(2)) for _ in range(self.WORLD_SIZE)
        ]
        for t in tensor_list:
            assert_type(t, {pg: I})

        with typecheck():
            dist.all_gather(tensor_list, x, group=pg)

        for t in tensor_list:
            self.assertIs(get_axis_local_type(t, pg), I)

    def test_raw_all_gather_wrong_input_type(self):
        """all_gather with R input raises SpmdTypeError."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([1.0, 2.0]))
        assert_type(x, {pg: R})
        tensor_list = [
            self.mode.rank_map(lambda r: torch.empty(2)) for _ in range(self.WORLD_SIZE)
        ]
        for t in tensor_list:
            assert_type(t, {pg: R})

        with typecheck():
            with self.assertRaises(SpmdTypeError):
                dist.all_gather(tensor_list, x, group=pg)

    def test_raw_all_gather_wrong_output_type(self):
        """all_gather with V output tensors raises SpmdTypeError."""
        pg = self.pg
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {pg: V})
        tensor_list = [
            self.mode.rank_map(lambda r: torch.empty(2)) for _ in range(self.WORLD_SIZE)
        ]
        for t in tensor_list:
            assert_type(t, {pg: V})

        with typecheck():
            with self.assertRaises(SpmdTypeError):
                dist.all_gather(tensor_list, x, group=pg)


class TestRawDistUnknownGroup(LocalTensorTestCase):
    """Test raw dist collectives with a group not in the tensor's type dict.

    In strict mode this should raise; in permissive mode it should skip.

    Tensors are typed on ``other_axis`` (a MeshAxis with a different layout
    than self.pg) so that when the collective runs on self.pg the axis
    lookup genuinely misses.
    """

    def setUp(self):
        super().setUp()
        # A MeshAxis with the same size but different stride than self.pg,
        # so it normalizes to a distinct key.
        self.other_axis = MeshAxis.of(self.WORLD_SIZE, 2)

    # -----------------------------------------------------------------
    # Strict mode (default): unknown group raises SpmdTypeError
    # -----------------------------------------------------------------

    def test_strict_unknown_group_input_raises(self):
        """Strict mode raises when input is typed but missing the group axis."""
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {self.other_axis: V})
        output = self.mode.rank_map(lambda r: torch.empty(self.WORLD_SIZE * 2))
        assert_type(output, {self.other_axis: R})

        with typecheck(strict_mode="strict"):
            with self.assertRaises(SpmdTypeError) as ctx:
                dist.all_gather_into_tensor(output, x, group=self.pg)
            self.assertIn("no type for axis", str(ctx.exception))

    def test_strict_unknown_group_output_raises(self):
        """Strict mode raises when output is typed but missing the group axis."""
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {self.pg: V})
        output = self.mode.rank_map(lambda r: torch.empty(self.WORLD_SIZE * 2))
        # Output only typed on other_axis, not self.pg.
        assert_type(output, {self.other_axis: R})

        with typecheck(strict_mode="strict"):
            with self.assertRaises(SpmdTypeError) as ctx:
                dist.all_gather_into_tensor(output, x, group=self.pg)
            self.assertIn("no type for axis", str(ctx.exception))

    def test_strict_unknown_group_all_reduce_raises(self):
        """Strict mode raises for in-place all_reduce with unknown group."""
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)]))
        assert_type(x, {self.other_axis: P})

        with typecheck(strict_mode="strict"):
            with self.assertRaises(SpmdTypeError) as ctx:
                dist.all_reduce(x, group=self.pg)
            self.assertIn("no type for axis", str(ctx.exception))

    # -----------------------------------------------------------------
    # Permissive mode: unknown group skips validation
    # -----------------------------------------------------------------

    def test_permissive_unknown_group_input_skips(self):
        """Permissive mode skips validation for unknown group on input."""
        x = self.mode.rank_map(lambda r: torch.tensor([float(r), float(r) + 0.5]))
        assert_type(x, {self.other_axis: V})
        output = self.mode.rank_map(lambda r: torch.empty(self.WORLD_SIZE * 2))
        assert_type(output, {self.other_axis: R})

        with typecheck(strict_mode="permissive"):
            # Should not raise.
            dist.all_gather_into_tensor(output, x, group=self.pg)

        # Types on the other axis are preserved.
        self.assertIs(get_axis_local_type(x, self.other_axis), V)
        self.assertIs(get_axis_local_type(output, self.other_axis), R)

    def test_permissive_unknown_group_all_reduce_skips_and_mutates(self):
        """Permissive mode skips validation but still applies mutate_src_to."""
        x = self.mode.rank_map(lambda r: torch.tensor([float(r)]))
        assert_type(x, {self.other_axis: P})

        with typecheck(strict_mode="permissive"):
            # Should not raise -- self.pg axis not in type dict, skips.
            dist.all_reduce(x, group=self.pg)

        # Other axis type is unchanged.
        self.assertIs(get_axis_local_type(x, self.other_axis), P)
        # mutate_src_to=R is applied unconditionally -- we know the
        # post-collective type even without validating the pre-collective type.
        self.assertIs(get_axis_local_type(x, self.pg), R)

    def test_permissive_known_group_still_checks(self):
        """Permissive mode still validates when the group IS in the type dict."""
        x = self.mode.rank_map(lambda r: torch.tensor([1.0, 2.0]))
        assert_type(x, {self.pg: R})

        with typecheck(strict_mode="permissive"):
            with self.assertRaises(SpmdTypeError):
                # R is wrong for all_reduce (expects P).
                dist.all_reduce(x, group=self.pg)


class TestFlattenedAxisCollectives(LocalTensorTestCase):
    """Test collective type checking with flattened/merged mesh axes."""

    WORLD_SIZE = 6

    def _make_mesh(self):
        """Create a (2, 3) mesh with dp, cp, and flattened dp_cp axes."""
        mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "cp"))
        dp = mesh.get_group("dp")
        cp = mesh.get_group("cp")
        dp_cp_mesh = mesh["dp", "cp"]._flatten("dp_cp")
        dp_cp = dp_cp_mesh.get_group("dp_cp")
        return dp, cp, dp_cp

    def test_all_reduce_flattened_axis(self):
        """all_reduce on flattened dp_cp axis decomposes to update both dp and cp."""
        dp, cp, dp_cp = self._make_mesh()
        x = self.mode.rank_map(lambda r: torch.randn(4) + r)
        assert_type(x, {dp: P, cp: P})
        with typecheck():
            result = all_reduce(x, dp_cp, src=P, dst=R)
        self.assertIs(get_axis_local_type(result, dp), R)
        self.assertIs(get_axis_local_type(result, cp), R)

    def test_all_reduce_flattened_axis_src_mismatch(self):
        """all_reduce on flattened axis errors when sub-axes have mismatched types."""
        dp, cp, dp_cp = self._make_mesh()
        x = self.mode.rank_map(lambda r: torch.randn(4) + r)
        assert_type(x, {dp: P, cp: R})
        with typecheck():
            with self.assertRaises(SpmdTypeError):
                all_reduce(x, dp_cp, src=P, dst=R)

    def test_all_gather_flattened_axis(self):
        """all_gather on flattened dp_cp axis decomposes to update both dp and cp."""
        dp, cp, dp_cp = self._make_mesh()
        x = self.mode.rank_map(lambda r: torch.randn(4) + r)
        assert_type(x, {dp: V, cp: V})
        with typecheck():
            result = all_gather(x, dp_cp, src=V, dst=R)
        self.assertIs(get_axis_local_type(result, dp), R)
        self.assertIs(get_axis_local_type(result, cp), R)

    def test_reduce_scatter_flattened_axis(self):
        """reduce_scatter on flattened dp_cp axis decomposes to update both dp and cp."""
        dp, cp, dp_cp = self._make_mesh()
        x = self.mode.rank_map(
            lambda r: torch.arange(6, dtype=torch.float).reshape(6, 1) + r
        )
        assert_type(x, {dp: P, cp: P})
        with typecheck():
            result = reduce_scatter(x, dp_cp, src=P, dst=V, scatter_dim=0)
        self.assertIs(get_axis_local_type(result, dp), V)
        self.assertIs(get_axis_local_type(result, cp), V)

    def test_flattened_axis_strict_error_non_decomposable(self):
        """Strict mode errors when flattened axis cannot decompose to tensor axes."""
        dp, cp, dp_cp = self._make_mesh()
        x = self.mode.rank_map(lambda r: torch.randn(4) + r)
        # Only type on dp, not cp -- dp_cp cannot fully decompose
        assert_type(x, {dp: P})
        with typecheck(strict_mode="strict"):
            with self.assertRaises(SpmdTypeError):
                all_reduce(x, dp_cp, src=P, dst=R)


if __name__ == "__main__":
    unittest.main()
