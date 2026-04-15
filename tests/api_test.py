"""
Tests for cross-module SPMD operations: redistribute, negative dim sharding.

Covers: __init__.py (public API), integration across modules.
"""

import unittest

import torch
from spmd_types import (
    all_gather,
    all_reduce,
    all_to_all,
    convert,
    I,
    P,
    R,
    redistribute,
    reduce_scatter,
    reinterpret,
    S,
    V,
)
from spmd_types._checker import assert_type, typecheck
from spmd_types._test_utils import LocalTensorTestCase
from spmd_types._type_attr import get_axis_local_type


class TestRedistribute(LocalTensorTestCase):
    """Test redistribute operation (semantics-preserving with comms)."""

    def test_redistribute_v_to_r(self):
        """redistribute(V,R) uses all_gather."""
        x = self.mode.rank_map(lambda r: torch.tensor(float(r)))
        assert_type(x, {self.pg: V})

        with typecheck():
            result = redistribute(x, self.pg, src=V, dst=R)

        expected = torch.tensor([0.0, 1.0, 2.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_redistribute_v_to_i(self):
        """redistribute(V,I) uses all_gather."""
        x = self.mode.rank_map(lambda r: torch.tensor(float(r)))
        assert_type(x, {self.pg: V})

        with typecheck():
            result = redistribute(x, self.pg, src=V, dst=I)

        expected = torch.tensor([0.0, 1.0, 2.0])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_redistribute_p_to_r(self):
        """redistribute(P,R) uses all_reduce."""
        x = self._generate_inputs((4,), self.pg, P)
        expected_sum = sum(x._local_tensors[r].clone() for r in range(self.WORLD_SIZE))

        with typecheck():
            result = redistribute(x, self.pg, src=P, dst=R)

        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected_sum)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_redistribute_p_to_s(self):
        """redistribute(P,S(0)) uses reduce_scatter."""
        chunk_size = 2
        x = self.rank_map(
            lambda r: torch.arange(self.WORLD_SIZE * chunk_size, dtype=torch.float) + r
        )
        assert_type(x, {self.pg: P})

        with typecheck():
            result = redistribute(x, self.pg, src=P, dst=S(0))

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

    def test_redistribute_r_to_v_uses_convert(self):
        """redistribute(R,V) delegates to convert."""
        base = torch.arange(6, dtype=torch.float).reshape(self.WORLD_SIZE, 2)
        x = self.rank_map(lambda r: base.clone())
        assert_type(x, {self.pg: R})

        with typecheck():
            result = redistribute(x, self.pg, src=R, dst=V)

        for r in range(self.WORLD_SIZE):
            expected = base[r]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_redistribute_r_to_p_uses_convert(self):
        """redistribute(R,P) delegates to convert."""
        base = torch.tensor([1.0, 2.0, 3.0])
        x = self.mode.rank_map(lambda r: base.clone())
        assert_type(x, {self.pg: R})

        with typecheck():
            result = redistribute(x, self.pg, src=R, dst=P)

        torch.testing.assert_close(result._local_tensors[0], base)
        for r in range(1, self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], torch.zeros_like(base))
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_redistribute_same_type_noop(self):
        """redistribute with same src and dst is identity."""
        x = self._generate_inputs((4,), self.pg, R)
        result = redistribute(x, self.pg, src=R, dst=R)
        self.assertIs(result, x)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_redistribute_shard_to_shard_uses_all_to_all(self):
        """redistribute(S(i),S(j)) with different dims uses all_to_all."""
        # dim 1 must be divisible by world_size for S(0)->S(1)
        # since all_to_all splits on dst.dim=1.
        x = self.mode.rank_map(
            lambda r: torch.arange(6, dtype=torch.float).reshape(2, 3) + r * 10
        )
        assert_type(x, {self.pg: V})

        with typecheck():
            result = redistribute(x, self.pg, src=S(0), dst=S(1))

        # Split on dim 1 (size 3/3=1), concat on dim 0 (2*3=6)
        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape[0], 6)
            self.assertEqual(result._local_tensors[r].shape[1], 1)
        self.assertIs(get_axis_local_type(result, self.pg), V)


class TestShardNegativeDim(LocalTensorTestCase):
    """Test that Shard(-1) (negative dimension indexing) works correctly.

    S(-1) should work like S(ndim-1), similar to how PyTorch handles negative
    dim arguments in operations like torch.cat, torch.chunk, etc.
    """

    def test_reduce_scatter_shard_neg1(self):
        """reduce_scatter with dst=S(-1) should scatter along the last dim.

        Bug: _ReduceScatter.forward always divides output_shape[0] regardless of
        scatter_dim, so dst=S(-1) on a 2D tensor produces shape (2, 9) instead
        of (6, 3).
        """
        # Each rank has shape (6, 9), world_size=3.
        # dst=S(-1) means scatter on last dim (dim 1): each rank gets shape (6, 3).
        x = self.mode.rank_map(
            lambda r: torch.arange(54, dtype=torch.float).reshape(6, 9) + r
        )
        assert_type(x, {self.pg: P})

        result = reduce_scatter(x, self.pg, src=P, dst=S(-1))

        for r in range(self.WORLD_SIZE):
            self.assertEqual(
                result._local_tensors[r].shape,
                (6, 3),
                f"rank {r}: expected shape (6, 3) but got "
                f"{tuple(result._local_tensors[r].shape)}",
            )

    def test_all_gather_shard_neg1(self):
        """all_gather with src=S(-1) should gather along the last dim.

        Forward uses torch.cat(dim=-1) which handles negative dims, so this
        should work. Included to confirm forward is fine (backward calls
        reduce_scatter with S(-1) which is broken).
        """
        # Each rank has shape (2, 1), world_size=3.
        # src=S(-1) = S(1) on 2D, result should be (2, 3).
        x = self.mode.rank_map(lambda r: torch.tensor([[float(r)], [float(r + 10)]]))
        assert_type(x, {self.pg: V})

        with typecheck():
            result = all_gather(x, self.pg, src=S(-1), dst=R)

        expected = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
        self._assert_all_ranks_equal(result)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], expected)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_redistribute_p_to_shard_neg1(self):
        """redistribute(P, S(-1)) delegates to reduce_scatter, which is broken for S(-1)."""
        x = self.mode.rank_map(
            lambda r: torch.arange(54, dtype=torch.float).reshape(6, 9) + r
        )
        assert_type(x, {self.pg: P})

        result = redistribute(x, self.pg, src=P, dst=S(-1))

        for r in range(self.WORLD_SIZE):
            self.assertEqual(
                result._local_tensors[r].shape,
                (6, 3),
                f"rank {r}: expected shape (6, 3) but got "
                f"{tuple(result._local_tensors[r].shape)}",
            )

    def test_convert_r_to_shard_neg1(self):
        """convert(R, S(-1)) should chunk along the last dim."""
        # Each rank has shape (2, 3), world_size=3.
        # S(-1) = S(1) on 2D, each rank gets a (2, 1) chunk.
        base = torch.arange(6, dtype=torch.float).reshape(2, 3)
        x = self.rank_map(lambda r: base.clone())
        assert_type(x, {self.pg: R})

        result = convert(x, self.pg, src=R, dst=S(-1))

        for r in range(self.WORLD_SIZE):
            expected = base[:, r : r + 1]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )


class TestBackwardCorrectness(LocalTensorTestCase):
    """Verify backward correctness for all SPMD operations via adjoint identity.

    For each linear SPMD operation A with backward A*, verifies:
        <A*(g), dx>_{src} = <g, A(dx)>_{dst}
    """

    # --- reinterpret ---

    def test_backward_reinterpret_r_to_v(self):
        x = self._generate_inputs((4,), self.pg, R)
        self.spmd_gradcheck(
            lambda x: reinterpret(x, self.pg, src=R, dst=V, expert_mode=True),
            x,
            axis=self.pg,
            src_type=R,
            dst_type=V,
        )

    def test_backward_reinterpret_r_to_i(self):
        x = self._generate_inputs((4,), self.pg, R)
        self.spmd_gradcheck(
            lambda x: reinterpret(x, self.pg, src=R, dst=I, expert_mode=True),
            x,
            axis=self.pg,
            src_type=R,
            dst_type=I,
        )

    def test_backward_reinterpret_i_to_r(self):
        x = self._generate_inputs((4,), self.pg, I)
        self.spmd_gradcheck(
            lambda x: reinterpret(x, self.pg, src=I, dst=R, expert_mode=True),
            x,
            axis=self.pg,
            src_type=I,
            dst_type=R,
        )

    def test_backward_reinterpret_v_to_p(self):
        x = self._generate_inputs((4,), self.pg, V)
        self.spmd_gradcheck(
            lambda x: reinterpret(x, self.pg, src=V, dst=P),
            x,
            axis=self.pg,
            src_type=V,
            dst_type=P,
        )

    def test_backward_reinterpret_r_to_p(self):
        x = self._generate_inputs((4,), self.pg, R)
        self.spmd_gradcheck(
            lambda x: reinterpret(x, self.pg, src=R, dst=P, expert_mode=True),
            x,
            axis=self.pg,
            src_type=R,
            dst_type=P,
        )

    # --- convert ---

    def test_backward_convert_r_to_v(self):
        x = self._generate_inputs((3, 4), self.pg, R)
        self.spmd_gradcheck(
            lambda x: convert(x, self.pg, src=R, dst=V),
            x,
            axis=self.pg,
            src_type=R,
            dst_type=V,
        )

    def test_backward_convert_r_to_s0(self):
        x = self._generate_inputs((6,), self.pg, R)
        self.spmd_gradcheck(
            lambda x: convert(x, self.pg, src=R, dst=S(0)),
            x,
            axis=self.pg,
            src_type=R,
            dst_type=S(0),
        )

    def test_backward_convert_i_to_v(self):
        x = self._generate_inputs((3, 4), self.pg, I)
        self.spmd_gradcheck(
            lambda x: convert(x, self.pg, src=I, dst=V),
            x,
            axis=self.pg,
            src_type=I,
            dst_type=V,
        )

    def test_backward_convert_i_to_s0(self):
        x = self._generate_inputs((6,), self.pg, I)
        self.spmd_gradcheck(
            lambda x: convert(x, self.pg, src=I, dst=S(0)),
            x,
            axis=self.pg,
            src_type=I,
            dst_type=S(0),
        )

    def test_backward_convert_r_to_p(self):
        x = self._generate_inputs((4,), self.pg, R)
        self.spmd_gradcheck(
            lambda x: convert(x, self.pg, src=R, dst=P),
            x,
            axis=self.pg,
            src_type=R,
            dst_type=P,
        )

    def test_backward_convert_i_to_p(self):
        x = self._generate_inputs((4,), self.pg, I)
        self.spmd_gradcheck(
            lambda x: convert(x, self.pg, src=I, dst=P, expert_mode=True),
            x,
            axis=self.pg,
            src_type=I,
            dst_type=P,
        )

    def test_backward_convert_v_to_p(self):
        x = self._generate_inputs((4,), self.pg, V)
        self.spmd_gradcheck(
            lambda x: convert(x, self.pg, src=V, dst=P, expert_mode=True),
            x,
            axis=self.pg,
            src_type=V,
            dst_type=P,
        )

    def test_backward_convert_s0_to_p(self):
        x = self._generate_inputs((2,), self.pg, S(0))
        self.spmd_gradcheck(
            lambda x: convert(x, self.pg, src=S(0), dst=P, expert_mode=True),
            x,
            axis=self.pg,
            src_type=S(0),
            dst_type=P,
        )

    # --- all_reduce ---

    def test_backward_all_reduce_p_to_r(self):
        x = self._generate_inputs((4,), self.pg, P)
        self.spmd_gradcheck(
            lambda x: all_reduce(x, self.pg, src=P, dst=R),
            x,
            axis=self.pg,
            src_type=P,
            dst_type=R,
        )

    def test_backward_all_reduce_p_to_i(self):
        x = self._generate_inputs((4,), self.pg, P)
        self.spmd_gradcheck(
            lambda x: all_reduce(x, self.pg, src=P, dst=I),
            x,
            axis=self.pg,
            src_type=P,
            dst_type=I,
        )

    # --- all_gather ---

    def test_backward_all_gather_v_to_r(self):
        x = self._generate_inputs((4,), self.pg, V)
        self.spmd_gradcheck(
            lambda x: all_gather(x, self.pg, src=V, dst=R),
            x,
            axis=self.pg,
            src_type=V,
            dst_type=R,
        )

    def test_backward_all_gather_s0_to_r(self):
        x = self._generate_inputs((2,), self.pg, S(0))
        self.spmd_gradcheck(
            lambda x: all_gather(x, self.pg, src=S(0), dst=R),
            x,
            axis=self.pg,
            src_type=S(0),
            dst_type=R,
        )

    def test_backward_all_gather_v_to_i(self):
        x = self._generate_inputs((4,), self.pg, V)
        self.spmd_gradcheck(
            lambda x: all_gather(x, self.pg, src=V, dst=I),
            x,
            axis=self.pg,
            src_type=V,
            dst_type=I,
        )

    def test_backward_all_gather_s0_to_i(self):
        x = self._generate_inputs((2,), self.pg, S(0))
        self.spmd_gradcheck(
            lambda x: all_gather(x, self.pg, src=S(0), dst=I),
            x,
            axis=self.pg,
            src_type=S(0),
            dst_type=I,
        )

    # --- reduce_scatter ---

    def test_backward_reduce_scatter_p_to_v(self):
        x = self._generate_inputs((3, 4), self.pg, P)
        self.spmd_gradcheck(
            lambda x: reduce_scatter(x, self.pg, src=P, dst=V),
            x,
            axis=self.pg,
            src_type=P,
            dst_type=V,
        )

    def test_backward_reduce_scatter_p_to_s0(self):
        x = self._generate_inputs((6,), self.pg, P)
        self.spmd_gradcheck(
            lambda x: reduce_scatter(x, self.pg, src=P, dst=S(0)),
            x,
            axis=self.pg,
            src_type=P,
            dst_type=S(0),
        )

    # --- all_to_all ---

    def test_backward_all_to_all_v_to_v(self):
        x = self._generate_inputs((3, 4), self.pg, V)
        self.spmd_gradcheck(
            lambda x: all_to_all(x, self.pg, src=V, dst=V),
            x,
            axis=self.pg,
            src_type=V,
            dst_type=V,
        )

    def test_backward_all_to_all_s0_to_s1(self):
        x = self._generate_inputs((3, 6), self.pg, S(0))
        self.spmd_gradcheck(
            lambda x: all_to_all(x, self.pg, src=S(0), dst=S(1)),
            x,
            axis=self.pg,
            src_type=S(0),
            dst_type=S(1),
        )


if __name__ == "__main__":
    unittest.main()
