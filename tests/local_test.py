# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for local (no-comms) operations: reinterpret, convert.

Covers: _local.py
"""

import unittest

import torch
from spmd_types import convert, I, P, R, reinterpret, S, V
from spmd_types._checker import assert_type, typecheck
from spmd_types._test_utils import LocalTensorTestCase
from spmd_types._type_attr import get_axis_local_type


class TestReinterpret(LocalTensorTestCase):
    """Test reinterpret operations (no-op forwards, possibly comms in backwards)."""

    def test_reinterpret_r_to_v(self):
        """reinterpret(R,V): R -> V, no-op forward. Tests forward and backward."""
        x = self._generate_inputs((4,), self.pg, R)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        with typecheck():
            result = reinterpret(x, self.pg, src=R, dst=V, expert_mode=True)

        # Forward is no-op, values unchanged
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_reinterpret_r_to_i(self):
        """reinterpret(R,I): R -> I, no-op forward. Tests forward and backward."""
        x = self._generate_inputs((4,), self.pg, R)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        with typecheck():
            result = reinterpret(x, self.pg, src=R, dst=I, expert_mode=True)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_reinterpret_r_to_p(self):
        """reinterpret(R,P): R -> P, no-op forward. Tests forward and backward."""
        x = self._generate_inputs((4,), self.pg, R)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        with typecheck():
            result = reinterpret(x, self.pg, src=R, dst=P, expert_mode=True)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_reinterpret_i_to_r(self):
        """reinterpret(I,R): I -> R, no-op forward. Tests forward and backward."""
        x = self._generate_inputs((4,), self.pg, I)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        with typecheck():
            result = reinterpret(x, self.pg, src=I, dst=R, expert_mode=True)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_reinterpret_v_to_p(self):
        """reinterpret(V,P): V -> P, no-op forward. Tests forward and backward."""
        x = self._generate_inputs((4,), self.pg, V)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        with typecheck():
            result = reinterpret(x, self.pg, src=V, dst=P)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_reinterpret_same_type_noop(self):
        """reinterpret with same src and dst is identity."""
        x = self._generate_inputs((4,), self.pg, R)
        result = reinterpret(x, self.pg, src=R, dst=R)
        # Should return same tensor
        self.assertIs(result, x)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_reinterpret_shard_rejected(self):
        """reinterpret rejects Shard types."""
        x = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(ValueError) as ctx:
            reinterpret(x, self.pg, src=R, dst=S(0))
        self.assertIn("does not support S(i)", str(ctx.exception))

    def test_reinterpret_unsupported_transition(self):
        """reinterpret rejects unsupported transitions."""
        x = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(ValueError) as ctx:
            reinterpret(x, self.pg, src=P, dst=R)
        self.assertIn("not supported", str(ctx.exception))

    def test_reinterpret_expert_mode_required(self):
        """reinterpret(R,V), reinterpret(R,I), reinterpret(R,P), reinterpret(I,R) require expert_mode."""
        x = self._generate_inputs((4,), self.pg, R)
        for dst in [V, I, P]:
            with self.assertRaises(ValueError) as ctx:
                reinterpret(x, self.pg, src=R, dst=dst)
            self.assertIn("expert_mode", str(ctx.exception))
        # I -> R also requires expert_mode
        x_i = self._generate_inputs((4,), self.pg, I)
        with self.assertRaises(ValueError) as ctx:
            reinterpret(x_i, self.pg, src=I, dst=R)
        self.assertIn("expert_mode", str(ctx.exception))


class TestConvert(LocalTensorTestCase):
    """Test convert operations (semantics-preserving type coercion)."""

    def test_convert_r_to_v(self):
        """convert(R,V): R -> V, slices to local portion. Tests forward and backward."""
        base = torch.arange(6, dtype=torch.float).reshape(self.WORLD_SIZE, 2)
        x = self.rank_map(lambda r: base.clone())
        assert_type(x, {self.pg: R})

        with typecheck():
            result = convert(x, self.pg, src=R, dst=V)

        # Each rank gets its chunk: rank 0 gets [0,1], rank 1 gets [2,3], rank 2 gets [4,5]
        for r in range(self.WORLD_SIZE):
            expected = base[r]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_convert_r_to_shard(self):
        """convert(R,S(i)): R -> S(i), slices to local portion."""
        base = torch.arange(6, dtype=torch.float)
        x = self.rank_map(lambda r: base.clone())
        assert_type(x, {self.pg: R})

        with typecheck():
            result = convert(x, self.pg, src=R, dst=S(0))

        for r in range(self.WORLD_SIZE):
            expected = base[r * 2 : (r + 1) * 2]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_convert_i_to_v(self):
        """convert(I,V): I -> V, slices to local portion. Tests forward and backward."""
        base = torch.arange(6, dtype=torch.float).reshape(self.WORLD_SIZE, 2)
        x = self.rank_map(lambda r: base.clone())
        assert_type(x, {self.pg: I})

        with typecheck():
            result = convert(x, self.pg, src=I, dst=V)

        for r in range(self.WORLD_SIZE):
            expected = base[r]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_convert_i_to_shard(self):
        """convert(I,S(i)): I -> S(i), slices to local portion."""
        base = torch.arange(6, dtype=torch.float)
        x = self.rank_map(lambda r: base.clone())
        assert_type(x, {self.pg: I})

        with typecheck():
            result = convert(x, self.pg, src=I, dst=S(0))

        for r in range(self.WORLD_SIZE):
            expected = base[r * 2 : (r + 1) * 2]
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_convert_r_to_p(self):
        """convert(R,P): R -> P, zeros out non-rank-0. Tests forward and backward."""
        base = torch.tensor([1.0, 2.0, 3.0])
        x = self.rank_map(lambda r: base.clone())
        assert_type(x, {self.pg: R})

        with typecheck():
            result = convert(x, self.pg, src=R, dst=P)

        # Rank 0 keeps values, others are zeroed
        torch.testing.assert_close(result._local_tensors[0], base)
        for r in range(1, self.WORLD_SIZE):
            torch.testing.assert_close(
                result._local_tensors[r],
                torch.zeros_like(base),
                msg=f"rank {r} should be zeros",
            )
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_convert_i_to_p(self):
        """convert(I,P): I -> P, zeros out non-rank-0. Tests forward and backward."""
        base = torch.tensor([1.0, 2.0, 3.0])
        x = self.rank_map(lambda r: base.clone())
        assert_type(x, {self.pg: I})

        with typecheck():
            result = convert(x, self.pg, src=I, dst=P, expert_mode=True)

        torch.testing.assert_close(result._local_tensors[0], base)
        for r in range(1, self.WORLD_SIZE):
            torch.testing.assert_close(
                result._local_tensors[r],
                torch.zeros_like(base),
                msg=f"rank {r} should be zeros",
            )
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_convert_v_to_p(self):
        """convert(V,P): V -> P, places data in disjoint positions. Tests forward and backward."""
        # Each rank has [r]
        x = self.rank_map(lambda r: torch.tensor(float(r)))
        assert_type(x, {self.pg: V})

        with typecheck():
            result = convert(x, self.pg, src=V, dst=P, expert_mode=True)

        # Each rank has a tensor of size world_size with its value at its position
        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(self.WORLD_SIZE)
            expected[r] = float(r)
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_convert_shard_to_p(self):
        """convert(S(i),P): S(i) -> P, places data in disjoint positions."""
        x = self.rank_map(lambda r: torch.tensor([float(r)]))
        assert_type(x, {self.pg: V})

        with typecheck():
            result = convert(x, self.pg, src=S(0), dst=P, expert_mode=True)

        for r in range(self.WORLD_SIZE):
            expected = torch.zeros(self.WORLD_SIZE)
            expected[r] = float(r)
            torch.testing.assert_close(
                result._local_tensors[r], expected, msg=f"rank {r}"
            )
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_convert_same_type_noop(self):
        """convert with same src and dst is identity."""
        x = self._generate_inputs((4,), self.pg, R)
        result = convert(x, self.pg, src=R, dst=R)
        self.assertIs(result, x)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_convert_r_to_i_same_as_reinterpret(self):
        """convert(R,I) should work like reinterpret(R,I)."""
        x = self._generate_inputs((4,), self.pg, R)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        with typecheck():
            result = convert(x, self.pg, src=R, dst=I, expert_mode=True)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_convert_i_to_r_same_as_reinterpret(self):
        """convert(I,R) should work like reinterpret(I,R)."""
        x = self._generate_inputs((4,), self.pg, I)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        with typecheck():
            result = convert(x, self.pg, src=I, dst=R)

        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_convert_from_partial_error(self):
        """convert cannot convert from partial."""
        x = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(ValueError) as ctx:
            convert(x, self.pg, src=P, dst=R)
        self.assertIn("not supported", str(ctx.exception))

    def test_convert_expert_mode_required(self):
        """convert(R,I), convert(I,P) and convert(V,P) require expert_mode."""
        # R -> I requires expert_mode
        x_r = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(ValueError) as ctx:
            convert(x_r, self.pg, src=R, dst=I)
        self.assertIn("expert_mode", str(ctx.exception))

        x_i = self._generate_inputs((4,), self.pg, I)
        with self.assertRaises(ValueError) as ctx:
            convert(x_i, self.pg, src=I, dst=P)
        self.assertIn("expert_mode", str(ctx.exception))

        x_v = self.rank_map(lambda r: torch.tensor(float(r)))
        assert_type(x_v, {self.pg: V})
        with self.assertRaises(ValueError) as ctx:
            convert(x_v, self.pg, src=V, dst=P)
        self.assertIn("expert_mode", str(ctx.exception))

        # S(i) -> P also requires expert_mode (S normalizes to V)
        x_s = self.rank_map(lambda r: torch.tensor([float(r)]))
        assert_type(x_s, {self.pg: V})
        with self.assertRaises(ValueError) as ctx:
            convert(x_s, self.pg, src=S(0), dst=P)
        self.assertIn("expert_mode", str(ctx.exception))


class TestReinterpretCompositions(LocalTensorTestCase):
    """Test compositional reinterpret operations: I->V and I->P."""

    def test_reinterpret_i_to_v(self):
        """reinterpret(I,V): I -> R -> V composition, no-op forward."""
        x = self._generate_inputs((4,), self.pg, I)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        with typecheck():
            result = reinterpret(x, self.pg, src=I, dst=V)

        # Forward is no-op (composition of two no-ops)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_reinterpret_i_to_p(self):
        """reinterpret(I,P): I -> R -> P composition, no-op forward."""
        x = self._generate_inputs((4,), self.pg, I)
        original = {r: x._local_tensors[r].clone() for r in range(self.WORLD_SIZE)}

        with typecheck():
            result = reinterpret(x, self.pg, src=I, dst=P)

        # Forward is no-op (composition of two no-ops)
        for r in range(self.WORLD_SIZE):
            torch.testing.assert_close(result._local_tensors[r], original[r])
        self.assertIs(get_axis_local_type(result, self.pg), P)


class TestConvertSubmesh(LocalTensorTestCase):
    """Test convert operations on submesh axes (group size < world size).

    Verifies that convert correctly maps global ranks to group-local ranks
    when using process groups from a multi-dimensional mesh. Previously,
    tensor_map passed global ranks directly to chunk/select, causing
    IndexError when global rank >= group size.
    """

    WORLD_SIZE = 4

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        cls.tp = mesh.get_group("tp")
        cls.dp = mesh.get_group("dp")

    def test_convert_r_to_shard_submesh(self):
        """convert(R, S(1)) on a submesh axis uses group-local rank for chunking."""
        x = self.rank_map(lambda r: torch.arange(32).float().reshape(4, 8))
        assert_type(x, {self.tp: R})

        result = convert(x, self.tp, src=R, dst=S(1))

        # tp groups are [0,1] and [2,3], each with world_size=2
        # Chunking dim=1 of shape [4,8] into 2 gives [4,4] per rank
        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape, (4, 4))
        # Ranks in the same tp group should get different chunks
        self.assertFalse(
            torch.equal(result._local_tensors[0], result._local_tensors[1])
        )
        # Ranks in different tp groups but same position should get same chunk
        torch.testing.assert_close(result._local_tensors[0], result._local_tensors[2])
        torch.testing.assert_close(result._local_tensors[1], result._local_tensors[3])

    def test_convert_r_to_shard_strided_submesh(self):
        """convert(R, S(0)) on a strided submesh (dp groups [0,2] and [1,3])."""
        x = self.rank_map(lambda r: torch.arange(8).float().reshape(4, 2))
        assert_type(x, {self.dp: R})

        result = convert(x, self.dp, src=R, dst=S(0))

        # dp groups are [0,2] and [1,3], each with world_size=2
        # Chunking dim=0 of shape [4,2] into 2 gives [2,2] per rank
        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape, (2, 2))
        # Ranks 0 and 2 are in the same dp group, different positions
        self.assertFalse(
            torch.equal(result._local_tensors[0], result._local_tensors[2])
        )
        # Ranks 0 and 1 are in different dp groups, same position -> same chunk
        torch.testing.assert_close(result._local_tensors[0], result._local_tensors[1])


if __name__ == "__main__":
    unittest.main()
