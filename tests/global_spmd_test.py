# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for global SPMD shard propagation: propagating S(i) through torch ops.

In global SPMD, the per-axis types are R, I, P, and S(i) (no V). S(i)
propagation reuses DTensor's sharding propagation to determine how shard
dimensions flow through ops, while local SPMD inference handles R/I/P.

Decision tree per mesh axis (when S is present):
1. Replace S->V, run local SPMD as compatibility check.
2. If local SPMD rejects -> propagate error.
3. If local SPMD accepts -> run DTensor shard propagation for S(i) output.

When no S is present: local SPMD only.

When DTensor says output is Partial but local SPMD (with S->V) says Varying,
the local type is upgraded from V to P.
"""

import unittest
import unittest.mock

import torch
import torch.distributed as dist
from spmd_types import (
    all_gather,
    all_reduce,
    all_to_all,
    convert,
    I,
    P,
    PartitionSpec,
    R,
    redistribute,
    reduce_scatter,
    S,
    V,
)
from spmd_types._checker import (
    _collect_shard_axes,
    _infer_global_output_type,
    _set_partition_spec,
    _validate_and_update_local_global_correspondence,
    assert_type,
    dtensor_placement_to_spmd_type,
    get_local_type,
    get_partition_spec,
    no_typecheck,
    typecheck,
)
from spmd_types._mesh_axis import _reset
from spmd_types._test_utils import FakeProcessGroupTestCase, LocalTensorTestCase
from spmd_types._type_attr import get_axis_local_type
from spmd_types.types import (
    DeviceMeshAxis,
    DTensorPropagationError,
    normalize_axis,
    partition_spec_to_shard_types,
    PerMeshAxisSpmdType,
    PerMeshAxisSpmdTypes,
    RedistributeError,
    Shard,
    SpmdTypeError,
    to_local_type,
)
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


def _type_name(t):
    """Short name for SPMD types, used in parametrized test names."""
    return t.name if hasattr(t, "name") else str(t)


def _get_global_type(tensor: torch.Tensor) -> PerMeshAxisSpmdTypes:
    """Reconstruct S(i) from local type + PartitionSpec (test helper)."""
    typ = get_local_type(tensor).copy()
    spec = get_partition_spec(tensor)
    if spec:
        for axis, shard in partition_spec_to_shard_types(spec).items():
            typ[normalize_axis(axis)] = shard
    return typ


def _get_axis_type(tensor: torch.Tensor, axis: DeviceMeshAxis) -> PerMeshAxisSpmdType:
    """Get the full SPMD type for a specific axis, including S(i)."""
    return _get_global_type(tensor).get(normalize_axis(axis), V)


def _to_global(lt, typ, axis):
    """Reconstruct global tensor given LocalTensor and its SPMD type.

    Uses only the ranks along the given axis (one representative group)
    rather than all WORLD_SIZE ranks.
    """
    ranks = list(axis.layout.all_ranks_from_zero())
    if isinstance(typ, Shard):
        return torch.cat([lt._local_tensors[r].detach() for r in ranks], dim=typ.dim)
    elif typ is R or typ is I:
        return lt._local_tensors[ranks[0]].detach().clone()
    elif typ is P:
        return sum(lt._local_tensors[r].detach() for r in ranks)
    elif typ is V:
        return None
    raise ValueError(f"Cannot reconstruct global tensor from {typ}")


class GlobalSpmdTestCase(LocalTensorTestCase):
    """Base test class for global SPMD tests."""

    WORLD_SIZE = 27
    AXIS_SIZE = 3

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        n = cls.AXIS_SIZE
        cls.mesh = init_device_mesh("cpu", (n, n, n), mesh_dim_names=("ep", "dp", "tp"))
        cls.tp_pg = cls.mesh.get_group("tp")
        cls.tp = normalize_axis(cls.tp_pg)
        cls.dp = normalize_axis(cls.mesh.get_group("dp"))
        cls.ep = normalize_axis(cls.mesh.get_group("ep"))

    def setUp(self):
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()
        self._tc_cm = typecheck(
            local=False,
        )
        self._tc_cm.__enter__()

    def tearDown(self):
        self._tc_cm.__exit__(None, None, None)
        self.mode.__exit__(None, None, None)

    def rank_map(self, cb):
        """Create a LocalTensor, pausing type checking during the callback."""
        with no_typecheck():
            return self.mode.rank_map(cb)

    def _make_input(self, shape, axis, typ):
        if typ is R or typ is I:
            base = torch.randn(shape)
            result = self.rank_map(lambda r: base.clone())
        else:
            result = self.rank_map(lambda r: torch.randn(shape) + r)
        # Annotate all global axes: the target axis gets the requested type,
        # other global axes default to R (replicated).
        types = {self.tp: R, self.dp: R}
        types[axis] = typ
        assert_type(result, types)
        return result

    def _make_multi_axis_input(self, shape, types):
        has_shard_or_varying = any(
            isinstance(t, Shard) or t is V or t is P for t in types.values()
        )
        if not has_shard_or_varying:
            base = torch.randn(shape)
            result = self.rank_map(lambda r: base.clone())
        else:
            result = self.rank_map(lambda r: torch.randn(shape) + r)
        assert_type(result, types)
        return result


# =============================================================================
# S(i) storage
# =============================================================================


class TestShardTypeStorage(GlobalSpmdTestCase):
    """Test that S(i) can be stored on and read from tensors."""

    def test_assert_type_stores_shard(self):
        """assert_type with S(i) stores PartitionSpec on tensor."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        self.assertEqual(_get_axis_type(x, self.tp), S(0))
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.tp, None))

    def test_assert_type_stores_shard_with_other_axes(self):
        """S(i) on one axis, R on another."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: S(0), self.tp: R})
        self.assertEqual(_get_axis_type(x, self.dp), S(0))
        self.assertEqual(_get_axis_type(x, self.tp), R)

    def test_assert_type_shard_check_matches(self):
        """Re-asserting the same S(i) should succeed."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        assert_type(x, {self.tp: S(0)})

    def test_assert_type_shard_check_mismatch_raises(self):
        """Re-asserting conflicting S(i) should raise."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: S(1)})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: R})

    def test_check_v_then_r(self):
        """Asserting V then R on same axis."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: V})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: R})

    def test_check_r_then_v(self):
        """Asserting R then V on same axis."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: R})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: V})

    def test_get_local_type_returns_v_for_sharded(self):
        """get_local_type returns V for sharded axes."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        local = get_local_type(x)
        self.assertIs(local[self.tp], V)
        # get_global_type reconstructs S from PartitionSpec.
        self.assertEqual(_get_global_type(x)[self.tp], S(0))

    def test_get_local_type_preserves_non_shard_types(self):
        """get_local_type returns R/I/P normally."""
        x = self._generate_inputs((4, 3), self.pg, R)
        self.assertIs(get_local_type(x)[normalize_axis(self.pg)], R)

    def test_get_global_type_reconstructs_s(self):
        """get_global_type reconstructs S from PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        gtype = _get_global_type(x)
        self.assertEqual(gtype[self.tp], S(0))
        spec = get_partition_spec(x)
        self.assertEqual(spec, PartitionSpec(self.tp, None))

    def test_negative_dim_shard(self):
        """S(-1) resolves to S(ndim-1)."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(-1)})
        self.assertEqual(_get_axis_type(x, self.tp), S(1))
        self.assertEqual(get_partition_spec(x), PartitionSpec(None, self.tp))

    def test_recheck_v_then_s(self):
        """Asserting V then S(i) on same axis refines with PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: V})
        assert_type(x, {self.tp: S(0)})
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.tp, None))

    def test_recheck_s_then_v(self):
        """Asserting S(i) then V keeps existing PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        assert_type(x, {self.tp: V})  # V is less specific, doesn't overwrite
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.tp, None))

    def test_conflicting_s_rejected(self):
        """Same axis, different dim -> SpmdTypeError."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: S(1)})

    def test_multi_axis_same_dim_with_partition_spec(self):
        """Multi-axis on same dim requires explicit PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(
            x,
            {self.dp: V, self.tp: V},
            PartitionSpec((self.dp, self.tp), None),
        )
        spec = get_partition_spec(x)
        self.assertEqual(spec, PartitionSpec((self.dp, self.tp), None))

    def test_merge_different_axes_different_dims(self):
        """S(i) on tp then S(j) on dp merges into combined PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        assert_type(x, {self.dp: S(1)})
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.tp, self.dp))

    def test_recheck_spec_refine_none_to_sharded(self):
        """PartitionSpec(dp, None) then PartitionSpec(dp, tp) refines dim 1."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: V}, PartitionSpec(self.dp, None))
        assert_type(
            x,
            {self.dp: V, self.tp: V},
            PartitionSpec(self.dp, self.tp),
        )
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.dp, self.tp))

    def test_recheck_spec_single_to_multi_axis_rejected(self):
        """PartitionSpec(dp) then PartitionSpec((dp, tp)) conflicts."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: V}, PartitionSpec(self.dp, None))
        with self.assertRaises(SpmdTypeError):
            assert_type(
                x,
                {self.dp: V, self.tp: V},
                PartitionSpec((self.dp, self.tp), None),
            )

    def test_multi_axis_same_dim_without_spec_rejected(self):
        """{tp: S(0)} then {dp: S(0)}: error without explicit PartitionSpec."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: S(0)})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.dp: S(0)})

    def test_recheck_s_then_consistent_spec(self):
        """{dp: S(0)} then PartitionSpec(dp, None): OK (consistent)."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: S(0)})
        assert_type(x, {self.dp: V}, PartitionSpec(self.dp, None))
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.dp, None))

    def test_recheck_s_then_spec_adds_new_axis(self):
        """{dp: S(0)} then PartitionSpec(dp, tp): OK, adds tp on dim 1."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: S(0)})
        assert_type(
            x,
            {self.dp: V, self.tp: V},
            PartitionSpec(self.dp, self.tp),
        )
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.dp, self.tp))

    def test_partition_spec_ordering_matters(self):
        """PartitionSpec ordering matters: (tp, dp) != (dp, tp)."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(
            x,
            {self.tp: V, self.dp: V},
            PartitionSpec((self.tp, self.dp), None),
        )
        y = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(
            y,
            {self.dp: V, self.tp: V},
            PartitionSpec((self.dp, self.tp), None),
        )
        self.assertNotEqual(get_partition_spec(x), get_partition_spec(y))

    def test_recheck_s_to_multi_axis_rejected(self):
        """S(0) on dp then PartitionSpec((dp, tp)) on same dim conflicts."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.dp: S(0)})
        with self.assertRaises(SpmdTypeError):
            assert_type(
                x,
                {self.dp: V, self.tp: V},
                PartitionSpec((self.dp, self.tp), None),
            )

    def test_recheck_multi_axis_to_s_rejected(self):
        """PartitionSpec((dp, tp)) then S(0) on dp conflicts."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(
            x,
            {self.dp: V, self.tp: V},
            PartitionSpec((self.dp, self.tp), None),
        )
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.dp: S(0)})

    def test_explicit_v_is_stored(self):
        """assert_type with explicit V stores V in the local type dict."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.ep: V, self.tp: R})
        local = get_local_type(x)
        self.assertIs(local[self.ep], V)
        self.assertIs(local[self.tp], R)


# =============================================================================
# S(i) propagation through pointwise ops
# =============================================================================


class TestGlobalSpmdPointwise(GlobalSpmdTestCase):
    """Test S(i) propagation through pointwise/elementwise ops."""

    @parametrize(
        "op",
        [
            torch.neg,
            torch.abs,
            torch.exp,
            torch.tanh,
            torch.sigmoid,
            torch.relu,
        ],
        name_fn=lambda op: op.__name__,
    )
    def test_unary_s0(self, op):
        x = self._make_input((4, 3), self.tp, S(0))
        self.assertEqual(_get_axis_type(op(x), self.tp), S(0))

    @parametrize(
        "op",
        [torch.sqrt, torch.rsqrt, torch.reciprocal],
        name_fn=lambda op: op.__name__,
    )
    def test_positive_unary_s0(self, op):
        """Ops requiring positive input: apply abs+1 first."""
        x = self._make_input((4, 3), self.tp, S(0))
        self.assertEqual(_get_axis_type(op(torch.abs(x) + 1), self.tp), S(0))

    def test_clone_s0(self):
        x = self._make_input((4, 3), self.tp, S(0))
        self.assertEqual(_get_axis_type(x.clone(), self.tp), S(0))

    @parametrize(
        "op", [torch.add, torch.sub, torch.mul], name_fn=lambda op: op.__name__
    )
    def test_binary_s0_s0(self, op):
        x = self._make_input((4, 3), self.tp, S(0))
        y = self._make_input((4, 3), self.tp, S(0))
        self.assertEqual(_get_axis_type(op(x, y), self.tp), S(0))

    @parametrize(
        "op", [torch.add, torch.mul, torch.div], name_fn=lambda op: op.__name__
    )
    def test_s0_r_rejected(self, op):
        x = self._make_input((4, 3), self.tp, S(0))
        y = self._make_input((4, 3), self.tp, R)
        with self.assertRaises(SpmdTypeError):
            op(x, y)

    def test_r_s0_rejected(self):
        """R + S(0) rejected: order doesn't matter."""
        x = self._make_input((4, 3), self.tp, R)
        y = self._make_input((4, 3), self.tp, S(0))
        with self.assertRaises(SpmdTypeError):
            x + y

    def test_s0_r_fix_with_redistribute(self):
        """S(0) + R rejected, but redistribute R -> S(0) fixes it."""
        x = self._make_input((4, 3), self.tp, S(0))
        # R tensor has global shape (12, 3) to match S(0) global shape
        y = self._make_input((12, 3), self.tp, R)
        # Local shapes mismatch: (4, 3) + (12, 3) raises RuntimeError
        with self.assertRaises(RuntimeError):
            x + y
        y_s = redistribute(y, self.tp_pg, src=R, dst=S(0))
        result = x + y_s
        self.assertEqual(_get_axis_type(result, self.tp), S(0))

    @parametrize(
        "op,scalar",
        [
            (torch.add, 1),
            (torch.mul, 2.0),
            (torch.sub, 0.5),
            (torch.div, 3.0),
        ],
        name_fn=lambda op, scalar: op.__name__,
    )
    def test_scalar(self, op, scalar):
        x = self._make_input((4, 3), self.tp, S(0))
        self.assertEqual(_get_axis_type(op(x, scalar), self.tp), S(0))


instantiate_parametrized_tests(TestGlobalSpmdPointwise)


# =============================================================================
# S(i) propagation through matmul ops
# =============================================================================


class TestGlobalSpmdMatmul(GlobalSpmdTestCase):
    """Test S(i) propagation through matmul ops."""

    @parametrize(
        "op,x_shape,w_shape,x_type,w_type,expected",
        [
            (torch.mm, (4, 3), (3, 5), S(0), R, S(0)),
            (torch.mm, (4, 3), (3, 5), R, S(1), S(1)),
            (torch.mm, (4, 3), (3, 5), R, R, R),
            (torch.matmul, (4, 3), (3, 5), S(0), R, S(0)),
            (torch.bmm, (2, 6, 3), (2, 3, 4), S(1), R, S(1)),
            (torch.bmm, (2, 4, 3), (2, 3, 6), R, S(2), S(2)),
        ],
        name_fn=lambda op, x_shape, w_shape, x_type, w_type, expected: (
            f"{op.__name__}_{_type_name(x_type)}_{_type_name(w_type)}"
        ),
    )
    def test_mm(self, op, x_shape, w_shape, x_type, w_type, expected):
        x = self._make_input(x_shape, self.tp, x_type)
        w = self._make_input(w_shape, self.tp, w_type)
        self.assertEqual(_get_axis_type(op(x, w), self.tp), expected)

    def test_matmul_operator_s0_r(self):
        """x @ w with S(0)@R -> S(0) via __matmul__."""
        x = self._make_input((4, 3), self.tp, S(0))
        w = self._make_input((3, 5), self.tp, R)
        self.assertEqual(_get_axis_type(x @ w, self.tp), S(0))

    def test_matmul_batched_3d_decomposition(self):
        """torch.matmul with 3D inputs decomposes to aten.bmm.

        This is a regression test: the old name-based _resolve_aten_op would
        map torch.matmul to aten.mm, but 3D x 3D dispatches to aten.bmm.
        The dispatch-based approach handles this naturally.
        """
        x = self._make_input((2, 4, 3), self.tp, S(1))
        w = self._make_input((2, 3, 6), self.tp, R)
        out = torch.matmul(x, w)
        self.assertEqual(_get_axis_type(out, self.tp), S(1))

    def test_matmul_batched_3d_output_sharded(self):
        """torch.matmul 3D: shard on output dim via weight."""
        x = self._make_input((2, 4, 3), self.tp, R)
        w = self._make_input((2, 3, 6), self.tp, S(2))
        out = torch.matmul(x, w)
        self.assertEqual(_get_axis_type(out, self.tp), S(2))


instantiate_parametrized_tests(TestGlobalSpmdMatmul)


# =============================================================================
# Incompatible type combinations rejected in global SPMD
# =============================================================================


class TestGlobalSpmdRejection(GlobalSpmdTestCase):
    """Test cases that should be rejected in global SPMD."""

    @parametrize(
        "op,x_shape,w_shape,x_type,w_type",
        [
            # S + P rejection
            (torch.add, (4, 3), (4, 3), S(0), P),
            (torch.mul, (4, 3), (4, 3), S(0), P),
            (torch.mm, (4, 3), (3, 5), S(0), P),
            # Incompatible shard dims
            (torch.add, (4, 3), (4, 3), S(0), S(1)),
            # mm: weight K dim sharded, no feasible strategy
            (torch.mm, (4, 3), (3, 5), R, S(0)),
        ],
        name_fn=lambda op, x_shape, w_shape, x_type, w_type: (
            f"{op.__name__}_{_type_name(x_type)}_{_type_name(w_type)}"
        ),
    )
    def test_rejected(self, op, x_shape, w_shape, x_type, w_type):
        x = self._make_input(x_shape, self.tp, x_type)
        w = self._make_input(w_shape, self.tp, w_type)
        with self.assertRaises(SpmdTypeError):
            op(x, w)

    def test_add_s_p_has_fix_suggestion(self):
        """S+P rejection goes through local SPMD and gets fix suggestions."""
        x = self._make_input((4, 3), self.tp, S(0))
        y = self._make_input((4, 3), self.tp, P)
        with self.assertRaises(SpmdTypeError) as ctx:
            x + y
        self.assertIn("all_reduce", str(ctx.exception))


instantiate_parametrized_tests(TestGlobalSpmdRejection)


# =============================================================================
# V -> P promotion
# =============================================================================


class TestGlobalSpmdPartialPromotion(GlobalSpmdTestCase):
    """Test that V is upgraded to P when global shard propagation detects Partial."""

    def test_mm_contracted_sharded_dim(self):
        """mm(x@S(1), w@S(0)) -> P: contracted K dim is sharded."""
        x = self._make_input((4, 3), self.tp, S(1))
        w = self._make_input((3, 5), self.tp, S(0))
        result = torch.mm(x, w)
        self.assertIs(get_axis_local_type(result, self.tp), P)
        self.assertIsNone(get_partition_spec(result))

    # TODO(zpcore): Check if this is the right behavior.
    def test_sum_sharded_dim(self):
        """sum(x@S(0)) -> P: reducing over a sharded dim."""
        x = self._make_input((6, 4), self.tp, S(0))
        result = torch.sum(x)
        self.assertIs(get_axis_local_type(result, self.tp), P)


# =============================================================================
# I fallback to local SPMD inference
# =============================================================================


class TestGlobalSpmdInvariant(GlobalSpmdTestCase):
    """Test I type: I+I passes via local SPMD, I mixed with S/R/P is rejected."""

    @parametrize(
        "op,x_shape,w_shape",
        [
            (torch.add, (4, 3), (4, 3)),
            (torch.mul, (4, 3), (4, 3)),
            (torch.mm, (4, 3), (3, 5)),
        ],
        name_fn=lambda op, x_shape, w_shape: op.__name__,
    )
    def test_i_i_ok(self, op, x_shape, w_shape):
        x = self._make_input(x_shape, self.tp, I)
        w = self._make_input(w_shape, self.tp, I)
        self.assertEqual(_get_axis_type(op(x, w), self.tp), I)

    @parametrize(
        "op,x_shape,w_shape,x_type,w_type",
        [
            (torch.add, (4, 3), (4, 3), I, R),
            (torch.mul, (4, 3), (4, 3), I, R),
            (torch.add, (4, 3), (4, 3), I, P),
            (torch.mm, (4, 3), (3, 5), I, R),
            (torch.mm, (4, 3), (3, 5), S(0), I),
            (torch.mm, (4, 6), (6, 5), I, S(1)),
            (torch.add, (4, 3), (4, 3), S(0), I),
        ],
        name_fn=lambda op, x_shape, w_shape, x_type, w_type: (
            f"{op.__name__}_{_type_name(x_type)}_{_type_name(w_type)}"
        ),
    )
    def test_i_mixed_rejected(self, op, x_shape, w_shape, x_type, w_type):
        x = self._make_input(x_shape, self.tp, x_type)
        w = self._make_input(w_shape, self.tp, w_type)
        with self.assertRaises(SpmdTypeError):
            op(x, w)


instantiate_parametrized_tests(TestGlobalSpmdInvariant)


# =============================================================================
# Ops accepting list of tensors and multi-output ops
# =============================================================================


class TestGlobalSpmdListAndMultiOutput(GlobalSpmdTestCase):
    """Test ops that accept tensor lists (cat, stack) and multi-output ops (sort)."""

    @parametrize(
        "op,shape,typ,expected",
        [
            (torch.cat, (4, 6), S(1), S(1)),
            (torch.cat, (4, 3), R, R),
            (torch.cat, (4, 3), P, P),
            (torch.cat, (4, 3), I, I),
        ],
        name_fn=lambda op, shape, typ, expected: f"{op.__name__}_{_type_name(typ)}",
    )
    def test_cat(self, op, shape, typ, expected):
        a = self._make_input(shape, self.tp, typ)
        b = self._make_input(shape, self.tp, typ)
        self.assertEqual(_get_axis_type(torch.cat([a, b], dim=0), self.tp), expected)

    @parametrize(
        "typ,expected",
        [
            (S(0), S(1)),
            (R, R),
            (P, P),
        ],
        name_fn=lambda typ, expected: _type_name(typ),
    )
    def test_stack(self, typ, expected):
        shape = (4, 6) if isinstance(typ, Shard) else (4, 3)
        a = self._make_input(shape, self.tp, typ)
        b = self._make_input(shape, self.tp, typ)
        self.assertEqual(_get_axis_type(torch.stack([a, b], dim=0), self.tp), expected)

    @parametrize(
        "typ,shape",
        [
            (R, (4, 3)),
            (S(0), (4, 6)),  # global shape (12, 6)
        ],
        name_fn=lambda typ, shape: _type_name(typ),
    )
    def test_sort(self, typ, shape):
        x = self._make_input(shape, self.tp, typ)
        values, indices = torch.sort(x)
        self.assertEqual(_get_axis_type(values, self.tp), typ)
        self.assertEqual(_get_axis_type(indices, self.tp), typ)

    def test_sort_shard_rejected(self):
        x = self._make_input((4, 6), self.tp, S(0))
        with self.assertRaises(SpmdTypeError):
            torch.sort(x, dim=0)

    def test_sort_p_rejected(self):
        x = self._make_input((4, 3), self.tp, P)
        with self.assertRaises(SpmdTypeError):
            torch.sort(x)


instantiate_parametrized_tests(TestGlobalSpmdListAndMultiOutput)


# =============================================================================
# Multiple independent mesh axes
# =============================================================================


class TestGlobalSpmdMultiAxis(GlobalSpmdTestCase):
    """Test S(i) propagation with multiple independent mesh axes."""

    def test_mm_shard_one_axis_replicate_other(self):
        """S(0)@dp, R@tp propagates independently per axis."""
        x = self._make_multi_axis_input((4, 3), {self.dp: S(0), self.tp: R})
        w = self._make_multi_axis_input((3, 5), {self.dp: R, self.tp: R})
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, self.dp), S(0))
        self.assertEqual(_get_axis_type(result, self.tp), R)

    def test_mm_shard_both_axes(self):
        """S(0)@dp, S(0)@tp: both axes shard dim 0, propagate independently.

        Output PartitionSpec preserves input tuple ordering: ('dp', 'tp')
        in the input produces ('dp', 'tp') in the output.
        """
        x = self.rank_map(lambda r: torch.randn(4, 3) + r)
        assert_type(
            x, {self.dp: V, self.tp: V}, PartitionSpec((self.dp, self.tp), None)
        )
        w = self._make_multi_axis_input((3, 5), {self.dp: R, self.tp: R})
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, self.dp), S(0))
        self.assertEqual(_get_axis_type(result, self.tp), S(0))
        spec = get_partition_spec(result)
        self.assertEqual(spec, PartitionSpec((self.dp, self.tp), None))

    def test_mm_different_shard_dims_on_different_axes(self):
        """S(0)@dp on input, S(1)@tp on weight: independent axes, independent shards."""
        x = self._make_multi_axis_input((4, 3), {self.dp: S(0), self.tp: R})
        w = self._make_multi_axis_input((3, 5), {self.dp: R, self.tp: S(1)})
        result = torch.mm(x, w)
        self.assertEqual(_get_axis_type(result, self.dp), S(0))
        self.assertEqual(_get_axis_type(result, self.tp), S(1))

    def test_ordering_conflict_errors(self):
        """('tp', 'dp') on one input and ('dp', 'tp') on another is an error."""
        x = self.rank_map(lambda r: torch.randn(4, 3) + r)
        assert_type(
            x, {self.dp: V, self.tp: V}, PartitionSpec((self.tp, self.dp), None)
        )
        w = self.rank_map(lambda r: torch.randn(4, 3) + r)
        assert_type(
            w, {self.dp: V, self.tp: V}, PartitionSpec((self.dp, self.tp), None)
        )
        with self.assertRaises(SpmdTypeError):
            x + w

    def test_ordering_same_order_different_dims(self):
        """Same tuple ordering on different dims across inputs is OK.

        x has ('tp', 'dp') on dim 0.
        w has ('tp', 'dp') on dim 0 (same ordering, same dim).
        No conflict -- ordering is consistent.
        """
        x = self.rank_map(lambda r: torch.randn(4, 3) + r)
        assert_type(
            x, {self.dp: V, self.tp: V}, PartitionSpec((self.tp, self.dp), None)
        )
        w = self.rank_map(lambda r: torch.randn(4, 3) + r)
        assert_type(
            w, {self.dp: V, self.tp: V}, PartitionSpec((self.tp, self.dp), None)
        )
        result = x + w
        self.assertEqual(_get_axis_type(result, self.tp), S(0))
        self.assertEqual(_get_axis_type(result, self.dp), S(0))
        spec = get_partition_spec(result)
        self.assertEqual(spec, PartitionSpec((self.tp, self.dp), None))


# =============================================================================
# Redistribute with S(i) types
# =============================================================================


class TestGlobalSpmdRedistribute(GlobalSpmdTestCase):
    """Test redistribute with S(i) types and its role in enabling ops."""

    def test_redistribute_enables_mm(self):
        """Redistribute weight from S(0) (K sharded) to R, then mm works.

        Without redistribute, mm(R, S(0)) errors because K is sharded.
        After gathering weight to R, mm(R, R) -> R succeeds.
        """
        x = self._make_input((4, 3), self.tp, R)
        # S(0) per-rank (1, 5) -> global (3, 5), matching x's K=3
        w = self._make_input((1, 5), self.tp, S(0))

        # Direct mm fails: local shapes (4,3) @ (1,5) mismatch
        with self.assertRaises(RuntimeError):
            torch.mm(x, w)

        # Redistribute weight: S(0) -> R (all_gather)
        w_r = redistribute(w, self.tp_pg, src=S(0), dst=R)
        self.assertEqual(_get_axis_type(w_r, self.tp), R)

        # Now mm works: R @ R -> R
        result = torch.mm(x, w_r)
        self.assertEqual(_get_axis_type(result, self.tp), R)

    def test_redistribute_enables_mm_reshard(self):
        """Redistribute weight from S(0) (K sharded) to S(1) (N sharded).

        mm(R, S(1)) is valid: output N dim is sharded -> S(1).
        """
        x = self._make_input((4, 3), self.tp, R)
        # S(0) per-rank (1, 6) -> global (3, 6). After S(0)->S(1): per-rank (3, 2).
        w = self._make_input((1, 6), self.tp, S(0))

        # Redistribute weight: S(0) -> S(1) (all-to-all, K-shard to N-shard)
        w_s1 = redistribute(w, self.tp_pg, src=S(0), dst=S(1))
        self.assertEqual(_get_axis_type(w_s1, self.tp), S(1))

        # Now mm works: R @ S(1) -> S(1)
        result = torch.mm(x, w_s1)
        self.assertEqual(_get_axis_type(result, self.tp), S(1))

    def test_redistribute_rejects_non_innermost_axis(self):
        """redistribute rejects axes that are not innermost in PartitionSpec.

        For PartitionSpec(('tp', 'dp'), None), only 'dp' (the innermost) can
        be redistributed.  'tp' must be rejected until 'dp' is peeled off.
        """
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(
            x, {self.dp: V, self.tp: V}, PartitionSpec((self.tp, self.dp), None)
        )
        spec = get_partition_spec(x)
        self.assertEqual(spec, PartitionSpec((self.tp, self.dp), None))

        # Outer axis 'tp' is rejected (tp is a real PG, so the test executes
        # the function which runs the innermost check before the collective).
        with self.assertRaises(RedistributeError):
            redistribute(x, self.tp_pg, src=S(0), dst=R)

    def test_redistribute_allows_single_axis_entries(self):
        """Single-axis PartitionSpec entries are always allowed."""
        x = self._make_input((4, 3), self.tp, S(0))
        spec = get_partition_spec(x)
        self.assertEqual(spec, PartitionSpec(self.tp, None))
        # Single axis -- no ordering constraint.
        y = redistribute(x, self.tp_pg, src=S(0), dst=R)
        self.assertEqual(_get_axis_type(y, self.tp), R)


class TestGlobalSpmdDirectCollectives(GlobalSpmdTestCase):
    """Test direct collective calls on global axes."""

    def test_all_gather_s0_to_r(self):
        """all_gather(S(0), R) on a global axis updates PartitionSpec."""
        x = self._make_input((4, 3), self.tp, S(0))
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.tp, None))
        y = all_gather(x, self.tp_pg, src=S(0), dst=R)
        self.assertEqual(_get_axis_type(y, self.tp), R)
        self.assertEqual(get_partition_spec(y), PartitionSpec(None, None))

    def test_all_gather_s0_to_i(self):
        """all_gather(S(0), I) on a global axis."""
        x = self._make_input((4, 3), self.tp, S(0))
        y = all_gather(x, self.tp_pg, src=S(0), dst=I)
        self.assertEqual(_get_axis_type(y, self.tp), I)
        self.assertEqual(get_partition_spec(y), PartitionSpec(None, None))

    def test_reduce_scatter_p_to_s0(self):
        """reduce_scatter(P, S(0)) on a global axis."""
        x = self._make_input((6, 3), self.tp, R)
        x_p = convert(x, self.tp_pg, src=R, dst=P)
        y = reduce_scatter(x_p, self.tp_pg, src=P, dst=S(0))
        self.assertEqual(_get_axis_type(y, self.tp), S(0))
        self.assertEqual(get_partition_spec(y), PartitionSpec(self.tp, None))

    def test_all_reduce_p_to_r(self):
        """all_reduce(P, R) on a global axis clears shard from spec."""
        x = self._make_input((4, 3), self.tp, R)
        x_p = convert(x, self.tp_pg, src=R, dst=P)
        y = all_reduce(x_p, self.tp_pg, src=P, dst=R)
        self.assertEqual(_get_axis_type(y, self.tp), R)
        self.assertEqual(get_partition_spec(y), PartitionSpec(None, None))

    def test_all_to_all_s0_to_s1(self):
        """all_to_all(S(0), S(1)) on a global axis updates shard dim."""
        x = self._make_input((6, 3), self.tp, S(0))
        y = all_to_all(x, self.tp_pg, src=S(0), dst=S(1))
        self.assertEqual(_get_axis_type(y, self.tp), S(1))
        self.assertEqual(get_partition_spec(y), PartitionSpec(None, self.tp))

    def test_redistribute_p_to_r_numerics(self):
        """redistribute(P, R) via all_reduce preserves global value."""
        global_x = torch.randn(4, 3)
        x = self.rank_map(lambda r: global_x.clone())
        assert_type(x, {self.tp: R, self.dp: R})
        x_p = convert(x, self.tp_pg, src=R, dst=P)
        global_before = _to_global(x_p, P, self.tp)
        y = redistribute(x_p, self.tp_pg, src=P, dst=R)
        global_after = _to_global(y, R, self.tp)
        torch.testing.assert_close(global_after, global_before)

    def test_redistribute_p_to_s0_numerics(self):
        """redistribute(P, S(0)) via reduce_scatter preserves global value."""
        n = self.AXIS_SIZE
        global_x = torch.randn(n * 2, 3)
        x = self.rank_map(lambda r: global_x.clone())
        assert_type(x, {self.tp: R, self.dp: R})
        x_p = convert(x, self.tp_pg, src=R, dst=P)
        global_before = _to_global(x_p, P, self.tp)
        y = redistribute(x_p, self.tp_pg, src=P, dst=S(0))
        global_after = _to_global(y, S(0), self.tp)
        torch.testing.assert_close(global_after, global_before)

    # -- Error tests ----------------------------------------------------------

    def test_all_gather_mismatched_src_shard_dim(self):
        """all_gather with src=S(1) but tensor is S(0) should error."""
        x = self._make_input((4, 3), self.tp, S(0))
        self.assertEqual(get_partition_spec(x), PartitionSpec(self.tp, None))
        with self.assertRaisesRegex(
            SpmdTypeError,
            r"expected input shard S\(1\).*got S\(0\)",
        ):
            all_gather(x, self.tp_pg, src=S(1), dst=R)

    def test_all_to_all_mismatched_src_shard_dim(self):
        """all_to_all with src=S(1) but tensor is S(0) should error."""
        x = self._make_input((6, 3), self.tp, S(0))
        with self.assertRaisesRegex(
            SpmdTypeError,
            r"expected input shard S\(1\).*got S\(0\)",
        ):
            all_to_all(x, self.tp_pg, src=S(1), dst=S(0))


# =============================================================================
# Local/global SPMD transition
# =============================================================================


class TestLocalGlobalTransition(GlobalSpmdTestCase):
    """Test transitions between local and global SPMD."""

    def test_global_to_local_pointwise(self):
        """S(0) -> local V pointwise -> re-stamp S(0)."""
        x = self._make_input((4, 3), self.tp, S(0))
        self.assertEqual(_get_axis_type(x, self.tp), S(0))
        self.assertIsNotNone(get_partition_spec(x))

        with typecheck(local=True):
            # In local mode, V propagates through pointwise ops.
            y = x + 1
            self.assertIs(_get_axis_type(y, self.tp), V)
            self.assertIsNone(get_partition_spec(y))

            y = torch.relu(y)
            self.assertIs(_get_axis_type(y, self.tp), V)
            self.assertIsNone(get_partition_spec(y))

        # After exiting local: y has V with no PartitionSpec.
        self.assertIs(_get_axis_type(y, self.tp), V)
        self.assertIsNone(get_partition_spec(y))

        # Re-stamp S(0) to re-enter global mode.
        assert_type(y, {self.tp: S(0)})
        self.assertEqual(_get_axis_type(y, self.tp), S(0))
        self.assertIsNotNone(get_partition_spec(y))

        # Back in global mode: S(0) propagates through ops.
        z = y * 2
        self.assertEqual(_get_axis_type(z, self.tp), S(0))

    def test_local_to_global_matmul(self):
        """Local V matmul -> stamp S(0) -> global S(0) propagation.

        Starts in local mode: inputs have V, matmul produces V.
        Then transitions to global by stamping S(0) via assert_type,
        and verifies S(0) propagates through subsequent ops.
        """
        with typecheck(local=True):
            h = self.rank_map(lambda r: torch.randn(4, 8))
            w = self.rank_map(lambda r: torch.randn(8, 5))
            assert_type(h, {self.tp: V})
            assert_type(w, {self.tp: R})

            # Local matmul: V @ R -> V (no shard tracking).
            y = torch.mm(h, w)
            self.assertIs(_get_axis_type(y, self.tp), V)
            self.assertIsNone(get_partition_spec(y))

        # After exiting local: y has V with no PartitionSpec.
        self.assertIs(_get_axis_type(y, self.tp), V)
        self.assertIsNone(get_partition_spec(y))

        # Transition to global: stamp S(0).
        assert_type(y, {self.tp: S(0), self.dp: R})
        self.assertEqual(_get_axis_type(y, self.tp), S(0))
        self.assertEqual(get_partition_spec(y), PartitionSpec(self.tp, None))

        # In global mode: S(0) propagates through ops.
        z = y + 1
        self.assertEqual(_get_axis_type(z, self.tp), S(0))

    def test_local_to_global_rejects_v_without_partition_spec(self):
        """V without PartitionSpec is rejected when entering global mode.

        After local ops produce V tensors, using them in global mode
        without first stamping S(i) raises SpmdTypeError.
        """
        with typecheck(local=True):
            x = self.rank_map(lambda r: torch.randn(4, 3))
            assert_type(x, {self.tp: V, self.dp: R})

            y = x + 1
            self.assertIs(_get_axis_type(y, self.tp), V)
            self.assertIsNone(get_partition_spec(y))

        # Back in global mode: using y (V without PartitionSpec) raises.
        with self.assertRaises(SpmdTypeError):
            y + 1


instantiate_parametrized_tests(TestGlobalSpmdDirectCollectives)


# =============================================================================
# Backward ground truth tests
# =============================================================================


class TestGlobalSpmdBackwardGroundTruth(GlobalSpmdTestCase):
    """Verify backward correctness by comparing against non-distributed ground truth.

    Reconstructs global (non-distributed) tensors from per-rank SPMD data,
    runs the same computation on them, and compares gradients. Unlike the
    adjoint identity check, this works for ALL ops including nonlinear.
    """

    def _gradient_type(self, typ):
        if isinstance(typ, Shard):
            return typ
        return {R: P, I: I, V: V, P: R}[typ]

    def _make_random_typed(self, like_lt, typ, axis):
        with no_typecheck():
            if isinstance(typ, Shard) or typ is V or typ is P:
                result = self.mode.rank_map(
                    lambda r: torch.randn_like(like_lt._local_tensors[r])
                )
            else:
                t = torch.randn_like(like_lt._local_tensors[0])
                result = self.mode.rank_map(lambda r: t.clone())
        assert_type(result, {axis: typ})
        return result

    def _gradcheck(self, fn, inputs, input_types, axis):
        """Compare SPMD gradients against global ground truth."""
        for inp in inputs:
            inp.requires_grad_(True)

        # SPMD forward + backward
        y = fn(*inputs)
        output_type = _get_global_type(y).get(normalize_axis(axis), V)
        grad_type = self._gradient_type(output_type)
        g = self._make_random_typed(y, grad_type, axis)
        y.backward(g)

        # Ground truth: reconstruct global tensors, run the same computation
        # without SPMD type annotations, and compare gradients.
        # The global tensors are non-distributed (no ranks, no sharding),
        # so collectives like redistribute are meaningless — we stub them
        # as identity.
        def _noop(x, *a, **kw):
            return x

        _mod = type(self).__module__
        with (
            no_typecheck(),
            unittest.mock.patch(f"{_mod}.redistribute", side_effect=_noop),
        ):
            global_inputs = [
                _to_global(inp, typ, axis) for inp, typ in zip(inputs, input_types)
            ]
            global_g = _to_global(g, grad_type, axis)

            for gi in global_inputs:
                gi.requires_grad_(True)

            global_y = fn(*global_inputs)
            global_y.backward(global_g)

            # Compare gradients: reconstruct SPMD grad using the gradient type
            for inp, gi, typ in zip(inputs, global_inputs, input_types):
                if typ is V:
                    continue
                grad_typ = self._gradient_type(typ)
                spmd_grad_global = _to_global(inp.grad, grad_typ, axis)
                torch.testing.assert_close(
                    spmd_grad_global, gi.grad, atol=1e-4, rtol=1e-4
                )

    @parametrize(
        "op,shapes,types",
        [
            (torch.mm, [(6, 3), (3, 5)], [S(0), R]),
            (torch.mm, [(4, 3), (3, 6)], [R, S(1)]),
            (torch.add, [(6, 3), (6, 3)], [S(0), S(0)]),
            (torch.mul, [(6, 3), (6, 3)], [S(0), S(0)]),
            (torch.neg, [(6, 3)], [S(0)]),
            (torch.relu, [(6, 3)], [S(0)]),
        ],
        name_fn=lambda op, shapes, types: (
            f"{op.__name__}_{'_'.join(t.name if hasattr(t, 'name') else str(t) for t in types)}"
        ),
    )
    def test_op(self, op, shapes, types):
        inputs = [self._make_input(s, self.tp, t) for s, t in zip(shapes, types)]
        self._gradcheck(op, inputs, types, self.tp)

    def test_mm_relu_mm_chain(self):
        """mm -> relu -> mm: nonlinear in the middle."""
        x = self._make_input((6, 3), self.tp, S(0))
        w1 = self._make_input((3, 5), self.tp, R)
        w2 = self._make_input((5, 2), self.tp, R)

        def fn(x, w1, w2):
            h = torch.mm(x, w1)
            h = torch.relu(h)
            return torch.mm(h, w2)

        self._gradcheck(fn, [x, w1, w2], [S(0), R, R], self.tp)

    def test_ddp_pattern(self):
        """DDP: data-parallel with replicated weights.

        x: S(0) (batch-sharded); w1, w2: R (replicated)
        Forward: relu(mm(mm(x, w1), w2)) -> S(0)
        Backward: x.grad: S(0); w1.grad, w2.grad: P (needs all-reduce in real DDP)
        """
        x = self._make_input((6, 3), self.dp, S(0))
        w1 = self._make_input((3, 5), self.dp, R)
        w2 = self._make_input((5, 7), self.dp, R)

        def fn(x, w1, w2):
            y1 = x @ w1
            y2 = y1 @ w2
            return torch.relu(y2)

        self._gradcheck(fn, [x, w1, w2], [S(0), R, R], self.dp)

    def test_megatron_tp(self):
        """Megatron TP end-to-end: column-parallel mm, relu, then row-parallel mm.

        The full Megatron pattern with redistribute for the "Megatron trick":

          x: I
          redistribute(I, R)   -- forward: no-op; backward: all-reduce (P → I)
          R @ S(1) → S(1)     -- column parallel
          relu: S(1) → S(1)
          S(1) @ S(0) → P     -- row parallel (contracted sharded dim)
          redistribute(P, I)   -- forward: all-reduce; backward: no-op (I → R)
        """
        x = self._make_input((4, 3), self.tp, I)
        w_col = self._make_input((3, 8), self.tp, S(1))
        w_row = self._make_input((8, 5), self.tp, S(0))

        tp = self.tp_pg

        def fn(x, w_col, w_row):
            # The Megatron trick: I → R before column parallel
            # - Forward: no-op (data already identical across ranks)
            # - Backward: inserts all-reduce (R's grad is P, P → I)
            x = redistribute(x, tp, src=I, dst=R)

            # Column parallel
            h = x @ w_col  # R @ S(1) → S(1)
            h = torch.relu(h)  # S(1) → S(1)

            # Row parallel
            out = h @ w_row  # S(1) @ S(0) → P

            # The Megatron trick: P → I after row parallel
            # - Forward: inserts all-reduce
            # - Backward: no-op (I → R)
            out = redistribute(out, tp, src=P, dst=I)
            return out

        # Verify forward types
        y = fn(x, w_col, w_row)
        assert_type(y, {self.tp: I})

        self._gradcheck(fn, [x, w_col, w_row], [I, S(1), S(0)], self.tp)


instantiate_parametrized_tests(TestGlobalSpmdBackwardGroundTruth)


# =============================================================================
# _collect_shard_axes
# =============================================================================


class TestCollectShardAxes(unittest.TestCase):
    """Test _collect_shard_axes extracts and orders shard axes from PartitionSpecs."""

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=64, store=store)
        cls.mesh = init_device_mesh("cpu", (4, 4, 4), mesh_dim_names=("ep", "dp", "tp"))

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def setUp(self):
        self.tp = normalize_axis(self.mesh.get_group("tp"))
        self.dp = normalize_axis(self.mesh.get_group("dp"))
        self.ep = normalize_axis(self.mesh.get_group("ep"))

    def test_no_partition_spec_returns_empty(self):
        """Tensor without PartitionSpec contributes no axes."""
        result, edges = _collect_shard_axes([None], {self.tp})
        self.assertEqual(result, [])
        self.assertEqual(edges, {})

    def test_multi_axis_tuple_ordering(self):
        """Tuple entry (tp, dp) yields tp before dp."""
        specs = [
            PartitionSpec(None, (self.tp, self.dp), None),
            PartitionSpec((self.tp, self.dp), None, None),
            PartitionSpec((self.dp, self.ep), None, None),
        ]
        result, edges = _collect_shard_axes(specs, {self.tp, self.dp, self.ep})
        self.assertEqual(result, [self.tp, self.dp, self.ep])
        # Verify edges: tp->dp, dp->ep, tp->ep
        self.assertIn(self.dp, edges[self.tp])
        self.assertIn(self.ep, edges[self.dp])

    def test_ordering_conflict_raises(self):
        """Conflicting orderings across inputs raise SpmdTypeError."""
        specs = [
            PartitionSpec((self.tp, self.dp), None, None),
            PartitionSpec((self.dp, self.tp), None, None),
        ]
        with self.assertRaises(SpmdTypeError):
            _collect_shard_axes(specs, {self.tp, self.dp})


# =============================================================================
# DTensor shard propagator and _infer_global_output_type
# =============================================================================


class TestInferGlobalOutputType(unittest.TestCase):
    """Test _infer_global_output_type with plain tensors.

    Uses plain tensors with SPMD type annotations (no LocalTensorMode)
    since propagation only needs tensor shapes, dtypes, and type metadata.
    """

    WORLD_SIZE = 3

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=cls.WORLD_SIZE, store=store
        )
        cls.mesh = init_device_mesh("cpu", (cls.WORLD_SIZE,), mesh_dim_names=("tp",))

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def setUp(self):
        self.tp = normalize_axis(self.mesh.get_group("tp"))

    def _make_typed(self, shape, typ):
        """Create a plain tensor with SPMD type and optional PartitionSpec."""
        x = torch.randn(shape)
        local_typ = V if isinstance(typ, S) else typ
        with typecheck(strict_mode="permissive"):
            assert_type(x, {self.tp: local_typ})
        if isinstance(typ, S):
            spec = [self.tp if d == typ.dim else None for d in range(len(shape))]
            _set_partition_spec(x, PartitionSpec(*spec))
        return x

    def test_add_s0_s0(self):
        """add(S(0), S(0)) -> output PartitionSpec has S(0)."""
        tp = self.tp
        x = self._make_typed((8, 3), S(0))
        y = self._make_typed((8, 3), S(0))
        result = torch.add(x, y)
        pls, specs = _infer_global_output_type(
            func=torch.add,
            args=(x, y),
            kwargs=None,
            global_shard_axes=[tp],
            flat_results=[result],
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0], PartitionSpec(tp, None))
        global_results = [to_local_type(dtensor_placement_to_spmd_type(pls[0][tp]))]
        _validate_and_update_local_global_correspondence(V, global_results, tp)

    def test_add_s0_s0_out_mismatch(self):
        """add(S(0), S(0), out=out) -> should fail due to output PartitionSpec
        mismatch."""
        tp = self.tp
        x = self._make_typed((8, 3), S(0))
        y = self._make_typed((8, 3), S(0))
        out = self._make_typed((8, 3), S(1))
        result = torch.add(x, y)
        with self.assertRaises(SpmdTypeError):
            _infer_global_output_type(
                func=torch.add,
                args=(x, y),
                kwargs={"out": out},
                global_shard_axes=[tp],
                flat_results=[result],
            )

    def test_add_s0_s1(self):
        """add(S(0), S(1)) -> should fail for DTensor propagation due to
        redistribution needed."""
        tp = self.tp
        x = self._make_typed((8, 3), S(0))
        y = self._make_typed((8, 3), S(1))
        result = torch.add(x, y)
        with self.assertRaises(SpmdTypeError):
            _infer_global_output_type(
                func=torch.add,
                args=(x, y),
                kwargs=None,
                global_shard_axes=[tp],
                flat_results=[result],
            )

    def test_mm_s0_r(self):
        """matmul(S(0), R) -> output PartitionSpec has S(0)."""
        tp = self.tp
        x = self._make_typed((8, 3), S(0))
        w = self._make_typed((3, 5), R)
        result = torch.mm(x, w)
        pls, specs = _infer_global_output_type(
            func=torch.mm,
            args=(x, w),
            kwargs=None,
            global_shard_axes=[tp],
            flat_results=[result],
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0], PartitionSpec(tp, None))
        global_results = [to_local_type(dtensor_placement_to_spmd_type(pls[0][tp]))]
        _validate_and_update_local_global_correspondence(V, global_results, tp)

    def test_mm_r_s1(self):
        """matmul(R, S(1)) -> output PartitionSpec has S(1)."""
        tp = self.tp
        x = self._make_typed((8, 3), R)
        w = self._make_typed((3, 5), S(1))
        result = torch.mm(x, w)
        pls, specs = _infer_global_output_type(
            func=torch.mm,
            args=(x, w),
            kwargs=None,
            global_shard_axes=[tp],
            flat_results=[result],
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0], PartitionSpec(None, tp))
        global_results = [to_local_type(dtensor_placement_to_spmd_type(pls[0][tp]))]
        _validate_and_update_local_global_correspondence(V, global_results, tp)

    def test_mm_s1_s0_gives_partial(self):
        """matmul(S(1), S(0)) -> Partial: accepted by correspondence check."""
        tp = self.tp
        x = self._make_typed((8, 3), S(1))
        w = self._make_typed((3, 5), S(0))
        result = torch.mm(x, w)
        pls, specs = _infer_global_output_type(
            func=torch.mm,
            args=(x, w),
            kwargs=None,
            global_shard_axes=[tp],
            flat_results=[result],
        )
        # DTensor produces Partial; local SPMD produces V.
        global_results = [to_local_type(dtensor_placement_to_spmd_type(pls[0][tp]))]
        out = _validate_and_update_local_global_correspondence(V, global_results, tp)
        self.assertEqual(out, [P])

    def test_transpose_s0_gives_s1(self):
        """transpose(S(0), 0, 1) -> output PartitionSpec has S(1)."""
        tp = self.tp
        x = self._make_typed((8, 3), S(0))
        result = torch.transpose(x, 0, 1)
        pls, specs = _infer_global_output_type(
            func=torch.transpose,
            args=(x, 0, 1),
            kwargs=None,
            global_shard_axes=[tp],
            flat_results=[result],
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0], PartitionSpec(None, tp))
        global_results = [to_local_type(dtensor_placement_to_spmd_type(pls[0][tp]))]
        _validate_and_update_local_global_correspondence(V, global_results, tp)


class TestShapePropagation(FakeProcessGroupTestCase):
    """Test _ShardPropagator propagates shard axes through reshape/view ops."""

    def setUp(self) -> None:
        self.tp = normalize_axis(self.mesh.get_group("tp"))
        self._tc_cm = typecheck(local=False)
        self._tc_cm.__enter__()

    def tearDown(self) -> None:
        self._tc_cm.__exit__(None, None, None)

    def _make_typed(
        self, shape: tuple[int, ...], typ: PerMeshAxisSpmdType
    ) -> torch.Tensor:
        """Create a plain tensor with SPMD type and optional PartitionSpec."""
        x = torch.randn(shape)
        local_typ = V if isinstance(typ, Shard) else typ
        with typecheck(strict_mode="permissive"):
            assert_type(x, {self.tp: local_typ})
        if isinstance(typ, Shard):
            spec = [self.tp if d == typ.dim else None for d in range(len(shape))]
            _set_partition_spec(x, PartitionSpec(*spec))
        return x

    def _shard_global(
        self, global_tensor: torch.Tensor, shard_dim: int
    ) -> list[torch.Tensor]:
        """Slice a global tensor into per-rank shards using ceiling division."""
        global_dim_size = global_tensor.shape[shard_dim]
        chunk = (global_dim_size + self.WORLD_SIZE - 1) // self.WORLD_SIZE
        shards = []
        for r in range(self.WORLD_SIZE):
            start = min(chunk * r, global_dim_size)
            end = min(chunk * (r + 1), global_dim_size)
            shards.append(global_tensor.narrow(shard_dim, start, end - start).clone())
        return shards

    @parametrize(
        "input_shape,shard_dim,target_shape,expected",
        [
            # Forward: shard dim passes through unchanged.
            ((6, 4), 0, (6, 4), S(0)),
            ((6, 4), 1, (6, 4), S(1)),
            # Unsqueeze: insert size-1 dim, shard dim shifts.
            ((6, 4), 0, (6, 1, 4), S(0)),
            ((6, 4), 1, (6, 1, 4), S(2)),
            ((6, 4), 0, (1, 6, 4), S(1)),
            # Squeeze: remove size-1 dim.
            ((6, 1, 4), 0, (6, 4), S(0)),
            ((6, 1, 4), 2, (6, 4), S(1)),
            # Flatten: non-shard dims merge, shard dim untouched.
            ((2, 3, 4), 0, (2, 12), S(0)),
            ((2, 3, 4), 2, (6, 4), S(1)),
            # Singleton shard dim (size 1) with matching size-1 output dim:
            # view_groups maps InputDim(0) to output dim 0 (both size 1), so
            # scaling works normally.
            ((1, 6), 0, (1, 2, 3), S(0)),
            # Target shape contains -1 (inferred dim).
            ((6, 4), 0, (-1, 4), S(0)),
            ((6, 4), 1, (6, -1), S(1)),
            ((6, 4), 0, (6, 1, -1), S(0)),
        ],
        name_fn=lambda input_shape, shard_dim, target_shape, expected: (
            f"{'x'.join(map(str, input_shape))}_S{shard_dim}"
            f"_to_{'x'.join(map(str, target_shape))}"
        ),
    )
    def test_reshape_op(
        self,
        input_shape: tuple[int, ...],
        shard_dim: int,
        target_shape: tuple[int, ...],
        expected: PerMeshAxisSpmdType,
    ) -> None:
        """Reshape preserves shard dim through view_groups dim mapping."""
        x = self._make_typed(input_shape, S(shard_dim))
        y = x.reshape(*target_shape)
        assert_type(y, {self.tp: expected})

    @parametrize(
        "global_shape,shard_dim,target_dims,expected",
        [
            # global_shape is the full (unsharded) tensor shape.  With
            # WORLD_SIZE=3, dim 0 of size 8 yields local sizes [3, 3, 2].
            # _scale_size_args multiplies the local shard dim by mesh_size
            # (e.g. 3*3=9), which differs from the true global (8), but
            # DTensor propagation still resolves the correct output shard dim.
            #
            # None entries in target_dims are replaced with the local shard
            # dim size for each rank.
            #
            # Identity reshape (same shape).
            ((8, 4), 0, (None, 4), S(0)),
            ((6, 8), 1, (6, None), S(1)),
            # Unsqueeze: insert size-1 dim, shard dim shifts.
            ((8, 4), 0, (None, 1, 4), S(0)),
            ((8, 4), 0, (1, None, 4), S(1)),
            ((6, 8), 1, (6, 1, None), S(2)),
            # Squeeze: remove size-1 dim.
            ((8, 1, 4), 0, (None, 4), S(0)),
            ((6, 1, 8), 2, (6, None), S(1)),
            # Flatten: non-shard dims merge, shard dim untouched.
            ((8, 3, 4), 0, (None, 12), S(0)),
            ((2, 3, 8), 2, (6, None), S(1)),
            # Target shape contains -1 (inferred dim).
            ((8, 4), 0, (-1, 4), S(0)),
            ((6, 8), 1, (6, -1), S(1)),
        ],
        name_fn=lambda global_shape, shard_dim, target_dims, expected: (
            f"{'x'.join(map(str, global_shape))}_S{shard_dim}"
            f"_to_{'x'.join('N' if d is None else str(d) for d in target_dims)}"
        ),
    )
    def test_uneven_reshape_op(
        self,
        global_shape: tuple[int, ...],
        shard_dim: int,
        target_dims: tuple,
        expected: PerMeshAxisSpmdType,
    ) -> None:
        """Reshape with uneven sharding on the shard dim.

        Slices a global tensor into per-rank shards using ceiling division
        (matching DTensor's default), then tests each rank's local tensor
        independently.  None entries in target_dims are replaced with the
        rank's local shard dim size.
        """
        global_tensor = torch.randn(*global_shape)
        for shard in self._shard_global(global_tensor, shard_dim):
            local_shard_size = shard.shape[shard_dim]
            target_shape = tuple(
                local_shard_size if d is None else d for d in target_dims
            )
            x = self._make_typed(shard.shape, S(shard_dim))
            y = x.reshape(*target_shape)
            assert_type(y, {self.tp: expected})

    def test_reshape_split_error_6x8_S1_to_6x2x4(self) -> None:
        """Shard dim split across multiple output dims (dim 1)."""
        x = self._make_typed((6, 8), S(1))
        with self.assertRaisesRegex(
            DTensorPropagationError,
            r"Reshaping a tensor sharded on dim 1 into multiple smaller dimensions",
        ):
            x.reshape(6, 2, 4)

    def test_reshape_split_error_12x4_S0_to_3xNx4(self) -> None:
        """Shard dim split with -1 inferred dim."""
        x = self._make_typed((12, 4), S(0))
        with self.assertRaisesRegex(
            DTensorPropagationError,
            r"Reshaping a tensor sharded on dim 0 into multiple smaller dimensions",
        ):
            x.reshape(3, -1, 4)

    def test_reshape_singleton_drop_error_1x6_S0_to_6(self) -> None:
        """Shard dim has local size 1, view_groups drops it."""
        x = self._make_typed((1, 6), S(0))
        with self.assertRaisesRegex(
            DTensorPropagationError,
            r"Reshaping a tensor sharded on dim 0 \(local size 1\) is ambiguous",
        ):
            x.reshape(6)

    def test_reshape_singleton_drop_error_1x6_S0_to_2x3(self) -> None:
        """Shard dim has local size 1, view_groups drops it (multi-dim target)."""
        x = self._make_typed((1, 6), S(0))
        with self.assertRaisesRegex(
            DTensorPropagationError,
            r"Reshaping a tensor sharded on dim 0 \(local size 1\) is ambiguous",
        ):
            x.reshape(2, 3)


instantiate_parametrized_tests(TestShapePropagation)


# =============================================================================
# Cross-mesh PartitionSpec remapping
# =============================================================================


class TestCrossMeshSpecRemapping(unittest.TestCase):
    """Test that auto-reinterpret remaps PartitionSpec when switching meshes."""

    WORLD_SIZE = 16

    @classmethod
    def setUpClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=cls.WORLD_SIZE, store=store
        )
        mesh = init_device_mesh("cpu", (2, 2, 4), mesh_dim_names=("dp", "cp", "tp"))
        cls.dp = normalize_axis(mesh.get_group("dp"))
        cls.cp = normalize_axis(mesh.get_group("cp"))
        cls.tp = normalize_axis(mesh.get_group("tp"))
        dp_cp_mesh = mesh["dp", "cp"]._flatten("dp_cp")
        cls.dp_cp = normalize_axis(dp_cp_mesh.get_group("dp_cp"))
        cls.fine_mesh = frozenset({cls.dp, cls.cp, cls.tp})

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def test_shared_axis_spec_preserved(self):
        """S(0)@tp on {dp_cp, tp} mesh auto-reinterprets to {dp, cp, tp}, spec preserved."""
        from spmd_types import set_current_mesh

        with typecheck(local=False, strict_mode="strict"):
            x = torch.randn(4, 3)
            assert_type(x, {self.dp_cp: R, self.tp: S(0)})
            with set_current_mesh(self.fine_mesh):
                result = torch.add(x, x)
                self.assertIs(get_axis_local_type(result, self.dp), R)
                self.assertIs(get_axis_local_type(result, self.cp), R)
                self.assertIs(get_axis_local_type(result, self.tp), V)
                self.assertEqual(
                    get_partition_spec(result), PartitionSpec(self.tp, None)
                )

    def test_reversed_multi_axis_spec_passed(self):
        """PartitionSpec((dp, cp), None) remaps to PartitionSpec(dp_cp, None)."""
        from spmd_types import set_current_mesh

        with typecheck(local=False, strict_mode="strict"):
            x = torch.randn(4, 3)
            assert_type(
                x,
                {self.dp: V, self.cp: V, self.tp: R},
                PartitionSpec((self.dp, self.cp), None),
            )
            with set_current_mesh(frozenset({self.dp_cp, self.tp})):
                result = torch.add(x, x)
                self.assertEqual(
                    get_partition_spec(result), PartitionSpec(self.dp_cp, None)
                )

    def test_reversed_multi_axis_spec_rejected(self):
        """PartitionSpec((cp, dp), None) rejected: wrong axis order for dp_cp."""
        from spmd_types import set_current_mesh

        with typecheck(local=False, strict_mode="strict"):
            x = torch.randn(4, 3)
            assert_type(
                x,
                {self.dp: V, self.cp: V, self.tp: R},
                PartitionSpec((self.cp, self.dp), None),
            )
            with set_current_mesh(frozenset({self.dp_cp, self.tp})):
                with self.assertRaises(SpmdTypeError):
                    torch.add(x, x)

    def test_multi_segment_spec_remapped(self):
        """PartitionSpec((tp, dp, cp),) segments into tp + (dp,cp)->dp_cp."""
        from spmd_types import set_current_mesh

        with typecheck(local=False, strict_mode="strict"):
            x = torch.randn(4, 3)
            assert_type(
                x,
                {self.dp: V, self.cp: V, self.tp: V},
                PartitionSpec((self.tp, self.dp, self.cp), None),
            )
            with set_current_mesh(frozenset({self.dp_cp, self.tp})):
                result = torch.add(x, x)
                self.assertEqual(
                    get_partition_spec(result),
                    PartitionSpec((self.tp, self.dp_cp), None),
                )

    def test_separate_axes_spec_remapped_rejected(self):
        """PartitionSpec(dp, cp) remaps individual axes to dp_cp."""
        from spmd_types import set_current_mesh

        with typecheck(local=False, strict_mode="strict"):
            x = torch.randn(4, 3)
            assert_type(
                x, {self.dp: V, self.cp: V, self.tp: R}, PartitionSpec(self.dp, self.cp)
            )
            with set_current_mesh(frozenset({self.dp_cp, self.tp})):
                with self.assertRaises(SpmdTypeError):
                    result = torch.add(x, x)


if __name__ == "__main__":
    run_tests()
