# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for composable spmd_typecheck rules (spmd_types.rules).

Covers: rules.einsum (the einsum sharding rule, `_`, `...`, multiple outputs),
the type-only collectives and local transitions, rules.output, the coverage
checks the runner applies to hooks that use rules, and the einsum equation
parser.  ``TestUseCases`` inlines kernels from torchtitan and Megatron-style
pretraining code written as composed rules.
"""

import unittest

import expecttest
import torch
from spmd_types import (
    assert_type,
    get_partition_spec,
    I,
    P,
    PartitionSpec,
    R,
    rules,
    S,
    set_current_mesh,
    Shard,
    SpmdTypeError,
    V,
)
from spmd_types._checker import no_typecheck, typecheck
from spmd_types._test_utils import LocalTensorTestCase
from spmd_types._type_attr import get_axis_local_type, get_local_type
from spmd_types.types import (
    normalize_axis,
    normalize_partition_spec,
    partition_spec_get_shard,
)


class _MeshTestCase(LocalTensorTestCase, expecttest.TestCase):
    """2x2 named mesh (dp, tp) with an active typecheck() and current mesh."""

    MESH_SHAPE = (2, 2)
    MESH_DIM_NAMES = ("dp", "tp")
    LOCAL = False

    def setUp(self):
        super().setUp()
        self.enterContext(typecheck(local=self.LOCAL))
        self.enterContext(set_current_mesh(self.mesh))
        self.tp_group = self.mesh.get_group("tp")  # for Functions that take a pg

    def rank_map(self, cb):
        with no_typecheck():
            return self.mode.rank_map(cb)

    def typed(self, shape, **axes):
        """Random tensor annotated with per-axis types (by name), no spec."""
        if all(typ is R or typ is I for typ in axes.values()):
            base = torch.randn(shape)
            t = self.rank_map(lambda r: base.clone())
        else:
            t = self.rank_map(lambda r: torch.randn(shape))
        assert_type(t, dict(axes))
        return t

    def sharded(self, shape, spec, **local):
        """Random tensor with an explicit PartitionSpec plus local types."""
        t = self.rank_map(lambda r: torch.randn(shape))
        assert_type(t, local, spec)
        return t

    def dims(self, ndim, spec=None, **local):
        d = rules.NdimWithSpmdType(ndim)
        assert_type(d, local, spec)
        return d

    def assertTyped(self, value, local, spec):
        self.assertEqual(
            dict(get_local_type(value)),
            {normalize_axis(a): t for a, t in local.items()},
        )
        self.assertEqual(
            get_partition_spec(value),
            None if spec is None else normalize_partition_spec(spec),
        )


class GlobalSigTestCase(_MeshTestCase):
    LOCAL = False


class LocalSigTestCase(_MeshTestCase):
    LOCAL = True


# =============================================================================
# rules.einsum
# =============================================================================


class TestEinsum(GlobalSigTestCase):
    def test_batch_and_free_dims_keep_sharding(self):
        x = self.sharded((4, 6), PartitionSpec("dp", None), tp=R)
        w = self.sharded((5, 6), PartitionSpec("tp", None), dp=R)
        y = rules.einsum("mk,nk->mn", x, w)
        self.assertTyped(y, {"dp": V, "tp": V}, PartitionSpec("dp", "tp"))
        self.assertEqual(y.ndim, 2)

    def test_contraction_over_sharded_dim_is_partial(self):
        x = self.sharded((4, 6), PartitionSpec("dp", "tp"))
        w = self.sharded((5, 6), PartitionSpec(None, "tp"), dp=R)
        y = rules.einsum("mk,nk->mn", x, w)
        self.assertTyped(y, {"dp": V, "tp": P}, PartitionSpec("dp", None))
        # linear_in is about Partial operands; it neither enables nor changes
        # the Partial that a sharded contraction produces.
        y = rules.einsum("mk,nk->mn", x, w, linear_in=(0, 1))
        self.assertTyped(y, {"dp": V, "tp": P}, PartitionSpec("dp", None))

    def test_scalar_output_contracts_every_sharded_dim(self):
        x = self.sharded((4, 3), PartitionSpec("dp", "tp"))
        y = self.sharded((4, 3), PartitionSpec("dp", "tp"))
        dot = rules.einsum("tv,tv->", x, y)
        self.assertTyped(dot, {"dp": P, "tp": P}, None)
        self.assertEqual(dot.ndim, 0)

    def test_named_contraction_is_a_sum_and_underscore_is_opaque(self):
        """A named contracted label claims the op is additive along it (so
        (x ** 2).sum(0) is "bi->i" outright); a reduction that is not
        (max, mean, logsumexp) is written "_" and needs the dim complete."""
        x = self.sharded((4, 6), PartitionSpec("dp", None), tp=R)
        self.assertTyped(rules.einsum("bi->i", x), {"dp": P, "tp": R}, None)
        with self.assertRaises(SpmdTypeError) as cm:
            rules.einsum("_i->i", x)
        self.assertExpectedInline(
            str(cm.exception),
            """rules.einsum: operand 0 of '_i->i' is sharded on (mesh_dp) at dim 0, which the equation marks '_' (must not be sharded)""",
        )
        full = self.sharded((4, 6), PartitionSpec(None, "tp"), dp=R)
        self.assertTyped(
            rules.einsum("_i->i", full),
            {"dp": R, "tp": V},
            PartitionSpec("tp"),
        )

    def test_broadcast_operand_must_be_replicated(self):
        x = self.sharded((4, 6), PartitionSpec("dp", None), tp=R)
        bias_ok = self.typed((6,), dp=R, tp=R)
        rules.einsum("td,d->td", x, bias_ok)
        bias_bad = self.typed((6,), dp=V, tp=R)
        with self.assertRaises(SpmdTypeError) as cm:
            rules.einsum("td,d->td", x, bias_bad)
        self.assertExpectedInline(
            str(cm.exception),
            """rules.einsum: operand 1 of 'td,d->td' broadcasts over 't', which is sharded on mesh_dp, so it must be replicated on mesh_dp, but it is Varying""",
        )

    def test_labels_sharded_inconsistently(self):
        x = self.sharded((4, 6), PartitionSpec("dp", "tp"))
        y = self.sharded((4, 6), PartitionSpec("tp", "dp"))
        with self.assertRaises(SpmdTypeError) as cm:
            rules.einsum("td,td->td", x, y)
        self.assertExpectedInline(
            str(cm.exception),
            """rules.einsum: label 't' of 'td,td->td' is sharded inconsistently: (mesh_dp) vs (mesh_tp) (operand 1, dim 0)""",
        )

    def test_axis_may_shard_only_one_label(self):
        x = self.sharded((4, 6), PartitionSpec("tp", None), dp=R)
        w = self.sharded((5, 6), PartitionSpec("tp", None), dp=R)
        with self.assertRaises(SpmdTypeError) as cm:
            rules.einsum("mk,nk->mn", x, w)
        self.assertExpectedInline(
            str(cm.exception),
            """rules.einsum: axis mesh_tp shards both 'm' and 'n' in 'mk,nk->mn'; an axis may shard only one dim""",
        )

    def test_opaque_dim_must_be_unsharded(self):
        x = self.sharded((4, 6), PartitionSpec("dp", None), tp=I)
        w = self.typed((6,), dp=R, tp=I)
        y = rules.einsum("t_,_->t_", x, w)
        self.assertTyped(y, {"dp": V, "tp": I}, PartitionSpec("dp", None))
        with self.assertRaises(SpmdTypeError) as cm:
            rules.einsum("t_,_->t_", self.sharded((4, 6), PartitionSpec("dp", "tp")), w)
        self.assertExpectedInline(
            str(cm.exception),
            """rules.einsum: operand 0 of 't_,_->t_' is sharded on (mesh_tp) at dim 1, which the equation marks '_' (must not be sharded)""",
        )

    def test_ellipsis(self):
        spec = PartitionSpec("dp", None, "tp")
        g = self.sharded((2, 3, 4), spec)
        u = self.sharded((2, 3, 4), spec)
        y = rules.einsum("...d,...d->...d", g, u)
        self.assertTyped(y, {"dp": V, "tp": V}, spec)
        self.assertEqual(y.ndim, 3)
        w = self.sharded((4,), PartitionSpec("tp"), dp=R)
        self.assertTyped(rules.einsum("...d,d->...d", g, w), {"dp": V, "tp": V}, spec)
        with self.assertRaises(SpmdTypeError) as cm:
            rules.einsum(
                "...d,...d->...d",
                g,
                self.sharded((3, 4), PartitionSpec(None, "tp"), dp=V),
            )
        self.assertExpectedInline(
            str(cm.exception),
            """rules.einsum: '...d,...d->...d': '...' must stand for the same number of dims on every operand, but operand 0 ('...d', shape (2, 3, 4)) gives it 2 and operand 1 ('...d', shape (3, 4)) gives it 1""",
        )

    def test_one_output_per_call(self):
        x = self.sharded((4, 6), PartitionSpec("dp", None), tp=R)
        wq = self.sharded((5, 6), PartitionSpec("tp", None), dp=R)
        wk = self.sharded((5, 6), PartitionSpec("tp", None), dp=R)
        with self.assertRaises(TypeError) as cm:
            rules.einsum("tk,ak,bk->ta,tb", x, wq, wk)
        self.assertExpectedInline(
            str(cm.exception),
            """rules.einsum: 'tk,ak,bk->ta,tb' lists several outputs; write one rules.einsum per output instead""",
        )
        q = rules.einsum("tk,ak->ta", x, wq)
        k = rules.einsum("tk,bk->tb", x, wk)
        for y in (q, k):
            self.assertTyped(y, {"dp": V, "tp": V}, PartitionSpec("dp", "tp"))

    def test_transpose_moves_sharding(self):
        x = self.sharded((4, 6), PartitionSpec("dp", "tp"))
        self.assertTyped(
            rules.einsum("ab->ba", x),
            {"dp": V, "tp": V},
            PartitionSpec("tp", "dp"),
        )

    def test_operands_may_be_dims(self):
        x = self.dims(2, PartitionSpec("dp", "tp"))
        w = self.dims(2, PartitionSpec(None, "tp"), dp=R)
        y = rules.einsum("mk,nk->mn", x, w)
        self.assertIsInstance(y, rules.NdimWithSpmdType)
        self.assertTyped(y, {"dp": V, "tp": P}, PartitionSpec("dp", None))

    def test_partial_operand_requires_linear_in(self):
        x = self.typed((4, 6), dp=R, tp=P)
        w = self.typed((5, 6), dp=R, tp=R)
        with self.assertRaises(SpmdTypeError) as cm:
            rules.einsum("mk,nk->mn", x, w)
        self.assertExpectedInline(
            str(cm.exception),
            """rules.einsum: 'mk,nk->mn': operand(s) [0] are Partial on axis mesh_tp; a Partial may only pass through operands the op is linear in, all within one linear_in group (declare linear_in=(0,) if the op is linear in it)""",
        )
        self.assertTyped(
            rules.einsum("mk,nk->mn", x, w, linear_in=(0,)),
            {"dp": R, "tp": P},
            None,
        )
        self.assertTyped(
            rules.einsum("mk,nk->mn", x, w, linear_in=(0, 1)),
            {"dp": R, "tp": P},
            None,
        )
        # Linear only in w: a Partial x is not allowed through.
        with self.assertRaises(SpmdTypeError):
            rules.einsum("mk,nk->mn", x, w, linear_in=(1,))
        # P * P: two multilinear factors both Partial.
        p2 = self.typed((5, 6), dp=R, tp=P)
        with self.assertRaises(SpmdTypeError):
            rules.einsum("mk,nk->mn", x, p2, linear_in=(0, 1))
        # Jointly linear (add): all-Partial is fine.
        q = self.typed((4, 6), dp=R, tp=P)
        self.assertTyped(
            rules.einsum("td,td->td", x, q, linear_in=((0, 1),)),
            {"dp": R, "tp": P},
            None,
        )
        # Jointly linear: P + R is affine and rejected; all-P is required.
        with self.assertRaises(SpmdTypeError) as cm:
            rules.einsum(
                "td,td->td", x, self.typed((4, 6), dp=R, tp=R), linear_in=((0, 1),)
            )
        self.assertExpectedInline(
            str(cm.exception),
            """rules.einsum: 'td,td->td': operands [0, 1] are declared jointly linear, so on axis mesh_tp either all or none of them may be Partial; [0] are Partial and the rest are not (P + R is affine: the Replicate term would be summed once per rank)""",
        )
        # Partial with Varying on the same axis never combines.
        with self.assertRaises(SpmdTypeError):
            rules.einsum("td,td->td", x, self.typed((4, 6), dp=R, tp=V), linear_in=(0,))
        for bad in ((5,), (0, 0), ("x",)):
            with self.subTest(bad):
                with self.assertRaises(TypeError):
                    rules.einsum("mk,nk->mn", x, w, linear_in=bad)

    def test_local_rule_applies_to_unsharded_axes(self):
        x = self.typed((4, 6), dp=R, tp=I)
        w = self.typed((5, 6), dp=R, tp=V)
        with self.assertRaises(SpmdTypeError) as cm:
            rules.einsum("mk,nk->mn", x, w)
        self.assertExpectedInline(
            str(cm.exception),
            """\
rules.einsum: 'mk,nk->mn': Invariant type on axis mesh_tp cannot mix with other types. Found types: [I, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  convert(tensor, mesh_tp, src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)""",
        )

    def test_bad_equations_and_operands(self):
        for bad in ("mk,nk", "m->n", "m@tp->m", "mm->m", "m->...", "->m"):
            with self.subTest(bad):
                with self.assertRaises(TypeError):
                    rules.einsum(bad, self.typed((4,), dp=R, tp=R))
        with self.assertRaises(TypeError):
            rules.einsum("m,n->mn", self.typed((4,), dp=R, tp=R), None)
        with self.assertRaises(SpmdTypeError) as cm:
            rules.einsum("mk->mk", self.typed((4,), dp=R, tp=R))
        self.assertExpectedInline(
            str(cm.exception),
            """rules.einsum: 'mk->mk': operand 0 ('mk', shape (4,)) has 2 label(s) but 1 dim(s)""",
        )


# =============================================================================
# Type-only transitions
# =============================================================================


class TestTransitions(GlobalSigTestCase):
    def transition_input(self, src):
        if isinstance(src, Shard):
            entries = [None, None]
            entries[src.dim] = "tp"
            return self.dims(2, PartitionSpec(*entries), dp=R, tp=V)
        return self.dims(2, dp=R, tp=src)

    def test_transition_matrices_match_runtime(self):
        types = (R, I, V, P, S(0), S(1))
        cases = (
            (
                rules.all_reduce,
                {(src, dst) for src in (P, V) for dst in (R, I)},
            ),
            (
                rules.all_gather,
                {(src, dst) for src in (V, S(0), S(1)) for dst in (R, I)},
            ),
            (
                rules.reduce_scatter,
                {(src, dst) for src in (P, V) for dst in (V, S(0), S(1))},
            ),
            (
                rules.all_to_all,
                {(src, dst) for src in (V, S(0), S(1)) for dst in (V, S(0), S(1))},
            ),
            (
                rules.reinterpret,
                {
                    (R, R),
                    (R, I),
                    (R, V),
                    (R, P),
                    (I, R),
                    (I, I),
                    (I, V),
                    (I, P),
                    (V, V),
                    (V, P),
                    (P, P),
                },
            ),
            (
                rules.convert,
                {
                    (R, R),
                    (R, I),
                    (R, V),
                    (R, P),
                    (R, S(0)),
                    (R, S(1)),
                    (I, R),
                    (I, I),
                    (I, V),
                    (I, P),
                    (I, S(0)),
                    (I, S(1)),
                    (V, V),
                    (V, P),
                    (V, S(0)),
                    (V, S(1)),
                    (P, P),
                    (S(0), V),
                    (S(0), P),
                    (S(0), S(0)),
                    (S(1), V),
                    (S(1), P),
                    (S(1), S(1)),
                },
            ),
        )

        for fn, accepted in cases:
            for src in types:
                for dst in types:
                    with self.subTest(fn=fn.__name__, src=src, dst=dst):
                        x = self.transition_input(src)
                        if (src, dst) not in accepted:
                            with self.assertRaises(SpmdTypeError):
                                fn(x, "tp", src=src, dst=dst)
                            continue

                        result = fn(x, "tp", src=src, dst=dst)
                        expected_local = V if isinstance(dst, Shard) else dst
                        self.assertIs(get_axis_local_type(result, "tp"), expected_local)
                        self.assertEqual(
                            partition_spec_get_shard(
                                get_partition_spec(result), normalize_axis("tp")
                            ),
                            dst if isinstance(dst, Shard) else None,
                        )

    def test_all_gather_concat_removes_axis(self):
        x = self.sharded((4, 6), PartitionSpec(("dp", "tp"), None))
        y = rules.all_gather(x, "tp", src=S(0), dst=R)
        self.assertTyped(y, {"dp": V, "tp": R}, PartitionSpec("dp", None))
        self.assertEqual(y.ndim, 2)

    def test_all_gather_stack_adds_dim(self):
        x = self.typed((4, 6), dp=R, tp=V)
        y = rules.all_gather(x, "tp", src=V, dst=R)
        self.assertEqual(y.ndim, 3)
        self.assertTyped(y, {"dp": R, "tp": R}, None)

    def test_reduce_scatter_shard_and_unbind(self):
        p = self.dims(2, PartitionSpec("dp", None), tp=P)
        y = rules.reduce_scatter(p, "tp", dst=S(0))
        self.assertTyped(y, {"dp": V, "tp": V}, PartitionSpec(("dp", "tp"), None))
        z = rules.reduce_scatter(self.dims(2, dp=R, tp=P), "tp", dst=V)
        self.assertEqual(z.ndim, 1)
        self.assertTyped(z, {"dp": R, "tp": V}, None)
        with self.assertRaises(SpmdTypeError):
            rules.reduce_scatter(p, "tp", dst=V)  # dim 0 is sharded on dp

    def test_reducing_varying_requires_explicit_src(self):
        v = self.dims(2, PartitionSpec("dp", None), tp=V)
        with self.assertRaises(SpmdTypeError) as cm:
            rules.all_reduce(v, "tp", dst=I)  # default src=P
        self.assertExpectedInline(
            str(cm.exception),
            """rules.all_reduce('tp', src=P, dst=I) expects its input to be P on axis mesh_tp, but it is V""",
        )
        self.assertTyped(
            rules.all_reduce(v, "tp", src=V, dst=I),
            {"dp": V, "tp": I},
            PartitionSpec("dp", None),
        )

    def test_src_is_checked(self):
        x = self.typed((4,), dp=R, tp=R)
        with self.assertRaises(SpmdTypeError) as cm:
            rules.all_reduce(x, "tp", dst=I)
        self.assertExpectedInline(
            str(cm.exception),
            """rules.all_reduce('tp', src=P, dst=I) expects its input to be P on axis mesh_tp, but it is R""",
        )
        with self.assertRaises(SpmdTypeError) as cm:
            rules.all_gather(
                self.sharded((4, 6), PartitionSpec("dp", "tp")),
                "tp",
                src=S(0),
                dst=R,
            )
        self.assertExpectedInline(
            str(cm.exception),
            """rules.all_gather('tp', src=S(0), dst=R) expects its input sharded on dim 0 along mesh_tp, but it is sharded on dim 1""",
        )
        with self.assertRaises(SpmdTypeError) as cm:
            rules.all_gather(self.dims(2, dp=R, tp=V), "tp", src=S(0), dst=R)
        self.assertIn("not sharded", str(cm.exception))

    def test_src_is_set_when_untyped(self):
        x = self.typed((4,), dp=R)
        y = rules.all_gather(x, "tp", src=V, dst=R)
        self.assertIs(get_axis_local_type(x, "tp"), V)
        self.assertIs(get_axis_local_type(y, "tp"), R)

    def test_singleton_axis_checks_nothing_but_keeps_the_rank_change(self):
        from spmd_types._mesh_axis import MeshAxis

        x = self.typed((4,), dp=R, tp=V)
        y = rules.all_gather(x, MeshAxis.of(1, 1), src=S(0), dst=R)
        self.assertEqual(y.ndim, 1)
        self.assertTyped(y, {"dp": R, "tp": V}, None)
        # The stack form still adds the leading dim the kernel produces.
        y = rules.all_gather(x, MeshAxis.of(1, 1), src=V, dst=R)
        self.assertEqual(y.ndim, 2)
        self.assertTyped(y, {"dp": R, "tp": V}, None)

    def test_inadmissible_pairs_rejected(self):
        x = self.typed((4,), dp=R, tp=R)
        for fn, src, dst in (
            (rules.all_gather, R, V),
            (rules.all_reduce, R, R),
            (rules.reduce_scatter, V, R),
            (rules.reinterpret, P, R),
            (rules.convert, V, R),
        ):
            with self.subTest(fn.__name__):
                with self.assertRaises(SpmdTypeError):
                    fn(x, "tp", src=src, dst=dst)
        with self.assertRaises(TypeError):
            rules.all_gather(x, "tp", src="V", dst=R)

    def test_reinterpret_and_convert(self):
        x = self.typed((4,), dp=R, tp=V)
        self.assertTyped(
            rules.reinterpret(x, "tp", src=V, dst=P), {"dp": R, "tp": P}, None
        )
        r = self.typed((4,), dp=R, tp=R)
        self.assertTyped(
            rules.convert(r, "tp", src=R, dst=S(0)),
            {"dp": R, "tp": V},
            PartitionSpec("tp"),
        )
        i = self.typed((4,), dp=R, tp=I)
        self.assertTyped(
            rules.reinterpret(i, "tp", src=I, dst=P),
            {"dp": R, "tp": P},
            None,
        )
        with self.assertRaises(SpmdTypeError):
            rules.reinterpret(x, "tp", src=S(0), dst=P)


# =============================================================================
# Hooks: rules.output, coverage, resolution
# =============================================================================


class TestHooks(GlobalSigTestCase):
    def test_output_stamps_type_and_spec(self):
        class Op(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x, w):
                rules.output(out, rules.einsum("mk,nk->mn", x, w))

            @staticmethod
            def forward(ctx, x, w):
                return x @ w.T

            @staticmethod
            def backward(ctx, g):
                return g, g

        x = self.sharded((4, 6), PartitionSpec("dp", None), tp=R)
        w = self.sharded((5, 6), PartitionSpec("tp", None), dp=R)
        out = Op.apply(x, w)
        self.assertTyped(out, {"dp": V, "tp": V}, PartitionSpec("dp", "tp"))
        out * 1.0  # valid under global checking

    def test_output_rank_mismatch(self):
        class Op(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x):
                rules.output(out, rules.einsum("ab->a", x))

            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, g):
                return g

        with self.assertRaises(SpmdTypeError) as cm:
            Op.apply(self.typed((4, 6), dp=R, tp=R))
        self.assertIn(
            "output has 2 dim(s) but its declared type has 1", str(cm.exception)
        )

    def test_output_rejects_conflicting_partition_spec(self):
        out = self.sharded((4, 6), PartitionSpec("dp", None), dp=V, tp=R)
        typed = self.dims(2, PartitionSpec(None, "dp"), dp=V, tp=R)

        with self.assertRaisesRegex(SpmdTypeError, "PartitionSpec conflict"):
            rules.output(out, typed)

    def test_forgotten_tensor_argument(self):
        class Op(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x):
                rules.output(out, rules.einsum("td->td", x))

            @staticmethod
            def forward(ctx, x, w):
                return x * w

            @staticmethod
            def backward(ctx, g):
                return g, g

        with self.assertRaises(SpmdTypeError) as cm:
            Op.apply(self.typed((4, 6), dp=R, tp=R), self.typed((4, 6), dp=R, tp=R))
        self.assertExpectedInline(
            str(cm.exception),
            """\
Op.spmd_typecheck: tensor argument 'w' was not accounted for. Pass it to the rules operation that consumes it, assert_type it, or declare rules.ignore(w) if it does not affect the forward value.

  In apply(
    args[0]: f32[4, 6] {mesh_dp: R, mesh_tp: R},
    args[1]: f32[4, 6] {mesh_dp: R, mesh_tp: R},
  ) under mesh {mesh_dp, mesh_tp}""",
        )

    def test_ignore_and_identity(self):
        class SaveForRecompute(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, activation, residual, router_logits):
                rules.ignore(residual, router_logits)
                rules.output(out, activation)

            @staticmethod
            def forward(ctx, activation, residual, router_logits, layer_id):
                ctx.save_for_backward(residual, router_logits)
                return activation.clone()

            @staticmethod
            def backward(ctx, g):
                return g, None, None, None

        act = self.sharded((4, 6), PartitionSpec("dp", None), tp=I)
        out = SaveForRecompute.apply(
            act, self.typed((4, 6), dp=P, tp=P), self.typed((4, 6), dp=R, tp=I), 3
        )
        self.assertTyped(out, {"dp": V, "tp": I}, PartitionSpec("dp", None))

    def test_untyped_output_is_an_error(self):
        class Op(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x):
                rules.einsum("t->t", x)  # forgot rules.output

            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, g):
                return g

        with self.assertRaises(SpmdTypeError) as cm:
            Op.apply(self.typed((4,), dp=R, tp=R))
        self.assertExpectedInline(
            str(cm.exception),
            """\
Op.spmd_typecheck: output 0 was left untyped; return its type from the hook, or finish the rule with rules.output(out, <NdimWithSpmdType>) or assert_type(out, ...)

  In apply(
    args[0]: f32[4] {mesh_dp: R, mesh_tp: R},
  ) under mesh {mesh_dp, mesh_tp}""",
        )

    def test_multiple_outputs_and_optional_output(self):
        class Op(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x, wq, wk):
                q = rules.einsum("tk,ak->ta", x, wq)
                k = rules.einsum("tk,bk->tb", x, wk)
                rules.output(out, (q, k, None))

            @staticmethod
            def forward(ctx, x, wq, wk):
                return x @ wq.T, x @ wk.T, None

            @staticmethod
            def backward(ctx, gq, gk, gn):
                return gq + gk, gq, gk

        x = self.sharded((4, 6), PartitionSpec("dp", None), tp=R)
        w = lambda: self.sharded((5, 6), PartitionSpec("tp", None), dp=R)  # noqa: E731
        q, k, n = Op.apply(x, w(), w())
        self.assertIsNone(n)
        for y in (q, k):
            self.assertTyped(y, {"dp": V, "tp": V}, PartitionSpec("dp", "tp"))

    def test_classical_hook_must_account_for_every_input(self):
        class Op(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x):
                assert_type(x, {"tp": R})  # says nothing about w
                assert_type(out, {"tp": R})

            @staticmethod
            def forward(ctx, x, w):
                return x * w

            @staticmethod
            def backward(ctx, g):
                return g, g

        with self.assertRaises(SpmdTypeError) as cm:
            Op.apply(self.typed((4,), dp=R, tp=R), self.typed((4,), dp=R, tp=R))
        self.assertIn("'w' was not accounted for", str(cm.exception))

    def test_return_form_stamps_outputs(self):
        class Op(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x, wq, wk):
                return (
                    rules.einsum("tk,ak->ta", x, wq),
                    rules.einsum("tk,bk->tb", x, wk),
                    None,
                )

            @staticmethod
            def forward(ctx, x, wq, wk):
                return x @ wq.T, x @ wk.T, 7

            @staticmethod
            def backward(ctx, gq, gk, gn):
                return gq + gk, gq, gk

        x = self.sharded((4, 6), PartitionSpec("dp", None), tp=R)
        w = lambda: self.sharded((5, 6), PartitionSpec("tp", None), dp=R)  # noqa: E731
        q, k, n = Op.apply(x, w(), w())
        self.assertEqual(n, 7)
        for y in (q, k):
            self.assertTyped(y, {"dp": V, "tp": V}, PartitionSpec("dp", "tp"))

    def test_return_form_identity_and_coverage(self):
        class Marker(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, activation, residual):
                rules.ignore(residual)
                return activation

            @staticmethod
            def forward(ctx, activation, residual):
                return activation.clone()

            @staticmethod
            def backward(ctx, g):
                return g, None

        act = self.sharded((4, 6), PartitionSpec("dp", None), tp=I)
        out = Marker.apply(act, self.typed((4, 6), dp=P, tp=P))
        self.assertTyped(out, {"dp": V, "tp": I}, PartitionSpec("dp", None))

        class Forgetful(Marker):
            @staticmethod
            def spmd_typecheck(*, activation):
                return activation

        with self.assertRaises(SpmdTypeError) as cm:
            Forgetful.apply(act, self.typed((4, 6), dp=P, tp=P))
        self.assertIn(
            "tensor argument 'residual' was not accounted for", str(cm.exception)
        )

        class Nothing(Marker):
            @staticmethod
            def spmd_typecheck(*, activation, residual):
                rules.ignore(residual)

        with self.assertRaises(SpmdTypeError) as cm:
            Nothing.apply(act, self.typed((4, 6), dp=P, tp=P))
        self.assertExpectedInline(
            str(cm.exception),
            """\
Nothing.spmd_typecheck: returned None; a keyword-only spmd_typecheck must return the output type(s) (a NdimWithSpmdType, a tensor, or a tuple)

  In apply(
    args[0]: f32[4, 6] {mesh_tp: I, mesh_dp: V} PartitionSpec(mesh_dp, None),
    args[1]: f32[4, 6] {mesh_dp: P, mesh_tp: P},
  ) under mesh {mesh_dp, mesh_tp}""",
        )

    def test_out_argument_stamps_and_returns_out(self):
        x = self.sharded((4, 6), PartitionSpec("dp", "tp"))
        target = self.rank_map(lambda r: torch.randn(4, 6))
        ret = rules.einsum("ab->ba", x, out=target)
        self.assertIs(ret, target)
        self.assertTyped(target, {"dp": V, "tp": V}, PartitionSpec("tp", "dp"))
        with self.assertRaises(SpmdTypeError):
            rules.all_gather(x, "tp", src=V, dst=R, out=target)  # stack form has rank 3

    def test_assert_type_on_an_input_counts_as_coverage(self):
        """A hook may still assert_type an input it does not feed to a rules
        op; that counts as using it, and the assertion is enforced."""

        class ScaledCopy(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x, scale):
                assert_type(scale, {"dp": R, "tp": R})
                return rules.einsum("...->...", x)

            @staticmethod
            def forward(ctx, x, scale):
                return x * scale.mean()

            @staticmethod
            def backward(ctx, g):
                return None, None

        x = self.sharded((4, 6), PartitionSpec("dp", "tp"))
        out = ScaledCopy.apply(x, self.typed((3,), dp=R, tp=R))
        self.assertTyped(out, {"dp": V, "tp": V}, PartitionSpec("dp", "tp"))
        with self.assertRaises(SpmdTypeError):
            ScaledCopy.apply(x, self.typed((3,), dp=R, tp=V))

    def test_type_by_fiat(self):
        """When nothing derives a type, build a NdimWithSpmdType of the right
        rank and assert_type it, exactly as for a real tensor."""

        class Opaque(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x):
                rules.ignore(x)
                d = rules.NdimWithSpmdType(2)
                assert_type(d, {"dp": V, "tp": I}, PartitionSpec("dp", None))
                return d

            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, g):
                return None

        out = Opaque.apply(self.typed((4, 6), dp=R, tp=R))
        self.assertTyped(out, {"dp": V, "tp": I}, PartitionSpec("dp", None))
        d = rules.NdimWithSpmdType(1)
        assert_type(d, {"dp": R, "tp": S(0)})
        self.assertTyped(d, {"dp": R, "tp": V}, PartitionSpec("tp"))

    def test_hook_names_must_be_forward_parameters(self):
        class Op(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, nope):
                pass

            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, g):
                return g

        with self.assertRaises(TypeError) as cm:
            Op.apply(self.typed((4,), dp=R, tp=R))
        self.assertExpectedInline(
            str(cm.exception),
            """Op.spmd_typecheck: 'nope' is not a parameter of Op.forward (has ['x'])""",
        )


# =============================================================================
# Use cases from torchtitan and Megatron-style pretraining code
# =============================================================================


class TestUseCases(GlobalSigTestCase):
    """Real kernels written as composed rules.  Layouts follow torchtitan's
    dense conventions collapsed onto a (dp, tp) mesh."""

    def test_all_gather_linear(self):
        """torchtitan AllGatherLinear: all-gather the sequence, then a
        column-parallel GEMM with an optional bias."""

        class AllGatherLinear(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x_shard_m, w_shard_n, bias_shard_n, group):
                x = rules.all_gather(x_shard_m, group, src=S(0), dst=R)
                y = rules.einsum("mk,nk->mn", x, w_shard_n)
                if bias_shard_n is not None:
                    y = rules.einsum("mn,n->mn", y, bias_shard_n)
                rules.output(out, y)

            @staticmethod
            def forward(ctx, x_shard_m, w_shard_n, bias_shard_n, group):
                y = x_shard_m.repeat(2, 1) @ w_shard_n.T
                return y if bias_shard_n is None else y + bias_shard_n

            @staticmethod
            def backward(ctx, g):
                return g, g, g, None

        x = self.sharded((4, 6), PartitionSpec(("dp", "tp"), None))
        w = self.sharded((5, 6), PartitionSpec("tp", None), dp=R)
        b = self.sharded((5,), PartitionSpec("tp"), dp=R)
        for bias in (b, None):
            y = AllGatherLinear.apply(x, w, bias, self.tp_group)
            self.assertTyped(y, {"dp": V, "tp": V}, PartitionSpec("dp", "tp"))
            y * 1.0

    def test_linear_reduce_scatter(self):
        """torchtitan LinearReduceScatter: row-parallel GEMM (partial over the
        sharded K), reduce-scattered over the sequence, then bias."""

        class LinearReduceScatter(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x_shard_k, w_shard_k, bias, group):
                y = rules.einsum("mk,nk->mn", x_shard_k, w_shard_k)
                if bias is None:
                    rules.reduce_scatter(y, group, src=P, dst=S(0), out=out)
                else:
                    y = rules.reduce_scatter(y, group, src=P, dst=S(0))
                    rules.einsum("mn,n->mn", y, bias, out=out)

            @staticmethod
            def forward(ctx, x_shard_k, w_shard_k, bias, group):
                y = (x_shard_k @ w_shard_k.T)[: x_shard_k.shape[0] // 2]
                return y if bias is None else y + bias

            @staticmethod
            def backward(ctx, g):
                return g, g, g, None

        x = self.sharded((4, 6), PartitionSpec("dp", "tp"))
        w = self.sharded((5, 6), PartitionSpec(None, "tp"), dp=R)
        bias = self.typed((5,), dp=R, tp=R)
        y = LinearReduceScatter.apply(x, w, bias, self.tp_group)
        self.assertTyped(y, {"dp": V, "tp": V}, PartitionSpec(("dp", "tp"), None))
        y * 1.0

    def test_linear_all_reduce(self):
        """Row-parallel GEMM completed by an all-reduce; backward passes the
        gradient through, so the output is Invariant."""

        class LinearAllReduce(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x, weight, group):
                y = rules.einsum("mk,nk->mn", x, weight)
                return rules.all_reduce(y, group, src=P, dst=I)

            @staticmethod
            def forward(ctx, x, weight, group):
                return x @ weight.T

            @staticmethod
            def backward(ctx, g):
                return g, g, None

        x = self.sharded((4, 6), PartitionSpec("dp", "tp"))
        w = self.sharded((5, 6), PartitionSpec(None, "tp"), dp=R)
        y = LinearAllReduce.apply(x, w, self.tp_group)
        self.assertTyped(y, {"dp": V, "tp": I}, PartitionSpec("dp", None))

    def test_vocab_parallel_cross_entropy(self):
        """torchtitan _LossParallelCrossEntropy / Megatron _VocabParallelCrossEntropy: vocab-sharded
        logits; per-token loss all-reduced to Invariant.  With reduction="sum"
        the token dim is contracted too and the loss is Partial on dp."""

        class LossParallelCE(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, logits, labels, tp_group, reduction):
                # The function is log(sum_v exp) - logits[t, label]; both
                # reductions over v are sums, the stabilizing max is not part
                # of the function.  See the case study in docs/rules.md.
                z = rules.all_reduce(
                    rules.einsum("tv->t", logits), tp_group, src=P, dst=I
                )
                pred = rules.all_reduce(
                    rules.einsum("tv,t->t", logits, labels), tp_group, src=P, dst=I
                )
                per_token = rules.einsum("t,t->t", z, pred)
                if reduction == "none":
                    rules.output(out, per_token)
                else:
                    rules.output(out, rules.einsum("t->", per_token))

            @staticmethod
            def forward(ctx, logits, labels, tp_group, reduction):
                per_token = torch.logsumexp(logits, dim=-1)
                return per_token if reduction == "none" else per_token.sum()

            @staticmethod
            def backward(ctx, g):
                return None, None, None, None

        logits = self.sharded((4, 3), PartitionSpec("dp", "tp"))
        labels = self.sharded((4,), PartitionSpec("dp"), tp=R)
        per_token = LossParallelCE.apply(logits, labels, self.tp_group, "none")
        self.assertTyped(per_token, {"dp": V, "tp": I}, PartitionSpec("dp"))
        total = LossParallelCE.apply(logits, labels, self.tp_group, "sum")
        self.assertTyped(total, {"dp": P, "tp": I}, None)

    def test_vocab_parallel_embedding(self):
        """Megatron VocabParallelEmbedding: a masked lookup into a vocab shard
        is a partial sum over the vocab contraction, left for the caller."""

        class Embed(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, tokens, weight):
                rules.output(out, rules.einsum("t,vd->td", tokens, weight))

            @staticmethod
            def forward(ctx, tokens, weight):
                return weight[tokens % weight.shape[0]]

            @staticmethod
            def backward(ctx, g):
                return None, g.sum(0, keepdim=True).expand(3, -1)

        tokens = self.rank_map(lambda r: torch.randint(0, 6, (4,)))
        assert_type(tokens, {"tp": R}, PartitionSpec("dp"))
        weight = self.sharded((3, 5), PartitionSpec("tp", None), dp=R)
        out = Embed.apply(tokens, weight)
        self.assertTyped(out, {"dp": V, "tp": P}, PartitionSpec("dp", None))

    def test_gather_then_norm(self):
        """Sequence-parallel norm: gather the sequence, RMS-normalize over an
        unsharded hidden dim."""

        class GatherThenNorm(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x, weight, group):
                full = rules.all_gather(x, group, src=S(0), dst=R)
                rules.output(out, rules.einsum("t_,_->t_", full, weight))

            @staticmethod
            def forward(ctx, x, weight, eps, group):
                full = x.repeat(2, 1)
                return torch.nn.functional.rms_norm(full, (full.shape[1],), weight, eps)

            @staticmethod
            def backward(ctx, g):
                return g[:2], g.sum(0), None, None

        x = self.sharded((4, 6), PartitionSpec(("dp", "tp"), None))
        w = self.typed((6,), dp=R, tp=R)
        out = GatherThenNorm.apply(x, w, 1e-5, self.tp_group)
        self.assertTyped(out, {"dp": V, "tp": R}, PartitionSpec("dp", None))
        with self.assertRaises(SpmdTypeError):
            GatherThenNorm.apply(
                self.sharded((4, 6), PartitionSpec("dp", "tp")),
                w,
                1e-5,
                self.tp_group,
            )

    def test_silu_and_mul_and_rope(self):
        """torchtitan _silu_and_mul_2d and _FusedMLAQ: opaque local kernels
        whose sharding follows the labels."""

        class SiluAndMul(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, gate, up):
                rules.output(out, rules.einsum("...d,...d->...d", gate, up))

            @staticmethod
            def forward(ctx, gate, up):
                return torch.nn.functional.silu(gate) * up

            @staticmethod
            def backward(ctx, g):
                return g, g

        class FusedMLAQ(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, q, rope_cache_real, positions):
                rules.output(
                    out, rules.einsum("bthd,se,bt->bthd", q, rope_cache_real, positions)
                )

            @staticmethod
            def forward(ctx, q, rope_cache_real, positions, q_nope_dim):
                return q * 1.0

            @staticmethod
            def backward(ctx, g):
                return g, None, None, None

        spec = PartitionSpec("dp", "tp")
        out = SiluAndMul.apply(self.sharded((4, 6), spec), self.sharded((4, 6), spec))
        self.assertTyped(out, {"dp": V, "tp": V}, spec)

        q = self.sharded((1, 4, 2, 3), PartitionSpec(None, "dp", "tp", None))
        cache = self.typed((8, 3), dp=R, tp=R)
        positions = self.sharded((1, 4), PartitionSpec(None, "dp"), tp=R)
        out = FusedMLAQ.apply(q, cache, positions, 2)
        self.assertTyped(out, {"dp": V, "tp": V}, PartitionSpec(None, "dp", "tp", None))

    def test_variadic_offload_identity(self):
        """Activation-offload commit marker: forward(ctx, *args), each
        output typed like its own input; composition handles it with a loop."""

        class OffloadCommit(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, args):
                tensors = args[:-1]
                rules.output(out, tensors)

            @staticmethod
            def forward(ctx, *args):
                return tuple(t.clone() for t in args[:-1])

            @staticmethod
            def backward(ctx, *grads):
                return grads + (None,)

        a = self.sharded((4, 6), PartitionSpec("dp", None), tp=I)
        b = self.typed((3,), dp=R, tp=P)
        oa, ob = OffloadCommit.apply(a, b, "name")
        self.assertTyped(oa, {"dp": V, "tp": I}, PartitionSpec("dp", None))
        self.assertTyped(ob, {"dp": R, "tp": P}, None)

    def test_gather_sp_and_reduce_sp(self):
        """MoE token dispatcher GatherSP / ReduceSP: sequence-parallel all-gather and reduce-scatter."""

        class GatherSP(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x, tp_group):
                rules.output(out, rules.all_gather(x, tp_group, src=S(0), dst=R))

            @staticmethod
            def forward(ctx, x, tp_group):
                return x.repeat(2, 1)

            @staticmethod
            def backward(ctx, g):
                return g[:2], None

        class ReduceSP(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x, tp_group):
                rules.output(out, rules.reduce_scatter(x, tp_group, src=P, dst=S(0)))

            @staticmethod
            def forward(ctx, x, tp_group):
                return x[:2].clone()

            @staticmethod
            def backward(ctx, g):
                return g.repeat(2, 1), None

        x = self.sharded((4, 6), PartitionSpec(("dp", "tp"), None))
        self.assertTyped(
            GatherSP.apply(x, self.tp_group),
            {"dp": V, "tp": R},
            PartitionSpec("dp", None),
        )
        p = self.sharded((4, 6), PartitionSpec("dp", None), tp=P)
        self.assertTyped(
            ReduceSP.apply(p, self.tp_group),
            {"dp": V, "tp": V},
            PartitionSpec(("dp", "tp"), None),
        )


class TestLocalUseCases(LocalSigTestCase):
    """Kernels whose inputs are Varying with no PartitionSpec, which local
    SPMD checking allows and global checking rejects up front."""

    def test_stack_gather(self):
        """all_gather_into_tensor into a new leading dim: stack semantics."""

        class StackGather(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(out, *, x, group):
                rules.output(out, rules.all_gather(x, group, src=V, dst=R))

            @staticmethod
            def forward(ctx, x, group):
                return torch.stack([x, x])

            @staticmethod
            def backward(ctx, g):
                return g[0], None

        out = StackGather.apply(self.typed((4, 6), dp=R, tp=V), self.tp_group)
        self.assertTyped(out, {"dp": R, "tp": R}, None)


# =============================================================================
# Einsum equation parsing
# =============================================================================


class TestEinsumParsing(unittest.TestCase):
    """The equation parser, independent of any mesh."""

    def parse(self, eq):
        from spmd_types.rules import _parse_equation

        return _parse_equation(eq)

    def test_labels_underscore_and_ellipsis(self):
        eq = self.parse("ab_,...b->a...")
        self.assertEqual(
            [op.labels for op in eq.inputs], [("a", "b", "_"), ("...", "b")]
        )
        self.assertEqual(eq.output.labels, ("a", "..."))
        self.assertEqual(eq.inputs[1].expand(2), ["...0", "...1", "b"])

    def test_scalar_output_and_whitespace(self):
        eq = self.parse(" a b , b -> ")
        self.assertEqual(eq.output.labels, ())
        self.assertEqual(eq.inputs[0].labels, ("a", "b"))

    def test_unicode_labels(self):
        eq = self.parse("\u03b1\u03b2,\u03b2->\u03b1")
        self.assertEqual(eq.output.labels, ("\u03b1",))

    def test_errors(self):
        cases = {
            "ab,b": "must contain exactly one '->'",
            "ab->b->a": "must contain exactly one '->'",
            "ab,,b->a": "empty operand",
            "a1->a": "unexpected '1'",
            "a@b->a": "unexpected '@'",
            "aa->a": "repeats a label",
            "......a->a": "two '...'",
            "ab->ac": "output label 'c' does not appear",
            "ab->...a": "output uses '...' but no input does",
            "ab,bc->ac,ab": "lists several outputs",
        }
        for eq, msg in cases.items():
            with self.subTest(eq):
                with self.assertRaises(TypeError) as cm:
                    self.parse(eq)
                self.assertIn(msg, str(cm.exception))
