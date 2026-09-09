# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for spmd_types.rulecheck: numerically checking an spmd_typecheck hook
against its kernel, on plain per-rank tensors or (local_tensor_mode=True) under
LocalTensorMode with a fake process group.
"""

import unittest

import torch
import torch.distributed as dist
from spmd_types import assert_type, I, P, R, rules, S, Shard, SpmdTypeError, V
from spmd_types.rulecheck import rulecheck, RuleCheckError

MESH = {"dp": 2, "tp": 2}


class AllGatherLinear(torch.autograd.Function):
    """Gather the sequence over the group, then a column-parallel GEMM."""

    @staticmethod
    def spmd_typecheck(*, x, w, bias, group):
        full = rules.all_gather(x, group, src=S(0), dst=R)
        y = rules.einsum("mk,nk->mn", full, w)
        return y if bias is None else rules.einsum("mn,n->mn", y, bias)

    @staticmethod
    def forward(ctx, x, w, bias, group):
        world = dist.get_world_size(group)
        full = torch.empty(world * x.shape[0], x.shape[1], dtype=x.dtype)
        dist.all_gather_into_tensor(full, x.contiguous(), group=group)
        y = full @ w.T
        return y if bias is None else y + bias

    @staticmethod
    def backward(ctx, g):
        return g, g, g, None


class LinearReduceScatter(torch.autograd.Function):
    """Row-parallel GEMM, reduce-scattered over the sequence."""

    @staticmethod
    def spmd_typecheck(*, x, w, group):
        y = rules.einsum("mk,nk->mn", x, w)
        return rules.reduce_scatter(y, group, src=P, dst=S(0))

    @staticmethod
    def forward(ctx, x, w, group):
        y = (x @ w.T).contiguous()
        out = torch.empty(
            y.shape[0] // dist.get_world_size(group), y.shape[1], dtype=y.dtype
        )
        dist.reduce_scatter_tensor(out, y, group=group)
        return out

    @staticmethod
    def backward(ctx, g):
        return g, g, None


class LinearAllReduce(torch.autograd.Function):
    @staticmethod
    def spmd_typecheck(*, x, w, group):
        return rules.all_reduce(rules.einsum("mk,nk->mn", x, w), group, src=P, dst=I)

    @staticmethod
    def forward(ctx, x, w, group):
        y = x @ w.T
        dist.all_reduce(y, group=group)
        return y

    @staticmethod
    def backward(ctx, g):
        return g, g, None


class VocabParallelLogSumExp(torch.autograd.Function):
    """logsumexp over a vocab sharded across the group.  The kernel stabilizes
    with an all-reduced max; the rule types the function, log(sum_v exp), in
    which the only reduction over v is a sum."""

    @staticmethod
    def spmd_typecheck(*, logits, group):
        z = rules.einsum("tv->t", logits)  # sum_v exp(logits)
        z = rules.all_reduce(z, group, src=P, dst=I)
        return rules.einsum("t->t", z)  # log

    @staticmethod
    def forward(ctx, logits, group):
        m = logits.max(dim=-1).values.contiguous()
        dist.all_reduce(m, op=dist.ReduceOp.MAX, group=group)
        s = (logits - m[:, None]).exp().sum(-1)
        dist.all_reduce(s, group=group)
        return m + s.log()

    @staticmethod
    def backward(ctx, g):
        return None, None


class VocabParallelCrossEntropy(torch.autograd.Function):
    """Megatron _VocabParallelCrossEntropy: max / sum-exp / masked target
    gather, each all-reduced; see the case study in docs/rules.md."""

    @staticmethod
    def spmd_typecheck(*, logits, target, group):
        z = rules.einsum("tv->t", logits)
        z = rules.all_reduce(z, group, src=P, dst=I)
        pred = rules.einsum("tv,t->t", logits, target)
        pred = rules.all_reduce(pred, group, src=P, dst=I)
        return rules.einsum("t,t->t", z, pred)

    @staticmethod
    def forward(ctx, logits, target, group):
        rank = dist.get_rank(group)
        m = logits.max(dim=-1).values.contiguous()
        dist.all_reduce(m, op=dist.ReduceOp.MAX, group=group)
        shifted = logits - m[:, None]
        s = shifted.exp().sum(-1)
        dist.all_reduce(s, group=group)
        v_local = logits.shape[-1]
        lo = rank * v_local
        in_range = (target >= lo) & (target < lo + v_local)
        idx = (target - lo).clamp(0, v_local - 1)
        pred = shifted.gather(1, idx[:, None]).squeeze(1) * in_range
        dist.all_reduce(pred, group=group)
        return s.log() - pred

    @staticmethod
    def backward(ctx, g):
        return None, None, None


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.softmax(x, dim=-1)

    @staticmethod
    def backward(ctx, g):
        return g


class TestRuleCheck(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_all_gather_linear(self):
        x, w, b = torch.randn(8, 6), torch.randn(4, 6), torch.randn(4)
        rulecheck(
            AllGatherLinear,
            (x, w, b, "tp"),
            {
                "x": {"dp": S(0), "tp": S(0)},
                "w": {"dp": R, "tp": S(0)},
                "bias": {"dp": R, "tp": S(0)},
            },
            mesh=MESH,
            local_tensor_mode=True,
        )

    def test_linear_reduce_scatter(self):
        x, w = torch.randn(8, 6), torch.randn(4, 6)
        rulecheck(
            LinearReduceScatter,
            (x, w, "tp"),
            {"x": {"dp": S(0), "tp": S(1)}, "w": {"dp": R, "tp": S(1)}},
            mesh=MESH,
            local_tensor_mode=True,
        )

    def test_linear_all_reduce(self):
        x, w = torch.randn(8, 6), torch.randn(4, 6)
        rulecheck(
            LinearAllReduce,
            (x, w, "tp"),
            {"x": {"dp": S(0), "tp": S(1)}, "w": {"dp": R, "tp": S(1)}},
            mesh=MESH,
            local_tensor_mode=True,
        )

    def test_vocab_parallel_logsumexp(self):
        logits = torch.randn(8, 6)
        rulecheck(
            VocabParallelLogSumExp,
            (logits, "tp"),
            {"logits": {"dp": S(0), "tp": S(1)}},
            mesh=MESH,
            local_tensor_mode=True,
        )

    def test_partial_input_is_split_non_degenerately(self):
        class ReduceScatterTokens(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x, group):
                return rules.reduce_scatter(x, group, src=P, dst=S(0))

            @staticmethod
            def forward(ctx, x, group):
                out = torch.empty(x.shape[0] // dist.get_world_size(group), x.shape[1])
                dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
                return out

            @staticmethod
            def backward(ctx, g):
                return g, None

        x = torch.randn(8, 6)
        rulecheck(
            ReduceScatterTokens,
            (x, "tp"),
            {"x": {"dp": S(0), "tp": P}},
            mesh=MESH,
            local_tensor_mode=True,
        )

    def test_wrong_free_label_claim_is_caught(self):
        """softmax mixes along d; claiming d passes through is a global lie."""

        class BadSoftmax(Softmax):
            @staticmethod
            def spmd_typecheck(*, x):
                return rules.einsum("...d->...d", x)

        with self.assertRaises(RuleCheckError) as cm:
            rulecheck(
                BadSoftmax,
                (torch.randn(4, 8),),
                {"x": {"dp": S(0), "tp": S(1)}},
                mesh=MESH,
            )
        self.assertIn("differs from the reference", str(cm.exception))

        class GoodSoftmax(Softmax):
            @staticmethod
            def spmd_typecheck(*, x):
                return rules.einsum("..._->..._", x)

        # The honest hook refuses the d-sharded input outright.
        with self.assertRaises(SpmdTypeError):
            rulecheck(
                GoodSoftmax,
                (torch.randn(4, 8),),
                {"x": {"dp": S(0), "tp": S(1)}},
                mesh=MESH,
            )
        rulecheck(
            GoodSoftmax,
            (torch.randn(4, 8),),
            {"x": {"dp": S(0), "tp": R}},
            mesh=MESH,
        )

    def test_forgotten_reduce_is_caught(self):
        """A row-parallel GEMM whose kernel never reduces: the hook claims the
        all-reduce happened, but ranks disagree."""

        class Liar(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x, w, group):
                return rules.all_reduce(
                    rules.einsum("mk,nk->mn", x, w),
                    group,
                    src=P,
                    dst=I,
                )

            @staticmethod
            def forward(ctx, x, w, group):
                return x @ w.T  # no all_reduce

            @staticmethod
            def backward(ctx, g):
                return g, g, None

        with self.assertRaises(RuleCheckError) as cm:
            rulecheck(
                Liar,
                (torch.randn(8, 6), torch.randn(4, 6), "tp"),
                {"x": {"dp": S(0), "tp": S(1)}, "w": {"dp": R, "tp": S(1)}},
                mesh=MESH,
                local_tensor_mode=True,
            )
        self.assertIn("typed I on 'tp'", str(cm.exception))
        self.assertIn("hold different values", str(cm.exception))

    def test_plain_backend_needs_no_process_group(self):
        """Kernels without collectives run on ordinary tensors; a process group
        may even already be initialized."""

        class Transpose(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x):
                return rules.einsum("ab->ba", x)

            @staticmethod
            def forward(ctx, x):
                return x.T.contiguous()

            @staticmethod
            def backward(ctx, g):
                return g.T.contiguous()

        from torch.testing._internal.distributed.fake_pg import FakeStore

        dist.init_process_group(backend="fake", rank=0, world_size=3, store=FakeStore())
        try:
            rulecheck(
                Transpose,
                (torch.randn(4, 6),),
                {"x": {"dp": S(0), "tp": S(1)}},
                mesh=MESH,
            )
            with self.assertRaises(RuleCheckError) as cm:
                rulecheck(
                    Transpose,
                    (torch.randn(4, 6),),
                    {"x": {"dp": S(0), "tp": S(1)}},
                    mesh=MESH,
                    local_tensor_mode=True,
                )
            self.assertIn("call it with none initialized", str(cm.exception))
        finally:
            dist.destroy_process_group()

    def test_local_tensor_mode_can_be_forced(self):
        class Twice(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x):
                return rules.einsum("...->...", x)

            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, g):
                return g * 2

        rulecheck(
            Twice,
            (torch.randn(4, 6),),
            {"x": {"dp": S(0), "tp": R}},
            mesh=MESH,
            local_tensor_mode=True,
        )

    def test_enumeration_skips_rejected_placements(self):
        """placements=None shards each dim in turn; the norm's hidden dim is
        rejected by the hook (`_`), the token dim is checked."""

        class RMSNorm(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x, weight, eps):
                return rules.einsum("t_,_->t_", x, weight)

            @staticmethod
            def forward(ctx, x, weight, eps):
                return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)

            @staticmethod
            def backward(ctx, g):
                return g, None, None

        report = rulecheck(
            RMSNorm,
            (torch.randn(4, 6), torch.randn(6), 1e-5),
        )
        checked = report.checked
        self.assertIn({"x": {"shard": S(0)}, "weight": {"shard": R}}, checked)
        self.assertIn({"x": {"shard": R}, "weight": {"shard": R}}, checked)
        self.assertTrue(all(rejection.reason for rejection in report.rejected))
        rejected = [rejection.placement for rejection in report.rejected]
        self.assertIn({"x": {"shard": S(1)}, "weight": {"shard": R}}, rejected)
        self.assertIn({"x": {"shard": R}, "weight": {"shard": S(0)}}, rejected)

        class BadSoftmax(Softmax):
            @staticmethod
            def spmd_typecheck(*, x):
                return rules.einsum("...d->...d", x)

        with self.assertRaises(RuleCheckError) as cm:
            rulecheck(
                BadSoftmax,
                (torch.randn(4, 8),),
            )
        self.assertIn("under placements {x: {shard: S(1)}}", str(cm.exception))

    def test_enumeration_uses_the_group_axis_and_collectives(self):
        report = rulecheck(
            LinearAllReduce,
            (torch.randn(8, 6), torch.randn(4, 6), "tp"),
            local_tensor_mode=True,
        )
        # The only placement this kernel is correct for shards k on both
        # operands; the joint candidate reaches it.  Every single-dim sharding
        # is rejected by the hook: all_reduce(src=P) refuses a Varying result
        # that never went through a contraction.
        self.assertEqual(report.checked, [{"x": {"tp": S(1)}, "w": {"tp": S(1)}}])
        rejected = [p for p, _ in report.rejected]
        self.assertIn({"x": {"tp": S(0)}, "w": {"tp": R}}, rejected)
        self.assertIn({"x": {"tp": R}, "w": {"tp": S(0)}}, rejected)
        self.assertIn({"x": {"tp": R}, "w": {"tp": R}}, rejected)

    def test_permissive_hook_is_caught_by_enumeration(self):
        """A hook that declares src=V (an explicit reinterpret) claims the
        kernel may sum ANY varying result; enumeration finds the placement
        where that is false."""

        class Permissive(LinearAllReduce):
            @staticmethod
            def spmd_typecheck(*, x, w, group):
                return rules.all_reduce(
                    rules.einsum("mk,nk->mn", x, w),
                    group,
                    src=V,
                    dst=I,
                )

        with self.assertRaises(RuleCheckError) as cm:
            rulecheck(
                Permissive,
                (torch.randn(8, 6), torch.randn(4, 6), "tp"),
                local_tensor_mode=True,
            )
        self.assertIn("under placements {x: {tp: S(0)}, w: {tp: R}}", str(cm.exception))

    def test_enumeration_with_nothing_checkable(self):
        class Refuses(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x):
                raise SpmdTypeError("no")

            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, g):
                return g

        with self.assertRaises(RuleCheckError) as cm:
            rulecheck(Refuses, (torch.randn(4),))
        self.assertIn("rejected every enumerated placement", str(cm.exception))

    def test_placements_must_cover_every_axis(self):
        with self.assertRaises(RuleCheckError) as cm:
            rulecheck(
                Softmax,
                (torch.randn(4, 8),),
                {"x": {"dp": S(0)}},
                mesh=MESH,
            )
        self.assertIn("missing 'tp'", str(cm.exception))


class Scale2(torch.autograd.Function):
    @staticmethod
    def spmd_typecheck(*, x):
        return rules.einsum("...->...", x, linear_in=(0,))

    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, g):
        return g


class FusedMulAdd(torch.autograd.Function):
    """Three operands sharing every dim: only a joint sharding of all three
    is a valid non-replicated placement."""

    @staticmethod
    def spmd_typecheck(*, g, u, r):
        return rules.einsum("...,...,...->...", g, u, r)

    @staticmethod
    def forward(ctx, g, u, r):
        return g * u + r

    @staticmethod
    def backward(ctx, grad):
        return grad, grad, grad


class SiluLyingAboutLinearity(torch.autograd.Function):
    @staticmethod
    def spmd_typecheck(*, x):
        return rules.einsum("...->...", x, linear_in=(0,))

    @staticmethod
    def forward(ctx, x):
        return torch.nn.functional.silu(x)

    @staticmethod
    def backward(ctx, g):
        return g


def _shards(report, name):
    return [p for p in report.checked if isinstance(p[name]["shard"], Shard)]


class TestEnumerationAndDefaultReference(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_default_reference_plain(self):
        report = rulecheck(Scale2, (torch.randn(4, 6),))
        self.assertTrue(_shards(report, "x"))

    def test_default_reference_collective(self):
        report = rulecheck(
            LinearAllReduce,
            (torch.randn(4, 6), torch.randn(8, 6), "tp"),
            local_tensor_mode=True,
        )
        self.assertTrue(
            any(
                isinstance(p["x"]["tp"], Shard) and isinstance(p["w"]["tp"], Shard)
                for p in report.checked
            )
        )

    def test_three_operands_are_sharded_jointly(self):
        report = rulecheck(
            FusedMulAdd, (torch.randn(4, 6), torch.randn(4, 6), torch.randn(4, 6))
        )
        joint = [
            p
            for p in report.checked
            if all(isinstance(p[n]["shard"], Shard) for n in ("g", "u", "r"))
        ]
        self.assertEqual(len(joint), 2)  # dim 0 and dim 1
        # Sharding only one or two of them is inconsistent and rejected.
        self.assertTrue(any("inconsistently" in why for _, why in report.rejected))

    def test_partial_placements_exercise_linear_in(self):
        report = rulecheck(Scale2, (torch.randn(4, 6),))
        self.assertTrue(any(p["x"]["shard"] is P for p in report.checked))
        with self.assertRaisesRegex(RuleCheckError, "differs from the reference"):
            rulecheck(SiluLyingAboutLinearity, (torch.randn(4, 6),))

    def test_invariant_placement(self):
        report = rulecheck(Scale2, (torch.randn(4, 6),))
        self.assertTrue(any(p["x"]["shard"] is I for p in report.checked))


class TestCaseStudy(unittest.TestCase):
    def test_vocab_parallel_cross_entropy(self):
        torch.manual_seed(0)
        logits, target = torch.randn(4, 8), torch.randint(0, 8, (4,))
        report = rulecheck(
            VocabParallelCrossEntropy,
            (logits, target, "tp"),
            local_tensor_mode=True,
        )
        # The kernel all-reduces over the group, so it is only correct when the
        # vocab is what the group shards: every other placement (all
        # replicated, tokens sharded) leaves nothing Partial for the
        # all_reduce(src=P) and the hook refuses it.
        self.assertEqual(
            report.checked, [{"logits": {"tp": S(1)}, "target": {"tp": R}}]
        )
        rejected = [p for p, _ in report.rejected]
        self.assertIn({"logits": {"tp": R}, "target": {"tp": R}}, rejected)
        self.assertIn({"logits": {"tp": S(0)}, "target": {"tp": S(0)}}, rejected)


class TestMultiAxisEnumeration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_product_of_per_axis_enumerations(self):
        report = rulecheck(
            LinearAllReduce,
            (torch.randn(4, 8), torch.randn(6, 8), "tp"),  # k=8 divides dp*tp
            mesh={"dp": 2, "tp": 2},
            local_tensor_mode=True,
        )
        # Every accepted placement has k sharded on tp (the kernel all-reduces
        # over tp unconditionally); on top of that dp may shard rows of x,
        # rows of w, or k again (a second Partial axis), or be Invariant.
        self.assertTrue(report.checked)
        for p in report.checked:
            self.assertEqual((p["x"]["tp"], p["w"]["tp"]), (S(1), S(1)))
        self.assertIn(
            {"x": {"dp": S(0), "tp": S(1)}, "w": {"dp": R, "tp": S(1)}}, report.checked
        )
        self.assertIn(
            {"x": {"dp": S(1), "tp": S(1)}, "w": {"dp": S(1), "tp": S(1)}},
            report.checked,
        )
        rejected = [p for p, _ in report.rejected]
        self.assertIn({"x": {"dp": S(0), "tp": R}, "w": {"dp": R, "tp": R}}, rejected)


class TestUnconstructiblePlacements(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_partial_integer_tensor_is_a_rejection_not_an_abort(self):
        class RowScale(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, k, x):
                return rules.einsum("t,td->td", k, x)

            @staticmethod
            def forward(ctx, k, x):
                return x * k[:, None]

            @staticmethod
            def backward(ctx, g):
                return None, None

        args = (torch.randint(1, 5, (4,)), torch.randn(4, 8))
        report = rulecheck(RowScale, args)
        self.assertIn({"k": {"shard": S(0)}, "x": {"shard": S(0)}}, report.checked)
        with self.assertRaisesRegex(RuleCheckError, "integer tensor"):
            rulecheck(
                RowScale,
                args,
                {"k": {"shard": P}, "x": {"shard": R}},
                mesh={"shard": 2},
            )

    def test_hook_cannot_name_axis_outside_mesh_under_test(self):
        from spmd_types._mesh_axis import MeshAxis

        outside = MeshAxis.of(3, 7)

        class AddsOutsideAxis(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x):
                typed = rules.einsum("...->...", x)
                assert_type(typed, {outside: R})
                return typed

            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, g):
                return g

        with self.assertRaisesRegex(RuleCheckError, "not part of the mesh under test"):
            rulecheck(
                AddsOutsideAxis,
                (torch.randn(4),),
                {"x": {"shard": R}},
                mesh={"shard": 2},
            )


class TestHookDefectsAreErrors(unittest.TestCase):
    def test_hook_naming_a_non_forward_parameter_is_a_type_error(self):
        class Typo(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, xx):
                return rules.einsum("...->...", xx)

            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, g):
                return g

        with self.assertRaisesRegex(
            TypeError, "'xx' is not a parameter of Typo.forward"
        ):
            rulecheck(Typo, (torch.randn(4, 6),))

    def test_runner_error_after_acceptance_fails_the_check(self):
        """A return-form hook that accepts a placement but returns nothing is a
        defect, not a rejection, and must not leave a passing report."""

        class Forgetful(torch.autograd.Function):
            @staticmethod
            def spmd_typecheck(*, x):
                rules.ignore(x)
                return None

            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, g):
                return g

        with self.assertRaisesRegex(
            RuleCheckError, "accepted this placement but running it failed"
        ):
            rulecheck(Forgetful, (torch.randn(4, 6),))
