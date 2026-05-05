# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for SPMD type checker: type inference, strict mode, error messages.

Covers: _checker.py
"""

import logging
import re
import unittest

import expecttest
import torch
import torch.distributed as dist
from spmd_types import I, Infer, local_map, P, R, S, Scalar, set_current_mesh, V
from spmd_types._checker import (
    _trace_logger,
    _validate_partition_spec_for_global_spmd,
    assert_type,
    get_partition_spec,
    infer_local_type_for_axis,
    mutate_type,
    no_typecheck,
    OpLinearity,
    trace,
    typecheck,
)
from spmd_types._collectives import all_reduce
from spmd_types._mesh_axis import MeshAxis
from spmd_types._test_utils import LocalTensorTestCase, SpmdTypeCheckedTestCase
from spmd_types._type_attr import get_axis_local_type, get_local_type
from spmd_types.types import normalize_axis, PartitionSpec, SpmdTypeError
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore


# Module-level custom_op that simulates an opaque fused RMSNorm kernel.
# Registered at module scope so re-running tests in the same process does
# not trigger re-registration errors.
@torch.library.custom_op(
    "spmd_types_test::rmsnorm_fused_kernel_for_test",
    mutates_args=(),
)
def _rmsnorm_fused_kernel_for_test(
    x: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * w


@_rmsnorm_fused_kernel_for_test.register_fake
def _rmsnorm_fused_kernel_for_test_fake(
    x: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch.empty_like(x)


class TestEinsumTypePropagation(unittest.TestCase):
    """Test einsum type propagation rules.

    TODO: These tests depend on LTensor which was removed. They need to be
    reimplemented once the type-checking layer is rebuilt.
    """

    pass


class TestEinsumSingleOperand(unittest.TestCase):
    """Test einsum with single operand (unary operations).

    TODO: These tests depend on LTensor which was removed. They need to be
    reimplemented once the type-checking layer is rebuilt.
    """

    pass


class TestLinearTypePropagation(SpmdTypeCheckedTestCase):
    """Test type propagation through F.linear (matmul + optional bias)."""

    def test_linear_r_v_gives_v(self):
        """F.linear with R weight and V input should give V output, not R."""
        inp = self._generate_inputs((2, 4), self.pg, V)
        weight = self._generate_inputs((3, 4), self.pg, R)
        result = torch.nn.functional.linear(inp, weight)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_matmul_r_v_gives_v(self):
        """torch.matmul with R and V should give V output."""
        x = self._generate_inputs((2, 4), self.pg, R)
        y = self._generate_inputs((4, 3), self.pg, V)
        result = torch.matmul(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_add_r_v_gives_v(self):
        """torch.add with R and V should give V output."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, V)
        result = torch.add(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_add_p_p_gives_p(self):
        """torch.add with P and P should give P (addition is linear)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        result = torch.add(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_sub_p_p_gives_p(self):
        """torch.sub with P and P should give P (subtraction is linear)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        result = torch.sub(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_matmul_p_rejected(self):
        """torch.matmul with all-P is rejected (matmul is not linear in this sense)."""
        x = self._generate_inputs((2, 4), self.pg, P)
        y = self._generate_inputs((4, 3), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.matmul(x, y)

    def test_mul_p_r_gives_p(self):
        """torch.mul with P and R should give P (multilinear: P in one factor, R in other)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, R)
        result = torch.mul(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_cat_p_p_gives_p(self):
        """torch.cat with P and P should give P (cat is linear)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        result = torch.cat([x, y])
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_matmul_p_r_gives_p(self):
        """torch.matmul with P and R should give P (multilinear: P in one factor)."""
        x = self._generate_inputs((2, 4), self.pg, P)
        y = self._generate_inputs((4, 3), self.pg, R)
        result = torch.matmul(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_add_p_r_rejected(self):
        """torch.add with mixed P and R is still rejected."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.add(x, y)

    def test_add_p_scalar_rejected(self):
        """torch.add(P, scalar) is affine, not linear -- must be rejected."""
        x = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.add(x, 1.0)

    def test_sub_p_scalar_rejected(self):
        """torch.sub(P, scalar) is affine -- must be rejected."""
        x = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.sub(x, 1.0)

    def test_add_p_int_scalar_rejected(self):
        """torch.add(P, int_scalar) is also affine -- must be rejected."""
        x = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.add(x, 1)

    def test_cat_p_with_dim_arg_allowed(self):
        """torch.cat([P, P], 0) -- dim is not a tensor input, should be allowed."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        result = torch.cat([x, y], 0)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_mul_p_scalar_allowed(self):
        """torch.mul(P, scalar) -- multilinear, scalar ignored, P propagates."""
        x = self._generate_inputs((4,), self.pg, P)
        result = torch.mul(x, 2.0)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_div_i_scalar_gives_i(self):
        """I / scalar should give I (scalar adopts I type)."""
        x = self._generate_inputs((4,), self.pg, I)
        result = x / 2.0
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_add_i_scalar_gives_i(self):
        """torch.add(I, scalar) should give I (scalar adopts I type)."""
        x = self._generate_inputs((4,), self.pg, I)
        result = torch.add(x, 1.0)
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_sub_i_scalar_gives_i(self):
        """torch.sub(I, scalar) should give I (scalar adopts I type)."""
        x = self._generate_inputs((4,), self.pg, I)
        result = torch.sub(x, 1.0)
        self.assertIs(get_axis_local_type(result, self.pg), I)

    def test_add_v_scalar_gives_v(self):
        """torch.add(V, scalar) should give V (scalar adopts V type)."""
        x = self._generate_inputs((4,), self.pg, V)
        result = torch.add(x, 1.0)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_mul_v_scalar_gives_v(self):
        """torch.mul(V, scalar) should give V (scalar adopts V type)."""
        x = self._generate_inputs((4,), self.pg, V)
        result = torch.mul(x, 2.0)
        self.assertIs(get_axis_local_type(result, self.pg), V)


class TestStrictMode(SpmdTypeCheckedTestCase):
    """Test strict mode which errors on unannotated tensors."""

    def setUp(self):
        """Enter LocalTensorMode and strict typecheck for each test."""
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()
        self._type_checking_cm = typecheck(strict_mode="strict")
        self._type_checking_cm.__enter__()

    def tearDown(self):
        """Exit typecheck and LocalTensorMode after each test."""
        self._type_checking_cm.__exit__(None, None, None)
        self.mode.__exit__(None, None, None)

    def test_strict_mixed_annotated_unannotated_fails(self):
        """Strict mode raises when one operand is annotated and the other is not."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self.rank_map(lambda r: torch.randn(4))
        with self.assertRaises(SpmdTypeError):
            torch.add(x, y)

    def test_strict_mixed_cat_fails(self):
        """Strict mode catches mixed typed/untyped tensors inside lists (e.g. torch.cat)."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self.rank_map(lambda r: torch.randn(4))
        with self.assertRaises(SpmdTypeError):
            torch.cat([x, y])

    def test_strict_all_annotated_passes(self):
        """Strict mode allows operations when all tensors are annotated."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, R)
        result = torch.add(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_strict_all_v_annotated_mixed_with_unannotated_fails(self):
        """Strict mode catches all-V tensor mixed with unannotated."""
        x = self._generate_inputs((4,), self.pg, V)
        # x has _spmd_types attr (set to {self.pg: V})
        y = self.rank_map(lambda r: torch.randn(4))
        # y has no _spmd_types attr at all
        with self.assertRaises(SpmdTypeError):
            torch.add(x, y)

    def test_strict_all_unannotated_produces_empty_type(self):
        """Strict mode: all-untyped tensors ({} + {}) infer {} output."""
        x = self.rank_map(lambda r: torch.randn(4))
        y = self.rank_map(lambda r: torch.randn(4))
        result = torch.add(x, y)  # {} + {} -> {}
        self.assertEqual(get_local_type(result), {})

    def test_strict_collective_typed_input_passes(self):
        """Strict mode allows collectives when the input tensor is typed."""
        x = self._generate_inputs((4,), self.pg, P)
        result = all_reduce(x, self.pg, src=P, dst=R)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_strict_collective_untyped_input_fails(self):
        """Strict mode raises when a collective receives an unannotated tensor."""
        y = self.rank_map(lambda r: torch.randn(4))
        with self.assertRaises(SpmdTypeError):
            all_reduce(y, self.pg, src=P, dst=R)

    def test_strict_out_kwarg_unannotated_passes(self):
        """Strict mode allows unannotated out= tensor (it's just a pre-allocated destination)."""
        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, R)
        c = self.rank_map(lambda r: torch.empty(4))  # unannotated
        self.assertEqual(get_local_type(c), {})
        torch.add(a, b, out=c)  # should NOT raise
        # out= tensor gets the inferred type
        self.assertIs(get_axis_local_type(c, self.pg), R)

    def test_strict_out_kwarg_inputs_still_checked(self):
        """Strict mode still checks actual input operands even when out= is present."""
        a = self.rank_map(lambda r: torch.randn(4))  # unannotated input
        b = self._generate_inputs((4,), self.pg, R)
        c = self.rank_map(lambda r: torch.empty(4))
        with self.assertRaises(SpmdTypeError):
            torch.add(a, b, out=c)

    def test_nonstrict_mixed_passes(self):
        """Non-strict mode infers type from all inputs (untyped = {} = unknown)."""
        with typecheck(strict_mode="permissive"):
            x = self._generate_inputs((4,), self.pg, R)
            y = self.rank_map(lambda r: torch.randn(4))
            result = torch.add(x, y)  # Should not raise
            # Untyped tensor contributes {} (unknown on all axes).
            # R + {} -> R (unknown axes are compatible with any type).
            self.assertIs(get_axis_local_type(result, self.pg), R)


class TestFactoryTensors(LocalTensorTestCase):
    """Test tensors from factory ops (e.g., torch.zeros) in strict mode.

    Factory ops produce tensors with {} type (typed but unknown on all axes).
    They propagate through ops on other {} tensors, and combining with a
    tensor that has real axes raises SpmdTypeError in strict mode.
    """

    def setUp(self):
        super().setUp()
        self._strict_cm = typecheck(strict_mode="strict")
        self._strict_cm.__enter__()

    def tearDown(self):
        self._strict_cm.__exit__(None, None, None)
        super().tearDown()

    def test_factory_creates_empty_typed_tensor(self):
        """A factory op in strict mode produces a {} typed tensor."""
        from spmd_types._type_attr import get_local_type

        z = torch.zeros(4)
        self.assertEqual(get_local_type(z), {})

    def test_factory_propagates_through_ops(self):
        """Ops on {} tensors produce {} results."""
        from spmd_types._type_attr import get_local_type

        a = torch.zeros(4)
        b = torch.ones(4)
        c = a + b
        self.assertEqual(get_local_type(c), {})

    def test_factory_meets_typed_raises(self):
        """Mixing a {} tensor with a typed tensor raises (missing axis)."""
        x = self._generate_inputs((4,), self.pg, R)
        z = torch.zeros(4)
        with self.assertRaises(SpmdTypeError):
            torch.add(x, z)

    def test_factory_annotated_then_used(self):
        """assert_type on a {} tensor sets the type."""
        from spmd_types._type_attr import get_local_type

        z = torch.zeros(4)
        self.assertEqual(get_local_type(z), {})
        assert_type(z, {self.pg: R})
        self.assertIs(get_axis_local_type(z, self.pg), R)

    def test_factory_annotated_combines_with_typed(self):
        """Once annotated, a {} tensor combines with typed tensors."""
        x = self._generate_inputs((4,), self.pg, R)
        z = torch.zeros(4)
        assert_type(z, {self.pg: R})
        result = torch.add(x, z)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_truly_untyped_produces_empty_type(self):
        """Truly untyped tensors (created outside the mode) produce {} output."""
        a = self.rank_map(lambda r: torch.randn(4))
        b = self.rank_map(lambda r: torch.randn(4))
        result = torch.add(a, b)  # {} + {} -> {}
        self.assertEqual(get_local_type(result), {})

    def test_factory_chain_then_annotate(self):
        """A chain of factory ops produces {}, then assert_type works."""
        from spmd_types._type_attr import get_local_type

        a = torch.zeros(4)
        b = torch.ones(4)
        c = a + b
        d = c * 2.0
        self.assertEqual(get_local_type(d), {})
        assert_type(d, {self.pg: V})
        self.assertIs(get_axis_local_type(d, self.pg), V)

    def test_nonstrict_factory_propagates(self):
        """Factory ops produce {} typed tensors in permissive mode too."""
        from spmd_types._type_attr import get_local_type

        with typecheck(strict_mode="permissive"):
            z = torch.zeros(4)
            self.assertEqual(get_local_type(z), {})
            w = z + torch.ones(4)
            self.assertEqual(get_local_type(w), {})

    def test_nonstrict_factory_plus_typed_no_error(self):
        """Permissive mode infers from typed operands, skipping {} operands."""
        with typecheck(strict_mode="permissive"):
            x = self._generate_inputs((4,), self.pg, R)
            z = torch.zeros(4)
            result = torch.add(x, z)  # Should not raise
            # Permissive: {} operand skipped, inferred from typed operand.
            self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_factory_spmd_collective_raises(self):
        """Passing a {} tensor to an SPMD collective raises (missing axis)."""
        z = torch.zeros(4)
        with self.assertRaises(SpmdTypeError):
            all_reduce(z, self.pg, dst=R)

    def test_arange_with_scalar_propagates_type(self):
        """torch.arange with Scalar args propagates the Scalar type."""
        start = Scalar(0, {self.pg: V})
        end = Scalar(5, {self.pg: V})
        indices = torch.arange(start, end)
        self.assertIs(get_axis_local_type(indices, self.pg), V)

    def test_arange_with_scalar_then_index_typed(self):
        """torch.arange(Scalar(V)) produces V which combines with R via indexing."""
        start = Scalar(0, {self.pg: V})
        end = Scalar(5, {self.pg: V})
        indices = torch.arange(start, end)
        data = self._generate_inputs((10, 4), self.pg, R)
        result = data[indices]
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_factory_raw_collective_raises(self):
        """Passing a {} tensor to a raw dist collective raises (missing axis)."""
        import torch.distributed as dist

        pg = self.pg
        z = torch.zeros(4)
        output = torch.empty(self.WORLD_SIZE * 4)
        with self.assertRaises(SpmdTypeError):
            dist.all_gather_into_tensor(output, z, group=pg)

    def test_torch_tensor_auto_replicates_under_mesh(self):
        """``torch.tensor`` is a factory: auto-R under an active mesh."""
        mesh = init_device_mesh("cpu", (3,), mesh_dim_names=("tp",))
        with set_current_mesh(mesh):
            t = torch.tensor(0.5, dtype=torch.float32)
        tp = mesh.get_group("tp")
        self.assertIs(get_axis_local_type(t, tp), R)

    def test_torch_tensor_with_scalar_propagates_type(self):
        """``torch.tensor(Scalar)`` propagates the Scalar's annotation."""
        s = Scalar(0.5, {self.pg: V})
        t = torch.tensor(s, dtype=torch.float32)
        self.assertIs(get_axis_local_type(t, self.pg), V)


class TestFactoryTensorsPermissive(LocalTensorTestCase):
    """Test factory tensor behavior in permissive mode.

    These tests need no type checking in setUp so that truly untyped tensors
    can be created outside any mode, then tested inside permissive mode.
    """

    def test_nonstrict_truly_untyped_gets_empty_type(self):
        """In permissive mode, truly untyped tensors get {} output type."""
        # Create tensor outside any type checking -- truly untyped.
        a = torch.randn(4)
        with typecheck(strict_mode="permissive"):
            self.assertEqual(get_local_type(a), {})
            # Op on truly untyped input: result gets {} (unknown on all axes).
            b = a + 1.0
            self.assertEqual(get_local_type(b), {})

    def test_nonstrict_typed_plus_truly_untyped_produces_empty(self):
        """In permissive mode, {} (typed) + truly untyped ({}) = {} output."""
        # Create tensor outside any type checking -- truly untyped.
        a = torch.randn(4)
        with typecheck(strict_mode="permissive"):
            z = torch.zeros(4)  # {} typed (created inside mode)
            self.assertEqual(get_local_type(z), {})
            self.assertEqual(get_local_type(a), {})
            result = a + z
            # Both contribute {}: {} + {} -> {}.
            self.assertEqual(get_local_type(result), {})

    def test_factory_spmd_collective_permissive_propagates(self):
        """Permissive mode propagates {} tensor through SPMD collectives."""
        with typecheck(strict_mode="permissive"):
            z = torch.zeros(4)
            # {} tensor has no type for the collective's axis.
            # Permissive mode skips validation but still propagates dst type.
            all_reduce(z, self.pg, dst=R)


class TestAssertTypeRefinement(LocalTensorTestCase):
    """Test that assert_type merges non-overlapping axes (refinement)."""

    WORLD_SIZE = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))
        cls.tp = mesh.get_group("tp")
        cls.dp = mesh.get_group("dp")

    def test_refine_disjoint_axes(self):
        """assert_type on new axes merges into existing type."""
        x = torch.randn(4)
        assert_type(x, {self.dp: R})
        assert_type(x, {self.tp: I})
        self.assertIs(get_axis_local_type(x, self.dp), R)
        self.assertIs(get_axis_local_type(x, self.tp), I)

    def test_refine_matching_axis_ok(self):
        """assert_type on an existing axis with the same type is fine."""
        x = torch.randn(4)
        assert_type(x, {self.dp: R})
        assert_type(x, {self.dp: R})  # same type, no error
        self.assertIs(get_axis_local_type(x, self.dp), R)

    def test_refine_conflicting_axis_raises(self):
        """assert_type on an existing axis with a different type raises."""
        x = torch.randn(4)
        assert_type(x, {self.dp: R})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.dp: V})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.dp: P})

    def test_refine_partial_overlap(self):
        """assert_type with both new and existing axes works."""
        x = torch.randn(4)
        assert_type(x, {self.dp: R})
        assert_type(x, {self.dp: R, self.tp: V})
        self.assertIs(get_axis_local_type(x, self.dp), R)
        self.assertIs(get_axis_local_type(x, self.tp), V)


class TestAssertTypeVtoP(LocalTensorTestCase):
    """Test that assert_type can transmute V -> P implicitly."""

    def test_v_then_p_transmutes(self):
        """assert_type({TP: P}) on a V tensor should transmute V to P."""
        x = self.rank_map(lambda r: torch.randn(4))
        assert_type(x, {self.pg: V})
        # V -> P should be allowed as an implicit transmute.
        assert_type(x, {self.pg: P})
        self.assertIs(get_axis_local_type(x, self.pg), P)

    def test_p_then_v_rejected(self):
        """assert_type({TP: V}) on a P tensor should still be rejected."""
        x = self.rank_map(lambda r: torch.randn(4))
        assert_type(x, {self.pg: P})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.pg: V})


class TestAssertTypeBareExpansion(LocalTensorTestCase):
    """Test assert_type with bare R/V (PerMeshAxisLocalSpmdType) expansion."""

    WORLD_SIZE = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))

    def test_bare_r_expands_to_all_mesh_axes(self):
        """assert_type(tensor, R) with active mesh sets R on all axes."""
        x = torch.randn(4)
        with set_current_mesh(self.mesh):
            assert_type(x, R)
        dp = self.mesh.get_group("dp")
        tp = self.mesh.get_group("tp")
        self.assertIs(get_axis_local_type(x, dp), R)
        self.assertIs(get_axis_local_type(x, tp), R)

    def test_bare_v_expands_to_all_mesh_axes(self):
        """assert_type(tensor, V) with active mesh sets V on all axes."""
        x = torch.randn(4)
        with set_current_mesh(self.mesh):
            assert_type(x, V)
        dp = self.mesh.get_group("dp")
        tp = self.mesh.get_group("tp")
        self.assertIs(get_axis_local_type(x, dp), V)
        self.assertIs(get_axis_local_type(x, tp), V)

    def test_bare_r_no_mesh_raises(self):
        """assert_type(tensor, R) without active mesh raises SpmdTypeError."""
        x = torch.randn(4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, R)


class TestTypeErrorMessages(LocalTensorTestCase, expecttest.TestCase):
    """Test that type errors include actionable fix suggestions.

    Tests call infer_local_type_for_axis directly rather than going through
    torch ops to test specific axis-type combinations in isolation.
    """

    WORLD_SIZE = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))
        cls.tp = mesh.get_group("tp")

    def test_general_I_R(self):
        """I + R: suggest reinterpret I->R (I->V filtered: changes output from R to V)."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis(self.tp, [I, R])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis mesh_tp cannot mix with other types. Found types: [I, R]
Are you missing a collective or a reinterpret/convert call? e.g.,
  convert(tensor, mesh_tp, src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)""",
        )

    def test_general_I_V(self):
        """I + V: suggest convert I->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis(self.tp, [I, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis mesh_tp cannot mix with other types. Found types: [I, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  convert(tensor, mesh_tp, src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)""",
        )

    def test_general_I_P(self):
        """I + P: suggest convert I->P or all_reduce P->I."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis(self.tp, [I, P])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis mesh_tp cannot mix with other types. Found types: [I, P]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, mesh_tp, src=P, dst=I) on the Partial operand (all-reduce in forward, no-op backward)""",
        )

    def test_general_P_R(self):
        """P + R: suggest all_reduce P->R or convert R->P."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis(self.tp, [P, R])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis mesh_tp cannot propagate through non-linear ops (non-linear of a partial sum != partial sum of non-linear). Reduce first with all_reduce or reduce_scatter. Found types: [P, R]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, mesh_tp, src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_general_P_V(self):
        """P + V: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis(self.tp, [P, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis mesh_tp cannot combine with Varying. Reduce Partial first (all_reduce -> R, or reduce_scatter -> V). Found types: [P, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, mesh_tp, src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_general_R_P_V(self):
        """R + P + V: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis(self.tp, [R, P, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis mesh_tp cannot combine with Varying. Reduce Partial first (all_reduce -> R, or reduce_scatter -> V). Found types: [R, P, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, mesh_tp, src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_general_I_R_V(self):
        """I + R + V: suggest convert I->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis(self.tp, [I, R, V])
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis mesh_tp cannot mix with other types. Found types: [I, R, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  convert(tensor, mesh_tp, src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)""",
        )

    def test_mul_P_P(self):
        """P * P: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis(
                self.tp, [P, P], linearity=OpLinearity.MULTILINEAR
            )
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial in multiple factors of multilinear op on axis mesh_tp is forbidden. Reduce all but one factor first. Found types: [P, P]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, mesh_tp, src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_mul_P_V(self):
        """P * V: suggest all_reduce P->R."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis(
                self.tp, [P, V], linearity=OpLinearity.MULTILINEAR
            )
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis mesh_tp cannot combine with Varying. Reduce Partial first (all_reduce -> R, or reduce_scatter -> V). Found types: [P, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, mesh_tp, src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)""",
        )

    def test_mul_P_I(self):
        """P * I: suggest reinterpret I->R or all_reduce P->I."""
        with self.assertRaises(SpmdTypeError) as ctx:
            infer_local_type_for_axis(
                self.tp, [P, I], linearity=OpLinearity.MULTILINEAR
            )
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Invariant type on axis mesh_tp cannot mix with other types. Found types: [P, I]
Are you missing a collective or a reinterpret/convert call? e.g.,
  convert(tensor, mesh_tp, src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)
  all_reduce(tensor, mesh_tp, src=P, dst=I) on the Partial operand (all-reduce in forward, no-op backward)""",
        )


class TestOpLinearity(LocalTensorTestCase):
    """Test the OpLinearity parameter on infer_local_type_for_axis."""

    def test_all_p_linear_allowed(self):
        """All-P with LINEAR should return P."""
        result = infer_local_type_for_axis(
            self.pg, [P, P], linearity=OpLinearity.LINEAR
        )
        self.assertIs(result, P)

    def test_all_p_nonlinear_rejected(self):
        """All-P with NONLINEAR (default) should raise."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis(self.pg, [P, P])

    def test_mixed_p_r_rejected_even_linear(self):
        """Mixed P+R is rejected even with LINEAR."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis(self.pg, [P, R], linearity=OpLinearity.LINEAR)

    def test_all_r_unaffected_by_linearity(self):
        """All-R works regardless of linearity."""
        self.assertIs(infer_local_type_for_axis(self.pg, [R, R]), R)
        self.assertIs(
            infer_local_type_for_axis(self.pg, [R, R], linearity=OpLinearity.LINEAR), R
        )

    def test_multilinear_p_r_gives_p(self):
        """MULTILINEAR with P and R should return P."""
        result = infer_local_type_for_axis(
            self.pg, [P, R], linearity=OpLinearity.MULTILINEAR
        )
        self.assertIs(result, P)

    def test_multilinear_p_p_rejected(self):
        """MULTILINEAR with P*P should raise."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis(
                self.pg, [P, P], linearity=OpLinearity.MULTILINEAR
            )

    def test_multilinear_p_v_rejected(self):
        """MULTILINEAR with P and V should raise."""
        with self.assertRaises(SpmdTypeError):
            infer_local_type_for_axis(
                self.pg, [P, V], linearity=OpLinearity.MULTILINEAR
            )


class TestAssertTypeShardSugar(LocalTensorTestCase):
    """Test assert_type with S(i) sugar and PartitionSpec."""

    WORLD_SIZE = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))
        cls.tp = mesh.get_group("tp")
        cls.dp = mesh.get_group("dp")

    def test_shard_sugar_sets_varying(self):
        """S(i) in types should set the axis to V in local types."""
        x = torch.randn(4)
        assert_type(x, {self.tp: S(0)})
        self.assertIs(get_axis_local_type(x, self.tp), V)

    def test_shard_sugar_duplicate_dim_rejected(self):
        """S(i) on multiple axes at the same dim requires explicit PartitionSpec."""
        x = torch.randn(4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.dp: S(0), self.tp: S(0)})

    def test_shard_sugar_negative_dim(self):
        """S(-1) should resolve to the last dimension."""
        x = torch.randn(3, 4)
        assert_type(x, {self.tp: S(-1)})
        self.assertIs(get_axis_local_type(x, self.tp), V)

    def test_shard_sugar_with_partition_spec_rejected(self):
        """S(i) in types and explicit partition_spec are mutually exclusive."""
        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: S(1)}, PartitionSpec(None, self.tp))

    def test_partition_spec_sets_varying(self):
        """Axes in partition_spec should be treated as Varying."""
        x = torch.randn(3, 4)
        assert_type(x, {}, PartitionSpec(None, self.tp))
        self.assertIs(get_axis_local_type(x, self.tp), V)

    def test_partition_spec_wrong_length_rejected(self):
        """PartitionSpec length must match tensor ndim."""
        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {}, PartitionSpec(self.tp))

    def test_partition_spec_conflicts_with_non_varying_rejected(self):
        """Axis in partition_spec that is R/I/P in types should be rejected."""
        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: R}, PartitionSpec(None, self.tp))

    def test_partition_spec_with_explicit_v_ok(self):
        """Explicit V in types for an axis in partition_spec should be allowed."""
        from spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        assert_type(x, {self.dp: V}, PartitionSpec(None, None))
        self.assertIs(get_axis_local_type(x, self.dp), V)

    def test_mixed_shard_and_local_types(self):
        """S(i) for one axis and R/I/P for another should work."""
        x = torch.randn(3, 4)
        assert_type(x, {self.dp: S(0), self.tp: I})
        self.assertIs(get_axis_local_type(x, self.dp), V)
        self.assertIs(get_axis_local_type(x, self.tp), I)

    def test_partition_spec_with_non_varying_types(self):
        """partition_spec with R/I axes in types should work if they don't overlap."""
        from spmd_types.types import PartitionSpec

        x = torch.randn(3, 4)
        assert_type(x, {self.tp: I}, PartitionSpec(self.dp, None))
        self.assertIs(get_axis_local_type(x, self.tp), I)
        self.assertIs(get_axis_local_type(x, self.dp), V)

    def test_shard_sugar_out_of_bounds_rejected(self):
        """S(i) with out-of-bounds dim is an error."""
        x = torch.randn(3, 4)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: S(2)})
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: S(-3)})

    def test_assert_type_accepts_both_pg_and_meshaxis(self):
        """assert_type works with both ProcessGroup and MeshAxis keys."""
        x = torch.randn(3, 4)
        ma = normalize_axis(self.tp)
        assert_type(x, {self.tp: V})
        assert_type(x, {self.tp: S(0)})
        assert_type(x, {ma: S(0)})

    def test_shard_sugar_0d_tensor_rejected(self):
        """S(i) on a 0-d tensor is out of bounds."""
        x = torch.tensor(1.0)
        with self.assertRaises(SpmdTypeError):
            assert_type(x, {self.tp: S(0)})


class TestOpLinearityRegistry(SpmdTypeCheckedTestCase):
    """Test that the expanded _OP_REGISTRY covers aliases, dunders, and structural ops."""

    def test_clone_functional_form(self):
        """torch.clone(P) should give P (functional form, not just Tensor.clone)."""
        x = self._generate_inputs((4,), self.pg, P)
        result = torch.clone(x)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_concat_aliases(self):
        """torch.concat and torch.concatenate should propagate P like torch.cat."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        for fn in [torch.concat, torch.concatenate]:
            result = fn([x, y])
            self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_structural_ops_preserve_p(self):
        """Structural ops (reshape, view, transpose, etc.) should preserve P."""
        x = self._generate_inputs((2, 3), self.pg, P)
        # NB: contiguous is registered but untestable here -- LocalTensorMode
        # decomposes it to per-rank calls that bypass type checking.
        ops = [
            ("reshape", lambda t: torch.reshape(t, (6,))),
            ("Tensor.view", lambda t: t.view(6)),
            ("transpose", lambda t: torch.transpose(t, 0, 1)),
            ("Tensor.T", lambda t: t.T),
            ("squeeze", lambda t: torch.squeeze(t.unsqueeze(0))),
            ("unsqueeze", lambda t: torch.unsqueeze(t, 0)),
            ("flatten", lambda t: torch.flatten(t)),
            ("expand", lambda t: t.unsqueeze(0).expand(2, 2, 3)),
            ("permute", lambda t: torch.permute(t, (1, 0))),
        ]
        for name, op in ops:
            result = op(x)
            self.assertIs(
                get_axis_local_type(result, self.pg),
                P,
                f"{name} did not preserve P type",
            )

    def test_tensor_T_property_preserves_v(self):
        """Tensor.T (property descriptor, not method) should preserve type.

        Regression: ``.T`` is a property accessor that bypasses
        ``__torch_function__`` in some PyTorch versions, causing the
        SPMD type to be silently dropped. ``.transpose()`` propagates
        correctly; ``.T`` should match.
        """
        x = self._generate_inputs((2, 3), self.pg, V)
        result = x.T
        self.assertIs(
            get_axis_local_type(result, self.pg),
            V,
            "Tensor.T property did not preserve V type",
        )

    def test_neg_preserves_p(self):
        """torch.neg(P) and -P (unary neg operator) should both give P."""
        x = self._generate_inputs((4,), self.pg, P)
        result_func = torch.neg(x)
        self.assertIs(get_axis_local_type(result_func, self.pg), P)
        result_op = -x
        self.assertIs(get_axis_local_type(result_op, self.pg), P)

    def test_operator_add_p_p(self):
        """P + P via the + operator should give P."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        result = x + y
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_split_chunk_unbind_preserve_p(self):
        """Multi-output ops (split, chunk, unbind) should propagate P."""
        x = self._generate_inputs((6,), self.pg, P)
        for name, fn in [
            ("split", lambda t: torch.split(t, 3)),
            ("chunk", lambda t: torch.chunk(t, 2)),
            ("unbind", lambda t: torch.unbind(t.view(2, 3))),
        ]:
            results = fn(x)
            for i, r in enumerate(results):
                self.assertIs(
                    get_axis_local_type(r, self.pg),
                    P,
                    f"{name}[{i}] did not preserve P type",
                )

    def test_inplace_add_preserves_p(self):
        """P.add_(P) should give P."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, self.pg), P)

    def test_div_p_r_gives_p(self):
        """torch.div(P, R) should give P (linear in numerator when denominator is fixed)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, R)
        result = torch.div(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_div_r_p_rejected(self):
        """torch.div(R, P) should raise (P in denominator is not linear)."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.div(x, y)

    def test_div_p_p_rejected(self):
        """torch.div(P, P) should raise (P in denominator is not linear)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.div(x, y)

    def test_div_r_v_gives_v(self):
        """torch.div(R, V) should give V (normal non-P path, no fixed_args filtering)."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, V)
        result = torch.div(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_all_registered_ops_are_callable(self):
        """Every key in _OP_REGISTRY should be a callable (no stale references)."""
        from spmd_types._checker import _OP_REGISTRY

        for func in _OP_REGISTRY:
            self.assertTrue(
                callable(func),
                f"Registry entry {func!r} is not callable",
            )

    def test_all_decomp_rules_are_callable(self):
        """Every key in _DECOMP_TYPE_RULES should be a callable."""
        from spmd_types._checker import _DECOMP_TYPE_RULES

        for func in _DECOMP_TYPE_RULES:
            self.assertTrue(
                callable(func),
                f"Decomp rule entry {func!r} is not callable",
            )


class TestAddmmTypeDecomposition(SpmdTypeCheckedTestCase):
    """Test type-level decomposition for addmm and related ops.

    addmm(self, mat1, mat2) = self + mm(mat1, mat2) decomposes into a sum
    (LINEAR) of self and a matmul (MULTILINEAR) of mat1 and mat2. This catches
    bugs where e.g. addmm(P, R, R) was incorrectly accepted as P.
    """

    # --- addmm ---

    def test_addmm_P_R_R_rejected(self):
        """addmm(P, R, R) must error: R@R=R, then P+R is invalid."""
        self_t = self._generate_inputs((2, 3), self.pg, P)
        mat1 = self._generate_inputs((2, 4), self.pg, R)
        mat2 = self._generate_inputs((4, 3), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_R_P_R_rejected(self):
        """addmm(R, P, R) must error: P@R=P, then P+R is invalid."""
        self_t = self._generate_inputs((2, 3), self.pg, R)
        mat1 = self._generate_inputs((2, 4), self.pg, P)
        mat2 = self._generate_inputs((4, 3), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_R_R_P_rejected(self):
        """addmm(R, R, P) must error: R@P=P, then P+R is invalid."""
        self_t = self._generate_inputs((2, 3), self.pg, R)
        mat1 = self._generate_inputs((2, 4), self.pg, R)
        mat2 = self._generate_inputs((4, 3), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_P_P_R_gives_P(self):
        """addmm(P, P, R): mm(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((2, 3), self.pg, P)
        mat1 = self._generate_inputs((2, 4), self.pg, P)
        mat2 = self._generate_inputs((4, 3), self.pg, R)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_addmm_P_R_P_gives_P(self):
        """addmm(P, R, P): mm(R,P)=P, then P+P=P."""
        self_t = self._generate_inputs((2, 3), self.pg, P)
        mat1 = self._generate_inputs((2, 4), self.pg, R)
        mat2 = self._generate_inputs((4, 3), self.pg, P)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_addmm_R_R_R_gives_R(self):
        """addmm(R, R, R) -> R (no change from before)."""
        self_t = self._generate_inputs((2, 3), self.pg, R)
        mat1 = self._generate_inputs((2, 4), self.pg, R)
        mat2 = self._generate_inputs((4, 3), self.pg, R)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_addmm_V_R_R_gives_V(self):
        """addmm(V, R, R): mm(R,R)=R, then V+R=V."""
        self_t = self._generate_inputs((2, 3), self.pg, V)
        mat1 = self._generate_inputs((2, 4), self.pg, R)
        mat2 = self._generate_inputs((4, 3), self.pg, R)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_addmm_R_P_P_rejected(self):
        """addmm(R, P, P): mm(P,P) is invalid (two P factors in multilinear)."""
        self_t = self._generate_inputs((2, 3), self.pg, R)
        mat1 = self._generate_inputs((2, 4), self.pg, P)
        mat2 = self._generate_inputs((4, 3), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            torch.addmm(self_t, mat1, mat2)

    def test_addmm_V_V_V_gives_V(self):
        """addmm(V, V, V): mm(V,V)=V, then V+V=V."""
        self_t = self._generate_inputs((2, 3), self.pg, V)
        mat1 = self._generate_inputs((2, 4), self.pg, V)
        mat2 = self._generate_inputs((4, 3), self.pg, V)
        result = torch.addmm(self_t, mat1, mat2)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    # --- addmv ---

    def test_addmv_P_R_R_rejected(self):
        """addmv(P, R, R) must error."""
        self_t = self._generate_inputs((3,), self.pg, P)
        mat = self._generate_inputs((3, 4), self.pg, R)
        vec = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.addmv(self_t, mat, vec)

    def test_addmv_P_P_R_gives_P(self):
        """addmv(P, P, R): mv(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((3,), self.pg, P)
        mat = self._generate_inputs((3, 4), self.pg, P)
        vec = self._generate_inputs((4,), self.pg, R)
        result = torch.addmv(self_t, mat, vec)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_addmv_R_R_R_gives_R(self):
        """addmv(R, R, R) -> R."""
        self_t = self._generate_inputs((3,), self.pg, R)
        mat = self._generate_inputs((3, 4), self.pg, R)
        vec = self._generate_inputs((4,), self.pg, R)
        result = torch.addmv(self_t, mat, vec)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    # --- baddbmm ---

    def test_baddbmm_P_R_R_rejected(self):
        """baddbmm(P, R, R) must error."""
        self_t = self._generate_inputs((2, 3, 5), self.pg, P)
        batch1 = self._generate_inputs((2, 3, 4), self.pg, R)
        batch2 = self._generate_inputs((2, 4, 5), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.baddbmm(self_t, batch1, batch2)

    def test_baddbmm_P_P_R_gives_P(self):
        """baddbmm(P, P, R): bmm(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((2, 3, 5), self.pg, P)
        batch1 = self._generate_inputs((2, 3, 4), self.pg, P)
        batch2 = self._generate_inputs((2, 4, 5), self.pg, R)
        result = torch.baddbmm(self_t, batch1, batch2)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_baddbmm_R_R_R_gives_R(self):
        """baddbmm(R, R, R) -> R."""
        self_t = self._generate_inputs((2, 3, 5), self.pg, R)
        batch1 = self._generate_inputs((2, 3, 4), self.pg, R)
        batch2 = self._generate_inputs((2, 4, 5), self.pg, R)
        result = torch.baddbmm(self_t, batch1, batch2)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    # --- addr ---

    def test_addr_P_R_R_rejected(self):
        """addr(P, R, R) must error."""
        self_t = self._generate_inputs((3, 4), self.pg, P)
        vec1 = self._generate_inputs((3,), self.pg, R)
        vec2 = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            torch.addr(self_t, vec1, vec2)

    def test_addr_P_P_R_gives_P(self):
        """addr(P, P, R): outer(P,R)=P, then P+P=P."""
        self_t = self._generate_inputs((3, 4), self.pg, P)
        vec1 = self._generate_inputs((3,), self.pg, P)
        vec2 = self._generate_inputs((4,), self.pg, R)
        result = torch.addr(self_t, vec1, vec2)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_addr_R_R_R_gives_R(self):
        """addr(R, R, R) -> R."""
        self_t = self._generate_inputs((3, 4), self.pg, R)
        vec1 = self._generate_inputs((3,), self.pg, R)
        vec2 = self._generate_inputs((4,), self.pg, R)
        result = torch.addr(self_t, vec1, vec2)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    # --- sum/mean (LINEAR reductions) ---

    def test_sum_P_gives_P(self):
        """torch.sum(P) -> P (sum is linear)."""
        x = self._generate_inputs((2, 3), self.pg, P)
        result = torch.sum(x)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_mean_P_gives_P(self):
        """torch.mean(P) -> P (mean is linear)."""
        # randn already produces float32 so torch.mean works directly
        x = self._generate_inputs((2, 3), self.pg, P)
        result = torch.mean(x)
        self.assertIs(get_axis_local_type(result, self.pg), P)


class TestScalarWrapper(SpmdTypeCheckedTestCase):
    """Test the Scalar wrapper for annotating Python scalars with SPMD types."""

    def test_scalar_get_local_type(self):
        """get_local_type returns the stored type."""
        s = Scalar(1.0, {self.pg: R})
        self.assertEqual(get_axis_local_type(s, self.pg), R)

    def test_add_r_scalar_v(self):
        """torch.add(R, Scalar(V)) -> V.

        With a plain 1.0, R + scalar -> R. With Scalar(1.0, {tp: V}),
        R + V -> V. This catches rank-dependent scalars.
        """
        x = self._generate_inputs((4,), self.pg, R)
        s = Scalar(1.0, {self.pg: V})
        result = torch.add(x, s)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_add_v_scalar_v(self):
        """torch.add(V, Scalar(V)) -> V."""
        x = self._generate_inputs((4,), self.pg, V)
        s = Scalar(1.0, {self.pg: V})
        result = torch.add(x, s)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_mul_p_scalar_r(self):
        """torch.mul(P, Scalar(R)) -> P (multilinear)."""
        x = self._generate_inputs((4,), self.pg, P)
        s = Scalar(2.0, {self.pg: R})
        result = torch.mul(x, s)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_add_p_scalar_p(self):
        """torch.add(P, Scalar(P)) -> P (linear)."""
        x = self._generate_inputs((4,), self.pg, P)
        s = Scalar(1.0, {self.pg: P})
        result = torch.add(x, s)
        self.assertIs(get_axis_local_type(result, self.pg), P)

    def test_add_r_scalar_p_error(self):
        """torch.add(R, Scalar(P)) -> SpmdTypeError (P+R is affine).

        With a plain 1.0, R + scalar -> R. With Scalar(local_sum, {tp: P}),
        R + P is invalid (affine, not linear).
        """
        x = self._generate_inputs((4,), self.pg, R)
        s = Scalar(1.0, {self.pg: P})
        with self.assertRaises(SpmdTypeError):
            torch.add(x, s)

    def test_add_i_scalar_v_error(self):
        """torch.add(I, Scalar(V)) -> SpmdTypeError (I can't mix)."""
        x = self._generate_inputs((4,), self.pg, I)
        s = Scalar(1.0, {self.pg: V})
        with self.assertRaises(SpmdTypeError):
            torch.add(x, s)

    def test_narrow_r_scalar_v_gives_v(self):
        """torch.narrow(R, dim, Scalar(V), Scalar(V)) -> V.

        narrow is LINEAR with only tensor_args=(0,), so the dim/start/length
        args are not checked by _collect_scalar_types.  But if start or length
        is Scalar(V), the output varies across ranks and must be V, not R.
        """
        x = self._generate_inputs((8,), self.pg, R)
        start = Scalar(0, {self.pg: V})
        length = Scalar(2, {self.pg: V})
        result = torch.narrow(x, 0, start, length)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_narrow_p_scalar_v_error(self):
        """torch.narrow(P, dim, Scalar(V), Scalar(V)) -> SpmdTypeError.

        P + V is always invalid.
        """
        x = self._generate_inputs((8,), self.pg, P)
        start = Scalar(0, {self.pg: V})
        length = Scalar(2, {self.pg: V})
        with self.assertRaises(SpmdTypeError):
            torch.narrow(x, 0, start, length)

    def test_narrow_r_scalar_r_gives_r(self):
        """torch.narrow(R, dim, Scalar(R), Scalar(R)) -> R.

        Replicated scalars at non-tensor-arg positions don't change the type.
        """
        x = self._generate_inputs((8,), self.pg, R)
        start = Scalar(0, {self.pg: R})
        length = Scalar(2, {self.pg: R})
        result = torch.narrow(x, 0, start, length)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_scalar_without_type_mode(self):
        """Scalar unwraps and computes without type checking."""
        from spmd_types._checker import no_typecheck

        with no_typecheck():
            x = torch.tensor([1.0, 2.0, 3.0])
            s = Scalar(2.0, {self.pg: V})
            result = torch.mul(x, s)
            expected = torch.tensor([2.0, 4.0, 6.0])
            torch.testing.assert_close(result, expected)

    def test_scalar_drops_singleton_axis(self):
        """Scalar.__init__ drops size-1 axes, matching _validate behavior."""
        import torch.distributed as dist

        singleton_pg = dist.new_group(ranks=[0])
        s = Scalar(1.0, {self.pg: V, singleton_pg: V})
        local_type = s.local_type
        # The size-1 axis should be dropped.
        self.assertEqual(len(local_type), 1)
        self.assertIs(get_axis_local_type(s, self.pg), V)

    def test_scalar_all_singleton_gives_empty_type(self):
        """Scalar with only size-1 axes produces an empty type dict."""
        import torch.distributed as dist

        singleton_pg = dist.new_group(ranks=[0])
        s = Scalar(42, {singleton_pg: V})
        self.assertEqual(s.local_type, {})

    def test_narrow_r_scalar_v_with_singleton_axis(self):
        """narrow(R, Scalar(V)) works when Scalar also has a size-1 axis.

        This is the HSDP DDP=1 scenario: tensor has type {shard_pg: R} and
        Scalar has {shard_pg: V, replicate_pg: V} where replicate_pg has
        size 1. Without dropping the singleton axis in Scalar, the key sets
        mismatch and type checking fails.
        """
        import torch.distributed as dist

        singleton_pg = dist.new_group(ranks=[0])
        x = self._generate_inputs((8,), self.pg, R)
        start = Scalar(0, {self.pg: V, singleton_pg: V})
        length = Scalar(2, {self.pg: V, singleton_pg: V})
        result = torch.narrow(x, 0, start, length)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_scalar_arithmetic_drops_singleton_axis(self):
        """Scalar arithmetic propagates only non-singleton axes."""
        import torch.distributed as dist

        singleton_pg = dist.new_group(ranks=[0])
        a = Scalar(3, {self.pg: V, singleton_pg: V})
        b = Scalar(4, {self.pg: R, singleton_pg: R})
        result = a + b
        self.assertIsInstance(result, Scalar)
        local_type = result.local_type
        # Only the non-singleton axis should survive.
        self.assertEqual(len(local_type), 1)
        self.assertIs(local_type[normalize_axis(self.pg)], V)


class TestThreadLocalState(SpmdTypeCheckedTestCase):
    """Test thread-local state helpers: is_type_checking, no_typecheck."""

    def test_is_type_checking_inside_context(self):
        """is_type_checking() returns True inside typecheck context."""
        from spmd_types._state import is_type_checking

        self.assertTrue(is_type_checking())

    def test_no_typecheck_pauses(self):
        """no_typecheck() makes is_type_checking() return False."""
        from spmd_types._checker import no_typecheck
        from spmd_types._state import is_type_checking

        self.assertTrue(is_type_checking())
        with no_typecheck():
            self.assertFalse(is_type_checking())
        self.assertTrue(is_type_checking())

    def test_no_typecheck_noop_outside(self):
        """no_typecheck() is a no-op when no mode is active."""
        from spmd_types._checker import no_typecheck
        from spmd_types._state import is_type_checking

        # Temporarily exit type checking entirely to test the no-op case.
        self._type_checking_cm.__exit__(None, None, None)
        try:
            self.assertFalse(is_type_checking())
            with no_typecheck():
                self.assertFalse(is_type_checking())
        finally:
            # Re-enter for tearDown.
            self._type_checking_cm = typecheck()
            self._type_checking_cm.__enter__()

    def test_reentrant_noop(self):
        """typecheck() inside active mode is a no-op (does not nest modes)."""
        from spmd_types._state import _current_mode, is_type_checking

        outer = _current_mode()
        self.assertIsNotNone(outer)
        with typecheck():
            self.assertIs(_current_mode(), outer)
        self.assertTrue(is_type_checking())
        self.assertIs(_current_mode(), outer)

    def test_typecheck_inside_no_typecheck_reuses_mode(self):
        """typecheck() inside no_typecheck() reuses the existing mode."""
        from spmd_types._checker import no_typecheck
        from spmd_types._state import _current_mode, is_type_checking

        outer = _current_mode()
        self.assertIsNotNone(outer)
        with no_typecheck():
            self.assertFalse(is_type_checking())
            with typecheck():
                # Should reuse the outer mode, not create a new one.
                self.assertIs(_current_mode(), outer)
                self.assertTrue(is_type_checking())
                # Type inference should work inside the inner typecheck.
                x = self._generate_inputs((4,), self.pg, R)
                y = self._generate_inputs((4,), self.pg, V)
                result = torch.add(x, y)
                self.assertIs(get_axis_local_type(result, self.pg), V)
            self.assertFalse(is_type_checking())

    def test_typecheck_resumes_inside_no_typecheck(self):
        """typecheck() re-enables checking inside no_typecheck()."""
        from spmd_types._checker import no_typecheck
        from spmd_types._state import is_type_checking

        self.assertTrue(is_type_checking())
        with no_typecheck():
            self.assertFalse(is_type_checking())
            with typecheck():
                self.assertTrue(is_type_checking())
                # Type inference should work.
                x = self._generate_inputs((4,), self.pg, R)
                y = self._generate_inputs((4,), self.pg, V)
                result = torch.add(x, y)
                self.assertIs(get_axis_local_type(result, self.pg), V)
            self.assertFalse(is_type_checking())
        self.assertTrue(is_type_checking())

    def test_typecheck_inherits_strict_mode(self):
        """typecheck() without strict_mode inherits the current strict mode."""
        from spmd_types._checker import no_typecheck
        from spmd_types._state import _current_mode

        # Outer mode is "strict" (the default from setUp).
        with no_typecheck():
            with typecheck():
                mode = _current_mode()
                self.assertIsNotNone(mode)
                self.assertTrue(mode._strict)

        # Now test with permissive outer mode.
        self._type_checking_cm.__exit__(None, None, None)
        try:
            self._type_checking_cm = typecheck(strict_mode="permissive")
            self._type_checking_cm.__enter__()
            with no_typecheck():
                with typecheck():
                    mode = _current_mode()
                    self.assertIsNotNone(mode)
                    self.assertFalse(mode._strict)
        finally:
            self._type_checking_cm.__exit__(None, None, None)
            self._type_checking_cm = typecheck()
            self._type_checking_cm.__enter__()

    def test_typecheck_resume_noop_when_not_disabled(self):
        """typecheck() without args is a no-op when checking is already active."""
        from spmd_types._state import is_type_checking

        self.assertTrue(is_type_checking())
        with typecheck():
            self.assertTrue(is_type_checking())
            x = self._generate_inputs((4,), self.pg, R)
            y = self._generate_inputs((4,), self.pg, R)
            result = torch.add(x, y)
            self.assertIs(get_axis_local_type(result, self.pg), R)
        self.assertTrue(is_type_checking())


class TestMutationTypeChecking(SpmdTypeCheckedTestCase):
    """Test that in-place/out operations validate SPMD type consistency.

    Mutating operations (trailing underscore, out= kwarg, result-is-input
    identity) must not silently change a tensor's SPMD type. For example,
    x.add_(y) where x=R and y=V would infer V output, but since x is R
    and is being mutated, this is an error.
    """

    # --- In-place ops that preserve type (should pass) ---

    def test_inplace_add_r_r_ok(self):
        """R.add_(R) -> R: type preserved, no error."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, R)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, self.pg), R)

    def test_inplace_add_v_v_ok(self):
        """V.add_(V) -> V: type preserved, no error."""
        x = self._generate_inputs((4,), self.pg, V)
        y = self._generate_inputs((4,), self.pg, V)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, self.pg), V)

    def test_inplace_add_v_r_ok(self):
        """V.add_(R) -> V: output is V, self is V, type preserved."""
        x = self._generate_inputs((4,), self.pg, V)
        y = self._generate_inputs((4,), self.pg, R)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, self.pg), V)

    def test_inplace_mul_v_scalar_ok(self):
        """V.mul_(2.0) -> V: type preserved."""
        x = self._generate_inputs((4,), self.pg, V)
        x.mul_(2.0)
        self.assertIs(get_axis_local_type(x, self.pg), V)

    def test_inplace_neg_r_ok(self):
        """R.neg_() -> R: type preserved."""
        x = self._generate_inputs((4,), self.pg, R)
        x.neg_()
        self.assertIs(get_axis_local_type(x, self.pg), R)

    # --- In-place ops that widen R -> V (allowed) ---

    def test_inplace_add_r_v_ok(self):
        """R.add_(V) -> V: R widens to V (replicated data becomes varying)."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, V)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, self.pg), V)

    def test_inplace_mul_r_v_ok(self):
        """R.mul_(V) -> V: R widens to V."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, V)
        x.mul_(y)
        self.assertIs(get_axis_local_type(x, self.pg), V)

    def test_iadd_r_v_ok(self):
        """R += V -> V: R widens to V via __iadd__."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, V)
        x += y
        self.assertIs(get_axis_local_type(x, self.pg), V)

    # --- out= kwarg ops ---

    def test_out_kwarg_matching_type_ok(self):
        """torch.add(R, R, out=R_tensor): output R matches out tensor R."""
        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, R)
        c = self._generate_inputs((4,), self.pg, R)
        torch.add(a, b, out=c)
        self.assertIs(get_axis_local_type(c, self.pg), R)

    def test_out_kwarg_r_to_v_ok(self):
        """torch.add(R, V, out=R_tensor): output V, out R widens to V."""
        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, V)
        c = self._generate_inputs((4,), self.pg, R)
        torch.add(a, b, out=c)
        self.assertIs(get_axis_local_type(c, self.pg), V)

    def test_out_kwarg_untyped_ok(self):
        """torch.add(R, R, out=untyped): untyped out tensor is fine."""
        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, R)
        c = self.rank_map(lambda r: torch.empty(4))
        torch.add(a, b, out=c)
        self.assertIs(get_axis_local_type(c, self.pg), R)

    def test_out_kwarg_v_output_into_v_ok(self):
        """torch.add(R, V, out=V_tensor): output V, out is V. OK."""
        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, V)
        c = self._generate_inputs((4,), self.pg, V)
        torch.add(a, b, out=c)
        self.assertIs(get_axis_local_type(c, self.pg), V)

    def test_inplace_p_p_add_ok(self):
        """P.add_(P) -> P: type preserved (linear op)."""
        x = self._generate_inputs((4,), self.pg, P)
        y = self._generate_inputs((4,), self.pg, P)
        x.add_(y)
        self.assertIs(get_axis_local_type(x, self.pg), P)

    # --- Regression: requires_grad must be read pre-op, not post-op ---

    def test_inplace_add_r_v_with_grad_input_ok(self):
        """R.add_(V_grad): autograd flips x.requires_grad to True post-op,
        but the R->V exemption must read pre-op grad state.
        """
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, V)
        y.requires_grad_(True)
        self.assertFalse(x.requires_grad)
        x.add_(y)
        self.assertTrue(x.requires_grad)
        self.assertIs(get_axis_local_type(x, self.pg), V)

    # --- In-place ops that would change type unsafely (still rejected) ---

    def test_out_kwarg_v_to_r_rejected(self):
        """torch.add(R, R, out=V_tensor): output R but out is V. Error."""
        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, R)
        c = self._generate_inputs((4,), self.pg, V)
        with self.assertRaises(SpmdTypeError) as ctx:
            torch.add(a, b, out=c)
        self.assertIn("in-place/out", str(ctx.exception))


class TestMutateType(LocalTensorTestCase):
    """Test mutate_type for explicit single-axis type transitions."""

    WORLD_SIZE = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))
        cls.tp = mesh.get_group("tp")
        cls.dp = mesh.get_group("dp")

    def test_basic_mutation(self):
        """Mutate a single axis from V to R."""
        x = torch.randn(4)
        assert_type(x, {self.dp: V})
        mutate_type(x, self.dp, src=V, dst=R)
        self.assertIs(get_axis_local_type(x, self.dp), R)

    def test_shard_sugar_src(self):
        """S(i) is accepted as src and compared as V."""
        x = torch.randn(4)
        assert_type(x, {self.dp: S(0)})
        mutate_type(x, self.dp, src=S(0), dst=R)
        self.assertIs(get_axis_local_type(x, self.dp), R)

    def test_shard_sugar_dst(self):
        """S(i) is accepted as dst and stored as V."""
        x = torch.randn(4)
        assert_type(x, {self.dp: R})
        mutate_type(x, self.dp, src=R, dst=S(0))
        self.assertIs(get_axis_local_type(x, self.dp), V)

    def test_preserves_other_axes(self):
        """Mutating one axis must not affect other axes."""
        x = torch.randn(4)
        assert_type(x, {self.dp: V, self.tp: I})
        mutate_type(x, self.dp, src=V, dst=R)
        self.assertIs(get_axis_local_type(x, self.dp), R)
        self.assertIs(get_axis_local_type(x, self.tp), I)

    def test_wrong_src_raises(self):
        """Mismatched src raises SpmdTypeError."""
        x = torch.randn(4)
        assert_type(x, {self.dp: V})
        with self.assertRaises(SpmdTypeError):
            mutate_type(x, self.dp, src=R, dst=I)

    def test_missing_axis_raises(self):
        """Axis not present in tensor's type raises SpmdTypeError."""
        x = torch.randn(4)
        assert_type(x, {self.dp: V})
        with self.assertRaises(SpmdTypeError):
            mutate_type(x, self.tp, src=V, dst=R)

    def test_unannotated_tensor_raises(self):
        """Tensor with no type at all raises SpmdTypeError."""
        x = torch.randn(4)
        with self.assertRaises(SpmdTypeError):
            mutate_type(x, self.dp, src=V, dst=R)

    def test_returns_tensor(self):
        """mutate_type returns the tensor for chaining."""
        x = torch.randn(4)
        assert_type(x, {self.dp: V})
        result = mutate_type(x, self.dp, src=V, dst=R)
        self.assertIs(result, x)


class TestAutogradFunctionApply(SpmdTypeCheckedTestCase):
    """Test autograd Function.apply monkeypatch and registration mechanism."""

    def _make_local_func(self):
        """Create and register a simple local-only autograd Function."""
        from spmd_types._checker import register_local_autograd_function

        @register_local_autograd_function
        class LocalOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, g):
                return g * 2

        return LocalOp

    def _make_unregistered_func(self):
        """Create an autograd Function without registering it."""

        class UnregisteredOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 3

            @staticmethod
            def backward(ctx, g):
                return g * 3

        return UnregisteredOp

    def test_registered_propagates_type(self):
        """Registered local autograd Function propagates SPMD types."""
        LocalOp = self._make_local_func()
        x = self._generate_inputs((4,), self.pg, R)
        result = LocalOp.apply(x)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_registered_propagates_v(self):
        """Registered local autograd Function propagates V type."""
        LocalOp = self._make_local_func()
        x = self._generate_inputs((4,), self.pg, V)
        result = LocalOp.apply(x)
        self.assertIs(get_axis_local_type(result, self.pg), V)

    def test_unregistered_raises_in_strict_mode(self):
        """Unregistered autograd Function raises SpmdTypeError in strict mode."""
        UnregisteredOp = self._make_unregistered_func()
        x = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(SpmdTypeError) as ctx:
            UnregisteredOp.apply(x)
        self.assertIn("not registered for SPMD type checking", str(ctx.exception))

    def test_unregistered_skips_in_nonstrict_mode(self):
        """Unregistered autograd Function leaves output untyped in non-strict mode."""
        UnregisteredOp = self._make_unregistered_func()
        # Exit strict mode, enter non-strict
        self._type_checking_cm.__exit__(None, None, None)
        self._type_checking_cm = typecheck(strict_mode="permissive")
        self._type_checking_cm.__enter__()

        x = self._generate_inputs((4,), self.pg, V)
        result = UnregisteredOp.apply(x)
        self.assertEqual(get_local_type(result), {})

    def test_registered_multi_tensor_inputs(self):
        """Registered autograd Function with multiple typed tensor inputs."""
        from spmd_types._checker import register_local_autograd_function

        @register_local_autograd_function
        class MultiInputOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                return a + b

            @staticmethod
            def backward(ctx, g):
                return g, g

        a = self._generate_inputs((4,), self.pg, R)
        b = self._generate_inputs((4,), self.pg, R)
        result = MultiInputOp.apply(a, b)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_typecheck_forward_sets_output_type(self):
        """typecheck_forward calls .apply() and sets output type."""
        from spmd_types._checker import assert_type, register_autograd_function

        @register_autograd_function
        class CheckedOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def typecheck_forward(x):
                assert_type(x, {self.pg: R})
                out = CheckedOp.apply(x)
                assert_type(out, {self.pg: R})
                return out

            @staticmethod
            def backward(ctx, g):
                return g * 2

        x = self._generate_inputs((4,), self.pg, R)
        result = CheckedOp.apply(x)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_typecheck_forward_raises_on_wrong_input(self):
        """typecheck_forward can validate inputs via assert_type."""
        from spmd_types._checker import assert_type, register_autograd_function

        @register_autograd_function
        class StrictCheckOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def typecheck_forward(x):
                # Expects V, but we'll pass R
                assert_type(x, {self.pg: V})
                out = StrictCheckOp.apply(x)
                assert_type(out, {self.pg: V})
                return out

            @staticmethod
            def backward(ctx, g):
                return g * 2

        x = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            StrictCheckOp.apply(x)

    def test_typecheck_forward_overrides_local_default(self):
        """register_autograd_function produces different types than local-only.

        The local-only rule for infer_output_type([V, R]) would produce V.
        typecheck_forward sets R instead (e.g., because the function does an
        internal all-reduce).
        """
        from spmd_types._checker import assert_type, register_autograd_function

        @register_autograd_function
        class OverrideOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x + y

            @staticmethod
            def typecheck_forward(x, y):
                out = OverrideOp.apply(x, y)
                assert_type(out, {self.pg: R})
                return out

            @staticmethod
            def backward(ctx, g):
                return g, g

        x = self._generate_inputs((4,), self.pg, V)
        y = self._generate_inputs((4,), self.pg, R)
        result = OverrideOp.apply(x, y)
        # Local-only rule would give V; typecheck_forward sets R.
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_typecheck_forward_receives_all_args(self):
        """typecheck_forward receives tensor and non-tensor args."""
        from spmd_types._checker import assert_type, register_autograd_function

        received = {}

        @register_autograd_function
        class ArgsOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, scale):
                return x * scale

            @staticmethod
            def typecheck_forward(x, scale):
                received["x"] = x
                received["scale"] = scale
                out = ArgsOp.apply(x, scale)
                assert_type(out, {self.pg: R})
                return out

            @staticmethod
            def backward(ctx, g):
                return g, None

        x = self._generate_inputs((4,), self.pg, R)
        result = ArgsOp.apply(x, 2.0)
        self.assertIs(received["x"], x)
        self.assertEqual(received["scale"], 2.0)
        self.assertIs(get_axis_local_type(result, self.pg), R)

    def test_patch_only_active_during_mode(self):
        """Monkeypatch is only active while typecheck is entered."""
        # Inside the mode, apply goes through __torch_function__
        LocalOp = self._make_local_func()
        x = self._generate_inputs((4,), self.pg, R)
        LocalOp.apply(x)

        # Exit all modes
        self._type_checking_cm.__exit__(None, None, None)

        # Outside the mode, apply should work normally (no type propagation)
        plain = torch.randn(4)
        result2 = LocalOp.apply(plain)
        self.assertEqual(get_local_type(result2), {})

        # Re-enter for tearDown
        self._type_checking_cm = typecheck()
        self._type_checking_cm.__enter__()

    def test_mixed_empty_typed_raises_even_if_registered(self):
        """Mixed typed + {} inputs raise in strict mode for all ops.

        {} typed tensors (from factory ops) must be annotated with
        assert_type() before being mixed with tensors that have real
        axes, even for registered local autograd Functions.
        """
        from spmd_types._checker import register_local_autograd_function

        @register_local_autograd_function
        class MixedOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, dummy):
                return x + dummy

            @staticmethod
            def backward(ctx, g):
                return g, None

        x = self._generate_inputs((4,), self.pg, R)
        dummy = torch.zeros(4)  # {} typed tensor
        with self.assertRaises(SpmdTypeError):
            MixedOp.apply(x, dummy)


class TestRegisterDecomposition(SpmdTypeCheckedTestCase):
    """Test register_decomposition: derive SPMD types by tracing a
    pure-PyTorch reference implementation through primitive rules.
    """

    def test_invalid_input_rejected_by_primitive_rules(self):
        """SPMD errors surface naturally from primitive ops in the decomp."""
        from spmd_types._checker import register_decomposition

        class FusedOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1.0

            @staticmethod
            def backward(ctx, g):
                return g

        def ref_impl(x):
            # add(P, scalar) is affine on P -- rejected by primitive rule.
            return x + 1.0

        register_decomposition(FusedOp, ref_impl)

        x = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            FusedOp.apply(x)

    def test_tuple_output_stamps_each_leaf(self):
        """Decomposition with tuple output stamps every leaf."""
        from spmd_types._checker import register_decomposition

        class FusedOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x + y, x - y

            @staticmethod
            def backward(ctx, g1, g2):
                return g1 + g2, g1 - g2

        def ref_impl(x, y):
            return x + y, x - y

        register_decomposition(FusedOp, ref_impl)

        x = self._generate_inputs((4,), self.pg, V)
        y = self._generate_inputs((4,), self.pg, R)
        out1, out2 = FusedOp.apply(x, y)
        self.assertIs(get_axis_local_type(out1, self.pg), V)
        self.assertIs(get_axis_local_type(out2, self.pg), V)

    def test_tree_structure_mismatch_raises(self):
        """Mismatched tree shapes between decomp and fused output raise."""
        from spmd_types._checker import register_decomposition

        class FusedOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1  # Single tensor.

            @staticmethod
            def backward(ctx, g):
                return g

        def ref_impl(x):
            return x + 1, x - 1  # Tuple -- structure mismatch.

        register_decomposition(FusedOp, ref_impl)

        x = self._generate_inputs((4,), self.pg, R)
        with self.assertRaises(SpmdTypeError):
            FusedOp.apply(x)

    def test_decorator_form(self):
        """``@register_decomposition(cls)`` registers and returns the ref_impl."""
        from spmd_types._checker import register_decomposition

        class FusedOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1.0

            @staticmethod
            def backward(ctx, g):
                return g

        @register_decomposition(FusedOp)
        def ref_impl(x):
            return x + 1.0

        # Decorator returns ref_impl unchanged so the bound name is the function.
        self.assertTrue(callable(ref_impl))
        self.assertEqual(ref_impl.__name__, "ref_impl")
        # Registration took effect: FusedOp now has a typecheck_forward that
        # surfaces primitive-op SPMD errors (P + scalar is rejected).
        x = self._generate_inputs((4,), self.pg, P)
        with self.assertRaises(SpmdTypeError):
            FusedOp.apply(x)


class TestRegisterDecompositionGlobalSpmd(LocalTensorTestCase):
    """``register_decomposition`` under global SPMD: primitive-op rules
    propagate PartitionSpec through a RMSNorm-like decomposition and
    reject sharding on the normalized dim.
    """

    WORLD_SIZE = 4

    def setUp(self):
        super().setUp()
        self._tc_cm = typecheck(local=False)
        self._tc_cm.__enter__()

        from spmd_types._checker import register_decomposition

        def _native_rms_norm(x, w, eps):
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(variance + eps) * w

        class _RMSNormLike(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w, eps):
                return _rmsnorm_fused_kernel_for_test(x, w, eps)

            @staticmethod
            def backward(ctx, g):
                return g, None, None

        register_decomposition(_RMSNormLike, _native_rms_norm)
        self.rmsnorm_like = _RMSNormLike.apply

    def tearDown(self):
        self._tc_cm.__exit__(None, None, None)
        super().tearDown()

    def test_replicated_input_stays_replicated(self):
        """R input -> R output, no sharding introduced."""
        x = torch.randn(8, 16)
        w = torch.randn(16)
        assert_type(x, {self.pg: R})
        assert_type(w, {self.pg: R})
        y = self.rmsnorm_like(x, w, 1e-6)
        self.assertIs(get_axis_local_type(y, self.pg), R)

    def test_shard_on_non_normalized_dim_propagates(self):
        """S(0) (not the normalized last dim) propagates to the output."""
        x = torch.randn(8, 16)
        w = torch.randn(16)
        assert_type(x, {self.pg: S(0)})
        assert_type(w, {self.pg: R})
        y = self.rmsnorm_like(x, w, 1e-6)
        self.assertEqual(
            get_partition_spec(y),
            PartitionSpec(self.pg, None),
        )

    def test_shard_on_normalized_dim_rejected(self):
        """Sharding the normalized (last) dim is rejected."""
        x = torch.randn(8, 16)
        w = torch.randn(16)
        assert_type(x, {self.pg: S(1)})
        assert_type(w, {self.pg: R})
        with self.assertRaises((SpmdTypeError, ValueError)):
            self.rmsnorm_like(x, w, 1e-6)


class TestVmapWithSpmdTypeMode(unittest.TestCase):
    """Test vmap compatibility with SpmdTypeMode's autograd apply monkeypatch."""

    def test_vmap_compatible_with_generate_vmap_rule(self):
        """Monkeypatch preserves vmap dispatch for autograd functions with generate_vmap_rule.

        Regression test: the old monkeypatch saved _FunctionBase.apply (raw C++
        apply) instead of Function.apply (Python classmethod that handles
        vmap/functorch dispatch).  This caused autograd functions with
        generate_vmap_rule=True to crash under vmap with a
        dynamicLayerStack assertion failure.
        """

        class VmapOp(torch.autograd.Function):
            generate_vmap_rule = True

            @staticmethod
            def forward(x):
                return x * 2

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

        with typecheck(strict_mode="permissive"):
            x = torch.randn(3, 4)
            result = torch.vmap(VmapOp.apply)(x)
            torch.testing.assert_close(result, x * 2)


class TestOrthogonalityValidation(LocalTensorTestCase):
    """Test that _validate() rejects non-orthogonal mesh axes.

    The core invariant: all mesh axes in a tensor's LocalSpmdType dict must be
    mutually orthogonal (they tile different dimensions of the mesh without
    rank collisions). Overlapping axes produce incorrect type inference results
    because type inference reasons about each axis independently.
    """

    WORLD_SIZE = 8

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from spmd_types._mesh_axis import MeshAxis

        # Orthogonal axes: dp (size=2, stride=4) and tp (size=4, stride=1)
        # dp generates ranks {0, 4}, tp generates ranks {0, 1, 2, 3}
        # Combined: 2*4 = 8 unique ranks -- orthogonal.
        cls.dp = MeshAxis.of(2, 4)
        cls.tp = MeshAxis.of(4, 1)

        # Overlapping axis: dp_cp (size=4, stride=2) overlaps with dp above.
        # dp generates ranks {0, 4}, dp_cp generates ranks {0, 2, 4, 6}
        # They share rank 0 and rank 4.
        cls.dp_cp = MeshAxis.of(4, 2)

        # A small axis for three-way tests: size=2, stride=1
        cls.small = MeshAxis.of(2, 1)

    def test_overlapping_axes_rejected(self):
        """Subset relationship: dp <= dp_cp, so they share ranks."""
        x = torch.randn(4)
        with self.assertRaises(SpmdTypeError) as ctx:
            assert_type(x, {self.dp: R, self.dp_cp: V})
        self.assertIn("not orthogonal", str(ctx.exception))

    def test_same_axis_handled_by_dict_dedup(self):
        """Same axis used twice in a dict -- Python dict deduplicates keys."""
        x = torch.randn(4)
        # Python dict dedup: {dp: R, dp: V} -> {dp: V}, so only one key.
        assert_type(x, {self.dp: R, self.dp: V})

    def test_orthogonal_axes_from_init_device_mesh(self):
        """Axes from init_device_mesh are orthogonal by construction."""
        mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "tp"))
        dp_pg = mesh.get_group("dp")
        tp_pg = mesh.get_group("tp")
        x = torch.randn(4)
        # Should not raise.
        assert_type(x, {dp_pg: R, tp_pg: V})

    def test_single_axis_skips_check(self):
        """Single-axis dicts skip the orthogonality check entirely."""
        x = torch.randn(4)
        # Should not raise -- only one axis, nothing to check.
        assert_type(x, {self.dp: R})

    def test_size_1_axes_always_orthogonal(self):
        """Size-1 axes are always orthogonal to anything."""
        from spmd_types._mesh_axis import MeshAxis

        trivial = MeshAxis.of(1, 1)
        x = torch.randn(4)
        # Size-1 axis is orthogonal to any other axis.
        assert_type(x, {trivial: R, self.tp: V})

    def test_three_way_overlap_detected(self):
        """Three axes where at least one pair overlaps are detected."""
        x = torch.randn(4)
        # dp and dp_cp overlap; tp is orthogonal to dp but that doesn't help.
        with self.assertRaises(SpmdTypeError) as ctx:
            assert_type(x, {self.dp: R, self.tp: V, self.dp_cp: R})
        msg = str(ctx.exception)
        self.assertIn("not orthogonal", msg)

    def test_error_message_names_offending_axes(self):
        """Error message identifies the specific overlapping pair."""
        x = torch.randn(4)
        with self.assertRaises(SpmdTypeError) as ctx:
            assert_type(x, {self.dp: R, self.dp_cp: V})
        msg = str(ctx.exception)
        # The message should mention both axes.
        self.assertIn("not orthogonal", msg)
        self.assertIn("share ranks", msg)

    def test_mutate_type_does_not_add_nonorthogonal_axes(self):
        """mutate_type changes values, so it still validates the full dict."""
        x = torch.randn(4)
        assert_type(x, {self.dp: R, self.tp: V})
        # Mutating an existing key is fine (changes value, not keys).
        mutate_type(x, self.dp, src=R, dst=V)

    def test_merge_overlapping_axis_rejected(self):
        """Re-check path: merging a new axis that overlaps existing axes raises."""
        x = torch.randn(4)
        # First call: set type on the overlapping dp_cp axis.
        assert_type(x, {self.dp_cp: V})
        # Second call: try to merge dp, which overlaps with dp_cp.
        with self.assertRaises(SpmdTypeError) as ctx:
            assert_type(x, {self.dp: R})
        self.assertIn("not orthogonal", str(ctx.exception))
        # Failed assert_type must not corrupt the tensor's type dict.
        existing = get_local_type(x)
        self.assertNotIn(normalize_axis(self.dp), existing)
        self.assertIs(existing[normalize_axis(self.dp_cp)], V)

    def test_merge_overlapping_default_pg_rejected(self):
        """Re-check path: default_pg overlaps with sub-mesh axes."""
        from spmd_types._mesh_axis import MeshAxis

        # default_pg covers all 8 ranks.
        default_pg = MeshAxis.of(self.pg)
        x = torch.randn(4)
        # First call: set type on default_pg (all ranks).
        assert_type(x, {default_pg: V})
        # Second call: try to merge dp (subset of default_pg ranks).
        with self.assertRaises(SpmdTypeError) as ctx:
            assert_type(x, {self.dp: R})
        self.assertIn("not orthogonal", str(ctx.exception))
        # Failed assert_type must not corrupt the tensor's type dict.
        existing = get_local_type(x)
        self.assertNotIn(normalize_axis(self.dp), existing)
        self.assertIs(existing[default_pg], V)


class TestTraceMode(SpmdTypeCheckedTestCase, expecttest.TestCase):
    """Test SPMD_TYPES_TRACE logging output."""

    def setUp(self):
        super().setUp()
        self._handler = logging.Handler()
        self._handler.emit = lambda record: self._records.append(record)
        self._records: list[logging.LogRecord] = []
        _trace_logger.addHandler(self._handler)
        _trace_logger.setLevel(logging.DEBUG)

    def tearDown(self):
        _trace_logger.removeHandler(self._handler)
        super().tearDown()

    def _trace_output(self) -> str:
        lines = [r.getMessage() for r in self._records]
        # Normalize /abs/path/to/file.py:123 -> file.py:N for stability.
        lines = [re.sub(r"^.*?([^/]+\.py):\d+", r"\1:N", line) for line in lines]
        return "\n".join(lines)

    def test_trace_logs_local_op(self):
        """Trace logs operator name, input types, and output type."""
        with trace():
            x = self._generate_inputs((4,), self.pg, R)
            y = self._generate_inputs((4,), self.pg, V)
            _ = x + y

        self.assertExpectedInline(
            self._trace_output(),
            """\
runtime.py:N  assert_type({}) -> {default_pg: R}
runtime.py:N  assert_type({}) -> {default_pg: V}""",
        )

    def test_trace_logs_multiple_ops(self):
        """Each non-trivial op gets its own trace line."""
        with trace():
            x = self._generate_inputs((4,), self.pg, R)
            y = self._generate_inputs((4,), self.pg, V)
            z = x + y
            _ = z * z

        self.assertExpectedInline(
            self._trace_output(),
            """\
runtime.py:N  assert_type({}) -> {default_pg: R}
runtime.py:N  assert_type({}) -> {default_pg: V}""",
        )

    def test_trace_silent_when_disabled(self):
        """No trace output when trace is explicitly disabled."""
        with trace(enabled=False):
            x = self._generate_inputs((4,), self.pg, R)
            y = self._generate_inputs((4,), self.pg, V)
            _ = x + y

        self.assertExpectedInline(self._trace_output(), """""")

    def test_trace_logs_collective(self):
        """Trace logs SPMD collectives (all_reduce)."""
        with trace():
            x = self._generate_inputs((4,), self.pg, P)
            _ = all_reduce(x, self.pg, dst=R)

        self.assertExpectedInline(
            self._trace_output(),
            """runtime.py:N  assert_type({}) -> {default_pg: P}""",
        )

    def test_trace_logs_assert_type_initial(self):
        """Trace logs assert_type when setting initial type on a tensor."""
        with trace():
            x = torch.zeros(4)
            assert_type(x, {self.pg: R})

        self.assertExpectedInline(
            self._trace_output(),
            """runtime.py:N  assert_type({}) -> {default_pg: R}""",
        )

    def test_trace_logs_assert_type_refinement(self):
        """Trace logs assert_type when refining/merging existing type."""
        x = self._generate_inputs((4,), self.pg, R)
        with trace():
            assert_type(x, {self.pg: R})

        self.assertExpectedInline(
            self._trace_output(),
            """runtime.py:N  assert_type({default_pg: R}) -> {default_pg: R}""",
        )

    def test_trace_omits_all_empty_types(self):
        """Trace suppresses lines where all inputs and output are empty."""
        with trace():
            # Two factory tensors with {} type -- op should be suppressed.
            x = torch.zeros(4)
            assert_type(x, {})
            y = torch.ones(4)
            assert_type(y, {})
            _ = x + y

        self.assertExpectedInline(self._trace_output(), """""")


class TestErrorContext(SpmdTypeCheckedTestCase, expecttest.TestCase):
    """Test enhanced error context on SpmdTypeError."""

    def test_context_field_none_by_default(self):
        """SpmdTypeError.context is None when not set."""
        err = SpmdTypeError("test message")
        self.assertIsNone(err.context)
        self.assertEqual(str(err), "test message")

    def test_context_field_appended_to_str(self):
        """SpmdTypeError.context appears in str(error) when set."""
        err = SpmdTypeError("base msg", context="  In foo()")
        self.assertExpectedInline(
            str(err),
            """\
base msg

  In foo()""",
        )

    def test_context_set_after_init(self):
        """Setting context after construction also works."""
        err = SpmdTypeError("base msg")
        err.context = "  In bar()"
        self.assertExpectedInline(
            str(err),
            """\
base msg

  In bar()""",
        )

    def test_add_p_r_error_has_context(self):
        """P + R error includes operator context with shapes, dtypes, types."""
        x = self._generate_inputs((8, 16), self.pg, P)
        y = self._generate_inputs((16,), self.pg, R)
        with self.assertRaises(SpmdTypeError) as ctx:
            torch.add(x, y)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis default_pg in a linear op requires all operands to be Partial (sum of partial sums is a partial sum, but adding a Replicate or scalar value makes the result affine -- the non-partial term gets summed N times across ranks). Found types: [P, R]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, default_pg, src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)
  convert(tensor, default_pg, src=R, dst=P) on the Replicate operand (zeros non-rank-0 in forward, zeros non-rank-0 in backward)

  In add(
    args[0]: f32[8, 16] {default_pg: P},
    args[1]: f32[16] {default_pg: R},
  )""",
        )

    def test_matmul_p_p_error_has_context(self):
        """P * P matmul error includes operator context."""
        x = self._generate_inputs((2, 4), self.pg, P)
        y = self._generate_inputs((4, 3), self.pg, P)
        with self.assertRaises(SpmdTypeError) as ctx:
            torch.matmul(x, y)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial in multiple factors of multilinear op on axis default_pg is forbidden. Reduce all but one factor first. Found types: [P, P]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, default_pg, src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)

  In matmul(
    args[0]: f32[2, 4] {default_pg: P},
    args[1]: f32[4, 3] {default_pg: P},
  )""",
        )

    def test_addmm_decomp_error_has_context(self):
        """addmm type decomposition errors also get context."""
        self_t = self._generate_inputs((2, 3), self.pg, P)
        mat1 = self._generate_inputs((2, 4), self.pg, R)
        mat2 = self._generate_inputs((4, 3), self.pg, R)
        with self.assertRaises(SpmdTypeError) as ctx:
            torch.addmm(self_t, mat1, mat2)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis default_pg in a linear op requires all operands to be Partial (sum of partial sums is a partial sum, but adding a Replicate or scalar value makes the result affine -- the non-partial term gets summed N times across ranks). Found types: [R, P]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, default_pg, src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)
  convert(tensor, default_pg, src=R, dst=P) on the Replicate operand (zeros non-rank-0 in forward, zeros non-rank-0 in backward)

  In addmm(
    args[0]: f32[2, 3] {default_pg: P},
    args[1]: f32[2, 4] {default_pg: R},
    args[2]: f32[4, 3] {default_pg: R},
  )""",
        )


class TestErrorContextMultiAxis(expecttest.TestCase):
    """Test error context with multiple mesh axes."""

    WORLD_SIZE = 4

    @classmethod
    def setUpClass(cls):
        from spmd_types._testing import fake_pg

        super().setUpClass()
        cls._pg_ctx = fake_pg(cls.WORLD_SIZE)
        cls._pg_ctx.__enter__()
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        cls.dp = mesh.get_group("dp")
        cls.tp = mesh.get_group("tp")
        cls._tc_ctx = typecheck()
        cls._tc_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._tc_ctx.__exit__(None, None, None)
        cls._pg_ctx.__exit__(None, None, None)
        super().tearDownClass()

    def test_multi_axis_context(self):
        """Error context with two axes shows both axis types."""
        x = torch.randn(8, 16)
        assert_type(x, {self.dp: P, self.tp: R})
        y = torch.randn(16)
        assert_type(y, {self.dp: R, self.tp: R})
        with self.assertRaises(SpmdTypeError) as ctx:
            torch.add(x, y)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
Partial type on axis mesh_dp in a linear op requires all operands to be Partial (sum of partial sums is a partial sum, but adding a Replicate or scalar value makes the result affine -- the non-partial term gets summed N times across ranks). Found types: [P, R]
Are you missing a collective or a reinterpret/convert call? e.g.,
  all_reduce(tensor, mesh_dp, src=P, dst=R) on the Partial operand (all-reduce in forward, all-reduce in backward)
  convert(tensor, mesh_dp, src=R, dst=P) on the Replicate operand (zeros non-rank-0 in forward, zeros non-rank-0 in backward)

  In add(
    args[0]: f32[8, 16] {mesh_dp: P, mesh_tp: R},
    args[1]: f32[16] {mesh_dp: R, mesh_tp: R},
  )""",
        )


class TestErrorContextRawDist(LocalTensorTestCase, expecttest.TestCase):
    """Test error context for raw torch.distributed collectives.

    Raw dist functions have inspectable signatures, so positional args should
    be labeled with parameter names.  Also tests list-of-tensors formatting.
    """

    def test_all_gather_into_tensor_wrong_input_context(self):
        """all_gather_into_tensor error shows named params and group."""
        x = self._generate_inputs((2,), self.pg, R)
        out = self._generate_inputs((6,), self.pg, R)
        with typecheck("strict"):
            with self.assertRaises(SpmdTypeError) as ctx:
                dist.all_gather_into_tensor(out, x, group=self.pg)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
all_gather_into_tensor: expected input type PerMeshAxisLocalSpmdType.V on axis default_pg, got PerMeshAxisLocalSpmdType.R

  In all_gather_into_tensor(
    output_tensor: f32[6] {default_pg: R},
    input_tensor: f32[2] {default_pg: R},
    group: default_pg,
    async_op: False,
  )""",
        )

    def test_all_gather_list_wrong_input_context(self):
        """all_gather (list variant) error shows list of tensors."""
        x = self._generate_inputs((4,), self.pg, R)
        out_list = [
            self._generate_inputs((4,), self.pg, R) for _ in range(self.WORLD_SIZE)
        ]
        with typecheck("strict"):
            with self.assertRaises(SpmdTypeError) as ctx:
                dist.all_gather(out_list, x, group=self.pg)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
all_gather: expected input type PerMeshAxisLocalSpmdType.V on axis default_pg, got PerMeshAxisLocalSpmdType.R

  In all_gather(
    tensor_list: [f32[4] {default_pg: R}] * 3,
    tensor: f32[4] {default_pg: R},
    group: default_pg,
    async_op: False,
  )""",
        )

    def test_all_reduce_wrong_input_context(self):
        """all_reduce error shows named params."""
        x = self._generate_inputs((4,), self.pg, R)
        with typecheck("strict"):
            with self.assertRaises(SpmdTypeError) as ctx:
                dist.all_reduce(x, group=self.pg)
        self.assertExpectedInline(
            str(ctx.exception),
            """\
all_reduce: expected input type PerMeshAxisLocalSpmdType.P on axis default_pg, got PerMeshAxisLocalSpmdType.R

  In all_reduce(
    tensor: f32[4] {default_pg: R},
    op: <RedOpType.SUM: 0>,
    group: default_pg,
    async_op: False,
  )""",
        )


class TestSingletonAxisDropped(expecttest.TestCase):
    """Singleton mesh axes (size 1) should be silently dropped from SPMD types."""

    WORLD_SIZE = 6

    @classmethod
    def setUpClass(cls):
        from spmd_types._testing import fake_pg

        super().setUpClass()
        cls._pg_ctx = fake_pg(cls.WORLD_SIZE)
        cls._pg_ctx.__enter__()
        # (1, 6) mesh: singleton_axis has size 1, real_axis has size 6
        mesh = init_device_mesh("cpu", (1, 6), mesh_dim_names=("singleton", "real"))
        cls.singleton = mesh.get_group("singleton")
        cls.real = mesh.get_group("real")

    @classmethod
    def tearDownClass(cls):
        cls._pg_ctx.__exit__(None, None, None)
        super().tearDownClass()

    def test_singleton_dropped_from_assert_type(self):
        """assert_type with only a singleton axis produces an empty type dict."""
        x = torch.randn(4)
        assert_type(x, {self.singleton: R})
        from spmd_types._checker import get_local_type

        self.assertEqual(get_local_type(x), {})

    def test_singleton_plus_real_keeps_real(self):
        """assert_type with singleton + real axis keeps only the real axis."""
        x = torch.randn(4)
        assert_type(x, {self.singleton: V, self.real: R})
        from spmd_types._checker import get_local_type

        local_type = get_local_type(x)
        self.assertNotIn(
            normalize_axis(self.singleton),
            local_type,
        )
        self.assertIs(local_type[normalize_axis(self.real)], R)

    def test_mutate_type_singleton_is_noop(self):
        """mutate_type on a singleton axis returns the tensor unchanged."""
        x = torch.randn(4)
        assert_type(x, {self.real: R})
        result = mutate_type(x, self.singleton, src=R, dst=V)
        self.assertIs(result, x)
        self.assertIs(get_axis_local_type(x, self.real), R)


class TestSingletonAxisStrictMode(expecttest.TestCase):
    """Strict mode should not error for singleton axes."""

    WORLD_SIZE = 6

    @classmethod
    def setUpClass(cls):
        from spmd_types._testing import fake_pg

        super().setUpClass()
        cls._pg_ctx = fake_pg(cls.WORLD_SIZE)
        cls._pg_ctx.__enter__()
        mesh = init_device_mesh("cpu", (1, 6), mesh_dim_names=("singleton", "real"))
        cls.singleton = mesh.get_group("singleton")
        cls.real = mesh.get_group("real")
        cls._tc_ctx = typecheck()
        cls._tc_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._tc_ctx.__exit__(None, None, None)
        cls._pg_ctx.__exit__(None, None, None)
        super().tearDownClass()

    def test_ops_with_singleton_axis(self):
        """Ops between tensors typed on a singleton axis should work fine."""
        x = torch.randn(4)
        assert_type(x, {self.singleton: R, self.real: R})
        y = torch.randn(4)
        assert_type(y, {self.singleton: V, self.real: R})
        # Both tensors are R on real axis; singleton is absent from both.
        # This should succeed (R + R -> R on real axis).
        result = torch.add(x, y)
        self.assertIs(get_axis_local_type(result, self.real), R)

    def test_strict_mode_no_error_for_singleton(self):
        """Strict mode should not raise for missing singleton axis on collectives."""
        x = torch.randn(4)
        assert_type(x, {self.real: P})
        # Calling all_reduce on singleton axis -- should not raise even in
        # strict mode, since singleton axes are simply not tracked.
        result = all_reduce(x, self.singleton, src=P, dst=R)
        # Type on real axis should be preserved
        self.assertIs(get_axis_local_type(result, self.real), P)


class TestSingletonAxisInMeshFiltered(expecttest.TestCase):
    """_push_mesh rejects size-1 axes; callers must filter them out.

    The contract: callers (e.g. utils_spmd.push_mesh) must drop trivial
    (size-1) mesh axes before calling _push_mesh.  _push_mesh asserts
    this invariant so that the checker never sees size-1 axes (which
    would cause spurious strict-mode errors for tensors not annotated
    on the trivial dimension).
    """

    WORLD_SIZE = 8

    @classmethod
    def setUpClass(cls):
        from spmd_types._testing import fake_pg

        super().setUpClass()
        cls._pg_ctx = fake_pg(cls.WORLD_SIZE)
        cls._pg_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._pg_ctx.__exit__(None, None, None)
        super().tearDownClass()

    def test_push_mesh_rejects_size1_axes(self):
        """_push_mesh must raise AssertionError if given a size-1 axis."""
        from spmd_types._state import _push_mesh

        trivial = MeshAxis.of(1, 8)  # size=1
        real = MeshAxis.of(self.WORLD_SIZE, 1)

        with self.assertRaises(AssertionError):
            _push_mesh(frozenset({trivial, real}))

    def test_filtered_mesh_works_with_type_checker(self):
        """When size-1 axes are filtered out, type checking works normally."""
        from spmd_types._state import _pop_mesh, _push_mesh

        real_axis = MeshAxis.of(self.WORLD_SIZE, 1)
        # Only push the non-trivial axis (as callers are required to do).
        _push_mesh(frozenset({real_axis}))
        try:
            with typecheck():
                x = torch.randn(4)
                assert_type(x, {real_axis: V})
                # This must not raise -- no size-1 axis in the mesh.
                y = x >= 0
                self.assertIs(get_axis_local_type(y, real_axis), V)
        finally:
            _pop_mesh()


class TestMissingAxesValidation(SpmdTypeCheckedTestCase):
    """Test that missing mesh axes are rejected at shard propagation time."""

    WORLD_SIZE = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        mesh = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))
        cls.tp = mesh.get_group("tp")
        cls.dp = mesh.get_group("dp")

    def test_missing_axis_in_binary_op(self):
        """Mixing operands with different axis sets raises SpmdTypeError."""
        x = self._generate_inputs((4, 3), self.tp, R)
        y = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(y, {self.tp: R, self.dp: R})
        with self.assertRaises(SpmdTypeError):
            x + y  # x is missing dp

    def test_missing_axis_in_matmul(self):
        """Missing axis in matmul operand raises SpmdTypeError."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: V, self.dp: V})
        w = self._generate_inputs((3, 5), self.tp, V)
        with self.assertRaises(SpmdTypeError):
            torch.mm(x, w)  # w is missing dp

    def test_same_axes_ok(self):
        """Operands with the same axis set should pass."""
        x = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(x, {self.tp: R, self.dp: R})
        y = self.rank_map(lambda r: torch.randn(4, 3))
        assert_type(y, {self.tp: R, self.dp: R})
        result = x + y
        self.assertIs(get_axis_local_type(result, self.tp), R)
        self.assertIs(get_axis_local_type(result, self.dp), R)


class TestReinterpretMeshIntegration(expecttest.TestCase):
    """Test explicit cross-mesh reinterpretation."""

    def setUp(self) -> None:
        from spmd_types._mesh_axis import _reset

        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=16, store=store)

        mesh = init_device_mesh("cpu", (2, 2, 4), mesh_dim_names=("dp", "cp", "tp"))
        self.dp = mesh.get_group("dp")
        self.cp = mesh.get_group("cp")
        self.tp = mesh.get_group("tp")
        dp_cp_mesh = mesh["dp", "cp"]._flatten("dp_cp")
        self.dp_cp = dp_cp_mesh.get_group("dp_cp")
        cp_tp_mesh = mesh["cp", "tp"]._flatten("cp_tp")
        self.cp_tp = cp_tp_mesh.get_group("cp_tp")

        # Register all axes so complement detection works (realistic --
        # in real programs all mesh dimensions would be used somewhere).
        for pg in (self.dp, self.cp, self.tp, self.dp_cp, self.cp_tp):
            normalize_axis(pg)

    def tearDown(self) -> None:
        from spmd_types._mesh_axis import _reset

        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def test_mesh_mismatch_cross_mesh_compatible(self) -> None:
        """Cross-mesh compatible axes should suggest reinterpret_mesh."""
        with typecheck(strict_mode="strict"):
            x = torch.randn(4)
            assert_type(x, {self.dp_cp: V})
            y = torch.randn(4)
            assert_type(y, {self.dp: R, self.cp: R})
            with self.assertRaises(SpmdTypeError) as cm:
                x + y
            self.assertExpectedInline(
                str(cm.exception),
                """\
args[0] missing axis mesh_cp. args[0] has axes {mesh_dp_cp}, but the union of all operand axes is {mesh_dp, mesh_cp, mesh_dp_cp}. All operands must be annotated on the same set of mesh axes.

Hint: The operand axes {mesh_dp_cp} and {mesh_dp, mesh_cp} are cross-mesh compatible. Use reinterpret_mesh() to convert the operand with axes {mesh_dp_cp} to axes {mesh_dp, mesh_cp} before this operation.

  In add(
    args[0]: f32[4] {mesh_dp_cp: V},
    args[1]: f32[4] {mesh_dp: R, mesh_cp: R},
  )""",
            )

    def test_mesh_mismatch_sub_axis(self) -> None:
        """Sub-axis relationship should identify missing axes."""
        with typecheck(strict_mode="strict"):
            x = torch.randn(4)
            assert_type(x, {self.dp: V, self.tp: V})
            y = torch.randn(4)
            assert_type(y, {self.dp_cp: V, self.tp: V})
            with self.assertRaises(SpmdTypeError) as cm:
                x + y
            self.assertExpectedInline(
                str(cm.exception),
                """\
args[1] missing axis mesh_dp. args[1] has axes {mesh_dp_cp, mesh_tp}, but the union of all operand axes is {mesh_dp, mesh_dp_cp, mesh_tp}. All operands must be annotated on the same set of mesh axes.

Hint: mesh_dp is a sub-axis of mesh_dp_cp (ranks {0, 8} vs {0, 4, 8, 12}), so the operand with axes {mesh_dp, mesh_tp} is missing axis mesh_cp.

  In add(
    args[0]: f32[4] {mesh_dp: V, mesh_tp: V},
    args[1]: f32[4] {mesh_dp_cp: V, mesh_tp: V},
  )""",
            )

    def test_mesh_mismatch_disjoint_no_hint(self) -> None:
        """Completely disjoint axes should not produce a hint."""
        with typecheck(strict_mode="strict"):
            x = torch.randn(4)
            assert_type(x, {self.dp: V})
            y = torch.randn(4)
            assert_type(y, {self.tp: R})
            with self.assertRaises(SpmdTypeError) as cm:
                x + y
            self.assertExpectedInline(
                str(cm.exception),
                """\
args[1] missing axis mesh_dp. args[1] has axes {mesh_tp}, but the union of all operand axes is {mesh_dp, mesh_tp}. All operands must be annotated on the same set of mesh axes.

  In add(
    args[0]: f32[4] {mesh_dp: V},
    args[1]: f32[4] {mesh_tp: R},
  )""",
            )

    def test_mesh_mismatch_partial_overlap(self) -> None:
        """Partial rank overlap (no sub-axis) should flag the overlap."""
        # dp_cp covers ranks {0, 4, 8, 12}, cp_tp covers ranks {0..7}.
        # They partially overlap at rank 4 (excluding trivial rank 0),
        # but neither is a sub-axis of the other.
        with typecheck(strict_mode="strict"):
            x = torch.randn(4)
            assert_type(x, {self.dp_cp: V})
            y = torch.randn(4)
            assert_type(y, {self.cp_tp: R})
            with self.assertRaises(SpmdTypeError) as cm:
                x + y
            self.assertExpectedInline(
                str(cm.exception),
                """\
args[0] missing axis mesh_cp_tp. args[0] has axes {mesh_dp_cp}, but the union of all operand axes is {mesh_dp_cp, mesh_cp_tp}. All operands must be annotated on the same set of mesh axes.

Hint: The non-shared axes {mesh_dp_cp} and {mesh_cp_tp} partially overlap. The operand with axes {mesh_dp_cp} is missing mesh_tp; the operand with axes {mesh_cp_tp} is missing mesh_dp.

  In add(
    args[0]: f32[4] {mesh_dp_cp: V},
    args[1]: f32[4] {mesh_cp_tp: R},
  )""",
            )

    def test_auto_reinterpret_with_shared_axis(self) -> None:
        """Auto-reinterpret keeps shared axes and remaps foreign ones."""
        mesh = frozenset(
            {
                normalize_axis(self.dp),
                normalize_axis(self.cp),
                normalize_axis(self.tp),
            }
        )
        with typecheck(strict_mode="strict"):
            # src typed on {dp_cp: R, tp: V}, mesh has {dp, cp, tp}
            # tp is shared, dp_cp is foreign -> should map to {dp: R, cp: R}
            x = torch.randn(4)
            assert_type(x, {self.dp_cp: R, self.tp: V})
            with set_current_mesh(mesh):
                result = torch.add(x, x)
                self.assertIs(get_axis_local_type(result, self.dp), R)
                self.assertIs(get_axis_local_type(result, self.cp), R)
                self.assertIs(get_axis_local_type(result, self.tp), V)

    def test_reinterpret_mesh_flatten_unflatten(self) -> None:
        """Explicit reinterpret_mesh should allow dp,cp <-> dp_cp retagging."""
        from spmd_types._checker import reinterpret_mesh

        with typecheck(strict_mode="strict"):
            x = torch.randn(4)
            assert_type(x, {self.dp: R, self.cp: R})
            y = reinterpret_mesh(x, {self.dp_cp: R})
            self.assertEqual(
                get_local_type(x),
                {normalize_axis(self.dp): R, normalize_axis(self.cp): R},
            )
            self.assertEqual(get_local_type(y), {normalize_axis(self.dp_cp): R})

            z = torch.randn(4)
            assert_type(z, {self.dp_cp: V})
            out = y + z
            self.assertIs(get_axis_local_type(out, self.dp_cp), V)

    def test_reinterpret_mesh_parallel_folding(self) -> None:
        """Explicit reinterpret_mesh should handle tuple-to-tuple retagging."""
        from spmd_types._checker import reinterpret_mesh

        with typecheck(strict_mode="strict"):
            dp = MeshAxis.of(2, 8)
            cp = MeshAxis.of(2, 4)
            tp = MeshAxis.of(4, 1)
            edp = MeshAxis.of(4, 4)
            ep = MeshAxis.of(2, 2)
            etp = MeshAxis.of(2, 1)

            x = torch.randn(4)
            assert_type(x, {dp: V, cp: V, tp: V})
            y = reinterpret_mesh(x, {edp: V, ep: V, etp: V})
            self.assertEqual(get_local_type(y), {edp: V, ep: V, etp: V})

            z = torch.randn(4)
            assert_type(z, {edp: R, ep: R, etp: R})
            out = y + z
            self.assertIs(get_axis_local_type(out, edp), V)
            self.assertIs(get_axis_local_type(out, ep), V)
            self.assertIs(get_axis_local_type(out, etp), V)

    def test_reinterpret_mesh_rejects_mixed_group_types(self) -> None:
        """Mixed local types within a grouped region are not compatible."""
        from spmd_types._checker import reinterpret_mesh

        with typecheck(strict_mode="strict"):
            x = torch.randn(4)
            assert_type(x, {self.dp: V, self.cp: R})
            with self.assertRaises(SpmdTypeError) as cm:
                reinterpret_mesh(x, {self.dp_cp: V})
            self.assertExpectedInline(
                str(cm.exception),
                """\
Axes {mesh_dp, mesh_cp} would need to be flattened, but mesh_dp has type V while mesh_cp has type R.

  In reinterpret_mesh(
    tensor: f32[4] {mesh_dp: V, mesh_cp: R},
    type: {mesh_dp_cp: V},
  )""",
            )

    def test_reinterpret_mesh_rejects_different_rank_patterns(self) -> None:
        """Axes with the same size but different rank patterns are not compatible."""
        from spmd_types._checker import reinterpret_mesh
        from spmd_types._mesh_axis import _register_name

        with typecheck(strict_mode="strict"):
            x = torch.randn(4)
            # Two axes of size 4 but with different strides (different rank patterns).
            a = MeshAxis.of(4, 1)
            b = MeshAxis.of(4, 2)
            _register_name(b, "mesh_ep")
            assert_type(x, {a: R})
            self.assertExpectedRaisesInline(
                SpmdTypeError,
                lambda: reinterpret_mesh(x, {b: R}),
                """\
Source group {mesh_tp} and destination group {mesh_ep} flatten to different rank regions. Set MESH_AXIS_DEBUG=1 to see underlying size/stride layouts.

  In reinterpret_mesh(
    tensor: f32[4] {mesh_tp: R},
    type: {mesh_ep: R},
  )""",
            )

    def test_reinterpret_mesh_remaps_partition_spec(self) -> None:
        """reinterpret_mesh remaps PartitionSpec across mesh-axis flattening."""
        from spmd_types._checker import reinterpret_mesh
        from spmd_types.runtime import get_partition_spec

        with typecheck(strict_mode="strict"):
            x = torch.randn(4, 4)
            assert_type(
                x, {self.dp: V, self.cp: V}, PartitionSpec((self.dp, self.cp), None)
            )
            y = reinterpret_mesh(x, {self.dp_cp: V})
            self.assertEqual(get_partition_spec(y), PartitionSpec(self.dp_cp, None))

    def test_reinterpret_mesh_axes_only_flatten_unflatten(self) -> None:
        """axes-only reinterpret_mesh: dp,cp -> dp_cp derives types from source."""
        from spmd_types._checker import reinterpret_mesh

        with typecheck(strict_mode="strict"):
            x = torch.randn(4)
            assert_type(x, {self.dp: R, self.cp: R})
            y = reinterpret_mesh(x, [self.dp_cp])
            self.assertIs(get_axis_local_type(y, self.dp_cp), R)

    def test_reinterpret_mesh_axes_only_parallel_folding(self) -> None:
        """axes-only reinterpret_mesh: dp,cp,tp -> edp,ep,etp derives V from source."""
        from spmd_types._checker import reinterpret_mesh

        with typecheck(strict_mode="strict"):
            expert_mesh = init_device_mesh(
                "cpu", (4, 2, 2), mesh_dim_names=("edp", "ep", "etp")
            )
            edp = expert_mesh.get_group("edp")
            ep = expert_mesh.get_group("ep")
            etp = expert_mesh.get_group("etp")

            x = torch.randn(4)
            assert_type(x, {self.dp: V, self.cp: V, self.tp: V})
            y = reinterpret_mesh(x, [edp, ep, etp])
            self.assertIs(get_axis_local_type(y, edp), V)
            self.assertIs(get_axis_local_type(y, ep), V)
            self.assertIs(get_axis_local_type(y, etp), V)

    def test_reinterpret_mesh_axes_only_rejects_mixed_types(self) -> None:
        """axes-only reinterpret_mesh rejects I/V mismatch within a flattened group."""
        from spmd_types._checker import reinterpret_mesh

        with typecheck(strict_mode="strict"):
            x = torch.randn(4)
            assert_type(x, {self.dp: V, self.cp: I})
            self.assertExpectedRaisesInline(
                SpmdTypeError,
                lambda: reinterpret_mesh(x, [self.dp_cp]),
                """\
Axes {mesh_dp, mesh_cp} would need to be flattened, but mesh_dp has type V while mesh_cp has type I. Check that the operand has the local SPMD types you expect and that the current mesh is correct for this region of code.

  In reinterpret_mesh(
    tensor: f32[4] {mesh_dp: V, mesh_cp: I},
    type: frozenset({mesh_dp_cp}),
  )""",
            )


class TestValidatePartitionSpecForGlobalSpmd(LocalTensorTestCase):
    """Test _validate_partition_spec_for_global_spmd bidirectional consistency."""

    def test_passes_s_with_partition_spec(self):
        """S(i) axis (V + PartitionSpec) passes validation in global mode."""
        x = torch.zeros(4, 3)
        assert_type(x, {self.pg: S(0)})
        _validate_partition_spec_for_global_spmd(
            get_local_type(x), get_partition_spec(x)
        )

    def test_passes_replicate_without_partition_spec(self):
        """R axis without PartitionSpec passes validation."""
        x = torch.zeros(4, 3)
        assert_type(x, {self.pg: R})
        _validate_partition_spec_for_global_spmd(
            get_local_type(x), get_partition_spec(x)
        )

    def test_fails_v_without_partition_spec(self):
        """V axis without PartitionSpec raises SpmdTypeError."""
        x = torch.zeros(4, 3)
        assert_type(x, {self.pg: V})
        with self.assertRaises(SpmdTypeError):
            _validate_partition_spec_for_global_spmd(
                get_local_type(x), get_partition_spec(x)
            )

    def test_global_mode_rejects_v_without_spec(self):
        """global region rejects V-typed tensors without PartitionSpec."""
        with typecheck(local=True):
            x = torch.zeros(4, 3)
            assert_type(x, {self.pg: V})
            torch.add(x, x)
            with typecheck(local=False):
                with self.assertRaises(SpmdTypeError):
                    torch.add(x, x)


class TestBackwardTypeCheck(SpmdTypeCheckedTestCase, expecttest.TestCase):
    """Test that backward() validates loss type is Invariant or Partial."""

    def test_backward_invariant_loss_ok(self):
        """backward() on Invariant loss should succeed."""
        loss_i = self._generate_inputs((), self.pg, I)
        loss_i.requires_grad_(True)
        y = loss_i * 1.0  # I * scalar -> I
        y.backward()

    def test_backward_partial_loss_ok(self):
        """backward() on Partial loss should succeed."""
        x = self._generate_inputs((4,), self.pg, V)
        x.requires_grad_(True)
        with torch.no_grad():
            w = self._generate_inputs((4,), self.pg, V)
        s = (x * w).sum()  # V -> V scalar
        from spmd_types._local import reinterpret

        reinterpret(s, self.pg, src=V, dst=P).backward()

    def test_backward_replicate_loss_errors(self):
        """backward() on Replicate loss should raise SpmdTypeError."""
        x = self._generate_inputs((4,), self.pg, R)
        x.requires_grad_(True)
        loss = (x * x).sum()  # R -> R
        with self.assertRaises(SpmdTypeError) as ctx:
            loss.backward()
        self.assertExpectedInline(
            str(ctx.exception),
            """\
backward() on a Replicate loss on axis default_pg would produce incorrect gradients. The implicit grad_output is a 1.0 scalar on each rank (Invariant), but only Invariant or Partial losses are valid for backward() without an explicit gradient.
The loss is Replicate on axis default_pg -- this usually means an upstream all_reduce reduced to R instead of I. Consider:
  - Changing the upstream reduction to all_reduce(..., dst=I) so the loss is Invariant, or
  - Removing the all_reduce entirely and calling backward() on the local (Partial) loss, which is also more efficient""",
        )

    def test_backward_varying_loss_errors(self):
        """backward() on Varying loss should raise SpmdTypeError."""
        x = self._generate_inputs((4,), self.pg, V)
        x.requires_grad_(True)
        loss = (x * x).sum()  # V -> V
        with self.assertRaises(SpmdTypeError) as ctx:
            loss.backward()
        self.assertExpectedInline(
            str(ctx.exception),
            """\
backward() on a Varying loss on axis default_pg would produce incorrect gradients. The implicit grad_output is a 1.0 scalar on each rank (Invariant), but only Invariant or Partial losses are valid for backward() without an explicit gradient.
The loss is Varying -- each rank holds a different local loss. Use reinterpret(loss, default_pg, src=V, dst=P) to declare that the semantics is that you want to differentiate with regards to the (pending) reduction of the loss on all ranks""",
        )

    def test_backward_with_explicit_gradient_skips_check(self):
        """backward() with explicit gradient kwarg should skip the loss type check."""
        x = self._generate_inputs((4,), self.pg, R)
        x.requires_grad_(True)
        loss = (x * x).sum()  # R -> R
        grad = self._generate_inputs((), self.pg, P)
        loss.backward(gradient=grad)

    def test_backward_with_explicit_gradient_positional_skips_check(self):
        """backward() with explicit gradient as positional arg should skip the check."""
        x = self._generate_inputs((4,), self.pg, R)
        x.requires_grad_(True)
        loss = (x * x).sum()  # R -> R
        grad = self._generate_inputs((), self.pg, P)
        loss.backward(grad)

    def test_backward_untyped_loss_strict_errors(self):
        """backward() on untyped loss in strict mode should raise SpmdTypeError."""
        x = self.rank_map(lambda r: torch.randn(4))
        x.requires_grad_(True)
        with no_typecheck():
            loss = (x * x).sum()
        with self.assertRaises(SpmdTypeError) as ctx:
            loss.backward()
        self.assertExpectedInline(
            str(ctx.exception),
            """\
backward() called on a loss tensor with no SPMD type annotation. In strict mode, annotate the loss with assert_type() before calling backward().""",
        )

    def test_backward_untyped_loss_permissive_ok(self):
        """backward() on untyped loss in permissive mode should pass through."""
        x = self.rank_map(lambda r: torch.randn(4))
        x.requires_grad_(True)
        with no_typecheck():
            loss = (x * x).sum()
        with typecheck(strict_mode="permissive"):
            loss.backward()


class TestScalarWithPartitionSpec(SpmdTypeCheckedTestCase):
    """Verify ops with Scalar args work when tensor has PartitionSpec."""

    def test_narrow_with_scalar_and_partition_spec(self):
        x = torch.randn(4, 8)
        assert_type(x, {self.pg: S(1)})
        result = torch.narrow(x, 1, Scalar(0, {self.pg: V}), Scalar(4, {self.pg: V}))
        self.assertIs(get_axis_local_type(result, self.pg), V)


class TestLocalMap(LocalTensorTestCase, expecttest.TestCase):
    """Tests for ``local_map``."""

    def test_basic(self):
        """Validates inputs, runs body opaquely, stamps outputs (local type + PartitionSpec)."""
        with typecheck():
            x = self._generate_inputs((4, 3), self.pg, S(0))
            w = self._generate_inputs((3, 5), self.pg, R)

            @local_map(
                in_types=({self.pg: S(0)}, {self.pg: R}),
                # Wrong on purpose: shows no spmd type propagates inside fn.
                out_types={self.pg: S(1)},
            )
            def fn(x, w):
                return x @ w

            y = fn(x, w)
            self.assertIs(get_axis_local_type(y, self.pg), V)
            self.assertEqual(get_partition_spec(y), PartitionSpec(None, self.pg))

    def test_pytree_flatten(self):
        """``in_types`` / ``out_types`` mirror the structure of args / return."""
        with typecheck():
            x = self._generate_inputs((4, 6), self.pg, S(0))
            w1 = self._generate_inputs((6, 5), self.pg, R)
            w2 = self._generate_inputs((6, 5), self.pg, R)

            @local_map(
                in_types=({self.pg: S(0)}, [{self.pg: R}, {self.pg: R}]),
                out_types=({self.pg: S(0)}, {self.pg: S(0)}),
            )
            def fn(x, weights):
                return x @ weights[0], x @ weights[1]

            a, b = fn(x, [w1, w2])
            for t in (a, b):
                self.assertIs(get_axis_local_type(t, self.pg), V)
                self.assertEqual(get_partition_spec(t), PartitionSpec(self.pg, None))

    def test_explicit_partition_spec_and_non_tensor(self):
        """Tuple spec carries an explicit ``PartitionSpec``; ``None`` skips a non-tensor arg."""
        with typecheck():
            x = self.rank_map(lambda r: torch.randn(4, 3))
            assert_type(x, {self.pg: S(0)})  # V + PartitionSpec(self.pg, None)

            @local_map(
                in_types=(({self.pg: V}, PartitionSpec(self.pg, None)), None),
                out_types=({self.pg: V}, PartitionSpec(None, None, self.pg)),
            )
            def fn(x, scale):
                # [4, 3] -> scale -> unsqueeze -> transpose so the sharded dim
                # ends up on axis 2 (where the output PartitionSpec expects).
                return (x * scale).unsqueeze(0).transpose(1, 2)

            y = fn(x, 2.0)
            self.assertIs(get_axis_local_type(y, self.pg), V)
            self.assertEqual(get_partition_spec(y), PartitionSpec(None, None, self.pg))

    def test_bare_partition_spec_leaf(self):
        """Bare ``PartitionSpec`` leaf -- local type V is inferred from the spec."""
        with typecheck():
            x = self.rank_map(lambda r: torch.randn(4, 3))
            assert_type(x, {self.pg: S(0)})  # V + PartitionSpec(self.pg, None)

            @local_map(
                in_types=(PartitionSpec(self.pg, None), None),
                out_types=PartitionSpec(None, None, self.pg),
            )
            def fn(x, scale):
                return (x * scale).unsqueeze(0).transpose(1, 2)

            y = fn(x, 2.0)
            # Output stamp: V on self.pg with the declared PartitionSpec.
            self.assertIs(get_axis_local_type(y, self.pg), V)
            self.assertEqual(get_partition_spec(y), PartitionSpec(None, None, self.pg))

    def test_infer_input_default(self):
        """``in_types`` defaults to bare ``Infer`` broadcast: no input checks.

        Mixed tensor/non-tensor args still pass through without complaint.
        """
        with typecheck():
            x = self._generate_inputs((4, 3), self.pg, R)

            @local_map(out_types={self.pg: R})
            def fn(x, scale):
                return x * scale

            y = fn(x, 2.0)
            self.assertIs(get_axis_local_type(y, self.pg), R)

    def test_infer_rejected_at_leaves(self):
        """``Infer`` is only valid as the bare value of ``in_types``; at a
        leaf -- in either ``in_types`` or ``out_types`` -- it raises."""
        with typecheck():
            x = self._generate_inputs((4, 3), self.pg, R)

            # Infer at an in_types leaf: rejected.
            @local_map(in_types=(Infer,), out_types={self.pg: R})
            def infer_leaf_in(x):
                return x.clone()

            with self.assertRaises(SpmdTypeError) as ctx:
                infer_leaf_in(x)
            self.assertExpectedInline(
                str(ctx.exception),
                """local_map[TestLocalMap.test_infer_rejected_at_leaves.<locals>.infer_leaf_in]: input_types contains Infer at a leaf; only the bare value of in_types may be Infer.""",
            )

            # Infer in out_types (bare or as a leaf): rejected.
            @local_map(in_types=({self.pg: R},), out_types=Infer)
            def infer_bare_out(x):
                return x.clone()

            with self.assertRaises(SpmdTypeError) as ctx:
                infer_bare_out(x)
            self.assertExpectedInline(
                str(ctx.exception),
                """local_map[TestLocalMap.test_infer_rejected_at_leaves.<locals>.infer_bare_out]: output_types contains Infer at a leaf; only the bare value of in_types may be Infer.""",
            )

    def test_structure_mismatch(self):
        """``in_types`` whose pytree structure differs from args raises ``SpmdTypeError``."""
        with typecheck():
            x = self._generate_inputs((4, 3), self.pg, R)

            @local_map(
                in_types=({self.pg: R}, {self.pg: R}),  # 2 specs for 1 arg
                out_types={self.pg: R},
            )
            def fn(x):
                return x.clone()

            with self.assertRaisesRegex(
                SpmdTypeError,
                r"input_types pytree structure does not match input",
            ):
                fn(x)

    def test_input_validation(self):
        """Boundary checks reject mismatches: local type, PartitionSpec, tensor/None alignment."""
        with typecheck():
            x = self._generate_inputs((4, 3), self.pg, R)

            # Local type mismatch (R vs declared V).
            @local_map(in_types=({self.pg: V},), out_types={self.pg: V})
            def bad_local_type(x):
                return x.clone()

            with self.assertRaises(SpmdTypeError):
                bad_local_type(x)

            # Tensor input with ``None`` spec.
            @local_map(in_types=(None,), out_types={self.pg: R})
            def tensor_with_none(x):
                return x.clone()

            with self.assertRaises(SpmdTypeError):
                tensor_with_none(x)

            # PartitionSpec mismatch.
            x_sharded = self.rank_map(lambda r: torch.randn(4, 3))
            assert_type(x_sharded, {self.pg: S(0)})  # PartitionSpec(self.pg, None)

            @local_map(
                in_types=(({self.pg: V}, PartitionSpec(None, self.pg)),),
                out_types={self.pg: V},
            )
            def bad_pspec(x):
                return x.clone()

            with self.assertRaises(SpmdTypeError):
                bad_pspec(x_sharded)


if __name__ == "__main__":
    unittest.main()
