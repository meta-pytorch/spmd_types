"""
Tests that spmd_types frontend constructs are compatible with torch.compile.

Type checking is NOT expected to work under Dynamo -- these tests verify that
code which *uses* spmd_types constructs can be traced by Dynamo when type
checking is off.  This covers every public API in spmd_types that is
not about turning on type checking itself.
"""

import unittest

import torch
import torch.distributed as dist
from spmd_types import (
    all_gather,
    all_reduce,
    all_to_all,
    assert_type,
    convert,
    get_local_type,
    I,
    invariant_to_replicate,
    is_type_checking,
    mutate_type,
    no_typecheck,
    P,
    R,
    redistribute,
    reduce_scatter,
    reinterpret,
    reinterpret_mesh,
    S,
    shard,
    TYPE_CHECKING,
    unshard,
    V,
)
from spmd_types._checker import trace
from spmd_types._mesh_axis import _reset
from spmd_types.types import normalize_axis
from torch.testing._internal.distributed.fake_pg import FakeStore


def _compile(f):
    torch._dynamo.reset()
    return torch.compile(f, backend="eager", fullgraph=True)


def _assert_dynamo_ok(test, f, *args, check_result=True):
    """Run f eagerly and compiled, assert results match.

    Args:
        check_result: If False, only verify that compilation succeeds
            (the compiled function runs without error) but don't compare
            the output to eager.  Useful when the fake distributed backend
            produces nondeterministic results.
    """
    expected = f(*args)
    compiled = _compile(f)
    result = compiled(*args)
    if check_result:
        if isinstance(expected, torch.Tensor):
            test.assertTrue(isinstance(result, torch.Tensor))
            torch.testing.assert_close(result, expected)
        else:
            test.assertEqual(result, expected)


class TestDynamoGuards(unittest.TestCase):
    """Guards that gate type-checking code in model forward passes."""

    def test_type_checking_sentinel(self):
        """if TYPE_CHECKING: ... is dead-code-eliminated by Dynamo."""

        def f(x):
            if TYPE_CHECKING:
                x = x + 999
            return x * 2

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_is_type_checking_call(self):
        """is_type_checking() call traces correctly under Dynamo."""

        def f(x):
            if is_type_checking():
                x = x + 999
            return x * 2

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_no_typecheck_context_manager(self):
        """no_typecheck() context manager traces correctly under Dynamo."""

        def f(x):
            with no_typecheck():
                y = x * 2
            return y + 1

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_trace_context_manager(self):
        """trace() context manager traces correctly under Dynamo."""

        def f(x):
            with trace(enabled=False):
                y = x * 2
            return y + 1

        _assert_dynamo_ok(self, f, torch.randn(4))


class TestDynamoTypeAnnotationAPIs(unittest.TestCase):
    """Type annotation APIs: assert_type, get_local_type, mutate_type, reinterpret_mesh."""

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
        cls.pg = dist.distributed_c10d._get_default_group()

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def test_assert_type(self):
        """assert_type() can be called inside Dynamo-traced code."""
        pg = self.pg

        def f(x):
            assert_type(x, {pg: R})
            return x * 2

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_get_local_type(self):
        """get_local_type() can be called inside Dynamo-traced code."""

        def f(x):
            _ = get_local_type(x)
            return x * 2

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_get_local_type_on_typed_tensor(self):
        """get_local_type() on a tensor with SPMD type annotation."""
        pg = self.pg

        def f(x):
            assert_type(x, {pg: R})
            lt = get_local_type(x)
            if lt:
                return x * 3
            return x * 2

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_mutate_type(self):
        """mutate_type() can be called inside Dynamo-traced code."""
        pg = self.pg
        axis = normalize_axis(pg)

        def f(x):
            assert_type(x, {pg: R})
            mutate_type(x, axis, src=R, dst=V)
            return x * 2

        # Cannot use _assert_dynamo_ok: the eager call mutates the tensor's
        # type annotation (R -> V), so re-running on the same tensor would
        # fail in assert_type.  Verify eager and compiled on separate tensors.
        x = torch.randn(4)
        expected = f(x.clone())
        compiled = _compile(f)
        result = compiled(x.clone())
        torch.testing.assert_close(result, expected)

    def test_reinterpret_mesh(self):
        """reinterpret_mesh() can be called inside Dynamo-traced code."""
        pg = self.pg

        def f(x):
            assert_type(x, {pg: R})
            reinterpret_mesh(x, {pg: R})
            return x * 2

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_guarded_assert_type(self):
        """Pattern from attention.py: if TYPE_CHECKING: assert_type(...)."""
        pg = self.pg

        def f(x):
            if TYPE_CHECKING:
                assert_type(x, {pg: R})
            return x * 2

        _assert_dynamo_ok(self, f, torch.randn(4))


class TestDynamoLocalOps(unittest.TestCase):
    """Local (no-comms) operations: reinterpret, convert, shard, invariant_to_replicate."""

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
        cls.pg = dist.distributed_c10d._get_default_group()

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def test_reinterpret_V_P(self):
        """reinterpret(V, P) is a no-op forward, should trace."""
        pg = self.pg

        def f(x):
            return reinterpret(x, pg, src=V, dst=P)

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_convert_I_R(self):
        """convert(I, R) is a no-op forward, should trace."""
        pg = self.pg

        def f(x):
            return convert(x, pg, src=R, dst=R)

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_convert_R_S(self):
        """convert(R, S(0)) chunks the tensor, should trace."""
        pg = self.pg

        def f(x):
            return convert(x, pg, src=R, dst=S(0))

        _assert_dynamo_ok(self, f, torch.randn(self.WORLD_SIZE * 4))

    def test_convert_R_P(self):
        """convert(R, P) zeros non-rank-0, should trace."""
        pg = self.pg

        def f(x):
            return convert(x, pg, src=R, dst=P)

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_shard(self):
        """shard(R, S(1)) convenience alias should trace."""
        pg = self.pg

        def f(x):
            return shard(x, pg, src=R, dst=S(1))

        _assert_dynamo_ok(self, f, torch.randn(4, self.WORLD_SIZE * 2))

    def test_invariant_to_replicate(self):
        """invariant_to_replicate() convenience alias should trace."""
        pg = self.pg

        def f(x):
            return invariant_to_replicate(x, pg)

        _assert_dynamo_ok(self, f, torch.randn(4))


class TestDynamoCollectives(unittest.TestCase):
    """Collective operations: all_reduce, all_gather, reduce_scatter, all_to_all, redistribute, unshard."""

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
        cls.pg = dist.distributed_c10d._get_default_group()

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def test_all_reduce(self):
        """all_reduce(P, R) should trace."""
        pg = self.pg

        def f(x):
            return all_reduce(x, pg, src=P, dst=R)

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_all_reduce_dst_I(self):
        """all_reduce(P, I) should trace."""
        pg = self.pg

        def f(x):
            return all_reduce(x, pg, src=P, dst=I)

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_all_gather_V_R(self):
        """all_gather(V, R) with stack semantics should trace."""
        pg = self.pg

        def f(x):
            return all_gather(x, pg, src=V, dst=R)

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_all_gather_S_R(self):
        """all_gather(S(0), R) with shard semantics should trace."""
        pg = self.pg

        def f(x):
            return all_gather(x, pg, src=S(0), dst=R)

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_reduce_scatter(self):
        """reduce_scatter(P, V) should trace."""
        pg = self.pg

        def f(x):
            return reduce_scatter(x, pg, src=P, dst=V)

        # check_result=False: fake backend's reduce_scatter_tensor is nondeterministic
        _assert_dynamo_ok(self, f, torch.randn(self.WORLD_SIZE, 4), check_result=False)

    def test_reduce_scatter_S(self):
        """reduce_scatter(P, S(0)) should trace."""
        pg = self.pg

        def f(x):
            return reduce_scatter(x, pg, src=P, dst=S(0))

        # check_result=False: fake backend's reduce_scatter_tensor is nondeterministic
        _assert_dynamo_ok(self, f, torch.randn(self.WORLD_SIZE * 4), check_result=False)

    @unittest.skip("dist.all_to_all is marked as skipped by Dynamo")
    def test_all_to_all_V_V(self):
        """all_to_all(V, V) should trace."""
        pg = self.pg

        def f(x):
            return all_to_all(x, pg, src=V, dst=V)

        _assert_dynamo_ok(self, f, torch.randn(self.WORLD_SIZE, 4))

    @unittest.skip("dist.all_to_all is marked as skipped by Dynamo")
    def test_all_to_all_S_S(self):
        """all_to_all(S(0), S(1)) should trace."""
        pg = self.pg

        def f(x):
            return all_to_all(x, pg, src=S(0), dst=S(1))

        _assert_dynamo_ok(
            self, f, torch.randn(self.WORLD_SIZE * 2, self.WORLD_SIZE * 2)
        )

    def test_redistribute_P_R(self):
        """redistribute(P, R) routes to all_reduce, should trace."""
        pg = self.pg

        def f(x):
            return redistribute(x, pg, src=P, dst=R)

        _assert_dynamo_ok(self, f, torch.randn(4))

    def test_redistribute_R_S(self):
        """redistribute(R, S(0)) routes to convert, should trace."""
        pg = self.pg

        def f(x):
            return redistribute(x, pg, src=R, dst=S(0))

        _assert_dynamo_ok(self, f, torch.randn(self.WORLD_SIZE * 4))

    def test_unshard(self):
        """unshard(S(0), R) convenience alias should trace."""
        pg = self.pg

        def f(x):
            return unshard(x, pg, src=S(0), dst=R)

        _assert_dynamo_ok(self, f, torch.randn(4))


if __name__ == "__main__":
    unittest.main()
