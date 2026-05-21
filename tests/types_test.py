# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for SPMD type hierarchy (R, I, V, P, S) and PartitionSpec.

Covers: types.py.
"""

import unittest

import expecttest
from spmd_types import (
    I,
    Invariant,
    normalize_partition_spec,
    P,
    Partial,
    PartitionSpec,
    PerMeshAxisLocalSpmdType,
    R,
    Replicate,
    S,
    set_current_mesh,
    V,
    Varying,
)
from spmd_types._mesh_axis import _register_name, _reset, MeshAxis
from spmd_types.types import (
    partition_spec_get_shard,
    partition_spec_to_shard_types,
    shard_types_to_partition_spec,
    SpmdTypeError,
)


class _NamedMeshAxisTestCase(expecttest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # Distinct MeshAxis objects for testing.  Different strides ensure they are
        # not equal to each other.
        cls.tp = MeshAxis.of(4, stride=1)  # ranks {0,1,2,3}
        cls.dp = MeshAxis.of(4, stride=4)  # ranks {0,4,8,12}
        cls.ep = MeshAxis.of(4, stride=16)  # ranks {0,16,32,48}
        cls.cp = MeshAxis.of(1, stride=64)  # singleton axis
        _register_name(cls.tp, "tp")
        _register_name(cls.dp, "dp")
        _register_name(cls.ep, "ep")
        _register_name(cls.cp, "cp")

    @classmethod
    def tearDownClass(cls) -> None:
        _reset()
        super().tearDownClass()


class TestTypeEnum(unittest.TestCase):
    """Test that R, I, V, P are enum members and aliases work."""

    def test_enum_identity(self):
        """R, I, V, P should be PerMeshAxisLocalSpmdType enum members."""
        self.assertIs(R, PerMeshAxisLocalSpmdType.R)
        self.assertIs(I, PerMeshAxisLocalSpmdType.I)
        self.assertIs(V, PerMeshAxisLocalSpmdType.V)
        self.assertIs(P, PerMeshAxisLocalSpmdType.P)

    def test_isinstance(self):
        """Enum members should be instances of PerMeshAxisLocalSpmdType."""
        self.assertIsInstance(R, PerMeshAxisLocalSpmdType)
        self.assertIsInstance(I, PerMeshAxisLocalSpmdType)
        self.assertIsInstance(V, PerMeshAxisLocalSpmdType)
        self.assertIsInstance(P, PerMeshAxisLocalSpmdType)

    def test_backward_compat_aliases(self):
        """Backward compat aliases should be the same enum members."""
        self.assertIs(Replicate, R)
        self.assertIs(Invariant, I)
        self.assertIs(Varying, V)
        self.assertIs(Partial, P)

    def test_shard_equality(self):
        """Shard objects with same dim should be equal."""
        self.assertEqual(S(0), S(0))
        self.assertEqual(S(1), S(1))
        self.assertNotEqual(S(0), S(1))
        self.assertNotEqual(S(0), V)

    def test_type_repr(self):
        """Test string representation of types."""
        self.assertEqual(repr(R), "R")
        self.assertEqual(repr(I), "I")
        self.assertEqual(repr(V), "V")
        self.assertEqual(repr(P), "P")
        self.assertEqual(repr(S(0)), "S(0)")
        self.assertEqual(repr(S(1)), "S(1)")


class TestBackwardType(unittest.TestCase):
    """Test backward_type method for all types."""

    def test_replicate_backward_type(self):
        """Replicate backward type is Partial."""
        self.assertIs(R.backward_type(), P)

    def test_invariant_backward_type(self):
        """Invariant backward type is Invariant."""
        self.assertIs(I.backward_type(), I)

    def test_varying_backward_type(self):
        """Varying backward type is Varying."""
        self.assertIs(V.backward_type(), V)

    def test_partial_backward_type(self):
        """Partial backward type is Replicate."""
        self.assertIs(P.backward_type(), R)

    def test_shard_backward_type(self):
        """Shard backward type is the same Shard."""
        s = S(0)
        self.assertEqual(s.backward_type(), s)
        s2 = S(1)
        self.assertEqual(s2.backward_type(), s2)


class TestPartitionSpec(_NamedMeshAxisTestCase):
    """Test PartitionSpec for Global SPMD."""

    def test_empty_partition_spec(self):
        """Empty partition spec is fully replicated."""
        ps = PartitionSpec()
        self.assertEqual(ps, PartitionSpec())

    def test_single_axis_partition_spec(self):
        """Partition spec with single mesh axis."""
        ps = PartitionSpec(self.tp, None)
        self.assertEqual(ps, PartitionSpec(self.tp, None))

    def test_multi_axis_partition_spec(self):
        """Partition spec with multiple mesh axes on same dim."""
        ps = PartitionSpec((self.dp, self.tp), None)
        self.assertEqual(ps, PartitionSpec((self.dp, self.tp), None))

    def test_partition_spec_iteration(self):
        """Partition spec should be iterable."""
        ps = PartitionSpec(self.tp, self.dp, None)
        axes = list(ps)
        self.assertEqual(axes, [self.tp, self.dp, None])

    def test_partition_spec_repr(self):
        """Test string representation of PartitionSpec."""
        ps = PartitionSpec()
        self.assertEqual(repr(ps), "PartitionSpec()")

        ps = PartitionSpec(self.tp, None)
        self.assertEqual(repr(ps), f"PartitionSpec({self.tp!r}, None)")

        ps = PartitionSpec((self.dp, self.tp), self.ep)
        self.assertEqual(
            repr(ps), f"PartitionSpec(({self.dp!r}, {self.tp!r}), {self.ep!r})"
        )


class TestShardTypesToPartitionSpec(_NamedMeshAxisTestCase):
    """Test shard_types_to_partition_spec conversion."""

    def test_basic(self):
        """Single axis -> single-entry spec (no axis_order needed)."""
        spec = shard_types_to_partition_spec({self.tp: S(0)}, 2)
        self.assertEqual(spec, PartitionSpec(self.tp, None))

    def test_multi_dim(self):
        """Two axes on different dims (no axis_order needed)."""
        spec = shard_types_to_partition_spec({self.dp: S(0), self.tp: S(1)}, 2)
        self.assertEqual(spec, PartitionSpec(self.dp, self.tp))

    def test_multi_axis_same_dim(self):
        """Two axes on the same dim uses axis_order for ordering."""
        spec = shard_types_to_partition_spec(
            {self.tp: S(0), self.dp: S(0)}, 2, axis_order=[self.dp, self.tp]
        )
        self.assertEqual(spec, PartitionSpec((self.dp, self.tp), None))

    def test_multi_axis_same_dim_no_order_rejected(self):
        """Two axes on the same dim without axis_order raises."""
        with self.assertRaises(SpmdTypeError) as ctx:
            shard_types_to_partition_spec({self.tp: S(0), self.dp: S(0)}, 2)
        self.assertExpectedInline(
            str(ctx.exception),
            """Tensor dim 0 is sharded on multiple axes (tp, dp) but axis_order does not resolve their ordering. Pass all conflicting axes in axis_order.""",
        )

    def test_axis_order_subset(self):
        """axis_order can be a subset covering only the conflicting dim."""
        types = {self.dp: S(0), self.tp: S(0), self.ep: S(1)}
        spec = shard_types_to_partition_spec(types, 2, axis_order=[self.dp, self.tp])
        self.assertEqual(spec, PartitionSpec((self.dp, self.tp), self.ep))

    def test_axis_order_insufficient_raises(self):
        """axis_order that doesn't resolve a conflict raises."""
        with self.assertRaises(SpmdTypeError) as ctx:
            shard_types_to_partition_spec(
                {self.dp: S(0), self.tp: S(0)}, 2, axis_order=[self.dp]
            )
        self.assertExpectedInline(
            str(ctx.exception),
            """Tensor dim 0 is sharded on multiple axes (dp, tp) but axis_order does not resolve their ordering. Pass all conflicting axes in axis_order.""",
        )

    def test_axis_order_does_not_mutate_input(self):
        """axis_order path must not mutate the caller's dict."""
        types = {self.dp: S(0), self.tp: S(0), self.ep: S(1)}
        original = dict(types)
        shard_types_to_partition_spec(types, 2, axis_order=[self.dp, self.tp])
        self.assertEqual(types, original)

    def test_ignores_non_shard(self):
        """R/I/P entries are ignored (no axis_order needed)."""
        spec = shard_types_to_partition_spec({self.dp: R, self.tp: S(1)}, 3)
        self.assertEqual(spec, PartitionSpec(None, self.tp, None))
        spec = shard_types_to_partition_spec({self.dp: P, self.tp: S(1)}, 3)
        self.assertEqual(spec, PartitionSpec(None, self.tp, None))


class TestPartitionSpecToShardTypes(_NamedMeshAxisTestCase):
    """Test partition_spec_to_shard_types conversion."""

    def test_basic(self):
        """Single-entry spec -> S(0)."""
        shards = partition_spec_to_shard_types(PartitionSpec(self.tp, None))
        self.assertEqual(shards, {self.tp: S(0)})

    def test_multi_dim(self):
        """Axes on different dims."""
        shards = partition_spec_to_shard_types(PartitionSpec(self.dp, self.tp))
        self.assertEqual(shards, {self.dp: S(0), self.tp: S(1)})

    def test_multi_axis_same_dim(self):
        """Multi-axis tuple preserves inner order."""
        shards = partition_spec_to_shard_types(PartitionSpec((self.tp, self.dp), None))
        self.assertEqual(shards, {self.tp: S(0), self.dp: S(0)})

    def test_empty(self):
        shards = partition_spec_to_shard_types(PartitionSpec(None, None))
        self.assertEqual(shards, {})

    def test_duplicate_axis_rejected(self):
        """Axis at two different dims raises SpmdTypeError."""
        with self.assertRaises(SpmdTypeError) as ctx:
            partition_spec_to_shard_types(PartitionSpec(self.tp, self.tp))
        self.assertExpectedInline(
            str(ctx.exception),
            """Mesh axes tp and tp are not orthogonal (they share ranks). All axes in a LocalSpmdType must be mutually orthogonal.""",
        )

    def test_round_trip_single_axis_per_dim(self):
        """PartitionSpec -> dict -> spec works when each dim has one axis."""
        spec = PartitionSpec(self.dp, self.ep)
        shards = partition_spec_to_shard_types(spec)
        rebuilt = shard_types_to_partition_spec(shards, 2)
        self.assertEqual(rebuilt, spec)

    def test_string_axes_resolve_before_conversion(self):
        with set_current_mesh({"tp": self.tp}):
            shards = partition_spec_to_shard_types(PartitionSpec("tp", None))
        self.assertEqual(shards, {self.tp: S(0)})

    def test_get_shard_accepts_unnormalized_spec(self):
        with set_current_mesh({"tp": self.tp}):
            self.assertEqual(
                partition_spec_get_shard(PartitionSpec("tp"), self.tp), S(0)
            )


class TestNormalizePartitionSpec(_NamedMeshAxisTestCase):
    """Test explicit PartitionSpec normalization."""

    def test_constructor_preserves_raw_entries(self):
        spec = PartitionSpec("tp", (), ("dp", "cp"))
        self.assertEqual(spec, ("tp", (), ("dp", "cp")))

    def test_resolves_names_and_drops_singletons(self):
        with set_current_mesh({"tp": self.tp, "cp": self.cp}):
            spec = normalize_partition_spec(PartitionSpec(None, ("tp", "cp"), "cp"))
        self.assertEqual(spec, PartitionSpec(None, self.tp, None))

    def test_empty_tuple_rejected_by_normalize(self):
        with self.assertRaises(ValueError):
            normalize_partition_spec(PartitionSpec(()))


if __name__ == "__main__":
    unittest.main()
