"""
Tests for SPMD type hierarchy (R, I, V, P, S) and PartitionSpec.

Covers: types.py.
"""

import unittest

from spmd_types import (
    I,
    Invariant,
    P,
    Partial,
    PartitionSpec,
    PerMeshAxisLocalSpmdType,
    R,
    Replicate,
    S,
    V,
    Varying,
)
from spmd_types._mesh_axis import MeshAxis
from spmd_types.types import (
    partition_spec_to_shard_types,
    shard_types_to_partition_spec,
    SpmdTypeError,
)

# Distinct MeshAxis objects for testing.  Different strides ensure they are
# not equal to each other.
tp = MeshAxis.of(4, stride=1)  # ranks {0,1,2,3}
dp = MeshAxis.of(4, stride=4)  # ranks {0,4,8,12}
ep = MeshAxis.of(4, stride=16)  # ranks {0,16,32,48}


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


class TestPartitionSpec(unittest.TestCase):
    """Test PartitionSpec for Global SPMD."""

    def test_empty_partition_spec(self):
        """Empty partition spec is fully replicated."""
        ps = PartitionSpec()
        self.assertEqual(ps, PartitionSpec())

    def test_single_axis_partition_spec(self):
        """Partition spec with single mesh axis."""
        ps = PartitionSpec(tp, None)
        self.assertEqual(ps, PartitionSpec(tp, None))

    def test_multi_axis_partition_spec(self):
        """Partition spec with multiple mesh axes on same dim."""
        ps = PartitionSpec((dp, tp), None)
        self.assertEqual(ps, PartitionSpec((dp, tp), None))

    def test_partition_spec_iteration(self):
        """Partition spec should be iterable."""
        ps = PartitionSpec(tp, dp, None)
        axes = list(ps)
        self.assertEqual(axes, [tp, dp, None])

    def test_partition_spec_repr(self):
        """Test string representation of PartitionSpec."""
        ps = PartitionSpec()
        self.assertEqual(repr(ps), "PartitionSpec()")

        ps = PartitionSpec(tp, None)
        self.assertEqual(repr(ps), f"PartitionSpec({tp!r}, None)")

        ps = PartitionSpec((dp, tp), ep)
        self.assertEqual(repr(ps), f"PartitionSpec(({dp!r}, {tp!r}), {ep!r})")


class TestShardTypesToPartitionSpec(unittest.TestCase):
    """Test shard_types_to_partition_spec conversion."""

    def test_basic(self):
        """Single axis -> single-entry spec (no axis_order needed)."""
        spec = shard_types_to_partition_spec({tp: S(0)}, 2)
        self.assertEqual(spec, PartitionSpec(tp, None))

    def test_multi_dim(self):
        """Two axes on different dims (no axis_order needed)."""
        spec = shard_types_to_partition_spec({dp: S(0), tp: S(1)}, 2)
        self.assertEqual(spec, PartitionSpec(dp, tp))

    def test_multi_axis_same_dim(self):
        """Two axes on the same dim uses axis_order for ordering."""
        spec = shard_types_to_partition_spec(
            {tp: S(0), dp: S(0)}, 2, axis_order=[dp, tp]
        )
        self.assertEqual(spec, PartitionSpec((dp, tp), None))

    def test_multi_axis_same_dim_no_order_rejected(self):
        """Two axes on the same dim without axis_order raises."""
        with self.assertRaises(SpmdTypeError):
            shard_types_to_partition_spec({tp: S(0), dp: S(0)}, 2)

    def test_axis_order_subset(self):
        """axis_order can be a subset covering only the conflicting dim."""
        types = {dp: S(0), tp: S(0), ep: S(1)}
        spec = shard_types_to_partition_spec(types, 2, axis_order=[dp, tp])
        self.assertEqual(spec, PartitionSpec((dp, tp), ep))

    def test_axis_order_insufficient_raises(self):
        """axis_order that doesn't resolve a conflict raises."""
        with self.assertRaises(SpmdTypeError):
            shard_types_to_partition_spec({dp: S(0), tp: S(0)}, 2, axis_order=[dp])

    def test_axis_order_does_not_mutate_input(self):
        """axis_order path must not mutate the caller's dict."""
        types = {dp: S(0), tp: S(0), ep: S(1)}
        original = dict(types)
        shard_types_to_partition_spec(types, 2, axis_order=[dp, tp])
        self.assertEqual(types, original)

    def test_ignores_non_shard(self):
        """R/I/P entries are ignored (no axis_order needed)."""
        spec = shard_types_to_partition_spec({dp: R, tp: S(1)}, 3)
        self.assertEqual(spec, PartitionSpec(None, tp, None))
        spec = shard_types_to_partition_spec({dp: P, tp: S(1)}, 3)
        self.assertEqual(spec, PartitionSpec(None, tp, None))


class TestPartitionSpecToShardTypes(unittest.TestCase):
    """Test partition_spec_to_shard_types conversion."""

    def test_basic(self):
        """Single-entry spec -> S(0)."""
        shards = partition_spec_to_shard_types(PartitionSpec(tp, None))
        self.assertEqual(shards, {tp: S(0)})

    def test_multi_dim(self):
        """Axes on different dims."""
        shards = partition_spec_to_shard_types(PartitionSpec(dp, tp))
        self.assertEqual(shards, {dp: S(0), tp: S(1)})

    def test_multi_axis_same_dim(self):
        """Multi-axis tuple preserves inner order."""
        shards = partition_spec_to_shard_types(PartitionSpec((tp, dp), None))
        self.assertEqual(shards, {tp: S(0), dp: S(0)})

    def test_empty(self):
        shards = partition_spec_to_shard_types(PartitionSpec(None, None))
        self.assertEqual(shards, {})

    def test_duplicate_axis_rejected(self):
        """Axis at two different dims raises SpmdTypeError."""
        with self.assertRaises(SpmdTypeError):
            partition_spec_to_shard_types(PartitionSpec(tp, tp))

    def test_round_trip_single_axis_per_dim(self):
        """PartitionSpec -> dict -> spec works when each dim has one axis."""
        spec = PartitionSpec(dp, ep)
        shards = partition_spec_to_shard_types(spec)
        rebuilt = shard_types_to_partition_spec(shards, 2)
        self.assertEqual(rebuilt, spec)


if __name__ == "__main__":
    unittest.main()
