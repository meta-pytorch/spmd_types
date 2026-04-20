# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for MeshAxis."""

from __future__ import annotations

import unittest

import torch.distributed as dist
from spmd_types._mesh_axis import (
    _register_name,
    _reset,
    flatten_axes,
    MeshAxis,
    set_printoptions,
)
from spmd_types._testing import fake_pg
from torch.distributed._mesh_layout import _MeshLayout


class TestMeshAxis(unittest.TestCase):
    """Unit tests for MeshAxis construction, equality, hashing, and display."""

    def setUp(self) -> None:
        _reset()

    def tearDown(self) -> None:
        _reset()

    def test_equality_same_layout(self) -> None:
        a = MeshAxis.of(4, 1)
        b = MeshAxis.of(4, 1)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_equality_coalesced(self) -> None:
        # (2,2):(2,1) coalesces to (4,):(1,) -- should be equal.
        a = MeshAxis(_MeshLayout((2, 2), (2, 1)))
        b = MeshAxis.of(4, 1)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_inequality(self) -> None:
        a = MeshAxis.of(4, 1)
        b = MeshAxis.of(4, 2)
        self.assertNotEqual(a, b)

    def test_not_equal_to_non_mesh_axis(self) -> None:
        a = MeshAxis.of(4, 1)
        self.assertNotEqual(a, "not a mesh axis")
        self.assertNotEqual(a, 42)

    def test_of_unit_stride(self) -> None:
        a = MeshAxis.of(4, 1)
        b = MeshAxis(_MeshLayout((4,), (1,)))
        self.assertEqual(a, b)

    def test_of_custom_stride(self) -> None:
        a = MeshAxis.of(4, 2)
        b = MeshAxis(_MeshLayout((4,), (2,)))
        self.assertEqual(a, b)

    def test_size(self) -> None:
        a = MeshAxis.of(4, 1)
        self.assertEqual(a.size(), 4)

        b = MeshAxis(_MeshLayout((2, 3), (3, 1)))
        self.assertEqual(b.size(), 6)

    def test_size_single_rank(self) -> None:
        a = MeshAxis.of(1, 1)
        self.assertEqual(a.size(), 1)

    def test_subgroup_equal(self) -> None:
        a = MeshAxis.of(4, 1)
        b = MeshAxis.of(4, 1)
        self.assertTrue(a <= b)
        self.assertTrue(a >= b)
        self.assertFalse(a < b)
        self.assertFalse(a > b)

    def test_subgroup_proper(self) -> None:
        # Ranks {0, 1} is a subgroup of {0, 1, 2, 3}.
        small = MeshAxis.of(2, 1)
        big = MeshAxis.of(4, 1)
        self.assertTrue(small <= big)
        self.assertTrue(small < big)
        self.assertFalse(big <= small)
        self.assertFalse(big < small)

    def test_subgroup_strided(self) -> None:
        # Ranks {0, 2} is a subgroup of {0, 1, 2, 3}.
        strided = MeshAxis.of(2, 2)
        full = MeshAxis.of(4, 1)
        self.assertTrue(strided <= full)
        self.assertTrue(strided < full)

    def test_subgroup_disjoint(self) -> None:
        # {0, 1} vs {0, 2}: neither is a subgroup of the other.
        a = MeshAxis.of(2, 1)
        b = MeshAxis.of(2, 2)
        self.assertFalse(a <= b)
        self.assertFalse(b <= a)

    def test_repr_no_names(self) -> None:
        # Without names, repr falls back to layout format in both modes.
        with set_printoptions(debug=False):
            self.assertEqual(repr(MeshAxis.of(4, 1)), "MeshAxis(4:1)")
            self.assertEqual(repr(MeshAxis.of(4, 2)), "MeshAxis(4:2)")
            self.assertEqual(
                repr(MeshAxis(_MeshLayout((2, 3), (3, 1)))),
                "MeshAxis(6:1)",  # coalesced
            )
            # Multi-dimensional layout that doesn't coalesce to 1D.
            self.assertEqual(
                repr(MeshAxis(_MeshLayout((3, 2), (4, 1)))),
                "MeshAxis((3, 2):(4, 1))",
            )

    def test_repr_trivial_unnamed(self) -> None:
        """Size-1 axis without names shows MeshAxis(trivial)."""
        trivial = MeshAxis.of(1, 1)
        with set_printoptions(debug=False):
            self.assertEqual(repr(trivial), "MeshAxis(trivial)")
        with set_printoptions(debug=True):
            self.assertEqual(repr(trivial), "MeshAxis(trivial){1:1}")

    def test_repr_trivial_named(self) -> None:
        """Size-1 axis with a name shows name(trivial)."""
        trivial = MeshAxis.of(1, 1)
        _register_name(trivial, "TP")
        with set_printoptions(debug=False):
            self.assertEqual(repr(trivial), "TP(trivial)")
        with set_printoptions(debug=True):
            self.assertEqual(repr(trivial), "TP(trivial){1:1}")

    def test_repr_with_names(self) -> None:
        from spmd_types._mesh_axis import _register_name

        a = MeshAxis.of(4, 1)
        _register_name(a, "TP")
        with set_printoptions(debug=False):
            self.assertEqual(repr(a), "TP")

    def test_repr_with_multiple_names(self) -> None:
        from spmd_types._mesh_axis import _register_name

        a = MeshAxis.of(4, 1)
        _register_name(a, "TP")
        _register_name(a, "CP")
        with set_printoptions(debug=False):
            # Only the first name (alphabetically).
            self.assertEqual(repr(a), "CP")

    def test_repr_prefers_non_default_pg(self) -> None:
        """When both 'default_pg' and another name exist, prefer the other."""
        a = MeshAxis.of(4, 1)
        _register_name(a, "default_pg")
        _register_name(a, "TP")
        with set_printoptions(debug=False):
            self.assertEqual(repr(a), "TP")

    def test_repr_default_pg_only(self) -> None:
        """When 'default_pg' is the only name, use it."""
        a = MeshAxis.of(4, 1)
        _register_name(a, "default_pg")
        with set_printoptions(debug=False):
            self.assertEqual(repr(a), "default_pg")

    def test_repr_debug_no_names(self) -> None:
        with set_printoptions(debug=True):
            self.assertEqual(repr(MeshAxis.of(4, 1)), "MeshAxis(4:1)")

    def test_repr_debug_with_names(self) -> None:
        from spmd_types._mesh_axis import _register_name

        a = MeshAxis.of(4, 1)
        _register_name(a, "TP")
        with set_printoptions(debug=True):
            self.assertEqual(repr(a), "TP{4:1}")

    def test_repr_debug_with_multiple_names(self) -> None:
        from spmd_types._mesh_axis import _register_name

        a = MeshAxis.of(4, 1)
        _register_name(a, "TP")
        _register_name(a, "SP")
        with set_printoptions(debug=True):
            # All names sorted, joined with '/'.
            self.assertEqual(repr(a), "SP/TP{4:1}")

    def test_set_printoptions_statement(self) -> None:
        """set_printoptions as a statement applies permanently."""
        from spmd_types._mesh_axis import _register_name

        a = MeshAxis.of(4, 1)
        _register_name(a, "TP")
        set_printoptions(debug=True)
        self.assertEqual(repr(a), "TP{4:1}")
        set_printoptions(debug=False)
        self.assertEqual(repr(a), "TP")

    def test_set_printoptions_context_restores(self) -> None:
        """set_printoptions as context manager restores on exit."""
        from spmd_types._mesh_axis import _register_name

        a = MeshAxis.of(4, 1)
        _register_name(a, "TP")
        set_printoptions(debug=False)
        with set_printoptions(debug=True):
            self.assertEqual(repr(a), "TP{4:1}")
        self.assertEqual(repr(a), "TP")

    def test_names_property(self) -> None:
        a = MeshAxis.of(4, 1)
        self.assertEqual(a.names, frozenset())

        from spmd_types._mesh_axis import _register_name

        _register_name(a, "DP")
        _register_name(a, "FSDP")
        self.assertEqual(a.names, frozenset({"DP", "FSDP"}))

    def test_dict_key(self) -> None:
        """MeshAxis can be used as dictionary keys."""
        a = MeshAxis.of(4, 1)
        b = MeshAxis.of(4, 1)
        d = {a: "hello"}
        self.assertEqual(d[b], "hello")

    def test_set_membership(self) -> None:
        a = MeshAxis.of(4, 1)
        b = MeshAxis.of(4, 1)
        s = {a}
        self.assertIn(b, s)

    def test_isorthogonal_basic(self) -> None:
        # Ranks {0,1,2,3} (stride 1) and {0,4} (stride 4) are orthogonal.
        a = MeshAxis.of(4, 1)
        b = MeshAxis.of(2, 4)
        self.assertTrue(a.isorthogonal(b))
        self.assertTrue(b.isorthogonal(a))

    def test_isorthogonal_same_axis(self) -> None:
        # An axis is NOT orthogonal with itself (ranks collide).
        a = MeshAxis.of(4, 1)
        self.assertFalse(a.isorthogonal(a))

    def test_isorthogonal_overlapping(self) -> None:
        # Two axes whose combined layout has rank collisions.
        a = MeshAxis.of(4, 1)  # ranks {0,1,2,3}
        b = MeshAxis.of(2, 1)  # ranks {0,1}
        self.assertFalse(a.isorthogonal(b))

    def test_isorthogonal_single_rank(self) -> None:
        # A single-rank axis is orthogonal with anything.
        one = MeshAxis.of(1, 1)
        big = MeshAxis.of(8, 1)
        self.assertTrue(one.isorthogonal(big))
        self.assertTrue(big.isorthogonal(one))

    def test_isorthogonal_variadic(self) -> None:
        # Three-way orthogonality: dp=2 stride 8, cp=2 stride 4, tp=4 stride 1.
        tp = MeshAxis.of(4, 1)
        cp = MeshAxis.of(2, 4)
        dp = MeshAxis.of(2, 8)
        self.assertTrue(tp.isorthogonal(cp, dp))
        self.assertTrue(dp.isorthogonal(cp, tp))


class TestMeshAxisFromProcessGroup(unittest.TestCase):
    """Tests for MeshAxis.of(ProcessGroup) using a FakeStore backend."""

    def setUp(self) -> None:
        _reset()

    def tearDown(self) -> None:
        _reset()

    def test_from_default_group(self) -> None:
        with fake_pg(world_size=8) as pg:
            axis = MeshAxis.of(pg)
            self.assertEqual(axis.size(), 8)

    def test_from_subgroup(self) -> None:
        with fake_pg(world_size=8):
            sub = dist.new_group(ranks=[0, 1, 2, 3])
            axis = MeshAxis.of(sub)
            self.assertEqual(axis.size(), 4)

    def test_caching(self) -> None:
        with fake_pg(world_size=4) as pg:
            a = MeshAxis.of(pg)
            b = MeshAxis.of(pg)
            self.assertEqual(a, b)

    def test_group_desc_name_registration(self) -> None:
        with fake_pg(world_size=8):
            sub = dist.new_group(ranks=[0, 1, 2, 3], group_desc="TP")
            axis = MeshAxis.of(sub)
            self.assertEqual(axis.names, frozenset({"TP"}))
            with set_printoptions(debug=False):
                self.assertEqual(repr(axis), "TP")

    def test_two_pgs_same_ranks_same_axis(self) -> None:
        """Two PGs with the same ranks produce equal MeshAxis objects."""
        with fake_pg(world_size=8):
            pg1 = dist.new_group(ranks=[0, 1, 2, 3])
            pg2 = dist.new_group(ranks=[0, 1, 2, 3])
            a1 = MeshAxis.of(pg1)
            a2 = MeshAxis.of(pg2)
            self.assertEqual(a1, a2)

    def test_names_accumulate_across_pgs(self) -> None:
        """Names from multiple PGs with same ranks accumulate."""
        with fake_pg(world_size=8):
            pg1 = dist.new_group(ranks=[0, 1, 2, 3], group_desc="TP")
            pg2 = dist.new_group(ranks=[0, 1, 2, 3], group_desc="CP")
            MeshAxis.of(pg1)
            MeshAxis.of(pg2)
            # Both names should be registered for the same axis.
            axis = MeshAxis.of(pg1)
            self.assertEqual(axis.names, frozenset({"TP", "CP"}))


class TestMeshAxisDeviceMesh(unittest.TestCase):
    """Tests for MeshAxis with DeviceMesh-created ProcessGroups."""

    def setUp(self) -> None:
        _reset()

    def tearDown(self) -> None:
        _reset()

    def test_dp_cp_tp_mesh(self) -> None:
        """Basic 3D mesh: dp=2, cp=2, tp=4 on 16 ranks."""
        with fake_pg(world_size=16):
            from torch.distributed.device_mesh import init_device_mesh

            mesh = init_device_mesh("cpu", (2, 2, 4), mesh_dim_names=("dp", "cp", "tp"))
            dp = MeshAxis.of(mesh.get_group("dp"))
            cp = MeshAxis.of(mesh.get_group("cp"))
            tp = MeshAxis.of(mesh.get_group("tp"))

            self.assertEqual(dp.size(), 2)
            self.assertEqual(cp.size(), 2)
            self.assertEqual(tp.size(), 4)

            # Names come from DeviceMesh's group_desc ("mesh_<name>").
            self.assertEqual(dp.names, frozenset({"mesh_dp"}))
            self.assertEqual(cp.names, frozenset({"mesh_cp"}))
            self.assertEqual(tp.names, frozenset({"mesh_tp"}))

            # All three axes are distinct.
            self.assertNotEqual(dp, cp)
            self.assertNotEqual(dp, tp)
            self.assertNotEqual(cp, tp)

            # Pairwise orthogonality.
            self.assertTrue(tp.isorthogonal(dp))
            self.assertTrue(tp.isorthogonal(cp))
            self.assertTrue(dp.isorthogonal(cp))

            # Variadic: all three at once.
            self.assertTrue(tp.isorthogonal(dp, cp))

            # Default repr: just the name.
            with set_printoptions(debug=False):
                self.assertEqual(repr(tp), "mesh_tp")
                self.assertEqual(repr(cp), "mesh_cp")
                self.assertEqual(repr(dp), "mesh_dp")

            # Debug repr: name{layout}.
            with set_printoptions(debug=True):
                self.assertEqual(repr(tp), "mesh_tp{4:1}")
                self.assertEqual(repr(cp), "mesh_cp{2:4}")
                self.assertEqual(repr(dp), "mesh_dp{2:8}")

    def test_dp_cp_flatten(self) -> None:
        """Flattening dp and cp into a single axis."""
        with fake_pg(world_size=16):
            from torch.distributed.device_mesh import init_device_mesh

            mesh = init_device_mesh("cpu", (2, 2, 4), mesh_dim_names=("dp", "cp", "tp"))
            dp = MeshAxis.of(mesh.get_group("dp"))
            cp = MeshAxis.of(mesh.get_group("cp"))
            tp = MeshAxis.of(mesh.get_group("tp"))

            # Flatten dp x cp into a single axis.
            dp_cp_mesh = mesh["dp", "cp"]._flatten("dp_cp")
            dp_cp = MeshAxis.of(dp_cp_mesh.get_group("dp_cp"))

            self.assertEqual(dp_cp.size(), 4)
            with set_printoptions(debug=False):
                self.assertEqual(repr(dp_cp), "mesh_dp_cp")

            # dp and cp are both subgroups of dp_cp.
            self.assertTrue(dp <= dp_cp)
            self.assertTrue(dp < dp_cp)
            self.assertTrue(cp <= dp_cp)
            self.assertTrue(cp < dp_cp)

            # tp is NOT a subgroup of dp_cp (disjoint rank sets).
            self.assertFalse(tp <= dp_cp)

    def test_ep_pattern(self) -> None:
        """Expert parallelism pattern: dp=2, ep=4, tp=2 on 16 ranks.

        In torchtitan-style EP, expert parameters are sharded across the ep
        axis.  The dp_ep flattened axis is used for data-parallel collectives
        that need to span both dp and ep (e.g., gradient all-reduce for
        non-expert parameters when dp and ep are interleaved).
        """
        with fake_pg(world_size=16):
            from torch.distributed.device_mesh import init_device_mesh

            mesh = init_device_mesh("cpu", (2, 4, 2), mesh_dim_names=("dp", "ep", "tp"))
            dp = MeshAxis.of(mesh.get_group("dp"))
            ep = MeshAxis.of(mesh.get_group("ep"))
            tp = MeshAxis.of(mesh.get_group("tp"))

            self.assertEqual(dp.size(), 2)
            self.assertEqual(ep.size(), 4)
            self.assertEqual(tp.size(), 2)

            with set_printoptions(debug=False):
                self.assertEqual(repr(tp), "mesh_tp")
                self.assertEqual(repr(ep), "mesh_ep")
                self.assertEqual(repr(dp), "mesh_dp")

            # Flatten dp x ep for non-expert gradient all-reduce.
            dp_ep_mesh = mesh["dp", "ep"]._flatten("dp_ep")
            dp_ep = MeshAxis.of(dp_ep_mesh.get_group("dp_ep"))

            self.assertEqual(dp_ep.size(), 8)
            with set_printoptions(debug=False):
                self.assertEqual(repr(dp_ep), "mesh_dp_ep")

            # dp and ep are subgroups of dp_ep.
            self.assertTrue(dp <= dp_ep)
            self.assertTrue(ep <= dp_ep)

            # tp is NOT a subgroup of dp_ep.
            self.assertFalse(tp <= dp_ep)

            # dp_ep is NOT a subgroup of dp or ep individually.
            self.assertFalse(dp_ep <= dp)
            self.assertFalse(dp_ep <= ep)


class TestNameDisambiguation(unittest.TestCase):
    """Tests for disambiguation when two different MeshAxis share a name."""

    def setUp(self) -> None:
        _reset()

    def tearDown(self) -> None:
        _reset()

    def test_same_name_different_layout_disambiguates(self) -> None:
        """Two axes with the same name but different layouts get suffixed."""
        a = MeshAxis.of(4, 1)
        b = MeshAxis.of(2, 1)
        _register_name(a, "mesh_ddp")
        _register_name(b, "mesh_ddp")  # collision
        with set_printoptions(debug=False):
            self.assertEqual(repr(a), "mesh_ddp[4:1]")
            self.assertEqual(repr(b), "mesh_ddp[2:1]")

    def test_same_name_same_layout_no_disambiguation(self) -> None:
        """Two axes with the same name AND layout are the same axis -- no suffix."""
        a = MeshAxis.of(4, 1)
        _register_name(a, "mesh_ddp")
        _register_name(a, "mesh_ddp")  # same axis, no collision
        with set_printoptions(debug=False):
            self.assertEqual(repr(a), "mesh_ddp")

    def test_disambiguation_with_process_groups(self) -> None:
        """Two PGs with same group_desc but different ranks get disambiguated."""
        with fake_pg(world_size=8):
            pg1 = dist.new_group(ranks=[0, 1, 2, 3], group_desc="mesh_ddp")
            pg2 = dist.new_group(ranks=[0, 1], group_desc="mesh_ddp")
            a1 = MeshAxis.of(pg1)
            a2 = MeshAxis.of(pg2)
            self.assertNotEqual(a1, a2)
            with set_printoptions(debug=False):
                self.assertIn("[", repr(a1))
                self.assertIn("[", repr(a2))
                self.assertNotEqual(repr(a1), repr(a2))

    def test_three_way_collision(self) -> None:
        """Three axes with the same name all get disambiguated."""
        a = MeshAxis.of(4, 1)
        b = MeshAxis.of(2, 1)
        c = MeshAxis.of(8, 1)
        _register_name(a, "mesh_ddp")
        _register_name(b, "mesh_ddp")  # first collision
        _register_name(c, "mesh_ddp")  # second collision (base already marked)
        with set_printoptions(debug=False):
            self.assertEqual(repr(a), "mesh_ddp[4:1]")
            self.assertEqual(repr(b), "mesh_ddp[2:1]")
            self.assertEqual(repr(c), "mesh_ddp[8:1]")

    def test_disambiguation_preserves_other_names(self) -> None:
        """Non-colliding names are preserved when a collision is resolved."""
        a = MeshAxis.of(4, 1)
        b = MeshAxis.of(2, 1)
        _register_name(a, "TP")
        _register_name(a, "mesh_ddp")
        _register_name(b, "mesh_ddp")  # collision on "mesh_ddp" only
        with set_printoptions(debug=False):
            # a has "TP" and "mesh_ddp[4:1]"; best_name picks "TP" (non-default)
            self.assertEqual(repr(a), "TP")
            self.assertEqual(repr(b), "mesh_ddp[2:1]")


class TestInferredFlattenedRepr(unittest.TestCase):
    """Tests for unnamed axes that infer names from named sub-axes."""

    def setUp(self) -> None:
        _reset()

    def tearDown(self) -> None:
        _reset()

    def test_unnamed_flattened_axis_shows_components(self) -> None:
        """An unnamed axis that equals flatten_axes of named axes shows {dp,cp}."""
        dp = MeshAxis.of(2, 8)
        cp = MeshAxis.of(2, 4)
        _register_name(dp, "dp")
        _register_name(cp, "cp")

        dp_cp = flatten_axes((dp, cp))
        with set_printoptions(debug=False):
            self.assertEqual(repr(dp_cp), "{dp,cp}")

    def test_unnamed_flattened_axis_debug_mode(self) -> None:
        """In debug mode, inferred name includes layout."""
        dp = MeshAxis.of(2, 8)
        cp = MeshAxis.of(2, 4)
        _register_name(dp, "dp")
        _register_name(cp, "cp")

        dp_cp = flatten_axes((dp, cp))
        with set_printoptions(debug=True):
            self.assertEqual(repr(dp_cp), "{dp,cp}{4:4}")

    def test_named_axis_prefers_own_name(self) -> None:
        """If an axis has its own name, it uses that even if decomposable."""
        dp = MeshAxis.of(2, 8)
        cp = MeshAxis.of(2, 4)
        _register_name(dp, "dp")
        _register_name(cp, "cp")

        dp_cp = flatten_axes((dp, cp))
        _register_name(dp_cp, "dp_cp")
        with set_printoptions(debug=False):
            self.assertEqual(repr(dp_cp), "dp_cp")

    def test_no_decomposition_falls_back(self) -> None:
        """An unnamed axis with no matching sub-axes falls back to MeshAxis(...)."""
        x = MeshAxis.of(4, 1)
        with set_printoptions(debug=False):
            self.assertEqual(repr(x), "MeshAxis(4:1)")

    def test_three_way_flattened(self) -> None:
        """Three named sub-axes compose into one inferred name."""
        dp = MeshAxis.of(2, 8)
        cp = MeshAxis.of(2, 4)
        tp = MeshAxis.of(4, 1)
        _register_name(dp, "dp")
        _register_name(cp, "cp")
        _register_name(tp, "tp")

        full = flatten_axes((dp, cp, tp))
        with set_printoptions(debug=False):
            self.assertEqual(repr(full), "{dp,cp,tp}")

    def test_ordering_is_outermost_first(self) -> None:
        """Sub-axes are ordered by stride descending (outermost first)."""
        tp = MeshAxis.of(4, 1)
        dp = MeshAxis.of(2, 4)
        _register_name(tp, "tp")
        _register_name(dp, "dp")

        flat = flatten_axes((tp, dp))
        with set_printoptions(debug=False):
            self.assertEqual(repr(flat), "{dp,tp}")


if __name__ == "__main__":
    unittest.main()
