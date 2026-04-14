"""Tests for mesh axis helpers (flatten_axes)."""

from __future__ import annotations

import unittest

from spmd_types._mesh_axis import _reset, flatten_axes, MeshAxis


class TestFlattenAxes(unittest.TestCase):
    def setUp(self) -> None:
        _reset()

    def tearDown(self) -> None:
        _reset()

    def test_flatten_single_axis_is_identity(self) -> None:
        dp_cp = MeshAxis.of(4, 4)
        self.assertEqual(flatten_axes((dp_cp,)), dp_cp)

    def test_flatten_dp_cp(self) -> None:
        dp = MeshAxis.of(2, 8)
        cp = MeshAxis.of(2, 4)
        dp_cp = MeshAxis.of(4, 4)
        self.assertEqual(flatten_axes((dp, cp)), dp_cp)

    def test_flatten_full_region(self) -> None:
        dp = MeshAxis.of(2, 8)
        cp = MeshAxis.of(2, 4)
        tp = MeshAxis.of(4, 1)
        full = MeshAxis.of(16, 1)
        self.assertEqual(flatten_axes((dp, cp, tp)), full)

    def test_flatten_requires_orthogonal_axes(self) -> None:
        dp = MeshAxis.of(2, 8)
        dp_cp = MeshAxis.of(4, 4)
        with self.assertRaises(ValueError):
            flatten_axes((dp, dp_cp))


if __name__ == "__main__":
    unittest.main()
