# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import pytest
import torch
from spmd_types import (
    assert_type,
    MeshAxis,
    no_typecheck,
    R,
    reinterpret_mesh,
    set_current_mesh,
    V,
)
from spmd_types._checker import typecheck
from spmd_types._mesh_region import _remap_partition_spec
from spmd_types.runtime import get_local_type, get_partition_spec
from spmd_types.types import PartitionSpec, SpmdTypeError


@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("local", [False, True])
def test_cross_mesh_shard_decay(local, implicit):
    outer = MeshAxis.of(2, 2)
    inner = MeshAxis.of(2, 1)
    merged = MeshAxis.of(4, 1)
    shared = MeshAxis.of(2, 4)
    with typecheck(local=local, strict_mode="strict"):
        x = torch.zeros(4, 4, 4)
        assert_type(
            x,
            {outer: V, inner: V, shared: V},
            PartitionSpec(outer, inner, shared),
        )
        with set_current_mesh(frozenset({merged, shared})):

            def run():
                return (
                    x + x
                    if implicit
                    else reinterpret_mesh(x, frozenset({merged, shared}))
                )

            if not local:
                with pytest.raises(SpmdTypeError, match="Cannot remap PartitionSpec"):
                    run()
                return
            y = run()
            assert get_local_type(y) == {merged: V, shared: V}
            assert get_partition_spec(y) == (
                None if implicit else PartitionSpec(None, None, shared)
            )
            assert get_partition_spec(x) == PartitionSpec(outer, inner, shared)


@pytest.mark.parametrize("reverse", [False, True])
def test_decay_preserves_only_outer_prefix(reverse):
    a, b, c, k = (
        MeshAxis.of(2, 2),
        MeshAxis.of(2, 1),
        MeshAxis.of(4, 1),
        MeshAxis.of(2, 4),
    )
    pairs = [([a, b], [c]), ([k], [k])]
    spec = PartitionSpec((a, k) if reverse else (k, a))
    expected = PartitionSpec(None if reverse else k)
    with typecheck(local=True):
        assert _remap_partition_spec(spec, pairs) == expected
    with typecheck(local=False), pytest.raises(SpmdTypeError):
        _remap_partition_spec(spec, pairs)


@pytest.mark.parametrize("layout", ["inner", "reversed", "exact"])
def test_partial_and_reordered_shards(layout):
    a, b, c = MeshAxis.of(2, 2), MeshAxis.of(2, 1), MeshAxis.of(4, 1)
    entry = {"inner": b, "reversed": (b, a), "exact": (a, b)}[layout]
    with typecheck(local=True):
        result = _remap_partition_spec(PartitionSpec(entry), [([a, b], [c])])
    assert result == PartitionSpec(c if layout == "exact" else None)


def test_mixed_policy_decay():
    a, b, c, k = (
        MeshAxis.of(2, 4),
        MeshAxis.of(2, 2),
        MeshAxis.of(4, 2),
        MeshAxis.of(2, 1),
    )
    with typecheck(local=False, strict_mode="strict"):
        x = torch.zeros(4, 4, 4)
        assert_type(x, {a: V, b: V, k: V}, PartitionSpec(a, b, k))
        with set_current_mesh({"outer": c, "inner": k}, local_axes=("outer",)):
            y = reinterpret_mesh(x, frozenset({c, k}), inplace=True)
            assert get_partition_spec(y) == PartitionSpec(None, None, k)
            assert get_partition_spec(y + y) == PartitionSpec(None, None, k)


@pytest.mark.parametrize("local", [None, True, False])
def test_explicit_destination_outside_current_mesh(local):
    a, b, c = MeshAxis.of(2, 2), MeshAxis.of(2, 1), MeshAxis.of(4, 1)
    with nullcontext() if local is None else typecheck(local=local):
        x = torch.zeros(4, 4)
        assert_type(x, {a: V, b: V}, PartitionSpec(a, b))
        with set_current_mesh({"a": a, "b": b}):
            if local is False:
                with pytest.raises(SpmdTypeError, match="Cannot remap PartitionSpec"):
                    reinterpret_mesh(x, {c: V}, inplace=True)
            else:
                y = reinterpret_mesh(x, {c: V}, inplace=True)
                assert y is x
                assert get_local_type(y) == {c: V}
                assert get_partition_spec(y) == PartitionSpec(None, None)


def test_mixed_policy_rejects_undefined_destination():
    a, b, c = MeshAxis.of(2, 2), MeshAxis.of(2, 1), MeshAxis.of(4, 1)
    with typecheck(local=False):
        x = torch.zeros(4, 4)
        assert_type(x, {a: V, b: V}, PartitionSpec(a, b))
        with set_current_mesh({"a": a, "b": b}, local_axes=("a",)):
            with pytest.raises(SpmdTypeError, match="policy is undefined"):
                reinterpret_mesh(x, {c: V}, inplace=True)


def test_no_checker_still_rejects_incompatible_local_types():
    a, b, c = MeshAxis.of(2, 2), MeshAxis.of(2, 1), MeshAxis.of(4, 1)
    x = torch.zeros(4)
    assert_type(x, {a: V, b: R}, PartitionSpec(a))
    with pytest.raises(SpmdTypeError, match="has type V while .* has type R"):
        reinterpret_mesh(x, {c: V})


@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_exact_remap_needs_no_destination_policy(paused, inplace):
    a, b, c = MeshAxis.of(2, 2), MeshAxis.of(2, 1), MeshAxis.of(4, 1)
    with typecheck(local=False):
        x = torch.zeros(4, 4)
        assert_type(x, {a: V, b: V}, PartitionSpec((a, b), None))
        with set_current_mesh({"a": a, "b": b}, local_axes=("a",)):
            with no_typecheck() if paused else nullcontext():
                y = reinterpret_mesh(x, {c: V}, inplace=inplace)
                assert (y is x) == inplace
                assert get_partition_spec(y) == PartitionSpec(c, None)


def test_decay_does_not_query_preserved_axis():
    a, b, c, k = (
        MeshAxis.of(2, 2),
        MeshAxis.of(2, 1),
        MeshAxis.of(4, 1),
        MeshAxis.of(2, 4),
    )

    with typecheck(local=False), set_current_mesh({"c": c}, local_axes=("c",)):
        assert _remap_partition_spec(
            PartitionSpec((k, a)), [([k], [k]), ([a, b], [c])]
        ) == PartitionSpec(k)


def test_paused_checker_preserves_explicit_global_contract():
    a, b, c = MeshAxis.of(2, 2), MeshAxis.of(2, 1), MeshAxis.of(4, 1)
    with typecheck(local=False), no_typecheck():
        x = torch.zeros(4, 4)
        assert_type(x, {a: V, b: V}, PartitionSpec(a, b))
        with pytest.raises(SpmdTypeError, match="Cannot remap PartitionSpec"):
            reinterpret_mesh(x, {c: V}, inplace=True)
