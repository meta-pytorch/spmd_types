# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerically check an ``spmd_typecheck`` hook against its kernel.

A hook is trusted: the checker believes whatever types it derives.  A local
SPMD reading of a hook is often right while its global reading is wrong, for
example ``rules.einsum("...d->...d", x)`` on a kernel that mixes elements
along ``d``: the output is Varying either way, but the claim that a shard of
the output is the output of a shard is false.  ``rulecheck`` tests exactly
that claim, the single-device-then-partition equivalence that global SPMD
types promise:

1. global inputs are sharded, replicated, or split into random partial
   contributions per rank according to the placements under test;
2. the kernel runs once per rank coordinate and the hook derives the output
   types.  By default this is on plain per-rank tensors with no distributed
   setup at all; with ``local_tensor_mode=True`` (required when an argument
   names a mesh axis as the kernel's process group) the ranks are simulated
   in-process under ``LocalTensorMode`` with a fake process group, where raw
   ``torch.distributed`` collectives and functional collectives both execute;
3. the per-rank outputs are reassembled *according to the derived types*:
   concatenated along ``S(i)`` dims in mesh order, summed over ``P`` axes,
   checked equal across ``R``/``I`` ranks;
4. the result is compared with the kernel itself evaluated on the global
   inputs, with every group standing for a world of size one.

The recommended call passes no placements and lets ``rulecheck`` enumerate::

    report = rulecheck(LinearAllReduce, (x, weight, bias, "tp"), local_tensor_mode=True)

``report.checked`` lists the placements that ran and matched; ``report.rejected``
lists the ones the hook refused, with the reason.  Explicit ``placements=`` (with
``mesh=``) check one configuration, for anything enumeration cannot construct::

    rulecheck(
        LinearAllReduce,
        args=(x, weight, bias, "tp"),             # global tensors; "tp" names the group arg
        placements={"x": {"dp": S(0), "tp": S(1)}, "weight": {"dp": R, "tp": S(1)},
                    "bias": {"dp": R, "tp": I}},
        mesh={"dp": 2, "tp": 2},
        local_tensor_mode=True,
    )

``local_tensor_mode`` sets up its own fake distributed environment and must be
called with no process group initialized; the plain backend has no such
requirement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, NamedTuple
from unittest import mock

import torch
import torch.distributed as dist
from spmd_types._checker import typecheck
from spmd_types._mesh import set_current_mesh
from spmd_types._mesh_axis import _reset, MeshAxis
from spmd_types._state import current_mesh_all_names
from spmd_types._type_attr import get_local_type
from spmd_types.rules import _iter_tensors, bind_forward_args, hook_kwargs, hook_shape
from spmd_types.runtime import assert_type, get_partition_spec
from spmd_types.types import (
    I,
    normalize_partition_spec,
    P,
    PartitionSpec,
    PerMeshAxisSpmdType,
    R,
    S,
    Shard,
    SpmdType,
    SpmdTypeError,
    V,
)
from torch.distributed._local_tensor import LocalIntNode, LocalTensor, LocalTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore


class RuleCheckError(AssertionError):
    """The hook's derived types do not describe what the kernel computed."""


type RuleCheckPlacement = dict[str, dict[str, PerMeshAxisSpmdType]]


class RuleCheckRejection(NamedTuple):
    """An enumerated placement refused by the hook, and its reason."""

    placement: RuleCheckPlacement
    reason: str


# =============================================================================
# Mesh layout shared by both backends
# =============================================================================


class _Mesh:
    """Backend-independent description of the mesh simulated by rulecheck.

    The insertion order of the supplied axis names defines a row-major rank
    layout, matching ``init_device_mesh``.  The description is available
    before any process group exists and supports coordinates for every
    simulated rank, which the plain-tensor backend and placement construction
    both need.

    ``axes()`` realizes the same topology as process-group-free ``MeshAxis``
    objects for the plain backend.  The LocalTensor backend instead uses the
    stored names and sizes to construct a real ``DeviceMesh`` after setting up
    its fake process group.
    """

    def __init__(self, mesh: Mapping[str, int]) -> None:
        self.names = list(mesh)
        self.sizes = {n: int(mesh[n]) for n in self.names}
        self.world = 1
        for n in self.names:
            self.world *= self.sizes[n]

    def size(self, axis: str) -> int:
        return self.sizes[axis]

    def coords(self, rank: int) -> dict[str, int]:
        out: dict[str, int] = {}
        for name in reversed(self.names):
            out[name] = rank % self.sizes[name]
            rank //= self.sizes[name]
        return out

    def axes(self) -> dict[str, MeshAxis]:
        """Process-group-free ``MeshAxis`` objects for the plain backend."""
        result: dict[str, MeshAxis] = {}
        stride = 1
        for name in reversed(self.names):
            result[name] = MeshAxis.of(self.sizes[name], stride)
            stride *= self.sizes[name]
        return {name: result[name] for name in self.names}


# =============================================================================
# Placements
# =============================================================================


def _to_spmd_type(
    placement: SpmdType | Mapping[str, PerMeshAxisSpmdType], ndim: int
) -> tuple[dict[str, PerMeshAxisSpmdType], list[tuple[str, ...]]]:
    """Normalize the two public placement forms used by rulecheck.

    A mapping can encode sharding directly with ``S(i)`` values, whereas an
    ``SpmdType`` stores local ``V`` types separately from a ``PartitionSpec``.
    Convert either form into the representation needed to construct rank-local
    inputs:

    * a local type for each named mesh axis, with every ``S(i)`` lowered to
      ``V``; and
    * for each of the tensor's ``ndim`` dimensions, the ordered mesh-axis names
      that shard it.

    Negative shard dimensions are resolved against ``ndim``.  Mesh membership,
    complete axis coverage, and shape divisibility are validated later by
    ``_Distributor``, which has the mesh and concrete tensor shape.
    """
    if isinstance(placement, SpmdType):
        local = dict(placement.local_type)
        spec = placement.partition_spec
    else:
        local = dict(placement)
        spec = None
    dims: list[tuple[str, ...]] = [() for _ in range(ndim)]
    for axis, typ in list(local.items()):
        if isinstance(typ, Shard):
            d = typ.dim + ndim if typ.dim < 0 else typ.dim
            dims[d] = dims[d] + (axis,)
            local[axis] = V
    if spec is not None:
        for d, entry in enumerate(spec):
            if entry is None:
                continue
            axes = entry if isinstance(entry, tuple) else (entry,)
            dims[d] = dims[d] + tuple(axes)
            for a in axes:
                local[a] = V
    return local, dims


def _seeded_noise(like: torch.Tensor, *key: object) -> torch.Tensor:
    """Normal noise shaped like ``like``, determined by ``key`` and nothing else."""
    if not like.is_floating_point():
        raise RuleCheckError(
            "rulecheck: cannot split an integer tensor into Partial contributions"
        )
    seed = int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], "little")
    try:
        from torch.func import _random as stateless  # stateless PRNG, newer PyTorch

        prng_key, normal = stateless.key, stateless.normal
        return normal(prng_key(seed, device=like.device), *like.shape, dtype=like.dtype)
    except (ImportError, AttributeError, TypeError):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(like.shape, generator=g, dtype=like.dtype).to(like.device)


def _spec_from_dims(dims: list[tuple[str, ...]]) -> PartitionSpec | None:
    if not any(dims):
        return None
    return PartitionSpec(
        *[None if not d else (d[0] if len(d) == 1 else d) for d in dims]
    )


class _Distributor:
    """Materialize one global input tensor under a proposed placement.

    Construction normalizes the placement into local per-axis types and
    ordered sharding axes per tensor dimension.  It then validates that the
    placement covers exactly the mesh under test and that every sharded tensor
    dimension is divisible by the product of the axes sharding it.  The
    corresponding PartitionSpec is retained for annotating local values.

    ``piece(rank)`` produces that rank's physical input.  It repeatedly chunks
    sharded dimensions in outer-to-inner mesh-axis order.  On Partial axes it
    instead constructs deterministic, nondegenerate rank-local contributions
    whose sum is the sliced global value; the tensor name and other mesh
    coordinates keep independent partial groups distinct.  The returned value
    is cloned so separate simulated ranks never alias storage.

    ``annotate`` attaches the normalized local types and PartitionSpec to a
    piece before the typechecked kernel runs.
    """

    def __init__(
        self,
        name: str,
        value: torch.Tensor,
        placement: SpmdType | Mapping[str, PerMeshAxisSpmdType],
        mesh: _Mesh,
    ) -> None:
        self.name, self.value, self.mesh = name, value, mesh
        local, dims = _to_spmd_type(placement, value.ndim)
        for axis in local:
            if axis not in mesh.names:
                raise RuleCheckError(
                    f"rulecheck: {name!r} names unknown mesh axis {axis!r}"
                )
        for axis in mesh.names:
            if axis not in local:
                raise RuleCheckError(
                    f"rulecheck: placement for {name!r} must cover every mesh axis; "
                    f"missing {axis!r}"
                )
        for d, axes in enumerate(dims):
            total = 1
            for a in axes:
                total *= mesh.size(a)
            if axes and value.shape[d] % total:
                raise RuleCheckError(
                    f"rulecheck: {name!r} dim {d} of size {value.shape[d]} is not "
                    f"divisible by the mesh axes sharding it {tuple(axes)} (total size "
                    f"{total})"
                )
        self.dims = dims
        self.local = {a: local[a] for a in mesh.names}
        self.spec = _spec_from_dims(dims)
        self.partial_axes = [a for a in mesh.names if local[a] is P]

    def piece(self, rank: int) -> torch.Tensor:
        coords = self.mesh.coords(rank)
        t = self.value
        for d, axes in enumerate(self.dims):
            for a in axes:  # outermost first
                t = t.chunk(self.mesh.size(a), dim=d)[coords[a]]
        # Partial: split the sliced value into contributions that sum to it.
        for a in self.partial_axes:
            n = self.mesh.size(a)
            others = tuple((k, v) for k, v in coords.items() if k != a)
            noises = [_seeded_noise(t, self.name, a, others, c) for c in range(n - 1)]
            if coords[a] < n - 1:
                t = noises[coords[a]]
            else:
                t = t - sum(noises) if noises else t
        return t.clone()

    def annotate(self, t: torch.Tensor) -> torch.Tensor:
        assert_type(t, dict(self.local), self.spec)
        return t


# =============================================================================
# Reassembly according to the derived types
# =============================================================================


def _axis_names() -> dict[MeshAxis, str]:
    # all_names (not current_mesh_names) so a size-1 mesh axis, which the type
    # may still name, is present in the map rather than raising KeyError below.
    names = current_mesh_all_names() or {}
    return {axis: name for name, axis in names.items()}


def _axis_name(axis: MeshAxis, by_name: Mapping[MeshAxis, str], label: str) -> str:
    try:
        return by_name[axis]
    except KeyError as e:
        raise RuleCheckError(
            f"rulecheck: {label} names mesh axis {axis!r}, which is not part of "
            "the mesh under test"
        ) from e


def _reassemble(  # noqa: C901
    label: str,
    typed: Any,
    by_rank: dict[int, torch.Tensor],
    mesh: _Mesh,
) -> torch.Tensor:
    """Combine per-rank outputs into the global value the derived type implies.

    ``typed`` carries the derived annotations (the LocalTensor output, or any
    rank's output in the plain backend); ``by_rank`` holds the per-rank values.
    Partial axes are summed and sharded axes concatenated.  Replicate and
    Invariant axes must agree exactly across ranks; numerical tolerances apply
    only later, when the reassembled value is compared with the reference.
    """
    by_name = _axis_names()
    local_type = {
        _axis_name(a, by_name, label): t for a, t in get_local_type(typed).items()
    }
    spec = get_partition_spec(typed)
    names = mesh.names
    for axis in names:
        if axis not in local_type:
            raise RuleCheckError(
                f"rulecheck: {label} has no type on mesh axis {axis!r}; the hook "
                f"must type every axis"
            )
    shard_dim: dict[str, int] = {}
    nesting: dict[str, int] = {}  # axis -> position within its dim's tuple
    if spec is not None:
        for d, entry in enumerate(normalize_partition_spec(spec)):
            if entry is None:
                continue
            axes = entry if isinstance(entry, tuple) else (entry,)
            for pos, a in enumerate(axes):
                name = _axis_name(a, by_name, label)
                shard_dim[name] = d
                nesting[name] = pos
    for axis, typ in local_type.items():
        if typ is V and axis not in shard_dim:
            raise RuleCheckError(
                f"rulecheck: {label} is Varying on {axis!r} with no shard dim; a "
                f"global claim needs S(i) (or P) so the ranks can be recombined"
            )

    # Start from per-rank tensors keyed by full coordinates; fold one axis at a
    # time.  Inner axes of a nested sharding fold before outer ones.
    table: dict[tuple[int, ...], torch.Tensor] = {}
    for rank, t in by_rank.items():
        c = mesh.coords(rank)
        table[tuple(c[a] for a in names)] = t
    order = sorted(names, key=lambda a: -nesting.get(a, 0))
    remaining = list(names)
    for axis in order:
        i = remaining.index(axis)
        n = mesh.size(axis)
        folded: dict[tuple[int, ...], torch.Tensor] = {}
        groups: dict[tuple[int, ...], list[torch.Tensor]] = {}
        for key, t in table.items():
            rest = key[:i] + key[i + 1 :]
            groups.setdefault(rest, [None] * n)[key[i]] = t  # type: ignore[list-item]
        typ = local_type[axis]
        for rest, pieces in groups.items():
            if typ is P:
                folded[rest] = sum(pieces[1:], pieces[0])
            elif typ is V:
                folded[rest] = torch.cat(pieces, dim=shard_dim[axis])
            else:  # R / I: all ranks along the axis must agree
                for j, piece in enumerate(pieces[1:], 1):
                    if piece.shape != pieces[0].shape:
                        raise RuleCheckError(
                            f"rulecheck: {label} is typed {typ!r} on {axis!r}, but "
                            f"ranks at coordinate 0 and {j} along {axis!r} hold "
                            f"values of different shapes {tuple(pieces[0].shape)} "
                            f"and {tuple(piece.shape)}. The kernel left this axis "
                            f"varying or partial."
                        )
                    if not torch.equal(piece, pieces[0]):
                        raise RuleCheckError(
                            f"rulecheck: {label} is typed {typ!r} on {axis!r}, but "
                            f"ranks at coordinate 0 and {j} along {axis!r} hold "
                            f"different values (max diff "
                            f"{(piece - pieces[0]).abs().max().item():.3g}). The "
                            f"kernel left this axis varying or partial."
                        )
                folded[rest] = pieces[0]
        table = folded
        remaining.pop(i)
    assert list(table) == [()]
    return table[()]


# =============================================================================
# rulecheck
# =============================================================================


@dataclass
class RuleCheckReport:
    """What an enumerating ``rulecheck`` exercised.

    ``checked`` lists placements (by argument name) that ran and matched the
    reference; ``rejected`` lists placements the hook refused, with the
    error message, e.g. a ``_`` dim or a label that must be sharded jointly
    with another operand.
    """

    checked: list[RuleCheckPlacement] = field(default_factory=list)
    rejected: list[RuleCheckRejection] = field(default_factory=list)


def rulecheck(  # noqa: C901
    cls: type,
    args: Sequence[Any],
    placements: Mapping[str, SpmdType | Mapping[str, PerMeshAxisSpmdType]]
    | None = None,
    *,
    mesh: Mapping[str, int] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    local_tensor_mode: bool = False,
) -> RuleCheckReport | None:
    """Check that ``cls.spmd_typecheck`` describes what ``cls.forward`` computes.

    Args:
        cls: an autograd Function with an ``spmd_typecheck`` hook.
        args: positional arguments for ``cls.apply`` with *global* tensors.
            A string equal to a mesh axis name stands for that axis's process
            group.  Without ``mesh=`` the single string argument, if any,
            names the axis; with several string arguments ``mesh=`` is
            required.
        placements: per tensor argument (by ``forward`` parameter name), the
            placement to test: a dict of mesh axis name to ``R``/``I``/``V``/
            ``P``/``S(i)``, or an ``SpmdType``.  Every mesh axis must be
            covered.  ``None`` (the default, and the recommended way to call
            this) enumerates, per mesh axis: all replicated; for
            every dim size, every way of sharding one dim of that size on
            each subset of the tensor arguments together (so labels shared by
            any number of operands are reached); each floating tensor Partial
            alone and all of them Partial together (exercises ``linear_in``);
            and all Invariant.  Placements the hook rejects are skipped and
            reported.  Distinct dim sizes keep the enumeration small and
            sharp.
        mesh: mesh axis names and sizes, e.g. ``{"dp": 2, "tp": 2}``.  In
            enumeration mode it defaults to the group argument's axis, or
            ``{"shard": 2}`` if there is none; with several axes the product
            of the per-axis enumerations is tried (the hook pre-checks every
            combination; the kernel runs only for the accepted ones).
        rtol, atol: tolerances for comparing the reassembled output with the
            reference. Replicate and Invariant ranks must instead agree
            exactly; these tolerances do not relax that check.
        local_tensor_mode: simulate the ranks in-process under
            ``LocalTensorMode`` with a fake process group, so that
            ``torch.distributed`` calls inside the kernel execute and
            ``dist.get_rank(group)`` is per rank.  Required
            when an argument names a mesh axis as a group; also the way to
            check a kernel that reaches a process group some other way.  Off
            by default: the kernel runs once per rank on plain tensors with no
            distributed setup at all.

    Returns:
        A ``RuleCheckReport`` in enumeration mode, else ``None``.

    Raises:
        RuleCheckError: the reassembled outputs do not match the reference,
            ranks that the type says agree do not, or (enumeration mode) the
            hook rejected every placement.
        SpmdTypeError: the hook rejected the explicit placements.
    """
    args = tuple(args)
    if placements is not None:
        if mesh is None:
            raise TypeError("rulecheck: mesh= is required with explicit placements")
        _check_one(
            cls, args, placements, _Mesh(mesh), None, rtol, atol, local_tensor_mode
        )
        return None

    strings = [a for a in args if isinstance(a, str)]
    if mesh is None:
        # Without a mesh we cannot tell a group name from an ordinary string
        # argument, so the sole string (if any) is taken to name the group; with
        # several, the caller must pass mesh= to disambiguate.
        if len(strings) > 1:
            raise TypeError(
                "rulecheck: several string arguments; pass mesh= to say which of "
                f"them name process groups (strings: {strings})"
            )
        mesh = {strings[0] if strings else "shard": 2}
    # With a mesh, only a string equal to a mesh axis name is a process group;
    # any other string is an ordinary argument passed through to the kernel.
    m = _Mesh(mesh)
    forward_args = bind_forward_args(cls, args)
    tensors = {n: v for n, v in forward_args.items() if isinstance(v, torch.Tensor)}
    # Product of the per-axis enumerations.  Only the hook (a type-level
    # pre-check, no kernel run) sees every combination; the kernel runs for the
    # ones it accepts.
    per_axis = [_enumerate_placements(tensors, axis, m.size(axis)) for axis in m.names]
    candidates: list[RuleCheckPlacement] = []
    for combo in product(*per_axis):
        merged: RuleCheckPlacement = {n: {} for n in tensors}
        for part in combo:
            for n, v in part.items():
                merged[n].update(v)
        candidates.append(merged)
    if local_tensor_mode and dist.is_initialized():
        # _single_device (below) and every _run_local_tensor_mode set up their
        # own fake process group and _reset() the global mesh state; guard here,
        # before _single_device runs, so an already-initialized group is a clear
        # error rather than a raw torch failure that also corrupts that state.
        raise RuleCheckError(
            "rulecheck: local_tensor_mode sets up its own fake process group; "
            "call it with none initialized"
        )
    expected = None
    if local_tensor_mode:
        # The reference does not depend on the placement; compute it once
        # rather than once per accepted placement (each is a fake-pg setup).
        expected = _single_device(cls, args, m)
    elif not any(isinstance(a, str) and a in m.names for a in args):
        # Same for the plain backend: hoist the placement-independent reference
        # so it is not recomputed for every candidate. (A group argument would
        # make _check_one raise before the kernel runs, so skip it here.)
        expected = _plain_reference(cls, args, m)
    returns_types, _ = hook_shape(cls)
    report = RuleCheckReport()
    for placement in candidates:
        rejection = _hook_rejects(cls, args, placement, m)
        if rejection is not None:
            report.rejected.append(RuleCheckRejection(placement, rejection))
            continue
        try:
            _check_one(cls, args, placement, m, expected, rtol, atol, local_tensor_mode)
        except SpmdTypeError as e:
            if returns_types:
                # The hook already accepted this placement in the pre-check, so a
                # type error now is the runner's (an untyped output, a rank
                # mismatch, uncovered inputs): a defect in the hook, not a
                # refusal.
                raise RuleCheckError(
                    f"rulecheck: the hook accepted this placement but running it "
                    f"failed: {str(e).splitlines()[0]}\n  under placements "
                    f"{_fmt_placement(placement)}"
                ) from e
            # An imperative hook (out first) cannot be pre-checked by
            # _hook_rejects, so it refuses a placement by raising here; that is a
            # rejection, not a failure of the whole enumeration.
            report.rejected.append(
                RuleCheckRejection(placement, str(e).splitlines()[0])
            )
        except RuleCheckError as e:
            raise RuleCheckError(
                f"{e}\n  under placements {_fmt_placement(placement)}"
            ) from e
        else:
            report.checked.append(placement)
    if not report.checked:
        raise RuleCheckError(
            "rulecheck: the hook rejected every enumerated placement; nothing was "
            "checked. Rejections: "
            + "; ".join(f"{_fmt_placement(p)}: {why}" for p, why in report.rejected)
        )
    return report


def _enumerate_placements(
    tensors: Mapping[str, torch.Tensor], axis: str, n: int
) -> list[RuleCheckPlacement]:
    """Generate the candidate placements for one mesh axis of size ``n``.

    The result starts with every tensor Replicated.  For each tensor-dimension
    size divisible by ``n``, it then generates every nonempty subset of tensors
    having a dimension of that size.  Each selected tensor is sharded on one
    such dimension (including every choice when it has several matching
    dimensions), while every unselected tensor remains Replicated.  This
    reaches both independently sharded operands and any number of operands
    sharded together as a potential shared einsum label.

    It additionally generates one placement with each floating-point tensor
    Partial by itself, one with all floating-point tensors Partial together,
    and one with every tensor Invariant.  Non-floating tensors remain
    Replicated in Partial candidates because rulecheck cannot construct
    numeric Partial contributions for them.  Duplicate candidates are removed
    while preserving their first occurrence.

    These are placements for only ``axis``.  ``rulecheck`` takes the Cartesian
    product of the lists for all mesh axes and merges their per-tensor maps to
    obtain multi-axis placements.
    """
    names = list(tensors)
    base: RuleCheckPlacement = {name: {axis: R} for name in names}
    candidates: list[RuleCheckPlacement] = []
    seen: set[tuple] = set()

    def add(p: RuleCheckPlacement) -> None:
        key = tuple((k, repr(v[axis])) for k, v in sorted(p.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(p)

    add(dict(base))
    by_size: dict[int, dict[str, list[int]]] = {}
    for name, t in tensors.items():
        for d in range(t.ndim):
            if t.shape[d] % n == 0:
                by_size.setdefault(t.shape[d], {}).setdefault(name, []).append(d)
    for per in by_size.values():
        members = list(per)
        for k in range(1, len(members) + 1):
            for subset in combinations(members, k):
                for choice in product(*(per[nm] for nm in subset)):
                    add({**base, **{nm: {axis: S(d)} for nm, d in zip(subset, choice)}})
    floats = [nm for nm in names if tensors[nm].is_floating_point()]
    for nm in floats:
        add({**base, nm: {axis: P}})
    if len(floats) > 1:
        add({nm: {axis: P if nm in floats else R} for nm in names})
    add({nm: {axis: I} for nm in names})
    return candidates


def _hook_rejects(
    cls: type, args: tuple[Any, ...], placement: Mapping[str, Any], m: _Mesh
) -> str | None:
    """Run only the hook (return form) on annotated inputs; None if it accepts.

    The runner executes the kernel before the hook, so a placement the hook
    would refuse could crash the kernel first.  Imperative hooks need the
    outputs and cannot be pre-run; for them only placement constructibility is
    pre-checked (returning None once construction succeeds).
    """
    returns_types, _ = hook_shape(cls)
    # A placement that cannot even be constructed (a dim not divisible by the
    # axis sizes sharding it, an integer tensor asked to be Partial) is a
    # rejection too, not a reason to abort the enumeration; _bind builds the
    # distributors and surfaces those errors for either hook form.  A hook
    # naming a parameter that ``forward`` lacks raises TypeError from
    # hook_kwargs, as run_typecheck does.
    try:
        forward_args, dists = _bind(cls, args, placement, m)
        if not returns_types:
            return None
        with set_current_mesh(m.axes()):
            local = {
                name: dists[name].annotate(dists[name].piece(0))
                if name in dists
                else value
                for name, value in forward_args.items()
            }
            with typecheck(local=False):
                cls.spmd_typecheck(**hook_kwargs(cls, local))
    except (SpmdTypeError, RuleCheckError) as e:
        return str(e).splitlines()[0]
    return None


def _fmt_placement(placement: Mapping[str, Mapping[str, Any]]) -> str:
    return (
        "{"
        + ", ".join(
            f"{n}: {{{', '.join(f'{a}: {t!r}' for a, t in p.items())}}}"
            for n, p in placement.items()
        )
        + "}"
    )


def _check_one(
    cls: type,
    args: tuple[Any, ...],
    placements: Mapping[str, Any],
    m: _Mesh,
    expected: Any | None,
    rtol: float,
    atol: float,
    local_tensor_mode: bool,
) -> None:
    uses_groups = any(isinstance(a, str) and a in m.names for a in args)
    if local_tensor_mode:
        _run_local_tensor_mode(cls, args, placements, m, expected, rtol, atol)
    elif uses_groups:
        raise RuleCheckError(
            "rulecheck: an argument names a mesh axis as a process group; pass "
            "local_tensor_mode=True to simulate the ranks in-process under "
            "LocalTensorMode with a fake process group"
        )
    else:
        _run_plain(cls, args, placements, m, expected, rtol, atol)


def _bind(
    cls: type, args: tuple[Any, ...], placements: Mapping[str, Any], m: _Mesh
) -> tuple[dict[str, object], dict[str, _Distributor]]:
    """Bind args and construct distributors for the tensor arguments."""
    forward_args = bind_forward_args(cls, args)
    dists: dict[str, _Distributor] = {}
    for name, value in forward_args.items():
        if isinstance(value, torch.Tensor):
            if name not in placements:
                raise RuleCheckError(f"rulecheck: no placement given for {name!r}")
            dists[name] = _Distributor(name, value, placements[name], m)
        else:
            # A tensor nested inside a container (e.g. a ``*args`` parameter,
            # which binds to a tuple) gets no _Distributor, so it would be fed
            # to the kernel whole and never sharded -- the check would pass
            # vacuously.  Refuse it rather than report a hollow success.
            nested = _iter_tensors(value)
            if nested:
                raise TypeError(
                    f"rulecheck: argument {name!r} carries {len(nested)} tensor(s) "
                    f"nested inside a {type(value).__name__} (e.g. a *args or a "
                    f"tuple/list forward parameter); rulecheck can only place a "
                    f"tensor passed as its own positional forward argument"
                )
    return forward_args, dists


def _leaves(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (tuple, list)) else [value]


def _compare(
    outputs: Any,
    expected: Any,
    by_rank_of: Callable[[int], dict[int, torch.Tensor]],
    m: _Mesh,
    rtol: float,
    atol: float,
) -> None:
    """``outputs`` is one rank's (or the LocalTensor) output carrying the
    derived types; ``by_rank_of(i)`` gives every rank's value of output i."""
    out_leaves, exp_leaves = _leaves(outputs), _leaves(expected)
    if len(out_leaves) != len(exp_leaves):
        raise RuleCheckError(
            f"rulecheck: kernel returned {len(out_leaves)} value(s) but reference "
            f"returned {len(exp_leaves)}"
        )
    for i, (typed, exp) in enumerate(zip(out_leaves, exp_leaves)):
        if not isinstance(typed, torch.Tensor):
            continue
        label = f"output {i}" if len(out_leaves) > 1 else "the output"
        got = _reassemble(label, typed, by_rank_of(i), m)
        if got.shape != exp.shape:
            raise RuleCheckError(
                f"rulecheck: {label} reassembles to shape {tuple(got.shape)} but the "
                f"reference has shape {tuple(exp.shape)}; the derived sharding does "
                f"not match the kernel"
            )
        if not torch.allclose(got.to(exp.dtype), exp, rtol=rtol, atol=atol):
            by_name = _axis_names()
            typ = {
                _axis_name(a, by_name, label): t
                for a, t in get_local_type(typed).items()
            }
            raise RuleCheckError(
                f"rulecheck: {label} reassembled per its derived type {typ} "
                f"{get_partition_spec(typed) or ''} differs from the reference "
                f"(max abs diff {(got.to(exp.dtype) - exp).abs().max().item():.3g}). "
                f"The hook's claim about which dims are sharded or partial does "
                f"not describe what the kernel computes."
            )


def _plain_reference(cls: type, args: tuple[Any, ...], m: _Mesh) -> Any:
    """The kernel on the global tensors: the reference the plain backend checks
    each placement's reassembly against."""
    with set_current_mesh(m.axes()), torch.no_grad():
        return cls.apply(*args)


def _run_plain(
    cls: type,
    args: tuple[Any, ...],
    placements: Mapping[str, Any],
    m: _Mesh,
    expected: Any | None,
    rtol: float,
    atol: float,
) -> None:
    """No collectives: run the kernel on ordinary tensors, once per rank."""
    forward_args, dists = _bind(cls, args, placements, m)
    per_rank: list[Any] = []
    with set_current_mesh(m.axes()):
        for rank in range(m.world):
            local_args = [
                dists[name].annotate(dists[name].piece(rank))
                if name in dists
                else value
                for name, value in forward_args.items()
            ]
            with typecheck(local=False):
                per_rank.append(cls.apply(*local_args))
        if expected is None:
            with torch.no_grad():
                expected = cls.apply(*args)
        _compare(
            per_rank[0],
            expected,
            lambda i: {r: _leaves(per_rank[r])[i] for r in range(m.world)},
            m,
            rtol,
            atol,
        )


def _per_rank_get_rank(device_mesh: Any, m: _Mesh):
    """Under ``LocalTensorMode`` every simulated rank shares one process, so
    ``torch.distributed.get_rank`` would answer 0 for all of them.  Patch it to
    return a per-rank ``SymInt`` (``LocalIntNode``), globally or along the axis
    whose group is passed, so rank-dependent kernels (a vocab offset computed
    from the rank, say) run correctly.  Only ``dist.get_rank`` looked up at call
    time is covered; ``from torch.distributed import get_rank`` is not."""
    groups = [(device_mesh.get_group(n), n) for n in m.names]  # keeps them alive

    def get_rank(group: Any = None) -> int | torch.SymInt:
        if group is None:
            return torch.SymInt(LocalIntNode({r: r for r in range(m.world)}))
        for g, axis in groups:
            if g is group:
                return torch.SymInt(
                    LocalIntNode({r: m.coords(r)[axis] for r in range(m.world)})
                )
        raise RuleCheckError(
            "rulecheck: the kernel called dist.get_rank on a process group that is "
            "not one of the mesh axes' groups; only groups passed as arguments (a "
            "string naming a mesh axis) are simulated"
        )

    return mock.patch.object(dist, "get_rank", get_rank)


def _single_device(cls: type, args: tuple[Any, ...], m: _Mesh) -> Any:
    """The kernel on the global tensors with every group a world of size one:
    the single-device program the sharded run is checked against."""
    _reset()
    dist.init_process_group(backend="fake", rank=0, world_size=1, store=FakeStore())
    try:
        device_mesh = init_device_mesh(
            "cpu", (1,) * len(m.names), mesh_dim_names=tuple(m.names)
        )
        with LocalTensorMode(1) as mode, set_current_mesh(device_mesh), torch.no_grad():
            local_args = [
                mode.rank_map(lambda r, v=a: v.clone())
                if isinstance(a, torch.Tensor)
                else device_mesh.get_group(a)
                if isinstance(a, str) and a in m.names
                else a
                for a in args
            ]
            outputs = cls.apply(*local_args)

        def unwrap(v: Any) -> Any:
            return v._local_tensors[0] if isinstance(v, LocalTensor) else v

        if isinstance(outputs, (tuple, list)):
            return type(outputs)(unwrap(v) for v in outputs)
        return unwrap(outputs)
    finally:
        dist.destroy_process_group()
        _reset()


def _run_local_tensor_mode(
    cls: type,
    args: tuple[Any, ...],
    placements: Mapping[str, Any],
    m: _Mesh,
    expected: Any | None,
    rtol: float,
    atol: float,
) -> None:
    """Simulate the ranks in-process under LocalTensorMode with a fake process group."""
    if dist.is_initialized():
        raise RuleCheckError(
            "rulecheck: local_tensor_mode sets up its own fake process group; "
            "call it with none initialized"
        )
    if expected is None:
        expected = _single_device(cls, args, m)
    _reset()
    dist.init_process_group(
        backend="fake", rank=0, world_size=m.world, store=FakeStore()
    )
    try:
        device_mesh = init_device_mesh(
            "cpu", tuple(m.sizes[n] for n in m.names), mesh_dim_names=tuple(m.names)
        )
        with (
            LocalTensorMode(m.world) as mode,
            set_current_mesh(device_mesh),
            _per_rank_get_rank(device_mesh, m),
        ):
            forward_args, dists = _bind(cls, args, placements, m)
            local_args = []
            for name, value in forward_args.items():
                if name in dists:
                    local_args.append(
                        dists[name].annotate(mode.rank_map(dists[name].piece))
                    )
                elif isinstance(value, str) and value in m.names:
                    local_args.append(device_mesh.get_group(value))
                else:
                    local_args.append(value)
            with typecheck(local=False):
                outputs = cls.apply(*local_args)

            def by_rank(i: int) -> dict[int, torch.Tensor]:
                out = _leaves(outputs)[i]
                if not isinstance(out, LocalTensor):
                    raise RuleCheckError(
                        f"rulecheck: output {i} is not a LocalTensor; did the kernel run?"
                    )
                return dict(out._local_tensors)

            _compare(outputs, expected, by_rank, m, rtol, atol)
    finally:
        dist.destroy_process_group()
        _reset()


__all__ = [
    "RuleCheckError",
    "RuleCheckPlacement",
    "RuleCheckRejection",
    "RuleCheckReport",
    "rulecheck",
]
