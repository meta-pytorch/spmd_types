# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Composable typing rules for ``spmd_typecheck`` hooks.

An autograd Function that performs collectives or wraps an opaque kernel has
to declare its own typing rule in an ``spmd_typecheck`` hook.  This module
provides the pieces to write that hook as a composition of type-only
operations: ``rules.einsum`` for the einsum sharding rule, the type-only
collectives and transitions (``rules.all_reduce``, ``rules.all_gather``, ...),
and ``rules.ignore`` for inputs that do not matter.  Intermediates are
``NdimWithSpmdType`` values (a rank plus SPMD annotations).  Every hook is
coverage checked: each tensor argument of ``forward`` must be accounted for and
every output typed.

The guide and reference is ``docs/rules.md``; ``spmd_types.rulecheck`` tests a
hook numerically against its kernel.  Example::

    class LinearAllReduce(torch.autograd.Function):
        @staticmethod
        def spmd_typecheck(*, x, weight, bias, group):
            y = rules.einsum("mk,nk->mn", x, weight)
            y = rules.all_reduce(y, group, src=P, dst=I)
            if bias is not None:
                y = rules.einsum("mn,n->mn", y, bias)
            return y
"""

from __future__ import annotations

import inspect
import re
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from spmd_types._coverage import _cov, _coverage, _mark_asserted, _touch  # noqa: F401
from spmd_types._mesh_axis import MeshAxis
from spmd_types._type_attr import _LOCAL_TYPE_ATTR, get_local_type
from spmd_types.runtime import (
    _PARTITION_SPEC_ATTR,
    _set_local_type,
    _set_partition_spec,
    _update_axis_in_partition_spec,
    assert_type,
    get_partition_spec,
    has_local_type,
)
from spmd_types.types import (
    _canonicalize_shard,
    DeviceMeshAxis,
    format_axis,
    I,
    LocalSpmdType,
    normalize_axis,
    normalize_partition_spec,
    P,
    partition_spec_get_shard,
    PartitionSpec,
    PerMeshAxisLocalSpmdType,
    PerMeshAxisSpmdType,
    R,
    Shard,
    SpmdTypeError,
    to_local_type,
    V,
)
from torch.utils._pytree import tree_flatten

# =============================================================================
# Phantoms
# =============================================================================


class NdimWithSpmdType:
    """A tensor as the checker sees it: a rank and an SPMD type, nothing else.

    Returned by every ``rules`` operation.  Carries the same two SPMD annotation
    attributes a real tensor would, so ``assert_type``, ``assert_type_like`` and
    ``get_partition_spec`` accept it unchanged, and real annotated tensors are
    accepted wherever one of these is.  The type machinery reads only ``ndim``
    from a tensor, so this stand-in never claims sizes an opaque kernel might
    not preserve.
    """

    __slots__ = ("ndim", _LOCAL_TYPE_ATTR, _PARTITION_SPEC_ATTR)

    def __init__(self, ndim: int) -> None:
        self.ndim = ndim

    def __repr__(self) -> str:
        spec = get_partition_spec(self)
        text = f"NdimWithSpmdType(ndim={self.ndim}, {get_local_type(self)}"
        if spec is not None:
            text += f", {spec!r}"
        return text + ")"


def _is_tensor_or_ndim(value: object) -> bool:
    """Whether ``value`` can carry an SPMD type (a tensor or an ``NdimWithSpmdType``)."""
    return isinstance(value, (torch.Tensor, NdimWithSpmdType))


def _make(
    ndim: int, local_type: LocalSpmdType, spec: PartitionSpec | None
) -> NdimWithSpmdType:
    """An ``NdimWithSpmdType`` of the given rank carrying ``local_type`` and ``spec``."""
    d = NdimWithSpmdType(ndim)
    _set_local_type(d, dict(local_type))
    _set_partition_spec(d, _maybe_spec(spec))
    return d


def _maybe_spec(spec: PartitionSpec | None) -> PartitionSpec | None:
    """``spec`` unless it shards nothing, in which case ``None``."""
    if spec is None or all(e is None for e in spec):
        return None
    return spec


# =============================================================================
# Coverage tracking (active while an spmd_typecheck hook runs)
# =============================================================================


def ignore(*tensors: torch.Tensor) -> None:
    """Declare tensors that take no part in the forward value (e.g. tensors
    saved only for backward), satisfying the input coverage check."""
    _touch(*tensors)


# =============================================================================
# einsum
# =============================================================================

_ELLIPSIS = "..."
_OPAQUE = "_"
_TOKEN_RE = re.compile(r"\s*(?:(\.\.\.)|([^\W\d]))")  # a letter (any script) or _


@dataclass(frozen=True)
class _Operand:
    """One side of an einsum term: its labels, with ``_`` and ``...`` as written."""

    labels: tuple[str, ...]  # single-char labels, '_' or '...'

    @property
    def has_ellipsis(self) -> bool:
        return _ELLIPSIS in self.labels

    def expand(self, ell_rank: int) -> list[str]:
        out: list[str] = []
        for lbl in self.labels:
            if lbl == _ELLIPSIS:
                out.extend(f"...{i}" for i in range(ell_rank))
            else:
                out.append(lbl)
        return out


@dataclass(frozen=True)
class _Equation:
    """A parsed einsum equation: the text, its input operands, and its output."""

    text: str
    inputs: tuple[_Operand, ...]
    output: _Operand


def _parse_operand(text: str, *, full_equation: str) -> _Operand:
    """Parse one operand term; ``full_equation`` is included in diagnostics."""
    labels: list[str] = []
    pos = 0
    stripped = text.strip()
    if not stripped:
        raise TypeError(f"rules.einsum: empty operand in {full_equation!r}")
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None or m.end() == pos:
            if text[pos:].strip() == "":
                break
            raise TypeError(
                f"rules.einsum: unexpected {text[pos:].strip()[0]!r} in "
                f"{full_equation!r}; "
                f"labels are single letters (any script), '_' or '...'"
            )
        pos = m.end()
        labels.append(m.group(1) or m.group(2))
    if labels.count(_ELLIPSIS) > 1:
        raise TypeError(
            f"rules.einsum: operand {stripped!r} has two '...' in {full_equation!r}"
        )
    named = [lbl for lbl in labels if lbl not in (_ELLIPSIS, _OPAQUE)]
    if len(named) != len(set(named)):
        raise TypeError(
            f"rules.einsum: operand {stripped!r} repeats a label in "
            f"{full_equation!r}; "
            f"repeated labels (diagonals) are not supported"
        )
    return _Operand(tuple(labels))


def _parse_equation(equation: str) -> _Equation:
    """Parse and validate an einsum equation (one output, output labels bound)."""
    if equation.count("->") != 1:
        raise TypeError(
            f"rules.einsum: equation {equation!r} must contain exactly one '->'"
        )
    lhs, rhs = equation.split("->")
    inputs = tuple(_parse_operand(p, full_equation=equation) for p in lhs.split(","))
    if "," in rhs:
        raise TypeError(
            f"rules.einsum: {equation!r} lists several outputs; write one "
            f"rules.einsum per output instead"
        )
    output = (
        _Operand(())
        if rhs.strip() == ""
        else _parse_operand(rhs, full_equation=equation)
    )
    in_labels = {lbl for op in inputs for lbl in op.labels}
    for lbl in output.labels:
        if lbl not in (_ELLIPSIS, _OPAQUE) and lbl not in in_labels:
            raise TypeError(
                f"rules.einsum: output label {lbl!r} does not appear on any "
                f"input in {equation!r}"
            )
    if output.has_ellipsis and not any(op.has_ellipsis for op in inputs):
        raise TypeError(
            f"rules.einsum: output uses '...' but no input does in {equation!r}"
        )
    return _Equation(equation, inputs, output)


def _actual_axes(
    value: torch.Tensor | NdimWithSpmdType,
) -> list[tuple[MeshAxis, ...]]:
    """Per dim of ``value``, the mesh axes sharding it (outermost first)."""
    spec = get_partition_spec(value)
    if spec is None:
        return [()] * value.ndim
    out: list[tuple[MeshAxis, ...]] = []
    for entry in normalize_partition_spec(spec):
        if entry is None:
            out.append(())
        elif isinstance(entry, tuple):
            out.append(tuple(entry))
        else:
            out.append((entry,))
    return out


def _fmt_axes(axes: Sequence[MeshAxis]) -> str:
    """Format a tuple of mesh axes for an error message."""
    if not axes:
        return "None"
    return "(" + ", ".join(format_axis(a) for a in axes) + ")"


def _entry(axes: Sequence[Any]) -> Any:
    """A ``PartitionSpec`` entry for the axes sharding one dim."""
    if not axes:
        return None
    if len(axes) == 1:
        return axes[0]
    return tuple(axes)


def _describe_shape(value: torch.Tensor | NdimWithSpmdType) -> str:
    """``shape (...)`` for a tensor, ``N dim(s)`` for an ``NdimWithSpmdType``."""
    if isinstance(value, torch.Tensor):
        return f"shape {tuple(value.shape)}"
    return f"{value.ndim} dim(s)"


def _bind_einsum(
    eq: _Equation, operands: tuple[torch.Tensor | NdimWithSpmdType, ...]
) -> tuple[list[list[str]], int]:
    """Bind the parsed input terms of ``eq`` to the concrete operand ranks.

    Validates the operand count and that every operand is a tensor-like value.
    A term without ``...`` must have exactly one dimension per label.  For a
    term with ``...``, the remaining rank after its explicit labels determines
    the ellipsis rank; every ellipsis in the equation must represent that same
    number of dimensions.

    Returns one label list per input with ``...`` expanded to shared synthetic
    labels (``...0``, ``...1``, and so on), plus the ellipsis rank used later
    to expand the output term.
    """
    if len(operands) != len(eq.inputs):
        raise TypeError(
            f"rules.einsum: {eq.text!r} has {len(eq.inputs)} operand(s) but "
            f"{len(operands)} value(s) were passed"
        )
    ell_rank: int | None = None
    ell_from: tuple[int, str, str] | None = None  # (operand index, term, shape)
    for i, (op, value) in enumerate(zip(eq.inputs, operands)):
        if not _is_tensor_or_ndim(value):
            raise TypeError(
                f"rules.einsum: operands must be tensors or NdimWithSpmdType, got "
                f"{type(value).__name__}; write a separate rule for the None case"
            )
        term = "".join(op.labels)
        where = f"operand {i} ({term!r}, {_describe_shape(value)})"
        explicit = len(op.labels) - (1 if op.has_ellipsis else 0)
        if not op.has_ellipsis:
            if value.ndim != explicit:
                raise SpmdTypeError(
                    f"rules.einsum: {eq.text!r}: {where} has {explicit} label(s) but "
                    f"{value.ndim} dim(s)"
                )
            continue
        rank = value.ndim - explicit
        if rank < 0:
            raise SpmdTypeError(
                f"rules.einsum: {eq.text!r}: {where} needs at least {explicit} dim(s) "
                f"for its explicit labels but has {value.ndim}"
            )
        if ell_rank is not None and rank != ell_rank:
            assert ell_from is not None
            j, jterm, jshape = ell_from
            raise SpmdTypeError(
                f"rules.einsum: {eq.text!r}: '...' must stand for the same number of "
                f"dims on every operand, but operand {j} ({jterm!r}, {jshape}) gives "
                f"it {ell_rank} and {where} gives it {rank}"
            )
        ell_rank, ell_from = rank, (i, term, _describe_shape(value))
    ell = ell_rank or 0
    return [op.expand(ell) for op in eq.inputs], ell


def _einsum_label_axes(  # noqa: C901
    eq: _Equation,
    operands: tuple[torch.Tensor | NdimWithSpmdType, ...],
    labels: list[list[str]],
    out_labels: set[str],
) -> tuple[dict[str, tuple[MeshAxis, ...]], set[MeshAxis]]:
    """Derive the mesh-axis meaning of the expanded einsum labels.

    ``labels`` contains the per-input labels produced by ``_bind_einsum``;
    ``out_labels`` contains the expanded labels retained by the output.  This
    function matches each input dimension with its actual PartitionSpec and
    records the ordered mesh axes that shard each label.

    While doing so it enforces the global einsum invariants: ``_`` dimensions
    cannot be sharded, repeated occurrences of a label must have identical
    sharding, one mesh axis cannot shard two different labels, and an operand
    that omits a sharded label cannot be Varying on that label's mesh axes.

    Returns the sharding axes for every label and the mesh axes that shard
    input labels omitted from the output.  The latter are contraction axes, so
    ``einsum`` marks the result Partial on them.
    """
    label_axes: dict[str, tuple[MeshAxis, ...]] = {}
    axis_label: dict[MeshAxis, str] = {}
    for i, (value, lbls) in enumerate(zip(operands, labels)):
        for dim, (lbl, axes) in enumerate(zip(lbls, _actual_axes(value))):
            if lbl == _OPAQUE:
                if axes:
                    raise SpmdTypeError(
                        f"rules.einsum: operand {i} of {eq.text!r} is sharded on "
                        f"{_fmt_axes(axes)} at dim {dim}, which the equation "
                        f"marks '_' (must not be sharded)"
                    )
                continue
            prev = label_axes.setdefault(lbl, axes)
            if prev != axes:
                raise SpmdTypeError(
                    f"rules.einsum: label {lbl!r} of {eq.text!r} is sharded "
                    f"inconsistently: {_fmt_axes(prev)} vs {_fmt_axes(axes)} "
                    f"(operand {i}, dim {dim})"
                )
            for a in axes:
                other = axis_label.setdefault(a, lbl)
                if other != lbl:
                    raise SpmdTypeError(
                        f"rules.einsum: axis {format_axis(a)} shards both {other!r} "
                        f"and {lbl!r} in {eq.text!r}; an axis may shard only one dim"
                    )
    # Broadcast rule: an operand without label l must not be Varying on an
    # axis that shards l (it would be an irregular per-rank value, not a
    # broadcast of one value).
    for a, lbl in axis_label.items():
        for i, (value, lbls) in enumerate(zip(operands, labels)):
            if lbl in lbls:
                continue
            typ = get_local_type(value).get(a)
            if typ is V:
                raise SpmdTypeError(
                    f"rules.einsum: operand {i} of {eq.text!r} broadcasts over "
                    f"{lbl!r}, which is sharded on {format_axis(a)}, so it must "
                    f"be replicated on {format_axis(a)}, but it is Varying"
                )
    partial_axes = {
        a for lbl, axes in label_axes.items() if lbl not in out_labels for a in axes
    }
    return label_axes, partial_axes


def _linear_groups(
    linear_in: Sequence[int | Sequence[int]], n: int
) -> list[frozenset[int]]:
    """Normalize ``linear_in`` into disjoint operand groups (``(0, 1)`` two singletons, ``((0, 1),)`` one joint)."""
    groups: list[frozenset[int]] = []
    seen: set[int] = set()
    for entry in linear_in:
        members = (entry,) if isinstance(entry, int) else tuple(entry)
        for i in members:
            if not isinstance(i, int) or not 0 <= i < n:
                raise TypeError(
                    f"rules.einsum: linear_in refers to operand {i!r}; there are {n} operand(s)"
                )
            if i in seen:
                raise TypeError(f"rules.einsum: operand {i} appears twice in linear_in")
            seen.add(i)
        if members:
            groups.append(frozenset(members))
    return groups


def _partial_axes_from_inputs(
    equation: str, types: list[dict], groups: list[frozenset[int]]
) -> dict:
    """Per mesh axis, the Partial that ``linear_in`` lets pass through (all Partials in one group, the rest Replicate)."""
    axes = {a for t in types for a in t}
    forced: dict = {}
    for a in axes:
        p_ops = [i for i, t in enumerate(types) if t.get(a) is P]
        if not p_ops:
            continue
        owners = [g for g in groups if any(i in g for i in p_ops)]
        if any(not any(i in g for g in groups) for i in p_ops) or len(owners) != 1:
            joined = ", ".join(map(str, p_ops))
            if len(p_ops) == 1:
                # Joint linearity of a single operand is just linearity in it;
                # linear_in=((0),) would read as linear_in=(0,), so only the
                # one meaningful form is suggested.
                hint = f"declare linear_in=({joined},) if the op is linear in it"
            else:
                hint = (
                    f"declare linear_in=({joined},) for multilinear, or "
                    f"linear_in=(({joined}),) for jointly linear"
                )
            raise SpmdTypeError(
                f"rules.einsum: {equation!r}: operand(s) {p_ops} are Partial on axis "
                f"{format_axis(a)}; a Partial may only pass through operands the op "
                f"is linear in, all within one linear_in group ({hint})"
            )
        (owner,) = owners
        # An operand with no entry on this axis is unsharded there, i.e.
        # Replicate; ``get(a, R)`` applies that reading uniformly to both the
        # joint-linearity check and the combination check below.
        if len(owner) > 1 and any(types[i].get(a, R) is not P for i in owner):
            raise SpmdTypeError(
                f"rules.einsum: {equation!r}: operands {sorted(owner)} are declared "
                f"jointly linear, so on axis {format_axis(a)} either all or none of "
                f"them may be Partial; {p_ops} are Partial and the rest are not "
                f"(P + R is affine: the Replicate term would be summed once per rank)"
            )
        for i, t in enumerate(types):
            if i in p_ops:
                t[a] = R  # combine as if replicated; restored to P below
            elif t.get(a, R) is not R:
                raise SpmdTypeError(
                    f"rules.einsum: {equation!r}: operand {i} is {t[a]!r} on axis "
                    f"{format_axis(a)} while operand(s) {p_ops} are Partial there; "
                    f"a Partial combines only with Replicate"
                )
        forced[a] = P
    return forced


def einsum(
    equation: str,
    *operands: torch.Tensor | NdimWithSpmdType,
    linear_in: Sequence[int | Sequence[int]] = (),
    out: torch.Tensor | None = None,
) -> torch.Tensor | NdimWithSpmdType:
    """Type an einsum-shaped computation over tensors or ``NdimWithSpmdType``.

    ``equation`` is einsum notation with single-letter labels, plus ``_`` for
    a dim that must not be sharded and ``...`` for pass-through dims.  A
    scalar output is an empty right-hand side (``"tv,t->"``).  One output per
    call: a kernel with several outputs composes several ``rules.einsum`` calls.

    Per mesh axis, from the operands' actual placements:

    - a sharded label that reaches the output keeps its sharding there;
    - a sharded label that reaches no output (a contraction) makes the
      output Partial on that axis;
    - an operand that lacks a sharded label must be replicated on that axis;
    - labels are sharded identically wherever they appear, an axis shards at
      most one label, and ``_`` dims are unsharded.

    A named contracted label is a sum, as in einsum: the op's result is the
    sum of its results over slices of that dim (additive along the index).
    That is what makes a sharded contraction a Partial.  A reduction that is
    not additive (``max``, ``logsumexp``, ``mean``, a norm) must use ``_``
    for that dim, which requires it to be complete on every rank.

    Axes that shard nothing are combined with the ordinary typing rule (``R``
    with ``V`` is ``V``, ``I`` does not mix).  A Partial *operand* is a
    different claim, about linearity in the operand's value rather than
    additivity along an index, and must be declared with ``linear_in``: the
    operand positions the op is linear in.  ``(0, 1)`` means linear in each
    separately (multilinear, as a real einsum is); ``((0, 1),)`` means jointly
    linear (``add``); the default ``()`` claims nothing.  Per mesh axis,
    Partial operands may pass through only if they all lie in one group and
    every other operand is Replicate there.

    With ``out=``, the derived type is stamped onto that real output tensor
    (rank checked) and ``out`` is returned, as with torch's ``out=``.
    """
    from spmd_types._checker import infer_output_type, OpLinearity

    eq = _parse_equation(equation)
    groups = _linear_groups(linear_in, len(operands))
    _touch(*operands)
    labels, ell_rank = _bind_einsum(eq, operands)
    out_labels = eq.output.expand(ell_rank)
    label_axes, partial_axes = _einsum_label_axes(eq, operands, labels, set(out_labels))
    types = [dict(get_local_type(v)) for v in operands]
    forced_partial = _partial_axes_from_inputs(equation, types, groups)
    try:
        local_type = infer_output_type(
            types, out_partial_axes=partial_axes, linearity=OpLinearity.NONLINEAR
        )
    except SpmdTypeError as e:
        e.args = (f"rules.einsum: {equation!r}: {e.args[0]}", *e.args[1:])
        raise
    local_type.update(forced_partial)
    entries = [
        None if lbl == _OPAQUE else _entry(label_axes.get(lbl, ()))
        for lbl in out_labels
    ]
    return _finish(_make(len(out_labels), local_type, PartitionSpec(*entries)), out)


def _finish(
    result: NdimWithSpmdType, out: torch.Tensor | None
) -> torch.Tensor | NdimWithSpmdType:
    """Return ``result``, or stamp it onto ``out`` and return ``out`` when given."""
    if out is None:
        return result
    output(out, result)
    return out


# =============================================================================
# Type-only collectives and local transitions
# =============================================================================


@dataclass(frozen=True)
class _Transition:
    """A type-only collective or local transition: its kind, axis, ``src`` and ``dst``."""

    kind: str
    axis: DeviceMeshAxis
    src: PerMeshAxisSpmdType
    dst: PerMeshAxisSpmdType

    def __repr__(self) -> str:
        return (
            f"rules.{self.kind}({format_axis(self.axis)}, "
            f"src={self.src!r}, dst={self.dst!r})"
        )

    def _accepts_src(self, actual: PerMeshAxisLocalSpmdType) -> bool:
        # Unlike the runtime collectives, no implicit V -> P: a rule that
        # reduces a Varying value must say src=V, so the reinterpret is on
        # record.  A hook that accepted V under src=P would type a kernel
        # summing unrelated shards as if it had completed a contraction.
        return actual is to_local_type(self.src)


def _check_pair(
    kind: str,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
    allowed: Sequence[tuple[object, object]],
) -> None:
    """Reject a ``src -> dst`` pair the type system does not admit for this collective."""

    def matches(pattern: object, typ: PerMeshAxisSpmdType) -> bool:
        if pattern == "V|S":
            return typ is V or isinstance(typ, Shard)
        if pattern == "P|V":
            return typ in (P, V)
        return typ is pattern

    if not isinstance(src, (PerMeshAxisLocalSpmdType, Shard)) or not isinstance(
        dst, (PerMeshAxisLocalSpmdType, Shard)
    ):
        raise TypeError(
            f"rules.{kind}: src and dst must be R, I, V, P or S(i); "
            f"got src={src!r}, dst={dst!r}"
        )
    if not any(matches(s, src) and matches(d, dst) for s, d in allowed):

        def fmt_pattern(p: object) -> str:
            return p if isinstance(p, str) else repr(p)

        options = ", ".join(f"{fmt_pattern(s)} -> {fmt_pattern(d)}" for s, d in allowed)
        raise SpmdTypeError(
            f"rules.{kind}: {src!r} -> {dst!r} is not a transition the type system "
            f"admits for {kind}. Allowed: {options}"
        )


def _rank_change(
    tr: _Transition, ndim: int, spec: PartitionSpec | None
) -> tuple[int, PartitionSpec | None]:
    """The rank and spec after the stack (``all_gather(src=V)``) or unbind (``reduce_scatter(dst=V)``) form."""
    if tr.kind == "all_gather" and tr.src is V:
        return ndim + 1, PartitionSpec(None, *(spec or [None] * ndim))
    if tr.kind == "reduce_scatter" and tr.dst is V:
        if ndim == 0:
            raise SpmdTypeError(f"{tr!r}: cannot unbind dim 0 of a scalar")
        if spec is not None and spec[0] is not None:
            raise SpmdTypeError(
                f"{tr!r}: dim 0 is unbound across ranks but is sharded on {spec[0]!r}"
            )
        return ndim - 1, _maybe_spec(
            PartitionSpec(*spec[1:])
        ) if spec is not None else None
    return ndim, spec


def _transition(  # noqa: C901
    x: torch.Tensor | NdimWithSpmdType,
    kind: str,
    axis: DeviceMeshAxis,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
    allowed: Sequence[tuple[object, object]],
    out: torch.Tensor | None = None,
) -> torch.Tensor | NdimWithSpmdType:
    """Apply one collective or local type transition to ``x``.

    ``allowed`` describes the operation-specific ``src``/``dst`` pairs.  Once
    that declaration and ``x`` are validated, the transition checks that
    ``x`` actually has ``src`` on ``axis``.  An axis not yet present on ``x``
    is established as ``src``; a singleton axis needs no stored annotation and
    skips this check.

    ``S(i)`` is checked in two parts: its local type must be ``V`` and its
    PartitionSpec must shard dimension ``i`` on this axis.  Negative shard
    dimensions are canonicalized against the input rank.  The result copies
    every other annotation, replaces this axis with ``dst``, and updates its
    PartitionSpec entry accordingly.  Stack-form ``all_gather(src=V)`` then
    adds an output dimension, while unbind-form ``reduce_scatter(dst=V)``
    removes one.

    Returns a new ``NdimWithSpmdType``, or stamps those derived annotations
    onto ``out`` and returns it when ``out`` is supplied.
    """
    _check_pair(kind, src, dst, allowed)
    if not _is_tensor_or_ndim(x):
        raise TypeError(
            f"rules.{kind}: expected a tensor or NdimWithSpmdType, got {type(x).__name__}"
        )
    _touch(x)
    tr = _Transition(kind, axis, src, dst)
    mesh_axis = normalize_axis(axis)
    ndim = x.ndim
    if mesh_axis.size() == 1:
        # Nothing to check on a singleton axis, but the stack/unbind forms
        # still change the kernel's output rank.
        ndim, new_spec = _rank_change(tr, ndim, get_partition_spec(x))
        return _finish(_make(ndim, dict(get_local_type(x)), new_spec), out)
    if mesh_axis not in get_local_type(x):
        # Untyped on this axis: the operation's src establishes it, the way
        # the real collective would.
        assert_type(x, {axis: src})
    local_type = dict(get_local_type(x))
    spec = get_partition_spec(x)
    actual = local_type[mesh_axis]
    if not tr._accepts_src(actual):
        raise SpmdTypeError(
            f"{tr!r} expects its input to be {src!r} on axis "
            f"{format_axis(mesh_axis)}, but it is {actual!r}"
        )
    src_c = _canonicalize_shard(src, ndim)
    dst_c = _canonicalize_shard(dst, ndim)
    if (
        kind == "convert"
        and isinstance(src_c, Shard)
        and isinstance(dst_c, Shard)
        and src_c != dst_c
    ):
        raise SpmdTypeError(
            f"{tr!r}: cannot change the shard dimension from {src_c!r} to "
            f"{dst_c!r} without communication; use all_to_all instead"
        )
    if isinstance(src_c, Shard):
        want = src_c
        have = partition_spec_get_shard(spec, mesh_axis)
        if have is None:
            raise SpmdTypeError(
                f"{tr!r} expects its input sharded on dim {want.dim} along "
                f"{format_axis(mesh_axis)}, but it is not sharded there"
            )
        if have != want:
            raise SpmdTypeError(
                f"{tr!r} expects its input sharded on dim {want.dim} along "
                f"{format_axis(mesh_axis)}, but it is sharded on dim {have.dim}"
            )
    local_type[mesh_axis] = to_local_type(dst_c)
    new_spec = _maybe_spec(
        _update_axis_in_partition_spec(
            spec, mesh_axis, dst_c if isinstance(dst_c, Shard) else None, ndim
        )
    )
    ndim, new_spec = _rank_change(tr, ndim, new_spec)
    return _finish(_make(ndim, local_type, new_spec), out)


def all_reduce(
    x: torch.Tensor | NdimWithSpmdType,
    axis: DeviceMeshAxis,
    *,
    src: PerMeshAxisSpmdType = P,
    dst: PerMeshAxisSpmdType,
    out: torch.Tensor | None = None,
) -> torch.Tensor | NdimWithSpmdType:
    """Type of ``spmd.all_reduce(x, axis, src=, dst=)``: ``P|V -> R|I``.

    ``src`` must match the value: pass ``src=V`` to sum a Varying value (an
    explicit reinterpret); the default ``src=P`` requires a pending sum.
    """
    return _transition(
        x, "all_reduce", axis, src, dst, [("P|V", R), ("P|V", I)], out=out
    )


def all_gather(
    x: torch.Tensor | NdimWithSpmdType,
    axis: DeviceMeshAxis,
    *,
    src: PerMeshAxisSpmdType = V,
    dst: PerMeshAxisSpmdType,
    out: torch.Tensor | None = None,
) -> torch.Tensor | NdimWithSpmdType:
    """Type of ``spmd.all_gather(x, axis, src=, dst=)``: ``V|S(i) -> R|I``.

    With ``src=V`` the result gains a leading dim (stack semantics); with
    ``src=S(i)`` the rank is unchanged (concat semantics).
    """
    return _transition(
        x, "all_gather", axis, src, dst, [("V|S", R), ("V|S", I)], out=out
    )


def reduce_scatter(
    x: torch.Tensor | NdimWithSpmdType,
    axis: DeviceMeshAxis,
    *,
    src: PerMeshAxisSpmdType = P,
    dst: PerMeshAxisSpmdType = V,
    out: torch.Tensor | None = None,
) -> torch.Tensor | NdimWithSpmdType:
    """Type of ``spmd.reduce_scatter(x, axis, src=, dst=)``: ``P|V -> V|S(i)``.

    With ``dst=V`` dim 0 is unbound across ranks (rank decreases by one);
    with ``dst=S(i)`` the rank is unchanged.
    """
    return _transition(x, "reduce_scatter", axis, src, dst, [("P|V", "V|S")], out=out)


def all_to_all(
    x: torch.Tensor | NdimWithSpmdType,
    axis: DeviceMeshAxis,
    *,
    src: PerMeshAxisSpmdType = V,
    dst: PerMeshAxisSpmdType = V,
    out: torch.Tensor | None = None,
) -> torch.Tensor | NdimWithSpmdType:
    """Type of ``spmd.all_to_all(x, axis, src=, dst=)``: ``V|S(i) -> V|S(j)``."""
    return _transition(x, "all_to_all", axis, src, dst, [("V|S", "V|S")], out=out)


def reinterpret(
    x: torch.Tensor | NdimWithSpmdType,
    axis: DeviceMeshAxis,
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
    out: torch.Tensor | None = None,
) -> torch.Tensor | NdimWithSpmdType:
    """Type of ``spmd.reinterpret(x, axis, src=, dst=)``: no comms, local data
    unchanged, semantic value may change (e.g. ``V -> P``)."""
    return _transition(
        x,
        "reinterpret",
        axis,
        src,
        dst,
        [
            (R, R),
            (I, I),
            (V, V),
            (P, P),
            (V, P),
            (R, V),
            (R, P),
            (I, V),
            (I, P),
            (R, I),
            (I, R),
        ],
        out=out,
    )


def convert(
    x: torch.Tensor | NdimWithSpmdType,
    axis: DeviceMeshAxis,
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
    out: torch.Tensor | None = None,
) -> torch.Tensor | NdimWithSpmdType:
    """Type of ``spmd.convert(x, axis, src=, dst=)``: no comms, semantic value
    preserved (e.g. ``R -> S(0)`` shards a replicated value locally)."""
    return _transition(
        x,
        "convert",
        axis,
        src,
        dst,
        [
            (R, R),
            (I, I),
            (P, P),
            (R, I),
            (R, "V|S"),
            (R, P),
            (I, R),
            (I, "V|S"),
            (I, P),
            ("V|S", "V|S"),
            ("V|S", P),
        ],
        out=out,
    )


# =============================================================================
# Outputs
# =============================================================================


def output(out: Any, typed: Any) -> None:
    """Stamp the real output(s) with the type(s) of ``typed``.

    ``out`` and ``typed`` are a tensor and a tensor/``NdimWithSpmdType``, or matching
    tuples/lists of them.  Ranks must agree.  A ``None`` in ``out`` (an
    optional output) is skipped.
    """
    if isinstance(out, (tuple, list)):
        if not isinstance(typed, (tuple, list)) or len(typed) != len(out):
            raise SpmdTypeError(
                f"rules.output: {len(out)} output(s) but "
                f"{len(typed) if isinstance(typed, (tuple, list)) else 1} type(s)"
            )
        for o, t in zip(out, typed):
            output(o, t)
        return
    if out is None or (typed is None and not isinstance(out, torch.Tensor)):
        return  # optional or non-tensor output
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            f"rules.output: expected a tensor output, got {type(out).__name__}"
        )
    if not _is_tensor_or_ndim(typed):
        raise TypeError(
            f"rules.output: expected a tensor or NdimWithSpmdType, got {type(typed).__name__}"
        )
    _touch(typed)
    if out.ndim != typed.ndim:
        raise SpmdTypeError(
            f"rules.output: output has {out.ndim} dim(s) but its declared type has "
            f"{typed.ndim}"
        )
    local_type = dict(get_local_type(typed))
    if has_local_type(out) and dict(get_local_type(out)) != local_type:
        raise SpmdTypeError(
            f"rules.output: output already has type {get_local_type(out)}, but the "
            f"rule derives {local_type}"
        )
    derived_spec = get_partition_spec(typed)
    try:
        assert_type(out, local_type, derived_spec)
    except SpmdTypeError as e:
        raise SpmdTypeError(
            f"rules.output: output annotations conflict with the derived type: {e}"
        ) from e


# =============================================================================
# Running an spmd_typecheck hook
# =============================================================================


_SIGNATURES: "weakref.WeakKeyDictionary[Callable[..., object], inspect.Signature]" = (
    weakref.WeakKeyDictionary()
)


def _signature(fn: Callable[..., object]) -> inspect.Signature:
    """``inspect.signature`` memoized per function (it runs on every ``apply``)."""
    sig = _SIGNATURES.get(fn)
    if sig is None:
        sig = _SIGNATURES[fn] = inspect.signature(fn)
    return sig


def hook_shape(cls: type) -> tuple[bool, tuple[str, ...]]:
    """``(returns_types, names)`` for ``cls.spmd_typecheck``: whether it is the
    return form (keyword-only parameters), and the forward arguments it names."""
    params = list(_signature(cls.spmd_typecheck).parameters.values())
    returns_types = not params or params[0].kind is inspect.Parameter.KEYWORD_ONLY
    return returns_types, tuple(
        p.name for p in (params if returns_types else params[1:])
    )


def hook_kwargs(
    cls: type,
    forward_args: dict[str, object],
    names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """The keyword arguments for ``cls.spmd_typecheck``; a name that is not a
    ``forward`` parameter is a ``TypeError`` naming it.  ``names`` may be passed
    when ``hook_shape(cls)`` was already computed by the caller."""
    if names is None:
        _, names = hook_shape(cls)
    for name in names:
        if name not in forward_args:
            raise TypeError(
                f"{cls.__name__}.spmd_typecheck: {name!r} is not a parameter of "
                f"{cls.__name__}.forward (has {sorted(forward_args)})"
            )
    return {name: forward_args[name] for name in names}


def bind_forward_args(cls: type, args: tuple[object, ...]) -> dict[str, object]:
    """Bind ``cls.apply(*args)`` positionals to ``cls.forward`` parameter names.

    The ``ctx`` parameter (when ``forward`` takes one) is omitted.  A
    ``*args`` parameter binds to a tuple.
    """
    signature = _signature(cls.forward)
    has_ctx = cls.setup_context is torch.autograd.Function.setup_context
    bound = signature.bind(*((None, *args) if has_ctx else args))
    bound.apply_defaults()
    forward_args = dict(bound.arguments)
    if has_ctx:
        forward_args.pop(next(iter(signature.parameters)))
    return forward_args


def _iter_tensors(value: object) -> list[torch.Tensor]:
    """The tensors inside ``value`` (a tensor, a tuple/list/dict of them, or nothing)."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(
        value, (int, float, bool, complex, str, type(None), NdimWithSpmdType)
    ):
        return []
    flat, _ = tree_flatten(value)
    return [t for t in flat if isinstance(t, torch.Tensor)]


def run_typecheck(cls: type, func: Callable[..., object], args: tuple[object, ...]):
    """Run ``func(*args)`` (the real ``apply``) and then ``cls.spmd_typecheck``.

    The hook names the forward arguments it needs as keyword-only parameters.
    If it also takes a leading positional parameter it receives the outputs
    there and stamps them itself (``rules.output`` or ``out=``); otherwise it
    returns the output type(s) and the runner stamps them.  Every hook is
    coverage checked: each tensor argument must reach a ``rules`` function,
    ``assert_type``, or ``rules.ignore``, and every tensor output must be typed
    afterwards.
    """
    who = f"{cls.__name__}.spmd_typecheck"
    forward_args = bind_forward_args(cls, args)
    hook = cls.spmd_typecheck
    # Two shapes: ``spmd_typecheck(outputs, *, x, ...)`` stamps the outputs
    # itself; ``spmd_typecheck(*, x, ...)`` returns the type(s) and the runner
    # stamps them, so the hook reads like a type-level forward.
    returns_types, names = hook_shape(cls)
    kwargs = hook_kwargs(cls, forward_args, names)

    outputs = func(*args)
    out_tensors = _iter_tensors(outputs)
    with _coverage() as cov:
        if returns_types:
            result = hook(**kwargs)
            if result is None and out_tensors:
                raise SpmdTypeError(
                    f"{who}: returned None; a keyword-only spmd_typecheck must "
                    f"return the output type(s) (a NdimWithSpmdType, a tensor, or a tuple)"
                )
            output(outputs, result)
        else:
            hook(outputs, **kwargs)

    for name, value in forward_args.items():
        missed = [t for t in _iter_tensors(value) if id(t) not in cov.touched]
        if missed:
            raise SpmdTypeError(
                f"{who}: tensor argument {name!r} was not accounted for. Pass "
                f"it to the rules operation that consumes it, assert_type it, or "
                f"declare rules.ignore({name}) if it does not affect the "
                f"forward value."
            )
    for i, t in enumerate(out_tensors):
        if not has_local_type(t):
            raise SpmdTypeError(
                f"{who}: output {i} was left untyped; return its type from the "
                f"hook, or finish the rule with rules.output(out, "
                f"<NdimWithSpmdType>) or assert_type(out, ...)"
            )

    return outputs


__all__ = [
    "NdimWithSpmdType",
    "all_gather",
    "all_reduce",
    "all_to_all",
    "bind_forward_args",
    "convert",
    "einsum",
    "hook_kwargs",
    "hook_shape",
    "ignore",
    "output",
    "reduce_scatter",
    "reinterpret",
    "run_typecheck",
]
