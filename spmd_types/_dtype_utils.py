# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dtype conversion utilities for SPMD collective operations."""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass(frozen=True, slots=True)
class _DtypeOptions:
    """Processed dtype options for an autograd Function.

    Constructed by ``_process_dtype_options`` and passed as a single argument
    to each autograd Function's ``forward``.  The Function applies
    ``op_dtype`` / ``out_dtype`` inside forward (so the conversion is part of
    the same autograd node) and saves ``backward_options`` for the backward
    pass.

    All fields are always concrete (never None).  When no conversion is
    needed, ``op_dtype`` and ``out_dtype`` equal the input tensor's dtype.
    Do not construct directly; use ``_process_dtype_options``.
    """

    op_dtype: torch.dtype
    out_dtype: torch.dtype
    backward_options: dict


def _resolve_dtypes(
    op_dtype: torch.dtype | None,
    out_dtype: torch.dtype | None,
    reducing: bool,
    input_dtype: torch.dtype,
) -> tuple[torch.dtype, torch.dtype]:
    """Resolve op_dtype and out_dtype based on inference rules.

    When only ``op_dtype`` is specified, ``out_dtype`` defaults to ``op_dtype``.

    When only ``out_dtype`` is specified, ``op_dtype`` is inferred to minimize
    bandwidth:

    - Reducing ops: ``op_dtype`` stays as ``input_dtype`` (preserve precision).
    - Non-reducing downcast (``out_dtype`` smaller than ``input_dtype``):
      ``op_dtype`` defaults to ``out_dtype`` (communicate at lower precision).
    - Non-reducing upcast (``out_dtype`` larger than ``input_dtype``):
      ``op_dtype`` stays as ``input_dtype`` (communicate at lower precision,
      upcast after).

    When neither is specified, both default to ``input_dtype`` (no conversion).

    Args:
        op_dtype: Explicit operation dtype (cast before op), or None.
        out_dtype: Explicit output dtype (cast after op), or None.
        reducing: If True, the operation involves a reduction or makes the
            tensor smaller, so lowering op_dtype would lose precision.
        input_dtype: The dtype of the input tensor.
    """
    if op_dtype is None and out_dtype is None:
        return input_dtype, input_dtype
    if op_dtype is not None and out_dtype is None:
        return op_dtype, op_dtype
    if op_dtype is None:
        if reducing:
            return input_dtype, out_dtype
        smaller = min(input_dtype, out_dtype, key=lambda d: torch.finfo(d).bits)
        return smaller, out_dtype
    return op_dtype, out_dtype


def _check_backward_options(
    op_dtype: torch.dtype | None,
    out_dtype: torch.dtype | None,
    backward_options: dict | None,
    backward_has_reduction: bool,
    requires_grad: bool,
    input_dtype: torch.dtype | None = None,
) -> None:
    """Validate backward_options when using reduced precision.

    When the backward involves a reduction and the operation will run in
    reduced precision, the user must explicitly specify
    ``backward_options={"op_dtype": ...}`` to control the precision of
    the gradient reduction.

    Reduced precision is detected from explicit ``op_dtype``/``out_dtype``
    arguments OR from ``input_dtype`` (since the resolved operation dtype
    defaults to ``input_dtype`` when nothing is specified).

    Raises:
        ValueError: If reduced precision is used with a reducing backward
            and backward_options doesn't specify op_dtype.
    """
    if not backward_has_reduction or not requires_grad or not torch.is_grad_enabled():
        return
    if backward_options is not None and "op_dtype" in backward_options:
        return
    has_reduced_precision = (
        (
            op_dtype is not None
            and torch.finfo(op_dtype).bits < torch.finfo(torch.float32).bits
        )
        or (
            out_dtype is not None
            and torch.finfo(out_dtype).bits < torch.finfo(torch.float32).bits
        )
        or (
            input_dtype is not None
            and torch.finfo(input_dtype).bits < torch.finfo(torch.float32).bits
        )
    )
    if not has_reduced_precision:
        return
    raise ValueError(
        "When using reduced precision with an operation whose backward "
        "involves a reduction (all_reduce or reduce_scatter), you must "
        'explicitly specify backward_options={"op_dtype": ...} to control '
        "the precision of the gradient reduction. For example, FSDP "
        "typically uses bf16 all_gather in forward with fp32 reduce_scatter "
        "in backward: all_gather(..., op_dtype=torch.bfloat16, "
        'backward_options={"op_dtype": torch.float32}). This also applies '
        "when the input tensor is already reduced precision (e.g. bf16)."
    )


_VALID_BACKWARD_OPTIONS_KEYS = frozenset({"op_dtype", "out_dtype", "backward_options"})


def _validate_backward_options(backward_options: dict | None) -> dict:
    """Validate and normalize backward_options.

    Checks that all keys are recognized and returns a (possibly empty) dict
    that is safe to ``**``-splat into a collective call.

    Raises:
        ValueError: If backward_options contains unrecognized keys.
    """
    if backward_options is None:
        return {}
    unknown = set(backward_options) - _VALID_BACKWARD_OPTIONS_KEYS
    if unknown:
        raise ValueError(
            f"Unknown keys in backward_options: {unknown}. "
            f"Valid keys are: {sorted(_VALID_BACKWARD_OPTIONS_KEYS)}"
        )
    return backward_options


def _process_dtype_options(
    op_dtype: torch.dtype | None,
    out_dtype: torch.dtype | None,
    backward_options: dict | None,
    *,
    reducing: bool,
    backward_has_reduction: bool,
    input_dtype: torch.dtype,
    requires_grad: bool,
    inplace: bool = False,
) -> _DtypeOptions:
    """Validate, resolve, and normalize all dtype options in one pass.

    Checks backward_options against the raw user-specified dtypes (before
    inference), resolves missing dtypes against ``input_dtype``, and validates
    backward_options keys.  Returns a ``_DtypeOptions`` with all fields
    concrete.
    """
    if inplace and (op_dtype is not None or out_dtype is not None):
        raise ValueError(
            "inplace=True does not support op_dtype or out_dtype, "
            "because the operation cannot be performed in-place when a dtype "
            "conversion is required. Use inplace=False instead."
        )
    _check_backward_options(
        op_dtype,
        out_dtype,
        backward_options,
        backward_has_reduction,
        requires_grad,
        input_dtype,
    )
    op_dtype, out_dtype = _resolve_dtypes(op_dtype, out_dtype, reducing, input_dtype)
    backward_options = _validate_backward_options(backward_options)
    if "out_dtype" not in backward_options:
        backward_options = {**backward_options, "out_dtype": input_dtype}
    return _DtypeOptions(
        op_dtype=op_dtype,
        out_dtype=out_dtype,
        backward_options=backward_options,
    )


def _split_composition_dtype_options(
    dtype_options: _DtypeOptions,
) -> tuple[_DtypeOptions, _DtypeOptions]:
    """Split dtype options for a two-step forward composition.

    For a composition ``second(first(x))``, the first op applies ``op_dtype``
    on the forward input and the second op applies ``out_dtype`` on the forward
    output.  In backward (reverse execution order) the second op's backward
    applies ``backward op_dtype`` first, then the first op's backward applies
    ``backward out_dtype`` to produce the final gradient.

    Returns ``(first_options, second_options)``.
    """
    bw = dtype_options.backward_options
    bw_op = bw.get("op_dtype")
    bw_out = bw.get("out_dtype")
    if "backward_options" in bw:
        raise NotImplementedError(
            "backward_options with nested backward_options (double backward) "
            "is not supported for composed operations (reinterpret I->V, I->P)"
        )
    first = _DtypeOptions(
        op_dtype=dtype_options.op_dtype,
        out_dtype=dtype_options.op_dtype,
        backward_options={"op_dtype": bw_op, "out_dtype": bw_out},
    )
    second = _DtypeOptions(
        op_dtype=dtype_options.op_dtype,
        out_dtype=dtype_options.out_dtype,
        backward_options={"op_dtype": bw_op, "out_dtype": bw_op},
    )
    return first, second


def _apply_op_dtype(x: torch.Tensor, op_dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to op_dtype if different from current dtype."""
    if x.dtype != op_dtype:
        return x.to(op_dtype)
    return x


def _apply_out_dtype(x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to out_dtype if different from current dtype."""
    if x.dtype != out_dtype:
        return x.to(out_dtype)
    return x
