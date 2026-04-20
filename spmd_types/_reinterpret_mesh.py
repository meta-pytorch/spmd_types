# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cross-mesh reinterpretation for SPMD types.

Split into its own module so that changes to ``_checker.py`` (the type
inference engine) do not invalidate downstream targets that only need
``reinterpret_mesh``.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from typing import Callable, NamedTuple

import torch
from spmd_types import _dist
from spmd_types._mesh_axis import MeshAxis
from spmd_types._state import current_mesh
from spmd_types._traceback import api_boundary
from spmd_types._type_attr import get_local_type, set_local_type as _set_local_type_raw
from spmd_types.runtime import (
    _TRACE,
    _trace_op,
    _validate,
    get_partition_spec,
    has_local_type,
)
from spmd_types.types import (
    DeviceMeshAxis,
    format_axis,
    normalize_axis,
    PerMeshAxisSpmdTypes,
    SpmdTypeError,
)
from torch.distributed.device_mesh import DeviceMesh

# =============================================================================
# Error formatting helpers (moved from _checker.py)
# =============================================================================


def _format_axis_set(axes: Iterable[DeviceMeshAxis]) -> str:
    """Format a set of mesh axes for display (stride-descending, comma-separated, in braces)."""
    normalized = [(normalize_axis(ax), format_axis(ax)) for ax in axes]
    # Sort by max stride descending (outermost first), then by name for ties.
    normalized.sort(
        key=lambda item: (-max(d for _, d in item[0].layout.sizes_and_strides), item[1])
    )
    return "{" + ", ".join(name for _, name in normalized) + "}"


def _format_tensor_for_context(t: torch.Tensor) -> str:
    """Format a single tensor for error context display."""
    from torch.utils._dtype_abbrs import dtype_abbrs

    lt = get_local_type(t)
    dtype_str = dtype_abbrs.get(t.dtype, str(t.dtype).removeprefix("torch."))
    shape_str = ", ".join(str(d) for d in t.shape)
    type_items = ", ".join(f"{format_axis(axis)}: {typ!r}" for axis, typ in lt.items())
    result = f"{dtype_str}[{shape_str}] {{{type_items}}}"
    ps = get_partition_spec(t)
    if ps is not None:
        result += f" {ps}"
    return result


def _format_non_tensor_for_context(val: object) -> str:
    """Format a non-tensor argument for error context display.

    Special-cases ProcessGroup to show the mesh axis name instead of the
    raw repr.
    """
    if isinstance(val, _dist.dist.ProcessGroup):
        return repr(normalize_axis(val))
    return repr(val)


def _format_arg_for_context(val: object) -> str:
    """Format a single argument value for error context display.

    Handles tensors, lists of tensors, ProcessGroups, and other values.
    """
    if isinstance(val, torch.Tensor):
        return _format_tensor_for_context(val)
    if isinstance(val, (list, tuple)):
        has_tensors = any(isinstance(item, torch.Tensor) for item in val)
        if has_tensors:
            bracket = "[" if isinstance(val, list) else "("
            close = "]" if isinstance(val, list) else ")"
            items = []
            for item in val:
                if isinstance(item, torch.Tensor):
                    items.append(_format_tensor_for_context(item))
                else:
                    items.append(_format_non_tensor_for_context(item))
            # If all items are identical, abbreviate: [item] * N
            if len(items) > 1 and all(it == items[0] for it in items):
                return f"{bracket}{items[0]}{close} * {len(items)}"
            # Multi-line format; items on separate lines.
            inner = ",\n".join(items)
            return bracket + "\n" + inner + ",\n" + close
    return _format_non_tensor_for_context(val)


def _get_param_names(func: Callable) -> list[str] | None:
    """Try to get positional parameter names from a function signature.

    Returns None if the signature cannot be inspected (e.g., C builtins).
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return None
    return [
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]


class _RawArgEntry(NamedTuple):
    """One argument (positional or keyword) captured for error display."""

    location: str | None
    value: object


def _format_operator_context(
    func: Callable,
    raw_entries: list[_RawArgEntry],
    mesh: frozenset[MeshAxis] | None = None,
) -> str:
    """Format an operator context string for error messages.

    Produces a multi-line block showing all arguments (tensor and
    non-tensor) like::

        In all_gather_into_tensor(
          output_tensor: f32[64] {},
          input_tensor: f32[32] {},
          group: DP,
        )

    Positional args use parameter names when available from the function
    signature, otherwise they fall back to ``args[i]``. Keyword args use
    their key name directly. ProcessGroup arguments are shown by their
    mesh axis name. Lists of tensors are shown with each element
    formatted inline.
    """
    #   4 spaces = arg indent (items in the function call)
    #   6 spaces = nested indent (items inside a list/tuple arg)
    _ARG_INDENT = "    "
    _NESTED_INDENT = "      "

    mesh_suffix = ""
    if mesh is not None:
        mesh_suffix = f" under mesh {_format_axis_set(mesh)}"

    op_name = getattr(func, "__name__", repr(func))
    if not raw_entries:
        return f"  In {op_name}(){mesh_suffix}"
    param_names = _get_param_names(func)
    parts = []
    positional_index = 0
    for entry in raw_entries:
        formatted = _format_arg_for_context(entry.value)
        if entry.location is not None:
            raw = f"{entry.location}: {formatted}"
        else:
            if param_names is not None and positional_index < len(param_names):
                label = param_names[positional_index]
            else:
                label = f"args[{positional_index}]"
            positional_index += 1
            raw = f"{label}: {formatted}"
        if "\n" not in raw:
            parts.append(f"{_ARG_INDENT}{raw}")
        else:
            # Multi-line value (e.g., list of tensors).  First line stays
            # at arg indent, middle lines at nested indent, last line
            # (closing bracket) back at arg indent.
            lines = raw.split("\n")
            result = f"{_ARG_INDENT}{lines[0]}"
            for line in lines[1:-1]:
                result += f"\n{_NESTED_INDENT}{line}"
            result += f"\n{_ARG_INDENT}{lines[-1]}"
            parts.append(result)
    joined = ",\n".join(parts)
    return f"  In {op_name}(\n{joined},\n  ){mesh_suffix}"


# =============================================================================
# reinterpret_mesh
# =============================================================================


@api_boundary
def reinterpret_mesh(
    tensor: torch.Tensor,
    type: PerMeshAxisSpmdTypes | frozenset[MeshAxis] | DeviceMesh | Sequence,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Explicitly reinterpret a tensor onto a cross-mesh-compatible local type.

    This is a no-op on the underlying tensor data. It only retags the tensor
    with a new local SPMD type after verifying the source and destination mesh
    presentations are compatible under one-hop grouping.

    When *type* specifies axes only (frozenset, DeviceMesh, or Sequence of
    ProcessGroups), destination types are derived from the source tensor.
    """
    if not has_local_type(tensor):
        raise SpmdTypeError(
            "reinterpret_mesh requires a tensor with an existing local SPMD type."
        )
    if get_partition_spec(tensor) is not None:
        raise SpmdTypeError(
            "reinterpret_mesh does not support tensors carrying PartitionSpec. "
            "Cross-mesh reinterpretation is currently local-SPMD-only."
        )

    from spmd_types._mesh import _resolve_axes
    from spmd_types._mesh_region import check_reinterpret_mesh_compatible

    if isinstance(type, dict):
        dst = _validate(type)
    else:
        dst = _resolve_axes(type)

    src_type = get_local_type(tensor)
    compat = check_reinterpret_mesh_compatible(src_type, dst)
    if isinstance(compat, str):
        context = _format_operator_context(
            reinterpret_mesh,
            [_RawArgEntry(None, tensor), _RawArgEntry(None, dst)],
            mesh=current_mesh(),
        )
        raise SpmdTypeError(compat, context=context)

    result = tensor if inplace else torch.ops.aten.alias.default(tensor)
    _set_local_type_raw(result, compat)
    if _TRACE:
        _trace_op(reinterpret_mesh, [src_type], compat)
    return result
