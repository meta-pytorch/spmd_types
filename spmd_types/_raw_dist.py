# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Type rules for raw torch.distributed collectives.

When users call torch.distributed collectives directly (instead of the
spmd_types wrappers), we still type-check the operands.  Raw collectives
are not differentiable, so we only propagate R/I/V/P without autograd
concerns.  The output tensor is passed as an argument (not returned), so
the user must pre-annotate it with the desired R or I type.

Why enforce typing here even though these collectives have no backward?
We cannot reuse the out= kwarg handling from D94912643 because raw dist
collectives don't have out=.  Even if we let users assert_type on the
output directly, we still need something to verify the output type is
consistent with the input type -- e.g., all_gather(V, R, out=R) is
obviously wrong, but the standard rules (V and R can mix freely) won't
complain about it.

For in-place collectives like all_reduce(x), the type checker will
in-place modify the spmd type on the input tensor (e.g., P -> R).
mutate is reserved for views only.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as _torch_dist
from spmd_types import _state
from spmd_types._type_attr import get_local_type
from spmd_types.types import (
    format_axis,
    I,
    normalize_axis,
    P,
    PerMeshAxisLocalSpmdType,
    R,
    SpmdTypeError,
    V,
)
from torch.fx.operator_schemas import normalize_function


def _check_raw_collective(
    func_name: str,
    input_tensor: torch.Tensor,
    output_tensors: torch.Tensor | list[torch.Tensor] | None,
    group: object,
    src: PerMeshAxisLocalSpmdType,
    dst: tuple[PerMeshAxisLocalSpmdType, ...],
    *,
    mutate_src_to: PerMeshAxisLocalSpmdType | None = None,
) -> None:
    """Shared type-check logic for raw dist collectives.

    Validates that ``input_tensor`` has type ``src`` on the group axis and
    that every output tensor is pre-annotated with one of the ``dst`` types.

    For in-place collectives (e.g. ``all_reduce``), pass ``mutate_src_to``
    to change the input tensor's type after validation.  This models a
    read-write operand that is both checked as input and mutated as output.

    In strict mode, a typed tensor missing the group axis raises
    ``SpmdTypeError``.  In permissive mode, missing axes skip src/dst
    validation but ``mutate_src_to`` is still applied (we know the
    post-collective type even if we didn't validate the pre-collective type).

    Args:
        func_name: Name of the collective (for error messages).
        input_tensor: The input tensor operand.
        output_tensors: A single output tensor or a list of output tensors
            (e.g. for list-based ``all_gather``).  Each must be pre-annotated
            by the user.  Pass None for in-place collectives like all_reduce.
        group: The ProcessGroup used as the axis key.
        src: Expected input type on the group axis.
        dst: Allowed output types on the group axis.
        mutate_src_to: If set, mutate the input tensor's type to this value
            after validation.  Used for in-place collectives (e.g. P -> R
            for all_reduce).
    """
    if group is None:
        group = _torch_dist.distributed_c10d._get_default_group()

    # Normalize group to MeshAxis for dict lookups (types are stored under
    # normalized keys).
    group = normalize_axis(group)

    if group.size() == 1:
        return  # singleton axes are not tracked

    local_type = get_local_type(input_tensor)
    if group in local_type:
        input_type = local_type[group]
        if input_type is not src:
            raise SpmdTypeError(
                f"{func_name}: expected input type {src} on axis "
                f"{format_axis(group)}, got {input_type}"
            )
    elif _state.is_strict():
        raise SpmdTypeError(
            f"{func_name}: tensor has no type for axis "
            f"{format_axis(group)}. Use assert_type() to annotate "
            f"the tensor on this axis before calling collectives."
        )
    # Always apply mutate_src_to (even if axis was previously missing).
    if mutate_src_to is not None:
        local_type[group] = mutate_src_to

    if output_tensors is not None:
        if isinstance(output_tensors, list):
            for i, out_tensor in enumerate(output_tensors):
                _check_output_tensor(func_name, out_tensor, group, dst, index=i)
        else:
            _check_output_tensor(func_name, output_tensors, group, dst)


def _check_output_tensor(
    func_name: str,
    tensor: torch.Tensor,
    group: object,
    dst: tuple[PerMeshAxisLocalSpmdType, ...],
    index: int | None = None,
) -> None:
    """Check a single output tensor's SPMD type."""
    # Normalize group to MeshAxis for dict lookups.
    group = normalize_axis(group)
    if group.size() == 1:
        return  # singleton axes are not tracked
    label = f"output tensor_list[{index}]" if index is not None else "output tensor"
    local_type = get_local_type(tensor)
    if group in local_type:
        output_type = local_type[group]
        if output_type not in dst:
            raise SpmdTypeError(
                f"{func_name}: expected {label} type "
                f"{'/'.join(str(t) for t in dst)} on axis "
                f"{format_axis(group)}, got {output_type}"
            )
    elif _state.is_strict():
        raise SpmdTypeError(
            f"{func_name}: {label} has no type for axis "
            f"{format_axis(group)}. Use assert_type() to annotate "
            f"the tensor on this axis before calling collectives."
        )


def _make_rule(input_arg, output_arg, src, dst, *, mutate_src_to=None):
    """Create a raw dist rule from a declarative spec.

    Args:
        input_arg: Name of the input tensor parameter.
        output_arg: Name of the output tensor/list parameter, or None
            for in-place collectives.
        src: Expected input SPMD type.
        dst: Allowed output SPMD types.
        mutate_src_to: If set, mutate input type after validation
            (for in-place collectives like all_reduce).
    """

    def rule(func, args, kwargs):
        ba = normalize_function(
            func, args, kwargs, normalize_to_only_use_kwargs=True
        ).kwargs
        output = ba[output_arg] if output_arg is not None else None
        _check_raw_collective(
            func.__name__,
            ba[input_arg],
            output,
            ba["group"],
            src,
            dst,
            mutate_src_to=mutate_src_to,
        )

    return rule


def _not_yet_implemented(func, args, kwargs):
    raise SpmdTypeError(
        f"spmd_types: type checking for raw torch.distributed.{func.__name__} "
        f"is not yet implemented; use the spmd_types wrapper instead"
    )


# Functions that have __torch_function__ support in PyTorch and will be
# intercepted by SpmdTypeMode:
#   all_gather, all_gather_into_tensor, all_reduce,
#   all_to_all_single, reduce_scatter_tensor
#
# Functions that do NOT have __torch_function__ support yet (entries here
# are dormant until PyTorch adds support or we add manual patching):
#   all_gather_coalesced, all_reduce_coalesced, all_to_all,
#   broadcast, gather, reduce, reduce_scatter, scatter,
#   send, recv, isend, irecv

RAW_DIST_RULES: dict[Callable, Callable] = {
    # -- Implemented (have __torch_function__ support) --
    _torch_dist.all_gather: _make_rule("tensor", "tensor_list", V, (R, I)),
    _torch_dist.all_gather_into_tensor: _make_rule(
        "input_tensor", "output_tensor", V, (R, I)
    ),
    # R vs I doesn't directly matter here because all_reduce isn't
    # differentiable, but we choose R because the result is more likely to
    # participate in further computation (R semantics) than to be treated as
    # a fixed constant (I semantics).
    _torch_dist.all_reduce: _make_rule("tensor", None, P, (), mutate_src_to=R),
    _torch_dist.all_to_all_single: _make_rule("input", "output", V, (V,)),
    _torch_dist.reduce_scatter_tensor: _make_rule("input", "output", P, (V,)),
    # -- Stubs (no __torch_function__ support yet, dormant) --
    _torch_dist.all_gather_coalesced: _not_yet_implemented,
    _torch_dist.all_reduce_coalesced: _not_yet_implemented,
    _torch_dist.all_to_all: _not_yet_implemented,
    _torch_dist.broadcast: _not_yet_implemented,
    _torch_dist.gather: _not_yet_implemented,
    _torch_dist.reduce: _not_yet_implemented,
    _torch_dist.reduce_scatter: _not_yet_implemented,
    _torch_dist.scatter: _not_yet_implemented,
    _torch_dist.send: _not_yet_implemented,
    _torch_dist.recv: _not_yet_implemented,
    _torch_dist.isend: _not_yet_implemented,
    _torch_dist.irecv: _not_yet_implemented,
}
