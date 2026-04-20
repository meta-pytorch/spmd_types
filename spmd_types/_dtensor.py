# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DTensor bridge utilities for spmd_types.

Converts between spmd_types per-axis types and DTensor placements, and
provides a ``spmd_redistribute`` helper that redistributes a DTensor using
spmd_types collectives while keeping DTensor metadata in sync.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest

import torch
from spmd_types._collectives import redistribute
from spmd_types.types import (
    I,
    normalize_dim_sharding,
    P,
    PerMeshAxisSpmdType,
    R,
    S,
    Shard,
    TensorSharding,
    V,
)
from torch.distributed.tensor import (
    DTensor,
    Partial as DtPartial,
    Placement as DtPlacement,
    Replicate as DtReplicate,
    Shard as DtShard,
)


def dtensor_placement_to_spmd_type(
    placement: DtPlacement,
    grad_placement: DtPlacement | None = None,
) -> PerMeshAxisSpmdType:
    """Converts a DTensor placement to an spmd_types per-axis type.

    This is the reverse of ``spmd_type_to_dtensor_placement``.

    For Replicate placements, the forward type is ambiguous between R and I.
    The ``grad_placement`` parameter disambiguates:

    - ``Partial`` gradient -> R (backward does all-reduce)
    - ``Replicate`` gradient (or None, the default) -> I (backward is no-op)

    Args:
        placement: A DTensor placement (Replicate, Shard, or Partial).
        grad_placement: The gradient placement for this axis, used to
            disambiguate Replicate (R vs I).  When None, defaults to I
            (matching DTensor's default of keeping Replicate gradients
            as Replicate).

    Returns:
        The corresponding spmd_types per-axis type.

    Raises:
        ValueError: If the placement type is unknown.
    """
    if isinstance(placement, DtReplicate):
        # Disambiguate R vs I via the gradient placement.
        # DTensor default: grad of Replicate stays Replicate (= I semantics).
        # Explicit Partial grad: backward does all-reduce (= R semantics).
        if isinstance(grad_placement, DtPartial):
            return R
        return I
    elif isinstance(placement, DtShard):
        return S(placement.dim)
    elif isinstance(placement, DtPartial):
        return P
    raise ValueError(f"Unknown DTensor placement: {placement}")


def spmd_type_to_dtensor_placement(
    spmd_type: PerMeshAxisSpmdType,
) -> DtPlacement:
    """Converts an spmd_types per-axis type to a DTensor placement.

    Args:
        spmd_type: An spmd type (R, I, V, P, or S(i)).

    Returns:
        The corresponding DTensor placement.

    Raises:
        ValueError: If the spmd type is Varying without a Shard dim.
    """
    if spmd_type is R or spmd_type is I:
        return DtReplicate()
    elif isinstance(spmd_type, Shard):
        return DtShard(spmd_type.dim)
    elif spmd_type is P:
        return DtPartial()
    elif spmd_type is V:
        raise ValueError(
            "Cannot convert plain Varying to a DTensor placement. "
            "Use S(dim) to specify which tensor dimension is sharded."
        )
    raise ValueError(f"Unknown spmd type: {spmd_type}")


def _apply_transitions(
    local: torch.Tensor,
    mesh,
    mesh_dim_names: list[str],
    transitions: dict[str, tuple],
) -> torch.Tensor:
    """Apply spmd_types.redistribute for each axis transition."""
    for dim_name, (dim_src, dim_dst) in transitions.items():
        dim_idx = mesh_dim_names.index(dim_name)
        pg = mesh.get_group(dim_idx)
        local = redistribute(local, pg, src=dim_src, dst=dim_dst)
    return local


class _SpmdRedistribute(torch.autograd.Function):
    """Autograd function wrapping spmd_types.redistribute for DTensor bridge.

    Handles the DTensor <-> local tensor conversion at the boundary:
    forward extracts the local tensor via ``_local_tensor``, applies the
    spmd_types collectives, and the caller wraps the result back into a
    DTensor.  Backward receives a DTensor gradient (from
    ``DTensor.from_local``'s backward), extracts its local tensor, and
    applies the inverse transitions (dst->src, reversed order).

    This custom autograd.Function is necessary because ``_local_tensor``
    is an attribute access with no autograd node -- without it, gradients
    cannot flow from the output DTensor back to the input DTensor.
    """

    @staticmethod
    def forward(ctx, x, mesh, mesh_dim_names, transitions):
        ctx.mesh = mesh
        ctx.mesh_dim_names = mesh_dim_names
        ctx.src_placements = x.placements if isinstance(x, DTensor) else None
        # Reverse transitions for backward: swap src/dst, reverse order.
        ctx.bwd_transitions = dict(
            reversed([(name, (dst, src)) for name, (src, dst) in transitions.items()])
        )
        local = x._local_tensor if isinstance(x, DTensor) else x
        return _apply_transitions(local, mesh, mesh_dim_names, transitions)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output may be a DTensor (from _FromTorchTensor backward);
        # extract the local tensor so spmd_types operates on plain tensors.
        if isinstance(grad_output, DTensor):
            grad_output = grad_output._local_tensor
        grad_input = _apply_transitions(
            grad_output, ctx.mesh, ctx.mesh_dim_names, ctx.bwd_transitions
        )
        # Return a DTensor if the forward input was a DTensor, so the gradient
        # matches the input type for _FromTorchTensor's backward chain.
        if ctx.src_placements is not None:
            grad_input = DTensor.from_local(
                grad_input,
                ctx.mesh,
                ctx.src_placements,
                run_check=False,
            )
        return grad_input, None, None, None


def spmd_redistribute(
    x: DTensor,
    src: TensorSharding,
    dst: TensorSharding,
) -> DTensor:
    """Redistribute a DTensor using spmd_types collectives.

    This is a bridge function for incrementally migrating from DTensor's
    ``redistribute()`` to ``spmd_types.redistribute()``.  It accepts
    ``TensorSharding`` (the same type used in ``ModuleShardings``) and
    derives per-axis transitions automatically::

        spmd_redistribute(x, TensorSharding("tp"), TensorSharding())
        # equivalent to all_gather(S(0)->R) on tp

        spmd_redistribute(
            x,
            TensorSharding("fsdp", "tp"),
            TensorSharding(),
        )
        # gathers on both fsdp and tp

    Multi-axis sharding on a single tensor dim (e.g.,
    ``TensorSharding(("dp", "fsdp"), ...)`` where dim 0 is sharded across both
    ``dp`` and ``fsdp``) is allowed as long as it does not change between
    ``src`` and ``dst``.  If it does change, the per-axis diff algorithm
    needs to determine the correct sequence of collectives; for now, a
    ``NotImplementedError`` is raised.

    Note: TensorSharding cannot express Partial or Invariant placements.
    Dims not mentioned in the sharding are treated as Replicate. For
    transitions involving Partial (e.g., after a sharded matmul), use
    ``spmd_types.all_reduce`` or ``spmd_types.reduce_scatter`` directly.

    The backward is the symmetric inverse: each axis transition is reversed
    (dst->src) in reverse order.  Since TensorSharding only expresses
    Shard and Replicate, the inverse always recovers the original placements
    (e.g., all_gather(S->R) backward is redistribute(R->S)).

    Args:
        x: The input DTensor.
        src: The current sharding (which tensor dims map to which mesh axes).
        dst: The desired sharding.

    Returns:
        A DTensor with updated placements matching ``dst``.

    Raises:
        NotImplementedError: If any tensor dim is sharded across multiple mesh
        axes and that sharding differs between ``src`` and ``dst``.
    """
    mesh = x.device_mesh
    mesh_dim_names = list(mesh.mesh_dim_names)

    # Build per-dim src/dst spmd types directly from TensorSharding.
    # Trailing dims beyond len(sharding) are implicitly replicated (None -> ()).
    # Dims not mentioned default to Replicate.
    src_per_dim: dict[str, PerMeshAxisSpmdType] = defaultdict(lambda: R)
    dst_per_dim: dict[str, PerMeshAxisSpmdType] = defaultdict(lambda: R)
    for tensor_dim, (s, d) in enumerate(zip_longest(src, dst)):
        src_dims = normalize_dim_sharding(s)
        dst_dims = normalize_dim_sharding(d)
        if src_dims != dst_dims and (len(src_dims) > 1 or len(dst_dims) > 1):
            raise NotImplementedError(
                f"Multi-axis sharding on a single tensor dim is not yet supported "
                f"by spmd_redistribute. Tensor dim {tensor_dim} changes from "
                f"{src_dims!r} to {dst_dims!r}."
            )
        for dim in src_dims:
            src_per_dim[dim] = S(tensor_dim)
        for dim in dst_dims:
            dst_per_dim[dim] = S(tensor_dim)

    # Only include dims where the type actually changes.
    transitions = {
        name: (src_per_dim[name], dst_per_dim[name])
        for name in mesh_dim_names
        if src_per_dim[name] != dst_per_dim[name]
    }

    if not transitions:
        return x

    local = _SpmdRedistribute.apply(x, mesh, mesh_dim_names, transitions)
    new_placements = list(x.placements)

    for dim_name, (_, dim_dst) in transitions.items():
        dim_idx = mesh_dim_names.index(dim_name)
        new_placements[dim_idx] = spmd_type_to_dtensor_placement(dim_dst)

    return DTensor.from_local(local, mesh, tuple(new_placements), run_check=False)
