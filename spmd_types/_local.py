# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local SPMD type coercion operations: reinterpret and convert."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from spmd_types import _dist
from spmd_types._dtype_utils import (
    _apply_op_dtype,
    _apply_out_dtype,
    _process_dtype_options,
    _split_composition_dtype_options,
)
from spmd_types._traceback import api_boundary
from spmd_types.types import _canonicalize_shard, I, P, PerMeshAxisSpmdType, R, Shard, V

try:
    from torch.distributed._local_tensor import local_tensor_mode, LocalTensor
    from torch.distributed._local_tensor._c10d import _prepare_collective_groups

    _HAS_LOCAL_TENSOR = True
except ImportError:
    _HAS_LOCAL_TENSOR = False

from torch.overrides import handle_torch_function, has_torch_function_unary

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def _get_local_tensor_mode(x: torch.Tensor):
    """Return the active LocalTensorMode if x is a LocalTensor, else None."""
    if not _HAS_LOCAL_TENSOR:
        return None
    mode = local_tensor_mode()
    if mode is not None and isinstance(x, LocalTensor):
        return mode
    return None


def _build_rank_to_group_local(pg) -> dict[int, int]:
    """Build a mapping from global rank to group-local rank.

    In LocalTensor mode, tensor_map iterates over all global ranks, but
    operations like chunk/select need the rank's position within its process
    group (group-local rank). This uses the complement computation from
    PyTorch's _c10d.py to handle both contiguous and strided process groups.

    For example, with world_size=4 and pg=[0,1]:
      - Groups are [0,1] and [2,3]
      - Returns {0: 0, 1: 1, 2: 0, 3: 1}
    """
    ranks, group_offsets, _offset = _prepare_collective_groups(pg)
    mapping = {}
    for group_offset in group_offsets:
        for local_rank, r in enumerate(ranks):
            mapping[group_offset + r] = local_rank
    return mapping


# =============================================================================
# reinterpret autograd Functions
# =============================================================================


class _ReplicateToVarying(torch.autograd.Function):
    """reinterpret(R,V): R -> V, backward is reinterpret(V,P): V -> P (no-op)."""

    @staticmethod
    def forward(ctx, x, axis, dtype_options):
        ctx.axis = axis
        ctx.backward_options = dtype_options.backward_options
        x = _apply_op_dtype(x, dtype_options.op_dtype)
        return _apply_out_dtype(x, dtype_options.out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # reinterpret(V,P) is a no-op in forward direction
        return (
            reinterpret(grad_out, ctx.axis, src=V, dst=P, **ctx.backward_options),
            None,
            None,
        )


class _ReplicateToInvariant(torch.autograd.Function):
    """convert(R,I): R -> I, backward is convert(I,P): I -> P."""

    @staticmethod
    def forward(ctx, x, axis, dtype_options):
        ctx.axis = axis
        ctx.backward_options = dtype_options.backward_options
        x = _apply_op_dtype(x, dtype_options.op_dtype)
        return _apply_out_dtype(x, dtype_options.out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # convert(I,P): I -> P
        return (
            convert(
                grad_out,
                ctx.axis,
                src=I,
                dst=P,
                expert_mode=True,
                **ctx.backward_options,
            ),
            None,
            None,
        )


class _InvariantToReplicate(torch.autograd.Function):
    """convert(I,R): I -> R, backward is all_reduce(I): P -> I."""

    @staticmethod
    def forward(ctx, x, axis, dtype_options):
        ctx.axis = axis
        ctx.backward_options = dtype_options.backward_options
        x = _apply_op_dtype(x, dtype_options.op_dtype)
        return _apply_out_dtype(x, dtype_options.out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is all_reduce(I): P -> I
        from spmd_types._collectives import all_reduce  # @manual

        return (
            all_reduce(grad_out, ctx.axis, src=P, dst=I, **ctx.backward_options),
            None,
            None,
        )


class _VaryingToPartial(torch.autograd.Function):
    """reinterpret(V,P): V -> P, backward is reinterpret(R,V): R -> V (no-op)."""

    @staticmethod
    def forward(ctx, x, axis, dtype_options):
        ctx.axis = axis
        ctx.backward_options = dtype_options.backward_options
        x = _apply_op_dtype(x, dtype_options.op_dtype)
        return _apply_out_dtype(x, dtype_options.out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # reinterpret(R,V): R -> V
        return (
            reinterpret(
                grad_out,
                ctx.axis,
                src=R,
                dst=V,
                expert_mode=True,
                **ctx.backward_options,
            ),
            None,
            None,
        )


class _ReplicateToPartial(torch.autograd.Function):
    """reinterpret(R,P): R -> P, backward is reinterpret(R,P): R -> P."""

    @staticmethod
    def forward(ctx, x, axis, dtype_options):
        ctx.axis = axis
        ctx.backward_options = dtype_options.backward_options
        x = _apply_op_dtype(x, dtype_options.op_dtype)
        return _apply_out_dtype(x, dtype_options.out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # reinterpret(R,P): R -> P (self-dual)
        return (
            reinterpret(
                grad_out,
                ctx.axis,
                src=R,
                dst=P,
                expert_mode=True,
                **ctx.backward_options,
            ),
            None,
            None,
        )


@api_boundary
def reinterpret(  # noqa: C901
    x,
    axis: ProcessGroup,
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
    expert_mode: bool = False,
    op_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
    backward_options: dict | None = None,
):
    """``reinterpret(x, mesh_axis, src, dst)``

    Coerce from one local SPMD type to another local SPMD type without changing
    the local tensor.  It is guaranteed to be a no-op in forwards.

    **Important:** Unlike ``convert``, ``reinterpret`` can change the semantic
    value of a tensor.  For example, ``reinterpret(R,P)`` treats a replicated
    value as if it were partial, meaning after reduction you get N times the
    original value (where N is the mesh axis size).  If you want to preserve
    semantics, use ``convert``.

    This API does not support shard for src/dst, because the restriction on no
    local tensor change means that the local SPMD semantics would be precisely
    the same as the corresponding varying operation.

    Args:
        x: Input tensor
        axis: The mesh axis to operate on (ProcessGroup)
        src: Source local SPMD type (R, I, V, P)
        dst: Destination local SPMD type (R, I, V, P)
        op_dtype: Cast input to this dtype before the operation.
        out_dtype: Cast output to this dtype after the operation.
        backward_options: Dict of options passed to the backward operation.

    **Supported coercions:**

    ``reinterpret(R,I): R -> I`` delegates to ``convert(R,I)``; see ``convert``
    for the full specification and diagrams.

    ``reinterpret(R,V): R -> V``, the backwards is ``reinterpret(V,P): V -> P``::

        def reinterpret_R_V_spec(x: f32[*shape]) -> List[f32[*shape]]:
            # Makes N copies of the input
            return [x] * mesh_axis_size

        Forward:
               A
        A  =>  A
               A

        Backward:
        +A      A
        +B  <=  B
        +C      C

    ``reinterpret(I,R): I -> R`` delegates to ``convert(I,R)``; see ``convert``
    for the full specification and diagrams.

    ``reinterpret(V,P): V -> P``, the backwards is ``reinterpret(R,V): R -> V``::

        def reinterpret_V_P_spec(xs: List[f32[*shape]]) -> f32[*shape]:
            # Semantically does a sum, even if physically it hasn't happened yet!
            return sum(xs)

        Forward:
        A      +A
        B  =>  +B
        C      +C

        Backward:
        A
        A  <=  A
        A

    ``reinterpret(R,P): R -> P``, the backwards is ``reinterpret(R,P): R -> P``::

        def reinterpret_R_P(x: f32[*shape]) -> f32[*shape]:
            # Summing each replicated entry together scales the value by axis size
            return x * mesh_axis_size

        Forward:
               +A
        A  =>  +A
               +A

        Backward:
        +A
        +A  <=  A
        +A

    ``reinterpret(I,V): I -> V`` is the composition of ``I -> R -> V``.
    ``reinterpret(R,P): R -> P`` is the composition of ``R -> V -> P``.
    ``reinterpret(I,P): I -> P`` is the composition of ``I -> R -> P``.  Note
    that these reinterprets have unusual semantics: the resulting tensor has
    been scaled by the mesh axis size (because you are now obligated to sum
    each of the (equal) quantities of the rank together!)  If you instead
    wanted to *preserve* the original semantic meaning of the tensor, use
    ``convert``.

    Here is a table of permissible reinterprets (``-`` is no-op, ``X`` is
    direct coercion, ``/`` is transitive coercion.)::

               dst
               R I V P
        src R  - X X /
            I  X - / /
            V      - X
            P        -
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            reinterpret,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
            expert_mode=expert_mode,
            op_dtype=op_dtype,
            out_dtype=out_dtype,
            backward_options=backward_options,
        )
    # Validate no Shard types
    if isinstance(src, Shard) or isinstance(dst, Shard):
        raise ValueError(
            f"reinterpret does not support S(i). Use V instead, or use convert for "
            f"semantics-preserving conversions. Got src={src}, dst={dst}"
        )

    if src is dst:
        return x  # no-op

    # Gate expert-only coercions
    if not expert_mode:
        if src is R and dst is I:
            raise ValueError(
                "reinterpret(R, I) requires expert_mode=True. "
                "This is rarely what you want in the forward pass; "
                "it exists as a backward for convert(I, P). "
                "Prefer convert(R, I) which is the primary API for R<->I transitions."
            )
        if src is R and dst is P:
            raise ValueError(
                "reinterpret(R, P) requires expert_mode=True. "
                "This scales the semantic value by the mesh axis size, which is rarely "
                "intentional. If you want to preserve semantics, use convert(R, P) instead."
            )
        if src is I and dst is R:
            raise ValueError(
                "reinterpret(I, R) requires expert_mode=True. "
                "Use convert(I, R) instead, which is the primary API for I<->R transitions."
            )
        if src is R and dst is V:
            raise ValueError(
                "reinterpret(R, V) requires expert_mode=True. "
                "This makes N copies of the input, which is rarely what you want in the "
                "forward pass. If you want to shard a replicated value, use convert(R, V) "
                "or convert(R, S(i)) instead."
            )

    # Delegated cases: pass raw kwargs through to convert
    if src is R and dst is I:
        return convert(
            x,
            axis,
            src=R,
            dst=I,
            expert_mode=True,
            op_dtype=op_dtype,
            out_dtype=out_dtype,
            backward_options=backward_options,
        )
    if src is I and dst is R:
        return convert(
            x,
            axis,
            src=I,
            dst=R,
            op_dtype=op_dtype,
            out_dtype=out_dtype,
            backward_options=backward_options,
        )

    # Non-delegated cases: process dtype options and use autograd functions
    dtype_options = _process_dtype_options(
        op_dtype,
        out_dtype,
        backward_options,
        reducing=False,
        backward_has_reduction=False,
        input_dtype=x.dtype,
        requires_grad=x.requires_grad,
    )

    if src is R and dst is V:
        return _ReplicateToVarying.apply(x, axis, dtype_options)
    elif src is R and dst is P:
        return _ReplicateToPartial.apply(x, axis, dtype_options)
    elif src is I and dst is V:
        # Composition: I -> R -> V
        first_opts, second_opts = _split_composition_dtype_options(dtype_options)
        return _ReplicateToVarying.apply(
            _InvariantToReplicate.apply(x, axis, first_opts),
            axis,
            second_opts,
        )
    elif src is I and dst is P:
        # Composition: I -> R -> P
        first_opts, second_opts = _split_composition_dtype_options(dtype_options)
        return _ReplicateToPartial.apply(
            _InvariantToReplicate.apply(x, axis, first_opts),
            axis,
            second_opts,
        )
    elif src is V and dst is P:
        return _VaryingToPartial.apply(x, axis, dtype_options)
    else:
        if src is P:
            raise ValueError(
                f"reinterpret({src}, {dst}) is not supported; it is semantically ill-defined. "
                "Call all_reduce(src=P, dst=R) first to materialize the sum, "
                "then do whatever conversion you need from R."
            )
        elif src is V:
            # V -> R or V -> I
            raise ValueError(
                f"reinterpret({src}, {dst}) is not supported. "
                f"We cannot unsafely assert that varying values on all ranks are actually the same. "
                f"Ensure your source is already {dst} instead."
            )
        else:
            raise ValueError(f"reinterpret({src}, {dst}) is not supported.")


# =============================================================================
# convert helper functions
# =============================================================================


@torch.library.custom_op("spmd_types::replicate_to_varying", mutates_args=())
def _replicate_to_varying(
    x: torch.Tensor, world_size: int, split_dim: int, rank: int, stack: bool
) -> torch.Tensor:
    """Forward: split and take local portion based on rank (never aliases input)."""
    if stack:
        assert x.shape[split_dim] == world_size
        result = x.select(split_dim, rank).contiguous()
    else:
        chunks = torch.chunk(x, world_size, dim=split_dim)
        result = chunks[rank].contiguous()
    if result.untyped_storage() is x.untyped_storage():
        return result._lazy_clone()
    return result


@_replicate_to_varying.register_fake
def _replicate_to_varying_fake(
    x: torch.Tensor, world_size: int, split_dim: int, rank: int, stack: bool
) -> torch.Tensor:
    if stack:
        shape = list(x.shape)
        del shape[split_dim]
        return x.new_empty(shape)
    shape = list(x.shape)
    shape[split_dim] = (shape[split_dim] + world_size - 1) // world_size
    return x.new_empty(shape)


@torch.library.custom_op("spmd_types::varying_to_partial", mutates_args=())
def _varying_to_partial(
    x: torch.Tensor, world_size: int, split_dim: int, rank: int, stack: bool
) -> torch.Tensor:
    """Forward: pad with zeros, place data at rank position."""
    if stack:
        pad_shape = list(x.shape)
        pad_shape.insert(split_dim, world_size)
        result = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        slices: list[int | slice] = [slice(None)] * len(pad_shape)
        slices[split_dim] = rank
        result[tuple(slices)] = x
        return result
    pad_shape = list(x.shape)
    pad_shape[split_dim] = pad_shape[split_dim] * world_size
    result = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    chunk_size = x.shape[split_dim]
    slices_c: list[int | slice] = [slice(None)] * len(pad_shape)
    slices_c[split_dim] = slice(rank * chunk_size, (rank + 1) * chunk_size)
    result[tuple(slices_c)] = x
    return result


@_varying_to_partial.register_fake
def _varying_to_partial_fake(
    x: torch.Tensor, world_size: int, split_dim: int, rank: int, stack: bool
) -> torch.Tensor:
    if stack:
        shape = list(x.shape)
        shape.insert(split_dim, world_size)
        return x.new_empty(shape)
    shape = list(x.shape)
    shape[split_dim] = shape[split_dim] * world_size
    return x.new_empty(shape)


@torch.library.custom_op("spmd_types::replicate_to_partial", mutates_args=())
def _replicate_to_partial(x: torch.Tensor, rank: int) -> torch.Tensor:
    """Forward: keep value on rank 0, zero elsewhere."""
    if rank == 0:
        return x.clone()
    else:
        return torch.zeros_like(x)


@_replicate_to_partial.register_fake
def _replicate_to_partial_fake(x: torch.Tensor, rank: int) -> torch.Tensor:
    return torch.empty_like(x)


# =============================================================================
# convert autograd Functions
# =============================================================================


class _ConvertReplicateToVarying(torch.autograd.Function):
    """convert(R,V): R -> V, backward is convert(V,P): V -> P."""

    @staticmethod
    def forward(ctx, x, axis, split_dim, stack, dtype_options):
        ctx.axis = axis
        ctx.split_dim = split_dim
        ctx.stack = stack
        ctx.backward_options = dtype_options.backward_options
        x = _apply_op_dtype(x, dtype_options.op_dtype)
        pg = axis
        world_size = _dist.dist.get_world_size(pg)

        mode = _get_local_tensor_mode(x)
        if mode is not None:
            r2l = _build_rank_to_group_local(pg)
            result = mode.tensor_map(
                x,
                lambda r, t: _replicate_to_varying(
                    t, world_size, split_dim, r2l[r], stack
                ),
            )
        else:
            rank = _dist.dist.get_rank(pg)
            result = _replicate_to_varying(x, world_size, split_dim, rank, stack)
        return _apply_out_dtype(result, dtype_options.out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # convert(V/S(i),P): V -> P or S(i) -> P
        src = V if ctx.stack else Shard(ctx.split_dim)
        return (
            convert(
                grad_out,
                ctx.axis,
                src=src,
                dst=P,
                expert_mode=True,
                **ctx.backward_options,
            ),
            None,
            None,
            None,
            None,
        )


class _ConvertInvariantToVarying(torch.autograd.Function):
    """convert(I,V): I -> V, backward is all_gather(I): V -> I."""

    @staticmethod
    def forward(ctx, x, axis, split_dim, stack, dtype_options):
        ctx.axis = axis
        ctx.split_dim = split_dim
        ctx.stack = stack
        ctx.backward_options = dtype_options.backward_options
        x = _apply_op_dtype(x, dtype_options.op_dtype)
        pg = axis
        world_size = _dist.dist.get_world_size(pg)

        mode = _get_local_tensor_mode(x)
        if mode is not None:
            r2l = _build_rank_to_group_local(pg)
            result = mode.tensor_map(
                x,
                lambda r, t: _replicate_to_varying(
                    t, world_size, split_dim, r2l[r], stack
                ),
            )
        else:
            rank = _dist.dist.get_rank(pg)
            result = _replicate_to_varying(x, world_size, split_dim, rank, stack)
        return _apply_out_dtype(result, dtype_options.out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # backward is all_gather(I): V -> I
        from spmd_types._collectives import all_gather  # @manual

        src = V if ctx.stack else Shard(ctx.split_dim)
        return (
            all_gather(
                grad_out,
                ctx.axis,
                src=src,
                dst=I,
                **ctx.backward_options,
            ),
            None,
            None,
            None,
            None,
        )


class _ConvertReplicateToPartial(torch.autograd.Function):
    """convert(R,P): R -> P, backward is convert(R,P): R -> P."""

    @staticmethod
    def forward(ctx, x, axis, dtype_options):
        ctx.axis = axis
        ctx.backward_options = dtype_options.backward_options
        x = _apply_op_dtype(x, dtype_options.op_dtype)
        pg = axis

        mode = _get_local_tensor_mode(x)
        if mode is not None:
            r2l = _build_rank_to_group_local(pg)
            result = mode.tensor_map(x, lambda r, t: _replicate_to_partial(t, r2l[r]))
        else:
            rank = _dist.dist.get_rank(pg)
            result = _replicate_to_partial(x, rank)
        return _apply_out_dtype(result, dtype_options.out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # convert(R,P): R -> P (self-dual)
        return (
            convert(grad_out, ctx.axis, src=R, dst=P, **ctx.backward_options),
            None,
            None,
        )


class _ConvertInvariantToPartial(torch.autograd.Function):
    """convert(I,P): I -> P, backward is convert(R,I): R -> I (no-op)."""

    @staticmethod
    def forward(ctx, x, axis, dtype_options):
        ctx.axis = axis
        ctx.backward_options = dtype_options.backward_options
        x = _apply_op_dtype(x, dtype_options.op_dtype)
        pg = axis

        mode = _get_local_tensor_mode(x)
        if mode is not None:
            r2l = _build_rank_to_group_local(pg)
            result = mode.tensor_map(x, lambda r, t: _replicate_to_partial(t, r2l[r]))
        else:
            rank = _dist.dist.get_rank(pg)
            result = _replicate_to_partial(x, rank)
        return _apply_out_dtype(result, dtype_options.out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # convert(R,I): R -> I
        return (
            convert(
                grad_out,
                ctx.axis,
                src=R,
                dst=I,
                expert_mode=True,
                **ctx.backward_options,
            ),
            None,
            None,
        )


class _ConvertVaryingToPartial(torch.autograd.Function):
    """convert(V,P): V -> P, backward is convert(R,V): R -> V."""

    @staticmethod
    def forward(ctx, x, axis, split_dim, stack, dtype_options):
        ctx.axis = axis
        ctx.split_dim = split_dim
        ctx.stack = stack
        ctx.backward_options = dtype_options.backward_options
        x = _apply_op_dtype(x, dtype_options.op_dtype)
        pg = axis
        world_size = _dist.dist.get_world_size(pg)

        mode = _get_local_tensor_mode(x)
        if mode is not None:
            r2l = _build_rank_to_group_local(pg)
            result = mode.tensor_map(
                x,
                lambda r, t: _varying_to_partial(
                    t, world_size, split_dim, r2l[r], stack
                ),
            )
        else:
            rank = _dist.dist.get_rank(pg)
            result = _varying_to_partial(x, world_size, split_dim, rank, stack)
        return _apply_out_dtype(result, dtype_options.out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # convert(R,V/S(i)): R -> V or R -> S(i)
        dst = V if ctx.stack else Shard(ctx.split_dim)
        return (
            convert(grad_out, ctx.axis, src=R, dst=dst, **ctx.backward_options),
            None,
            None,
            None,
            None,
        )


@api_boundary
def convert(  # noqa: C901
    x,
    axis: ProcessGroup,
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
    expert_mode: bool = False,
    op_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
    backward_options: dict | None = None,
):
    """``convert(x, mesh_axis, src, dst)``

    Convert from one local SPMD to another while preserving the semantics of
    the tensor, without doing communications.  When src/dst is shard, this
    means preserving the global SPMD semantics of the tensor per this sharding;
    when src/dst is varying, this means preserving the local SPMD semantics (we
    simply say that a non-varying tensor can be interpreted as varying by
    unbinding dim 0, following the natural behavior of collectives like
    all-gather and reduce-scatter that stack/unbind on dim 0).  The
    shard/varying conversions are actually exactly identical, except for being
    rank-preserving or not, so in the summary tables we will only include the
    varying versions of operations.  However, we include both in the API
    description for clarity.

    You cannot convert out of P: the only way to eliminate the pending
    reduction is to do the actual all-reduce.

    We also support ``convert(R,I)`` and ``convert(I,R)``.  Since R and I
    have identical local representations (same data on all ranks), both are
    no-op forwards that only differ in backward semantics.  ``reinterpret``
    also accepts R<->I and delegates to ``convert``.

    Args:
        x: Input tensor
        axis: The mesh axis to operate on (ProcessGroup)
        src: Source local SPMD type (R, I, V, P, or S(i))
        dst: Destination local SPMD type (R, I, V, P, or S(i)).
             When src or dst is S(i), the split/concat dimension is
             derived from the shard index.  Plain V always uses dim 0.
        op_dtype: Cast input to this dtype before the operation.
        out_dtype: Cast output to this dtype after the operation.
        backward_options: Dict of options passed to the backward operation.
            Supports ``op_dtype``, ``out_dtype``, and ``backward_options`` keys.

    **Supported conversions:**

    ``convert(I,R): I -> R``, the backward is ``all_reduce(I): P -> I``

    Input is invariant across ranks; the output is replicate, still holding
    the same data on every rank.  The only effect is on backward semantics:
    replicate gradients are partial (each rank contributes independently),
    while invariant gradients are invariant.

    Two common use cases:

    1. A replicated parameter (I@tp, e.g., norm weights) entering computation
       where each rank may contribute a different gradient.  The backward
       all-reduce synchronizes the gradient for the optimizer.
    2. Megatron ``CopyToModelParallelRegion`` (SP=False only): inter-block
       activations are I@tp, and this op transitions them to R@tp at the
       column-parallel linear entry.

    ::

        def convert_I_R_spec(x: f32[*shape]) -> f32[*shape]:
            return x

        Forward:
        A => A

        Backward:
                          +Ax
        Ax + Ay + Az  <=  +Ay
                          +Az

    ``convert(R,I): R -> I``, the backward is ``convert(I,P): I -> P``

    Input is replicate across ranks; the output is invariant, still holding
    the same data on every rank.  The only effect is on backward semantics:
    invariant gradients remain invariant, whereas replicate gradients would
    be partial.  Rarely needed in forward code (``expert_mode`` required);
    exists primarily as the backward of ``convert(I,P)``.

    ::

        def convert_R_I_spec(x: f32[*shape]) -> f32[*shape]:
            return x

        Forward:
        A  =>  A

        Backward:
        +A
        +0  <=  A
        +0

    ``convert(R,V): R -> V``, the backward is ``convert(V,P): V -> P``

    Input is replicated across ranks, so each rank holds the full tensor.  The
    output keeps only the local shard along tensor dim 0, producing a varying
    value.  This operation reduces the rank of the tensor.

    ::

        def convert_R_V_spec(x: f32[mesh_axis_size, *shape]) -> List[f32[*shape]]:
            return x.unbind()

        Forward:
                       A
        [A, B, C]  =>  B
                       C

        Backward:
        +[A, 0, 0]      A
        +[0, B, 0]  <=  B
        +[0, 0, C]      C

    ``convert(R,S(i)): R -> S(i)``, the backward is ``convert(S(i),P): S(i) -> P``

    Like above, but the rank of the tensor is not reduced, and an arbitrary
    tensor dim can be specified to be sharded.

    ::

        def convert_R_S_spec(x, i):
            return x.chunk(mesh_axis_size, i)

        Forward (for i = 0):
                                      [A0, A1]
        [A0, A1, B0, B1, C0, C1]  =>  [B0, B1]
                                      [C0, C1]

        Backward (for i = 0):
        +[A0, A1, 0,  0,  0,  0 ]      [A0, A1]
        +[0,  0,  B0, B1, 0,  0 ]  <=  [B0, B1]
        +[0,  0,  0,  0,  C0, C1]      [C0, C1]

    ``convert(I,V): I -> V``, the backwards is ``all_gather(V,I): V -> I``

    Input is invariant across ranks, so each rank holds the full tensor.  The
    output keeps only the local slice along dim 0, producing a varying value.
    The rank of the tensor is reduced.

    ::

        def convert_I_V_spec(x: f32[mesh_axis_size, *shape]) -> List[f32[*shape]]:
            return x.unbind()

        Forward:
                       A
        [A, B, C]  =>  B
                       C

        Backward:
                       A
        [A, B, C]  <=  B
                       C

    A common use case: shard a parameter (I) across ranks for memory savings.
    Each rank stores and computes on only its local slice; in backward, the
    per-rank gradient shards are all-gathered to reconstruct the full invariant
    gradient needed by the optimizer.

    ``convert(I,S(i)): I -> S(i)``, the backwards is ``all_gather(S(i),I): S(i) -> I``

    Like above, but the rank of the tensor is not reduced, and an arbitrary
    tensor dim can be specified to be sharded.

    ::

        def convert_I_S_spec(x, i):
            return x.chunk(mesh_axis_size, i)

        Forward (for i = 0):
                                      [A0, A1]
        [A0, A1, B0, B1, C0, C1]  =>  [B0, B1]
                                      [C0, C1]

        Backward (for i = 0):
                                      [A0, A1]
        [A0, A1, B0, B1, C0, C1]  <=  [B0, B1]
                                      [C0, C1]

    ``convert(R,P): R -> P``, the backwards is ``convert(R,P): R -> P``

    Input is replicated across ranks.  The output keeps the same per-rank tensor
    shape, but all ranks except the first are zeroed out, producing a partial
    value that sums to the original tensor after a cross-rank reduction.

    ::

        def convert_R_P_spec(x: f32[*shape]) -> f32[*shape]:
            return x

        Forward:
                 +[A]
        [A]  =>  +[0]
                 +[0]

        Backward:
        +[A]
        +[0]  <=  [A]
        +[0]

    This operation is its own backward (self-dual).  A common forward use
    case: when a replicated scalar (e.g., a regularization term) needs to be
    added to a partial loss without being counted N times.  Converting R -> P
    puts the value on only one rank, so the subsequent all-reduce produces
    the correct total.

    ``convert(I,P): I -> P``, the backwards is ``convert(R,I): R -> I``

    Input is invariant across ranks.  The output keeps the same per-rank tensor
    shape, but all ranks except the first are zeroed out, producing a partial
    value that sums to the original tensor after a cross-rank reduction.

    ::

        def convert_I_P_spec(x: f32[*shape]) -> f32[*shape]:
            return x

        Forward:
                 +[A]
        [A]  =>  +[0]
                 +[0]

        Backward:
        [A]  <=  [A]

    ``convert(V,P): V -> P``, the backwards is ``convert(R,V): R -> V``

    Input is varying, with each rank holding a shard or distinct value.  The
    output places each rank's value into a disjoint position of a partial
    tensor (zeros elsewhere) so that summing across ranks reconstructs the
    stacked value.  The rank of the tensor is increased.

    ::

        def convert_V_P_spec(xs: List[f32[*shape]]) -> f32[mesh_axis_size, *shape]:
            return torch.stack(xs)

        Forward:
        A      +[A, 0, 0]
        B  =>  +[0, B, 0]
        C      +[0, 0, C]

        Backward:
        A
        B  <=  [A, B, C]
        C

    ``convert(S(i),P): S(i) -> P``, the backwards is ``convert(R,S(i)): R -> S(i)``

    Like above, but the rank of the tensor is not reduced, and an arbitrary
    tensor dim can be specified to be scattered on.

    ::

        def convert_S_P_spec(xs, i):
            return torch.concat(xs, i)

        Forward (for i = 0):
        [A0, A1]      +[A0, A1, 0,  0,  0,  0 ]
        [B0, B1]  =>  +[0,  0,  B0, B1, 0,  0 ]
        [C0, C1]      +[0,  0,  0,  0,  C0, C1]

        Backward (for i = 0):
        [A0, A1]
        [B0, B1]  <=  [A0, A1, B0, B1, C0, C1]
        [C0, C1]

    Here is a table of permissible converts (``-`` is no-op, ``N`` is
    a no-op forward that only changes backward semantics (R<->I), ``O``
    is supported.)::

               dst
               R I V P
        src R  - N O O
            I  N - O O
            V      - O
            P        -
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            convert,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
            expert_mode=expert_mode,
            op_dtype=op_dtype,
            out_dtype=out_dtype,
            backward_options=backward_options,
        )
    # Canonicalize negative Shard dims
    src = _canonicalize_shard(src, x.ndim)
    dst = _canonicalize_shard(dst, x.ndim)

    # Derive dim from Shard types (V always uses dim 0)
    dim = 0
    if isinstance(src, Shard):
        dim = src.dim
    if isinstance(dst, Shard):
        dim = dst.dim

    # Normalize Shard to V for dispatch (they use the same underlying functions)
    src_base = V if isinstance(src, Shard) else src
    dst_base = V if isinstance(dst, Shard) else dst

    # Gate expert-only coercions
    if not expert_mode:
        if src_base is R and dst_base is I:
            raise ValueError(
                "convert(R, I) requires expert_mode=True. "
                "This treats a replicate tensor as invariant, which is rarely what you "
                "want in the forward pass. It exists as a backward for convert(I, P)."
            )
        if src_base is I and dst_base is P:
            raise ValueError(
                "convert(I, P) requires expert_mode=True. "
                "This zeros out all non-rank-0 tensors, which is rarely what you want. "
                "It exists as a backward for convert(R, I)."
            )
        if src_base is V and dst_base is P:
            raise ValueError(
                f"convert({src}, P) requires expert_mode=True. "
                f"This pads with zeros to create a partial tensor, which is rarely what you "
                f"want in the forward pass. It exists as a backward for convert(R, {src})."
            )

    if src_base is dst_base:
        if isinstance(src, Shard) and isinstance(dst, Shard) and src.dim != dst.dim:
            raise ValueError(
                f"convert(S({src.dim}), S({dst.dim})) cannot change the shard dimension "
                f"without communication. Use all_to_all() instead."
            )
        return x  # no-op

    # R/I -> V/S(i): tensor gets smaller, keep input precision
    reducing = (src_base is R or src_base is I) and dst_base is V
    backward_has_reduction = src_base is I and dst_base is R

    dtype_options = _process_dtype_options(
        op_dtype,
        out_dtype,
        backward_options,
        reducing=reducing,
        backward_has_reduction=backward_has_reduction,
        input_dtype=x.dtype,
        requires_grad=x.requires_grad,
    )

    if src_base is R and dst_base is V:
        stack = not isinstance(dst, Shard)
        return _ConvertReplicateToVarying.apply(x, axis, dim, stack, dtype_options)
    elif src_base is R and dst_base is P:
        return _ConvertReplicateToPartial.apply(x, axis, dtype_options)
    elif src_base is R and dst_base is I:
        # Same as reinterpret
        return _ReplicateToInvariant.apply(x, axis, dtype_options)
    elif src_base is I and dst_base is V:
        stack = not isinstance(dst, Shard)
        return _ConvertInvariantToVarying.apply(x, axis, dim, stack, dtype_options)
    elif src_base is I and dst_base is P:
        return _ConvertInvariantToPartial.apply(x, axis, dtype_options)
    elif src_base is I and dst_base is R:
        # Same as reinterpret
        return _InvariantToReplicate.apply(x, axis, dtype_options)
    elif src_base is V and dst_base is P:
        stack = not isinstance(src, Shard)
        return _ConvertVaryingToPartial.apply(x, axis, dim, stack, dtype_options)
    else:
        if src_base is P:
            if dst_base is R or dst_base is I:
                raise ValueError(
                    f"convert({src}, {dst}) is not supported. "
                    f"Use all_reduce(src=P, dst={dst}) to perform the reduction and get the full sum."
                )
            elif dst_base is V or isinstance(dst, Shard):
                raise ValueError(
                    f"convert({src}, {dst}) is not supported. "
                    f"Use reduce_scatter(src=P, dst={dst}) to perform the reduction and get shards of the sum."
                )
            else:
                raise ValueError(
                    f"convert({src}, {dst}) is not supported. Cannot convert out of P."
                )
        else:
            raise ValueError(f"convert({src}, {dst}) is not supported.")


def shard(
    x,
    axis: ProcessGroup,
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
    op_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
    backward_options: dict | None = None,
):
    """Convenience alias: ``shard(src, S(i))`` is ``convert(src, S(i))``.

    Shards a replicated or invariant tensor along a given dimension without
    communication.

    Args:
        x: Input tensor
        axis: The mesh axis to operate on (ProcessGroup)
        src: Source local SPMD type (R or I)
        dst: Destination shard type, must be S(i)
        op_dtype: Cast input to this dtype before the operation.
        out_dtype: Cast output to this dtype after the operation.
        backward_options: Dict of options passed to the backward operation.
    """
    if not isinstance(dst, Shard):
        raise ValueError(f"shard dst must be S(i), got {dst}")
    return convert(
        x,
        axis,
        src=src,
        dst=dst,
        op_dtype=op_dtype,
        out_dtype=out_dtype,
        backward_options=backward_options,
    )


def invariant_to_replicate(
    x,
    axis: ProcessGroup,
    *,
    op_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
    backward_options: dict | None = None,
):
    """Convenience alias: ``invariant_to_replicate`` is ``convert(I, R)``.

    Converts an invariant tensor to replicated.  The local tensor is
    unchanged; the only effect is that the backward will perform an
    all-reduce (P -> I) instead of expecting invariant gradients.

    Two common use cases:

    1. A replicated parameter (I@tp, e.g., norm weights) entering
       computation where each rank may contribute a different gradient.
       The backward all-reduce synchronizes the gradient for the optimizer.
    2. Megatron ``CopyToModelParallelRegion`` (SP=False only): inter-block
       activations are I@tp, and this op transitions them to R@tp at the
       column-parallel linear entry.  With sequence parallelism (the common
       case), this role is filled by ``all_gather(dst=R): V->R`` instead.

    Args:
        x: Input tensor with I type on the mesh axis
        axis: The mesh axis to operate on (ProcessGroup)
        op_dtype: Cast input to this dtype before the operation.
        out_dtype: Cast output to this dtype after the operation.
        backward_options: Dict of options passed to the backward operation.
    """
    # Directly call the autograd function rather than going through
    # convert(), so that callers (e.g. copy_to_tensor_model_parallel_region)
    # can read straight through to the actual work without needing to match
    # src/dst and trace into reinterpret's dispatch table.  This duplicates
    # the I->R case from reinterpret() intentionally for readability.
    if has_torch_function_unary(x):
        return handle_torch_function(
            invariant_to_replicate,
            (x,),
            x,
            axis,
            op_dtype=op_dtype,
            out_dtype=out_dtype,
            backward_options=backward_options,
        )
    dtype_options = _process_dtype_options(
        op_dtype,
        out_dtype,
        backward_options,
        reducing=False,
        backward_has_reduction=True,
        input_dtype=x.dtype,
        requires_grad=x.requires_grad,
    )
    return _InvariantToReplicate.apply(x, axis, dtype_options)
