"""Distributed collective operations: all_reduce, all_gather, reduce_scatter, all_to_all, redistribute."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import torch
from spmd_types import _dist
from spmd_types._local import convert, reinterpret
from spmd_types._traceback import api_boundary
from spmd_types.types import _canonicalize_shard, I, P, PerMeshAxisSpmdType, R, Shard, V
from torch.overrides import handle_torch_function, has_torch_function_unary

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

# =============================================================================
# all_reduce: P -> R | I
# =============================================================================


class _AllReduce(torch.autograd.Function):
    """all_reduce: P -> R|I.

    When dst=R, backward is all_reduce(R): P -> R.
    When dst=I, backward is convert(I,R): I -> R (no-op).
    """

    @staticmethod
    def forward(ctx, x, axis, dst, inplace):
        ctx.axis = axis
        ctx.dst = dst
        pg = axis
        # TODO: check if contiguous assertion is really necessary
        assert x.is_contiguous(), "all_reduce input must be contiguous"
        # TODO: check if world_size == 1 short-circuit is really necessary
        if _dist.dist.get_world_size(pg) == 1:
            if inplace:
                return x
            return x.clone()
        if inplace:
            _dist.dist.all_reduce(x, op=_dist.dist.ReduceOp.SUM, group=pg)
            ctx.mark_dirty(x)
            return x
        else:
            result = x.clone()
            _dist.dist.all_reduce(result, op=_dist.dist.ReduceOp.SUM, group=pg)
            return result

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.dst is R:
            # backward of P -> R is P -> R (same operation)
            grad = all_reduce(grad_out, ctx.axis, src=P, dst=R)
        else:
            # backward of P -> I: convert(I,R) is identity but sets up autograd for double backward
            grad = convert(grad_out, ctx.axis, src=I, dst=R)
        return grad, None, None, None


@api_boundary
def all_reduce(
    x,
    axis: ProcessGroup,
    *,
    src: PerMeshAxisSpmdType = P,
    dst: PerMeshAxisSpmdType,
    inplace: bool = False,
):
    """``all_reduce(x: Partial | Varying, mesh_axis, dst) -> Replicate | Invariant``

    Reduce shards along the mesh axis, so that every rank has the full summed value.

    ::

        def all_reduce_spec(x: f32[*shape]) -> f32[*shape]:
            return x  # Identity! The summation already occured on x's conversion to Partial

        +Ax
        +Ay  =>  Ax + Ay + Az
        +Az

    Args:
        x: Input tensor with P or V type on the mesh axis
        axis: The mesh axis to reduce over (ProcessGroup)
        src: Source type (P or V; if V, automatically reinterpreted as P first)
        dst: Destination type (R or I)
        inplace: If True, perform the all-reduce in-place on the input tensor
            using ``dist.all_reduce`` instead of allocating a new output tensor.

    Returns:
        Tensor with R or I type depending on dst

    **Backward cases:**

    When ``dst=R``, aka ``all_reduce(R): P -> R``, the backwards is ``all_reduce(R): P -> R``::

                          +Ax
        Ax + Ay + Az  <=  +Ay
                          +Az

    When ``dst=I``, aka ``all_reduce(I): P -> I``, the backwards is ``convert(I,R): I -> R``::

        A  <=  A

    It is common to want to ``all_reduce`` on varying data; just
    ``reinterpret(V,P)`` the data as partial before calling ``all_reduce``.

    **Why is** ``dst`` **required?** The backward pass differs materially:
    ``all_reduce(R): P -> R`` has backward ``reduce_scatter: P -> V``, while
    ``all_reduce(I): P -> I`` has backward ``convert(I,R): I -> R``
    (identity).  If a library always produced I and the caller later converted
    ``I -> R``, the backward would miss the reduce-scatter optimization you'd
    get from ``all_reduce(R)`` directly.  For library code that doesn't know
    the caller's intent, accept ``dst`` as a parameter rather than hardcoding
    a choice.

    **When to choose R vs I:** If you intend to do *duplicated* work next
    (same computation on every rank), use ``dst=I``.  If you intend to do
    *different* work next (each rank uses the result differently), use
    ``dst=R``.

    **Example:** In Megatron-style TP without sequence parallelism,
    ``ReduceFromModelParallelRegion`` (forward: all-reduce, backward:
    identity) is ``all_reduce(dst=I)``.  The identity backward confirms
    the choice of I--``all_reduce(dst=R)`` is self-dual and would require
    an all-reduce in its backward too.  With sequence parallelism (the
    common case), this role is filled by ``reduce_scatter(): P->V``
    instead.
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            all_reduce,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
            inplace=inplace,
        )
    if src is not P:
        if src is V:
            x = reinterpret(x, axis, src=V, dst=P)
        elif src is R or src is I:
            raise ValueError(
                f"all_reduce src must be P, got {src}. "
                "all_reduce on replicated/invariant data is usually a bug. "
                f"If you really want to scale by mesh size, use reinterpret(src={src}, dst=P) first."
            )
        else:
            raise ValueError(f"all_reduce src must be P, got {src}")
    if dst is R or dst is I:
        return _AllReduce.apply(x, axis, dst, inplace)
    else:
        raise ValueError(f"all_reduce dst must be R or I, got {dst}")


# =============================================================================
# all_gather: V -> R | I
# =============================================================================


class _AllGatherStack(torch.autograd.Function):
    """all_gather with stack semantics: V -> R|I.

    When dst=R, backward is reduce_scatter(V): P -> V.
    When dst=I, backward is convert(I,V): I -> V.
    """

    @staticmethod
    def forward(ctx, x, axis, dst, gather_dim):
        ctx.axis = axis
        ctx.dst = dst
        ctx.gather_dim = gather_dim
        pg = axis
        world_size = _dist.dist.get_world_size(pg)
        gathered = [torch.empty_like(x) for _ in range(world_size)]
        _dist.dist.all_gather(gathered, x, group=pg)
        return torch.stack(gathered, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.dst is R:
            grad = reduce_scatter(
                grad_out, ctx.axis, src=P, dst=V, scatter_dim=ctx.gather_dim
            )
        else:
            grad = convert(grad_out, ctx.axis, src=I, dst=V)
        return grad, None, None, None


class _AllGatherShard(torch.autograd.Function):
    """all_gather with shard semantics: S(i) -> R|I.

    When dst=R, backward is reduce_scatter(S(i)): P -> S(i).
    When dst=I, backward is convert(I,S(i)): I -> S(i).
    """

    @staticmethod
    def forward(ctx, x, axis, dst, gather_dim):
        ctx.axis = axis
        ctx.dst = dst
        ctx.gather_dim = gather_dim
        pg = axis
        world_size = _dist.dist.get_world_size(pg)
        gathered = [torch.empty_like(x) for _ in range(world_size)]
        _dist.dist.all_gather(gathered, x, group=pg)
        return torch.cat(gathered, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.dst is R:
            grad = reduce_scatter(
                grad_out,
                ctx.axis,
                src=P,
                dst=Shard(ctx.gather_dim),
                scatter_dim=ctx.gather_dim,
            )
        else:
            grad = convert(grad_out, ctx.axis, src=I, dst=Shard(ctx.gather_dim))
        return grad, None, None, None


class _AllGatherUneven(torch.autograd.Function):
    """all_gather with uneven split_sizes: S(i) -> R|I.

    When dst=R, backward is reduce_scatter(S(i)): P -> S(i) with split_sizes.
    When dst=I, backward is manual split and select rank's chunk.
    """

    @staticmethod
    def forward(ctx, x, axis, dst, gather_dim, split_sizes):
        ctx.axis = axis
        ctx.dst = dst
        ctx.gather_dim = gather_dim
        ctx.split_sizes = split_sizes
        pg = axis
        ctx.rank = _dist.dist.get_rank(pg)
        gathered = []
        for s in split_sizes:
            shape = list(x.shape)
            shape[gather_dim] = s
            gathered.append(torch.empty(shape, dtype=x.dtype, device=x.device))
        _dist.dist.all_gather(gathered, x, group=pg)
        return torch.cat(gathered, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.dst is R:
            output = reduce_scatter(
                grad_out,
                ctx.axis,
                src=P,
                dst=Shard(ctx.gather_dim),
                split_sizes=ctx.split_sizes,
            )
        else:
            chunks = list(
                torch.split(grad_out, list(ctx.split_sizes), dim=ctx.gather_dim)
            )
            output = chunks[ctx.rank].contiguous()
        return output, None, None, None, None


@api_boundary
def all_gather(
    x,
    axis: ProcessGroup,
    *,
    src: PerMeshAxisSpmdType = V,
    dst: PerMeshAxisSpmdType,
    split_sizes: Optional[List[int]] = None,
):
    """``all_gather(x: Varying, mesh_axis, src, dst) -> Replicate | Invariant``

    Gather shards along the mesh axis, so that every rank has the full copy of
    the data.  PyTorch's ``all_gather`` can either concat or stack inputs
    together.  When the source tensor is interpreted as a varying tensor, we
    stack the inputs together, creating a new dimension of size mesh axis.  But
    if the source tensor is a sharded tensor, we concat along the sharded
    dimension.

    ::

        def all_gather_spec(xs, src):
            # NB: dst only affects autograd
            match src:
                case V:
                    '''
                    A
                    B  =>  [A, B, C]
                    C
                    '''
                    return torch.stack(xs)

                case S(i):
                    '''
                    When i == 0:
                    [A0, A1]
                    [B0, B1]  =>  [A0, A1, B0, B1, C0, C1]
                    [C0, C1]
                    '''
                    return torch.concat(xs, i)

    Args:
        x: Input tensor with V or S(i) type on the mesh axis
        axis: The mesh axis to gather over (ProcessGroup)
        src: Source type (V or S(i)). When V, stacks on dim 0. When S(i), concatenates on dim i.
        dst: Destination type (R or I)
        split_sizes: Per-rank sizes for uneven gathering.

    Returns:
        Tensor with R or I type depending on dst

    **Backward cases:**

    ``all_gather(V,R): V -> R``, the backwards is ``reduce_scatter(V): P -> V``::

        Ax + Ay + Az      +[Ax, Bx, Cx]
        Bx + By + Bz  <=  +[Ay, By, Cy]
        Cx + Cy + Cz      +[Az, Bz, Cz]

    ``all_gather(V,I): V -> I``, the backwards is ``convert(I,V): I -> V``::

        A
        B  <=  [A, B, C]
        C

    ``all_gather(S(i),R): S(i) -> R``, the backwards is ``reduce_scatter(S(i)): P -> S(i)``::

        When i == 0:
        [A0x + A0y + A0z, A1x + A1y + A1z]      +[A0x, A1x, B0x, B1x, C0x, C1x]
        [B0x + B0y + B0z, B1x + B1y + B1z]  <=  +[A0y, A1y, B0y, B1y, C0y, C1y]
        [C0x + C0y + C0z, C1x + C1y + C1z]      +[A0z, A1z, B0z, B1z, C0z, C1z]

    ``all_gather(S(i),I): S(i) -> I``, the backwards is ``convert(I,S(i)): I -> S(i)``::

        [A0, A1]
        [B0, B1]  <=  [A0, A1, B0, B1, C0, C1]
        [C0, C1]

    **When to choose R vs I:** If you intend to do *duplicated* work next
    (same computation on every rank), use ``dst=I``.  If you intend to do
    *different* work next (each rank uses the result differently), use
    ``dst=R``.

    **Example:** Megatron-style vocab-parallel cross-entropy computes
    loss from TP-sharded logits.  At the type level, this decomposes as
    ``all_gather(V, dst=I)`` (reconstruct full logits) followed by
    ``CE(I, I) -> I`` (duplicated loss computation).  (The actual
    implementation avoids materializing full logits by using scalar
    all-reduces; the type decomposition describes the semantic contract,
    not the communication pattern.)  The ``dst=I``
    backward--``convert(I, V)``--matches the actual backward, which
    computes each rank's logit-gradient partition locally with no
    communication.  Using ``dst=R`` would incorrectly insert a
    reduce-scatter.
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            all_gather,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
            split_sizes=split_sizes,
        )
    # Canonicalize negative Shard dims
    src = _canonicalize_shard(src, x.ndim)

    # Validate src is V or S(i)
    if not (src is V or isinstance(src, Shard)):
        raise ValueError(f"all_gather src must be V or S(i), got {src}")

    if split_sizes is not None and not isinstance(src, Shard):
        raise ValueError(
            f"all_gather split_sizes is only supported with src=S(i), got src={src}"
        )

    gather_dim = src.dim if isinstance(src, Shard) else 0
    stack = src is V
    if dst is R or dst is I:
        if split_sizes is not None:
            return _AllGatherUneven.apply(x, axis, dst, gather_dim, split_sizes)
        if stack:
            return _AllGatherStack.apply(x, axis, dst, gather_dim)
        return _AllGatherShard.apply(x, axis, dst, gather_dim)
    else:
        raise ValueError(f"all_gather dst must be R or I, got {dst}")


# =============================================================================
# reduce_scatter: P -> V
# =============================================================================


class _ReduceScatterStack(torch.autograd.Function):
    """reduce_scatter with stack semantics: P -> V, backward is all_gather(V,R): V -> R."""

    @staticmethod
    def forward(ctx, x, axis, scatter_dim):
        ctx.axis = axis
        ctx.scatter_dim = scatter_dim
        pg = axis
        # x stacked on dim 0: shape[0] == world_size
        result = x.new_empty([1] + list(x.shape[1:]))
        _dist.dist.reduce_scatter_tensor(
            result, x, op=_dist.dist.ReduceOp.SUM, group=pg
        )
        return result.squeeze(0)

    @staticmethod
    def backward(ctx, grad_out):
        return (
            all_gather(grad_out, ctx.axis, src=V, dst=R),
            None,
            None,
        )


class _ReduceScatterShard(torch.autograd.Function):
    """reduce_scatter with shard semantics: P -> S(i), backward is all_gather(S(i),R): S(i) -> R."""

    @staticmethod
    def forward(ctx, x, axis, scatter_dim):
        ctx.axis = axis
        ctx.scatter_dim = scatter_dim
        pg = axis
        world_size = _dist.dist.get_world_size(pg)
        # reduce_scatter_tensor always scatters along dim 0, so we
        # movedim before/after when scatter_dim != 0.
        needs_permute = scatter_dim != 0
        if needs_permute:
            x = x.movedim(scatter_dim, 0).contiguous()

        output_shape = list(x.shape)
        output_shape[0] //= world_size
        result = x.new_empty(output_shape)
        _dist.dist.reduce_scatter_tensor(
            result, x, op=_dist.dist.ReduceOp.SUM, group=pg
        )

        if needs_permute:
            result = result.movedim(0, scatter_dim)
        return result

    @staticmethod
    def backward(ctx, grad_out):
        return (
            all_gather(grad_out, ctx.axis, src=Shard(ctx.scatter_dim), dst=R),
            None,
            None,
        )


class _ReduceScatterUneven(torch.autograd.Function):
    """reduce_scatter with uneven split_sizes: P -> S(i), backward is all_gather(S(i),R) with split_sizes."""

    @staticmethod
    def forward(ctx, x, axis, scatter_dim, split_sizes):
        ctx.axis = axis
        ctx.scatter_dim = scatter_dim
        ctx.split_sizes = split_sizes
        pg = axis
        ctx.rank = _dist.dist.get_rank(pg)
        x_list = list(torch.split(x, list(split_sizes), dim=scatter_dim))
        output = torch.empty_like(x_list[ctx.rank])
        _dist.dist.reduce_scatter(output, x_list, op=_dist.dist.ReduceOp.SUM, group=pg)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        return (
            all_gather(
                grad_out,
                ctx.axis,
                src=Shard(ctx.scatter_dim),
                dst=R,
                split_sizes=ctx.split_sizes,
            ),
            None,
            None,
            None,
        )


@api_boundary
def reduce_scatter(
    x,
    axis: ProcessGroup,
    *,
    src: PerMeshAxisSpmdType = P,
    dst: PerMeshAxisSpmdType = V,
    scatter_dim: int | None = None,
    split_sizes: Optional[List[int]] = None,
):
    """``reduce_scatter(x, mesh_axis, dst): Partial -> Varying``

    Reduce shards along the mesh axis, but only get one shard of the result
    (e.g., an inefficient implementation of reduce-scatter would be to
    all-reduce and then drop the data you did not need.)  Like ``all_gather``,
    ``dst`` can either be varying for stack semantics, or shard for concat
    semantics.

    ::

        def reduce_scatter_spec(x: f32[mesh_axis_size, *shape], dst) -> List[f32[*shape]]:
            # NB: The semantic summation already occured on x's conversion to Partial
            match dst:
                case V:
                    '''
                    +[Ax, Bx, Cx]      Ax + Ay + Az
                    +[Ay, By, Cy]  =>  Bx + By + Bz
                    +[Az, Bz, Cz]      Cx + Cy + Cz
                    '''
                    return x.unbind()
                case S(i):
                    '''
                    When i == 0:
                    +[A0x, A1x, B0x, B1x, C0x, C1x]      [A0x + A0y + A0z, A1x + A1y + A1z]
                    +[A0y, A1y, B0y, B1y, C0y, C1y]  =>  [B0x + B0y + B0z, B1x + B1y + B1z]
                    +[A0z, A1z, B0z, B1z, C0z, C1z]      [C0x + C0y + C0z, C1x + C1y + C1z]
                    '''
                    return x.chunk(mesh_axis_size, i)

    Args:
        x: Input tensor with P type on the mesh axis
        axis: The mesh axis to reduce-scatter over (ProcessGroup)
        src: Source type (must be P)
        dst: Destination type (V or S(i))
        scatter_dim: The tensor dimension to scatter along. Defaults to 0 when
            dst is V; inferred from the shard dim when dst is S(i). If both
            scatter_dim and dst=S(i) are provided, they must agree.
        split_sizes: Per-rank sizes along ``scatter_dim`` for uneven scatter.

    Returns:
        Tensor with V or S(i) type depending on dst

    **Backward cases:**

    ``reduce_scatter(V): P -> V``, the backwards is ``all_gather(V,R): V -> R``::

                       A
        [A, B, C]  <=  B
                       C

    ``reduce_scatter(S(i)): P -> S(i)``, the backwards is ``all_gather(S(i),R): S(i) -> R``::

        When i == 0:
                                      [A0, A1]
        [A0, A1, B0, B1, C0, C1]  <=  [B0, B1]
                                      [C0, C1]

    It is common to want to reduce-scatter on varying data; just
    ``reinterpret(V,P)`` the data as partial before calling ``reduce_scatter``.
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            reduce_scatter,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
            scatter_dim=scatter_dim,
            split_sizes=split_sizes,
        )
    # Canonicalize negative Shard dims
    dst = _canonicalize_shard(dst, x.ndim)

    if src is not P:
        if src is V:
            x = reinterpret(x, axis, src=V, dst=P)
        elif src is R or src is I:
            raise ValueError(
                f"reduce_scatter src must be P, got {src}. "
                "reduce_scatter on replicated/invariant data is usually a bug. "
                f"If you really want to scale by mesh size, use reinterpret(src={src}, dst=P) first."
            )
        else:
            raise ValueError(f"reduce_scatter src must be P, got {src}")
    if not (dst is V or isinstance(dst, Shard)):
        raise ValueError(f"reduce_scatter dst must be V or S(i), got {dst}")

    if split_sizes is not None and not isinstance(dst, Shard):
        raise ValueError(
            f"reduce_scatter split_sizes is only supported with dst=S(i), got dst={dst}"
        )

    if isinstance(dst, Shard):
        if scatter_dim is not None and scatter_dim != dst.dim:
            raise ValueError(
                f"reduce_scatter got scatter_dim={scatter_dim} but dst=S({dst.dim}); "
                f"these conflict. Either use dst=S({scatter_dim}) or remove scatter_dim."
            )
        scatter_dim = dst.dim
    else:
        if scatter_dim is None:
            scatter_dim = 0

    if dst is V:
        return _ReduceScatterStack.apply(x, axis, scatter_dim)
    elif split_sizes is not None:
        return _ReduceScatterUneven.apply(x, axis, scatter_dim, split_sizes)
    else:
        return _ReduceScatterShard.apply(x, axis, scatter_dim)


# =============================================================================
# all_to_all: V -> V
# =============================================================================


class _AllToAllStack(torch.autograd.Function):
    """all_to_all with stack semantics: V -> V, backward is all_to_all(V,V) with swapped dims."""

    @staticmethod
    def forward(ctx, x, axis, split_dim, concat_dim):
        ctx.axis = axis
        ctx.split_dim = split_dim
        ctx.concat_dim = concat_dim
        pg = axis
        input_chunks = list(torch.unbind(x, dim=split_dim))
        output_chunks = [
            torch.empty_like(input_chunks[0]) for _ in range(len(input_chunks))
        ]
        _dist.dist.all_to_all(output_chunks, input_chunks, group=pg)
        return torch.stack(output_chunks, dim=concat_dim)

    @staticmethod
    def backward(ctx, grad_out):
        grad = all_to_all(
            grad_out,
            ctx.axis,
            src=V,
            dst=V,
            split_dim=ctx.concat_dim,
            concat_dim=ctx.split_dim,
        )
        return grad, None, None, None


class _AllToAllUneven(torch.autograd.Function):
    """all_to_all with uneven split sizes on dim 0 for S(0) -> S(0)."""

    @staticmethod
    def forward(ctx, x, axis, output_split_sizes, input_split_sizes):
        ctx.axis = axis
        pg = axis
        world_size = _dist.dist.get_world_size(pg)
        if input_split_sizes is None:
            input_split_sizes = [x.shape[0] // world_size] * world_size
        if output_split_sizes is None:
            output_split_sizes = [x.shape[0] // world_size] * world_size
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        input_chunks = [
            c.contiguous() for c in torch.split(x, input_split_sizes, dim=0)
        ]
        output_chunks = [
            torch.empty([s] + list(x.shape[1:]), dtype=x.dtype, device=x.device)
            for s in output_split_sizes
        ]
        _dist.dist.all_to_all(output_chunks, input_chunks, group=pg)
        return torch.cat(output_chunks, dim=0)

    @staticmethod
    def backward(ctx, grad_out):
        # Backward reverses input/output splits.
        grad = all_to_all(
            grad_out,
            ctx.axis,
            src=Shard(0),
            dst=Shard(0),
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
        )
        return grad, None, None, None


class _AllToAllShard(torch.autograd.Function):
    """all_to_all with shard semantics: S(i) -> S(j), backward is all_to_all(S(j),S(i))."""

    @staticmethod
    def forward(ctx, x, axis, split_dim, concat_dim):
        ctx.axis = axis
        ctx.split_dim = split_dim
        ctx.concat_dim = concat_dim
        pg = axis
        world_size = _dist.dist.get_world_size(pg)
        # TODO: support uneven splits by accepting explicit input/output
        # sizes, like the underlying dist.all_to_all collective supports.
        if x.shape[split_dim] % world_size != 0:
            raise ValueError(
                f"all_to_all: tensor dimension {split_dim} (size {x.shape[split_dim]}) "
                f"is not evenly divisible by world_size ({world_size})"
            )
        input_chunks = [
            c.contiguous() for c in torch.chunk(x, world_size, dim=split_dim)
        ]
        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
        _dist.dist.all_to_all(output_chunks, input_chunks, group=pg)
        return torch.cat(output_chunks, dim=concat_dim)

    @staticmethod
    def backward(ctx, grad_out):
        # Backward reverses: split on old concat_dim, concat on old split_dim.
        grad = all_to_all(
            grad_out,
            ctx.axis,
            src=Shard(ctx.split_dim),
            dst=Shard(ctx.concat_dim),
        )
        return grad, None, None, None


@api_boundary
def all_to_all(
    x,
    axis: ProcessGroup,
    *,
    src: PerMeshAxisSpmdType = V,
    dst: PerMeshAxisSpmdType = V,
    split_dim: int | None = None,
    concat_dim: int | None = None,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
):
    """``all_to_all(x, mesh_axis, src, dst): Varying -> Varying``

    An all-to-all transposes a mesh axis with a tensor axis.

    ::

        def all_to_all_spec(xs, src, dst):
            match src, dst:
                case V, V:
                    x = torch.stack(xs)
                    x = x.transpose(0, 1)
                    return x.unbind()

                case S(i), S(j):
                    x = torch.concat(xs, i)
                    return x.chunk(mesh_axis_size, j)

    The varying and shard versions of ``all_to_all`` are pretty different (even
    though under the hood they both have an all-to-all communication pattern),
    so we describe them separately.

    * ``all_to_all(V,V)`` transposes the logical mesh axis with the dim 0 local
      tensor axis.

    * ``all_to_all(S(i),S(j))`` intuitively unshards the tensor on dim i, and
      then reshards it on dim j (but skips actually doing the all-gather).

    ::

        Diagram for all_to_all(V,V):
        [A0, A1, A2]      [A0, B0, C0]
        [B0, B1, B2]  =>  [A1, B1, C1]
        [C0, C1, C2]      [A2, B2, C2]

    Args:
        x: Input tensor with V or S(i) type on the mesh axis
        axis: The mesh axis to transpose with (ProcessGroup)
        src: Source type (V or S(i))
        dst: Destination type (V or S(j))
        split_dim: The tensor dimension to split along. Defaults to 0 when
            src is V; inferred from the shard dim when src is S(i). If both
            split_dim and src=S(i) are provided, they must agree.
        concat_dim: The tensor dimension to concatenate along. Defaults to 0
            when dst is V; inferred from the shard dim when dst is S(j). If
            both concat_dim and dst=S(j) are provided, they must agree.
        output_split_sizes: Per-rank output sizes for uneven all-to-all-v.
            Only supported with src=S(0), dst=S(0). If None, defaults to
            even splits. Length must equal world_size.
        input_split_sizes: Per-rank input sizes for uneven all-to-all-v.
            Only supported with src=S(0), dst=S(0). If None, defaults to
            even splits. Length must equal world_size and sum must equal
            x.shape[0].

    Returns:
        Tensor with V or S(j) type depending on dst

    **Backward cases:**

    The forwards is ``V -> V``, the backwards is ``all_to_all: V -> V`` (with
    src/dst and split_dim/concat_dim swapped).

    ``all_to_all(V,V): V -> V``, the backwards is ``all_to_all(V,V): V -> V``::

        [A0, A1, A2]      [A0, B0, C0]
        [B0, B1, B2]  <=  [A1, B1, C1]
        [C0, C1, C2]      [A2, B2, C2]

    ``all_to_all(S(i),S(j)): S(i) -> S(j)``, the backwards is
    ``all_to_all(S(j),S(i)): S(j) -> S(i)``.
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            all_to_all,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
            split_dim=split_dim,
            concat_dim=concat_dim,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )
    # Canonicalize negative Shard dims
    src = _canonicalize_shard(src, x.ndim)
    dst = _canonicalize_shard(dst, x.ndim)

    # Validate src and dst are V or S(i)
    if not (src is V or isinstance(src, Shard)):
        raise ValueError(f"all_to_all src must be V or S(i), got {src}")
    if not (dst is V or isinstance(dst, Shard)):
        raise ValueError(f"all_to_all dst must be V or S(i), got {dst}")

    if isinstance(src, Shard) and isinstance(dst, Shard):
        # S(i) -> S(j): physically split on new shard dim (j), concat on old (i).
        # Conceptual spec: concat(xs, i) then chunk(ws, j).
        if split_dim is not None and split_dim != dst.dim:
            raise ValueError(
                f"all_to_all S({src.dim})->S({dst.dim}) got split_dim={split_dim} "
                f"but expected {dst.dim} (dst dim)."
            )
        if concat_dim is not None and concat_dim != src.dim:
            raise ValueError(
                f"all_to_all S({src.dim})->S({dst.dim}) got concat_dim={concat_dim} "
                f"but expected {src.dim} (src dim)."
            )
        split_dim = dst.dim
        concat_dim = src.dim
    elif isinstance(src, Shard):
        if split_dim is not None and split_dim != src.dim:
            raise ValueError(
                f"all_to_all got split_dim={split_dim} but src=S({src.dim}); "
                f"these conflict. Either use src=S({split_dim}) or remove split_dim."
            )
        split_dim = src.dim
        if concat_dim is None:
            concat_dim = 0
    elif isinstance(dst, Shard):
        if concat_dim is not None and concat_dim != dst.dim:
            raise ValueError(
                f"all_to_all got concat_dim={concat_dim} but dst=S({dst.dim}); "
                f"these conflict. Either use dst=S({concat_dim}) or remove concat_dim."
            )
        concat_dim = dst.dim
        if split_dim is None:
            split_dim = 0
    else:
        if split_dim is None:
            split_dim = 0
        if concat_dim is None:
            concat_dim = 0

    if output_split_sizes is not None or input_split_sizes is not None:
        if split_dim != 0 or concat_dim != 0:
            raise ValueError(
                f"output_split_sizes/input_split_sizes only supported with "
                f"src=S(0), dst=S(0), got src={src}, dst={dst}"
            )
        return _AllToAllUneven.apply(x, axis, output_split_sizes, input_split_sizes)

    if src is V and dst is V:
        return _AllToAllStack.apply(x, axis, split_dim, concat_dim)
    else:
        return _AllToAllShard.apply(x, axis, split_dim, concat_dim)


# =============================================================================
# redistribute: semantics-preserving type change with comms
# =============================================================================


@api_boundary
def redistribute(  # noqa: C901
    x,
    axis: ProcessGroup,
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
):
    """Semantics-preserving type change that allows comms.

    It is helpful to have a version of ``convert`` that is semantics preserving
    but allows for comms.  ``redistribute`` routes to the following
    collectives::

        redistribute(S(i),R)    =   all_gather(S(i),R)
        redistribute(S(i),I)    =   all_gather(S(i),I)
        redistribute(P,R)       =   all_reduce(P,R)
        redistribute(P,I)       =   all_reduce(P,I)
        redistribute(P,S(i))    =   reduce_scatter(P,S(i))
        redistribute(S(i),S(j)) =   all_to_all(S(i),S(j))

    These only work if the mesh axis is the LAST to shard a particular tensor
    dimension.

    For conversions that don't require comms (R<->I, R->V, R->P, I->V, I->P,
    V->P), this function delegates to ``convert`` or ``reinterpret`` as
    appropriate.

    Args:
        x: Input tensor
        axis: The mesh axis to operate on (ProcessGroup)
        src: Source local SPMD type
        dst: Destination local SPMD type
    """
    if has_torch_function_unary(x):
        return handle_torch_function(
            redistribute,
            (x,),
            x,
            axis,
            src=src,
            dst=dst,
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

    # Normalize to base types for dispatch
    src_is_shard = isinstance(src, Shard)
    dst_is_shard = isinstance(dst, Shard)
    src_base = V if src_is_shard else src
    dst_base = V if dst_is_shard else dst

    if src_base is dst_base:
        if src_is_shard and dst_is_shard and src.dim != dst.dim:
            # S(i) -> S(j): need all_to_all
            return all_to_all(x, axis, src=src, dst=dst)
        return x  # no-op

    # Varying/Shard -> Replicate: all_gather
    if src_base is V and dst_base is R:
        return all_gather(x, axis, src=src, dst=R)

    # Varying/Shard -> Invariant: all_gather
    if src_base is V and dst_base is I:
        return all_gather(x, axis, src=src, dst=I)

    # Partial -> Replicate: all_reduce
    if src_base is P and dst_base is R:
        return all_reduce(x, axis, src=P, dst=R)

    # Partial -> Invariant: all_reduce
    if src_base is P and dst_base is I:
        return all_reduce(x, axis, src=P, dst=I)

    # Partial -> Varying/Shard: reduce_scatter
    if src_base is P and dst_base is V:
        return reduce_scatter(x, axis, src=P, dst=dst, scatter_dim=dim)

    # For non-comm conversions, delegate to convert
    # R -> I, I -> R, R -> V, R -> P, I -> V, I -> P, V -> P
    if src_base is R and dst_base is I:
        return convert(x, axis, src=R, dst=I, expert_mode=True)
    if src_base is I and dst_base is R:
        return convert(x, axis, src=I, dst=R, expert_mode=True)
    if src_base is R and dst_base is V:
        return convert(x, axis, src=R, dst=dst, expert_mode=True)
    if src_base is R and dst_base is P:
        return convert(x, axis, src=R, dst=P, expert_mode=True)
    if src_base is I and dst_base is V:
        return convert(x, axis, src=I, dst=dst, expert_mode=True)
    if src_base is I and dst_base is P:
        return convert(x, axis, src=I, dst=P, expert_mode=True)
    if src_base is V and dst_base is P:
        return convert(x, axis, src=src, dst=P, expert_mode=True)

    raise ValueError(f"redistribute({src}, {dst}) is not supported.")


def unshard(
    x,
    axis: ProcessGroup,
    *,
    src: PerMeshAxisSpmdType,
    dst: PerMeshAxisSpmdType,
):
    """Convenience alias: ``unshard(S(i), dst)`` is ``all_gather(S(i), dst)``.

    Gathers a sharded tensor along the mesh axis so every rank has the full
    copy.

    Args:
        x: Input tensor with S(i) type on the mesh axis
        axis: The mesh axis to gather over (ProcessGroup)
        src: Source shard type, must be S(i)
        dst: Destination type (R or I)
    """
    if not isinstance(src, Shard):
        raise ValueError(f"unshard src must be S(i), got {src}")
    return all_gather(x, axis, src=src, dst=dst)
