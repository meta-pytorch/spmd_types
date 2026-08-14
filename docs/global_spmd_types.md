# Global SPMD types

Global SPMD types augment local SPMD types with a `PartitionSpec` that
describes how varying mesh axes map to tensor dimensions, giving a precise
global interpretation of sharded tensors.  See
[local SPMD types](local_spmd_types.md) for the underlying type system.

## Partition Spec

Varying mesh axes can be omitted from the `local_spmd_type` dictionary.  When
you omit them, you can optionally provide a `partition_spec` to describe which
tensor dimensions are sharded on which mesh axes, giving global SPMD semantics.

A `PartitionSpec` is a tuple with one entry per tensor dimension.  Each entry
is either `None` (not sharded on this axis), a mesh axis (sharded on that
axis), or a tuple of mesh axes (sharded on multiple axes).

If a mesh axis is R, I, or P, you **must** specify it in `local_spmd_type`
even if it could be inferred from `partition_spec`, because it would be
ambiguous whether it is Replicate or Invariant.

If a mesh axis is neither mentioned in `local_spmd_type` (as R/I/P) nor in
`partition_spec`, it is assumed to be Varying **without** a global SPMD type
(i.e., purely local SPMD -- the sharding may be irregular or otherwise not
expressible via `PartitionSpec`).

As a shortcut, you can use `S(i)` in the `local_spmd_type` dictionary to mean
"this mesh axis shards tensor dimension i", which is equivalent to specifying
a `partition_spec` entry.  However, we reject `S(i)` if the same tensor
dimension is sharded by multiple mesh axes, since the sharding order would be
ambiguous; use an explicit `PartitionSpec` in that case.

Here are some examples with mesh axes `dp` and `tp`:

```python
from spmd_types import PartitionSpec

# input is [batch@dp, seq, hidden@tp]
spmd.assert_type(x, {}, PartitionSpec(dp, None, tp))
# or, using the per-mesh S(i) sugar:
spmd.assert_type(x, {dp: spmd.S(0), tp: spmd.S(2)})

# prior to colwise linear, input is [batch@dp, seq, in_feature]
# tp axis is Invariant (not sharded), must be explicit
spmd.assert_type(x, {tp: spmd.I}, PartitionSpec(dp, None, None))

# input, but dp is irregularly sharded (not expressible in global SPMD)
spmd.assert_type(x, {}, PartitionSpec(None, None, tp))
# or, being explicit about dp being Varying without a global type:
spmd.assert_type(x, {dp: spmd.V}, PartitionSpec(None, None, tp))
```

## Local matrix blocks

Optimizers such as Muon apply matrix operations independently to complete
local blocks. Validate that a globally annotated plain tensor is only sharded
on leading dimensions before entering that computation:

```python
param = spmd.assert_local_block(param)  # final two dimensions are unsharded
```

Set `trailing_dims` for other block ranks. The assertion rejects Partial
tensors, local-only `V` annotations without a `PartitionSpec`, and shards on
any protected trailing dimension.

## Storage-to-compute views

Use `dtensor_compute_view` when a local computation requires a different
placement from persistent DTensor storage:

```python
with spmd.dtensor_compute_view(
    grad,
    placements=compute_placements,
    writeback=True,
) as local_grad:
    local_grad.mul_(precomputed_clip_scale)
```

The context owns placement redistribution and writeback but preserves the
physical tensor shape. Any logical reshape required by a particular algorithm
belongs to that algorithm's wrapper, outside this API.

The caller still owns the computation's global semantics. For example,
gradient clipping must compute `precomputed_clip_scale` using a globally
correct norm, either by gathering gradients or reducing sharded norm
contributions before entering the mutable scaling region.
