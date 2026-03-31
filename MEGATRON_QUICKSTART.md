# Megatron Quick Start

This quick start guide is to help developers working on Megatron-derived
training frameworks quickly adapt their code to use `spmd_types`.  It will
only cover local SPMD types.  For a more theoretical and complete discussion of
the design, see `DESIGN.md`.

## Why use this?

Have you ever:

* Forgotten to call `copy_to_tensor_model_parallel_region` when you should
  have?

* Accidentally all-reduced a tensor too many times because you forgot that
  you had already reduced it?

* Forgotten that you had a pending reduction and used a non-linear function on
  a tensor without all-reducing first?

* Gotten annoyed at how slow E2E testing that a model gives the same results
  with no results compared to with DP+CP+TP, because you can only test
  numerics in a multi-process setup?

`spmd_types` provides an optional type system that keeps track of whether or
not your data (and gradients) are varying or have pending reductions, which
are able to catch all of these bugs without needing to do any numerics testing
at all.  Annotate your module inputs and parameters with their SPMD types, and
then run your code with type checking on to verify that you haven't made
mistakes that would result in incorrect gradients.

## Mesh Axes

A **mesh axis** identifies one dimension of the device mesh -- essentially,
"which ranks participate and in what order."  In common use, you just pass
the same `ProcessGroup` you were already using for collectives.  `spmd_types`
accepts a `ProcessGroup` anywhere a mesh axis is expected and normalizes it
into a lightweight `MeshAxis` internally (which only remembers which ranks are
in the group, not PG-specific state like `pg_options` or the communicator).
Two process groups over the same ranks produce the same `MeshAxis`.

In code examples below, variables like `tp` and `dp` are the process groups
you already have for your tensor-parallel and data-parallel collectives.

## Types

Every tensor has a local SPMD type.  You can set it with `assert_type`,
and when typechecking is enabled using the `typecheck()` context manager,
we will propagate them through PyTorch functions.

```python
import spmd_types as spmd
from spmd_types import PartitionSpec

t = torch.ones(20)
print(spmd.get_local_type(t))  # {}, no type specified yet
spmd.assert_type(t, {tp: spmd.I})
print(spmd.get_local_type(t))  # {tp: spmd.I}
with spmd.typecheck():
    print(spmd.get_local_type(t * 4))  # {tp: spmd.I}
spmd.assert_type(t, {tp: spmd.R})  # Error, because type is inconsistent!
```

The full signature is:

```python
spmd.assert_type(tensor, local_spmd_type, partition_spec=None)
```

A local SPMD type is a dictionary from mesh axes to a per-axis local
type: Replicate (R), Invariant (I), Varying (V) or Partial (P).  These types
describe how the tensor is distributed over the mesh axis:

* Varying means you have differing values across the axis (e.g., the tensor is
  sharded across that axis).  The gradient of Varying is Varying.

* Partial means you have a pending reduction across the axis (e.g., you did a
  contraction on a sharded dimension).  The gradient of Partial is Replicate.

* Replicate and Invariant mean you have the same values across the axis (e.g.,
  a parameter, or a non-sharded quantity).  Replicate and Invariant differ in
  their backwards behavior: the gradients of a Replicate tensor are Partial,
  while gradients of an Invariant tensor are Invariant.  (How can you figure
  out which one to use? More on this shortly.)

The local types of gradients are summarized by this handy table:

```text
Forward     Backward
----------------------
Replicate   Partial
Partial     Replicate
Invariant   Invariant
Varying     Varying
```

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

## Operators

`spmd_types` provides its own versions of distributed collectives and local
operations which interact with the types in a non-trivial way.  An easy way to
get started is to see [which function from Megatron](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py) you are using, and then use the corresponding function
from our API:

```text
Megatron function                                       spmd_types function
---------------------------------------------------------------------------------------------------------
copy_to_tensor_model_parallel_region(x)                 spmd.convert       (x, tp, src=spmd.I,      dst=spmd.R)
reduce_from_tensor_model_parallel_region(x)             spmd.all_reduce    (x, tp,                  dst=spmd.I)
scatter_to_tensor_model_parallel_region(x)              spmd.reduce_scatter(x, tp,                  dst=spmd.S(-1))
gather_from_tensor_model_parallel_region(x)             spmd.all_gather    (x, tp, src=spmd.S(-1),  dst=spmd.R)
scatter_to_sequence_parallel_region(x)                  spmd.convert       (x, tp, src=spmd.I,      dst=spmd.S(0))
gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True)
                                                        spmd.all_gather    (x, tp, src=spmd.S(0),   dst=spmd.R)
gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False)
                                                        spmd.all_gather    (x, tp, src=spmd.S(0),   dst=spmd.I)
reduce_scatter_to_sequence_parallel_region(x)           spmd.reduce_scatter(x, tp,                  dst=spmd.S(0))
all_gather_last_dim_from_tensor_parallel_region(x)      spmd.all_gather    (x, tp, src=spmd.S(-1),  dst=spmd.R)
reduce_scatter_last_dim_to_tensor_parallel_region(x)    spmd.reduce_scatter(x, tp,                  dst=spmd.S(-1))
all_to_all(group, x)                                    spmd.all_to_all    (x, group)
```

In the table above, `tp` is the tensor-parallel mesh axis.
Instead of having an API for each variation of forward-backward combination, we
instead provide a unified API with a function per collective.  We distinguish
between which autograd function is desired via the new src/dst arguments to our
collectives, which tell you what the input and output local spmd type of the
function is on the particular specified mesh axis.  Some other things of note:

* We omit `src` on the reduction APIs (`all_reduce`, `reduce_scatter`),
  because these default to Partial (`spmd.P`).  If the input is actually
  Varying, the V->P reinterpret is done implicitly

* If you pass `Shard(i)` (abbreviated as `S(i)`) to src/dst, this means that,
  not only is the tensor varying over a mesh axis, but it is semantically
  (according to global SPMD) sharded on tensor axis i, and so we wish to do a
  concat/split on that particular tensor dimension.  If you pass Varying,
  instead, we will do a stack/unbind on dimension 0.

* `spmd.reinterpret` and `spmd.convert` are not collectives, but instead
  type "conversion" functions that change the local SPMD type of their tensor.
  A `reinterpret` is guaranteed to do no work (it keeps the local data exactly
  exactly as is, unchanged), while a `convert` is guaranteed to be global SPMD
  semantics preserving (e.g., a tensor that sharded on tensor dim 0, is
  semantically equivalent to the full tensor you get by concatenating all the
  shards on dim 0).  TODO: maybe Megatron style alias (copy/scatter) might be
  intuitive for people; unfortunately, scatter could also denote a comms, so
  it's not a great name.

Although it can be somewhat counter-intuitive to directly deal with these new
functions (in particular, if you forked Megatron-LM, we suggest you keep the old
functions and replace their implementations with calls to our API), the explicit
types in the new API are important for understanding the local SPMD types of
your tensors.  Assuming that your original training code wasn't buggy, if you
see a call to `reduce_from_tensor_model_parallel_region`, you know that your
output tensor is invariant--and more importantly, you know that its gradient is
invariant (not partial!)

## Propagation rules

How do types propagate through your program on operators that are not the
collectives listed above?  The rules are fairly simple:

```text
op(Replicate..) -> Replicate
op(Invariant..) -> Invariant  # NB: this case is uncommon
op(Varying..) -> Varying
linear_op(Partial) -> Partial

# Invariant not allowed to mix with other types
op(Replicate, Varying) -> Varying
Partial + Partial -> Partial
```

To summarize:

* You usually need to convert Invariant into Replicate before you
  can do any compute with it (unless you are doing compute entirely with
  Invariant)

* Replicate and Varying can mix freely without requiring extra conversions;
  if any input is Varying, the output is Varying

* You can only propagate Partial through linear functions.

## Generating Partial values

The reduction collectives (`all_reduce`, `reduce_scatter`) nominally require
the input to be Partial, but for convenience they also accept `src=spmd.V`
and will implicitly reinterpret the varying tensor as partial before reducing.
So in most cases, you can simply pass your varying tensor directly:

```python
# These are equivalent:
out = spmd.all_reduce(x, tp, src=spmd.V, dst=spmd.R)
out = spmd.all_reduce(spmd.reinterpret(x, tp, src=spmd.V, dst=spmd.P), tp, dst=spmd.R)

out = spmd.reduce_scatter(x, tp, src=spmd.V, dst=spmd.S(0))
out = spmd.reduce_scatter(spmd.reinterpret(x, tp, src=spmd.V, dst=spmd.P), tp, dst=spmd.S(0))
```

You can still produce an explicit partial tensor earlier with
`spmd.reinterpret(x, pg, src=spmd.V, dst=spmd.P)` if you want to.  In local
SPMD this can be useful for catching non-linear ops on partial tensors sooner.
In global SPMD, producing partial at the point of contraction (via
`out_partial_axes`) is required for the operation to be well-typed.

A strategy that works better with global SPMD types is to explicitly annotate
a local operation (e.g., a sum, a linear or an einsum) as producing a partial
output.  For example, when you sum over a sharded
dimension, in local SPMD semantics you would do a per-rank summation; in
global SPMD semantics, you still need to do a summation across all the ranks.
You can clearly specify that you meant to do the full summation (but are
delaying it) with:

```python
spmd.sum(x, pg, src=spmd.S(0), dst=spmd.P)
```

This function says that the input tensor is sharded on tensor dim 0 across pg,
and so to do a *global* summation over this dimension, we will do the local
summation and declare a pending summation over this pg as well.

The following operations are supported by spmd:

```python
spmd.sum(x, pg, src, dst)
spmd.linear(x, w, pg, src, dst)
spmd.einsum(equation, *args, pg, src, dst)
```

It is also supported to provide tuples for pg/src/dst, indicating that there
are multiple mesh axes which are producing pending reductions, e.g.,

```python
spmd.sum(x, (ax1, ax2), src=(spmd.S(0), spmd.S(1)), dst=(spmd.P, spmd.P))
```

TODO: These functions are not implemented yet

TODO: spmd.P is very "explicit", there's probably a shorter form, is
uniformity better?

TODO: Is three tuples good, or you want to keep `ax1, src=spmd.S(0),
dst=spmd.P` together?

## Advice about Invariant vs Replicate

Although Megatron contains many functions for working with the Invariant type,
it is actually not recommended: in general, you want to have Replicate instead
in your forwards, so that you can delay all-reduce as long as possible, having
Invariant only on parameters.  However, we recommend porting an existing
training codebase as-is, applying types accurately for what it does at the
moment, before considering refactors that take advantage of the type checking
to verify correctness.

## Forwards/Backwards

Here is a table that summarizes the forward-backward relationships between the
operators in our API (abbreviating the function calls to only include src/dst
when it would disambiguate between multiple different operators).  We've
separated the less efficient invariant conversions from the rest.

```text
Fwd Type    Forward                 Bwd Type    Backward
----------------------------------------------------------------------------
R -> V      convert(R,V)            V -> P      convert(V,P)
R -> P      convert(R,P)            R -> P      convert(R,P)
I -> R      convert(I,R)            P -> I      all_reduce(I)
V -> R      all_gather(R)           P -> V      reduce_scatter()
V -> V      all_to_all()            V -> V      all_to_all()
V -> R      all_reduce(src=V,R)     P -> R      all_reduce(R)
V -> V      reduce_scatter(src=V)   V -> R      all_gather(R)
P -> R      all_reduce(R)           P -> R      all_reduce(R)
P -> V      reduce_scatter()        V -> R      all_gather(R)
----------------------------------------------------------------------------
P -> I      all_reduce(I)           I -> R      convert(I,R)
V -> I      all_gather(I)           I -> V      convert(I,V)
I -> V      convert(I,V)            V -> I      all_gather(I)
----------------------------------------------------------------------------
```
