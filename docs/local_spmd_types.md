# Local SPMD types

Local SPMD types are a lightweight type system which enforce correct usage of
differentiable collectives without requiring an E2E loss validation run.

## Motivating example

The biggest source of bugs related to differentiable collectives arise from
tensor parallelism.  Megatron-like training frameworks have custom autograd
functions that support differentiating through the tensor parallel region.
For example, Megatron-LM defines these functions in
[`mappings.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py).
The simplest pair, used in the non-sequence-parallel case, is
`_CopyToModelParallelRegion` and `_ReduceFromModelParallelRegion`:

```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_                                    # no-op!
    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None     # all-reduce

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""
    @staticmethod
    def forward(ctx, input_, group):
        return _reduce(input_, group)                    # all-reduce
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None                         # no-op!
```

The problem: `copy_to_tensor_model_parallel_region` is a **no-op in the
forward pass**.  If you forget it, the forward pass produces identical results
-- but the backward pass silently computes wrong gradients because the
all-reduce on the activation gradient is missing.  Each rank only sees its
local partial gradient, not the sum across all ranks.

Local SPMD types has two aims:

- It wants to catch errors related to use of differentiable collectives, without
  requiring you to do a full E2E training run to see if the loss curve looks good.

- The parallelism-relevant calling convention between modules can be
  complicated to describe in its own right (e.g., is the gradient reduced or
  not?): local SPMD types provides a language by which we can easily describe
  what expectations a module has on its inputs/outputs/gradients.

## Basics

To typecheck your model, there are three things you must do:

1. You have to specify your compute mesh with `spmd.set_current_mesh`,
   see key concepts for a discussion about what a compute mesh is (TODO: link)

2. You must annotate all input tensors and parameters with appropriate local
   SPMD types.  There is some interaction with distributed module wrappers
   like FSDP, which is discussed in more detail in Examples.

3. You need to ensure all collectives called inside your model are
   typecheckable, either by switching to `spmd_types` built-in collectives or
   using our custom autograd function annotation to make your existing
   collectives typecheckable.

4. You run the forward (backward not needed) of your model under the
   `spmd_types.checker.typecheck(local=True)` context manager (`local=True`
   indicates that you only want local SPMD type checking.)

### Types

A local SPMD type describes how the tensor and its gradient are distributed
over the mesh axis.  There are four distinct per-axis types: Replicate (R),
Invariant (I), Varying (V) or Partial (P):

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

The local SPMD type for any tensor can be set using `spmd.assert_type`.
Assuming an appropriate `set_current_mesh` has been set, the idiomatic usage
of `spmd.assert_type` is to give it a dictionary of mesh axes names to
per-axis type, e.g.:

```
spmd.assert_type(t, {"tp": spmd.I})
```

In order to do typechecking, you must specify local SPMD types on all
parameters and inputs to your model.  We also think it's good practice to
assert types on inputs/outputs to modules, in the same way it's helpful to
write explicit type signatures on top-level functions.  For more
guidance on how exactly you should add these annotations, check the Examples
section.

### Operators

`spmd_types` provides its own versions of distributed collectives and local
operations which interact with the types in a non-trivial way.  You can
read the full API reference at TODO.  Here are some high level conceptual
ideas to keep in mind while reading the API reference:

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
  shards on dim 0).  We also have a dedicated `spmd.invariant_to_replicate`
  for one particularly common `spmd.reinterpret` call.

If you are coming from a Megatron-based framework, this table maps [Megatron
APIs](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py)
to the corresponding function in our API:

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

You'll notice that instead of having an API for each variation of
forward-backward combination, we instead provide a unified API with a function
per collective.  We distinguish between which autograd function is desired via
the new src/dst arguments to our collectives, which tell you what the input
and output local spmd type of the function is on the particular specified mesh
axis.

Although it can be somewhat counter-intuitive to directly deal with these new
functions (in particular, if you forked Megatron-LM, we suggest you keep the old
functions and replace their implementations with calls to our API), the explicit
types in the new API are important for understanding the local SPMD types of
your tensors.  Assuming that your original training code wasn't buggy, if you
see a call to `reduce_from_tensor_model_parallel_region`, you know that your
output tensor is invariant--and more importantly, you know that its gradient is
invariant (not partial!)

### Custom autograd functions

TODO: write

### Propagation rules

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

### Generating Partial values

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

### Type checking

We typecheck a model for local SPMD types by running its forward pass with
typechecking enabled.  Every tensor is annotated with a local SPMD type, and
typechecking propagates the annotations through the execution of the model.
This means you must also annotate the inputs and parameters

```
from spmd_types.checker import typecheck

with typecheck(local=True):
    loss = model(input)
```

## Examples

Here, we'll go through some examples of how to add local SPMD types to common
parallelisms found in LLM training codebases.  One recurring theme is that
some of the most interesting local SPMD typing problems emerge from the
"calling conventions" across modules: e.g., when a module needs to coordinate
with some other module or distributed wrapper about whether or not its weight
gradients are all-reduced.  One of the primary values of local SPMD types is
to help detect when two modules are individually correct but disagree about
the types at the interface boundary.

### Data/context parallelism

The calling convention for data and context parallelism is quite simple: it is
essentially universally understood that all PyTorch modules take inputs that
are sharded on the batch/context dimension `{"dp": spmd.V, "cp": spmd.V}` and
will compute gradients parameters with a pending reduction on the DP/CP mesh
axes `{"dp": spmd.R, "cp": spmd.R}`.  That is to say, a module never takes
care of this reduction by itself; instead, there is typically some out-of-band
mechanism (e.g., a backward hook registered by FSDP2) that actually takes care
of this reduction.  This is not a coincidence: the enduring stickiness of
FSDP-style approaches to parallelism is that they work on code written for a
single GPU without modification.

This suggests a specific strategy for perform local SPMD types annotations for
DP/CP, in the absence of a distributed wrapper like FSDP:

- Because every parameter essentially universally has the same local SPMD
  type, the most convenient thing to do is to have a helper that traverses
  all parameters and runs `spmd.assert_type(param, {"dp": spmd.R, "cp": spmd.R})` on them.

- Buffers don't meaningfully distinguish between `spmd.R` and `spmd.I`, as they
  don't accept gradients, but it's usually more convenient to annotate them as
  `spmd.R` for symmetry with parameters.

- Expert parallelism changes the degree of data parallelism (i.e., a separate
  EDP mesh axis distinct from DP), so something special has to happen for
  expert parallelism!

A default helper that doesn't handle expert parallelism might look like this:

```
def assert_dp_cp_params(module: nn.Module) -> None:
    for tensor in itertools.chain(m.parameters(), m.buffers()):
        spmd.assert_type(tensor, {"dp": spmd.R, "cp": spmd.R})
```

Here's the typical flow of local SPMD types for DP/CP; the inputs/outputs are
varying and the weights are replicated:

```
def forward(self, x):
    spmd.assert_type(x,           {"dp": spmd.V, "cp": spmd.V})
    spmd.assert_type(self.weight, {"dp": spmd.R, "cp": spmd.R})

    out = F.linear(x, self.weight)
    spmd.assert_type(out,         {"dp": spmd.V, "cp": spmd.V})

    return out
```

These asserts are just illustrative: because the DP/CP types are so formulaic,
we don't actually recommend putting these asserts in your code (with an
exception for CP-aware modules, like context-parallel attention, where you may
specifically require a particular sharding on input tensors for context
parallelism).

#### FSDP

Ideally, this section would simply be "PyTorch FSDP2 does the right
thing with local SPMD types, you don't have to think about it."  But
unfortunately, (1) PyTorch FSDP2 does not currently support local SPMD types
(in progress at https://github.com/pytorch/pytorch/pull/181519/ ), but also
(2) there are many forks of FSDP, and there needs to be a description of how
to apply local SPMD types to your fork.

Fortunately, it is simpler than you might think.  Superficially, it would seem
that FSDP requires some work: you will be typechecking your module, it will
enter FSDP code, and you will error because something in FSDP doesn't know how
to propagate local SPMD types.  Here, however, we assert an important guiding
principle: FSDP's distributed storage handling is *independent* of local SPMD
types.  This is because a typical FSDP implementation does not make use of
differentiable collectives; instead, it is transparently instrumenting the
all-gathers and reduce-scatters necessary to make your original single GPU
code work without modification.  So there isn't any benefit from attempting to
do local SPMD types in FSDP's internals.  (Additionally, it's also often
quite confusing to do so, because FSDP operates on the model mesh while
your SPMD type checking is operating on the compute mesh -- TODO link.)

What you should do is:

- Liberally apply `spmd.no_typecheck` onto everything that can get called into
  during forwards.

- Ensure that the all-gathered parameter that is put on the module and to be
  used during compute has a correct local SPMD type.

How exactly do you determine the "correct local SPMD type"?  For example, if
the original module (prior to FSDP being applied to it) had local SPMD types
applied to it, should FSDP preserve these types?  For DP/CP, the answer is
simple: the original types should be consistent with the types that FSDP wants
to put on these axes--and FSDP is going to want `spmd.R` on DP and CP, because
those are the axes it's going to do gradient reduce-scatters on.

For the most part, your choice here doesn't make a big difference (I suppose
you could potentially catch an error where you didn't configure FSDP to
include the CP axis in the reduce-scatter--you would notice that FSDP was
specifying the local SPMD type should be `"cp": spmd.I` whereas the model code
was written assuming `"cp": spmd.R`, but you should always have FSDP take care
of the CP reduce-scatter, so this distinction doesn't really matter in
practice.)  (By the way, this question is going to become more complicated for
TP, so keep your eyes peeled.)

Now, let's discuss some variations on the theme.

**FSDP on DTensor parameters.**  PyTorch FSDP2 supports applying `fully_shard`
on modules that already have DTensor parameters; inside the module body, the
parameters continue to be DTensor (even if they are all replicated).  This is
most commonly done to handle tensor parallel parameters, but there's nothing
stopping you from having a DP/CP only model have all your parameters as
DTensor.

This situation is actually quite simple!  If your entire model does compute
entirely with DTensor, there is no point in local SPMD types; DTensor subsumes
local SPMD types in the sense that if you write a program entirely with
DTensor, you are also guaranteed to get correct gradients.  You could also
unwrap the DTensor into a plain tensor immediately inside your module body (if
you are patching FSDP, you may have patched FSDP to automatically unwrap the
DTensor in this case): during this unwrapping process, it is necessary to
ascribe local SPMD types to the tensor.  DTensor Replicate will translate into
`spmd.R` by default, and this is the correct choice for FSDP2, as FSDP2 will
unconditionally run a reduce-scatter on all parameters it knows about,
regardless of if the gradient happened to already have been replicated on the
DP axis.

**SimpleFSDP.**  SimpleFSDP is an extremely simple implementation of FSDP that
doesn't handle prefetching or bucketing at all; instead, you simply
differentiably all-gather your parameters right when you need them.  Because
the backwards of all-gather is reduce-scatter, this sets up all the comms you
need, albeit inefficiently.  SimpleFSDP then relies on a compiler to perform
bucketing and prefetching for performance.

Today's implementation of SimpleFSDP utilizes DTensor (mooting the question of
local SPMD typing), but it is not difficult to imagine a variant of SimpleFSDP
that operates entirely on plain tensors.  Because a differentiable collective
is used here, it is relatively simple to local SPMD typecheck through the
all-gather: your parameters are stored as `{"dp": spmd.V, "cp": spmd.V}`, and
the all-gather takes them to `{"dp": spmd.R, "cp": spmd.R}` as before.

Is this a violation of our philosophical principle above (local SPMD types is
independent of FSDP)?  Probably.  But because SimpleFSDP directly uses
differentiable collectives, I think it's fine if you want to mix them up.
If your model mesh and data mesh diverge, you can still make it work by having
the FSDP collectives operate on a different mesh (this works.)  The
philosophical claim above is mostly necessitated by the fact that distributed
wrappers like FSDP don't use differentiable collectives, and instead do
the all-gather and reduce-scatter by out-of-band mechanisms that don't look
like "compute".  It's tough to make only compute performant, but a compiler
can help, and maybe you do want to experiment with weird schemes beyond what
regular eager FSDP can support for you.

### Tensor parallelism

Tensor parallelism poses several distinct calling convention questions:
sequence parallel or not?  If sequence parallel, do you have module or wrapper
responsibility for grad all-reduce for TP replicated weights?  What about FSDP?

#### No sequence parallelism

Sequence parallelism avoids wasted compute in the regions of code that don't
directly perform tensor parallelism, by utilizing the TP mesh axis to shard on
sequence.  However, it is a valid (if wasteful) choice to run without sequence
parallelism (for example, the original Megatron paper described parallelism in
this way), and it's instructive to see how local SPMD types can express this.

In the classic Megatron formulation, when you have TP without sequence
parallel, in the non-TP regions you are running exactly the same compute across
the TP axis: we indicate that this has occurred by having `spmd.I` on the TP
axis of modules outside of the TP region.

**Non-TP region module.**  Consider an RMSNorm outside of the TP region.  It
will have the following types:

```
def forward(self, input: torch.Tensor) -> torch.Tensor:
    spmd.assert_type(input,       {"tp": spmd.I})
    spmd.assert_type(self.weight, {"tp": spmd.I})

    out = rms_norm(input, self.weight)
    spmd.assert_type(out,         {"tp": spmd.I})

    return out
```

Most other modules outside the TP region will be typed similarly.

**Col-parallel linear.**   The column-parallel linear moves us into the TP
region: so we transition from being `"tp": spmd.I` to `"tp": spmd.V`.  A
backwards collective must be introduced to ensure gradients are computed in
lock-step in the non-TP region.  Here is what the types look like here:

```
def forward(self, x):
    spmd.assert_type(x,           {"tp": spmd.I})
    spmd.assert_type(self.weight, {"tp": spmd.V})

    x = spmd.invariant_to_replicate(x, tp_group)
    spmd.assert_type(x,           {"tp": spmd.R})

    out = F.linear(x, self.weight)
    spmd.assert_type(out,         {"tp": spmd.V})

    return out
```

**Row-parallel linear.**  Conversely, the row-parallel linear moves us out
of the TP region, transitioning from `"tp": spmd.V` to `"tp": spmd.I`.  To
do this, we have to issue the all-reduce on the result of the linear,
to discharge the pending reduction.

```
def forward(self, x):
    spmd.assert_type(x,           {"tp": spmd.V})
    spmd.assert_type(self.weight, {"tp": spmd.V})

    out = F.linear(x, self.weight)
    spmd.assert_type(out,         {"tp": spmd.P})

    out = spmd.all_reduce(out, tp_group, dst=spmd.I)
    spmd.assert_type(out,         {"tp": spmd.I})

    return out
```

**Delayed all-reduce alternate universe.**  At the beginning of this section,
we adopted `"tp": spmd.I` as the calling convention between all modules in the
non-TP region (as this is how Megatron-style TP was originally described).  You
don't have to do it this way.  We could have instead specified `"tp": spmd.R`
as the calling convention instead: in backwards, this is equivalent to delaying
the backwards all-reduce out of col-parallel linear, and forcing someone else
to take care of the all-reduce on the weight gradients.  We won't work out this
example in detail, though, because an analogous problem arises in the context
of sequence parallelism, which is much more interesting.

#### Sequence parallelism: Module versus wrapper responsibility

With sequence parallelism, we ensure we don't "waste" the TP mesh axis outside
of TP regions by sharding the input on sequence outside of TP regions.
This changes the calling convention for the col-parallel/row-parallel linear, where
we uniformly have `"tp": spmd.V` throughout the entire module (but the cause
of this variation differs: in the SP region we are varying because we sharded on
sequence; in the TP region we are varying because we sharded on the TP weights).
The modification to col-parallel and row-parallel linear are fairly modest but
I include them here for completeness.

**Col-parallel linear with sequence parallel.**  Note that we do have
to explicitly specify in the collective what dimension the sequence dim
is for the all-gather, so we can concat (not stack) on it!

```
def forward(self, x):
    spmd.assert_type(x,           {"tp": spmd.V})
    spmd.assert_type(self.weight, {"tp": spmd.V})

    x = spmd.all_gather(x, tp_group, src=spmd.S(seq_dim), dst=spmd.R)
    spmd.assert_type(x,           {"tp": spmd.R})

    out = F.linear(x, self.weight)
    spmd.assert_type(out,         {"tp": spmd.V})

    return out
```

**Row-parallel linear with sequence parallel.**

```
def forward(self, x):
    spmd.assert_type(x,           {"tp": spmd.V})
    spmd.assert_type(self.weight, {"tp": spmd.V})

    out = F.linear(x, self.weight)
    spmd.assert_type(out,         {"tp": spmd.P})

    out = spmd.reduce_scatter(out, tp_group, dst=spmd.S(seq_dim))
    spmd.assert_type(out,         {"tp": spmd.V})

    return out
```

What is more interesting is that we've introduced a degree of freedom for
trainable parameters in the sequence parallel region: their gradients will need
an extra all-reduce over TP, analogous to how we need to do a DP/CP reduction
on gradients to incorporate the gradients from the other DP/CP shards.  Who is
responsible for doing this all-reduce?

For DP and CP, we said the contract is clear: someone else (the distributed
wrapper) is responsible for reducing gradients over the DP/CP mesh axis.  But
for TP, we have seen different codebases make different choices on this.
Specifically, there are two plausible options:

- Wrapper's responsibility: some other wrapper (usually FSDP) takes care of
  the reduction over TP: in this case, the parameter has the type `{"tp": spmd.R}`.
  This is directly analogous to how DP/CP reductions are handled.

- Module's responsibility: the module is responsible for accessing the
  replicated weight with an appropriate differentiable operation so that the
  backwards issues an all-reduce over TP; in this case the parameter has the
  type `{"tp": spmd.I}`.

Local SPMD types supports expressing both modes of operation.

**SP region module, wrapper responsibility.**  The nice thing about this scheme
is, like DP/CP, you don't have to change the module in the SP region at all.
But there is some hidden complexity: only *some* parameters require a TP
all-reduce (the non-TP sharded ones), and your wrapper has to keep track of who
should get a reduction and who doesn't!

```
def forward(self, input: torch.Tensor) -> torch.Tensor:
    spmd.assert_type(input,       {"tp": spmd.V})
    spmd.assert_type(self.weight, {"tp": spmd.R})

    out = rms_norm(input, self.weight)
    spmd.assert_type(out,         {"tp": spmd.V})

    return out
```

**SP region module, module responsibility.**  In module responsibility, the
weight is annotated `spmd.I`, and local SPMD types forces us to insert the
conversion `spmd.invariant_to_replicate` which ensures we run the all-reduce in
backwards.

```
def forward(self, input: torch.Tensor) -> torch.Tensor:
    spmd.assert_type(input,       {"tp": spmd.V})
    spmd.assert_type(self.weight, {"tp": spmd.I})

    weight = spmd.invariant_to_replicate(self.weight, tp_group)
    spmd.assert_type(weight, {"tp": spmd.R})

    out = rms_norm(input, weight)
    spmd.assert_type(out,         {"tp": spmd.V})

    return out
```

We generally advise you to follow whatever the prevailing conventions in your
training codebase are, when adding local SPMD types.  However, we *do* think
it's better to follow a wrapper's responsibility convention.  It simplifies
your SP-region model code, and it also makes it possible to apply a small
corretness fix when you have a small parameter is neither TP nor FSDP sharded.
In this case, in backwards you need to all-reduce on DP/CP and an all-reduce on
TP for this gradient's weights.  NCCL doesn't guarantee that doing two sequential
all-reduces on different PGs gives bitwise equivalent results across all ranks
(and in fact you can directly reproduce non equivalent results across ranks if
you select a weird mesh topology, where the PGs on one axis are sometimes fully
inside an NVLink domain and sometimes span across an NVLink domain); and you also
get a small benefit of only issuing one all-reduce instead of two.

To summarize:

- For all TP sharded parameters, you should annotate them with `"tp": spmd.V`.
  Typically, there will be some implicit understanding in your system already
  that they are TP-sharded (for example, the parameter may be initialized as a DTensor
  that is sharded on TP; or you may have some extra sharding metadata that
  your checkpointing implementation uses).  In those cases, you can piggy back
  off of that existing metadata to fill these annotations.

- For all other parameters, you should identify which calling convention your
  codebase uses, and then have your helper apply `"tp": spmd.I` or `"tp":
  spmd.R` accordingly.  If you are wrapper responsibility, consider having the
  wrapper apply this type according to the same rule by which it decides what
  parameters require a TP all-reduce: the presence of this TP all-reduce is
  the source of truth for the local SPMD type: if you get it wrong, the the
  typechecker won't be able to tell if we've missed or duplicated an
  all-reduce on a gradient.

#### FSDP

TP interacts with FSDP (unfortunately), primarily because TP sharding exists as
a concept *before* an FSDP wrapper is applied, and FSDP needs to "respect" the
TP sharding.  Additionally, if you are working in a module responsibility
regime, then FSDP is the most logical place to take care of TP all-reduce.
There are two primary situations to consider:

**FSDP on DTensor parameters (classic).**  This is the same setup we already
discussed for DP/CP under "FSDP on DTensor parameters" above -- parameters are
wrapped as DTensor before `fully_shard` is applied, and remain DTensor inside
the module body -- so the same two observations carry over: if your compute
stays on DTensor all the way through there is nothing for local SPMD types to
do, and otherwise you must ascribe a local SPMD type at the point where you
unwrap the DTensor into a plain tensor.  The only new question is how the TP
axis of the DTensor placement maps onto a local SPMD type.  Does the default
to `spmd.R` still seem appropriate?  We think so (prefer wrapper
responsibility); so this all works out of the box, given the same rules we
described for DP/CP.

**FSDP on Tensor parameters (new).**  What if I want to operate on plain
tensors?  Today, stock FSDP2 doesn't support tensor parallelism with an
original plain Tensor input.  We need a new API to support this.  (Note
that some FSDP2 forks already support this mode of us.)  The main reason why
you can't just implicitly have already TP sharded your weights is because
checkpointing needs to know that your weights are TP sharded, and there is
no PyTorch-native mechanism of indicating TP sharding without using DTensor.
There are two primary API types that apply in this case:

*A separate sharding metadata mechanism.*  Some training codebases have a
dedicated mechanism for saying that a plain Tensor has been sharded in some
way, usually fed directly into your checkpointing mechanism.  This annotation
should be repurposed to also apply the appropriate (local) SPMD type annotation
on those parameters.

We could also imagine extending PyTorch FSDP2 in this way.  Assuming that
there is some side-band annotation on the unsharded placement of the plain
Tensor, FSDP2 can incorporate this placement into the final DTensor that
represents the sharded parameter on the wrapped module.  The local SPMD type
annotation can then simply be inferred via the normal DTensor to SPMD type
mechanism.  You would need a separate configuration knob if you wanted to
switch to module-responsibility for backwards all-reduce.

*Use global SPMD types as the sharding metadata mechanism.*  Local SPMD types
doesn't give enough information about how a tensor is sharded, but global SPMD
types do.  So you could simply ensure that your original parameters are
annotated with global SPMD types, and then FSDP can directly use this
information to determine how to generate the DTensor.  This scheme also lets
you directly specify if you want wrapper-responsibility or
module-responsibility by selecting between `spmd.R` or `spmd.I` for TP
replicated parameters.

### Expert parallelism

```
def forward(hidden, topk, routes):                                   # {DP: V, CP: V, TP: V}
  x, topk, token_ids = prepare_activations(hidden, topk)
  with spmd.set_current_mesh({EDP, EP, ETP}):                        # {EDP: V, EP: V, ETP: V}
    global_counts = spmd.all_gather(routes.sum(dim=0), ETP, S(0), R) # {EDP: V, EP: V, ETP: R}
    global_counts = spmd.all_gather(global_counts, EP, S(0), R)      # {EDP: V, EP: R, ETP: R}
    ep_is, ep_os, etps = derive_splits(global_counts)
    x = spmd.all_to_all(x, EP, S(0), S(0), ep_is, ep_os)             # {EDP: V, EP: V, ETP: V}
    x = spmd.all_gather(x, ETP, S(0), R, split_sizes=etps)           # {EDP: V, EP: V, ETP: R}
    x = expert_fn(x)                                                 # {EDP: V, EP: V, ETP: P}
    x = spmd.reduce_scatter(x, ETP, P, S(0), split_sizes=etps)       # {EDP: V, EP: V, ETP: V}
    x = spmd.all_to_all(x, EP, S(0), S(0), ep_os, ep_is)             # {EDP: V, EP: V, ETP: V}
  return unpermute_and_combine(x, topk, token_ids, hidden)           # {DP: V, CP: V, TP: V}
```

### Putting it all together
