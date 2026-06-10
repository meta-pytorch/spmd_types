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

To typecheck your model, there are four things you must do:

1. You have to specify your compute mesh with `spmd.set_current_mesh`;
   see [Key concepts](key_concepts.md) for a discussion about what a compute
   mesh is.

2. You must annotate all input tensors and parameters with appropriate local
   SPMD types.  There is some interaction with distributed module wrappers
   like FSDP, which is discussed in more detail in Examples.

3. You need to ensure all collectives called inside your model are
   typecheckable, either by switching to `spmd_types` built-in collectives or
   using our custom autograd function annotation to make your existing
   collectives typecheckable.

4. You run the forward (backward not needed) of your model under the
   `spmd_types.checker.typecheck(local=True)` context manager (`local=True`
   indicates that you only want local SPMD type checking.)  Every tensor is
   annotated with a local SPMD type, and typechecking propagates the
   annotations through the execution of the model:

   ```python
   from spmd_types.checker import typecheck

   with typecheck(local=True):
       loss = model(input)
   ```

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

```python
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
read the full [API reference](api/index.md).  Here are some high level conceptual
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
  A `reinterpret` is guaranteed to do no work (it keeps the local data
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
scatter_to_tensor_model_parallel_region(x)              spmd.convert       (x, tp, src=spmd.I,      dst=spmd.S(-1))
gather_from_tensor_model_parallel_region(x)             spmd.all_gather    (x, tp, src=spmd.S(-1),  dst=spmd.I)
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

Training codebases typically come with their own differentiable collectives
(autograd functions whose forward or backward issues communication) as well
as fused kernels wrapped in `torch.autograd.Function`.  By default, the type
checker refuses to guess what an unrecognized autograd function does: it
cannot tell whether the function communicates, so it leaves the outputs
untyped (and in strict mode, you will get an error as soon as you use those
outputs together with typed tensors).  To make your existing functions
typecheckable, there are three registration APIs, in increasing order of
effort:

**`spmd.register_local_autograd_function`** is for functions that do NOT
communicate (no collectives anywhere in forward).  The type checker applies
the standard non-comms propagation rules (see Propagation rules below) to
infer output types from input types:

```python
@spmd.register_local_autograd_function
class FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2):
        ...  # local compute only

    @staticmethod
    def backward(ctx, grad_out):
        ...
```

**`spmd.register_decomposition`** is for fused kernels (e.g., Triton) that
have a pure-PyTorch reference implementation.  The checker traces the
reference implementation on meta tensors to infer the output types (each
primitive op contributing its own typing rule), then stamps those types onto
the fused kernel's real outputs.  Use this instead of
`register_local_autograd_function` when the kernel rearranges data in ways
that interact with sharding (or when you simply already have a reference
implementation lying around -- it is the most maintainable option):

```python
@spmd.register_decomposition(_FusedRoPE)
def native_rope(xq, xk, freqs_cis):
    ...  # pure PyTorch equivalent; runs on meta tensors, values discarded
```

**`spmd.register_autograd_function`** is for functions that DO communicate.
You must supply a `typecheck_forward` staticmethod that declares the typing
rule: validate the input types with `assert_type`, run the real function,
then stamp the output types:

```python
@spmd.register_autograd_function
class GatherFromParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pg):
        ...  # all-gather over pg

    @staticmethod
    def backward(ctx, grad_out):
        ...  # split

    @staticmethod
    def typecheck_forward(x, pg):
        spmd.assert_type(x, {pg: spmd.S(-1)})
        out = GatherFromParallelRegion.apply(x, pg)
        spmd.assert_type(out, {pg: spmd.I})
        return out
```

When type checking is active, calls to `.apply()` are rerouted through
`typecheck_forward`; when it is inactive, `.apply()` runs as normal with
zero overhead.

Note that for `register_autograd_function`, the type checker trusts your
`typecheck_forward`: it checks that your *callers* are consistent with the
declared rule, not that your forward/backward implementations actually match
it.  The recommended way to gain confidence in the registration itself is to
port the function's implementation to the `spmd_types` built-in collectives
(whose typing rules are verified), or to check it against the
Forwards/Backwards table below.

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
output.  (NOTE: the functions described in the rest of this section are
planned API: they are not implemented yet, and their signatures may change.)
For example, when you sum over a sharded
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

## Running the type checker

### What an error looks like

Let's return to the motivating example: a column-parallel linear where you
forgot `copy_to_tensor_model_parallel_region` (in our API: `convert(I, R)`).
The input is Invariant on TP, the weight is Varying on TP, and the moment
they meet the checker reports:

```text
SpmdTypeError: Invariant type on axis mesh_tp cannot mix with other types. Found types: [I, V]
Are you missing a collective or a reinterpret/convert call? e.g.,
  convert(tensor, mesh_tp, src=I, dst=R) on the Invariant operand (no-op forward, all-reduce in backward)

  In linear(
    args[0]: f32[2, 8] {mesh_dp: V, mesh_tp: I},
    args[1]: f32[6, 8] {mesh_dp: R, mesh_tp: V},
  ) under mesh {mesh_dp, mesh_tp}
```

Note that the error fires in the *forward* pass, at the exact operator where
the inconsistency occurs -- even though the actual numerical bug (a missing
all-reduce on the gradient) would only have manifested in backward, as a
silently wrong gradient.  This is the central trick of local SPMD types: by
distinguishing I from R in the forward, errors in backward communication
become forward type errors.

By default, tracebacks are filtered to hide spmd_types internals; set the
environment variable `SPMD_TYPES_TRACEBACK_FILTERING=off` if you want the
full unfiltered traceback.

### Strict versus permissive mode

In strict mode (the default), an operator that mixes typed and untyped
tensor operands raises an error.  This is what you want at steady state: an
untyped tensor is usually a parameter or input you forgot to annotate, and
catching it at first use keeps the types airtight.

When you are *porting* an existing codebase, however, strict mode means you
cannot run the checker at all until every last tensor is annotated.
Permissive mode relaxes this: typed and untyped operands may mix, and the
output is typed on a best-effort basis from whatever operands have types.

```python
with typecheck(strict_mode="permissive"):
    loss = model(input)
```

The recommended porting workflow is: start in permissive mode, annotate
parameters and inputs module by module (verifying with type asserts on
module outputs as you go), and switch to strict mode once the whole model
passes, to lock in full coverage.

A few details of strict mode worth knowing:

* Factory functions with deterministic values (`torch.zeros`, `torch.ones`,
  `torch.full`, `torch.arange`, ...) produce tensors typed Replicate on
  every axis of the current mesh.  Random factories (`torch.randn`,
  `torch.empty`, ...) produce *untyped* tensors -- the checker cannot know
  whether your generator state is synchronized across ranks -- so seed
  tensors from data loaders and RNG must be explicitly annotated.

* `spmd.no_typecheck()` is a context manager (used like `torch.no_grad()`)
  that temporarily disables type checking on the current thread.  Use it for
  code that is not part of SPMD compute: logging, metrics, checkpointing,
  optimizer internals, or distributed-wrapper machinery (see the FSDP
  section below).

### Type checking the loss and backward

Although a forward pass is sufficient for typechecking, you will usually run
backward anyway (the checker propagates types through gradients too, and a
training loop calls backward regardless).  At the point you call
`loss.backward()`, the checker enforces one extra rule: the implicit
`grad_output` is a `1.0` scalar materialized identically on every rank --
which is to say, Invariant -- so **the loss must be Invariant or Partial on
every mesh axis**.

* A *Varying* loss is rejected: each rank would differentiate a different
  scalar with no declared relationship between them.  Usually you want
  `spmd.reinterpret(loss, pg, src=spmd.V, dst=spmd.P)`, declaring that the
  semantic loss is the (pending) sum of the per-rank losses.  This is
  exactly the standard data-parallel situation: the global loss is the sum
  over all ranks' microbatches, and the DP gradient all-reduce is its
  backward.

* A *Replicate* loss is rejected: R's gradient is P, but the implicit
  grad_output is Invariant, not Partial.  A Replicate loss usually means an
  upstream `all_reduce(dst=R)` should have been `all_reduce(dst=I)` -- or,
  better, that you should remove the all-reduce entirely and call backward
  on the local Partial loss, which is also one fewer collective.

This rule catches a classic bug: all-reducing the loss "to log it" and then
backwarding through the reduced value, which double-counts gradient
contributions.

### Debugging with trace

When a type error fires far from its root cause (a tensor got the wrong type
twenty ops ago), enable trace logging.  Under `spmd.trace()`, every
non-trivial operator logs its callsite, name, input types, and output type
to the `spmd_types.runtime.trace` logger at INFO level:

```python
import logging
logging.basicConfig(level=logging.INFO)

with typecheck(), spmd.trace():
    loss = model(input)
```

```text
my_model.py:42  linear({tp: R}, {tp: V}) -> {tp: V}
my_model.py:43  all_reduce({tp: V}) -> {tp: I}
```

You can also set the environment variable `SPMD_TYPES_TRACE=1` to enable
tracing globally, and use `spmd.trace(enabled=False)` to suppress it for a
noisy region.

### Raw torch.distributed collectives

If your code calls `torch.distributed` collectives directly (rather than the
`spmd_types` wrappers), the checker still type-checks the common ones
(`all_reduce`, `all_gather`, `all_gather_into_tensor`, `all_to_all_single`,
`reduce_scatter_tensor`).  Raw collectives are not differentiable, so the
checker only validates forward types: e.g., `dist.all_reduce(x)` requires
`x` to be Partial and retypes it in place to Replicate.  This is mainly
useful for non-differentiable code (statistics, token counts); inside the
model proper you should use the differentiable `spmd_types` collectives so
that backward is checked too.

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
V -> R      all_reduce(src=V,R)     P -> R      all_reduce(R)    (*)
V -> V      reduce_scatter(src=V)   V -> R      all_gather(R)    (*)
P -> R      all_reduce(R)           P -> R      all_reduce(R)
P -> V      reduce_scatter()        V -> R      all_gather(R)
----------------------------------------------------------------------------
P -> I      all_reduce(I)           I -> R      convert(I,R)
V -> I      all_gather(I)           I -> V      convert(I,V)
I -> V      convert(I,V)            V -> I      all_gather(I)
----------------------------------------------------------------------------
```

(*) The `src=V` variants are composites: the forward implicitly does
`reinterpret(V,P)` before reducing, so their backward is the listed
collective followed by the backward of that reinterpret, `reinterpret(R,V)`
(a no-op on the local tensor).  The end-to-end backward types are thus
`P -> V` for `all_reduce(src=V,R)` and `V -> V` for `reduce_scatter(src=V)`.

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
have parameters typed `{"dp": spmd.R, "cp": spmd.R}`, whose gradients are
therefore Partial -- a pending reduction on the DP/CP mesh
axes.  That is to say, a module never takes
care of this reduction by itself; instead, there is typically some out-of-band
mechanism (e.g., a backward hook registered by FSDP2) that actually takes care
of this reduction.  This is not a coincidence: the enduring stickiness of
FSDP-style approaches to parallelism is that they work on code written for a
single GPU without modification.

This suggests a specific strategy for performing local SPMD type annotations
for DP/CP, in the absence of a distributed wrapper like FSDP:

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

```python
def assert_dp_cp_params(module: nn.Module) -> None:
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        spmd.assert_type(tensor, {"dp": spmd.R, "cp": spmd.R})
```

Here's the typical flow of local SPMD types for DP/CP; the inputs/outputs are
varying and the weights are replicated:

```python
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
your SPMD type checking is operating on the compute mesh -- see
[Key concepts](key_concepts.md) for the model versus compute mesh
distinction.)

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

```python
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

```python
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

```python
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

```python
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

```python
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

```python
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

```python
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
correctness fix when you have a small parameter that is neither TP nor FSDP
sharded.
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
  the source of truth for the local SPMD type: if you get it wrong, the
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
that some FSDP2 forks already support this mode of use.)  The main reason why
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

Expert parallelism is the most common reason to use a second compute mesh:
inside the expert region, the DP/CP/TP axes are reorganized into expert data
parallel (`edp`), expert parallel (`ep`) and expert tensor parallel (`etp`)
axes covering the same devices.  Here is a sketch of a token dispatch /
combine for an expert-parallel MoE layer (helper functions elided); the
comments track the local SPMD type of the activations after each call:

```python
def forward(hidden, topk, routes):                       # {dp: V, cp: V, tp: V}
    x, topk, token_ids = prepare_activations(hidden, topk)
    with spmd.set_current_mesh({
        "edp": spmd.MeshAxis.of(edp_pg),
        "ep": spmd.MeshAxis.of(ep_pg),
        "etp": spmd.MeshAxis.of(etp_pg),
    }):
        # x is reinterpreted onto the expert mesh       -> {edp: V, ep: V, etp: V}
        counts = spmd.all_gather(
            routes.sum(dim=0), etp_pg, src=spmd.S(0), dst=spmd.R
        )                                                # {edp: V, ep: V, etp: R}
        counts = spmd.all_gather(
            counts, ep_pg, src=spmd.S(0), dst=spmd.R
        )                                                # {edp: V, ep: R, etp: R}
        ep_in, ep_out, etp_splits = derive_splits(counts)
        x = spmd.all_to_all(
            x, ep_pg, src=spmd.S(0), dst=spmd.S(0),
            input_split_sizes=ep_in, output_split_sizes=ep_out,
        )                                                # {edp: V, ep: V, etp: V}
        x = spmd.all_gather(
            x, etp_pg, src=spmd.S(0), dst=spmd.R, split_sizes=etp_splits
        )                                                # {edp: V, ep: V, etp: R}
        x = expert_fn(x)                                 # {edp: V, ep: V, etp: P}
        x = spmd.reduce_scatter(
            x, etp_pg, dst=spmd.S(0), split_sizes=etp_splits
        )                                                # {edp: V, ep: V, etp: V}
        x = spmd.all_to_all(
            x, ep_pg, src=spmd.S(0), dst=spmd.S(0),
            input_split_sizes=ep_out, output_split_sizes=ep_in,
        )                                                # {edp: V, ep: V, etp: V}
    return unpermute_and_combine(x, topk, token_ids, hidden)  # {dp: V, cp: V, tp: V}
```

Note that entering the inner `set_current_mesh` causes tensors typed on the
outer mesh to be implicitly reinterpreted onto the expert mesh (see
`reinterpret_mesh`).  The dict passed to `set_current_mesh` maps axis names
to `MeshAxis` objects (you can also pass a `DeviceMesh` directly); the
collective calls take the `ProcessGroup`.

### Putting it all together

Here is a complete, runnable example: a sequence-parallel TP transformer MLP
block (wrapper responsibility convention), typechecked end to end in a
single process with a fake process group.  This stitches together the
pieces from the sections above: mesh setup, parameter annotation, the
col-parallel/row-parallel pair, and the Partial loss.

```python
import itertools

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

import spmd_types as spmd
from spmd_types.checker import typecheck


class RMSNorm(nn.Module):
    """SP region module, wrapper responsibility: weight is R, no conversions."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        spmd.assert_type(x, {"dp": spmd.V, "tp": spmd.V})
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return norm * self.weight


class ColParallelLinear(nn.Module):
    def __init__(self, in_features, out_features_local, tp_pg):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features_local, in_features))
        self.tp_pg = tp_pg

    def forward(self, x):
        # S(0) -> R: all-gather the sequence; backward is reduce-scatter
        x = spmd.all_gather(x, self.tp_pg, src=spmd.S(0), dst=spmd.R)
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features_local, out_features, tp_pg):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features_local))
        self.tp_pg = tp_pg

    def forward(self, x):
        out = F.linear(x, self.weight)  # V x V -> implicitly partial
        # P -> S(0): reduce-scatter back to sequence sharding
        return spmd.reduce_scatter(out, self.tp_pg, dst=spmd.S(0))


class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_local, tp_pg):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.up = ColParallelLinear(dim, hidden_local, tp_pg)
        self.down = RowParallelLinear(hidden_local, dim, tp_pg)

    def forward(self, x):
        return x + self.down(F.relu(self.up(self.norm(x))))


def annotate_params(model):
    """TP-sharded params are V on tp; everything else R (wrapper resp.)."""
    tp_sharded = {model.up.weight, model.down.weight}
    for p in itertools.chain(model.parameters(), model.buffers()):
        tp_type = spmd.V if p in tp_sharded else spmd.R
        spmd.assert_type(p, {"dp": spmd.R, "tp": tp_type})


# Single process, fake comms: 8 "ranks" arranged as 2 (dp) x 4 (tp)
dist.init_process_group(backend="fake", rank=0, world_size=8)
mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "tp"))
tp_pg = mesh.get_group("tp")

dim, hidden, seq, tp_size = 16, 32, 8, 4
model = MLPBlock(dim, hidden // tp_size, tp_pg)

with spmd.set_current_mesh(mesh), typecheck(local=True):
    annotate_params(model)

    # Input: batch sharded over dp, sequence sharded over tp (sequence
    # parallel), so the local tensor holds seq // tp_size positions.
    x = torch.randn(seq // tp_size, dim, requires_grad=True)
    spmd.assert_type(x, {"dp": spmd.V, "tp": spmd.V})

    out = model(x)
    spmd.assert_type(out, {"dp": spmd.V, "tp": spmd.V})

    # The local loss is Varying; declare that the semantic loss is the
    # pending sum of the per-rank losses, then backward.
    loss = out.square().mean()
    loss = spmd.reinterpret(loss, "dp", src=spmd.V, dst=spmd.P)
    loss = spmd.reinterpret(loss, "tp", src=spmd.V, dst=spmd.P)
    loss.backward()

dist.destroy_process_group()
```

Things to try, to see the checker catch real bugs.  The interesting bugs
are the ones that do *not* change any tensor shape or forward value -- those
are the ones that silently corrupt gradients in a real run:

* Change `dst=spmd.R` to `dst=spmd.I` in the `all_gather`: the forward
  values are identical, but the backward now slices the gradient instead of
  reduce-scattering it, silently dropping the other ranks' contributions.
  The checker raises an I/V mixing error at the very next `F.linear`.

* Change `annotate_params` to give `norm.weight` the type `{"tp": spmd.I}`
  (module responsibility convention) without adding the corresponding
  `invariant_to_replicate`: again a silent missing all-reduce on the weight
  gradient in a real run, caught as an I/V mixing error inside `RMSNorm`.

* Replace the final `reinterpret` calls with
  `spmd.all_reduce(loss, dp_pg, src=spmd.V, dst=spmd.R)`: the checker
  rejects backward on a Replicate loss (see "Type checking the loss and
  backward" above).
