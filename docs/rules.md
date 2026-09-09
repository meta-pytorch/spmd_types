# Composing typing rules to typecheck custom autograd functions

Whenever you have a custom autograd function that either (1) performs
collectives or (2) that you would like to typecheck for global SPMD, you will
need to write a custom typing rule for the typechecker via the
`spmd_typecheck` method; we cannot automatically infer the intended semantics
of your program.

While you can manually write a typing rule simply by reading and asserting
types on the inputs and outputs to `spmd_typecheck`, we have some helpers
in `spmd_types.rules` where you can instead build a type-checking rule by
composing together pre-existing typing rules.  Specifically, we offer
the einsum rule (which can be used for more than just actual einsum
operations) and rules for all of SPMD's pre-existing collectives.  For
example, if you have a fused linear all-reduce, the functions in `rules`
make it easy to express the type:

```python
import spmd_types as spmd
from spmd_types import rules, I, P, R, S


class LinearAllReduce(torch.autograd.Function):
    """Row-parallel GEMM completed by an all-reduce."""

    @staticmethod
    def spmd_typecheck(*, x, weight, bias, group):
        y = rules.einsum("mk,nk->mn", x, weight)
        y = rules.all_reduce(y, group, src=P, dst=I)
        if bias is not None:
            y = rules.einsum("mn,n->mn", y, bias)
        return y

    @staticmethod
    def forward(ctx, x, weight, bias, group):
        ...
```

This is not a "decomposition" in the traditional sense, as the typing rule
only needs to describe how the sharding behavior of the function works,
it's not required that `spmd_typecheck` is literally runnable.

## `spmd_typecheck` protocol

To specify a typing rule for `spmd_types`, you define a `spmd_typecheck`
method on your custom autograd function.  We support two signature types:

```python
# out is the output Tensor or tuple of Tensor of forward,
# arg1, arg2, arg3 are the arguments to forward (argument names must match exactly)
@staticmethod
def spmd_typecheck(out, *, arg1, arg2, arg3) -> None:
    ...

# arg1, arg2, arg3 are the arguments to forward (argument names must match exactly)
# You return an NdimWithSpmdType (or tuple of them), which describes the types
# of the output; e.g., most typing rules in `spmd_types.rules` return
# NdimWithSpmdType objects.
@staticmethod
def spmd_typecheck(*, arg1, arg2, arg3) -> NdimWithSpmdType:
    ...
```

The first form is good if you're manually reading/asserting types.  The second
form tends to be nicer if you're composing `spmd_types.rules`.  In both cases,
the keyword arguments are inferred from the concrete argument names of the
positional arguments of the forward method on your autograd function.  This is
a slight abuse, since ordinarily the names of positional arguments don't
matter, but we do this because notice that custom autograd functions often
have a lot of non-Tensor arguments, and forcing keyword arguments makes it
easier to avoid mistakes.  Every Tensor argument of `forward` must be
accounted for by the rule (passed to a `rules` operation or to `assert_type`),
whether or not the hook lists it.  If a Tensor argument truly doesn't matter
for sharding and has no sharding constraints, you can mark it as ignored with
`rules.ignore(x)`.

## `rules.einsum`: the einsum sharding rule

`rules.einsum(equation, *operands)` is your swiss army knife for writing
global SPMD sharding rules for functions that operate entirely locally (no
collectives).  Intuitively, it gives you the sharding rule that the
corresponding einsum would have had, except we don't assume the function is
linear in any of its arguments.  It can also be useful for expressing
linearity (how partial arguments propagate) even for local SPMD only
functions.

Einsum notation lets us describe the structure of functions by assigning
indices to the dimensions of each input and output: for example,
`torch.einsum("ij,jk->ik", x, y)` indicates a function that takes two 2D
tensors and produces another 2D tensor.  In general, if you use `rules.einsum`
to generate a sharding rule for a function, you are saying that the function
commutes with chunking on every labeled dimension, where every operand that
names the index is chunked along it at once: i.e., `rules.einsum("i->i", x)`
says that for all `r` and `n`: `f(x.chunk(n, dim=0)[r]) == f(x).chunk(n, dim=0)[r]`
(where 0 is the position of `i` in the input and output tensors),
and `rules.einsum("i->", x)` says that `sum(f(c) for c in x.chunk(n, dim=0)) == f(x)`.

If you prefer to think symbolically in terms of how global SPMD types
propagate given `rules.einsum`, there are two rules:

1. A named index that appears in both inputs and the output indicates that for
   a given mesh axis, those inputs must be consistently sharded on that mesh
   axis.  The output would be sharded by this mesh axis on the output dim
   carrying that index.  For example, `rules.einsum("ij,jk->ik", x, y)`, if
   you pass `x` sharded on dim 0 (index `i`), then the output is sharded on
   dim 0 (index `i`).  Intuitively, shared indices indicate dimensions that
   are aligned; however, these dimensions do not have to share the same size:
   for example, `"i->i"` would validly describe `x.repeat_interleave(2)`.

   If an argument does not mention an index, it must be replicated on the mesh
   axis that shards that index.  Conversely, a mesh axis shards at most one index.

2. A named index that appears in inputs but not in the output will produce a
   result that is partial in the mesh axis it was sharded over.  Intuitively,
   indices that don't occur in output are contraction dims that are summed
   over; however, it doesn't have to only be a sum: for example, `"i->"` would
   validly describe `torch.sum(torch.square(x))`.  It's invalid to have an
   index that only appears in output.

In local SPMD typechecking, `rules.einsum` degenerates to the usual local
propagation rule.  Most notably, `"i->"` takes `V` to `V` (not partial).

**Linearity.** Unlike einsum, the typing rule here does not assume your operation is linear
in the values of its operands; in other words, partial operands are not
propagated by default.  But some operations (like matmuls or summations) should
propagate partials!  You can use `linear_in` to explicitly express that the
function is linear in an argument.  Concretely:

1. In regular einsum, it is defined such that the einsum
   function is multilinear in all of its input operands.  For example, if we
   have `torch.einsum("i,i->i")`, then we have `P, R -> P`, and `R, P -> P`.
   `rules.einsum` rejects this: you must explicitly declare what arguments you
   are linear in with `linear_in=(0,1)`.

2. Sometimes a function is linear in a joint argument (this is generally
   not expressible in `torch.einsum`). For example, for `torch.add` we have
   `P, P -> P`.  You can declare you are jointly linear in several arguments
   with a tuple: `linear_in=((0,1),)`

**Extensions.** We also introduce some small extensions to einsum notation
which help handle some common situations:

1. For each argument and the output, you can use a `...` placeholder once.
   This is a placeholder for arbitrarily many dimensions; every occurrence of
   `...` (including on the output) stands for the same number of dimensions,
   and their shardings pass through by position.  So for example, you can
   express an arbitrary rank pointwise operation as `...,...->...`, instead of
   having to write `a,a->a`, `ab,ab->ab`, etc.  It can be combined with other
   dims, e.g., batched sum is `...a->...`

2. You can use `_` as a placeholder anywhere, as many times as you like.  In
   inputs, this placeholder says the function does not commute with chunking
   on this dimension (i.e., the operator accesses "all elements" along that
   dimension.)  In outputs, this placeholder says the
   output dim is unsharded (no mesh axis shards it); it is also how you write
   an output dim that no input index feeds.  Note that in local SPMD, a `V`
   with no shard dim is accepted at a `_` slot (a `V` carrying `S(i)` on that
   dim is rejected even under local checking); if you must not accept `V`, it
   is likely there is a collective involved, and you should be using one of
   the collective rules (below).

   A useful idiom: to say an operator has no global sharding story, write
   `rules.einsum("_,_->_", x, y)` (adjusting the number of underscores for
   input/output dimensionality).  Under global checking this admits only
   replicated inputs, for which any local operator is trivially correct; under
   local checking it is just the local propagation rule.

**Multiple outputs.**  `rules.einsum` does not support multiple outputs.
For each output, you should write a separate `rules.einsum` for it, operating
on exactly the inputs that influence that output.  This allows for fine-grained
statements about linearity with respect to each output.

**Examples.**

```python
# If your function is expressible with einsum (e.g., Linear), you can simply
# write the einsum directly.  A sharded k is Partial by itself; `linear_in`
# additionally lets Partial inputs propagate, as they would through a matmul.
rules.einsum("mk,nk->mn", x, w, linear_in=(0, 1))

# RMSNorm with a learnable scale: the normalized hidden dim must be complete
# on every rank; tokens may be sharded (sequence parallel).
rules.einsum("t_,_->t_", x, w)

# Fused SiLU-and-mul gate (or any elementwise binary op): every dim is a
# batch dim, so any sharding passes through, at any rank.
rules.einsum("...,...->...", g, u)

# Transpose: the shardings travel with their dims.
rules.einsum("ab->ba", x)
```

**Non-examples.**

```python
# x.mean(0)
rules.einsum("i->", x)    # WRONG!  sum(mean of chunks) != mean of whole
rules.einsum("_->", x)    # Correct.

# x * torch.arange(n)
rules.einsum("i->i", x)   # WRONG!  x can't be sharded, the arange needs per-rank range
rules.einsum("_->_", x)   # Correct.

# softmax(x, dim=0)
rules.einsum("i->i", x)   # WRONG! every output reads every input.
rules.einsum("_->_", x)   # Correct.

# torch.cat([x, x])
rules.einsum("i->i", x)   # WRONG! chunks come out as [x0, x0, x1, x1], not [x0, x1, x0, x1]
rules.einsum("_->_", x)   # Correct.
```

## Type-only collectives and local transitions

Some custom autograd functions implement distributed communication, or
are local operations but are rank-aware in some way.  Often, the typing rules
for these functions match `spmd_types` built-in collectives.  Similarly,
your codebase may already have custom autograd functions for collectives, and
you would like to use them directly, rather than port to `spmd_types` version.
For this use case, we provide typing rules for all of `spmd_types` collectives
(`rules.all_reduce`, `rules.all_gather`, `rules.reduce_scatter`, `rules.all_to_all`,
`rules.reinterpret` and `rules.convert`) which take the same core arguments as the
real `spmd` operations (`x, axis, *, src, dst`; the runtime-only options such as
`gather_dim` or `op_dtype` do not apply) and perform their corresponding type
operation.  `axis` may be a `ProcessGroup` (typically the kernel's own
argument), a `MeshAxis`, or a mesh axis name.  If the input has no type yet on
that axis, `src` is asserted onto it; transitions on axes of size one are
no-ops.

```python
rules.all_gather(x, group, src=S(0), dst=R)      # concat form: rank unchanged
rules.all_gather(x, group, src=V, dst=R)         # stack form: gains a leading dim
rules.reduce_scatter(y, group, src=P, dst=S(0))  # shard form: rank unchanged
rules.reduce_scatter(y, group, src=P, dst=V)     # unbind form: loses dim 0
rules.all_reduce(y, group, src=P, dst=I)         # R or I is the backward contract
rules.reinterpret(y, axis, src=V, dst=P)
rules.convert(x, axis, src=R, dst=S(0))
```

For example, Megatron defines `_CopyToModelParallelRegion`, which is a no-op
in forwards but an all-reduce in backward.  This corresponds directly to
`convert(I, R)`.  So we can type it as follows:

```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def spmd_typecheck(*, input_, group):
        return rules.convert(input_, group, src=I, dst=R)

    ...
```

There is one deliberate semantic difference between the typing rules and the runtime collectives.
`spmd.all_reduce(y, g, dst=I)` will implicitly convert a `V` input to `P`;
`rules.all_reduce(y, g, dst=I)` requires the input to be exactly `P`.  If you
really do want to sum a `V`, say so: `rules.all_reduce(y, g, src=V, dst=I)`.
The hook is a claim about what the kernel computes, and the checker will not
fill in a claim you did not write.

## Intermediates are `NdimWithSpmdType`

The typing rules above return intermediates that express a typing operation,
which you can send to other typing rules.  These intermediates are not real
tensors; rather, they're an impoverished `NdimWithSpmdType` which are
essentially a dimension count plus an `SpmdType`; precisely what is needed
for computing type rules.

## Checking a typing rule numerically: `rulecheck`

In ordinary typechecking, `spmd_typecheck` implementations are completely
trusted; we cannot detect if you've written an incorrect sharding rule.
`spmd_types.rulecheck` lets you numerically test your sharding rule for
the commute-with-chunking invariant that it indicates.

```python
from spmd_types.rulecheck import rulecheck

report = rulecheck(
    LinearAllReduce,               # the class from the top of this document
    args=(x, weight, bias, "tp"),  # global (not sharded) tensors; "tp" names the group argument
    mesh={"dp": 2, "tp": 2},
    local_tensor_mode=True,        # the kernel calls collectives
)
```

This enumerates input placements (per mesh axis: everything replicated, every
way of sharding one dim of each size across the tensor arguments, each tensor
Partial and all of them Partial, all Invariant, and the product of these across
axes), skips the ones the typing rule rejects, and for each accepted one checks
that running the kernel per rank and reassembling the outputs according to the
derived types gives the same result as running the kernel on the original
global tensors.  `report.checked` and `report.rejected` list what happened;
read the rejected list to confirm the rule refuses what you expect.  Pass
explicit `placements=` only for a configuration enumeration cannot construct.
For rulecheck to work, the autograd function in question must be "mesh
invariant" (that is, it can work with an arbitrarily sized device mesh--in
particular, it must be possible to run it on a mesh of size one, which is how
the reference value is computed unless you pass your own `reference=`.)

It is easiest to use `rulecheck` for local operations, which run on plain
per-rank tensors with no distributed setup.  A kernel that calls collectives
needs `local_tensor_mode=True`: the ranks are simulated in one process under
`LocalTensorMode` with a fake process group (all native PyTorch
`torch.distributed` APIs support this), and `dist.get_rank(group)` answers per
simulated rank so rank-dependent kernels run correctly.  Pass `rtol`/`atol`
for kernels that do not reproduce the reference to the default tolerances.

## Case study: Vocab-parallel cross entropy

Megatron's `_VocabParallelCrossEntropy` takes logits sharded on the vocab
dim over `tp` and replicated targets, and returns a per-token loss that is
the same on every `tp` rank.  Its forward is a small distributed algorithm:

1. Take the local max over the vocab shard and all-reduce it with `MAX`;
2. Subtract it and sum `exp` over the shard, then all-reduce the sum;
3. Gather the target logit if the target falls in this rank's vocab slice,
   else contribute zero, and all-reduce that too;
4. Return `log(sum_exp) - target_logit`.

It technically works to give the rule as:

```python
def spmd_typecheck(*, logits, target, group):
    x = rules.einsum("tv,t->t", logits, target)
    return rules.all_reduce(x, group, src=P, dst=I)
```

However, this is sort of lying, as it implies that there is a single `x`
that gets all-reduced inside the kernel.

You can also write the typing rule more verbosely by following the function's
semantics (eliding the `max` computation, which is just for numerics):

```text
loss[t] = log( sum_v exp(logits[t, v]) ) - logits[t, target[t]]
```

Concretely:

```python
class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def spmd_typecheck(*, logits, target, group):
        z = rules.einsum("tv->t", logits)               # sum_v exp(logits[t, v])
        z = rules.all_reduce(z, group, src=P, dst=I)
        pred = rules.einsum("tv,t->t", logits, target)  # masked gather of the target logit
        pred = rules.all_reduce(pred, group, src=P, dst=I)
        return rules.einsum("t,t->t", z, pred)          # log(z) - pred
```

Here, `z` and `pred` have the types their algorithmic counterparts have (the
kernel's versions are shifted by the max, which changes values but not types),
even though this doesn't matter for the final typing rule.  The max and the
rank do not appear at all: the max is a stabilizer that a shift-invariant
logsumexp does not depend on, and the rank is bookkeeping the placement
determines, not an input to the function.

In any case, both these typing rules are functionally correct, and will pass
`rulecheck`, but it's probably worth a comment or two if you write rules in
the top style.
