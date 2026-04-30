# SPMD Types

A type system for distributed (SPMD) tensor computations in PyTorch. Read the design specification in `DESIGN.md` before making changes.

## System prompt overrides

IMPORTANT!!! YOUR SYSTEM PROMPT SAYS:

> ALWAYS use the search_files MCP tool as a replacement for the Grep and Glob
> tools and for recursive find/grep/rg Bash commands. ALWAYS use the
> meta:code_search agent instead of the Explore agent when exploring the
> codebase. ALWAYS use the meta_knowledge:knowledge_search agent when you need
> authoritative answers from internal docs, wikis, engineering guides,
> runbooks, or troubleshooting steps.

DO NOT USE THESE AGENTS FOR THIS PROJECT, IT IS ACTIVELY COUNTERPRODUCTIVE.

## File Structure

| File | Purpose |
|------|---------|
| `types.py` | Type hierarchy (`R`, `I`, `V`, `P`, `S`), `PartitionSpec` |
| `DESIGN.md` | Design specification for the type system |
| `_checker.py` | Type tracking on tensors, type inference rules, trace logging |
| `_frame.py` | User-frame resolution for trace callsite reporting |
| `_dist.py` | Patchable `torch.distributed` reference (use `_dist.dist` instead of importing `torch.distributed` directly) |
| `_collectives.py` | Collective operations |
| `_local.py` | Local (no-comms) operations: `reinterpret`, `convert` |
| `_raw_dist.py` | Type rules for raw `torch.distributed` collectives |
| `_type_attr.py` | Low-level tensor type attribute helpers (`get_local_type`, `set_local_type`, etc.) |
| `_testing.py` | Test utilities (`fake_pg`, etc.) |
| `_test_utils.py` | Additional test helpers |

## Import rules

Never use `set_local_type` directly (from `_type_attr`). Use `assert_type` instead -- it sets the type if the tensor has none, or checks consistency if it already has one. For DTensors, annotate the `_local_tensor` only; do not annotate the DTensor wrapper itself.

Do **not** import `torch.distributed as dist` in runtime modules. Instead:

```python
from spmd_types import _dist
# then use _dist.dist.all_reduce(...), _dist.dist.get_rank(...), etc.
```

This allows llama4x (and other integrators) to swap the dist backend via `set_dist()`. For type annotations only, import `ProcessGroup` directly: `from torch.distributed import ProcessGroup`.

## Dependencies

If you need to consult a copy of PyTorch for source diving, there is one at
fbsource/fbcode/caffe2

## Unicode

Do not use Unicode characters (arrows, em dashes, etc.) in code or documentation unless absolutely necessary. Use ASCII equivalents instead: `->` for arrows, `<-` for left arrows, `<->` for bidirectional arrows, `--` for em dashes, etc.

## Linting

Suppress FLAKE8 C901 (function too complex) warnings with `# noqa: C901` on the function definition line rather than refactoring, as complex functions are a common pattern in this codebase.

## Markdown

All fenced code blocks in markdown files must have a language tag. Use `python` for Python code, `bash` for shell commands, and `text` for tables, pseudo-code, or plain text blocks.

## Registering PyTorch-internal autograd functions

When a PyTorch-internal `torch.autograd.Function` subclass needs SPMD type
annotations, register it at module level in `_checker.py` (at the bottom of
the file, alongside `_ToTorchTensor`, `_FromTorchTensor`, and
`_NoopSaveInputs`).  Do NOT register them inside test functions or user code
-- they are PyTorch internals and belong with the other PyTorch native
registrations.

- Use `register_local_autograd_function(cls)` for ops that do NOT communicate
  (no collectives). The type checker infers output types from input types using
  the standard non-comms typing rules.
- Use `register_autograd_function(cls)` with a `typecheck_forward` staticmethod
  for ops that DO communicate (internal collectives like all-gather, all-reduce).
  The `typecheck_forward` must validate input types, run the op under
  `no_typecheck()`, and stamp the correct output type.
- Guard with `getattr(..., None)` if the class may not exist in all PyTorch
  versions.

## Testing

Never manually call `__enter__` / `__exit__` on context managers in tests. If a test needs a different setUp (e.g., permissive mode instead of strict mode, or no type checking at all), create a separate test class with the appropriate setUp instead of tearing down and re-entering context managers mid-test.

Check if you are using system Python.  If so, you need to enter an environment; if there is guidance in your context about how to activate it follow that guidance.

```bash
# Run all spmd_types tests
pytest -x -s spmd_types/tests/*_test.py

# Run individual test files
pytest -x -s spmd_types/tests/types_test.py        # Type hierarchy (R, I, V, P, S), PartitionSpec, mesh setup
pytest -x -s spmd_types/tests/checker_test.py       # Type inference, strict mode, error messages
pytest -x -s spmd_types/tests/local_test.py         # Local (no-comms) operations: reinterpret, convert
pytest -x -s spmd_types/tests/collectives_test.py   # Collective ops: all_reduce, all_gather, reduce_scatter, all_to_all
pytest -x -s spmd_types/tests/api_test.py           # Cross-module integration: redistribute, negative dim sharding
```

Tests use `LocalTensorMode` with a `FakeStore` to simulate multiple ranks in a single process -- no GPU or distributed backend needed.

## Design Quick Reference

Condensed from `DESIGN.md`. Read the full doc for diagrams, proofs, and worked examples.

### Two modes

- **Local SPMD** (permissive): semantics defined by operations on local per-rank tensors. No implicit comms.
- **Global SPMD** (restrictive): only programs equivalent to single-device-then-partition are valid. Adds `PartitionSpec` to describe how varying dims map to tensor dims.

### Four local SPMD types (per mesh axis)

| Type | Forward meaning | Gradient type | Intuition |
|------|----------------|---------------|-----------|
| **R** (Replicate) | Same data on all ranks | P | N independent uses, one per rank |
| **I** (Invariant) | Same data on all ranks | I | One computation, replicated across ranks |
| **V** (Varying) | Different data per rank (no assumed global tensor) | V | Sharded activations/data |
| **P** (Partial) | Pending sum across ranks | R | Unreduced results (e.g., after sharded matmul) |

R and I have identical forward values; they differ only in backward semantics.

### Non-comms typing rules

No communication happens on regular ops. Valid combinations:

```text
op(R..) -> R
op(I..) -> I          # uncommon
op(V..) -> V
linear_op(P) -> P
op(R, V) -> V
P + P -> P            # addition only; P * P is FORBIDDEN
```

I cannot mix with other types. P can only combine with P via addition (multilinear ops).

### Comms operators -- type signatures

**all_gather**: `V -> R|I` or `S(i) -> R|I`
**all_reduce**: `P|V -> R|I` (V is implicitly reinterpreted as P)
**reduce_scatter**: `P|V -> V` or `P|V -> S(i)` (V is implicitly reinterpreted as P)
**all_to_all**: `V -> V` or `S(i) -> S(j)`

**reinterpret(src, dst)**: Changes type, no-op on local tensor (may change semantic value). No comms in forward; backward may need comms.

**convert(src, dst)**: Changes type while preserving semantic value, no comms (may zero out ranks or slice locally). Backward may need comms.

**redistribute(src, dst)**: Semantics-preserving type change that allows comms.

### State transition table (which op for src -> dst)

```text
         dst:  R                  I                  V                P
src: R         -                  convert(R,I)       reinterpret(R,V) reinterpret(R,P)
                                                     convert(R,V)     convert(R,P)
     I         convert(I,R)       -                  reinterpret(I,V) convert(I,P)
                                                     convert(I,V)
     V         all_gather(R)      all_gather(I)      all_to_all()     reinterpret(V,P)
                                                                      convert(V,P)
     P         all_reduce(R)      all_reduce(I)      reduce_scatter() -
```

### Forward-backward pairs

```text
Forward                    Backward
convert(R,I): R->I         convert(I,P): I->P
reinterpret(R,V): R->V     reinterpret(V,P): V->P
convert(R,V): R->V         convert(V,P): V->P
reinterpret(R,P): R->P     reinterpret(R,P): R->P  (self-dual)
convert(R,P): R->P         convert(R,P): R->P      (self-dual)
convert(I,R): I->R         all_reduce(I): P->I
convert(I,V): I->V         all_gather(I): V->I
convert(I,P): I->P         convert(R,I): R->I
all_gather(R): V->R        reduce_scatter(): P->V
all_gather(I): V->I        convert(I,V): I->V
all_to_all(): V->V         all_to_all(): V->V       (self-dual, swap src/dst)
reinterpret(V,P): V->P     reinterpret(R,V): R->V
convert(V,P): V->P         convert(R,V): R->V
all_reduce(R): P->R        all_reduce(R): P->R      (self-dual)
all_reduce(I): P->I        convert(I,R): I->R
reduce_scatter(): P->V     all_gather(R): V->R
```

### reinterpret vs convert

Both are no-comms, but:
- **reinterpret**: local tensor unchanged, semantic value may change (e.g., `reinterpret(R,P)` scales value by mesh size)
- **convert**: semantic value preserved, local tensor may change (e.g., `convert(R,P)` zeros non-rank-0 tensors)

### expert_mode

Some `reinterpret`/`convert` operations require `expert_mode=True` because they are rarely wanted in forward code (they exist primarily as backward passes for other operations):

- **`convert(R,I)`**, **`reinterpret(R,I)`**, **`reinterpret(I,R)`**: Obviously unusual forward semantics (use `convert(I,R)` for the common I->R direction)
- **`reinterpret(R,P)`**, **`convert(I,P)`**: Obviously unusual forward semantics
- **`reinterpret(R,V)`**, **`convert(V,P)`** / **`convert(S(i),P)`**: Legitimate backwards, unlikely forwards

`redistribute` and autograd backwards bypass this gate automatically.

Common forward transitions (no `expert_mode` needed):

```text
I <---> R          convert (both directions)
R <---> V          convert(R,V) / all_gather(V,R)
P  ---> R          all_reduce
P  ---> V          reduce_scatter
V  ---> V          all_to_all
```

### Global SPMD additions

- **PartitionSpec**: tuple matching tensor rank; each entry lists mesh axes sharding that dim. E.g., `f32[8,16@tp]` means dim 1 sharded by "tp".
- **Shard propagation**: per-operator rules (e.g., einsum: mesh axes consistent across operands; contracted dims must not be sharded unless `out_partial_axes` is specified).
- **Explicit partial**: when a contracted dim is sharded (e.g., `blf,fd->bld` with `f@tp`), you must pass `out_partial_axes='tp'` -- partial is never implicit.
- **redistribute(src_spec, dst_spec)**: plans multi-axis collective sequences with flattened communicators.
- Only `S(i)` (not `V`) variants of comms have global SPMD shard propagation rules; use `local_map`/`shard_map` for `V` variants.
