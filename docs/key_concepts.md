# Key concepts

## Device mesh

A **device mesh** is a grid of devices with named axes that let you describe
how communication occurs in your program.  A **mesh axis** identifies one
dimension of the device mesh -- when you do a collective on a mesh axis, the
mesh axis says what ranks communicate with each other: when doing
communication across a mesh axis, only devices in the same "fiber" communicate
with each other.  `spmd_types` is all about describing how data/gradients vary
across each mesh axis in a device mesh.

### Model versus compute mesh

Confusingly, there can be multiple device meshes in a program.  SPMD types
should be used with the "computation mesh" of your program: the mesh which
describes the organization of computation on activations.  This is distinct
from the "model mesh", which describes how your weights are stored across
nodes, e.g., that FSDP cares about.

This can be described more clearly with an example.  Suppose that we are
training a model with hybrid FSDP (split into separate `dp_replicate` and
`dp_shard` axes), context parallel (`cp`) and tensor parallel (`tp`).  There
are two distinct meshes in play here:

* **Model mesh.**  FSDP communications need to understand how weights are
  replicated and sharded across the device mesh: we must all-gather and
  reduce-scatter across the sharding axis, and do a final all-reduce across
  the replicate axis.  In the parallelism setup above, replicate axis is
  `dp_replicate`, while the shard axis is the combination of `dp_shard` and
  `cp` (torchtitan calls this `dp_shard_cp`).

* **Compute mesh.**  When computing on activations, we don't care about the
  fine details of FSDP storage: we simply are interested in how our input data
  and tensor parallel weights have been sharded.  Concretely, the data
  parallel axis is the combined `dp_replicate` and `dp_shard` (you should call
  this combined mesh axis `dp`), the context parallel axis is `cp`, and the
  tensor parallel axis is `tp`.

These two meshes cover the same devices, but they emphasize different internal
structure.  You should always use the compute mesh when writing SPMD types.

### Current mesh

The **current mesh** is conceptually the mesh that your compute is operating
on.  In traditional PyTorch code working with ProcessGroup, this mesh is
implicit and is inferred by inspecting what comms are executed in a codepath;
for DTensor code, this mesh is propagated in a data-dependent way on the
DTensor objects themselves.

`spmd_types` asks you to directly specify the current mesh which all compute
implicitly operates under, using the context manager `spmd.set_current_mesh`.
You can pass it a direct PyTorch `DeviceMesh`:

```python
from torch.distributed.device_mesh import init_device_mesh
import spmd_types as spmd

mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
with spmd.set_current_mesh(mesh):
    ...
```

Or you can pass a dictionary of string axis names to `MeshAxis`.  (This is
useful if you aren't using `DeviceMesh` to manage creation of your process
groups.)  Construct a `MeshAxis` from a `ProcessGroup` with `MeshAxis.of`:

```python
with spmd.set_current_mesh({
    "dp": spmd.MeshAxis.of(dp_pg),
    "tp": spmd.MeshAxis.of(tp_pg),
}):
    ...
```

This context manager should enclose the region of code you are type checking.
You can nest `set_current_mesh` to change the current mesh of your program;
this is most commonly done in expert parallelism, when a different mesh is used
for expert parallel communications.  SPMD types are always computed relative
to the current mesh; if a tensor is on a different mesh than the current mesh,
we will attempt to implicitly reinterpret it onto the current mesh (see
`reinterpret_mesh`).

### Process groups versus mesh axis

There are two classes of APIs in `spmd_types`: APIs that actually do runtime
collectives, and APIs that work with types and otherwise have no runtime
effect.  You must pass a `ProcessGroup` to the runtime collective APIs (e.g.,
`spmd.all_gather`), as there can be potentially multiple `ProcessGroup` with
different `pg_options` for a mesh axis, and it matters which specific
`ProcessGroup` is used for the communication:

```python
# Example: with a ProcessGroup you created directly
spmd.all_reduce(x, tp_pg, src=spmd.P, dst=spmd.R)

# Example: with a ProcessGroup obtained from a DeviceMesh
tp_group = mesh.get_group("tp")
spmd.all_reduce(x, tp_group, src=spmd.P, dst=spmd.R)
```

For all other APIs (e.g., `spmd.assert_type`), we accept several ways to spell
an SPMD type.  While you can still use a `ProcessGroup` directly in this
context as well, the recommended method is to use strings, which will be
looked up per `spmd.set_current_mesh`:

```python
with spmd.set_current_mesh(mesh):
    spmd.assert_type(x, {"tp": spmd.R})
```

If you pass a string to an SPMD API that supports being used as a decorator
(like `local_map`), the string will be interpreted per the current mesh at
runtime (not decorator time).

## Fake process group

You should run typechecking in a single process (not in a production training
run).  However, the single process must be configured "as if" it were running a
real parallel configuration (e.g., you need to actually execute all the
collectives that the parallelism needs!)  In PyTorch, you can do this
conveniently by using fake process groups (fake PG), which stubs out
collectives to not do real communications when you invoke them.

```python
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore

dist.init_process_group(backend="fake", rank=0, world_size=8192)
mesh = init_device_mesh("cuda", (256, 4, 8), mesh_dim_names=("dp", "cp", "tp"))
with spmd.set_current_mesh(mesh):
    ...
dist.destroy_process_group()
```

We recommend having fake process group support directly baked into your trainer
script; see `--comm.mode fake_backend` for an implementation of this in
`torchtitan`.

The simplest way to retrofit fake PG support into a training codebase is to
switch to a fake PG, but keep everything else the same: in particular, you
would be running real kernels on exactly the same hardware type that you would
have done the real training run on.  If you have any data-dependent kernels
(e.g., indexing or host-to-device syncs), you need to add some fake PG specific
branches to fill in data with reasonable values.  If you are not familiar with
the modeling code, LLMs are quite good at doing this, highly recommended!
If you have kernels using symmetric memory, you will need to explicitly stub
the actual kernels (as fake PG only renders classic NCCL collectives inert.)

If your codebase supports fake tensors, you can also run with fake PG and fake
tensors, in this situation, you can potentially even run typechecking on a CPU
only machine (good for CI!)

### Memory debugging with fake PG

A benefit for running with real kernels is that you can actually exactly match
the memory allocation pattern that would happen on a real run (good for
debugging fragmentation problems!)  There may be slight mismatches but, once
again, LLMs are good at comparing real and fake PG profiles and adding extra
allocations as needed to match.  Here is sample script for reading memory
snapshot profiles:

```python
import pickle

def summarize(path):
    snap = pickle.load(open(path, 'rb'))
    segs = snap['segments']
    total = sum(s['total_size'] for s in segs)
    alloc = sum(
        sum(b['size'] for b in s.get('blocks', []) if b['state'] == 'active_allocated')
        for s in segs
    )
    return len(segs), total / 1e9, alloc / 1e9

# Should match exactly for LOCAL_DEBUG configs
real = summarize('real.pickle')
fake = summarize('fake.pickle')
print('real:', real)
print('fake:', fake)
```

You're looking for reserved and allocated memory to match exactly.  If there is
a small mismatch, you can often eliminate it by adding extra allocations to the
fake PG path to mimic what the real code does.

## MPMD

SPMD types, true to its name, is designed for single-program multiple-data
(SPMD) parallelism.  This means it is inappropriate for checking safety
properties of MPMD parallelism; most notably pipeline parallelism.

That being said, pipeline parallelism is typically combined with SPMD
parallelism,
and it is still profitable to check the SPMD parallel bits.  The easiest setup,
if possible, is to just SPMD typecheck the entire model, without PP, prior to
splitting into PP stages.  If you can only typecheck an individual PP stage,
note that you can only really SPMD typecheck the forward stages.
