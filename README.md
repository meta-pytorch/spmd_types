# `spmd_types`

A type system for distributed (SPMD) tensor computations in PyTorch.  This
package provides two type systems:

* **Local SPMD types**, which allow you to use Megatron-style differentiable
  collectives in a safe way by tracking whether or not your backward gradients
  are pending reduction or not.

* **Global SPMD types**, a DTensor-like abstraction for writing code that has
  the same semantics whether run on a single device or in a distributed
  fashion, but with explicit communication operations so you are never
  guessing when a redistribute occurs.

In both cases, the SPMD types makes it possible for you to check that your
code computes correct gradients (local SPMD) or gives equivalent results
across different parallelizations (global SPMD), without having to actually
run a full E2E distributed training run to check for loss matching.

The goal of this package is to provide a flexible type system that can
typecheck realistic training code.  We have used local SPMD types to typecheck
a realistic pretraining codebase, and global SPMD types is actively under
construction!

## Installation

```bash
pip install spmd_types
```

## Quick start

```python
import torch
import torch.distributed as dist
import spmd_types as spmd
import spmd_types.checker
from torch.distributed.device_mesh import init_device_mesh

# Set up a fake process group (no GPUs needed)
dist.init_process_group(backend="fake", rank=0, world_size=8)
mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "tp"))
dp = mesh.get_group("dp")
tp = mesh.get_group("tp")

with spmd.set_current_mesh(mesh), spmd.checker.typecheck():
    x = torch.randn(4)
    spmd.assert_type(x, {dp: spmd.R, tp: spmd.P})       # R on dp, partial on tp
    y = spmd.all_reduce(x, tp, src=spmd.P, dst=spmd.R)  # sum across tp ranks
    spmd.assert_type(y, {dp: spmd.R, tp: spmd.R})       # now replicated everywhere
    z = torch.mul(y, y)                                 # type inference: R * R -> R
    spmd.assert_type(z, {dp: spmd.R, tp: spmd.R})

dist.destroy_process_group()
```

## Documentation

See [Local SPMD types](docs/local_spmd_types.md) for a hands-on guide on
porting Megatron-derived training frameworks, including the Megatron-to-spmd_types
function mapping table and advice on Invariant vs Replicate.

See [Design](docs/design.md) for the full type system specification, including
local vs global SPMD modes, collective signatures with diagrams, forward-backward
pairs, expert mode, cross-mesh compatibility, and partition spec redistribute.

## License

BSD 3-Clause License. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.
