# spmd_types

A type system for distributed (SPMD) tensor computations in PyTorch.

spmd_types tracks per-mesh-axis types on tensors -- Replicate (R), Invariant (I),
Varying (V), Partial (P), and Shard (S) -- and enforces type-correct transitions
through collective operations and local rewrites. It catches distributed
programming errors at development time without requiring a GPU cluster.

## Installation

```bash
pip install spmd_types
```

## Quick start

```python
import torch
import torch.distributed as dist
import spmd_types as spmd
from torch.distributed.device_mesh import init_device_mesh

# Set up a fake process group (no GPUs needed)
dist.init_process_group(backend="fake", rank=0, world_size=8)
mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "tp"))
dp = mesh.get_group("dp")
tp = mesh.get_group("tp")

with spmd.set_current_mesh(mesh), spmd.typecheck():
    x = torch.randn(4)
    spmd.assert_type(x, {dp: spmd.R, tp: spmd.P})  # R on dp, partial on tp
    y = spmd.all_reduce(x, tp, src=spmd.P, dst=spmd.R)  # sum across tp ranks
    spmd.assert_type(y, {dp: spmd.R, tp: spmd.R})   # now replicated everywhere
    z = torch.mul(y, y)                              # type inference: R * R -> R
    spmd.assert_type(z, {dp: spmd.R, tp: spmd.R})

dist.destroy_process_group()
```

## Documentation

See [MEGATRON_QUICKSTART.md](MEGATRON_QUICKSTART.md) for a guide on porting
Megatron-derived training frameworks to use spmd_types.

See [DESIGN.md](DESIGN.md) for the full type system specification, including
type inference rules, collective signatures, and forward-backward pairs.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
