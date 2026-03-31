# spmd_types

A type system for distributed (SPMD) tensor computations in PyTorch.

spmd_types tracks per-mesh-axis types on tensors -- Replicate (R), Invariant (I),
Varying (V), Partial (P), and Shard (S) -- and enforces type-correct transitions
through collective operations and local rewrites. It catches distributed
programming errors at development time without requiring a GPU cluster.

## Installation

```bash
pip install spmd-types
```

## Quick start

```python
import torch
import torch.distributed as dist
from spmd_types import R, V, P, S, typecheck, assert_type, all_reduce

dist.init_process_group(backend="nccl")
pg = dist.distributed_c10d._get_default_group()

with typecheck():
    x = torch.randn(4, device="cuda")
    assert_type(x, {pg: P})       # x is partial (pending sum)
    y = all_reduce(x, pg, src=P, dst=R)  # sum across ranks
    assert_type(y, {pg: R})       # y is now replicated
```

## Documentation

See [DESIGN.md](DESIGN.md) for the full type system specification, including
type inference rules, collective signatures, and forward-backward pairs.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
