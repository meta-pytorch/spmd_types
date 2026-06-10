# `spmd_types`

All public functions and types are importable from the top-level `spmd_types` package:

```python
import spmd_types as spmd
```

## Types

Core type hierarchy for distributed tensor expressions.

### Per-Mesh-Axis Types

```{eval-rst}
.. autoclass:: spmd_types.types.PerMeshAxisLocalSpmdType
   :members: R, I, V, P, backward_type

.. autoclass:: spmd_types.types.Shard
```

### PartitionSpec

```{eval-rst}
.. autoclass:: spmd_types.types.PartitionSpec
```

### TensorSharding

```{eval-rst}
.. autoclass:: spmd_types.types.TensorSharding
```

### Scalar

```{eval-rst}
.. autoclass:: spmd_types.Scalar
```

### SpmdTypeError

```{eval-rst}
.. autoclass:: spmd_types.types.SpmdTypeError
```

## Mesh

Device mesh setup and axis management.

```{eval-rst}
.. autofunction:: spmd_types.set_current_mesh

.. autofunction:: spmd_types.current_mesh

.. autoclass:: spmd_types.MeshAxis

.. autofunction:: spmd_types.reinterpret_mesh

.. autofunction:: spmd_types.set_dist
```

## Runtime

Lightweight runtime API for SPMD type annotations. This module does
**not** depend on the type inference engine (`_checker`), so it can
be imported in model code without pulling in the heavy checker logic.

```{eval-rst}
.. autofunction:: spmd_types.runtime.assert_type

.. autofunction:: spmd_types.runtime.assert_local_type

.. autofunction:: spmd_types.runtime.assert_type_like

.. autofunction:: spmd_types.runtime.mutate_type

.. autofunction:: spmd_types.runtime.has_local_type

.. autofunction:: spmd_types.runtime.get_partition_spec

.. autofunction:: spmd_types.runtime.trace

.. autofunction:: spmd_types.runtime.local_map

.. autofunction:: spmd_types.runtime.local

.. autofunction:: spmd_types.runtime.register_autograd_function

.. autofunction:: spmd_types.runtime.register_local_autograd_function

.. autofunction:: spmd_types.runtime.register_decomposition
```

## Local Operations

No-comms type transitions: change the SPMD type without performing any communication in the forward pass.

```{eval-rst}
.. autofunction:: spmd_types.reinterpret

.. autofunction:: spmd_types.convert

.. autofunction:: spmd_types.shard

.. autofunction:: spmd_types.invariant_to_replicate
```

## Collective Operations

Wrappers around the classic NCCL collectives provided by
`torch.distributed`. Unlike those functions, all of these collectives are
differentiable and have typing rules under `spmd_types`.

```{eval-rst}
.. autofunction:: spmd_types.all_reduce

.. autofunction:: spmd_types.all_gather

.. autofunction:: spmd_types.reduce_scatter

.. autofunction:: spmd_types.all_to_all

.. autofunction:: spmd_types.redistribute

.. autofunction:: spmd_types.unshard
```
