# _checker/ -- SPMD Type Inference Engine

This package contains the type inference engine that runs inside
`typecheck()`. It is the heaviest dependency in spmd_types and
changes here trigger the most downstream CI tests.

## Architecture

The runtime annotation API (assert_type, mutate_type,
register_autograd_function, etc.) lives in `spmd_types.runtime` and
does NOT depend on this package. Model code that only needs to
annotate tensors should depend on `:runtime` instead of `:_checker`.

```text
runtime tier (stable, lightweight)
  types.py, _mesh_axis.py, _state.py, _type_attr.py, _frame.py,
  _traceback.py, _dist.py, _scalar_sentinel.py, runtime.py,
  _collectives.py, _local.py, _scalar.py, _raw_dist.py, _vmap.py,
  _mesh_region.py, _reinterpret_mesh.py

checker tier (this package -- the inference engine)
  _checker/__init__.py  -- TorchFunctionMode, type inference, shard propagation
```

Only `_checker/` depends on the runtime tier, never the reverse.

## What lives here

Currently everything is in `__init__.py`. As the codebase grows,
the plan is to split into focused modules:

- **Type inference**: `OpLinearity`, `_infer_local_type_for_axis`,
  `infer_output_type`, cross-mesh advice, fix suggestion engine
- **Op registry**: `_OpSpec`, per-op type specs, type-level
  decompositions for compound ops
- **Global SPMD / shard propagation**: `_ShardPropagator`,
  `_collect_shard_axes`, DTensor integration. Shard prop rules for
  PyTorch builtins will likely get their own file here.
- **Partial propagation**: `out_partial` handling in type inference.
  Currently inlined in `_infer_local_type_for_axis_raw`; may be
  extracted as the rules grow.
- **TorchFunctionMode**: `_SpmdTypeMode`, `typecheck()`,
  `no_typecheck()`, passthrough op set, autograd Function.apply patch
