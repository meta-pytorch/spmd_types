# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monkey-patch torch.func.vmap to propagate SPMD type annotations.

vmap wraps input tensors as C++-level BatchedTensors, which drops Python
attributes (including ``_local_type``).  This module patches the internal
``_flat_vmap`` function so that:

1. Input types are saved before batching and copied onto the BatchedTensors.
2. The vmapped function runs normally -- TorchFunctionMode (already on the
   stack) sees correctly-typed inputs and propagates types as usual.
3. Output types are captured from the batched outputs and copied onto the
   unbatched results.

The vmap batch dimension is orthogonal to SPMD mesh dimensions, so types
pass through unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from spmd_types._type_attr import get_local_type, set_local_type
from torch._functorch import vmap as _vmap_module  # for _flat_vmap
from torch.utils._pytree import tree_flatten, tree_unflatten

_orig_flat_vmap = None


def _patched_flat_vmap(
    func: Callable,
    batch_size: int,
    flat_in_dims: list[int | None],
    flat_args: list[Any],
    args_spec: Any,
    out_dims: Any,
    randomness: str,
    **kwargs: Any,
) -> Any:
    assert _orig_flat_vmap is not None  # install() must be called first

    # Save input types before C++ wraps them as BatchedTensors.
    input_types = [
        get_local_type(arg) if isinstance(arg, torch.Tensor) else None
        for arg in flat_args
    ]

    output_types: list[dict | None] = []

    def wrapped(*args: Any, **kw: Any) -> Any:
        # Apply saved input types to the BatchedTensors we receive.
        flat_batched, _ = tree_flatten(args)
        for i, (typ, in_dim) in enumerate(zip(input_types, flat_in_dims)):
            if typ and in_dim is not None and i < len(flat_batched):
                set_local_type(flat_batched[i], typ)

        result = func(*args, **kw)

        # Capture output types before _unwrap_batched strips them.
        flat_out, _ = tree_flatten(result)
        for t in flat_out:
            output_types.append(
                get_local_type(t) if isinstance(t, torch.Tensor) else None
            )
        return result

    result = _orig_flat_vmap(
        wrapped,
        batch_size,
        flat_in_dims,
        flat_args,
        args_spec,
        out_dims,
        randomness,
        **kwargs,
    )

    # Restore output types on unbatched results.
    if output_types:
        flat_result, result_spec = tree_flatten(result)
        for i, typ in enumerate(output_types):
            if (
                typ
                and i < len(flat_result)
                and isinstance(flat_result[i], torch.Tensor)
            ):
                set_local_type(flat_result[i], typ)
        result = tree_unflatten(flat_result, result_spec)

    return result


def install() -> None:
    """Install the vmap monkey-patch.  Idempotent."""
    global _orig_flat_vmap

    if _orig_flat_vmap is not None:
        return

    _orig_flat_vmap = _vmap_module._flat_vmap
    _vmap_module._flat_vmap = _patched_flat_vmap


def uninstall() -> None:
    """Remove the vmap monkey-patch.  Idempotent."""
    global _orig_flat_vmap

    if _orig_flat_vmap is None:
        return

    _vmap_module._flat_vmap = _orig_flat_vmap
    _orig_flat_vmap = None
