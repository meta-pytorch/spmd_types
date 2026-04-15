"""DTensor <-> local tensor boundary typecheck_forward implementations.

This module bridges _checker and _dtensor: it imports from both and wires up
the typecheck_forward methods on _ToTorchTensor / _FromTorchTensor.  Keeping
this in a separate module avoids a circular Buck dependency between _checker
and _dtensor.

Importing this module has the side effect of registering the typecheck_forward
implementations; __init__.py imports it after both _checker and _dtensor.
"""

from __future__ import annotations

import torch
from spmd_types._checker import _TYPECHECK_AUTOGRAD_FUNCTIONS, assert_type
from spmd_types._dtensor import dtensor_placement_to_spmd_type
from spmd_types.types import LocalSpmdType, normalize_axis, Shard, V
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._api import _FromTorchTensor, _ToTorchTensor

if hasattr(torch.distributed.tensor._api, "_normalize_placements_for_grad"):
    _normalize_placements_for_grad = (
        torch.distributed.tensor._api._normalize_placements_for_grad
    )
else:
    from torch.distributed.tensor import (
        Placement as DtPlacement,
        Replicate as DtReplicate,
    )

    def _normalize_placements_for_grad(
        placements: tuple[DtPlacement, ...],
    ) -> tuple[DtPlacement, ...]:
        normalized: list[DtPlacement] = []
        for p in placements:
            if p.is_partial():
                normalized.append(DtReplicate())
            else:
                normalized.append(p)
        return tuple(normalized)


def _placements_to_local_type(mesh, placements, grad_placements):
    """Convert DTensor placements to an SPMD local type dict."""
    assert len(placements) == len(grad_placements), (
        f"placements and grad_placements must have the same length, "
        f"got {len(placements)} and {len(grad_placements)}"
    )
    local_type: LocalSpmdType = {}
    for i, placement in enumerate(placements):
        pg = mesh.get_group(i)
        axis = normalize_axis(pg)
        spmd_type = dtensor_placement_to_spmd_type(placement, grad_placements[i])
        # S(i) decays to V for local type storage.
        if isinstance(spmd_type, Shard):
            spmd_type = V
        local_type[axis] = spmd_type
    return local_type


def _typecheck_forward_to_torch_tensor(*args, **kwargs):
    """typecheck_forward for _ToTorchTensor: run apply, then assert_type."""
    # _ToTorchTensor.apply(input_dtensor, grad_placements)
    input_dtensor = args[0]
    assert isinstance(input_dtensor, DTensor), (
        f"_ToTorchTensor input must be a DTensor, got {type(input_dtensor)}"
    )
    grad_placements = args[1] if len(args) > 1 else None
    if grad_placements is None:
        grad_placements = _normalize_placements_for_grad(input_dtensor.placements)
    result = _ToTorchTensor.apply(*args, **kwargs)
    local_type = _placements_to_local_type(
        input_dtensor.device_mesh, input_dtensor.placements, grad_placements
    )
    assert_type(result, local_type)
    return result


def _typecheck_forward_from_torch_tensor(*args, **kwargs):
    """typecheck_forward for _FromTorchTensor: validate types, then run apply."""
    # _FromTorchTensor.apply(input, device_mesh, placements, run_check,
    #                        shape, stride, grad_placements)
    input_tensor = args[0]
    device_mesh = args[1]
    placements = args[2]
    grad_placements = args[6] if len(args) > 6 else None
    if grad_placements is None:
        grad_placements = _normalize_placements_for_grad(placements)
    expected_type = _placements_to_local_type(device_mesh, placements, grad_placements)
    assert_type(input_tensor, expected_type)
    return _FromTorchTensor.apply(*args, **kwargs)


# =============================================================================
# Registration: wire up the typecheck_forward methods
# =============================================================================

_ToTorchTensor.typecheck_forward = staticmethod(_typecheck_forward_to_torch_tensor)
_FromTorchTensor.typecheck_forward = staticmethod(_typecheck_forward_from_torch_tensor)
_TYPECHECK_AUTOGRAD_FUNCTIONS.add(_ToTorchTensor)
_TYPECHECK_AUTOGRAD_FUNCTIONS.add(_FromTorchTensor)
