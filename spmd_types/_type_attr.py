# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Low-level helpers for reading/writing the SPMD type attribute on tensors.

This module is intentionally minimal so that both ``_checker`` and
``_raw_dist`` can depend on it without creating a cycle.
"""

from __future__ import annotations

import torch
from spmd_types.types import (
    DeviceMeshAxis,
    format_axis,
    LocalSpmdType,
    normalize_axis,
    PerMeshAxisLocalSpmdType,
)

# Attribute name for storing SPMD types on tensors.
#
# The attribute holds one of two states:
#   - absent: truly untyped (created outside SpmdTypeMode)
#   - dict:   real LocalSpmdType (including {} for typed-but-unknown-on-all-axes)
_LOCAL_TYPE_ATTR = "_local_type"


def get_local_type(value: object) -> LocalSpmdType:
    """Get the SPMD types stored on a tensor or Scalar.

    Returns an empty dict if the object has no SPMD type annotations,
    meaning all axes are unknown.
    """
    result = getattr(value, _LOCAL_TYPE_ATTR, None)
    if result is None:
        return {}
    return result


def set_local_type(tensor: torch.Tensor, type: LocalSpmdType) -> torch.Tensor:
    """Set SPMD type on a tensor (internal). Returns the tensor for chaining.

    The caller is responsible for validating ``type`` before calling this.
    """
    setattr(tensor, _LOCAL_TYPE_ATTR, type)
    return tensor


def get_axis_local_type(
    tensor: torch.Tensor, axis: DeviceMeshAxis
) -> PerMeshAxisLocalSpmdType:
    """Get the SPMD type for a specific mesh axis.

    Raises:
        ValueError: If the tensor has no SPMD type annotations, or if the
            axis is not present in the tensor's SPMD type dict.

    Args:
        tensor: The tensor to query.
        axis: The mesh axis to look up (MeshAxis or ProcessGroup).
    """
    result = maybe_get_axis_local_type(tensor, axis)
    if result is None:
        axis = normalize_axis(axis)
        raise ValueError(
            f"Axis {format_axis(axis)} not found in tensor's SPMD type. "
            f"Tensor has axes: {set(get_local_type(tensor).keys())}"
        )
    return result


def maybe_get_axis_local_type(
    tensor: torch.Tensor, axis: DeviceMeshAxis
) -> PerMeshAxisLocalSpmdType | None:
    """Get the SPMD type for a specific mesh axis, or None if not found.

    Like ``get_axis_local_type`` but returns None instead of raising when
    the axis is not present in the tensor's SPMD type dict.

    Args:
        tensor: The tensor to query.
        axis: The mesh axis to look up (MeshAxis or ProcessGroup).
    """
    axis = normalize_axis(axis)
    local_type = get_local_type(tensor)
    return local_type.get(axis)
