"""Patchable reference to torch.distributed.

Callers should ``import spmd_types._dist as _dist`` and access
``_dist.dist`` to use the current dist implementation. Do NOT use
``from ... import dist`` as that captures a snapshot.
"""

from __future__ import annotations

import torch.distributed as _torch_dist

dist = _torch_dist


def set_dist(module) -> None:
    """Replace the dist implementation (e.g., with comms_wrapper).

    Args:
        module: A module providing the same interface as ``torch.distributed``.
            Pass None to reset to the default ``torch.distributed``.
    """
    global dist
    if module is None:
        module = _torch_dist
    dist = module
