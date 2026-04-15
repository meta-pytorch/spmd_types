"""Test utilities for SPMD types."""

from __future__ import annotations

import contextlib

import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.testing._internal.distributed.fake_pg import FakeStore


@contextlib.contextmanager
def fake_pg(world_size: int = 4) -> ProcessGroup:
    """Context manager that sets up and tears down a fake process group.

    Args:
        world_size: The number of ranks to simulate (default: 4).
    """
    store = FakeStore()
    dist.init_process_group(backend="fake", rank=0, world_size=world_size, store=store)
    try:
        yield dist.distributed_c10d._get_default_group()
    finally:
        dist.destroy_process_group()
