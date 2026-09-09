# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coverage bookkeeping for ``spmd_typecheck`` hooks.

A leaf module so that both ``spmd_types.rules`` (which runs the hook and
checks coverage) and ``spmd_types.runtime`` (whose ``assert_type`` counts as
covering a tensor) can depend on it without a cycle.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

import torch


class _Coverage:
    """The ids of the tensors a running hook has accounted for."""

    def __init__(self) -> None:
        self.touched: set[int] = set()


# A thread-local rather than a ContextVar: ``assert_type`` runs under Dynamo
# tracing in compiled code, and Dynamo cannot trace ``ContextVar.get``.
_tls = threading.local()


def _cov() -> _Coverage | None:
    """The coverage record of the hook currently running, if any."""
    return getattr(_tls, "coverage", None)


def _touch(*values: object) -> None:
    """Record every tensor among ``values`` as accounted for."""
    cov = _cov()
    if cov is None:
        return
    for v in values:
        if isinstance(v, torch.Tensor):
            cov.touched.add(id(v))


def _mark_asserted(value: object) -> None:
    """``assert_type`` on an input counts as accounting for it."""
    _touch(value)


@contextmanager
def _coverage() -> Iterator[_Coverage]:
    """Run a hook with a fresh coverage record."""
    prev = _cov()
    cov = _Coverage()
    _tls.coverage = cov
    try:
        yield cov
    finally:
        _tls.coverage = prev
