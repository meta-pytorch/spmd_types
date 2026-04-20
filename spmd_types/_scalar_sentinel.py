# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scalar sentinel for SPMD type inference.

Separated into its own module so that both ``_checker.py`` and
``runtime.py`` can reference the sentinel without circular imports.
"""

from __future__ import annotations


class _ScalarType:
    """Sentinel for Python scalars in SPMD type inference.

    We assume a Python scalar is the same value on all ranks and carries no
    gradient.  This assumption can be wrong: a rank-dependent scalar (e.g.
    ``rank / world_size``) is really Varying, and a scalar that is the sum of
    per-rank contributions is really Partial.  We choose to assume Replicate
    anyway because (1) the vast majority of scalars in practice *are* the same
    on every rank, and (2) requiring users to annotate every literal ``2.0``
    or ``eps`` would be extremely noisy for little safety gain.  A future
    improvement could add an explicit wrapper (e.g. ``Varying(scalar)``) for
    the rare rank-dependent case; until then Dynamo tracing with SymInt offers
    partial safety.

    Given the assumption, _Scalar is compatible with both R and I -- analogous
    to how Python scalars are "weak" in dtype promotion and do not determine
    the specific dtype.

    _Scalar participates in linearity validation (P + scalar is affine, not
    linear) but does not influence the inferred output type (R vs I).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "Scalar"


_Scalar = _ScalarType()
