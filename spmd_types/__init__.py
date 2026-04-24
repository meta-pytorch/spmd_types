# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# spmd_types package
from __future__ import annotations

from spmd_types._backward_hooks import (  # noqa: F401
    register_local_backward_hook,
)
from spmd_types._collectives import (  # noqa: F401
    all_gather,
    all_reduce,
    all_to_all,
    redistribute,
    reduce_scatter,
    unshard,
)
from spmd_types._dist import set_dist  # noqa: F401
from spmd_types._dtensor import (  # noqa: F401
    dtensor_placement_to_spmd_type,
    spmd_redistribute,
    spmd_type_to_dtensor_placement,
)
from spmd_types._local import (  # noqa: F401
    convert,
    invariant_to_replicate,
    reinterpret,
    shard,
)
from spmd_types._mesh import set_current_mesh  # noqa: F401
from spmd_types._mesh_axis import MeshAxis  # noqa: F401

# reinterpret_mesh lives in its own module
from spmd_types._reinterpret_mesh import reinterpret_mesh  # noqa: F401
from spmd_types._scalar import Scalar  # noqa: F401
from spmd_types._state import (  # noqa: F401
    current_mesh,
    is_type_checking,
    no_typecheck,
)
from spmd_types._traceback import traceback_filtering  # noqa: F401
from spmd_types._type_attr import (  # noqa: F401
    get_axis_local_type,
    get_local_type,
    maybe_get_axis_local_type,
)

# Collectives and operations -- runtime API (no _checker dependency)
from spmd_types.runtime import (  # noqa: F401
    assert_local_type,
    assert_type,
    assert_type_like,
    mutate_type,
    register_autograd_function,
    register_local_autograd_function,
    trace,
)

# Types
from spmd_types.types import (  # noqa: F401
    DimSharding,
    I,
    Invariant,
    LocalSpmdType,
    normalize_axis,
    normalize_mesh,
    P,
    Partial,
    PartitionSpec,
    PerMeshAxisLocalSpmdType,
    PerMeshAxisSpmdType,
    PerMeshAxisSpmdTypes,
    R,
    Replicate,
    S,
    Shard,
    SpmdTypeError,
    TensorSharding,
    V,
    Varying,
)


class _TypeCheckingSentinel:
    """Singleton whose bool value reflects whether type checking is active.

    ``bool(TYPE_CHECKING)`` returns True when a ``typecheck()`` context is
    active on the current thread, False otherwise.  This avoids the
    sys.modules replacement trick which breaks torch.compile / Dynamo.
    """

    def __bool__(self) -> bool:
        return is_type_checking()

    def __repr__(self) -> str:
        return f"TYPE_CHECKING({is_type_checking()})"


TYPE_CHECKING = _TypeCheckingSentinel()
