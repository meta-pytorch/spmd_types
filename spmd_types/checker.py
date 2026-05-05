# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types._dtensor_checker as _dtensor_checker  # noqa: F401
from spmd_types._checker import (  # noqa: F401
    _SpmdTypeBackwardCompatibleMode as SpmdTypeMode,
    typecheck,
)
from spmd_types._state import is_type_checking, no_typecheck  # noqa: F401
from spmd_types.runtime import local_map  # noqa: F401
