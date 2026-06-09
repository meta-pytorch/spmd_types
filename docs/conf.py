# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

project = "spmd_types"
copyright = "Meta Platforms, Inc."

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

myst_commonmark_compat = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

autodoc_member_order = "bysource"
napoleon_google_docstring = True

# External (PyTorch) types that appear in autodoc'd signatures but have no
# documentation target in this project. We do not host PyTorch's inventory,
# so suppress the nitpicky cross-reference warnings for them rather than
# emit dead links. (Internal type aliases like DeviceMeshAxis are
# intentionally left unsuppressed so missing internal docs stay visible.)
nitpick_ignore = [
    ("py:class", "torch.Tensor"),
    ("py:class", "torch.dtype"),
    ("py:class", "ProcessGroup"),
    ("py:class", "DeviceMesh"),
    ("py:class", "torch.distributed._mesh_layout._MeshLayout"),
]
