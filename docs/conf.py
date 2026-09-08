# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytorch_sphinx_theme2

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

exclude_patterns = ["_build"]

html_theme = "pytorch_sphinx_theme2"
html_theme_path = [pytorch_sphinx_theme2.get_html_theme_path()]
html_baseurl = "https://meta-pytorch.org/spmd_types/"
html_theme_options = {
    "navigation_with_keys": False,
    "show_lf_header": False,
    "show_lf_footer": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/meta-pytorch/spmd_types",
            "icon": "fa-brands fa-github",
        },
    ],
    "use_edit_page_button": True,
    "navbar_center": "navbar-nav",
}

theme_variables = pytorch_sphinx_theme2.get_theme_variables()
templates_path = [
    "_templates",
    os.path.join(os.path.dirname(pytorch_sphinx_theme2.__file__), "templates"),
]
html_context = {
    "theme_variables": theme_variables,
    "display_github": True,
    "github_url": "https://github.com",
    "github_user": "meta-pytorch",
    "github_repo": "spmd_types",
    "github_version": "main",
    "doc_path": "docs",
    "library_links": theme_variables.get("library_links", []),
    "community_links": theme_variables.get("community_links", []),
    "language_bindings_links": html_theme_options.get("language_bindings_links", []),
}

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
