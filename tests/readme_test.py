# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test that the README code samples run correctly."""

import re
import textwrap
import unittest
from pathlib import Path

from spmd_types._mesh_axis import _reset


class TestReadmeQuickStart(unittest.TestCase):
    def _extract_python_blocks(self):
        """Extract all python code blocks after '## Quick start' in README.md."""
        readme = Path(__file__).resolve().parent.parent / "README.md"
        content = readme.read_text()

        # Get everything after "## Quick start" until the next ## heading
        section = re.search(r"## Quick start\s*(.*?)(?=\n## |\Z)", content, re.DOTALL)
        self.assertIsNotNone(section, "Could not find Quick start section in README.md")

        blocks = re.findall(r"```python\n(.*?)```", section.group(1), re.DOTALL)
        self.assertGreater(len(blocks), 0, "No python code blocks found")
        return blocks

    def test_quick_start_runs(self):
        """Run each python code block in the Quick start section."""
        for i, block in enumerate(self._extract_python_blocks()):
            code = textwrap.dedent(block)
            _reset()
            try:
                exec(compile(code, f"README.md:block{i}", "exec"), {})
            finally:
                _reset()


if __name__ == "__main__":
    unittest.main()
