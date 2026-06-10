# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test that runnable code samples in docs/ run correctly."""

import re
import textwrap
import unittest
from pathlib import Path

from spmd_types._mesh_axis import _reset


def _extract_section_blocks(doc_name, heading):
    """Extract python code blocks under *heading* in docs/<doc_name>."""
    doc = Path(__file__).resolve().parent.parent / "docs" / doc_name
    content = doc.read_text()

    # Collect lines after the heading until the next markdown heading of
    # the same or higher level.  Track code fence state so that Python
    # comments inside code blocks are not mistaken for headings.
    level = len(heading) - len(heading.lstrip("#"))
    lines = content.splitlines(keepends=True)
    section_lines = []
    in_section = False
    in_fence = False
    for line in lines:
        if line.rstrip() == heading:
            in_section = True
            continue
        if not in_section:
            continue
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and re.match(rf"^#{{1,{level}}} ", line):
            break
        section_lines.append(line)
    if not in_section:
        raise AssertionError(f"Could not find section {heading!r} in {doc_name}")
    section = "".join(section_lines)
    return re.findall(r"```python\n(.*?)```", section, re.DOTALL)


class TestLocalSpmdTypesDoc(unittest.TestCase):
    def test_putting_it_all_together_runs(self):
        """The end-to-end example in local_spmd_types.md typechecks."""
        blocks = _extract_section_blocks(
            "local_spmd_types.md", "### Putting it all together"
        )
        self.assertEqual(
            len(blocks), 1, "Expected exactly one python block in the section"
        )
        code = textwrap.dedent(blocks[0])
        _reset()
        try:
            exec(compile(code, "local_spmd_types.md:putting_it_all_together", "exec"), {})
        finally:
            _reset()


if __name__ == "__main__":
    unittest.main()
