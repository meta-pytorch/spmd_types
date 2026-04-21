"""
Test that importing spmd_types does not transitively pull in the checker.

The checker is heavy (imports torch.overrides, DTensor internals, etc.) and
should only be loaded when the user explicitly imports spmd_types.checker.
"""

import subprocess
import sys
import unittest


class TestImportIsolation(unittest.TestCase):
    def test_spmd_types_does_not_import_checker(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import spmd_types; import sys; "
                "assert 'spmd_types._checker' not in sys.modules, "
                "'spmd_types._checker was transitively imported by spmd_types'",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"spmd_types transitively imported _checker:\n{result.stderr}",
        )

    def test_checker_is_importable(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import spmd_types.checker; import sys; "
                "assert 'spmd_types._checker' in sys.modules, "
                "'spmd_types._checker should be loaded after importing spmd_types.checker'",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"spmd_types.checker failed to import _checker:\n{result.stderr}",
        )
