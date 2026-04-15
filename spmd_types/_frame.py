"""User-frame resolution for SPMD type trace logging.

Walks the Python call stack to find the first frame that belongs to user
code rather than spmd_types internals or torch infrastructure.
"""

from __future__ import annotations

import os
import sys

# Directory containing this package's private modules (_checker.py, etc.).
_SPMD_DIR = os.path.dirname(__file__)


def _is_internal_frame(filename: str) -> bool:
    """Return True if *filename* belongs to spmd_types internals or torch."""
    # Skip torch internals (torch/, torch/overrides.py, etc.).
    for mod_name in ("torch",):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "__file__") and mod.__file__:
            if filename.startswith(os.path.dirname(mod.__file__)):
                return True
    # Skip spmd_types private modules (_checker.py, _collectives.py, etc.)
    # but NOT test files or user code that happens to live in the same dir.
    if filename.startswith(_SPMD_DIR):
        basename = os.path.basename(filename)
        if basename.startswith("_"):
            return True
    return False


def _abbreviate_path(filename: str) -> str:
    """Shorten *filename* by stripping the longest matching ``sys.path`` prefix."""
    best = filename
    for entry in sys.path:
        if not entry:
            continue
        # Ensure the prefix ends with a separator so we don't match partial
        # directory names (e.g. /foobar matching /foo).
        prefix = entry if entry.endswith(os.sep) else entry + os.sep
        if filename.startswith(prefix):
            rel = filename[len(prefix) :]
            if len(rel) < len(best):
                best = rel
    return best


def _get_user_frame() -> str:
    """Walk the stack to find the first frame outside spmd_types / torch.

    Returns a ``"path/to/file.py:lineno"`` string for the first frame
    whose filename does not belong to the spmd_types private modules or
    the torch package.  The path is abbreviated relative to ``sys.path``
    so that trace output stays concise.
    """
    frame = sys._getframe(0)
    while frame is not None:
        if not _is_internal_frame(frame.f_code.co_filename):
            return f"{_abbreviate_path(frame.f_code.co_filename)}:{frame.f_lineno}"
        frame = frame.f_back
    return "<unknown>"
