"""JAX-style traceback filtering for SPMD type errors.

When ``SPMD_TYPES_TRACEBACK_FILTERING`` is set (default ``"auto"``), internal
frames from ``_checker.py``, ``_collectives.py``, torch internals, etc. are
stripped from ``SpmdTypeError`` tracebacks so users see only their own code.

Modes:

- ``off``: no filtering (original traceback)
- ``tracebackhide``: sets ``__tracebackhide__ = True`` on internal frames
  (works with pytest/IPython)
- ``quiet_remove_frames``: rebuilds traceback without internal frames, adds
  a note that frames were filtered
- ``remove_frames``: like ``quiet_remove_frames`` but also attaches the full
  traceback as ``__cause__`` via an ``UnfilteredStackTrace`` wrapper

``auto`` resolves to ``tracebackhide`` when pytest or IPython is detected,
``quiet_remove_frames`` otherwise.
"""

from __future__ import annotations

import functools
import os
import sys
import threading
import traceback
import types
from contextlib import contextmanager
from typing import Optional

from spmd_types._frame import _is_internal_frame
from spmd_types.types import SpmdTypeError

# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------

_FILTERING: str = os.environ.get("SPMD_TYPES_TRACEBACK_FILTERING", "auto")

_VALID_MODES = frozenset(
    {"off", "auto", "tracebackhide", "quiet_remove_frames", "remove_frames"}
)


def _resolve_mode(mode: str) -> str:
    """Resolve ``auto`` to a concrete mode; validate all others."""
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid SPMD_TYPES_TRACEBACK_FILTERING mode: {mode!r}. "
            f"Valid modes: {sorted(_VALID_MODES)}"
        )
    if mode != "auto":
        return mode
    # IPython or pytest present -> tracebackhide
    if "IPython" in sys.modules or "pytest" in sys.modules:
        return "tracebackhide"
    return "quiet_remove_frames"


# ---------------------------------------------------------------------------
# Traceback filtering
# ---------------------------------------------------------------------------


def filter_traceback(
    tb: Optional[types.TracebackType],
) -> Optional[types.TracebackType]:
    """Return a new traceback chain keeping only user (non-internal) frames."""
    out: Optional[types.TracebackType] = None
    frames = list(traceback.walk_tb(tb))
    for f, lineno in reversed(frames):
        if not _is_internal_frame(f.f_code.co_filename):
            out = types.TracebackType(out, f, f.f_lasti, lineno)
    return out


# ---------------------------------------------------------------------------
# Unfiltered wrapper (for ``remove_frames`` mode)
# ---------------------------------------------------------------------------


class UnfilteredStackTrace(Exception):
    """Wrapper carrying the original unfiltered traceback.

    Attached as ``__cause__`` when ``remove_frames`` mode is active so that
    the full traceback is still available for debugging.
    """


# ---------------------------------------------------------------------------
# Nesting guard (thread-local)
# ---------------------------------------------------------------------------

_tls = threading.local()


def _is_under_filter() -> bool:
    return getattr(_tls, "under_filter", False)


# ---------------------------------------------------------------------------
# Core: filter and reraise
# ---------------------------------------------------------------------------


def _filter_and_reraise(e: SpmdTypeError) -> None:  # noqa: C901
    """Apply traceback filtering to *e* in-place.

    Callers should invoke this from an ``except SpmdTypeError`` block and then
    use a bare ``raise`` so the original traceback is preserved without adding
    helper frames from this module.
    """
    mode = _resolve_mode(_FILTERING)

    if mode == "off":
        return

    tb = e.__traceback__

    if mode == "tracebackhide":
        # Walk the traceback and set __tracebackhide__ on internal frames.
        current = tb
        while current is not None:
            if _is_internal_frame(current.tb_frame.f_code.co_filename):
                current.tb_frame.f_locals["__tracebackhide__"] = True
            current = current.tb_next
        return

    # quiet_remove_frames or remove_frames: rebuild traceback
    filtered_tb = filter_traceback(tb)

    if mode == "remove_frames":
        # Attach full traceback as __cause__
        wrapper = UnfilteredStackTrace(
            "The above is the original, unfiltered traceback."
        )
        wrapper.__traceback__ = tb
        e.__cause__ = wrapper

    # Add a note about filtering
    e.add_note(
        "Set SPMD_TYPES_TRACEBACK_FILTERING=off to see the full, unfiltered traceback."
    )

    e.__traceback__ = filtered_tb


# ---------------------------------------------------------------------------
# API boundary decorator
# ---------------------------------------------------------------------------


def api_boundary(fn):
    """Decorator that catches ``SpmdTypeError`` and applies traceback filtering.

    Uses a thread-local nesting guard so that nested API boundaries (e.g.
    ``redistribute`` calling ``all_reduce``) do not double-filter.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        __tracebackhide__ = True  # noqa: F841
        if _is_under_filter() or _FILTERING == "off":
            return fn(*args, **kwargs)
        _tls.under_filter = True
        try:
            return fn(*args, **kwargs)
        except SpmdTypeError as e:
            _filter_and_reraise(e)
            raise
        finally:
            _tls.under_filter = False

    return wrapper


# ---------------------------------------------------------------------------
# Context manager for programmatic override
# ---------------------------------------------------------------------------


@contextmanager
def traceback_filtering(mode: str = "auto"):
    """Temporarily override the traceback filtering mode.

    Example::

        with traceback_filtering("off"):
            ...  # full tracebacks shown

    Args:
        mode: One of ``"off"``, ``"auto"``, ``"tracebackhide"``,
            ``"quiet_remove_frames"``, ``"remove_frames"``.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid traceback filtering mode: {mode!r}. "
            f"Valid modes: {sorted(_VALID_MODES)}"
        )
    global _FILTERING
    old = _FILTERING
    _FILTERING = mode
    try:
        yield
    finally:
        _FILTERING = old
