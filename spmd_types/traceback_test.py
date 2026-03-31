"""
Tests for JAX-style traceback filtering on SpmdTypeError.

Covers: _traceback.py, api_boundary decorator, traceback_filtering context manager
"""

import sys
import traceback as traceback_mod
import unittest

from spmd_types._checker import assert_type, mutate_type
from spmd_types._test_utils import SpmdTypeCheckedTestCase
from spmd_types._traceback import (
    _VALID_MODES,
    api_boundary,
    filter_traceback,
    traceback_filtering,
    UnfilteredStackTrace,
)
from spmd_types.types import P, R, SpmdTypeError, V


def _collect_frame_files(tb):
    """Collect all filenames from a traceback chain."""
    files = []
    current = tb
    while current is not None:
        files.append(current.tb_frame.f_code.co_filename)
        current = current.tb_next
    return files


def _collect_frame_funcs(tb):
    """Collect all (filename, funcname) pairs from a traceback chain."""
    frames = []
    current = tb
    while current is not None:
        frames.append(
            (current.tb_frame.f_code.co_filename, current.tb_frame.f_code.co_name)
        )
        current = current.tb_next
    return frames


class TestFilterTraceback(unittest.TestCase):
    """Test the low-level filter_traceback function."""

    def test_filter_removes_internal_frames(self):
        """Internal frames from _checker.py etc. should be removed."""
        try:
            raise SpmdTypeError("test error")
        except SpmdTypeError:
            _, _, tb = sys.exc_info()

        filtered = filter_traceback(tb)
        self.assertIsNotNone(filtered)
        for f in _collect_frame_files(filtered):
            self.assertIn("traceback_test.py", f)

    def test_filter_none_traceback(self):
        """filter_traceback(None) should return None."""
        self.assertIsNone(filter_traceback(None))


class TestModeOff(SpmdTypeCheckedTestCase):
    """With mode=off, tracebacks should include deep internal frames."""

    def test_off_preserves_full_traceback(self):
        with traceback_filtering("off"):
            x = self._generate_inputs((4,), self.pg, R)
            y = self._generate_inputs((4,), self.pg, P)
            try:
                _ = x + y
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError:
                _, _, tb = sys.exc_info()
                frame_funcs = _collect_frame_funcs(tb)
                # off mode: deep internal frames like _typecheck_core should be present
                internal_funcs = [
                    fn for _, fn in frame_funcs if fn == "_typecheck_core"
                ]
                self.assertGreater(
                    len(internal_funcs),
                    0,
                    "off mode should preserve _typecheck_core frames",
                )
                helper_funcs = [
                    fn for _, fn in frame_funcs if fn == "_filter_and_reraise"
                ]
                self.assertEqual(
                    helper_funcs,
                    [],
                    "off mode should not add _filter_and_reraise to the traceback",
                )


class TestModeTracebackhide(SpmdTypeCheckedTestCase):
    """With mode=tracebackhide, __tracebackhide__ is set on internal frames."""

    def test_tracebackhide_sets_flag(self):
        with traceback_filtering("tracebackhide"):
            x = self._generate_inputs((4,), self.pg, R)
            y = self._generate_inputs((4,), self.pg, P)
            try:
                _ = x + y
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError:
                _, _, tb = sys.exc_info()
                current = tb
                found_hidden = False
                while current is not None:
                    if "_checker.py" in current.tb_frame.f_code.co_filename:
                        if current.tb_frame.f_locals.get("__tracebackhide__"):
                            found_hidden = True
                    current = current.tb_next
                self.assertTrue(
                    found_hidden,
                    "tracebackhide mode should set __tracebackhide__",
                )


class TestModeQuietRemoveFrames(SpmdTypeCheckedTestCase):
    """With mode=quiet_remove_frames, deep internal frames are stripped."""

    def test_quiet_remove_frames_strips_deep_internals(self):
        with traceback_filtering("quiet_remove_frames"):
            x = self._generate_inputs((4,), self.pg, R)
            y = self._generate_inputs((4,), self.pg, P)
            try:
                _ = x + y
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError as e:
                _, _, tb = sys.exc_info()
                frame_funcs = _collect_frame_funcs(tb)
                # Deep internal frames like _typecheck_core should be gone
                deep_internal = [
                    fn
                    for _, fn in frame_funcs
                    if fn
                    in ("_typecheck_core", "_raise_strict_error", "infer_output_type")
                ]
                self.assertEqual(
                    len(deep_internal),
                    0,
                    f"quiet_remove_frames should strip deep internals, found: {deep_internal}",
                )
                # Should have a note about filtering
                self.assertTrue(hasattr(e, "__notes__"))
                notes = " ".join(e.__notes__)
                self.assertIn("SPMD_TYPES_TRACEBACK_FILTERING=off", notes)

    def test_quiet_remove_frames_hides_filter_helper(self):
        with traceback_filtering("quiet_remove_frames"):
            try:
                self._generate_inputs((4,), self.pg, R) + self._generate_inputs(
                    (4,), self.pg, P
                )
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError as e:
                formatted = "".join(traceback_mod.format_exception(e))

        self.assertNotIn("_filter_and_reraise", formatted)

    def test_quiet_remove_frames_fewer_frames_than_off(self):
        """Filtered traceback should have fewer frames than unfiltered."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, P)

        # Count frames with off mode
        with traceback_filtering("off"):
            try:
                _ = x + y
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError:
                _, _, tb = sys.exc_info()
                off_count = len(_collect_frame_files(tb))

        # Count frames with quiet_remove_frames mode
        with traceback_filtering("quiet_remove_frames"):
            try:
                _ = x + y
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError:
                _, _, tb = sys.exc_info()
                filtered_count = len(_collect_frame_files(tb))

        self.assertLess(
            filtered_count,
            off_count,
            f"Filtered ({filtered_count}) should have fewer frames than off ({off_count})",
        )


class TestModeRemoveFrames(SpmdTypeCheckedTestCase):
    """With mode=remove_frames, deep internals stripped and __cause__ has full tb."""

    def test_remove_frames_has_cause(self):
        with traceback_filtering("remove_frames"):
            x = self._generate_inputs((4,), self.pg, R)
            y = self._generate_inputs((4,), self.pg, P)
            try:
                _ = x + y
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError as e:
                _, _, tb = sys.exc_info()
                # Deep internal frames should be gone
                deep = [
                    fn
                    for _, fn in _collect_frame_funcs(tb)
                    if fn in ("_typecheck_core", "_raise_strict_error")
                ]
                self.assertEqual(len(deep), 0)
                # __cause__ should be UnfilteredStackTrace with full tb
                self.assertIsInstance(e.__cause__, UnfilteredStackTrace)
                self.assertIsNotNone(e.__cause__.__traceback__)


class TestNestingGuard(SpmdTypeCheckedTestCase):
    """Nested API boundaries should not double-filter."""

    def test_nested_api_boundary_no_double_filter(self):
        @api_boundary
        def outer():
            return inner()

        @api_boundary
        def inner():
            raise SpmdTypeError("inner error")

        with traceback_filtering("quiet_remove_frames"):
            try:
                outer()
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError as e:
                # Should have exactly one note (not two)
                notes = getattr(e, "__notes__", [])
                filter_notes = [
                    n for n in notes if "SPMD_TYPES_TRACEBACK_FILTERING" in n
                ]
                self.assertEqual(
                    len(filter_notes),
                    1,
                    f"Expected exactly 1 filter note, got {len(filter_notes)}",
                )


class TestContextManager(unittest.TestCase):
    """Test traceback_filtering context manager."""

    def test_override_and_restore(self):
        import spmd_types._traceback as tb_mod

        original = tb_mod._FILTERING
        with traceback_filtering("off"):
            self.assertEqual(tb_mod._FILTERING, "off")
            with traceback_filtering("remove_frames"):
                self.assertEqual(tb_mod._FILTERING, "remove_frames")
            self.assertEqual(tb_mod._FILTERING, "off")
        self.assertEqual(tb_mod._FILTERING, original)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            with traceback_filtering("invalid_mode"):
                pass

    def test_valid_modes(self):
        for mode in _VALID_MODES:
            with traceback_filtering(mode):
                pass  # should not raise


class TestAssertTypeApiB(SpmdTypeCheckedTestCase):
    """assert_type with conflicting types triggers filtered error."""

    def test_assert_type_conflict_filtered(self):
        with traceback_filtering("quiet_remove_frames"):
            x = self._generate_inputs((4,), self.pg, R)
            try:
                assert_type(x, {self.pg: V})
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError as e:
                # Should have a note about filtering
                self.assertTrue(hasattr(e, "__notes__"))
                notes = " ".join(e.__notes__)
                self.assertIn("SPMD_TYPES_TRACEBACK_FILTERING=off", notes)


class TestMutateTypeApiB(SpmdTypeCheckedTestCase):
    """mutate_type with wrong src triggers filtered error."""

    def test_mutate_type_wrong_src_filtered(self):
        with traceback_filtering("quiet_remove_frames"):
            x = self._generate_inputs((4,), self.pg, R)
            try:
                mutate_type(x, self.pg, src=V, dst=P)
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError as e:
                # Should have a note about filtering
                self.assertTrue(hasattr(e, "__notes__"))
                notes = " ".join(e.__notes__)
                self.assertIn("SPMD_TYPES_TRACEBACK_FILTERING=off", notes)


class TestEndToEnd(SpmdTypeCheckedTestCase):
    """End-to-end: format the traceback as a user would see it.

    Uses ``traceback.format_exception`` on the caught exception to verify that
    the formatted output (what ``python -c`` would print) contains only user
    frames when filtering is on, and internal frames when it is off.
    """

    def _trigger_error(self):
        """Trigger a SpmdTypeError through a user-like call chain."""
        x = self._generate_inputs((4,), self.pg, R)
        y = self._generate_inputs((4,), self.pg, P)

        def user_math(a, b):
            return a + b

        user_math(x, y)

    def test_filtered_traceback_shows_only_user_frames(self):
        with traceback_filtering("quiet_remove_frames"):
            try:
                self._trigger_error()
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError as e:
                formatted = "".join(traceback_mod.format_exception(e))

        # The error type should appear.
        self.assertIn("SpmdTypeError", formatted)

        # User code frames should be present.
        self.assertIn("user_math", formatted)
        self.assertIn("_trigger_error", formatted)

        # Deep internal frames should have been stripped.
        self.assertNotIn("_typecheck_core", formatted)
        self.assertNotIn("infer_output_type", formatted)
        self.assertNotIn("_raise_strict_error", formatted)
        # torch dispatch internals should be gone.
        self.assertNotIn("torch/overrides.py", formatted)

        # The filtering note should appear.
        self.assertIn("SPMD_TYPES_TRACEBACK_FILTERING=off", formatted)

    def test_off_mode_shows_internal_frames(self):
        with traceback_filtering("off"):
            try:
                self._trigger_error()
                self.fail("Expected SpmdTypeError")
            except SpmdTypeError as e:
                formatted = "".join(traceback_mod.format_exception(e))

        self.assertIn("SpmdTypeError", formatted)
        # With filtering off, deep internal frames should be visible.
        self.assertIn("_typecheck_core", formatted)
        self.assertIn("_checker.py", formatted)
        # No filtering note.
        self.assertNotIn("SPMD_TYPES_TRACEBACK_FILTERING=off", formatted)


if __name__ == "__main__":
    unittest.main()
