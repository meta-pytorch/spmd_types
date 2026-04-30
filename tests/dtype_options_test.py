# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for op_dtype, out_dtype, and backward_options on collectives and local ops.

Each test traces dtype conversions (to_copy) and collectives in forward and
backward via ``_TraceResult(fwd, bwd, error)``.  Ops are ``;``-separated,
``-`` means no ops.  Detects wasteful round-trips (e.g. bf16->f32->bf16) and
verifies cast ordering relative to the collective.

Three kinds of test per operation:

- **fwd_defaults_table**: Forward-only (requires_grad=False) over (in, op, out)
  where at least one of op/out is None.  Tests the defaulting logic: which
  dtype does the op run in when the user only specifies one?  Per-collective
  because the defaulting depends on whether the op is reducing.

- **bwd_defaults_table**: Backward-only over (in, op, out) with both specified
  but no backward_options.  Shows backward defaulting and ERROR for
  reduction-backward cases that require explicit backward_options.

- **full table (_table)**: Forward+backward over (in, op, out, bop) with all
  specified.  Since the defaulting logic is shared across collectives, we keep
  full tables only for structurally distinct cases: one comm-comm pair (P->R)
  and all pairs that lack a collective in forward or backward (P->I, S(0)->I,
  convert, reinterpret).  The remaining comm-comm pairs (S(0)->R, P->S(0),
  V->V) are covered by invariant tests.

- **invariant tests (_check_invariants)**: For all (in, op, out, bop) combos,
  programmatically check: fwd collective at op precision, bwd collective at
  bop precision, output dtype == out, gradient dtype == in.
"""

import unittest
from dataclasses import dataclass

import expecttest
import torch
import torch.distributed as dist
from spmd_types import (
    all_gather,
    all_reduce,
    all_to_all,
    convert,
    I,
    invariant_to_replicate,
    P,
    R,
    redistribute,
    reduce_scatter,
    reinterpret,
    S,
    shard,
    unshard,
    V,
)
from spmd_types._checker import assert_type
from spmd_types.types import Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

f32 = torch.float32
bf16 = torch.bfloat16
_S0 = S(0)

_BACKWARD_ERROR = 'error: When using reduced precision with an operation whose backward involves a reduction (all_reduce or reduce_scatter), you must explicitly specify backward_options={"op_dtype": ...} to control the precision of the gradient reduction.'


@dataclass
class _TraceResult:
    fwd: str = "-"
    bwd: str | None = None
    error: str | None = None

    def __str__(self) -> str:
        if self.error is not None:
            return self.error
        if self.bwd is not None:
            return f"{self.fwd} | {self.bwd}"
        return self.fwd


_DTYPE_SHORT = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16"}


def _short_dtype(dt: torch.dtype) -> str:
    return _DTYPE_SHORT.get(dt, str(dt).replace("torch.", ""))


_C10D_MAP = {
    "c10d.allreduce_": "all_reduce",
    "c10d.allgather_": "all_gather",
    "c10d._allgather_base_": "all_gather",
    "c10d._reduce_scatter_base_": "reduce_scatter",
    "c10d.reduce_scatter_": "reduce_scatter",
    "c10d.reduce_scatter_tensor": "reduce_scatter",
    "c10d.alltoall_": "all_to_all",
}


class _OpTrace(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops: list[str] = []

    @staticmethod
    def _find_dtype(args):
        for a in args:
            if hasattr(a, "dtype"):
                return a.dtype
            if isinstance(a, (list, tuple)):
                for item in a:
                    if hasattr(item, "dtype"):
                        return item.dtype
        return None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func.overloadpacket)
        if name == "aten._to_copy":
            kw = kwargs or {}
            target_dtype = kw.get("dtype")
            if target_dtype is not None and len(args) > 0 and hasattr(args[0], "dtype"):
                if target_dtype != args[0].dtype:
                    self.ops.append(f"to({_short_dtype(target_dtype)})")
        elif name in _C10D_MAP:
            dt = self._find_dtype(args)
            label = _C10D_MAP[name]
            if dt is not None:
                label += f"({_short_dtype(dt)})"
            self.ops.append(label)
        elif name.startswith("spmd_types."):
            dt = self._find_dtype(args)
            label = {
                "spmd_types.replicate_to_partial": "r2p",
                "spmd_types.replicate_to_varying": "r2v",
                "spmd_types.varying_to_partial": "v2p",
            }.get(name, name)
            if dt is not None:
                label += f"({_short_dtype(dt)})"
            self.ops.append(label)
        elif name.startswith("c10d."):
            self.ops.append(name)
        return func(*args, **(kwargs or {}))


class _DtypeTestBase(expecttest.TestCase):
    WORLD_SIZE = 3

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=cls.WORLD_SIZE, store=store
        )
        cls.pg = dist.distributed_c10d._get_default_group()

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _make_input(self, shape, input_dtype, src_type, requires_grad):
        x = torch.randn(shape, dtype=input_dtype)
        assert_type(x, {self.pg: V if isinstance(src_type, Shard) else src_type})
        if requires_grad:
            x.requires_grad_(True)
        return x

    def _trace(
        self, fn, *, src_type, shape=(4,), input_dtype=f32, requires_grad=True
    ) -> _TraceResult:
        x = self._make_input(shape, input_dtype, src_type, requires_grad)

        tracer = _OpTrace()
        try:
            with tracer:
                result = fn(x, self.pg)
        except (ValueError, RuntimeError) as e:
            msg = str(e)
            dot = msg.find(". ")
            return _TraceResult(error=f"error: {msg[: dot + 1] if dot >= 0 else msg}")

        fwd = "; ".join(tracer.ops) if tracer.ops else "-"

        if requires_grad and result.requires_grad:
            tracer.ops.clear()
            with tracer:
                torch.autograd.grad(result, x, grad_outputs=torch.ones_like(result))
            bwd = "; ".join(tracer.ops) if tracer.ops else "-"
            return _TraceResult(fwd=fwd, bwd=bwd)

        return _TraceResult(fwd=fwd)

    def _ex(self, actual: _TraceResult | str, expected: str) -> None:
        self.assertExpectedInline(str(actual), expected, skip=1)

    @staticmethod
    def _fmt_table(rows):
        ncols = len(rows[0])
        widths = [max(len(r[i]) for r in rows) for i in range(ncols)]
        lines = []
        for i, row in enumerate(rows):
            line = "  ".join(row[j].ljust(widths[j]) for j in range(ncols))
            lines.append(line.rstrip())
            if i == 0:
                lines.append("  ".join("-" * widths[j] for j in range(ncols)))
        return "\n".join(lines)

    @staticmethod
    def _dn(d):
        return "-" if d is None else ("f32" if d is f32 else "bf16")

    def _fwd_defaults_table(self, fn):
        """Forward-only table over in x op x out where at least one of op/out is None."""
        rows = [("in", "op", "out", "forward")]
        for in_dt in [f32, bf16]:
            for op_dt in [None, bf16, f32]:
                for out_dt in [None, bf16, f32]:
                    if op_dt is not None and out_dt is not None:
                        continue
                    kw = {}
                    if op_dt is not None:
                        kw["op_dtype"] = op_dt
                    if out_dt is not None:
                        kw["out_dtype"] = out_dt
                    result = fn(in_dt, requires_grad=False, **kw)
                    rows.append(
                        (
                            self._dn(in_dt),
                            self._dn(op_dt),
                            self._dn(out_dt),
                            str(result),
                        )
                    )
        return self._fmt_table(rows)

    def _bwd_defaults_table(self, fn):
        """Backward-only table over in x op x out (no None, no bop) to test backward defaulting."""
        rows = [("in", "op", "out", "backward")]
        for in_dt in [f32, bf16]:
            for op_dt in [bf16, f32]:
                for out_dt in [bf16, f32]:
                    result = fn(in_dt, op_dtype=op_dt, out_dtype=out_dt)
                    if result.error is not None:
                        bwd = "ERROR"
                    elif result.bwd is not None:
                        bwd = result.bwd
                    else:
                        bwd = "-"
                    rows.append(
                        (self._dn(in_dt), self._dn(op_dt), self._dn(out_dt), bwd)
                    )
        return self._fmt_table(rows)

    def _table(self, fn):
        """Fwd+bwd table over in x op x out x bop (no None -- defaults tested separately)."""
        rows = [("in", "op", "out", "bop", "forward", "backward")]
        for in_dt in [f32, bf16]:
            for op_dt in [bf16, f32]:
                for out_dt in [bf16, f32]:
                    for bop_dt in [bf16, f32]:
                        result = fn(
                            in_dt,
                            op_dtype=op_dt,
                            out_dtype=out_dt,
                            backward_options={"op_dtype": bop_dt},
                        )
                        if result.error is not None:
                            fwd, bwd = "ERROR", ""
                        else:
                            fwd = result.fwd
                            bwd = result.bwd if result.bwd is not None else ""
                        rows.append(
                            (
                                self._dn(in_dt),
                                self._dn(op_dt),
                                self._dn(out_dt),
                                self._dn(bop_dt),
                                fwd,
                                bwd,
                            )
                        )
        return self._fmt_table(rows)

    @staticmethod
    def _collective_dtypes(ops):
        result = []
        for op in ops:
            for coll in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all"):
                if op.startswith(coll + "("):
                    result.append(op[len(coll) + 1 : -1])
        return result

    def _check_invariants(self, fn, *, src_type, shape=(4,)):
        """For all (in, op, out, bop) combos, verify:
        1. Forward collective (if any) runs at op precision
        2. Backward collective (if any) runs at bop precision
        3. Output dtype == out
        4. Gradient dtype == in
        """
        for in_dt in [f32, bf16]:
            for op_dt in [bf16, f32]:
                for out_dt in [bf16, f32]:
                    for bop_dt in [bf16, f32]:
                        desc = f"in={_short_dtype(in_dt)} op={_short_dtype(op_dt)} out={_short_dtype(out_dt)} bop={_short_dtype(bop_dt)}"
                        with self.subTest(desc):
                            x = self._make_input(
                                shape, in_dt, src_type, requires_grad=True
                            )

                            tracer = _OpTrace()
                            with tracer:
                                result = fn(
                                    x,
                                    self.pg,
                                    op_dtype=op_dt,
                                    out_dtype=out_dt,
                                    backward_options={"op_dtype": bop_dt},
                                )

                            self.assertEqual(
                                result.dtype,
                                out_dt,
                                f"{desc}: output dtype",
                            )
                            for coll_dt in self._collective_dtypes(tracer.ops):
                                self.assertEqual(
                                    coll_dt,
                                    _short_dtype(op_dt),
                                    f"{desc}: fwd collective dtype",
                                )

                            tracer.ops.clear()
                            with tracer:
                                grads = torch.autograd.grad(
                                    result, x, grad_outputs=torch.ones_like(result)
                                )

                            self.assertEqual(
                                grads[0].dtype,
                                in_dt,
                                f"{desc}: gradient dtype",
                            )
                            for coll_dt in self._collective_dtypes(tracer.ops):
                                self.assertEqual(
                                    coll_dt,
                                    _short_dtype(bop_dt),
                                    f"{desc}: bwd collective dtype",
                                )


# =====================================================================
# all_reduce: P -> R | I
# =====================================================================


class TestAllReduceDtype(_DtypeTestBase):
    def _run(self, input_dtype, dst, *, requires_grad=True, **kw):
        return self._trace(
            lambda x, pg: all_reduce(x, pg, src=P, dst=dst, **kw),
            src_type=P,
            input_dtype=input_dtype,
            requires_grad=requires_grad,
        )

    # -- P -> I (backward: convert I->R, no reduction) --

    # reducing: out_dtype only -> op stays at input dtype (no min-bandwidth trick)
    def test_P_I_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, I, **kw)),
            """\
in    op    out   forward
----  ----  ----  --------------------------
f32   -     -     all_reduce(f32)
f32   -     bf16  all_reduce(f32); to(bf16)
f32   -     f32   all_reduce(f32)
f32   bf16  -     to(bf16); all_reduce(bf16)
f32   f32   -     all_reduce(f32)
bf16  -     -     all_reduce(bf16)
bf16  -     bf16  all_reduce(bf16)
bf16  -     f32   all_reduce(bf16); to(f32)
bf16  bf16  -     all_reduce(bf16)
bf16  f32   -     to(f32); all_reduce(f32)""",
        )

    # no bwd collective (convert I->R is a no-op)
    def test_P_I_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, I, **kw)),
            """\
in    op    out   backward
----  ----  ----  --------
f32   bf16  bf16  to(f32)
f32   bf16  f32   -
f32   f32   bf16  to(f32)
f32   f32   f32   -
bf16  bf16  bf16  -
bf16  bf16  f32   to(bf16)
bf16  f32   bf16  -
bf16  f32   f32   to(bf16)""",
        )

    def test_P_I(self):
        # TODO: arguably the "to(f32); to(bf16)" backward entries are a bug --
        # we should special case no-op backwards to skip this conversion.
        # (the other direction is not a bug: the user asked for truncation.)
        self._ex(
            self._table(lambda in_dt, **kw: self._run(in_dt, I, **kw)),
            """\
in    op    out   bop   forward                              backward
----  ----  ----  ----  -----------------------------------  -----------------
f32   bf16  bf16  bf16  to(bf16); all_reduce(bf16)           to(f32)
f32   bf16  bf16  f32   to(bf16); all_reduce(bf16)           to(f32)
f32   bf16  f32   bf16  to(bf16); all_reduce(bf16); to(f32)  to(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); all_reduce(bf16); to(f32)  -
f32   f32   bf16  bf16  all_reduce(f32); to(bf16)            to(f32)
f32   f32   bf16  f32   all_reduce(f32); to(bf16)            to(f32)
f32   f32   f32   bf16  all_reduce(f32)                      to(bf16); to(f32)
f32   f32   f32   f32   all_reduce(f32)                      -
bf16  bf16  bf16  bf16  all_reduce(bf16)                     -
bf16  bf16  bf16  f32   all_reduce(bf16)                     to(f32); to(bf16)
bf16  bf16  f32   bf16  all_reduce(bf16); to(f32)            to(bf16)
bf16  bf16  f32   f32   all_reduce(bf16); to(f32)            to(bf16)
bf16  f32   bf16  bf16  to(f32); all_reduce(f32); to(bf16)   -
bf16  f32   bf16  f32   to(f32); all_reduce(f32); to(bf16)   to(f32); to(bf16)
bf16  f32   f32   bf16  to(f32); all_reduce(f32)             to(bf16)
bf16  f32   f32   f32   to(f32); all_reduce(f32)             to(bf16)""",
        )

    # -- P -> R (backward: all_reduce P->R, self-dual reduction) --

    # reducing: out_dtype only -> op stays at input dtype (no min-bandwidth trick)
    def test_P_R_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, R, **kw)),
            """\
in    op    out   forward
----  ----  ----  --------------------------
f32   -     -     all_reduce(f32)
f32   -     bf16  all_reduce(f32); to(bf16)
f32   -     f32   all_reduce(f32)
f32   bf16  -     to(bf16); all_reduce(bf16)
f32   f32   -     all_reduce(f32)
bf16  -     -     all_reduce(bf16)
bf16  -     bf16  all_reduce(bf16)
bf16  -     f32   all_reduce(bf16); to(f32)
bf16  bf16  -     all_reduce(bf16)
bf16  f32   -     to(f32); all_reduce(f32)""",
        )

    # bwd is all_reduce (reducing): reduced precision requires explicit backward_options
    # because the gradient reduction would lose precision silently
    def test_P_R_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, R, **kw)),
            """\
in    op    out   backward
----  ----  ----  ---------------
f32   bf16  bf16  ERROR
f32   bf16  f32   ERROR
f32   f32   bf16  ERROR
f32   f32   f32   all_reduce(f32)
bf16  bf16  bf16  ERROR
bf16  bf16  f32   ERROR
bf16  f32   bf16  ERROR
bf16  f32   f32   ERROR""",
        )

    def test_P_R(self):
        self._ex(
            self._table(lambda in_dt, **kw: self._run(in_dt, R, **kw)),
            """\
in    op    out   bop   forward                              backward
----  ----  ----  ----  -----------------------------------  -----------------------------------
f32   bf16  bf16  bf16  to(bf16); all_reduce(bf16)           all_reduce(bf16); to(f32)
f32   bf16  bf16  f32   to(bf16); all_reduce(bf16)           to(f32); all_reduce(f32)
f32   bf16  f32   bf16  to(bf16); all_reduce(bf16); to(f32)  to(bf16); all_reduce(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); all_reduce(bf16); to(f32)  all_reduce(f32)
f32   f32   bf16  bf16  all_reduce(f32); to(bf16)            all_reduce(bf16); to(f32)
f32   f32   bf16  f32   all_reduce(f32); to(bf16)            to(f32); all_reduce(f32)
f32   f32   f32   bf16  all_reduce(f32)                      to(bf16); all_reduce(bf16); to(f32)
f32   f32   f32   f32   all_reduce(f32)                      all_reduce(f32)
bf16  bf16  bf16  bf16  all_reduce(bf16)                     all_reduce(bf16)
bf16  bf16  bf16  f32   all_reduce(bf16)                     to(f32); all_reduce(f32); to(bf16)
bf16  bf16  f32   bf16  all_reduce(bf16); to(f32)            to(bf16); all_reduce(bf16)
bf16  bf16  f32   f32   all_reduce(bf16); to(f32)            all_reduce(f32); to(bf16)
bf16  f32   bf16  bf16  to(f32); all_reduce(f32); to(bf16)   all_reduce(bf16)
bf16  f32   bf16  f32   to(f32); all_reduce(f32); to(bf16)   to(f32); all_reduce(f32); to(bf16)
bf16  f32   f32   bf16  to(f32); all_reduce(f32)             to(bf16); all_reduce(bf16)
bf16  f32   f32   f32   to(f32); all_reduce(f32)             all_reduce(f32); to(bf16)""",
        )

    def test_P_R_op_bf16_no_grad(self):
        self._ex(
            self._run(f32, R, op_dtype=bf16, requires_grad=False),
            """to(bf16); all_reduce(bf16)""",
        )

    def test_P_R_bf16_input_no_grad(self):
        self._ex(self._run(bf16, R, requires_grad=False), """all_reduce(bf16)""")

    def test_P_I_invariants(self):
        self._check_invariants(
            lambda x, pg, **kw: all_reduce(x, pg, src=P, dst=I, **kw),
            src_type=P,
        )

    def test_P_R_invariants(self):
        self._check_invariants(
            lambda x, pg, **kw: all_reduce(x, pg, src=P, dst=R, **kw),
            src_type=P,
        )

    def test_P_I_inplace_op_bf16_error(self):
        self._ex(
            self._run(f32, I, inplace=True, op_dtype=bf16),
            """error: inplace=True does not support op_dtype or out_dtype, because the operation cannot be performed in-place when a dtype conversion is required.""",
        )


# =====================================================================
# all_gather: V | S(i) -> R | I
# =====================================================================


class TestAllGatherDtype(_DtypeTestBase):
    def _run(self, input_dtype, dst, *, src=_S0, requires_grad=True, **kw):
        return self._trace(
            lambda x, pg: all_gather(x, pg, src=src, dst=dst, **kw),
            src_type=V,
            shape=(6,),
            input_dtype=input_dtype,
            requires_grad=requires_grad,
        )

    # -- S(0) -> I (backward: convert I->S(0), no reduction) --

    # non-reducing: out_dtype only -> op at min(in, out) dtype (min-bandwidth trick)
    def test_S0_I_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, I, **kw)),
            """\
in    op    out   forward
----  ----  ----  --------------------------
f32   -     -     all_gather(f32)
f32   -     bf16  to(bf16); all_gather(bf16)
f32   -     f32   all_gather(f32)
f32   bf16  -     to(bf16); all_gather(bf16)
f32   f32   -     all_gather(f32)
bf16  -     -     all_gather(bf16)
bf16  -     bf16  all_gather(bf16)
bf16  -     f32   all_gather(bf16); to(f32)
bf16  bf16  -     all_gather(bf16)
bf16  f32   -     to(f32); all_gather(f32)""",
        )

    # no bwd collective (convert I->S(0) is local)
    def test_S0_I_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, I, **kw)),
            """\
in    op    out   backward
----  ----  ----  ------------------
f32   bf16  bf16  r2v(bf16); to(f32)
f32   bf16  f32   r2v(f32)
f32   f32   bf16  r2v(bf16); to(f32)
f32   f32   f32   r2v(f32)
bf16  bf16  bf16  r2v(bf16)
bf16  bf16  f32   r2v(f32); to(bf16)
bf16  f32   bf16  r2v(bf16)
bf16  f32   f32   r2v(f32); to(bf16)""",
        )

    def test_S0_I(self):
        self._ex(
            self._table(lambda in_dt, **kw: self._run(in_dt, I, **kw)),
            """\
in    op    out   bop   forward                              backward
----  ----  ----  ----  -----------------------------------  ----------------------------
f32   bf16  bf16  bf16  to(bf16); all_gather(bf16)           r2v(bf16); to(f32)
f32   bf16  bf16  f32   to(bf16); all_gather(bf16)           to(f32); r2v(f32)
f32   bf16  f32   bf16  to(bf16); all_gather(bf16); to(f32)  to(bf16); r2v(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); all_gather(bf16); to(f32)  r2v(f32)
f32   f32   bf16  bf16  all_gather(f32); to(bf16)            r2v(bf16); to(f32)
f32   f32   bf16  f32   all_gather(f32); to(bf16)            to(f32); r2v(f32)
f32   f32   f32   bf16  all_gather(f32)                      to(bf16); r2v(bf16); to(f32)
f32   f32   f32   f32   all_gather(f32)                      r2v(f32)
bf16  bf16  bf16  bf16  all_gather(bf16)                     r2v(bf16)
bf16  bf16  bf16  f32   all_gather(bf16)                     to(f32); r2v(f32); to(bf16)
bf16  bf16  f32   bf16  all_gather(bf16); to(f32)            to(bf16); r2v(bf16)
bf16  bf16  f32   f32   all_gather(bf16); to(f32)            r2v(f32); to(bf16)
bf16  f32   bf16  bf16  to(f32); all_gather(f32); to(bf16)   r2v(bf16)
bf16  f32   bf16  f32   to(f32); all_gather(f32); to(bf16)   to(f32); r2v(f32); to(bf16)
bf16  f32   f32   bf16  to(f32); all_gather(f32)             to(bf16); r2v(bf16)
bf16  f32   f32   f32   to(f32); all_gather(f32)             r2v(f32); to(bf16)""",
        )

    # -- S(0) -> R (backward: reduce_scatter, has reduction) --

    # non-reducing: out_dtype only -> op at min(in, out) dtype (min-bandwidth trick)
    def test_S0_R_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, R, **kw)),
            """\
in    op    out   forward
----  ----  ----  --------------------------
f32   -     -     all_gather(f32)
f32   -     bf16  to(bf16); all_gather(bf16)
f32   -     f32   all_gather(f32)
f32   bf16  -     to(bf16); all_gather(bf16)
f32   f32   -     all_gather(f32)
bf16  -     -     all_gather(bf16)
bf16  -     bf16  all_gather(bf16)
bf16  -     f32   all_gather(bf16); to(f32)
bf16  bf16  -     all_gather(bf16)
bf16  f32   -     to(f32); all_gather(f32)""",
        )

    # bwd is reduce_scatter (reducing): reduced precision requires explicit backward_options
    # because the gradient reduction would lose precision silently
    def test_S0_R_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, R, **kw)),
            """\
in    op    out   backward
----  ----  ----  -------------------
f32   bf16  bf16  ERROR
f32   bf16  f32   ERROR
f32   f32   bf16  ERROR
f32   f32   f32   reduce_scatter(f32)
bf16  bf16  bf16  ERROR
bf16  bf16  f32   ERROR
bf16  f32   bf16  ERROR
bf16  f32   f32   ERROR""",
        )

    # -- V (stack) -> R (backward: reduce_scatter, has reduction) --

    def test_V_R_fsdp(self):
        self._ex(
            self._run(f32, R, src=V, op_dtype=bf16, backward_options={"op_dtype": f32}),
            """to(bf16); all_gather(bf16) | to(f32); reduce_scatter(f32)""",
        )

    # -- V (stack) -> I (backward: convert I->V, no reduction) --

    def test_V_I_op_bf16(self):
        self._ex(
            self._run(f32, I, src=V, op_dtype=bf16),
            """to(bf16); all_gather(bf16) | r2v(bf16); to(f32)""",
        )

    def test_S0_I_invariants(self):
        self._check_invariants(
            lambda x, pg, **kw: all_gather(x, pg, src=S(0), dst=I, **kw),
            src_type=V,
            shape=(6,),
        )

    def test_S0_R_invariants(self):
        self._check_invariants(
            lambda x, pg, **kw: all_gather(x, pg, src=S(0), dst=R, **kw),
            src_type=V,
            shape=(6,),
        )

    def test_V_I_invariants(self):
        self._check_invariants(
            lambda x, pg, **kw: all_gather(x, pg, src=V, dst=I, **kw),
            src_type=V,
            shape=(6,),
        )

    def test_V_R_invariants(self):
        self._check_invariants(
            lambda x, pg, **kw: all_gather(x, pg, src=V, dst=R, **kw),
            src_type=V,
            shape=(6,),
        )


# =====================================================================
# reduce_scatter: P -> V | S(i)
# =====================================================================


class TestReduceScatterDtype(_DtypeTestBase):
    def _run(self, input_dtype, *, dst=_S0, requires_grad=True, **kw):
        return self._trace(
            lambda x, pg: reduce_scatter(x, pg, src=P, dst=dst, **kw),
            src_type=P,
            shape=(6,),
            input_dtype=input_dtype,
            requires_grad=requires_grad,
        )

    # reducing: out_dtype only -> op stays at input dtype (no min-bandwidth trick)
    def test_P_S0_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, **kw)),
            """\
in    op    out   forward
----  ----  ----  ------------------------------
f32   -     -     reduce_scatter(f32)
f32   -     bf16  reduce_scatter(f32); to(bf16)
f32   -     f32   reduce_scatter(f32)
f32   bf16  -     to(bf16); reduce_scatter(bf16)
f32   f32   -     reduce_scatter(f32)
bf16  -     -     reduce_scatter(bf16)
bf16  -     bf16  reduce_scatter(bf16)
bf16  -     f32   reduce_scatter(bf16); to(f32)
bf16  bf16  -     reduce_scatter(bf16)
bf16  f32   -     to(f32); reduce_scatter(f32)""",
        )

    # bwd is all_gather (non-reducing): op at min(in, out) dtype (min-bandwidth trick)
    def test_P_S0_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, **kw)),
            """\
in    op    out   backward
----  ----  ----  --------------------------
f32   bf16  bf16  all_gather(bf16); to(f32)
f32   bf16  f32   all_gather(f32)
f32   f32   bf16  all_gather(bf16); to(f32)
f32   f32   f32   all_gather(f32)
bf16  bf16  bf16  all_gather(bf16)
bf16  bf16  f32   to(bf16); all_gather(bf16)
bf16  f32   bf16  all_gather(bf16)
bf16  f32   f32   to(bf16); all_gather(bf16)""",
        )

    # -- P -> V (stack variant) --

    def _run_stack(self, input_dtype, **kw):
        return self._trace(
            lambda x, pg: reduce_scatter(x, pg, src=P, dst=V, **kw),
            src_type=P,
            shape=(self.WORLD_SIZE, 4),
            input_dtype=input_dtype,
        )

    def test_P_V(self):
        self._ex(self._run_stack(f32), """reduce_scatter(f32) | all_gather(f32)""")

    def test_P_V_op_bf16(self):
        self._ex(
            self._run_stack(f32, op_dtype=bf16),
            """to(bf16); reduce_scatter(bf16) | all_gather(bf16); to(f32)""",
        )

    def test_P_S0_invariants(self):
        self._check_invariants(
            lambda x, pg, **kw: reduce_scatter(x, pg, src=P, dst=S(0), **kw),
            src_type=P,
            shape=(6,),
        )

    def test_P_V_invariants(self):
        self._check_invariants(
            lambda x, pg, **kw: reduce_scatter(x, pg, src=P, dst=V, **kw),
            src_type=P,
            shape=(3, 4),
        )


# =====================================================================
# all_to_all: V -> V
# =====================================================================


class TestAllToAllDtype(_DtypeTestBase):
    def _run(self, input_dtype, *, requires_grad=True, **kw):
        return self._trace(
            lambda x, pg: all_to_all(x, pg, src=V, dst=V, **kw),
            src_type=V,
            shape=(3, 4),
            input_dtype=input_dtype,
            requires_grad=requires_grad,
        )

    # non-reducing: out_dtype only -> op at min(in, out) dtype (min-bandwidth trick)
    def test_V_V_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, **kw)),
            """\
in    op    out   forward
----  ----  ----  --------------------------
f32   -     -     all_to_all(f32)
f32   -     bf16  to(bf16); all_to_all(bf16)
f32   -     f32   all_to_all(f32)
f32   bf16  -     to(bf16); all_to_all(bf16)
f32   f32   -     all_to_all(f32)
bf16  -     -     all_to_all(bf16)
bf16  -     bf16  all_to_all(bf16)
bf16  -     f32   all_to_all(bf16); to(f32)
bf16  bf16  -     all_to_all(bf16)
bf16  f32   -     to(f32); all_to_all(f32)""",
        )

    # bwd is all_to_all (non-reducing): op at min(in, out) dtype (min-bandwidth trick)
    def test_V_V_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, **kw)),
            """\
in    op    out   backward
----  ----  ----  --------------------------
f32   bf16  bf16  all_to_all(bf16); to(f32)
f32   bf16  f32   all_to_all(f32)
f32   f32   bf16  all_to_all(bf16); to(f32)
f32   f32   f32   all_to_all(f32)
bf16  bf16  bf16  all_to_all(bf16)
bf16  bf16  f32   to(bf16); all_to_all(bf16)
bf16  f32   bf16  all_to_all(bf16)
bf16  f32   f32   to(bf16); all_to_all(bf16)""",
        )

    # -- S(i) -> S(j) (shard variant) --

    def test_S0_S1_op_bf16(self):
        self._ex(
            self._trace(
                lambda x, pg: all_to_all(x, pg, src=S(0), dst=S(1), op_dtype=bf16),
                src_type=V,
                shape=(6, 3),
                input_dtype=f32,
            ),
            """to(bf16); all_to_all(bf16) | all_to_all(bf16); to(f32)""",
        )

    def test_V_V_invariants(self):
        self._check_invariants(
            lambda x, pg, **kw: all_to_all(x, pg, src=V, dst=V, **kw),
            src_type=V,
            shape=(3, 4),
        )

    def test_S0_S1_invariants(self):
        self._check_invariants(
            lambda x, pg, **kw: all_to_all(x, pg, src=S(0), dst=S(1), **kw),
            src_type=V,
            shape=(6, 3),
        )


# =====================================================================
# convert: local (no-comms) type changes
# =====================================================================


class TestConvertDtype(_DtypeTestBase):
    def _run(self, input_dtype, src, dst, *, requires_grad=True, **kw):
        src_type = V if isinstance(src, Shard) else src
        shape = (4,)
        if (src is R or src is I) and dst is V:
            shape = (3, 4)
        elif (src is R or src is I) and isinstance(dst, Shard):
            shape = (6,)
        return self._trace(
            lambda x, pg: convert(x, pg, src=src, dst=dst, **kw),
            src_type=src_type,
            shape=shape,
            input_dtype=input_dtype,
            requires_grad=requires_grad,
        )

    # -- I -> R (backward: all_reduce P->I, has reduction) --

    # no fwd collective
    def test_I_R_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, I, R, **kw)),
            """\
in    op    out   forward
----  ----  ----  --------
f32   -     -     -
f32   -     bf16  to(bf16)
f32   -     f32   -
f32   bf16  -     to(bf16)
f32   f32   -     -
bf16  -     -     -
bf16  -     bf16  -
bf16  -     f32   to(f32)
bf16  bf16  -     -
bf16  f32   -     to(f32)""",
        )

    # bwd is all_reduce (reducing): reduced precision requires explicit backward_options
    # because the gradient reduction would lose precision silently
    def test_I_R_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, I, R, **kw)),
            """\
in    op    out   backward
----  ----  ----  ---------------
f32   bf16  bf16  ERROR
f32   bf16  f32   ERROR
f32   f32   bf16  ERROR
f32   f32   f32   all_reduce(f32)
bf16  bf16  bf16  ERROR
bf16  bf16  f32   ERROR
bf16  f32   bf16  ERROR
bf16  f32   f32   ERROR""",
        )

    def test_I_R(self):
        self._ex(
            self._table(lambda in_dt, **kw: self._run(in_dt, I, R, **kw)),
            """\
in    op    out   bop   forward            backward
----  ----  ----  ----  -----------------  -----------------------------------
f32   bf16  bf16  bf16  to(bf16)           all_reduce(bf16); to(f32)
f32   bf16  bf16  f32   to(bf16)           to(f32); all_reduce(f32)
f32   bf16  f32   bf16  to(bf16); to(f32)  to(bf16); all_reduce(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); to(f32)  all_reduce(f32)
f32   f32   bf16  bf16  to(bf16)           all_reduce(bf16); to(f32)
f32   f32   bf16  f32   to(bf16)           to(f32); all_reduce(f32)
f32   f32   f32   bf16  -                  to(bf16); all_reduce(bf16); to(f32)
f32   f32   f32   f32   -                  all_reduce(f32)
bf16  bf16  bf16  bf16  -                  all_reduce(bf16)
bf16  bf16  bf16  f32   -                  to(f32); all_reduce(f32); to(bf16)
bf16  bf16  f32   bf16  to(f32)            to(bf16); all_reduce(bf16)
bf16  bf16  f32   f32   to(f32)            all_reduce(f32); to(bf16)
bf16  f32   bf16  bf16  to(f32); to(bf16)  all_reduce(bf16)
bf16  f32   bf16  f32   to(f32); to(bf16)  to(f32); all_reduce(f32); to(bf16)
bf16  f32   f32   bf16  to(f32)            to(bf16); all_reduce(bf16)
bf16  f32   f32   f32   to(f32)            all_reduce(f32); to(bf16)""",
        )

    # -- R -> V (backward: convert V->P, no reduction, expert_mode) --

    # no fwd collective
    def test_R_V_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, R, V, expert_mode=True, **kw)
            ),
            """\
in    op    out   forward
----  ----  ----  -------------------
f32   -     -     r2v(f32)
f32   -     bf16  r2v(f32); to(bf16)
f32   -     f32   r2v(f32)
f32   bf16  -     to(bf16); r2v(bf16)
f32   f32   -     r2v(f32)
bf16  -     -     r2v(bf16)
bf16  -     bf16  r2v(bf16)
bf16  -     f32   r2v(bf16); to(f32)
bf16  bf16  -     r2v(bf16)
bf16  f32   -     to(f32); r2v(f32)""",
        )

    # no bwd collective (convert V->P is local)
    def test_R_V_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, R, V, expert_mode=True, **kw)
            ),
            """\
in    op    out   backward
----  ----  ----  -------------------
f32   bf16  bf16  v2p(bf16); to(f32)
f32   bf16  f32   v2p(f32)
f32   f32   bf16  v2p(bf16); to(f32)
f32   f32   f32   v2p(f32)
bf16  bf16  bf16  v2p(bf16)
bf16  bf16  f32   to(bf16); v2p(bf16)
bf16  f32   bf16  v2p(bf16)
bf16  f32   f32   to(bf16); v2p(bf16)""",
        )

    def test_R_V(self):
        self._ex(
            self._table(
                lambda in_dt, **kw: self._run(in_dt, R, V, expert_mode=True, **kw)
            ),
            """\
in    op    out   bop   forward                       backward
----  ----  ----  ----  ----------------------------  ----------------------------
f32   bf16  bf16  bf16  to(bf16); r2v(bf16)           v2p(bf16); to(f32)
f32   bf16  bf16  f32   to(bf16); r2v(bf16)           to(f32); v2p(f32)
f32   bf16  f32   bf16  to(bf16); r2v(bf16); to(f32)  to(bf16); v2p(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); r2v(bf16); to(f32)  v2p(f32)
f32   f32   bf16  bf16  r2v(f32); to(bf16)            v2p(bf16); to(f32)
f32   f32   bf16  f32   r2v(f32); to(bf16)            to(f32); v2p(f32)
f32   f32   f32   bf16  r2v(f32)                      to(bf16); v2p(bf16); to(f32)
f32   f32   f32   f32   r2v(f32)                      v2p(f32)
bf16  bf16  bf16  bf16  r2v(bf16)                     v2p(bf16)
bf16  bf16  bf16  f32   r2v(bf16)                     to(f32); v2p(f32); to(bf16)
bf16  bf16  f32   bf16  r2v(bf16); to(f32)            to(bf16); v2p(bf16)
bf16  bf16  f32   f32   r2v(bf16); to(f32)            v2p(f32); to(bf16)
bf16  f32   bf16  bf16  to(f32); r2v(f32); to(bf16)   v2p(bf16)
bf16  f32   bf16  f32   to(f32); r2v(f32); to(bf16)   to(f32); v2p(f32); to(bf16)
bf16  f32   f32   bf16  to(f32); r2v(f32)             to(bf16); v2p(bf16)
bf16  f32   f32   f32   to(f32); r2v(f32)             v2p(f32); to(bf16)""",
        )

    # -- R -> P (backward: convert R->P, self-dual, expert_mode) --

    # no fwd collective
    def test_R_P_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, R, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   forward
----  ----  ----  -------------------
f32   -     -     r2p(f32)
f32   -     bf16  to(bf16); r2p(bf16)
f32   -     f32   r2p(f32)
f32   bf16  -     to(bf16); r2p(bf16)
f32   f32   -     r2p(f32)
bf16  -     -     r2p(bf16)
bf16  -     bf16  r2p(bf16)
bf16  -     f32   r2p(bf16); to(f32)
bf16  bf16  -     r2p(bf16)
bf16  f32   -     to(f32); r2p(f32)""",
        )

    # no bwd collective (convert R->P is local)
    def test_R_P_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, R, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   backward
----  ----  ----  -------------------
f32   bf16  bf16  r2p(bf16); to(f32)
f32   bf16  f32   r2p(f32)
f32   f32   bf16  r2p(bf16); to(f32)
f32   f32   f32   r2p(f32)
bf16  bf16  bf16  r2p(bf16)
bf16  bf16  f32   to(bf16); r2p(bf16)
bf16  f32   bf16  r2p(bf16)
bf16  f32   f32   to(bf16); r2p(bf16)""",
        )

    def test_R_P(self):
        self._ex(
            self._table(
                lambda in_dt, **kw: self._run(in_dt, R, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   bop   forward                       backward
----  ----  ----  ----  ----------------------------  ----------------------------
f32   bf16  bf16  bf16  to(bf16); r2p(bf16)           r2p(bf16); to(f32)
f32   bf16  bf16  f32   to(bf16); r2p(bf16)           to(f32); r2p(f32)
f32   bf16  f32   bf16  to(bf16); r2p(bf16); to(f32)  to(bf16); r2p(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); r2p(bf16); to(f32)  r2p(f32)
f32   f32   bf16  bf16  r2p(f32); to(bf16)            r2p(bf16); to(f32)
f32   f32   bf16  f32   r2p(f32); to(bf16)            to(f32); r2p(f32)
f32   f32   f32   bf16  r2p(f32)                      to(bf16); r2p(bf16); to(f32)
f32   f32   f32   f32   r2p(f32)                      r2p(f32)
bf16  bf16  bf16  bf16  r2p(bf16)                     r2p(bf16)
bf16  bf16  bf16  f32   r2p(bf16)                     to(f32); r2p(f32); to(bf16)
bf16  bf16  f32   bf16  r2p(bf16); to(f32)            to(bf16); r2p(bf16)
bf16  bf16  f32   f32   r2p(bf16); to(f32)            r2p(f32); to(bf16)
bf16  f32   bf16  bf16  to(f32); r2p(f32); to(bf16)   r2p(bf16)
bf16  f32   bf16  f32   to(f32); r2p(f32); to(bf16)   to(f32); r2p(f32); to(bf16)
bf16  f32   f32   bf16  to(f32); r2p(f32)             to(bf16); r2p(bf16)
bf16  f32   f32   f32   to(f32); r2p(f32)             r2p(f32); to(bf16)""",
        )

    # -- I -> V (backward: all_gather V->I, no reduction) --

    # no fwd collective
    def test_I_V_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, I, V, **kw)),
            """\
in    op    out   forward
----  ----  ----  -------------------
f32   -     -     r2v(f32)
f32   -     bf16  r2v(f32); to(bf16)
f32   -     f32   r2v(f32)
f32   bf16  -     to(bf16); r2v(bf16)
f32   f32   -     r2v(f32)
bf16  -     -     r2v(bf16)
bf16  -     bf16  r2v(bf16)
bf16  -     f32   r2v(bf16); to(f32)
bf16  bf16  -     r2v(bf16)
bf16  f32   -     to(f32); r2v(f32)""",
        )

    # bwd is all_gather (non-reducing): op at min(in, out) dtype (min-bandwidth trick)
    def test_I_V_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, I, V, **kw)),
            """\
in    op    out   backward
----  ----  ----  --------------------------
f32   bf16  bf16  all_gather(bf16); to(f32)
f32   bf16  f32   all_gather(f32)
f32   f32   bf16  all_gather(bf16); to(f32)
f32   f32   f32   all_gather(f32)
bf16  bf16  bf16  all_gather(bf16)
bf16  bf16  f32   to(bf16); all_gather(bf16)
bf16  f32   bf16  all_gather(bf16)
bf16  f32   f32   to(bf16); all_gather(bf16)""",
        )

    def test_I_V(self):
        self._ex(
            self._table(lambda in_dt, **kw: self._run(in_dt, I, V, **kw)),
            """\
in    op    out   bop   forward                       backward
----  ----  ----  ----  ----------------------------  -----------------------------------
f32   bf16  bf16  bf16  to(bf16); r2v(bf16)           all_gather(bf16); to(f32)
f32   bf16  bf16  f32   to(bf16); r2v(bf16)           to(f32); all_gather(f32)
f32   bf16  f32   bf16  to(bf16); r2v(bf16); to(f32)  to(bf16); all_gather(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); r2v(bf16); to(f32)  all_gather(f32)
f32   f32   bf16  bf16  r2v(f32); to(bf16)            all_gather(bf16); to(f32)
f32   f32   bf16  f32   r2v(f32); to(bf16)            to(f32); all_gather(f32)
f32   f32   f32   bf16  r2v(f32)                      to(bf16); all_gather(bf16); to(f32)
f32   f32   f32   f32   r2v(f32)                      all_gather(f32)
bf16  bf16  bf16  bf16  r2v(bf16)                     all_gather(bf16)
bf16  bf16  bf16  f32   r2v(bf16)                     to(f32); all_gather(f32); to(bf16)
bf16  bf16  f32   bf16  r2v(bf16); to(f32)            to(bf16); all_gather(bf16)
bf16  bf16  f32   f32   r2v(bf16); to(f32)            all_gather(f32); to(bf16)
bf16  f32   bf16  bf16  to(f32); r2v(f32); to(bf16)   all_gather(bf16)
bf16  f32   bf16  f32   to(f32); r2v(f32); to(bf16)   to(f32); all_gather(f32); to(bf16)
bf16  f32   f32   bf16  to(f32); r2v(f32)             to(bf16); all_gather(bf16)
bf16  f32   f32   f32   to(f32); r2v(f32)             all_gather(f32); to(bf16)""",
        )

    # -- R -> I (backward: convert I->P, expert_mode) --

    # no fwd collective
    def test_R_I_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, R, I, expert_mode=True, **kw)
            ),
            """\
in    op    out   forward
----  ----  ----  --------
f32   -     -     -
f32   -     bf16  to(bf16)
f32   -     f32   -
f32   bf16  -     to(bf16)
f32   f32   -     -
bf16  -     -     -
bf16  -     bf16  -
bf16  -     f32   to(f32)
bf16  bf16  -     -
bf16  f32   -     to(f32)""",
        )

    # no bwd collective (convert I->P is local)
    def test_R_I_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, R, I, expert_mode=True, **kw)
            ),
            """\
in    op    out   backward
----  ----  ----  -------------------
f32   bf16  bf16  r2p(bf16); to(f32)
f32   bf16  f32   r2p(f32)
f32   f32   bf16  r2p(bf16); to(f32)
f32   f32   f32   r2p(f32)
bf16  bf16  bf16  r2p(bf16)
bf16  bf16  f32   to(bf16); r2p(bf16)
bf16  f32   bf16  r2p(bf16)
bf16  f32   f32   to(bf16); r2p(bf16)""",
        )

    def test_R_I(self):
        self._ex(
            self._table(
                lambda in_dt, **kw: self._run(in_dt, R, I, expert_mode=True, **kw)
            ),
            """\
in    op    out   bop   forward            backward
----  ----  ----  ----  -----------------  ----------------------------
f32   bf16  bf16  bf16  to(bf16)           r2p(bf16); to(f32)
f32   bf16  bf16  f32   to(bf16)           to(f32); r2p(f32)
f32   bf16  f32   bf16  to(bf16); to(f32)  to(bf16); r2p(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); to(f32)  r2p(f32)
f32   f32   bf16  bf16  to(bf16)           r2p(bf16); to(f32)
f32   f32   bf16  f32   to(bf16)           to(f32); r2p(f32)
f32   f32   f32   bf16  -                  to(bf16); r2p(bf16); to(f32)
f32   f32   f32   f32   -                  r2p(f32)
bf16  bf16  bf16  bf16  -                  r2p(bf16)
bf16  bf16  bf16  f32   -                  to(f32); r2p(f32); to(bf16)
bf16  bf16  f32   bf16  to(f32)            to(bf16); r2p(bf16)
bf16  bf16  f32   f32   to(f32)            r2p(f32); to(bf16)
bf16  f32   bf16  bf16  to(f32); to(bf16)  r2p(bf16)
bf16  f32   bf16  f32   to(f32); to(bf16)  to(f32); r2p(f32); to(bf16)
bf16  f32   f32   bf16  to(f32)            to(bf16); r2p(bf16)
bf16  f32   f32   f32   to(f32)            r2p(f32); to(bf16)""",
        )

    # -- I -> P (backward: convert R->I, expert_mode) --

    # no fwd collective
    def test_I_P_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, I, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   forward
----  ----  ----  -------------------
f32   -     -     r2p(f32)
f32   -     bf16  to(bf16); r2p(bf16)
f32   -     f32   r2p(f32)
f32   bf16  -     to(bf16); r2p(bf16)
f32   f32   -     r2p(f32)
bf16  -     -     r2p(bf16)
bf16  -     bf16  r2p(bf16)
bf16  -     f32   r2p(bf16); to(f32)
bf16  bf16  -     r2p(bf16)
bf16  f32   -     to(f32); r2p(f32)""",
        )

    # no bwd collective (convert R->I is local)
    def test_I_P_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, I, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   backward
----  ----  ----  --------
f32   bf16  bf16  to(f32)
f32   bf16  f32   -
f32   f32   bf16  to(f32)
f32   f32   f32   -
bf16  bf16  bf16  -
bf16  bf16  f32   to(bf16)
bf16  f32   bf16  -
bf16  f32   f32   to(bf16)""",
        )

    def test_I_P(self):
        self._ex(
            self._table(
                lambda in_dt, **kw: self._run(in_dt, I, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   bop   forward                       backward
----  ----  ----  ----  ----------------------------  -----------------
f32   bf16  bf16  bf16  to(bf16); r2p(bf16)           to(f32)
f32   bf16  bf16  f32   to(bf16); r2p(bf16)           to(f32)
f32   bf16  f32   bf16  to(bf16); r2p(bf16); to(f32)  to(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); r2p(bf16); to(f32)  -
f32   f32   bf16  bf16  r2p(f32); to(bf16)            to(f32)
f32   f32   bf16  f32   r2p(f32); to(bf16)            to(f32)
f32   f32   f32   bf16  r2p(f32)                      to(bf16); to(f32)
f32   f32   f32   f32   r2p(f32)                      -
bf16  bf16  bf16  bf16  r2p(bf16)                     -
bf16  bf16  bf16  f32   r2p(bf16)                     to(f32); to(bf16)
bf16  bf16  f32   bf16  r2p(bf16); to(f32)            to(bf16)
bf16  bf16  f32   f32   r2p(bf16); to(f32)            to(bf16)
bf16  f32   bf16  bf16  to(f32); r2p(f32); to(bf16)   -
bf16  f32   bf16  f32   to(f32); r2p(f32); to(bf16)   to(f32); to(bf16)
bf16  f32   f32   bf16  to(f32); r2p(f32)             to(bf16)
bf16  f32   f32   f32   to(f32); r2p(f32)             to(bf16)""",
        )

    # -- V -> P (backward: convert R->V, expert_mode) --

    # no fwd collective
    def test_V_P_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, V, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   forward
----  ----  ----  -------------------
f32   -     -     v2p(f32)
f32   -     bf16  to(bf16); v2p(bf16)
f32   -     f32   v2p(f32)
f32   bf16  -     to(bf16); v2p(bf16)
f32   f32   -     v2p(f32)
bf16  -     -     v2p(bf16)
bf16  -     bf16  v2p(bf16)
bf16  -     f32   v2p(bf16); to(f32)
bf16  bf16  -     v2p(bf16)
bf16  f32   -     to(f32); v2p(f32)""",
        )

    # no bwd collective (convert R->V is local)
    def test_V_P_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, V, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   backward
----  ----  ----  ------------------
f32   bf16  bf16  r2v(bf16); to(f32)
f32   bf16  f32   r2v(f32)
f32   f32   bf16  r2v(bf16); to(f32)
f32   f32   f32   r2v(f32)
bf16  bf16  bf16  r2v(bf16)
bf16  bf16  f32   r2v(f32); to(bf16)
bf16  f32   bf16  r2v(bf16)
bf16  f32   f32   r2v(f32); to(bf16)""",
        )

    def test_V_P(self):
        self._ex(
            self._table(
                lambda in_dt, **kw: self._run(in_dt, V, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   bop   forward                       backward
----  ----  ----  ----  ----------------------------  ----------------------------
f32   bf16  bf16  bf16  to(bf16); v2p(bf16)           r2v(bf16); to(f32)
f32   bf16  bf16  f32   to(bf16); v2p(bf16)           to(f32); r2v(f32)
f32   bf16  f32   bf16  to(bf16); v2p(bf16); to(f32)  to(bf16); r2v(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); v2p(bf16); to(f32)  r2v(f32)
f32   f32   bf16  bf16  v2p(f32); to(bf16)            r2v(bf16); to(f32)
f32   f32   bf16  f32   v2p(f32); to(bf16)            to(f32); r2v(f32)
f32   f32   f32   bf16  v2p(f32)                      to(bf16); r2v(bf16); to(f32)
f32   f32   f32   f32   v2p(f32)                      r2v(f32)
bf16  bf16  bf16  bf16  v2p(bf16)                     r2v(bf16)
bf16  bf16  bf16  f32   v2p(bf16)                     to(f32); r2v(f32); to(bf16)
bf16  bf16  f32   bf16  v2p(bf16); to(f32)            to(bf16); r2v(bf16)
bf16  bf16  f32   f32   v2p(bf16); to(f32)            r2v(f32); to(bf16)
bf16  f32   bf16  bf16  to(f32); v2p(f32); to(bf16)   r2v(bf16)
bf16  f32   bf16  f32   to(f32); v2p(f32); to(bf16)   to(f32); r2v(f32); to(bf16)
bf16  f32   f32   bf16  to(f32); v2p(f32)             to(bf16); r2v(bf16)
bf16  f32   f32   f32   to(f32); v2p(f32)             r2v(f32); to(bf16)""",
        )

    # -- expert_mode errors --

    def test_V_P_expert_error(self):
        self._ex(
            self._run(f32, V, P, requires_grad=False),
            """error: convert(PerMeshAxisLocalSpmdType.V, P) requires expert_mode=True.""",
        )


# =====================================================================
# reinterpret: no-op forward, changes type only
# =====================================================================


class TestReinterpretDtype(_DtypeTestBase):
    def _run(self, input_dtype, src, dst, *, requires_grad=True, **kw):
        src_type = V if isinstance(src, Shard) else src
        shape = (3, 4) if src is I and dst is V else (4,)
        return self._trace(
            lambda x, pg: reinterpret(x, pg, src=src, dst=dst, **kw),
            src_type=src_type,
            shape=shape,
            input_dtype=input_dtype,
            requires_grad=requires_grad,
        )

    # -- V -> P (backward: reinterpret R->V, no reduction) --

    # no fwd collective
    def test_V_P_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, V, P, **kw)),
            """\
in    op    out   forward
----  ----  ----  --------
f32   -     -     -
f32   -     bf16  to(bf16)
f32   -     f32   -
f32   bf16  -     to(bf16)
f32   f32   -     -
bf16  -     -     -
bf16  -     bf16  -
bf16  -     f32   to(f32)
bf16  bf16  -     -
bf16  f32   -     to(f32)""",
        )

    # no bwd collective (reinterpret R->V is local)
    def test_V_P_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(lambda in_dt, **kw: self._run(in_dt, V, P, **kw)),
            """\
in    op    out   backward
----  ----  ----  --------
f32   bf16  bf16  to(f32)
f32   bf16  f32   -
f32   f32   bf16  to(f32)
f32   f32   f32   -
bf16  bf16  bf16  -
bf16  bf16  f32   to(bf16)
bf16  f32   bf16  -
bf16  f32   f32   to(bf16)""",
        )

    def test_V_P(self):
        self._ex(
            self._table(lambda in_dt, **kw: self._run(in_dt, V, P, **kw)),
            """\
in    op    out   bop   forward            backward
----  ----  ----  ----  -----------------  -----------------
f32   bf16  bf16  bf16  to(bf16)           to(f32)
f32   bf16  bf16  f32   to(bf16)           to(f32)
f32   bf16  f32   bf16  to(bf16); to(f32)  to(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); to(f32)  -
f32   f32   bf16  bf16  to(bf16)           to(f32)
f32   f32   bf16  f32   to(bf16)           to(f32)
f32   f32   f32   bf16  -                  to(bf16); to(f32)
f32   f32   f32   f32   -                  -
bf16  bf16  bf16  bf16  -                  -
bf16  bf16  bf16  f32   -                  to(f32); to(bf16)
bf16  bf16  f32   bf16  to(f32)            to(bf16)
bf16  bf16  f32   f32   to(f32)            to(bf16)
bf16  f32   bf16  bf16  to(f32); to(bf16)  -
bf16  f32   bf16  f32   to(f32); to(bf16)  to(f32); to(bf16)
bf16  f32   f32   bf16  to(f32)            to(bf16)
bf16  f32   f32   f32   to(f32)            to(bf16)""",
        )

    # -- R -> P (self-dual backward, expert_mode) --

    # no fwd collective
    def test_R_P_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, R, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   forward
----  ----  ----  --------
f32   -     -     -
f32   -     bf16  to(bf16)
f32   -     f32   -
f32   bf16  -     to(bf16)
f32   f32   -     -
bf16  -     -     -
bf16  -     bf16  -
bf16  -     f32   to(f32)
bf16  bf16  -     -
bf16  f32   -     to(f32)""",
        )

    # no bwd collective (reinterpret R->P is local)
    def test_R_P_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, R, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   backward
----  ----  ----  --------
f32   bf16  bf16  to(f32)
f32   bf16  f32   -
f32   f32   bf16  to(f32)
f32   f32   f32   -
bf16  bf16  bf16  -
bf16  bf16  f32   to(bf16)
bf16  f32   bf16  -
bf16  f32   f32   to(bf16)""",
        )

    def test_R_P(self):
        self._ex(
            self._table(
                lambda in_dt, **kw: self._run(in_dt, R, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   bop   forward            backward
----  ----  ----  ----  -----------------  -----------------
f32   bf16  bf16  bf16  to(bf16)           to(f32)
f32   bf16  bf16  f32   to(bf16)           to(f32)
f32   bf16  f32   bf16  to(bf16); to(f32)  to(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); to(f32)  -
f32   f32   bf16  bf16  to(bf16)           to(f32)
f32   f32   bf16  f32   to(bf16)           to(f32)
f32   f32   f32   bf16  -                  to(bf16); to(f32)
f32   f32   f32   f32   -                  -
bf16  bf16  bf16  bf16  -                  -
bf16  bf16  bf16  f32   -                  to(f32); to(bf16)
bf16  bf16  f32   bf16  to(f32)            to(bf16)
bf16  bf16  f32   f32   to(f32)            to(bf16)
bf16  f32   bf16  bf16  to(f32); to(bf16)  -
bf16  f32   bf16  f32   to(f32); to(bf16)  to(f32); to(bf16)
bf16  f32   f32   bf16  to(f32)            to(bf16)
bf16  f32   f32   f32   to(f32)            to(bf16)""",
        )

    # -- composition: I -> R -> V (backward: all_reduce, expert_mode) --

    # no fwd collective
    def test_I_V_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, I, V, expert_mode=True, **kw)
            ),
            """\
in    op    out   forward
----  ----  ----  --------
f32   -     -     -
f32   -     bf16  to(bf16)
f32   -     f32   -
f32   bf16  -     to(bf16)
f32   f32   -     -
bf16  -     -     -
bf16  -     bf16  -
bf16  -     f32   to(f32)
bf16  bf16  -     -
bf16  f32   -     to(f32)""",
        )

    # bwd is all_reduce (reducing): no min-bandwidth trick
    def test_I_V_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, I, V, expert_mode=True, **kw)
            ),
            """\
in    op    out   backward
----  ----  ----  -----------------------------------
f32   bf16  bf16  all_reduce(bf16); to(f32)
f32   bf16  f32   to(bf16); all_reduce(bf16); to(f32)
f32   f32   bf16  to(f32); all_reduce(f32)
f32   f32   f32   all_reduce(f32)
bf16  bf16  bf16  all_reduce(bf16)
bf16  bf16  f32   to(bf16); all_reduce(bf16)
bf16  f32   bf16  to(f32); all_reduce(f32); to(bf16)
bf16  f32   f32   all_reduce(f32); to(bf16)""",
        )

    def test_I_V(self):
        self._ex(
            self._table(
                lambda in_dt, **kw: self._run(in_dt, I, V, expert_mode=True, **kw)
            ),
            """\
in    op    out   bop   forward            backward
----  ----  ----  ----  -----------------  ------------------------------------------------------
f32   bf16  bf16  bf16  to(bf16)           all_reduce(bf16); to(f32)
f32   bf16  bf16  f32   to(bf16)           to(f32); to(bf16); to(f32); all_reduce(f32)
f32   bf16  f32   bf16  to(bf16); to(f32)  to(bf16); all_reduce(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); to(f32)  to(bf16); to(f32); all_reduce(f32)
f32   f32   bf16  bf16  to(bf16)           to(f32); to(bf16); all_reduce(bf16); to(f32)
f32   f32   bf16  f32   to(bf16)           to(f32); all_reduce(f32)
f32   f32   f32   bf16  -                  to(bf16); to(f32); to(bf16); all_reduce(bf16); to(f32)
f32   f32   f32   f32   -                  all_reduce(f32)
bf16  bf16  bf16  bf16  -                  all_reduce(bf16)
bf16  bf16  bf16  f32   -                  to(f32); to(bf16); to(f32); all_reduce(f32); to(bf16)
bf16  bf16  f32   bf16  to(f32)            to(bf16); all_reduce(bf16)
bf16  bf16  f32   f32   to(f32)            to(bf16); to(f32); all_reduce(f32); to(bf16)
bf16  f32   bf16  bf16  to(f32); to(bf16)  to(f32); to(bf16); all_reduce(bf16)
bf16  f32   bf16  f32   to(f32); to(bf16)  to(f32); all_reduce(f32); to(bf16)
bf16  f32   f32   bf16  to(f32)            to(bf16); to(f32); to(bf16); all_reduce(bf16)
bf16  f32   f32   f32   to(f32)            all_reduce(f32); to(bf16)""",
        )

    # -- composition: I -> R -> P (backward: all_reduce, expert_mode) --

    # no fwd collective
    def test_I_P_fwd_defaults(self):
        self._ex(
            self._fwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, I, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   forward
----  ----  ----  --------
f32   -     -     -
f32   -     bf16  to(bf16)
f32   -     f32   -
f32   bf16  -     to(bf16)
f32   f32   -     -
bf16  -     -     -
bf16  -     bf16  -
bf16  -     f32   to(f32)
bf16  bf16  -     -
bf16  f32   -     to(f32)""",
        )

    # bwd is all_reduce (reducing): no min-bandwidth trick
    def test_I_P_bwd_defaults(self):
        self._ex(
            self._bwd_defaults_table(
                lambda in_dt, **kw: self._run(in_dt, I, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   backward
----  ----  ----  -----------------------------------
f32   bf16  bf16  all_reduce(bf16); to(f32)
f32   bf16  f32   to(bf16); all_reduce(bf16); to(f32)
f32   f32   bf16  to(f32); all_reduce(f32)
f32   f32   f32   all_reduce(f32)
bf16  bf16  bf16  all_reduce(bf16)
bf16  bf16  f32   to(bf16); all_reduce(bf16)
bf16  f32   bf16  to(f32); all_reduce(f32); to(bf16)
bf16  f32   f32   all_reduce(f32); to(bf16)""",
        )

    def test_I_P(self):
        self._ex(
            self._table(
                lambda in_dt, **kw: self._run(in_dt, I, P, expert_mode=True, **kw)
            ),
            """\
in    op    out   bop   forward            backward
----  ----  ----  ----  -----------------  ------------------------------------------------------
f32   bf16  bf16  bf16  to(bf16)           all_reduce(bf16); to(f32)
f32   bf16  bf16  f32   to(bf16)           to(f32); to(bf16); to(f32); all_reduce(f32)
f32   bf16  f32   bf16  to(bf16); to(f32)  to(bf16); all_reduce(bf16); to(f32)
f32   bf16  f32   f32   to(bf16); to(f32)  to(bf16); to(f32); all_reduce(f32)
f32   f32   bf16  bf16  to(bf16)           to(f32); to(bf16); all_reduce(bf16); to(f32)
f32   f32   bf16  f32   to(bf16)           to(f32); all_reduce(f32)
f32   f32   f32   bf16  -                  to(bf16); to(f32); to(bf16); all_reduce(bf16); to(f32)
f32   f32   f32   f32   -                  all_reduce(f32)
bf16  bf16  bf16  bf16  -                  all_reduce(bf16)
bf16  bf16  bf16  f32   -                  to(f32); to(bf16); to(f32); all_reduce(f32); to(bf16)
bf16  bf16  f32   bf16  to(f32)            to(bf16); all_reduce(bf16)
bf16  bf16  f32   f32   to(f32)            to(bf16); to(f32); all_reduce(f32); to(bf16)
bf16  f32   bf16  bf16  to(f32); to(bf16)  to(f32); to(bf16); all_reduce(bf16)
bf16  f32   bf16  f32   to(f32); to(bf16)  to(f32); all_reduce(f32); to(bf16)
bf16  f32   f32   bf16  to(f32)            to(bf16); to(f32); to(bf16); all_reduce(bf16)
bf16  f32   f32   f32   to(f32)            all_reduce(f32); to(bf16)""",
        )

    # -- expert_mode error --

    def test_R_V_expert_error(self):
        self._ex(
            self._run(f32, R, V, requires_grad=False),
            """error: reinterpret(R, V) requires expert_mode=True.""",
        )


# =====================================================================
# convenience aliases
# =====================================================================


class TestConvenienceAliasesDtype(_DtypeTestBase):
    # -- invariant_to_replicate (convert I->R) --

    def test_i2r(self):
        self._ex(
            self._trace(lambda x, pg: invariant_to_replicate(x, pg), src_type=I),
            """- | all_reduce(f32)""",
        )

    def test_i2r_op_bf16_bwd_f32(self):
        self._ex(
            self._trace(
                lambda x, pg: invariant_to_replicate(
                    x, pg, op_dtype=bf16, backward_options={"op_dtype": f32}
                ),
                src_type=I,
            ),
            """to(bf16) | to(f32); all_reduce(f32)""",
        )

    def test_i2r_op_bf16_error(self):
        result = self._trace(
            lambda x, pg: invariant_to_replicate(x, pg, op_dtype=bf16), src_type=I
        )
        self.assertEqual(result.error, _BACKWARD_ERROR)

    def test_i2r_bf16_input_error(self):
        result = self._trace(
            lambda x, pg: invariant_to_replicate(x, pg),
            src_type=I,
            input_dtype=bf16,
        )
        self.assertEqual(result.error, _BACKWARD_ERROR)

    def test_i2r_bf16_input_bwd_ok(self):
        self._ex(
            self._trace(
                lambda x, pg: invariant_to_replicate(
                    x, pg, backward_options={"op_dtype": bf16}
                ),
                src_type=I,
                input_dtype=bf16,
            ),
            """- | all_reduce(bf16)""",
        )

    # -- shard (convert R -> S(i)) --

    def test_shard_op_bf16(self):
        self._ex(
            self._trace(
                lambda x, pg: shard(x, pg, src=R, dst=S(0), op_dtype=bf16),
                src_type=R,
                shape=(6,),
            ),
            """to(bf16); r2v(bf16) | v2p(bf16); to(f32)""",
        )

    # -- unshard (all_gather S(i) -> R|I) --

    def test_unshard_op_bf16(self):
        self._ex(
            self._trace(
                lambda x, pg: unshard(x, pg, src=S(0), dst=I, op_dtype=bf16),
                src_type=V,
                shape=(6,),
            ),
            """to(bf16); all_gather(bf16) | r2v(bf16); to(f32)""",
        )

    # -- redistribute --

    def test_redistribute_V_R(self):
        self._ex(
            self._trace(
                lambda x, pg: redistribute(
                    x,
                    pg,
                    src=V,
                    dst=R,
                    op_dtype=bf16,
                    backward_options={"op_dtype": f32},
                ),
                src_type=V,
                shape=(6,),
            ),
            """to(bf16); all_gather(bf16) | to(f32); reduce_scatter(f32)""",
        )

    def test_redistribute_P_R(self):
        self._ex(
            self._trace(
                lambda x, pg: redistribute(
                    x,
                    pg,
                    src=P,
                    dst=R,
                    op_dtype=bf16,
                    backward_options={"op_dtype": f32},
                ),
                src_type=P,
            ),
            """to(bf16); all_reduce(bf16) | to(f32); all_reduce(f32)""",
        )


# =====================================================================
# validation
# =====================================================================


class TestDtypeValidation(_DtypeTestBase):
    def test_unknown_backward_options_key(self):
        self._ex(
            self._trace(
                lambda x, pg: all_reduce(
                    x, pg, src=P, dst=I, backward_options={"op_dtypo": bf16}
                ),
                src_type=P,
            ),
            """error: Unknown keys in backward_options: {'op_dtypo'}.""",
        )

    def test_no_grad_bypasses_backward_check(self):
        with torch.no_grad():
            self._ex(
                self._trace(
                    lambda x, pg: all_reduce(x, pg, src=P, dst=R, op_dtype=bf16),
                    src_type=P,
                    requires_grad=False,
                ),
                """to(bf16); all_reduce(bf16)""",
            )

    def test_composition_nested_backward_options_error(self):
        self._ex(
            self._trace(
                lambda x, pg: reinterpret(
                    x,
                    pg,
                    src=I,
                    dst=V,
                    expert_mode=True,
                    op_dtype=bf16,
                    backward_options={
                        "op_dtype": f32,
                        "backward_options": {"op_dtype": bf16},
                    },
                ),
                src_type=I,
                shape=(3, 4),
                input_dtype=f32,
            ),
            """error: backward_options with nested backward_options (double backward) is not supported for composed operations (reinterpret I->V, I->P)""",
        )

    def test_double_backward_options(self):
        self._ex(
            self._trace(
                lambda x, pg: all_reduce(
                    x,
                    pg,
                    src=P,
                    dst=R,
                    op_dtype=bf16,
                    backward_options={
                        "op_dtype": f32,
                        "backward_options": {"op_dtype": bf16},
                    },
                ),
                src_type=P,
            ),
            """to(bf16); all_reduce(bf16) | to(f32); all_reduce(f32)""",
        )


if __name__ == "__main__":
    unittest.main()
