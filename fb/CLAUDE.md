# Internal (fbsource) instructions

Imported by `../CLAUDE.md`. ShipIt strips `fb/` from the meta-pytorch/spmd_types
GitHub export, so Meta-internal agent guidance goes here, not in `../CLAUDE.md`.

## Python environment and build system

- Never use Buck here: no `buck2 build`, `buck2 test`, or `buck2 run`, even though
  this directory has `BUCK` and `PACKAGE` files and fbsource guidance says most
  projects use Buck. Those targets exist for downstream Buck consumers (llama4x,
  sixlib) and legacy conda CI.
- Like `genai/msl/farm`, spmd_types runs under **uv**. Its environment is the
  internal uv project in `fb/ci` (the MSL torch wheel set Buildkite CI runs).
  Never call the system `python` or `python3`; run everything with
  `uv run --project fb/ci`.

```bash
# From genai/msl/spmd_types
uv sync --project fb/ci
uv run --project fb/ci pytest -q tests
uv run --project fb/ci pytest -x -s tests/checker_test.py
```

## Dependencies

If you need to consult a copy of PyTorch for source diving, there is one at
fbsource/fbcode/caffe2
