# Conservation preflight — implementation

Zero-training design audit for `open-audit-20260904`. No neural framework is
imported and no model is fitted beyond two-variable constant least squares.

## Files

| file | role |
|---|---|
| `preflight_lib.py` | constant-fit algebra: weighted moments, unconstrained normal equations, exact box-constrained solve, closure gate, plug-in `G`/`Gamma`, balanced-flow interval, one-flow collapse, deterministic hashing, independent `to_hourly` |
| `preflight_data.py` | hourly transition construction and the outcome-blind exogenous active-48 window |
| `run_preflight.py` | integrity gates, all three metric tables, `RUN_PROVENANCE.json` |
| `test_preflight.py` | the eight mandatory algebra and data-path tests (13 test functions) |
| `validate_outputs.py` | Phase 6 independent validation of the generated tables |
| `make_reports.py` | `PREFLIGHT_SUMMARY.json` and `PREFLIGHT_REPORT.md`, generated programmatically |

## Reproducing

Two clean clones are required; no local checkout, dataset, cache or checkpoint
may be substituted.

```bash
ROOT="$(mktemp -d)"
REPO=https://github.com/ShuaiWang-Castle/asymode-open-data
git clone --branch open-audit-20260904 --single-branch "$REPO" "$ROOT/code"
git clone --filter=blob:none --no-checkout "$REPO" "$ROOT/public-data"
git -C "$ROOT/public-data" sparse-checkout init --cone
git -C "$ROOT/public-data" sparse-checkout set data configs
git -C "$ROOT/public-data" fetch --depth 1 origin 8dd47c5ccd829611f27b69a3d64c274a0a24c400
git -C "$ROOT/public-data" checkout --detach 8dd47c5ccd829611f27b69a3d64c274a0a24c400

python3.11 -m venv "$ROOT/venv"
"$ROOT/venv/bin/pip" install numpy==2.1.3 pandas==2.2.3 scipy==1.14.1 \
    pyarrow==18.1.0 pytest==8.3.4

cd "$ROOT/code/analysis/conservation_preflight_20260904/implementation"
"$ROOT/venv/bin/python" -m pytest -q test_preflight.py
"$ROOT/venv/bin/python" run_preflight.py --code "$ROOT/code" --data "$ROOT/public-data"
"$ROOT/venv/bin/python" validate_outputs.py "$ROOT/code"
"$ROOT/venv/bin/python" make_reports.py "$ROOT/code" pytest_output.txt
```

The virtual environment is created outside the code clone so the worktree stays
clean and the environment cannot be committed.

## Determinism

`run_preflight.py` was executed twice independently on the pinned inputs; all
three CSV tables were byte-identical. Every source of ordering is fixed:
row capping and query selection use a SHA-256 keyed hash of
`(event, county, physical_hour)` rather than Python's per-process `hash()`;
PCA uses an economy SVD with an explicit sign convention; and nearest-neighbour
ties are broken by row index via `lexsort`.

## Fixed parameters

`CAP_U=0.265`, `CAP_R=0.25`, closure tolerance `0.05`, `K=2`, `k=200`,
800 queries per fold and design, 12,000-row cap per source event, 5 PCA
dimensions, 48-transition active window centred on the NOAA footprint peak,
seed 0. None was swept.
