# Reproduction commands

Branch `open-audit-20260904`. All commands are run from the repository root with the
committed virtual environment. `PYTHONPATH` must include both `src` and
`experiments`.

## 0. Environment

```bash
./.venv/bin/python -c "import torch, numpy, pandas, sklearn, scipy; \
  print(torch.__version__, numpy.__version__, pandas.__version__)"
# 2.11.0 2.2.6 <pandas>
```

Deterministic on CPU. The pilot was verified to reproduce bitwise: rerunning one
complete pilot event with the same seed reproduced `selected_update`,
`selected_validation`, `test_path_mse_full`, `test_tf_mse_full` and
`test_h24_mse_full` with a maximum absolute difference of `0.000e+00`.

## 1. Freeze the data design (must run before the model is imported)

```bash
PYTHONPATH=src:experiments ./.venv/bin/python experiments/paper_v2_event_design.py
```

Writes `event_design_table.csv`, `event_design_table.md`, `event_folds_v2.json` and
`_origin_audit.json` under `analysis/gpt_rescue_20260904/cc_v2/`. The fold map and
its digest are written to disk here, before any model import.

Expected: 26 panels, panel digest `db286b4960a4`, fold digest `beb00a6762ba`.

## 2. Verify the exact constant-class solvers against brute force

```bash
PYTHONPATH=src ./.venv/bin/python - <<'PY'
import numpy as np
from asymode_paper.initialization import fit_two_flow, fit_u_ray, fit_r_ray
# 400 random problems; no grid point may beat the closed form
PY
```

The full script used for the audit is reproduced in
`MODEL_IMPLEMENTATION_AUDIT.md`; the reported result is a worst deficit of
`+0.000e+00` on all three solvers.

## 3. Verify the update-0 identity

```bash
PYTHONPATH=src ./.venv/bin/python - <<'PY'
from asymode_paper.asymmetric_flows import AsymmetricFlows, CAP_U_MAIN, CAP_U_BKG, CAP_R
from asymode_paper.initialization import modular_init
spec = modular_init(2.674e-05, 0.0267395, CAP_U_MAIN, CAP_U_BKG, CAP_R)
m = AsymmetricFlows(32, 6, 23, "asym_two_flow"); m.apply_modular_init(spec)
print(m.constant_flows())     # must match (U0, R0) to <= 1e-6
PY
```

## 4. Run the nine-job pilot

```bash
PYTHONPATH=src:experiments ./.venv/bin/python -u experiments/paper_v2_pilot.py --seed 0
```

Roughly 17 minutes on CPU. Writes `pilot_results.json`,
`pilot_event_effects.csv`, `pilot_training_diagnostics.csv` and
`pilot_run_config.json`.

Pilot events are selected inside the script by exogenous medoid distance and are
not passed in: `2024-05-08` (convective), `2022-03-12` (winter), `2018-10-11`
(tropical).

## 5. Reproducibility check

```bash
PYTHONPATH=src:experiments ./.venv/bin/python -u experiments/paper_v2_pilot.py \
  --seed 0 --events 2024-05-08
```

**This overwrites the three pilot output files with the single-event run.** Copy
them aside first, or rerun step 4 afterwards to restore the full set.

## Files

| path | role |
|---|---|
| `src/asymode_paper/features.py` | the three process-specific blocks |
| `src/asymode_paper/initialization.py` | exact bounded solvers and the modular init |
| `src/asymode_paper/asymmetric_flows.py` | the scaffold and the output-level collapse |
| `src/asymode_paper/trainer.py` | Stage A / Stage B update-budgeted training |
| `experiments/paper_v2_event_design.py` | Task 1, the frozen design |
| `experiments/paper_v2_pilot.py` | Tasks 4-6, the nine jobs and the baselines |

No arm was added to `src/asymode/dynamics.py`.
