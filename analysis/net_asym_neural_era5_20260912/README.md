# NET / ASYM neural round (2026-09-12)

One question: under the same available information and the same full-trajectory
supervision, which conditional response distributions favour a single net flow (NET)
and which favour explicit two-flow source pools (ASYM)?

Only two learned models exist in this round:

- **NET**: `y[k+1] = y[k] + scale * net(context, step_x[k], y[k]/scale)`; a general net
  recursion that reads its own predicted state. It is not a signed single-rate model.
- **ASYM**: `(u, r, stay) = softmax(rate_logits(context, step_x[k]), 0)`,
  `y[k+1] = y[k] + u (1 - y[k]) - r y[k]`; rates never read the recursive state.

Both come from the audited package core (`code/core_models_base.py`), are trained on the
full-path squared loss with no teacher forcing, and are compared as
`Delta = MSE_NET - MSE_ASYM` (positive favours ASYM).

## Tasks

| task | data | status |
|---|---|---|
| A. US EAGLE-I + ERA5 | 26 public panels and matched ERA5 drivers; retrospective, given-reanalysis conditional response | see `RESULTS_ZH.md` |
| B. ANEEL | fixed 24-collection ledger; targeted selection repair; reused 2019 retrospective evaluation | see `RESULTS_ZH.md` |
| C. Synthetic | exact binomial source-pool mixture laws with identical x and zero start | see `RESULTS_ZH.md` |

## Layout

| path | contents |
|---|---|
| `PROTOCOL_ZH.md` | the round protocol, verbatim |
| `CONFIG_LOCK.json`, `SOURCE_COMMITS.json` | locked configuration and consulted commits |
| `intake/` | data inventory, byte-level audits, schema drift, missing-data request |
| `locks/` | frozen splits and window index, data hashes, resource plan, selection locks |
| `code/`, `tests/` | implementation and correctness tests |
| `checkpoints/`, `predictions/` | final models and sharded predictions |
| `results/`, `figures/` | tables and figures built from saved predictions |
| `logs/` | trial registry, environment, gates, negative results and failures |

## Replay without training

```bash
./REPLAY_ONLY.sh                      # every checkpoint, all three tasks
./REPLAY_ONLY.sh --task us --max-models 1
```

`REPRODUCE_ALL.sh` reruns everything from the repository data and retrains every model.

## Data caveat that shapes Task A

The published driver files encode 1,265 of 2,625 panel counties as all-zero weather.
Those counties are excluded as missing inputs for both models, never fed as zero; see
`intake/SCHEMA_DRIFT.md` and `intake/MISSING_DATA_REQUEST.md`.

## Licences

EAGLE-I 2014-2022: CC BY 4.0. EAGLE-I 2024: no reuse restrictions. ERA5: Copernicus
licence, derived product with attribution. ANEEL-derived files remain under the ODbL
with attribution to ANEEL and stay in their own directories.
