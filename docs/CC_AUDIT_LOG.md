# CC audit log — AISTATS soundness repair

Branch `aistats-theory-protocol-repair`, opened 2026-09-02 from `3ec2a60`. Append-only.

## 0. Firewall first

`FIREWALL.md` read before anything else. One instruction in the mission prompt
conflicts with it and was **not followed**: the prompt names an external
"primary repository" outside this tree. This repository is the entire allowed
universe; nothing outside it was read, cloned, or referenced. All other prompt
rules are consistent with `FIREWALL.md`. The companion theory documents were
read from the package only, treated as a specification to verify, not as
authority; every identity in them was re-derived and tested (`docs/CC_THEORY_CHECK.md`).

## 1. Handoff identity (A1)

| item | recorded in prompt | measured | match |
|---|---|---|---|
| export zip SHA256 | acc481aa…e664d3 | acc481aac8e1cf736996cd9ba8978ecbdac91919f2582bf13a9e40e79fe664d3 | yes |
| exported HEAD | 679146c | export MANIFEST HEAD 679146c | yes |
| `paper/DRAFT.md` SHA256 | 33ed6da4…f9310 | 33ed6da4… at 679146c and at current HEAD | yes |
| root `DRAFT.md` | "duplicated" | not tracked in this repository (a loose copy was sent separately) | n/a |
| package SHA256SUMS | 4 files | all OK (`shasum -c`); loose copies in Downloads byte-identical | yes |

Current HEAD at branch creation: `3ec2a60`, two docs-only commits after the export
(`3e21b5f` pre-registration E1–E7; `3ec2a60` H-H prior). Tracked files 94 =
manifest 93 + `docs/PREREGISTRATION_long_horizon.md`. Working tree clean.

## 2. Environment

Python 3.11.6, arm64 macOS 13.7; numpy 2.2.6, torch 2.11.0 (CPU), scikit-learn
1.9.0, pandas 3.0.5, scipy 1.17.1, xarray 2026.7.0, geopandas 1.1.4; pytest 9.1.1
and hypothesis 6.167.1 installed on this branch; lock file `requirements.lock.txt`.
Apple M2, 8 cores, 16 GB, no GPU. **Public raw/interim data present** (gitignored):
`data/raw` 4.1 GB (census, eaglei, eia, era5 26 windows, nws, storm_events),
`data/interim` 451 MB (26 `panel_*.npz` with `ts`, 26 `drivers_*.npz`, EAGLE-I
parquet 2018–2024), `results/` 15 MB incl. 15 `oof_*.npz` exports. Nothing is
blocked by missing data; compute is CPU-only (a 135-fit run takes ~2–3 h).

## 3. Read order (§2 of the prompt)

1–11 read in full this session (FIREWALL, MANIFEST, STATUS, EVIDENCE_SUMMARY,
RESULTS_LEDGER, paper/DRAFT, THEORY_PLAN, DEEP_DIVE, AISTATS_FIT, DATA_CARD,
all PREREGISTRATION_*). 12–14 read for the audited paths: `evalproto.make_folds`,
`inner_split`, `to_hourly`; `panels.source_version`; `dynamics` (state equation,
`GatedRate`, `TwoRateConfig`); `features` (family registry); `exp05` (`load_pooled`,
`add_context`, main loop, config assembly, OOF writer); `exp06/07/08/10` call sites
of `add_context` and `make_folds`; `paired_review`, `check_comparable`,
`make_figures`. `fit.py`, `exp01/02/09` and `d1–d5` were read earlier in the
project and were not re-read line by line here.

## 4. Gates

| gate | status | evidence |
|---|---|---|
| theory unit tests (A3 items 1–8) | **PASS** 1,726 | `tests/test_theory.py` |
| split/schema unit tests | **PASS** 4 | `tests/test_splits_schema.py` |
| fixed-split harness test (A3 item 9) | pending wiring | — |
| timestamp clock test (A3 item 10) | pending wiring | — |
| result-schema / check_comparable fail-closed (item 11) | pending wiring | — |
| OOF uniqueness (item 12) | pending wiring | — |
| archive reproduction (A2) | done | `docs/CC_ARCHIVE_REPRO_AUDIT.md` |

**No scientific rerun has been launched.** Per the prompt, none may be until the
four pending gates pass.

## 5. Open question logged, not guessed

Timezone of EAGLE-I `run_start_time` (hence of panel `ts`). Evidence so far: the
dataset landing page gives resolution (15 min) but no timezone; the driver
builder aligns panel `ts` directly to ERA5 `valid_time` (UTC), so the pipeline
already *assumes* UTC. The metadata record is being fetched; until confirmed the
clock is labelled `utc_hour` with this assumption stated in the definition string.
