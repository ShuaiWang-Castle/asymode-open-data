# Input-gated versus state-feedback recurrences for bounded occupancies: proposal and evidence for review (2026-09-29)

Start with `PROPOSAL_ZH.md`, a summary in Chinese of the current proposal, the plan, the evidence, the corrections made
so far, the open risks, and a checklist for the reviewer.

## Question

A bounded occupancy, such as the share of customers without power during a storm, can be forecast with an
input-gated recurrence or with a state-feedback recurrence:

- input-gated: flow-decomposed, two-flow with exogenous rates, or a linear compartment model;
- state-feedback: net-flow, or a recurrent network or ODE whose gates read the state.

For an ideal learner the two differ by one statistical restriction: the update may not read the current state.

The general theory in `docs/THEORY_GENERAL.md` has three parts:
- when that restriction lowers multi-step risk, in terms of trajectory-distribution features;
- which noise pays for flexibility;
- a dilute-limit law for density-dependent population processes, Λ ∝ n·M·γ²·ℓ³.

The law is checked exactly on two different processes. Tropical-cyclone power outages are the real-data instance. They
come from 15 Atlantic systems, 2018–2024, with 1,633 county-events.

## Status

- There is no venue target.
- Experiments are paused at the request of the principal investigator.
- The tropical v2 protocol (nested storm-level validation and an R1/R2 decomposition) was frozen and started, then
  stopped with 18 of 180 inner fits done and no final fits. Those partial fits are not included here.

## Layout

| path | contents |
|---|---|
| `PROPOSAL_ZH.md` | the summary for review (Chinese) |
| `docs/THEORY_GENERAL.md` | general theory G0–G6, novelty table, experiment layers, exact checks |
| `docs/THEORY_T3.md` | detailed derivations for the outage example (first-order closed forms, feasibility sign rule, corrected Lemma 1) |
| `docs/PAPER_PLAN_TROPICAL.md` | paper plan after an independent review. It is kept for context; `PROPOSAL_ZH.md` supersedes it where they differ |
| `source/`, `results/`, `logs/` | exact-moment checks: `t3_check.py` (closed forms), `t2_boundary.py` (boundary dividend), `t_general_check*.py` (dilute-limit exponents on two processes and the gate-feasibility dichotomy); `dualflow.py` is the constrained two-flow fitter |
| `tropical/` | tropical-cyclone runs (details below) |

Contents of `tropical/`:
- the v1 plan, frozen before the fits (`PLAN_TROPICAL.md`, hashes in `PLAN_PROVENANCE.json`);
- the v2 plan (`PLAN_TROPICAL_V2.md`, `PLAN_V2_PROVENANCE.json`);
- the scripts;
- per-unit results for the registered 3,000-update budget (`results/`) and the post-hoc 6,000-update budget (`long/results/`);
- the retrospective Isaias figures (`figures/`);
- the data audit and the review checks (`logs/`).

B4′ and B5, the synthetic neural training grids, are in `analysis/two_flow_envelope_20260926/` on the same branch.

## Data and reproduction

- **Tropical scripts.**
  - They read the geo-weather development panel `features_v1D.npz` and `splits_v1D.json`, which are not included here.
  - They are run as `python3.11 tropical_run.py --panel <npz> --splits <json> --worker 0 --workers 1`, then `tropical_analyze.py <npz> <json>`.
  - The panel is built from public EAGLE-I county outage data and ERA5 weather.
  - The sealed confirmation tranche of that design was not used.
  - 1,367 of the 1,633 units belong to systems whose outcomes were read earlier in this repository, so the tropical results are development evidence.
- **Exact checks.**
  - They need no data (`python3.11 source/t_general_check_exact.py`, and so on).
  - The last step of `t3_check.py` also reads `results/b4_analysis.json` from `analysis/two_flow_envelope_20260926/`.
- **Frozen scripts.** They are byte-identical to the versions whose hashes appear in the provenance files. `make_example_figure.py` is post hoc; it now takes the panel paths as arguments.

`SHA256SUMS.txt` lists every file in this directory.
