# Experiment matrix — staged, with exact commands

Paths under `results/v2/` are the corrected protocol; `results/` (legacy) is never
overwritten. Every command is run from the repository root with
`./.venv/bin/python`. Compute: Apple M2, CPU only.

| id | what | command | status |
|---|---|---|---|
| G0 | gates | `./.venv/bin/python -m pytest -q tests` | pass (1,738) |
| G1 | smokes (non-archival) | `experiments/exp0{5,6,7,8,10}… --panels <2 panels> --k 2 --model-seeds 0 --epochs 1 --split-unit event --out results/smoke/…` | pass ×5 |
| D6-g2 | local information geometry, primary set | `experiments/d6_information_geometry.py --panels configs/panel_manifest_g2-convective-11.json --out results/d6_information_geometry_g2.json` | done |
| D6-g3 | same, all families | `… --panels configs/panel_manifest_g3-all-26.json --out results/d6_information_geometry_g3.json` | done |
| S2 | CRLB tracking, unconfounded synthetic sweep + negative control | `experiments/s2_crlb_tracking.py --out results/s2_crlb_tracking.json` | to run |
| V2-05a | stage 3: exp05 on g2, **event-held-out**, one model seed | `experiments/exp05_real_dynamics.py --panels configs/panel_manifest_g2-convective-11.json --split-unit event --model-seeds 0 --epochs 60 --patience 12 --save-oof --out results/v2/exp05_g2_event_seed0.json` | running |
| V2-05 | stage 5: same, three model seeds | `… --model-seeds 0 1 2 --out results/v2/exp05_g2_event.json` | after V2-05a inspection |
| V2-07 | trees, same split, cap 2,000 | `experiments/exp07_learned_baselines.py --panels configs/panel_manifest_g2-convective-11.json --split-unit event --model-seeds 0 1 2 --rounds 2000 --arms trees_matched --save-oof --out results/v2/exp07_g2_event.json` | after V2-05 |
| V2-05c | secondary protocol: county-held-out, fixed map | `… --split-unit county --model-seeds 0 1 2 --save-oof --out results/v2/exp05_g2_county.json` | after V2-07 |
| V2-06 | families on g3, event-held-out within family (k capped) | `experiments/exp06_by_family.py --panels configs/panel_manifest_g3-all-26.json --split-unit event --model-seeds 0 1 2 --epochs 60 --patience 12 --arms susceptible net_scaled --out results/v2/exp06_g3_event.json` | after V2-05 |
| E3 | event-level reviews | `scripts/event_level_review.py results/v2/oof_<A>.npz results/v2/oof_<B>.npz --json …` | after each export |
| D0 | transition semantics decision + numerical equivalence | test 7 in `tests/test_theory.py`; paper language: discrete-time two-rate transition model | done (tests) |
| D1 | auxiliary teacher-forced one-step loss, λ_tf ∈ {0, 0.1, 1} | flag to add to exp05; run after V2-05 | not started |
| D2 | information-gated concurrency (α, τ) model + controls | after the theory tests and D1; synthetic first (S2 oracle gate) | not started |
| D3 | restoration memory (minimal) | deferred until D0–D2 adjudicated | deferred |
| E5 | missingness / denominator / onset-threshold sensitivity | `scripts/build_panel.py` variants + onset audit reruns | not started |

Rough cost on this machine: exp05 six neural arms × 5 folds ≈ 50 min per model
seed at 60 epochs; exp07 trees ≈ 20–40 min per seed at 2,000 rounds; exp06 on
g3 ≈ 3–4 h for three seeds.
