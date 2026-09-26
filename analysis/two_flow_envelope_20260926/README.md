# When does the two-flow structure help trained multi-step forecasters? (2026-09-26)

This round asks which features of the conditional response distribution decide whether a two-flow (source-pool)
recursion forecasts better than a single net-flow recursion:

```
two-flow:  y' = y + u(c, x)(1 - y) - r(c, x) y,   u, r >= 0, u + r <= 1   (ASYM; rates do not read the state)
net-flow:  y' = y + f(c, x, y)                                               (NET)
```

The round also includes a two-flow model whose rates may read the recursive state (ASYM_STATE).

All data are synthetic. The finite-population outage process (M = 16 response groups, L = 32 blocks per event) comes
with exact conditional means and covariances, so every error is measured against the true conditional mean. The
neural models are the unchanged public NET and ASYM classes from
`analysis/net_asym_affected_cv_20260914/code/models_v2.py` at revision 97e1a5b. The training protocol is AdamW, full
path loss, early stopping on independent validation panels, and a fixed 3,000-update ablation.

Each batch followed a plan frozen before it ran (`refine-logs/EXPERIMENT_PLAN.md`, revisions 0–3). Kill criteria are
reported as triggered or passed. Verdicts in `refine-logs/CLAIMS_FROM_RESULTS.md` are the executor's self-assessment;
an independent cross-model review is pending.

## Batches and outcomes

| batch | question | outcome |
|---|---|---|
| B1: exact anatomy (original bridge law) | Is the two-flow model's feasibility gap F = dist(Pm, A)²/d material next to the non-affine signal S = ‖Qm‖²/d? | **No.** F/S = 0.014 at γ = .04 (kill criterion K1). The identity min‖m − g‖² = ‖Qm‖² + dist(Pm, A)² holds to 3e−11 relative error. The positivity constraint u ≥ 0 is active in every no-forcing hour |
| B1 / K3 check | Do activated constraints explain the neural shared-component variance gap? | **No.** 0.014e−6 predicted against 1.365e−6 observed (K3) |
| X1 (exploratory) | How does F scale with feedback strength and outage level? | F/S reaches 0.10–0.33 only with strong crowding feedback (γ ≈ .19) at high outage levels |
| X2 (exploratory) | Can the original bridge adjudicate neural theories? | **No.** Only 2 of 18 cells resolve the sign of NET − ASYM; neural training floors are about 10× S |
| B4′ phase 1 (4,320 fits; γ from 0 to .19, ρ ∈ {0, 1}, n ∈ {64, 256, 1024}, 60 reps, 5 initialisations) | Do distribution features plus architecture error envelopes (floor α + slope β × ideal variance, calibrated on γ = 0 only) predict the NET vs ASYM ranking? | K4 and K5 pass: 24/30 feedback cells resolve. The envelope gets 24/24 signs right, the classical ideal-estimator criterion 22/24. Held-out MAE is 6.6e−6 against 25.9e−6. K6 is literally triggered: NET misses 0.5–8% of S |
| B4′ phase 2 (2,160 fits) | Does the two-flow model's lower training floor come from restricting state access or from its parameterisation? | **Parameterisation.** Shared-component floor α: ASYM_STATE 4.70e−6, ASYM 6.23e−6, NET 7.42e−6. ASYM_STATE misses only 0.4–4.3% of S in strong-signal cells |
| B5 generality (5,760 fits) | Do these findings hold for a second law (recovery mobilisation, F = 0) and a smaller network (8,192 parameters)? | Running; results will be added |

Fitted envelopes (B4′, original early stop; x = the ideal estimator's variance in the component):

| component | floor α | slope β |
|---|---|---|
| NET, shared (affine-in-initial-state) part | 7.42e−6 | 0.899 |
| ASYM, shared part | 6.23e−6 | 1.017 |
| NET, excluded (non-affine) part | 0.15e−6 | 0.041 |

For this law, the trained net-flow model pays about 4% of the ideal estimator's variance in the directions the
two-flow restriction excludes. The noise-driven advantage of the restriction that the classical criterion predicts
therefore largely disappears. The two-flow model wins when the excluded signal lies below the difference in
shared-component training floors.

## Reproduce

Data generators are deterministic; `results/b4_data_checksums.json` lists the SHA-256 of the generated files. The exact
moments of the B4′ law (`results/b4_data/design.npz`) are included, so the packed fits can be analysed without regenerating
the panels; reproducing the analyses this way was checked to give identical numbers.

```bash
# from this directory; REPO must be a checkout of this repository at revision 97e1a5b (for models_v2.py)
python3.11 source/b1_exact_anatomy.py
python3.11 source/b4_data.py
REPO=/path/to/checkout-at-97e1a5b WORKERS=6 bash source/run_b4.sh     # 4,320 fits
bash source/run_b4s.sh                                                # 2,160 ASYM_STATE fits
python3.11 source/b4_analyze.py && python3.11 source/b4s_analyze.py
# or analyse the packed public fits without training:
python3.11 source/unpack_fits.py b4 && python3.11 source/b4_analyze.py && python3.11 source/b4s_analyze.py
```

`run_b4.sh` defaults `REPO` to `$HOME/asymode-open-data-work` and assumes it is at revision 97e1a5b; set `REPO` if
yours differs. The experiment scripts check that the checkout's revision is exactly 97e1a5b.

## Layout

| path | contents |
|---|---|
| `refine-logs/` | frozen plan with revisions, tracker, claim verdicts, plan provenance |
| `source/` | exact computations, data generators, training, analyses, packing |
| `inputs/neural_bridge/` | exact moments and summary of the earlier 1,080-fit bridge (inputs to B1 and X2) |
| `results/` | analysis outputs; `b4_predictions.npz` (every fit's early-stop and final predictions); `b4_fit_records.jsonl.gz` (per-fit configuration, stopping and validation curves); data checksums |
| `logs/` | analysis logs and run times |

`refine-logs/PLAN_PROVENANCE.json` records the SHA-256 and modification times of the unredacted local plan. The public
plan redacts two references to another manuscript under double-blind review. The only public script changes are the
input paths of `b1_exact_anatomy.py` and `x2_training_envelopes.py`.
