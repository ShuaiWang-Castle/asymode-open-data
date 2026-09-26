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
| B4′ phase 1 (4,320 fits; γ from 0 to .19, ρ ∈ {0, 1}, n ∈ {64, 256, 1024}, 60 reps, 5 initialisations) | Do distribution features plus architecture error envelopes (floor α + slope β × ideal variance, calibrated on γ = 0 only) predict the NET vs ASYM ranking? | **Descriptive only (K6 triggered).** All 30 feedback cells favour NET and all 24 resolved cells are NET wins, so always-NET also scores 24/24 (classical criterion 22/24). A post-hoc bias baseline −(S+F) has lower MAE (5.8e−6) than the envelope (6.5e−6); classical 25.9e−6. The fixed-3000 ablation has both winners (18 NET, 2 ASYM), and there the envelope's MAE (21.0e−6) is worse than the classical criterion's (15.2e−6). Per-cell e_N^Q/S is 0.5%–97.5% |
| B4′ phase 2 (2,160 fits) | Is restricting state access needed for the two-flow model's low shared-component error? | **No, it is not necessary.** Intercepts: ASYM_STATE 4.70e−6, ASYM 6.23e−6, NET 7.42e−6. Stratified bootstrap 95% intervals, conditional on five initialisations: ASYM_STATE − ASYM [−2.58, −0.47]e−6; ASYM_STATE − NET [−3.86, −1.55]e−6; NET − ASYM [−0.05, +2.45]e−6. The intercepts are extrapolations, not measured floors, and the mechanism is not isolated. In strong-signal low-noise cells, ASYM_STATE's e_Q/S is 0.4–4.3% |
| B5 generality (5,760 fits; 3 combinations × 4 feedback levels × 2 ρ × 2 n × 40 reps × 3 models) | Do these findings hold for a second law (recovery mobilisation, F = 0) and a smaller network (8,192 parameters)? | **Pre-registered hypotheses hold except H3 (no-feedback part) in B32768; K7 and K8 not triggered.** In A8192 and B32768 every resolved feedback cell is a NET win (10/10, 8/8), so always-NET is equally accurate. Only B8192 has both winners: ASYM wins at weak mobilisation (η = .2, ρ = 0; +4.31 ± 1.54 and +4.61 ± 1.78 e−6); the envelope gets 10/10, always-NET 8/10, the classical criterion 5/10. Post hoc, the envelope's MAE (4.6, 1.8, 3.6 e−6) is below the best bias-only baseline (6.7, 3.1, 4.9) in all three combinations, unlike B4′. NET's excluded-direction slope β_Q is 0.016, 0.021, 0.013 |

Fitted no-feedback intercepts and slopes (B4′, original early stop; x = the ideal estimator's variance in the component; descriptive):

| component | intercept α | slope β |
|---|---|---|
| NET, shared (affine-in-initial-state) part | 7.42e−6 | 0.899 |
| ASYM, shared part | 6.23e−6 | 1.017 |
| NET, excluded (non-affine) part | 0.15e−6 | 0.041 (bootstrap 95% 0.026–0.060) |

B5 repeats the same pattern for law A at 8,192 parameters and law B (mobilisation) at 32,768 and 8,192 parameters: the
two-flow shared intercept is lower (NET 8.51, 6.99, 7.77; ASYM 5.68, 4.70, 6.00; ASYM_STATE 4.46, 5.13, 4.44, in 1e−6),
its shared noise slope is steeper (NET 0.79, 0.78, 0.71; ASYM 1.16, 0.98, 0.92), and NET barely responds to noise in the
excluded directions (β_Q 0.016, 0.021, 0.013). The B5 intercepts are extrapolations from four no-feedback cells, without
intervals.

The robust finding is the last row. For this law and training protocol, the trained net-flow model's error responds to
data noise in the excluded directions with about 4% of the ideal saturated estimator's slope. The ideal estimator's
variance penalty (the classical restricted-versus-unrestricted criterion) therefore does not transfer to the trained
network. Whether the envelope adds predictive value beyond simple bias baselines is not settled: it does not in B4′
(`results/posthoc_baselines.json`) and does in all three B5 combinations (`results/b5_posthoc.json`, post hoc).

The ideal (classical) criterion does not transfer to these trained networks. In B5 its sign accuracy on resolved cells
(6/10, 6/8, 5/10) is below always-NET. In high-noise cells it favours the restricted model where NET wins, for example
law A at 8,192 parameters with no feedback, ρ = 1, n = 64: ideal difference +47.6e−6, observed −13.05 ± 10.09e−6. The two
ASYM wins in B8192 are cells where the ideal criterion favours NET, so they come from the lower shared intercept, not
from a variance saving of the restriction.

## Corrections after independent review (2026-09-26)

An independent cross-model review checked this package (all 41 checksums, byte-identical re-analysis from the packed
predictions) and found several interpretations too strong. They are corrected here, in `refine-logs/CLAIMS_FROM_RESULTS.md`
and in the plan's appended errata; the original verdicts are kept above those sections.

1. The 24/24 accuracy does not show two-sided selection skill. Every resolved cell is a NET win.
2. The earlier statement that NET captures 92–99.5% of S misread a regression slope. Per-cell e_N^Q/S is 0.5%–97.5%.
3. K6 was triggered, so phase 1 is descriptive.
4. Restricting state access is not necessary for the low shared-component error. That the advantage comes mainly from the
   parameterisation is not identified: branches, output map, initialisation and optimisation are confounded.
5. NET is an unbounded increment update, not a bounded net-flow model.
6. Hashes and local timestamps establish version correspondence, not time order.
7. The launchers now keep an externally set `REPO`.

B5 note: after the first B5 launch, the worker partition in `source/b5_experiment.py` was changed so that each worker
takes complete (NET, ASYM, ASYM_STATE) triplets; the first launch was stopped and the run resumed, skipping completed
fits (`logs/b4_run.log` records status 1 for the stopped launch). Each fit's seeds depend only on its task, so the
change affects scheduling, not results. `source/b5_posthoc.py` (completeness check, K8 reasons, baselines) is post hoc.

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
# generality run: data, 5,760 fits, registered analysis, post-hoc additions
python3.11 source/b5_data.py && REPO=/path/to/checkout-at-97e1a5b bash source/run_b5.sh
python3.11 source/b5_analyze.py && python3.11 source/b5_posthoc.py
# or from the packed public fits
python3.11 source/unpack_fits.py b5 && python3.11 source/b5_analyze.py && python3.11 source/b5_posthoc.py
```

Re-analysing the packed B5 fits this way was checked to reproduce `results/b5_analysis.json` and
`results/b5_posthoc.json` byte for byte.

`run_b4.sh` defaults `REPO` to `$HOME/asymode-open-data-work` and assumes it is at revision 97e1a5b; set `REPO` if
yours differs. The experiment scripts check that the checkout's revision is exactly 97e1a5b.

## Layout

| path | contents |
|---|---|
| `refine-logs/` | frozen plan with revisions, tracker, claim verdicts, plan provenance |
| `source/` | exact computations, data generators, training, analyses, packing |
| `inputs/neural_bridge/` | exact moments and summary of the earlier 1,080-fit bridge (inputs to B1 and X2) |
| `results/` | analysis outputs; `b4_predictions.npz` and `b5_predictions.npz` (every fit's early-stop and final predictions); `b4_fit_records.jsonl.gz` and `b5_fit_records.jsonl.gz` (per-fit configuration, stopping and validation curves); exact moments of both laws (`b4_data/design.npz`, `b5_data_mobilisation/design.npz`); data checksums |
| `logs/` | analysis logs and run times |

`refine-logs/PLAN_PROVENANCE.json` records the SHA-256 and modification times of the unredacted local plan. These confirm
version correspondence; the order of revisions rests on local records. The public plan redacts two references to another manuscript under double-blind review. The only public script changes are the
input paths of `b1_exact_anatomy.py` and `x2_training_envelopes.py`.
