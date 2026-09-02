# Evidence summary for the framing decision

Every line points into `RESULTS_LEDGER.md`; nothing here is new. Grades:
[A] provable/verifiable · [B] full protocol (county-held-out folds, ≥3 seeds, sign gate) · [C] preliminary, not for the paper · [B-synth] synthetic, known ground truth.

## What is established

| # | finding | grade | supports | undercuts |
|---|---|---|---|---|
| 1 | Onset from exactly zero is the dominant regime: 74% of interrupted counties on convective days, 80% on other families; 22 storm days agree; denominator-free (EXP03, EXP03b) | [A] | the structural case against `y·(1−y)` inflow | — |
| 2 | Identifiability: the interruption/restoration split is identified only through state variation under comparable drivers; determinant `(y₁−y₂)` (EXP01 argument; ridge confirmed on synthetic data, err-corr positive in 18/18 runs) | [A] + [B-synth] | a methods contribution no comparable paper has | — |
| 3 | On synthetic data the seeded epidemic form loses onset by 23x and reaches parity only by inflating its seed (EXP02) | [B-synth] | onset argument | — |
| 4 | On public data, seeded epidemic reaches parity by degenerating: ε dominates the inflow on ~77% of scored cells (EXP05, all families in EXP06 H3) | [B] | onset argument | the raw-RMSE gap to the seeded arm is only 1–2% |
| 5 | The two-rate model beats persistence, damped persistence, and pure epidemic at h+24/48 by 7–9%, 15/15 folds; **no advantage at h+1/h+6** (EXP05) | [B] | long-horizon claim vs statistical baselines | any short-horizon claim |
| 6 | **Gradient boosting on identical information beats the two-rate model at h+6/24/48 (7–8%, 0/15 folds at long horizons); the two-rate model wins h+1 by 21%** (EXP07, convergence-checked) | [B] | h+1 claim | "more accurate per point" at h ≥ 6 |
| 7 | Releasing the single-rollout constraint closes the h+6 gap entirely and none of the h+24/48 gap: the long-horizon loss is structural (EXP10, post-hoc control, labelled) | [B] | honest boundary of the model | "we lose because we produce trajectories" |
| 8 | Structural ladder: state scaling is worth 4–6% at every horizon; concurrency (two rates vs one signed rate, parameter-matched) is worth +2.0% at h+48 only and costs 1.3–1.7% at h+1/6 (EXP05-g2) | [B] h+48 / [C] h+24 | concurrency has a measurable, small, long-horizon value | any claim that concurrency is a large effect |
| 9 | **H-E: the two-rate advantage over the parameter-matched single rate orders the families by phase separation and reverses on winter** — tropical −3.5% (h+48, converged), winter +4.3% (0/5); negative control behaves as one (EXP06 + convergence probe) | [B] h+48 / [C] h+24 | the mechanism claim; the paper's strongest result | — |
| 10 | Within convective, the advantage does **not** track the county-event's own phase ratio (D-5) | [B] null | — | any claim that the mechanism resolves at county-event grain; H-E is family-level |
| 11 | Per-horizon direct regressors (trees and linear alike) carry 2.6x more excess reversals than the two-rate rollout (S1 0.61 vs 0.24, county-block CI above zero, 3 seeds); the rollout itself is not coherent (0.24 vs 0.0 for monotone baselines) (D-4) | [B] | "carries fewer reversals than a per-point regressor of equal accuracy" | "produces a coherent trajectory" |
| 12 | Ranking is intrinsically unpredictable at h+24/48 on convective/winter (ceiling 0.29–0.34) and less so on tropical (0.59); survives severity matching only at the tropical-vs-rest level; onset share does not order families (D-2 + control) | [A] | per-horizon reporting; the tropical "ranking win" possibility | "three independent measurements agree" (withdrawn) |
| 13 | Trees win on per-county magnitude, not ordering: their per-origin Spearman is lower than the two-rate model's; level term is ≤ 0.04 of MSE for every arm at long horizons (D-3) | [A] | mechanistic account of the trees' win | "level win/loss" wording (corrected) |
| 14 | Shrinkage is MSE-optimal: oracle λ* < 1 for the dynamics, > 1 for the trees, headroom < 1% either way (D-1) | [A] | closes the peak-weighting family | — |
| 15 | Capacity is doing work: both rates as GLMs lose 3.7–4.8%, 0/15 (EXP08) | [B] | model is not over-parameterised | — |
| 16 | Capacity asymmetry (H-D) dead on both sample sets; input asymmetry (H-A3) [C] — signal on both sides, one side below the fold bar; decisive g1-vs-g2 rerun **pending** | dead / [C] | — | the "three asymmetries" story as originally framed |

## What the three framing paths now rest on

* **Narrow to short horizon.** Rests on #6 (h+1 −21%, 15/15) and #1–#4 (onset). Solid. Leaves #9 as a separate mechanism section.
* **Change the headline metric.** No alternative metric was ever registered (ledger, EXP07); D-1 shows rescaling cannot help; D-4 gives a *registered, measured* trajectory property (#11) that could be reported alongside RMSE but was fixed after EXP07 landed and must be presented as such.
* **Accept the negative on per-point accuracy and lead with mechanism.** Rests on #9 (family-level, converged), #7 (structural boundary), #10 (honest limit), #12–#14 (why the trees win). This is the path where every load-bearing number is [B] or [A] and none was chosen after the fact.

## Not yet in evidence

* Decisive H-A3 rerun (g1 panels × 14 channels) — running; decides whether the g1/g2 ambient discrepancy is the panel set or the wind components.
* Nothing else registered remains unrun.
