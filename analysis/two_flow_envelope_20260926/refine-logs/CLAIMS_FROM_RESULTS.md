# CLAIMS_FROM_RESULTS (ARIS result-to-claim format; executor self-assessment; independent reviewer pending)

## C1 — The feasibility gap F is material, so a criterion that uses only S systematically underestimates the two-flow bias

- claim_supported: **no** (on the original bridge design)
- what_results_support: F > 0 at γ = .04. The implied affine inflow is negative during recovery, down to −0.0022 (this reproduces GPT's finding). The identity min‖m − g‖² = ‖Qm‖² + dist(Pm, 𝒜)² holds to a relative 3e−11.
- what_results_dont_support: F/S = 0.014 at γ = .04, below the K1 threshold of 0.05. F is numerically negligible in this design.
- missing_evidence: none for this design. Exploratory run X1 shows F/S reaching 0.10–0.33 only with strong feedback (γ ≥ .16) and high outage levels.
- suggested_claim_revision: "F becomes material only under strong state feedback at high outage levels. Otherwise the affine envelope is an accurate proxy for the two-flow class."
- next_experiments_needed: fold the γ and level dependence of F into the powered design B4'.
- confidence: high (exact, deterministic)
- integrity_status: unavailable (no cross-model reviewer on this machine)

## C2 — Activated constraints explain the neural common-component variance gap

- claim_supported: **no**
- what_results_support: the positivity constraint is active in every no-forcing hour. Among ideal estimators it saves up to about 7% of the Q-direction variance in the highest-noise cells.
- what_results_dont_support: at γ = .04, ρ = 0, n = 512 the predicted saving is 0.014e−6 against an observed neural gap of 1.365e−6 (**K3 triggered**). The neural P-component variance is 7–15× the ideal estimator's variance at low noise, so it is dominated by training (initialisation and optimisation), not by data noise.
- missing_evidence: none needed to reject.
- suggested_claim_revision: drop the claim. The neural gap must be modelled as a training-error envelope, not as constraint activation.
- confidence: high
- integrity_status: unavailable

## Side finding (exploratory, X2) — the bridge cannot adjudicate neural theories

Only 2 of 18 settings resolve the sign of NET − ASYM beyond Monte Carlo half-widths. The structural signal S = 0.5e−6 lies below both the neural error floors (about 5–7e−6) and the pairwise noise. Any theory of neural structure choice needs a design in which the structural signal spans the training floor, with more replications and initialisations.

## C3′ — Distribution features plus architecture envelopes predict the neural two-flow vs net-flow ranking (B4′ phase 1, pre-registered in revision 1)

- claim_supported: **partial**
- what_results_support:
  - K4 passed: 24 of 30 held-out cells (γ > 0) resolve the sign of Δ.
  - K5 passed: the envelope criterion gets 24/24 signs right on the resolved cells; the ideal-estimator (classical) criterion gets 22/24.
  - The two cells where the criteria disagree (γ = .08 and .12, ρ = 1, n = 64) both go the envelope's way. Observed Δ = −20.9 ± 12.0 and −51.2 ± 14.6 (×1e−6); the classical criterion predicted +40.6 and +9.3.
  - Magnitudes, held-out cells: MAE 6.6e−6 for the envelope against 25.9e−6 for the classical criterion; median relative error 8.0% against 11.9%.
- key fitted envelope (γ = 0 cells only, original early stop):

  | component | α | β |
  |---|---|---|
  | NET, shared (P) | 7.42e−6 | 0.899 |
  | ASYM, shared (P) | 6.23e−6 | 1.017 |
  | NET, excluded (Q) | 0.15e−6 | 0.041 |

  The net-flow network therefore pays only about 4% of the ideal estimator's variance in the excluded directions. The noise-driven preference for the restricted model that the classical criterion predicts largely disappears. What remains is the shared-component training-floor difference, about 1.2e−6.
- what_results_dont_support:
  - K6 is literally triggered: NET's excluded-component error grows with S in all six (ρ, n) groups (p < .003). The slopes are small, 0.0055–0.08, so NET still captures 92–99.5% of S. Additivity is therefore approximate, not exact.
  - Under the fixed-3000 ablation the envelope's sign accuracy is 20/20 against 19/20, but its magnitude MAE is worse (21.0e−6 against 15.2e−6).
  - One synthetic law, one architecture pair, and a calibration on six cells.
- missing_evidence:
  - a second law;
  - other widths and depths;
  - transfer to real storms;
  - a pre-registered κ·S correction for the K6 slope.
- suggested_claim_revision: "In the tested law, the two-flow restriction wins only when the excluded signal lies below the shared-component training-floor difference. Additional event noise does not restore the restricted model's advantage, because the net-flow network's excluded-direction variance is almost entirely suppressed."
- next_experiments_needed:
  - B4′-S (running): does state access or the two-flow parameterisation set the floor?
  - A second law.
  - Real-data envelope transfer.
- confidence: medium
- integrity_status: unavailable (awaiting the cross-model reviewer)

## C5 — The lower shared-component training floor of the two-flow model comes from its parameterisation, not from restricting state access (B4′-S, pre-registered in revision 2)

- claim_supported: **yes** (law A, 32,768 parameters)
- what_results_support: decision 1. The state-reading two-flow model (ASYM_STATE) has a shared-component floor α = 4.70e−6. This is below the pre-registered midpoint of 6.83e−6, and below both ASYM (6.23e−6) and NET (7.42e−6).
- what_results_dont_support: one law and one size only. Revision 3 is testing generality.
- confidence: medium

## C6 — With state-reading rates, the two-flow model uses the excluded signal (B4′-S, decision 2)

- claim_supported: **yes**
- what_results_support: in the strong-signal cells (γ ≥ .08, ρ = 0), ASYM_STATE's excluded-component error is 0.4–4.3% of S.
- observed consequences (reported, not pre-registered decisions):
  - ASYM_STATE beats ASYM in every cell with S > 0 and ρ = 0. At γ = .04 the differences are −3.3 to −3.8e−6 with half-widths of 0.5–1.0e−6. At γ = 0 the two models tie or ASYM_STATE is slightly better.
  - ASYM_STATE is no worse than NET in all low-noise cells and better in most.
  - NET beats ASYM_STATE in several high-noise (ρ = 1) cells.
- suggested_claim_revision (pending revision 3): "The two-flow source-pool form helps trained networks through a lower training floor. The classical restriction on state access adds little and costs the excluded signal. The flexible net-flow model gains with event noise."
- confidence: medium
