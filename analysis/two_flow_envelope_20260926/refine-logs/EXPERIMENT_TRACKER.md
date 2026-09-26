# EXPERIMENT_TRACKER

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| B1 | M0 | Exact anatomy: S, F, identity, active set, variance terms | Exact moments, bridge design (K = {0,2,4,8}/16) | exact | S, F, F/S, tr(·Σ) | MUST | DONE 2026-09-26 | **K1 triggered**: F/S = 0.014 at γ = .04. Identity holds (relative gap 3e-11). Positivity u ≥ 0 is active in every no-forcing hour, including at γ = 0 |
| X1 | exploratory | F versus feedback strength and outage level | Exact moments, γ ∈ {0, …, .19}, three initial-state grids | exact | S, F, F/S | — | DONE | F/S grows with γ and level: 0.10–0.33 at γ = .19. Not pre-registered |
| X2 | exploratory | Do neural component errors follow floor-plus-slope envelopes? | Saved 1,080 bridge fits | train design | α, β; sign accuracy | — | DONE | **Only 2 of 18 cells resolve the NET−ASYM sign.** The neural error floor (NET about 7.2e−6, ASYM about 4.7e−6) is roughly 10× S. Not pre-registered |
| B1b | M1 | Constrained LS on the 30 original datasets | Saturated / affine / two-flow LS vs neural | train, interp | risk, P/Q parts | MUST | ON HOLD | Superseded by the K3 check. The bridge is under-powered for neural comparisons |
| K3 check | M1 | Can activated constraints explain the neural common-component variance gap? | B1 face prediction vs neural summary | train | ratio | MUST | DONE | **K3 triggered**: at γ = .04, ρ = 0, n = 512 the prediction is 0.014e−6 against an observed 1.365e−6 (about 1%) |
| B4' | M3 | Powered neural test of the envelope criterion | NET vs ASYM, 36 cells × 60 reps (4,320 fits) | train design | Δ, P/Q parts, envelope | MUST | DONE 2026-09-26 (3.8 h) | K4 and K5 pass (24/24 vs 22/24); K6 literally triggered but small (92–99.5% of S captured) |
| B4'-S | M3 | Phase 2: state-reading two-flow control | ASYM_STATE, 36 cells × 60 reps (2,160 fits) | train design | α_S^P, Q error | MUST | DONE 2026-09-26 (4.4 h) | Floor source is the parameterisation (α_S = 4.70 < midpoint 6.83); ASYM_STATE uses the excluded signal (0.4–4.3% of S missed) |
| B5 | M3 | Generality: law A at 8k parameters; law B (mobilisation) at 32k and 8k | NET / ASYM / ASYM_STATE, 3 combos × 4 feedback × 2 ρ × 2 n × 40 reps (5,760 fits) | train design | H1–H4, K7–K8 | MUST | RUNNING | Frozen in revision 3 |
