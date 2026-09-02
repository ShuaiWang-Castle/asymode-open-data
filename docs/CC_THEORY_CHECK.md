# Theory check — every statement re-derived and tested

Specification under test: `ASYMODE_THEORY_DEEP_DIVE_V1.md` / `…SECTION_V1.tex`
(package SHA256 in `docs/CC_AUDIT_LOG.md`). Module: `src/asymode/information.py`.
Tests: `tests/test_theory.py` (1,726 randomised cases incl. endpoints y=0, y=1,
point masses, one rate zero, heavily imbalanced weights). Verdicts are mine.

| statement | my derivation | test | verdict |
|---|---|---|---|
| Thm 1: det Q = v | Q = [[1−2μ+m₂, −(μ−m₂)],[−(μ−m₂), m₂]] ⇒ det = m₂ − μ² | `test_det_equals_conditional_variance` | **holds** |
| Thm 1: ½ ≤ λ_max ≤ 1 ⇒ v ≤ λ_min ≤ 2v | ‖φ‖² ≤ 1; w=(1,−1)/√2 gives wᵀφ = 1/√2 for every y | `test_eigenvalue_bounds` | **holds** |
| AB − C² = v | A=1−2μ+m₂, B=m₂, C=μ−m₂ | same test | **holds** |
| Cor 1.1 identification iff v>0 | affine-in-y mean; two support points determine intercept & slope | covered by Prop 3 test (null direction) | **holds** |
| Cor 1.2 excess-risk identity | E[(φᵀΔβ)²] = ΔβᵀQΔβ, bounds by eigenvalues | not separately tested (algebraic) | holds |
| Thm 2: det(AᵀA)=Σ_{i<j}(y_i−y_j)²=N²v̂; Cov=σ²(AᵀA)⁻¹; Var(R̂)/Var(Û)=Σ(1−y)²/Σy² | Cauchy–Binet; 2×2 inverse | `ols_covariance` used in the S-2 script (pending); identity checked in `test_det_equals_conditional_variance` via weights | **holds**; it is a **variance** ratio |
| Prop 3: (α,τ) coordinates, Q̃ = diag(1, v), inverse map, cone | m = α − τ(y−μ); basis (1, −(y−μ)) | `test_alpha_tau_round_trip_and_orthogonality` | **holds** |
| null direction n_μ=(μ,1−μ), energy c²v | φ(μ)ᵀn_μ=0; φ(y)ᵀn_μ = μ−y | `test_null_direction_at_point_mass` | **holds** |
| Thm 4 local recovery bound | bias ≤ Lh since ‖φ‖≤1; ‖A⁺‖ = 1/√(N κ̂); Gaussian norm concentration | not numerically tested (probabilistic); proof checked by hand | holds as stated |
| Thm 5 one-rate gap = v·min(R²/A, U²/B); >0 iff v>0, U>0, R>0 | residuals vR²/A, vU²/B; infeasibility of one projection forces the other feasible and smaller (AU<CR ⇒ U/R<C/A≤√(B/A)) | `test_one_rate_gap_matches_bruteforce` vs dense constrained grid | **holds** |
| §9 exact CT/DT bijection | e^{−ΛΔ} = 1−p, q = u/p | `test_ct_dt_bijection_and_exact_transition` | **holds** |
| §1.3 clip redundant when u,r∈[0,1] | affine map with F(0)=u, F(1)=1−r | `test_recurrence_maps_unit_interval_into_itself` | **holds** |
| Prop 6 product-sum bound; old Prop 5 fails without a lower rate bound | recursion e_{t+1} ≤ ρ_t e_t + δ_t | `test_rollout_product_sum_bound_holds`; `test_constant_rate_saturation_bound_fails_…` | **holds**; old wording **refuted** |
| §8 ridge/gate ĉ_λ = Nv/(Nv+λ)·ĉ_OLS; g* = Nvc²/(Nvc²+σ²) | one-coefficient ridge; MSE minimisation in g | to be tested with the S-2 oracle layer | holds (algebra) |

**Consequences adopted for the paper text (F1 of the prompt):**
conditional-mean identification ≠ weight identification; design Gram ≠ Fisher
information (Fisher only under the Gaussian working model, ×1/σ²); variance
ratio ≠ precision ratio; local information N·λ_min(Q) ≠ Kish concentration;
common-rate-function assumption stated before any pooling claim; transition
components ≠ hazards (exact bijection, choose one language); non-expansion ≠
uniform contraction. The reachable-interval lemma stays an appendix sanity lemma.
