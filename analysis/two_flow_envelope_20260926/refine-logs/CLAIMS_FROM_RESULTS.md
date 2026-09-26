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


---

## 独立审查后的更正（2026-09-26；以上原判定保留，按本节更正）

- **C3′ 降级为“描述性”。** K6 已触发，按计划应降级，“字面触发但斜率小”不能追认通过。
  - 早停主实验中，30 个有反馈格点全部是 NET 胜；24 个可分辨格点也全是 NET 胜，“始终选 NET”同样 24/24。
  - 事后基线 −(S+F) 的 MAE 为 5.81e−6，低于包络的 6.55e−6。
  - 固定 3000 步消融出现两种胜者（NET 18，ASYM 2），但包络 MAE 21.0e−6 差于经典式 15.2e−6。
  - 可以保留的描述性结论：NET 在被排除方向上的数据噪声响应 β_Q ≈ 0.041，分层 bootstrap 95% 区间 [0.026, 0.060]，远小于理想饱和估计量的 1。因此理想估计量的方差惩罚不能直接移植到这个训练程序下的网络。
- **更正数字。** “NET 吃下 92–99.5% 的信号”是误读：0.0055–0.08 是跨 γ 的回归斜率。逐格 e_N^Q/S 为 0.52%–97.5%。
- **C5 改写。** 限制速率读状态，不是获得较低共同部分误差的必要条件。
  - 截距：ASYM_STATE 4.70e−6，ASYM 6.23e−6，NET 7.42e−6。
  - 分层 bootstrap 95% 区间，以五个初始化为条件：ASYM_STATE − ASYM 为 [−2.58, −0.47]e−6，ASYM_STATE − NET 为 [−3.86, −1.55]e−6，NET − ASYM 为 [−0.05, +2.45]e−6（包含 0）。
  - 截距是外推值，不是测得的不可约地板。分支结构、输出映射、初始化和优化的作用都没有分离，所以不能写成“主要来自两流参数化”。
- **标签更正。** NET 是无界增量更新，不是“有界净流”。
- **冻结凭证。** 公开的 hash 与本地修改时间只证明版本对应，不证明时间顺序。


## B5 泛化检验（修订 3 预注册；2026-09-26 完成，按勘误第 5 条的口径判定）

**完整性（事后检查）。** 5,760/5,760 次拟合全部完成，预测均为有限值，每个格点 40 个配对三元组齐全（`source/b5_posthoc.py`）。

| 组合 | H1 | H2 | H3（低噪声有信号 / 无反馈） | H4 | K7 | K8 |
|---|---|---|---|---|---|---|
| 规律 A，8,192 参数 | 成立 | 成立 | 成立 / 成立 | 成立 | 未触发 | 未触发 |
| 规律 B，32,768 参数 | 成立 | 成立 | 成立 / **不成立** | 成立 | 未触发 | 未触发 |
| 规律 B，8,192 参数 | 成立 | 成立 | 成立 / 成立 | 成立 | 未触发 | 未触发 |

**H1 与 H2 的数值。**
- 无反馈标定的共同部分截距 α^P（10⁻⁶）：

  | 组合 | NET | ASYM | ASYM_STATE |
  |---|---|---|---|
  | A8192 | 8.51 | 5.68 | 4.46 |
  | B32768 | 6.99 | 4.70 | 5.13 |
  | B8192 | 7.77 | 6.00 | 4.44 |

- 斜率 β^P：

  | 组合 | NET | ASYM | ASYM_STATE |
  |---|---|---|---|
  | A8192 | 0.79 | 1.16 | 1.12 |
  | B32768 | 0.78 | 0.98 | 0.89 |
  | B8192 | 0.71 | 0.92 | 0.82 |

- 截距是用 4 个无反馈格点外推出来的，B5 没有计算区间。
- 两流参数化的共同部分截距更低、噪声斜率更陡，这一模式在新规律和新规模下重复出现。

**H3。**
- B32768 的无反馈格点（ρ = 1，n = 1024）上，ASYM_STATE 比 ASYM 差 +5.87 ± 3.33（10⁻⁶），所以 H3 的无反馈部分不成立。
- 按勘误第 5 条，“不超过半宽”不是非劣性检验。

**H4 的正确读法。**
- A8192 与 B32768 的可分辨反馈格点全部是 NET 胜（10/10、8/8），“始终 NET”同样全对。这两个组合的 H4 成立，不能证明双向选型能力。
- 只有 B8192 出现两种胜者。ASYM 在 η = .2、ρ = 0 的两个格点胜出：+4.31 ± 1.54 和 +4.61 ± 1.78（10⁻⁶）。
  - 包络：10/10。
  - 始终 NET：8/10。
  - 经典理想判据：5/10。
- 这是第一次有记录的双向正确，但只有两个格点。

**事后基线（未预注册，`results/b5_posthoc.json`）。** 12 个反馈格点上风险差的 MAE（10⁻⁶）：

| 组合 | 包络 | 最好的纯偏差基线 | 经典式 |
|---|---|---|---|
| A8192 | 4.56 | 6.72 | 22.7 |
| B32768 | 1.77 | 3.08 | 12.7 |
| B8192 | 3.56 | 4.94 | 12.2 |

B4′ 的结果相反，−(S+F) 优于包络。所以包络是否有超出简单基线的价值，两批结果不一致。包络的拟合参数也比基线多。

**可以保留的描述性结论（算法层）。**
1. NET 在被排除方向上的噪声斜率 β_Q 为 0.016、0.021、0.013，比 B4′ 的 0.041 更小。“训练出来的 NET 几乎不付被排除方向的方差”在两个规律、两个规模下重复成立。
2. **理想判据的噪声逻辑在训练网络上是反的。**
   - 经典式对可分辨格点的符号准确率为 6/10、6/8、5/10，都低于“始终 NET”。
   - 在高噪声格点上，理想判据偏向受约束模型，训练结果却是 NET 胜。例：A8192 无反馈、ρ = 1、n = 64，理想差 +47.6，实测 −13.05 ± 10.09。
   - 原因是 ASYM 在共同方向上的噪声斜率更陡，NET 在被排除方向上又几乎不付方差。
3. B8192 中 ASYM 的两个胜出格点，理想判据给出的都是 NET（x_Q − S < 0）。所以 ASYM 在那里赢，靠的是更低的共同部分截距，不是约束带来的方差节省。
4. 由第 2 点，只看被排除方向的简化判据“Λ < β_Q”在高噪声格点上已经可以预见会失败。如果要做理想判据向训练网络的迁移检验，判据应当是同时包含共同方向和被排除方向斜率的完整包络，并在新设计上预先登记。
