# Asymmetric two-rate dynamics for county power-outage trajectories

## 最新地理核结果 2026年10月8日

先展示原始四场，再展示此前县域拓展和固定县域增加风暴的结果。数字沿用归档，不重训、不改值。以下为开发期证据[C]；RMSE越低越好，括号相对同列AsymODE，↓为下降、↑为上升。

完整设置、MAE、种子结果和同样本比较见[完整结果报告](experiments/geo_weather_20260924/notes/GEOGRAPHY_RESULTS_20261008_ZH.md)。

### 原始四场风暴

| 模型 | Michael | Delta | 1月24日冬季风暴 | 1月底冰冻风暴 |
|---|---:|---:|---:|---:|
| 全0预测 | 0.136812（↑49.66%） | 0.145082（↑25.28%） | 0.063833（↑37.53%） | 0.070845（↑6.74%） |
| TimesFM 3.0 | 0.136793（↑49.64%） | 0.144946（↑25.17%） | 0.063816（↑37.49%） | 0.070809（↑6.69%） |
| AsymODE | 0.091417 | 0.115802 | **0.046414** | 0.066372 |
| ＋原地理核 | 0.089976（↓1.58%） | 0.115497（↓0.26%） | 0.047171（↑1.63%） | 0.065455（↓1.38%） |
| ＋新地理核 | **0.085404（↓6.58%）** | **0.113360（↓2.11%）** | 0.047218（↑1.73%） | **0.065400（↓1.46%）** |

### 此前扩大县域

县数由147、113、113、115变为249、138、138、140。样本组成变化，不能直接用前后表RMSE差异说明提升。

| 模型 | Michael | Delta | 1月24日冬季风暴 | 1月底冰冻风暴 |
|---|---:|---:|---:|---:|
| 全0预测 | 0.122814（↑34.81%） | 0.131393（↑27.30%） | 0.057780（↑38.82%） | 0.066917（↑7.52%） |
| TimesFM 3.0 | 0.122801（↑34.80%） | 0.131271（↑27.19%） | 0.057765（↑38.78%） | 0.066886（↑7.47%） |
| AsymODE | 0.091100 | **0.103212** | **0.041622** | 0.062235 |
| ＋原地理核 | 0.094283（↑3.49%） | 0.104306（↑1.06%） | 0.044142（↑6.05%） | 0.061919（↓0.51%） |
| ＋新地理核 | **0.089343（↓1.93%）** | 0.103844（↑0.61%） | 0.042482（↑2.07%） | **0.061895（↓0.55%）** |

### 固定147县增加风暴

前三列分别训练，最后一列联合Michael共四场训练。仍为147个独立县，合计588条县—风暴样本。

| 模型 | Isaias | Zeta | Ophelia | 四场合并 |
|---|---:|---:|---:|---:|
| 全0预测 | 0.053227（↑65.48%） | 0.040506（↑25.09%） | 0.014528（↑3.32%） | 0.076489（↑40.52%） |
| TimesFM 3.0 | 0.053168（↑65.30%） | 0.040483（↑25.02%） | 0.014528（↑3.32%） | 0.076467（↑40.48%） |
| AsymODE | 0.032165 | 0.032381 | 0.014062 | 0.054434 |
| ＋原地理核 | 0.032225（↑0.19%） | 0.030745（↓5.05%） | **0.013951（↓0.79%）** | 0.055320（↑1.63%） |
| ＋新地理核 | **0.031806（↓1.12%）** | **0.030601（↓5.50%）** | 0.014190（↑0.91%） | **0.051931（↓4.60%）** |

合并训练中新核相对AsymODE降低RMSE 4.60%。同样588条记录上，新核分别训练为0.048593，合并训练为0.051931，合并后误差高6.87%；每场分别训练600步，合并模型总共仍为600步，训练预算影响未排除。

A county's outage fraction moves under two processes that have nothing in common:
storms interrupt service, and crews restore it. They respond to different
covariates, on different timescales, with different smoothness. Most sequence
models give the two directions one shared function and let it change sign. This
project asks what is gained by refusing to.

The state is the fraction of a county's tracked customers without power,
`y in [0, 1]`, and it evolves as

    y_{t+1} = clip( y_t + u_t (1 - y_t) - r_t y_t , 0, 1 )
    u_t = cap_u * sigmoid( f_U(x^U_t) )        interruption
    r_t = cap_r * sigmoid( f_R(x^R_t) )        restoration

with `f_U` and `f_R` separately parameterised. Three asymmetries are exposed as
independent switches so each can be ablated on its own:

| axis | what varies | off-switch |
|---|---|---|
| dynamical | inflow on the served pool `(1-y)` vs the epidemic `y(1-y)` | `inflow=` |
| input | the two rates read different driver channels | same `idx_u`/`idx_r` |
| capacity | the two rates get different network widths | same `hidden_u`/`hidden_r` |

**The dynamical axis is the load-bearing one.** An inflow proportional to `y`
is identically zero at `y = 0`: a model in that family cannot start an outage in
a county that does not already have one. Storms do exactly that, constantly. The
experiments are built to measure the cost of that structural gap rather than to
assert it, including against a steelmanned epidemic arm that is given a learnable
seed so it *can* ignite.

Rates are bounded by construction as `cap * sigmoid(logit)`, and every structural
pathway adds to the **logit**, never to the rate. Composing in rate space needs
`clamp(rate + bump, 0, cap)`, and the clamp zeroes the gradient wherever it binds
-- a pathway that drifts negative pins the rate at zero and can never be learned
back on. In logit space the bound is automatic and no gradient is destroyed.

## Layout

    src/asymode/     dynamics, synthetic generator, trainer, event catalog
    experiments/     numbered, each with its hypotheses and kill conditions in the docstring
    scripts/         data acquisition and panel construction
    docs/            DATA_CARD.md, ACCESS_TODO.md, MODEL_NOTES.md
    results/         raw JSON from every run; nothing is quoted without a path here
    RESULTS_LEDGER.md  every number that may enter a paper, with its file and its grade

## Evidence grading

No number is quoted anywhere without an archive path and a grade.

* **[A]** provable or directly verifiable
* **[B]** full protocol passed -- county-held-out folds, >= 3 seeds, sign consistent
* **[C]** preliminary -- internal discussion only, may not enter a paper

## Setup

    python3.11 -m venv .venv && ./.venv/bin/pip install torch numpy pandas scipy scikit-learn matplotlib pyarrow statsmodels
    ./.venv/bin/python scripts/build_event_catalog.py
    ./.venv/bin/python experiments/exp01_identifiability.py
    ./.venv/bin/python experiments/exp02_onset.py

Data is not committed. See `docs/ACCESS_TODO.md` for what still needs an account.

## Which panels a run used

Experiments that fit real observations pool a named set of panels, not whatever
happens to be on disk. The set lives in `data/interim/PANEL_MANIFEST.json` and its
digest is written into every result file, because the panel directory grows as
covariate downloads land and two runs a few hours apart can otherwise disagree
for that reason alone with nothing in either file to say so.

    ./.venv/bin/python scripts/build_panel_manifest.py --generation g1-example
    ./.venv/bin/python experiments/exp05_real_dynamics.py
    ./.venv/bin/python scripts/check_comparable.py results/exp05_*.json results/exp08_*.json

Rebuild the manifest when a batch of downloads finishes, and re-run the
experiments that are meant to be compared -- not one of them. `--panels auto`
pools the directory for exploration; such a run is not comparable to anything and
should not be archived.
