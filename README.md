# Asymmetric two-rate dynamics for county power-outage trajectories

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
