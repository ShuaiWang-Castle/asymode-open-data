# Research draft: bounded process states for weather–geography outage response

**Status (2026-09-25):** A mechanism proposal and falsification plan, not an empirical finding or a claim of novelty. This draft uses only public-data variables. It does not change the registered experiments or their results; a dated protocol amendment is needed before any new confirmatory run.

## Motivation and scientific boundary

County customer outages are stocks resulting from simultaneous disruptions and restoration. A model that multiplies a county descriptor by wind speed, or merely stores weather in a response kernel, does not distinguish *how* earlier rain, wind, snow, or cold conditions may change the response to later forcing. We propose a small, shared **process graph**: weather drives bounded intermediate states; geography controls their transitions and the conditional exposure of the damage and recovery rates. The graph is shared across counties and events and has no event-ID or county-ID embedding. It is meant to test transferable conditional dynamics, not to assert that a latent state is a measured tree, line, road, or repair crew.

The first comparison is the current **W+Cin host**, not the weather-only W. W+Cin injects six county context variables into the first damage layer, `ReLU(W_x x_{i,t} + A c_i + b)`, with `A` initially zero (192 additional parameters). It already lets county context interact nonlinearly with weather. A proposed mechanism must beat this informed host and a parameter-matched direct-input control before we attribute any improvement to process structure. The recovery network, occurrence gate, background rate, customer denominator, and forecast origin should otherwise be held fixed in the initial comparison.

In round E3R2, W+Cin is also **not memoryless**. Its damage network receives 42
hourly inputs, including 6/12/24-hour windows and 72-hour path summaries, and
its raw damage logit passes through a learned scalar recurrence
`lbar_t = f_t lbar_(t-1) + (1-f_t) l_t`. Consequently, showing that a proposed
state distinguishes rain-before-wind from wind-before-rain establishes neither
an information advantage nor a unique dynamic capability over the full host.

The experimental question is narrow: **do geographically conditioned, ordered intermediate states improve transfer to unseen counties and unseen events beyond static context and ordinary causal weather summaries?** An answer of no is useful. Existing customer records do not supply component failures, line-adjacent trees, or crew deployments.

## A minimal bounded dynamical family

Let `x[i,t]` contain weather available at hour `t`, `g[i]` the public geographic descriptors, `c[i]` the existing county context, and `z[i,t]` a vector of `K <= 3` shared process states. Initialize each state from *pre-origin* weather and geography only, using a bounded encoder or public ERA5 soil moisture where appropriate. For a proposed state `k`, define

```text
a[k]_(i,t) = sigmoid(A[k](x[i,t], g[i], z[parent(k)]_(i,t)))
d[k]_(i,t) = sigmoid(D[k](x[i,t], g[i], z[parent(k)]_(i,t)))
z[k]_(i,t+1) = z[k]_(i,t) + a[k]_(i,t)(1-z[k]_(i,t)) - d[k]_(i,t)z[k]_(i,t).
```

Since `0 <= a,d <= 1`, every transition maps `[0,1]` into `[0,1]`. Use soft constraints, positive weights where a well-founded directional constraint applies, and a small bounded learned closure for unknown effects. In particular, an instantaneous wind-load *pathway* may be constrained to increase with wind speed at fixed state, while a universal monotone soil-moisture effect should **not** be imposed: wet soil may favor uprooting, whereas very dry conditions may favor breakage. Do not label a state `ice accretion` merely because ERA5 reports snowfall; without a suitable freezing-rain or accretion observation it is a snow/cold *load proxy*.

The small, testable starting graph is:

| State or pathway | Allowed weather forcing and geography | Hypothesized downstream effect | Measurement caveat |
| --- | --- | --- | --- |
| Antecedent wetness / root-support proxy | Past precipitation and soil water; public soil drainage, shallow soils, terrain | Changes later wind-conditioned damage susceptibility | County soil moisture is not root anchorage at a particular tree or feeder. |
| Accumulated mechanical load | Gust duration, wind direction, seasonal leaf cover; public canopy and terrain | Carries earlier stress into later wind-conditioned damage | County canopy is not the number of trees that can strike lines. |
| Snow/cold load proxy | Snowfall, temperature, precipitation phase where defensible | Changes damage during a later wind or cold wave | Do not infer radial ice thickness from snowfall. |
| Access impedance **(deferred)** | Rain, snow, terrain, documented public road/flood observations | Slows restoration conditional on existing predicted outage burden | Weather and terrain alone cannot establish a blocked road or crew allocation. |

The first implementation should include at most two damage-side states. Add the access state only after a separate measurement audit and a frozen damage-side result. Let the W+Cin pre-gate damage logit be `L_base`. Attach a bounded, initially zero residual from nonnegative pathway intensities:

```text
q_m(i,t) = softplus(F_m(x[i,t], z[i,t], g[i]))   for pathway m
delta(i,t) = alpha * tanh( sum_m q_m(i,t) - b )  with 0 <= alpha <= alpha_max
L_damage(i,t) = L_base(i,t) + delta(i,t).
```

An initially zero, box-constrained `alpha` (optimized with an explicitly documented projection) makes the model exactly W+Cin at step 0 and with the residual disabled. Choose `alpha_max` and all initialization rules on FIT/INNER data before OUTER evaluation. Retain the host's existing bounded occurrence, damage, and recovery rates. Its stock update remains

```text
p_hat(i,t) = p_hat(i,t-1)
           + u(i,t)(1-p_hat(i,t-1)) - r(i,t)p_hat(i,t-1),
```

so `p_hat` stays in `[0,1]` whenever `u,r` are in `[0,1]`. Compute `u(i,t)` from the state *before* using hour `t` weather to update that state; this makes a rain-at-`t` → wetness-at-`t+1` → later-wind path explicit. A direct current-weather path remains available. Unroll only forward in time; never initialize or update `z` with outage observations beyond hour 71. Inputs from ERA5 over the forecast window imply a **realized-weather / perfect-weather-information** experiment, not an operational 144-hour weather forecast.

Automatic differentiation can report `d p_hat(i,t) / d x(i,tau)` for `tau <= t` and `d p_hat(i,t) / d g(i)` through the transition Jacobians. Test these gradients with finite differences and synthetic planted mechanisms before interpreting them. An integrated gradient, geography swap, or model intervention describes the *fitted model's response*, not a physical causal effect identified from observations.

## Two simpler alternatives that must remain in scope

1. **W+Cin + causal summaries:** give the same host strictly past-looking cumulative precipitation, gust duration/path maximum, directional wind deviation, and seasonal leaf-cover proxies. Match the proposed model's parameter count where practical. This tests whether explicit intermediate states add anything beyond well-constructed inputs.
2. **Low-rank fragility mixture:** let `K` positive, geography-conditioned susceptibility weights multiply a few positive weather-load functions before the same bounded damage-rate readout. This is a cheap, strong structural control with no recursively updated state. Compare it directly with the process graph; neither a generic interaction nor a fragility mixture alone is an originality claim.

3. **Parameter-matched generic-state control:** attach the same number of
   bounded recurrent coordinates, with the same weather/geography information,
   initialization, residual scale and parameter budget, but without named
   wetness/load nodes or a predeclared process graph. A second diagnostic may
   replace these coordinates with `K` ordinary learned leaky smoothers. This
   separates gains from extra recurrent capacity from gains due to the claimed
   geography-conditioned topology. If the generic states match or beat the
   graph, report a capacity result rather than a mechanism result.

A gridded, within-county exposure model may be considered later if higher-resolution weather and defensible customer/line exposure proxies become available. Aggregating county-average forest to a purported feeder-specific tree threat would invent information. County adjacency is likewise not the electrical network.

## Identifiability, prior work, and a credible novelty claim

From the stock alone, damage and recovery are not separately identifiable. At one hour, `Delta p = u(1-p)-rp` is one observed equation for two unknown rates. For example, at `p=0.1` with `Delta p=0`, both `(u=0.01,r=0.09)` and `(u=0.05,r=0.45)` fit exactly. The same ambiguity extends to latent wetness, mechanical load, unknown line exposure, and repair capacity. A fitted state is therefore a **predictive susceptibility proxy under explicit assumptions**. Physical labels require independent component, vegetation, road, or crew observations and validation.

Novelty cannot consist of “physics-informed,” a neural ODE, a cumulative weather state, rain–wind interactions, hourly forecasts, geography, or fragility curves alone. Rackauckas et al.'s [universal differential equations](https://arxiv.org/abs/2001.04385) establish the general learned-closure approach; Zhu et al.'s [grid-resilience model](https://arxiv.org/html/2109.09711v3) already learns decaying weather accumulations and spatial outage dynamics. [Chen et al.'s GDF-NODE](https://arxiv.org/abs/2502.18321) already uses compartmental outage dynamics in a neural ODE. [Manning et al. (2025)](https://www.nature.com/articles/s43247-025-02176-6) investigate antecedent rainfall, wind direction, and season together, while warning about shared weather drivers and limited causal attribution. Additional work on vegetation and fragility must be checked before making a priority claim.

The *potential* contribution, if the comparisons succeed, is a **small bounded process graph for county customer-stock forecasting that provides testable temporal ordering and geography-conditioned transition responses under public-data transfer tests**. This exact conjunction still requires a focused prior-art audit before any claim of originality. Without a known county-scale governing PDE, adding an arbitrary residual penalty and calling the model a PINN would be misleading. Published [PINN gradient-pathology analysis](https://doi.org/10.1137/20M1318043) also argues against assuming a PDE-penalty architecture is easy to optimize here.

## Falsification and evaluation before any mechanistic claim

- **Data and scale gate:** Audit observation availability, outage denominator, UTC alignment, 30-day antecedent weather availability, and whether ERA5 resolves the event's gust footprint. Preserve outage-data masks; do not convert an unobserved feed into a true zero. Convective downbursts can be sub-grid, so compare local NOAA reports or a higher-resolution public weather product where feasible. No architectural comparison can repair label/forcing mismatch.
- **Synthetic recovery:** Plant a known order-dependent wetness → wind mechanism, a direct-weather mechanism, and a no-geography mechanism. Confirm the optimizer recovers the *predictive distinction* under the actual horizon, noise, and sample size. Test gradient direction and exact residual-closed parity with W+Cin.
- **Information-set collision:** `mechanism_information_audit.py` constructs
  two synthetic histories with identical total rain, current wind-hour weather,
  and the same 42-dimensional E3R2 damage feature vector at the wind hour. The
  two rain pulses and the response all occur after the forecast origin. The
  bounded wetness state differs by 0.0315, demonstrating information not present
  in that *single response-hour vector*. But a legal scalar host-smoother
  capacity witness also differs (0.00285), because the full host sees the prior
  hourly sequence. This is a negative necessity result: temporal-order plots
  alone cannot promote the process graph. It is not a trained comparison.
- **Temporal-order control:** Within credible weather ranges and matched gust peaks, compare rain-before-wind with wind-before-rain. A wetness explanation predicts a difference only when the hypothesized antecedent path can act; report controls for a direct rainfall effect, season, and unusual wind direction. Do not present simulated reorderings as observed causal experiments.
- **Geography and negative controls:** Swap geography among counties matched on climate and basic context as well as unrestricted donors; compare to sham or permuted geography and a parameter-matched static readout. Inspect false activity in observed-zero hours. Weather sensitivity should not be attributed to geography if donor maps perform equally well.
- **Generalization and uncertainty:** Keep county-grouped outer folds and event-held-out tests, paired seeds `0..4`, and the fixed open-loop trajectory metrics. Report pooled RMSE/MAE, event-equal and per-event errors, horizon segments, peak errors, and false activity. Cluster uncertainty at least by event or event-by-state as well as county; a few large storms can dominate pooled RMSE. Report every event and seed, including adverse effects. If the model wins only after a global amplitude rescaling, characterize the result as calibration, not a demonstrated mechanism.
- **Reused tests:** The earlier twelve-event OUTER labels and model outcomes have already been inspected. An architecture selected on these same events cannot be described as independently confirmed on them. Lock fresh weather-selected events or an external cohort for a confirmatory transfer test.
- **Ablations:** Disable each state, freeze geography-controlled transition rates, remove time ordering, and close the residual at inference. A pathway is unsupported if ablating it does not change out-of-event error or if its learned state saturates, responds to irrelevant weather, or shifts harm into quiet hours.

## Compute and decision gate

First run only data auditing, deterministic toy tests, and an INNER/FIT-only one-fold, one-seed timing and optimization pilot against W+Cin and the two simple controls. No OUTER outcomes should be used to tune the graph or its hyperparameters. Record wall time for feature construction, one inner fit, refit, and inference, CPU/thread settings, memory peak, and model parameter counts. Extrapolate conservatively to **every planned outer county and event fold, three inner folds, and all five seeds**, with a safety margin for refits and diagnostic inference. If the projected local total exceeds approximately 24 hours, or higher-resolution data cannot be obtained, record the obstacle and propose a concrete reduced scope to the PI before launching confirmatory training; do not silently omit folds, events, or seeds. Run the final comparison only after the new protocol and stop/failure criteria are registered.

Promotion requires a stable margin over W+Cin, the direct/summary and fragility
controls, and the parameter-matched generic-state control on held-out tests
without worse false activity, a temporally specific and geography-sensitive
ablation signal, and uncertainty intervals that justify the strength of the
wording. Otherwise report the failure and retain W+Cin as the defensible host.

## Primary sources for measurement decisions

- [Han, Guikema, and Quiring (2009), hurricane outage predictors](https://doi.org/10.1111/j.1539-6924.2009.01280.x): gust peaks and duration, antecedent soil wetness, land cover, and infrastructure exposure are distinct predictors; wet and dry conditions can act through different failure modes.
- [Wanik et al. (2017), LiDAR line-adjacent tree risk](https://hartman.byu.edu/docs/files/WanikParentAnagnostouHartman_LIDAR.pdf): forest cover alone does not identify trees close and tall enough to hit a line.
- [USFS tree-damage/terrain study](https://research.fs.usda.gov/treesearch/64642) and [USFS root-water experiment](https://research.fs.usda.gov/treesearch/54716): useful hypotheses about breakage/uprouting and root support, not validation of county-level electrical damage states.
- [Danziger et al. (2022), flood-related road access and outage restoration](https://barabasi.com/media/Danziger_et_al-2022-Nature_Communications.pdf): motivates an access hypothesis, conditional on separately measured road disruptions; it does not identify repair access from outage stocks.
- [Copernicus ERA5 documentation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) and [NWS downburst description](https://www.weather.gov/lmk/downburst): document the spatial-scale limitation before claiming local gust mechanisms.
- [NRCS gridded soil survey](https://www.nrcs.usda.gov/resources/data-and-reports/gridded-soil-survey-geographic-gssurgo-database), [USGS 3DEP](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services), and [USFS tree canopy products](https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/): public origin data for reproducible geographic proxies.
