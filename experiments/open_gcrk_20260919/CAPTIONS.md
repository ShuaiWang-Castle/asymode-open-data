# Figure captions (drafts)

Numbers in the captions come from `RESULTS.md` (with their files). Figure files are in
`figures/` as PDF and PNG; include at full text width (`figure*`).

## Figure 1 — `fig1_event_county_impacts_open`

County-level impacts of the representative event (8–16 December 2021, UTC). Peak share of
customers without power in each panel county during wave A (forecast hours 72–95), wave B
(hours 168–215) and the whole forecast window (hours 72–215). EAGLE-I county records
divided by the publisher's modelled 2024 county customers; logarithmic colour scale; grey:
counties outside the panel (no wind report in the forecast window, or a failed data gate).

## Figure 3 — `fig3_gcrk_interpretation_open`

GCRK response analysis in Siskiyou County, California (December 2021 event; seed 0, the
county's own held-out fold), the county chosen by the pre-registered rule: among counties
with wind reports in both waves and at least 10,000 customers, the largest observed
outage rise during wave B. (a) Contribution of the deposit at hour v to the damage logit
at forecast hour t. The kernel K(t,v) = ν R_t⋯R_v is rebuilt from the trained model (state
error 3×10⁻⁷) and its contribution is split over deposit hours by integrated gradients
along kernel exit closed → open, exact for the piecewise-linear readout up to quadrature.
Strip: deposit norm ‖d_v‖; red bar: the push window (hours 168–215). (b) What drives the
push, the kernel's contribution to the damage logit summed over the push window:
integrated gradients over the damage network's weather inputs (14 current channels and
26 causal summaries, all hours) and the 31 geographic descriptors that enter only through
GCRK, from weather at the fitting-set mean (no departures, zero push) and the neutral
geographic code. Contributions that raise and lower damage are shown separately, weather
dark and geography light; they sum to the push (−11.7). (c) Top: net push per forecast
hour (row sums of a). Bottom: forecasts from the observed p_71 — observed; the weather host
W (same initialisation and protocol without GCRK); AsymODE + GCRK; and the same GCRK network
with its kernel exit closed, a configuration trained through drop-path. Here the push
falls while the damage rate is near zero, so the open and closed forecasts differ by at
most 0.07 percentage points, and neither model reaches the observed 31.9% peak. (d) RMSE
change of GCRK relative to W over all held-out county-hours by lead time: bars are means
over five seeds, dots are the seeds, labels give the mean and the number of seeds in
which GCRK is lower.

## Figure 3S (supplementary, post hoc) — `fig3s_gcrk_largest_kernel_effect_posthoc_open`

As Figure 3 for Oneida County, Wisconsin, **chosen after training from model output**: the
county of the same event with the largest difference between the forecasts with the kernel
exit open and closed during wave B (seed 0). The push (+78.9 on the logit summed over hours
168–215) is carried by weather inputs (+78.9; geography −0.01). With seed 0 the kernel lifts
the wave-B forecast from 9.6% (exit closed) to 46.5% at the observed 44.5% peak, against
14.8% for W; across the five seeds GCRK gives 19.4–46.5% and W 14.2–31.2% at that hour.
Shown to illustrate the mechanism, not as a typical county.

## Figure 4 — `fig4_county_trajectories_open`

Outage trajectories in eight counties chosen by a pre-registered rule: the Figure 3 county;
for each event, the county with at least 10,000 customers and the largest observed
forecast-window peak; then further counties of the December 2021 event with wind reports
in both waves, by observed wave-B rise. Observed (all 216 hours), AsymODE + GCRK (seed 0;
one open-loop rollout from the observed p_71, each county from its own held-out fold) and
TimesFM zero-shot with ERA5 covariates. Shading: outages observed, hours 0–71. All eight
counties start the forecast at p_71 = 0, where TimesFM stays near zero. Roscommon, Michigan
reaches the cap of 1 for two hours: the denominator is the publisher's modelled 2024
customer count.

## Table (main results) — `results/table_main_main.tex`

Forecast accuracy of the outage fraction on 2,660 held-out county-events (five county-
grouped folds; each county's events are held out together). RMSE over the whole forecast
(hours 72–215) and by lead time, and MAE, pooled over observed county-hours. W and
AsymODE + GCRK: mean ± standard deviation over five initialisation seeds, each trained by
its own inner early stopping and refit. TimesFM is evaluated zero-shot with ERA5
covariates. Lower is better.
