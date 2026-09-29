# D03: outcome-blind high-dimensional structure audit

Written before the weather audit on 2026-09-28, following the PI's request for a
data-motivated geographic kernel. This is a descriptive information audit, not a
new prediction screen or a test of physical mechanisms. It preserves D01/D02.

## Inputs and observation support

Use only the public D panel's first 12 raw weather channels and small event/county
metadata, plus the existing deduplicated D02 county lookup. Stream the `xu` NPY
member to retain only those 12 channels. Do not load any outage, prediction,
observation-mask, target, or recovery-feature array. Read only `unit` and `anchor`
from the existing D01 OOF cache to reproduce its 50,302 supported 24-hour windows.
Their support was originally selected using observation availability; the present
audit is therefore outcome-value-blind, conditional on that existing availability
selection, rather than an entirely outcome-independent sample design.

Reuse the original five event folds and family/shared-county-near-date merged
groups. Report county-events, unique counties and independent event-group support;
50,302 windows are not 50,302 independent weather realizations. No C, new downloads,
neural training, prediction regressions, or paper edits.

## County geometry

Use the existing unique-county median imputation and standardization. Report two
geometries separately: the 40 physical-geography descriptors with four equally
weighted semantic blocks, and those 40 plus six county/service-context descriptors
with five equal blocks. Each block is divided by the square root of its number of
features times the number of blocks. The context block includes historical SAIDI;
the 46D spectrum cannot be attributed to the current kernel's 40D geography input.

For each covariance spectrum report all eigenvalue fractions, cumulative fractions,
participation rank, entropy rank, and dimensions needed for 80%, 90% and 95% of
descriptor variance. These are linear descriptor-dimension summaries, not intrinsic
manifold-dimension estimates or sufficient predictive ranks. The basic 46D spectrum
and slope-aspect harmonic counts were inspected from the existing cache before
writing this scope; their extension to pure 40D is not a new preregistered test.

Retain the eight-bin slope-aspect distribution. Summarize the magnitudes of its
first (directed) and second (axial) circular Fourier harmonics. Report counts at
0.05, 0.10 and 0.15, including counties with low first but high second harmonic.
These descriptive cutoffs do not establish meaningful mechanical orientation;
opposite slopes can cancel a directed mean. This audit does not link orientation
to outage outcomes or claim that a wind-direction kernel has been identified.

## Weather geometry, trajectories, and linear redundancy

Use the same fixed `log1p(max(x,0))` transforms for precipitation, snowfall and CAPE
as D01/D02. Analyze the pooled sample and all five regimes, always reporting both:

* Original design weights divided by retained anchors per county-event.
* Equal merged-event-group weight, retaining the design-weight proportions inside
  each group. Inside a regime, each represented group receives equal weight.

The first describes the existing weighted D exposure distribution; the second
describes a typical represented event group, not an alternate national estimand.
Neither is a prediction-loss criterion.

Construct the same 54 non-orientation D02 summaries: 12 channel means, 12 late-half
minus early-half contrasts, 15 simultaneous products among the six core channels,
and 15 centered antisymmetric moments over lags 1--12. Audit the standardized
correlation spectrum and the event-group concentration of each driver's variance.
For each order summary, linearly project it onto (1) the 12 means, (2) means and
trends, and (3) those 24 plus the 15 simultaneous products. Report its remaining
variance fraction and event-group concentration, including sparse or zero results.
Use a fixed Gram eigenvalue tolerance `max(1e-12, 1e-10 * largest eigenvalue)`.
This is an in-sample linear information decomposition, not incremental forecasting
skill, conditional independence, causality, or a test against all nonlinear main effects.

Separately expand each real 24-hour, 12-channel path in a fixed orthonormal DCT-II
basis. Keep the first eight time coefficients per channel for the 96D spectral
summary. No time warping, peak alignment or outcome-selected basis. Scale each
channel by its weighted standard deviation over rows and hours, then center each
hour/channel coordinate across rows. Retain the full 24-hour path energy as the
denominator and report the fraction retained by mean-only (DCT coefficient 0) and
by coefficients 0--7, pooled and per channel. The discarded energy is real temporal
variation, but its relevance to outage dynamics is unknown. Spectral summaries of
the truncated representation must not be called the rank of the full weather path.

All summary settings are fixed here, with no winner selection. Validate DCT
Parseval equality, monotone retained energies, source/row consistency, positive
semidefinite covariance, and projection normal equations. Run one finite process
at nice >=15 with at most two numerical-library threads. Save the independent
script and compact JSON; no raw weather/outage arrays are published.
