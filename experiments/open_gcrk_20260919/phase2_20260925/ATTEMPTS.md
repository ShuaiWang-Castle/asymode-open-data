# Open-data research: attempts and decision gates

Snapshot checked on 2026-09-25 UTC. This is an audit record and a planning
note, not a trained-model result or a proposal to overwrite the existing
protocol.

## Attempt 01: archive, literature, and measurement review (2026-09-25)

Input: the public research branch at `da465857e6dbc266e1f2fad104d049c68076ad11`,
`../RESULTS.md`, `../PREREG.md`, `../selected_events_e3.json`, the E3R2
feature checksum manifest, and the primary references in
`LITERATURE_AND_NOVELTY.md`. No local nine-day source tables, 216-hour feature
array, or trained checkpoint was present. The read-only review found that
99.94% "observed" in the original five-event result **includes inferred zeros**;
raw positive EAGLE-I rows alone cannot certify the missing slots as actual
zeros. The twelve-event input has only a seed-0 W+Cin screen; its five-seed
confirmation was registered but is not yet reported. A historical episode was
replaced after a source-coverage gap (documented in the event manifest).
Outcome: no new empirical accuracy or mechanism claim. Next gate: run the
native 15-minute audit against the original public rows and recompute
matched-support targets. No model outcome was used to set audit thresholds.

## Attempt 02: reproducible synthetic feasibility checks (2026-09-25)

Input: only generated records and seeded, synthetic weather/rate arrays; no
real outage observation, model checkpoint, or ERA5 data. `measurement_audit.py
--self-test` checks partial and whole-hour source gaps, a source row excluded
by the national proxy, explicit zero, a county numerator above its panel
denominator, conflict detection, service causality, a cross-year source
requirement, mask mismatch, and that B leaves long missing gaps **unknown**.
The first draft of B incorrectly treated long gaps as zero; it was corrected
*before* committing this audit. `test_process_graph_prototype.py` checks exact
residual-closed baseline parity, bounded rates/states/stock, rain-before-wind
versus the reversed sequence, reversible geography swap, no future-to-past
leakage, and a finite-difference gradient. Outcome: audit self-test and seven
unit tests pass. These establish only code behavior on synthetic inputs; no
prediction improvement or physical mechanism was measured.

## Attempt 03: audit scenario B against its registered source-row rule (2026-09-25)

Input: the branch's `measurement_audit.py` (pre-edit SHA-256
`bfc201d4c121a458dddbeed56d8df6af0398ac8c2ec83c3b0114bed4d4ee976d`),
`src/asymode/panel.py`, and `PREPROCESSING_PROTOCOL.md`; **synthetic rows only**.
No original national 15-minute parquet, 216-hour research panel, or saved
checkpoint was available in this checkout. Inspection found a protocol/code
contradiction: B filtered actual positive source rows through A's inferred
national-run/county-service mask, so it could discard a recorded count when
fewer than five national rows happened to exist at that timestamp. B also
carried a positive count across the hour-71/72 forecast origin despite the
explicit no-cross-origin rule. These were code defects, **not** observed
real-data frequencies or evidence that B improves accuracy.

Correction: B now retains valid on-grid source rows independently of A's
proxy mask; carry is limited to proxy-run slots and reset at the forecast
origin or a no-run slot. Conflicting duplicates, nonfinite/negative counts,
and counts exceeding the panel denominator are quarantined from B rather
than interpreted as zeros or clipped positives. The script reports B-only
county-hours and quarantined quarters; numerical A/B contrasts remain on
their common support. A's historic reconstruction and original result files
were not changed. A separate source row at a no-run slot, a pre-origin
positive, an above-denominator count, and a conflicting duplicate are covered
by synthetic assertions. The synthetic self-test and all seven bounded-state
unit tests pass; `git diff --check` passes. Corrected script SHA-256:
`9c6b58487c600ff486b214307684513de17dc588a95255bfcbb40754073b2f93`.

Negative and next gate: no real-source B-only count, invalid-row prevalence,
label sensitivity, or predictive comparison can be stated from this synthetic
check. The original national public yearly EAGLE-I parquet(s), matching
216-hour panel and their SHA-256 digests must be restored before running the
row-level audit. Stop if A fails exact mask/label reconciliation or if source
quality flags are nonzero; investigate the relevant rows before any fit.

## Attempt 04: distinguish artifact replay from public-source rebuild (2026-09-25)

Input: tracked provenance only; no raw or derived data payload was available.
The audited files and their SHA-256 digests were: `panel216_checksums.json`
`708ac1e0861e9a272768f4e9a01f38a1eac70a0723c30a96478d61e837429a9b`,
`panel216r2_checksums.json`
`438277daffaadb4f4bd11b0dcdf2328f6e0bdd3a02b26bfb8238ca9e33c3380f`,
`features_e3r2_checksum.json`
`05b4138db12007ab16e474724f68f568dd5112878435001c863d891888cdff7e`,
`sources.json`
`d31747bd4d7d688a4d86d2b1c600bbcf4a5e2862ea4e46204ddf8fb9c135888a`,
and `era5_fetch_log.jsonl`
`6393a7a756ffaa2975c86b8c9e647463b5d3fcef08978b1ef10820a5a49214fe`.

Finding: the locked E3R2 derived cohort is internally enumerated: all 12
weather-selected events have one R1 216-hour panel hash, one R2 panel hash,
and the final feature manifest fixes 6,122 county-events, 2,409 counties,
`d_u=42`, `G=40`, and SHA-256
`32c14c55038c16c4fccdd9e96e6c03f4dffeded8e0dd051553b02f0d92e86178`.
Thus a recovered artifact can be accepted or rejected byte-for-byte. This is
an **identity/replay** statement only; none of those payloads exists here and
none was hashed during this attempt.

The stricter rebuild audit is incomplete. The separate hourly-maximum-gust
(`era5_fg10`) download log contains hashes and CDS request bodies for all 12
events. `sources.json` contains the main multivariable ERA5 files for only the
original five events. The seven added events lacking a tracked main-ERA5
source entry are `2019-02-24`, `2019-11-27`, `2021-08-11`, `2022-04-13`,
`2022-06-17`, `2024-05-08`, and `2024-06-26`. This may be an unmerged
provenance record rather than a missing historical download, but it currently
prevents independent source reconstruction. Also, `panel216r2_checksums.json`
stores only ERA5 basenames; where the main and `era5_fg10` directories use the
same name, its `era5_files` list is ambiguous and contains duplicate strings.
The derived-panel SHA remains unambiguous.

Added `provenance_gate.py` (SHA-256
`88af3e2ae51ff1a0940b93e35e4309f23f32eb02f95eeaff9f6ecdb8a9128413`)
to enforce the exact 12-event/216-hour cohort, report upstream source gaps,
and optionally stream all 25 restored artifacts through size and SHA-256
checks. Its manifest self-test passes. No data were downloaded, no training
was run, and no accuracy or mechanism conclusion changed.

Next gate: locate the seven main-ERA5 request/hash records on the originating
machine or redownload those public windows with archived CDS request JSON and
new hashes. Do not edit the historical derived hashes. Then restore the 12 R1
panels, 12 R2 panels and `features_e3r2.npz`; require
`local_files_all_verified=true` before replay. A rebuild that yields different
hashes is a new versioned panel and requires matched W/W+Cin controls rather
than reuse of any historical checkpoint or score.

## Attempt 05: does temporal order establish a new-state capability? (2026-09-25)

Input: synthetic weather only and the tracked feature/host definitions;
`build_features.py` SHA-256
`e50af1d447cf15ba2cf44b3dd551d1e49e692075173e17c5267f4256c78cb176`,
`build_features_r2.py`
`4b935354275ac2dacd34a5b43d3996ce86649b19b1a52d3a168ba2b37708b0e5`,
and `asym_host.py`
`da366c4a7b4bda278fc24180448f8f5a9863d163e5099ddd5357aa4d2982daa1`.
No outage label, trained parameter, panel, checkpoint, fold, or OUTER result was
read. This is an information/capacity audit, not a fitted baseline comparison.

The E3R2 W+Cin damage host has 42 hourly weather/causal-summary inputs and a
learned scalar logit recurrence. The audit placed equal 20 mm rain pulses at
post-origin hours 80 and 98, followed by the same 25 m/s wind at hour 144,
with identical geography, total rain and current weather. Both rain pulses lie
outside the response hour's 24-hour windows but inside its 72-hour window.
The complete 42-dimensional R2 damage vector at hour 144 is exactly identical
(maximum absolute difference `0.0`). The proposed bounded wetness state still
differs (`0.0172894` versus `0.0487708`, absolute difference `0.0314813`) because
it retains timing through decay.

Negative necessity result: this collision does **not** establish unique memory
for the process graph. The full W+Cin host sees the earlier post-origin hourly
sequence before the response hour. A legal capacity witness in which its damage
MLP emits a rain pulse and its existing scalar smoother has constant forget
`0.95` also distinguishes the histories at hour 144 (`0.0018762` versus
`0.0047234`, difference `0.0028472`). These are constructed internal states,
not learned values or prediction errors. Thus a rain-order plot alone cannot
justify a new process state or a novelty claim.

Added `mechanism_information_audit.py` (SHA-256
`d83ab75b013bdf9c053615b4b27cfc4d76d3aa9538895830a845c4a2ad46ce95`)
and two unit tests (`test_mechanism_information_audit.py`, SHA-256
`275500491846503cd21aa2b93dd156d91017a16239741d5d2736a19c2222aae6`).
The standalone self-test and both unit tests pass. The mechanism protocol now
requires a parameter-matched generic bounded-state or multi-smoother control,
in addition to W+Cin, causal summaries and the low-rank fragility mixture.

Next gate: after verified data are restored, run an INNER/FIT-only timing and
optimization pilot with identical inputs, split, seed, stopping rule and
residual budget for W+Cin, the generic-state control and the structured graph.
Do not use already inspected OUTER labels for topology selection. Promote a
mechanism interpretation only if the structured graph beats generic recurrent
capacity and its ordering/geography ablations on fresh event transfer without
increasing false activity.

## Attempt 06: primary-source novelty and comparator audit (2026-09-25)

Input: five versioned primary publications only; no outage label, panel,
checkpoint, trained parameter, fold result, or weather payload was read. The
frozen source/claim manifest is `PRIMARY_LITERATURE_SNAPSHOT.json`, SHA-256
`e173df372807eaa03994260a6551ca744d100cb2477ae3e926fb144b004933ea`.
Its exact identities are Zhu et al. arXiv `2109.09711v3` (2025-08-05), Chen
et al. arXiv `2502.18321v3` / related DOI `10.1109/TSG.2025.3642407`
(2025-12-08), Kidger et al. NeurIPS 2020 paper
`4a5876b450b45371f6cfe5047ac8cd45`, Rackauckas et al. arXiv
`2001.04385v4` (2021-11-02), and Manning et al. DOI
`10.1038/s43247-025-02176-6` (version of record 2025-03-26). JSON parsing and
repository whitespace checks pass.

Finding: the candidate's individual ingredients are already substantially
occupied. Zhu et al. use learned discounted histories of 34 HRRR variables,
a nonlinear weather network, geographic vulnerability, location-specific
decay/recovery, and decaying spatial outage kernels. Chen et al. conserve
unaffected + outaged + restored customer stocks while neural rates depend on
local weather and socioeconomic covariates. Neural CDEs already supply a
general path-driven differentiable state with a universal-approximation
result, and UDEs already cover learned closures and sensitivities in forced or
constrained differential equations. Manning et al. already test antecedent
rainfall, wind direction, and season jointly and explicitly limit causal
interpretation.

Negative novelty result: neither “complex geography-weather interaction,”
“weather memory/order,” “bounded differentiable state,” “customer
conservation,” nor “PINN/neural ODE” is a defensible standalone originality
claim. The exact conjunction in the proposed process graph was not found in
these five sources, but this bounded audit is not a systematic priority search
and cannot support “first” language. The candidate is therefore framed as a
design-and-evaluation contribution only.

The comparison gate is tightened. In addition to W+Cin, causal summaries, the
static fragility mixture and the parameter-matched generic bounded state, a
Zhu-style learned discounted multi-weather accumulation plus static
vulnerability must be included. A full Neural CDE solver is optional for the
regular hourly setting, but the generic control must consume the identical
weather path and geography with the same initialization, residual budget,
seeds, split and stopping rule. If the graph only beats W+Cin, attribute the
result to added state capacity; if it fails to beat the generic and
discounted-kernel controls, do not claim support for the process topology.

No code, data transformation, training, or empirical score was produced in
this attempt. The current raw-data and 216-hour artifact blockers are
unchanged. Next literature gate: broaden the search to outage fragility,
conditional neural dynamics/hypernetworks, and multivariate hazard-state
models before drafting any priority claim. Next empirical gate: after exact
artifact verification, preregister the INNER/FIT-only comparator ladder and
lock fresh weather-selected event transfer before inspecting outcomes.

## Attempt 07: executable comparator-identity and fresh-cohort gate (2026-09-25)

Input: tracked source and manifest metadata only. No panel payload, outage
label, checkpoint, trained parameter or fold result was read. The relevant
source hashes are `asym_host.py`
`da366c4a7b4bda278fc24180448f8f5a9863d163e5099ddd5357aa4d2982daa1`,
`gcrk_train.py`
`f293cfbebc1bf327fd065a447c1904b6f662567d1c4a8d17a4fb7d499b288b1c`,
`gcrk.py`
`2cd3922dca7834733ced3c81ca3bda2b201ec52cac2704320d114fd7d6245ade`,
and `splits_e3r2.json`
`9c3cfba1cd3563520fa50a0363e2fc8d571b1013d4f695ed4f5147e0bc80a34c`.

Source-level budget finding: E3R2 W+Cin adds exactly `6 * 32 = 192`
zero-initialized context-input weights. At `d=32`, geographic code rank `r=4`
and `G=40`, GCRK adds `rG + 3dr + 2d + 2 = 610` trainable parameters beyond
the wrapped host layer. These counts do not show which model predicts better,
but they establish that a direct W+Cin-versus-state comparison does not isolate
process topology from additional capacity.

Added a prospective six-arm ladder: W+Cin, causal summaries, Zhu-style
discounted multi-weather/static vulnerability, static fragility mixture,
generic bounded path state, and structured process graph. The generic and
structured states must match exactly on new-parameter count, input-view hash,
initialization pair, optimizer groups, stopping rule and residual-amplitude
cap. All six arms must have identical seed/fold cell sets and common panel,
split, weather-information, feature-name, observation-support, normalization,
host and training-protocol identities.

The critical phase boundary is now executable. The already inspected E3R2
panel/split is allowed only for `inner_fit_only`, seed-0 pilots and is forbidden
from producing an OUTER artifact. A confirmatory bundle must use a different
panel and split, a hashed weather-only event-selection rule locked before label
or result inspection, and seeds 0–4. This metadata gate cannot prove that a
source is truthful or a cohort is genuinely fresh; it only rejects internal
inconsistency. Source manifests and review remain mandatory.

Artifacts: `COMPARATOR_PROTOCOL.json` SHA-256
`37091751bfb52a63e3825bc318ee2d963e7d1e5db3633cad611b7779372f59cc`,
`COMPARATOR_PROTOCOL.md`
`4a19bd58be4c97047d53f0522a4480d48ca67cace918edd1106920d70605e482`,
`comparator_protocol_gate.py`
`fa19d476ac2fde8923353a6e68d12ad750a7bf45783e7f28f8ce3a86846624ae`,
and `test_comparator_protocol_gate.py`
`dc1620768b6cc6e140c254a222026dc2da35b2281b799f193d103e8af7dc5784`.
The standalone self-test accepted two internally valid pilot/confirmation
examples and rejected eight mixed or post-selected examples. Nine unit tests
pass, covering historical-panel reuse, OUTER leakage in a pilot, missing arm or
fold cells, identity overrides, unmatched inputs/capacity, and corrupted
source-derived parameter counts.

Negative boundary: this attempt does not implement or train any of the new
arms, does not establish that a 610-parameter model overfits, and does not make
E3R2 confirmatory again. The raw EAGLE-I records and 216-hour payload remain
absent. Next gate: recover and verify the historical artifacts for an
INNER/FIT-only timing pilot, implement the generic and structured states with
audited identical budgets, and separately freeze a genuinely new
weather-selected cohort before any confirmatory OUTER evaluation.

## Repository and access snapshot

- Local checkout: `open_data_work`, branch `research/open-gcrk-data-mechanism-20260925`, HEAD `da465857e6dbc266e1f2fad104d049c68076ad11`. The remote branch `research/open-gcrk-5seed-20260919` is at the **same SHA**. The remote `main` is `8dd47c5ccd829611f27b69a3d64c274a0a24c400` (2026-09-03); the research commit is dated 2026-09-21. GitHub reports **no common ancestor** between these histories; use explicit refs, not `git merge main` or a naive ahead/behind count.
- GitHub connector authentication is `ShuaiWang-Castle`; repository metadata reports `push: true` and `admin: true`. This establishes connector account permission, but an HTTPS `git push` from this container has **not** been tested. `gh` is absent. Never push to `main`; create/update only a named research branch after reviewing the changes. GitHub: https://github.com/ShuaiWang-Castle/asymode-open-data.
- The initial five-event W/GCRK experiment reports five seeds. The later twelve-event E3 round was stopped after **seed 0**, and the context-aware host W+Cin is also a seed-0 screen; Amendment 10 registers seeds 1–4 confirmation. Keep the published seed-0 screen distinct from a five-seed result; see `experiments/open_gcrk_20260919/PREREG.md` Amendments 3 and 10 and `RESULTS.md` sections 15 and 21.

## Data and execution gates

1. **Source/provenance gate.** Only public upstream sources or the public derived 26-panel release on the independent `main` history may be used. The release includes `data/SHA256SUMS.txt` but excludes about 4 GB of raw source downloads. Its old panels are 168-hour windows; they cannot by themselves reproduce the research branch's 216-hour, twelve-event R2 build. Check the specific rebuilt panel, source and feature checksums before training. Do not mix old 168-hour arrays into a 216-hour fold. The E3R2 manifest expects SHA-256 `32c14c55038c16c4fccdd9e96e6c03f4dffeded8e0dd051553b02f0d92e86178` for `features_e3r2.npz`, 6,122 county-events, 2,409 counties, 12 events, and descriptor dimension 40. If any input is regenerated differently, version the manifest and rerun matched controls on that same panel.
2. **Current data availability.** This checkout has no `data/`, `runs/`, or `.venv/`. Two old 168-hour panels inspected in a temporary workspace are insufficient. The standard Python has NumPy/Pandas, but lacks PyTorch, pytest, xarray, rasterio and cdsapi. Do not report a training run from this container until the missing dependencies, public raw/derived inputs and checksum gate have been satisfied. Restoring the public `main` release can bootstrap the old panels, **not** restore the 216-hour R2 features or research-round model checkpoints.
3. **Observation and causality gate.** EAGLE-I omissions are not equivalent to observed zero: retain the observation mask in all loss/metrics and investigate long collection gaps. County/hour weather features and antecedent wetness must be computable using weather available by that hour; static geography cannot encode an event's future outage. Split by county for the outer test and by event for transfer; fit scalers, imputation and any geographic regimes using development units only where the scientific claim concerns unseen geography. The existing published regime screen fitted K-means on all counties using static covariates: label it as its original protocol, not an unseen-geography test.
4. **Model evidence gate.** Establish W and W+Cin on identical data, folds, seeds, initial conditions and rollout time before evaluating a new geographic dynamics model. Include capacity-controlled shared-state, static-context, and true/permuted-geography controls; report RMSE plus MAE, false activity, onset/peak and each event, not only the pooled score. Use five seeds only after a predeclared development-screen criterion; reserve held-out event folds for a single final interpretation. Gradients and a named latent state require counterfactual perturbations and planted-data recovery checks before any mechanistic or causal attribution claim. The SIR-style county weather/covariate neural ODE of Chen et al. (https://arxiv.org/html/2502.18321v3) is a close prior; population balance or a generic "PINN" label is not a novelty claim.
5. **Compute gate.** This container exposes `nproc=9`, 9.7 GiB RAM, **no swap**, and 29 GiB free disk at snapshot time. Reserve at least two CPUs (maximum seven workers) and keep each worker at one BLAS/PyTorch thread. Start at one worker and measure peak resident memory for a representative fold, then choose `workers <= min(7, floor(memory_budget / peak_RSS))`; four workers is a conservative starting candidate only after the memory probe. The registered E3 round estimates roughly 90 CPU-hours, at least 22.5 hours with four workers before overhead. It cannot honestly be promised within an 8–10-hour session. Queue a bounded screen or preprocessing validation first; record any design shortening as an explicit protocol amendment rather than silently calling it the five-seed confirmation.
6. **Result acceptance gate.** Preserve every attempt, including failed or neutral ones, with code commit, panel SHA-256, split file SHA-256, environment, hypothesis, changed component, parameters, CPU-hours, seed/fold, success/failure reason and per-event results. Mark exploratory selection separately from locked confirmation. Follow the published Amendment 10 criterion for W+Cin versus W: the county-cluster interval excludes zero and at least four of five paired seeds have the same sign. When the primary comparison or data preprocessing changes, freeze a fresh decision rule before examining corresponding outer results.

The twelve previous event outcomes have already been inspected. Repeated
development against their OUTER labels is exploratory even with event folds;
a new confirmatory claim needs additional weather-selected episodes whose
outcomes have not been examined for this proposed model.

## Read-only commands runnable in this checkout

```bash
# Run from the root of this checkout.
git status --short --branch
git show -s --format='%H %cI %s' HEAD
env GIT_TERMINAL_PROMPT=0 git ls-remote --heads origin main 'research/*'
git show -s --format='%H %cI %s' 8dd47c5ccd829611f27b69a3d64c274a0a24c400
git ls-tree -r -l 8dd47c5ccd829611f27b69a3d64c274a0a24c400 data/interim | head
git show 8dd47c5ccd829611f27b69a3d64c274a0a24c400:data/README.md | head -n 60
cat experiments/open_gcrk_20260919/data_provenance/features_e3r2_checksum.json
nproc
free -h
df -h .
python3 -B -c 'import importlib.util as u; print({x: bool(u.find_spec(x)) for x in ("numpy", "pandas", "torch", "pytest", "xarray", "rasterio", "cdsapi")})'
```

## Commands for a prepared environment (do not run until gates pass)

The public 26-panel **old release only** can be extracted into ignored `data/` without changing the checked-out Git history, if the 216-hour builder needs it as a starting point:

```bash
# Run from the root of this checkout.
git archive 8dd47c5ccd829611f27b69a3d64c274a0a24c400 data | tar -x
sha256sum -c data/SHA256SUMS.txt
```

Once an authorized environment has the research-round raw ERA5/public geographic inputs, feature artifact and locked environment, verify first. `features_e3r2.npz` is **not** in the public main archive:

```bash
# Run from the root of this checkout.
test -x .venv/bin/python
test -f data/interim/open_gcrk/features_e3r2.npz
python3 -B - <<'PY'
import hashlib, json, pathlib
spec = json.loads(pathlib.Path('experiments/open_gcrk_20260919/data_provenance/features_e3r2_checksum.json').read_text())
path = pathlib.Path(spec['file'])
actual = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
assert actual == spec['sha256'], (actual, spec['sha256'])
print('E3R2 features checksum OK:', actual, 'units:', spec['units'])
PY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
./.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_gcrk_equivalence.py tests/test_harness_protocol.py
```

To reproduce the registered twelve-event screen only after its model inputs and runtime exist, use its exact split file and `OPEN_GCRK_ROUND=e3r2`. The following writes ignored checkpoints and must run on a **separate bounded worker allocation**; it is not an instruction to start the whole queue now:

```bash
# Run from the root of this checkout.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPEN_GCRK_ROUND=e3r2
test -f experiments/open_gcrk_20260919/splits_e3r2.json
./.venv/bin/python experiments/open_gcrk_20260919/run.py worker --design main --seed 0 --fold 1 --arm W
# After inspecting this cell's peak RSS and elapsed time, choose a safe worker count.
./.venv/bin/python experiments/open_gcrk_20260919/queue_multi.py --designs main --seeds 0 --arms W GCRK --workers 2
./.venv/bin/python experiments/open_gcrk_20260919/evaluate_e3.py
```

Do **not** run the registered 5-seed confirmation (`--seeds 0 1 2 3 4` across `main event`) until the single-cell cost is measured and the whole protocol fits the agreed compute budget. An existing `DONE.json` makes queue scripts skip that cell; inspect checkpoint identity before trusting a resumed run. Keep attempts on a new branch under `research/`, and record the exact source tree rather than silently updating this historical result branch.
