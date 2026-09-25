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
