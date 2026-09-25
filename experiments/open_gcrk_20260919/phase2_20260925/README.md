# Public-data measurement and mechanism study

This dated study follows the completed experiment recorded in
[`../RESULTS.md`](../RESULTS.md). Its historical numbers and files are not
overwritten. The status of each proposed intervention is recorded in
[`ATTEMPTS.md`](ATTEMPTS.md), including failures and missing inputs.

| Document or tool | Purpose | Status at creation |
| --- | --- | --- |
| [`PREPROCESSING_PROTOCOL.md`](PREPROCESSING_PROTOCOL.md) | Prospective row-level observation audit, alternative outage labels, and frozen comparison support | Design only; raw source files unavailable in this checkout |
| [`LITERATURE_AND_NOVELTY.md`](LITERATURE_AND_NOVELTY.md) | Relevant primary sources, methods that transfer or do not, and the novelty boundary | Evidence review |
| [`MECHANISM_DESIGN.md`](MECHANISM_DESIGN.md) | Bounded weather-to-geography process graph, strong baselines, and falsification gates | Hypothesis, not trained model |
| [`measurement_audit.py`](measurement_audit.py) | Native-resolution evidence audit on original public EAGLE-I rows and saved panels | Run only when raw inputs are restored |
| [`provenance_gate.py`](provenance_gate.py) | Refuse mixed/incomplete E3R2 artifact cohorts and distinguish replay identity from source-level rebuild provenance | Manifest audit available; file hashes require restored artifacts |
| [`process_graph_prototype.py`](process_graph_prototype.py) | Tiny synthetic-only dynamic-state feasibility checks | No real-data effectiveness claim |

The current checkout has tracked results and scripts but **not** the raw
EAGLE-I 15-minute records, nine-day ERA5 fields, 216-hour processed panels,
or saved model checkpoints. An older, differently configured 168-hour panel
on another branch cannot stand in for them. Refit and label-sensitivity
numbers will be reported only after restoring and hashing the actual public
inputs. The separate experimental histories of `main` and the research
branch are not merged by this study.

Run the synthetic checks from the repository root (NumPy and pandas only):

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/open_gcrk_20260919/phase2_20260925/measurement_audit.py --self-test
PYTHONDONTWRITEBYTECODE=1 python experiments/open_gcrk_20260919/phase2_20260925/provenance_gate.py --self-test
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s experiments/open_gcrk_20260919/phase2_20260925 -p 'test_process_graph_prototype.py' -v
```

After restoring a candidate E3R2 artifact directory, stream all twelve R1
panels, twelve R2 panels and the feature array through the locked hashes. A
missing file or any byte mismatch leaves `local_files_all_verified` false:

```bash
python experiments/open_gcrk_20260919/phase2_20260925/provenance_gate.py \
  --check-files --out runs/e3r2_provenance_gate.json
```

After restoring the **original national** public EAGLE-I parquet and the
corresponding research-round 216-hour panel, audit one event without modifying
the saved panel:

```bash
python experiments/open_gcrk_20260919/phase2_20260925/measurement_audit.py \
  --raw data/interim/eaglei_outages_2021.parquet \
  --panel data/interim/open_gcrk/panel216_2021-12-11.npz \
  --out runs/measurement_audit_2021-12-11_centered7.json --hash-inputs
```

Repeat with `--service-days 1`, `3`, and `14`; use `--service-mode past_only`
with a stated service-day lookback for the pre-origin-only sensitivity. A real
December event with a 14-day centered window needs **both adjoining yearly
parquets** on `--raw`; the script stops if a named annual file is missing.
For example, use `--raw data/interim/eaglei_outages_2021.parquet
data/interim/eaglei_outages_2022.parquet --service-days 14` for the December
2021 episode. Running on real data requires a parquet reader such as PyArrow,
plus the source files. The
script reports reconstruction disagreements explicitly; it does not turn
source-row density into a utility-uptime measure. Its capped forward carry is
a **label scenario**, not a validated fix for missing data. Use event-specific
parquet years, including the panel window's starting year if it differs from
the event-name year. Write outputs to ignored `runs/`, then commit only concise
provenance and audited aggregate findings to `ATTEMPTS.md`.

To keep the investigation reviewable, update `ATTEMPTS.md` at each substantial
data, code, or experiment milestone and push this research branch. Before any
new training, record event selection, exact eligible county/hour support,
observation assumptions, data provenance, resource estimate, and the
pre-registered W/W+Cin/simple-control comparisons. Report every event and
seed, including regressions and null results.
