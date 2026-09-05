# ANEEL interruption records — event-level outages with observed restoration times

Brazil's electricity regulator, **ANEEL** (Agência Nacional de Energia Elétrica),
publishes every interruption recorded on the country's distribution networks. One
row is one interruption event affecting one consumer-unit set, and — the reason
this dataset is here — it carries **both the start and the end timestamp of the
event**.

That is a different observational regime from county-aggregated outage counts. In
an aggregate series the restoration process is latent: you see a falling curve and
must infer when each interruption ended. Here the end time is recorded, so the
duration of every event is observed directly. For a study about separating an
interruption process from a restoration process, that makes this dataset a
natural external check rather than just more of the same data.

**Nothing is redistributed here.** The archives are about 1.8 GB compressed and
roughly 15 GB as CSV. `scripts/fetch_aneel.py` downloads them from the official
portal and verifies them against `MANIFEST.json`.

```bash
python scripts/fetch_aneel.py --all --format parquet      # recommended
python scripts/fetch_aneel.py --all --format zip          # CSV archives, checksummed
python scripts/fetch_aneel.py --all --verify-only         # check what you already have
```

## Source, licence and required attribution

| | |
|---|---|
| dataset | Interrupções de Energia Elétrica nas Redes de Distribuição |
| portal | <https://dadosabertos.aneel.gov.br/dataset/interrupcoes-de-energia-eletrica-nas-redes-de-distribuicao> |
| publisher | Agência Nacional de Energia Elétrica (ANEEL), Brazil |
| licence | **Open Data Commons Open Database License (ODbL)** — <http://www.opendefinition.org/licenses/odc-odbl> |
| retrieved | 2026-09-04 |

**The ODbL is share-alike, and that has consequences for this project.** Using the
data is unrestricted, but if we publish a *derived database* built from it — for
example ANEEL-derived panels analogous to the county panels already released here
— that derived database must itself be offered under the ODbL, with attribution
to ANEEL and a note of what was changed. A paper, a figure or a model trained on
the data is a "produced work" and is not itself forced open, but the derived data
behind it is. Two practical consequences:

1. do not merge ANEEL-derived rows into an existing table released under different
   terms without deciding the licence of the result first;
2. keep ANEEL-derived artifacts in their own directory with their own licence
   notice, so the share-alike obligation stays attached to the right files.

Attribution to carry: *Agência Nacional de Energia Elétrica (ANEEL), Interrupções
de Energia Elétrica nas Redes de Distribuição, dadosabertos.aneel.gov.br, licensed
under the ODbL.*

## Coverage

Ten years, 2017–2026, are on the portal. Nine are held locally; **2020 is
available on the portal and was not in the local set** — `MANIFEST.json` records
which is which, and `fetch_aneel.py --all` will retrieve the missing one.

| year | zip size | held locally |
|---|---|---|
| 2017 | 78 MB | yes |
| 2018 | 194 MB | yes |
| 2019 | 209 MB | yes |
| 2020 | — | **no, on the portal** |
| 2021 | 213 MB | yes |
| 2022 | 232 MB | yes |
| 2023 | 250 MB | yes |
| 2024 | 256 MB | yes |
| 2025 | 270 MB | yes |
| 2026 | 203 MB | yes (partial year) |

Uncompressed each year is roughly 1–3 GB of CSV; 2017 is 1.05 GB and 2025 is
3.04 GB. Every archive held locally passed `unzip -t`, and its SHA-256 is in
`MANIFEST.json`. The portal also publishes a Parquet copy of every year, which is
what `fetch_aneel.py` retrieves by default.

## Record layout

Semicolon-delimited, quoted, Latin-1, one header row. Eighteen columns:

| column | meaning |
|---|---|
| `DatGeracaoConjuntoDados` | date the extract was generated (not an event field) |
| `IdeConjuntoUnidadeConsumidora` | id of the consumer-unit set |
| `DscConjuntoUnidadeConsumidora` | name of the consumer-unit set |
| `DscAlimentadorSubestacao` | feeder |
| `DscSubestacaoDistribuicao` | distribution substation |
| `NumOrdemInterrupcao` | interruption order number |
| `DscTipoInterrupcao` | scheduled or unscheduled (`Programada` / `Não Programada`) |
| `IdeMotivoInterrupcao` | cause code |
| **`DatInicioInterrupcao`** | **event start timestamp** |
| **`DatFimInterrupcao`** | **event end timestamp** |
| `DscFatoGeradorInterrupcao` | free-text cause, e.g. equipment failure, weather, third party |
| `NumNivelTensao` | voltage level |
| `NumUnidadeConsumidora` | consumer units affected by this record |
| `NumConsumidorConjunto` | consumer units in the set (a denominator) |
| `NumAno` | year |
| `NomAgenteRegulado` | distribution company |
| `SigAgente` | company short code |
| `NumCPFCNPJ` | the company's CNPJ (a corporate registration number, not a personal one) |

A data dictionary PDF is published on the portal alongside the resources.

## Things to check before using it for anything

These are open questions, not findings; nothing here has been measured yet.

* **What the denominator means.** `NumConsumidorConjunto` is a set-level count and
  `NumUnidadeConsumidora` is per record; whether their ratio behaves like the
  customers-out fraction used elsewhere in this project has not been verified.
* **Overlapping events.** Several records can cover the same consumer-unit set at
  the same time. Turning event records into a state trajectory requires an
  explicit rule for overlaps, and that rule is a modelling choice to be
  pre-registered, not a detail.
* **Scheduled interruptions.** `Não Programada` versus `Programada` separates
  storm damage from planned maintenance. Any weather-driven analysis has to decide
  which it keeps, before looking at outcomes.
* **Weather coverage.** The county-level ERA5 pipeline in this repository is built
  for United States counties. Brazil would need its own spatial join and its own
  aggregation weights; none exists here yet.
* **`DatGeracaoConjuntoDados` is an extract date**, identical across rows in a
  file (2026-07-24 in the archives held here). It is not an event time and must
  not be used as one.
