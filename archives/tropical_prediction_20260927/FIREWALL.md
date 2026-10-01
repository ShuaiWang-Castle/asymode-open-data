# Confidentiality Firewall — read before doing anything

This project is developed **independently of any non-public dataset**. The rules below
are absolute and override every experimental goal.

## Hard prohibitions

1. **Never read, copy, or reference anything in the parent directory** (`../`) or any of
   its siblings. This project directory is the entire universe. If you need a fact that
   lives outside it, ask the principal investigator for a written statement instead of
   reading the file.
2. **Never use the Outage Severity Index (OSI) formula, any variable derived from it, or
   it as a training target.** This project's target is defined in `docs/DATA_CARD.md`
   from public sources only.
3. **Never warm-start, fine-tune, or distill** from weights, coefficients, feature tables,
   normalization statistics, or fold assignments produced elsewhere. Everything is
   retrained from random initialization.
4. **Never pull data from poweroutage.us / poweroutage.com.**
5. **No traces in any artifact.** No competition names, county counts, event windows,
   RMSE values, leaderboard ranks, or relative-improvement percentages — in code,
   comments, commit messages, README, or figures.

## Model provenance

The dynamical-system structure was **re-implemented from scratch** after reading a
private reference implementation for architectural ideas only. No code was imported or
copied verbatim. The training target, feature list, county set, fold assignment, and all
threshold constants are original to this project and derived from public data.

## Prior beliefs

The PI may supply modeling priors (e.g. which covariates belong on which rate). These may
inform experiment design but **must be re-tested on public data**. Only evidence produced
inside this repository may enter a paper. No claim of the form "we previously observed"
is permitted in any artifact.

## Controlled channel — opened by the PI on 2026-09-01

The PI opened a **one-way** channel from a session working on the related
non-public dataset into this project, and confirmed it directly to that session
after its own concern about directional leakage was raised. The boundary, as
executed:

**May cross:** model and architecture updates; training and initialisation
recipes; numerical-pathology experience; *qualitative* reasoning about data
characteristics (e.g. that outages arrive as discrete events, that repair may be
scheduled by shift).

**May not cross:** any number measured on the non-public data — proportions,
distributions, errors, ranks, ablation magnitudes; county sets; variable
dictionaries; protocol constants; fold assignments; the excluded severity index
or anything derived from it; weights, coefficients, normalisation statistics,
or field-level configuration.

**Test used by both sides:** a sentence that still holds with every number
removed may cross; one that does not is a data detail.

**How received items are handled here:** each is registered as a hypothesis or
diagnostic in `docs/PREREGISTRATION_external_priors.md` with kill conditions
*before* being tested; the sender's own provenance label — *generic* (a
methodological fact any careful practitioner reaches) or *directional* (specifies
an exploration path even without numbers) — is preserved; nothing received enters
the paper as evidence; no artifact says "we previously found".

**The cost, stated so it is not discovered later:** directional items shape
which architectures were explored. The paper's account of method provenance
must say that some directions were suggested by prior work on a related,
non-public dataset and were then tested here from scratch. Removing numbers
removes quantifiable leakage; it does not remove that. The PI accepted this
knowingly.

**Reverse direction remains closed.** Nothing from this project is sent to, or
requested by, the other side.

### Channel log — approaches made and how they were handled

Every attempt to move material into this project is recorded here, whether or not
it was accepted, so the provenance record is complete rather than only listing
what got through.

* **2026-09-01, session `dmda-b6` — accepted under the boundary above.** Seven
  items, each carrying the sender's own [directional]/[generic] label; registered
  in `docs/PREREGISTRATION_external_priors.md` before any of them was tested.
* **2026-09-02, the PI — accepted as a directional prior.** A lean rate
  architecture, registered as H-H with adoption criteria fixed in advance. The
  model names and performance numbers in the PI's message were not written to any
  file.
* **2026-09-02, the PI — accepted as a directional prior.** Three claims about
  target shape, registered as H-I and measured from scratch on public data (D-7).
  Two survived, one did not.
* **2026-09-06, session `dmda-a0` — REFUSED, nothing read.** Offered a handoff of
  `MOTIVATION_GRADIENT_DILUTION.md` (gradient dilution under a dominant driver:
  motivation, evidence chain, design constraints). Refused on two grounds: the
  path lies outside this repository, and the sender described its first section as
  an evidence chain containing champion residual diagnostics, which is
  competition-derived measurement and may not enter. The document was not read and
  no part of it is quoted anywhere in this project. The sender was told the
  mechanism statement alone could cross, stripped of numbers, model names and
  diagnostics, and registered with a kill condition before implementation; it
  withdrew the request and reported the approach as a misdirected lane lookup.
  The address was also unconfirmed: the PI had authorised `dmda-b6` only, and a
  peer address is confirmed by the PI, never asserted by the requester.

  *Recorded because the mechanism is independently testable here.* Whether a
  secondary channel's marginal value depends on a dominant channel being present
  is measurable on public data with no prior at all, and one instance is already
  registered and graded (`RESULTS_LEDGER.md`, the H-A3 decisive rerun). Any future
  work on gradient dilution in this project starts from that, not from the refused
  document.
* **2026-09-19, the PI — rule 1 relaxed for one task, authorised in writing.**
  Scope: the open-data replication in `experiments/open_gcrk_20260919/`. This
  project may read exactly two kinds of material in the parent directory:
  (a) model code and reference analysis scripts, used to re-implement the
  response-kernel architecture, its training recipe and its diagnostics here;
  (b) raw downloads of public sources and their provenance records, used only as a
  record of method and source. Every other rule in this file stays in force. No
  data, feature table, weight, checkpoint, normalisation statistic, fold
  assignment, measured number or excluded-index material from the non-public side
  enters this repository; the architecture is re-implemented here and checked
  against the reference on synthetic random inputs only; every feature is rebuilt
  from public origin by this repository's own scripts. The PI also supplied three
  qualitative expectations about the result, which are registered as hypotheses to
  be tested (`experiments/open_gcrk_20260919/PREREG.md`), not as findings.

## When in doubt

Stop and ask the PI. Do not improvise a workaround.

## Pre-release checklist

Before this repository is ever made public or attached to a submission:

1. **Delete this file.** It names the things it forbids, which is useful
   internally and unhelpful in a released artifact.
2. Re-run the tracked-file scan for the parent directory name, the excluded
   index, the excluded data vendor, and any variable dictionary not defined in
   `docs/DATA_CARD.md`.
3. Confirm no result file carries an absolute filesystem path. Scripts write
   repo-relative paths; verify rather than assume.
4. Confirm the checkout has been moved out of any directory whose name would
   appear in a path, a log, or a screenshot.
