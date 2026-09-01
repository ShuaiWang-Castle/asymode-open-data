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

## When in doubt

Stop and ask the PI. Do not improvise a workaround.
