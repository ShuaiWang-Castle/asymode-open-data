# Unified AISTATS v2 — start here

This branch replaces the previous parallel experiment directions with one locked study.

Read in order:

1. `docs/UNIFIED_AISTATS_V2_PROTOCOL.md`
2. `paper/DRAFT_UNIFIED_V2.md`
3. `configs/event_split_map_g2_two_validation.json`
4. `experiments/unified_aistats_v2.py`
5. `tests/test_unified_aistats_v2.py`

The single primary question is whether separating two opposing neural rates avoids the event-dependent projection error incurred by a parameter-matched signed-rate collapse.

Primary data: all eleven events in `g2-convective-11`.

Primary forecast scope: first 24 hours, with h+6 and h+24 endpoints.

Proposed model: the existing two-rate neural dynamics, unchanged in architecture.

Training: equal event weighting and a fixed 50/50 rollout/teacher-forced conditional-transition objective.

Main baselines: parameter-matched `net_scaled`, HGB on the same information, and damped persistence.

Authorized ablations: one-rate structural collapse and removal of the teacher-forced term. Nothing else.
