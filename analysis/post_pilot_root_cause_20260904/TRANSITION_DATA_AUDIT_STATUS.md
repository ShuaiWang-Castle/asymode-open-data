# Transition/data-path audit status

## Status: blocked in GitHub Actions; required in the original pilot environment

The source-level teacher-forced mask defect is established by inspection of `experiments/paper_v2_pilot.py`:

```text
pilot mask at step t: observed(origin) AND observed(next)
required mask:        observed(current) AND observed(next)
```

Because the hourly state array is zero-filled before packing, an unobserved intermediate current state can enter Stage A as a fabricated zero. The exact contaminated-row share cannot be computed from a normal branch checkout because the `data/interim/panel_*.npz` bytes used by the pilot are not present in GitHub Actions.

Run the committed audit in the same mounted environment that produced the pilot:

```bash
git checkout open-audit-20260904
git pull --ff-only origin open-audit-20260904
PYTHONPATH=src:experiments \
python analysis/post_pilot_root_cause_20260904/transition_data_audit.py
```

Expected outputs:

```text
analysis/post_pilot_root_cause_20260904/TRANSITION_DATA_AUDIT_GENERATED.md
analysis/post_pilot_root_cause_20260904/TRANSITION_DATA_AUDIT.json
analysis/post_pilot_root_cause_20260904/TRANSITION_EVENT_AUDIT.csv
```

This audit also compares:

- anchor-window transitions with all adjacent observed transitions;
- positive, negative and quiet transition shares;
- row-pooled and equal-event constant fits;
- sigmoid derivative at the resulting interruption initialization.

Do not rerun a neural fit until these outputs exist and the illegal-row share is zero after repair.
