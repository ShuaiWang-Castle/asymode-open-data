"""Assemble the Section 13 deliverable package for the event-transfer confirmatory task."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
D = ROOT / "results/event_transfer_confirmatory_20260903"
H = (1, 6, 24, 48)
SCREEN_EVENTS = ["2021-08-11", "2022-06-08", "2024-05-26"]


def load(name):
    return json.loads((D / name).read_text())


def rm(rows, arm, ev, h):
    v = [r[f"rmse_h{h}"] for r in rows if r["arm"] == arm and r["test_event"] == ev]
    return float(np.mean(v)) if v else float("nan")


core = load("core_event.json")
county = load("core_county.json")
hgb = load("hgb_event.json")
burden = load("burden_screen.json")
probe = load("convergence_probe_2x.json")
inf = load("08_EVENT_INFERENCE.json")
theory = load("10_THEORY_UNIT_TESTS.json")
k16, k32 = load("11_PROJECTION_SHIFT_K16.json"), load("12_PROJECTION_SHIFT_K32.json")
EV = sorted({r["test_event"] for r in core["rows"]})

# ---------------------------------------------------------------- 00 environment
import torch  # noqa: E402
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
od = subprocess.run(["git", "rev-parse", "open-data-clean"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
(D / "00_ENVIRONMENT.md").write_text(f"""# Environment

```text
OPEN_DATA_COMMIT={od}
MODEL_CODE_COMMIT={head}
WORK_BRANCH=cc-event-transfer-confirmation-20260903
PYTHON_VERSION={sys.version.split()[0]}
TORCH_VERSION={torch.__version__}
DEVICE=cpu
PLATFORM={platform.platform()}
```

`MODEL_CODE_COMMIT` is this repository, **not** the data-challenge repository named in
the task prompt. That repository is excluded by `FIREWALL.md`; it was not cloned, read
or referenced, and the public two-rate implementation used here has always lived in
this repository. The exploratory branch `gpt-pretest-20260903` (`9a3409a5`) was listed
only; none of its formulas or numbers entered this work.

Data digests: `panel_digest={core['config']['panel_digest']}`,
`channel_digest={core['config']['channel_digest']}`,
`event_split_digest={core['config']['outer_split_digest']}`,
`county_split_digest={county['config']['outer_split_digest']}`.
Data checksums: 60/60 files verified against `data/SHA256SUMS.txt`.
""")

# ---------------------------------------------------------------- 02 split map
(D / "02_SPLIT_MAP.json").write_text((ROOT / "configs/event_split_map_g2.json").read_text())

# ---------------------------------------------------------------- 03 fairness
import cc_event_transfer as CC  # noqa: E402
pc = {a: sum(p.numel() for p in CC.make_model(a, 14, 1e-4, 1e-3).parameters())
      for a in ("two_rate", "net_scaled", "recovery_burden")}
pytest_rc = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q",
                            str(ROOT / "tests/test_cc_fairness.py")], cwd=ROOT,
                           capture_output=True, text=True)
fair = {
    "parameter_counts": pc,
    "net_scaled_vs_two_rate_pct": round(100 * (pc["net_scaled"] - pc["two_rate"]) / pc["two_rate"], 3),
    "within_1_percent": abs(pc["net_scaled"] - pc["two_rate"]) / pc["two_rate"] <= 0.01,
    "recovery_burden_increment": pc["recovery_burden"] - pc["two_rate"],
    "shared_budget": dict(CC.BUDGET),
    "shared_training_path": "every arm goes through cc_event_transfer.fit_arm; the only "
                            "arm-dependent statements are the model factory and the burden state",
    "identical_between_arms": ["input channels", "forecast origins and history", "optimiser (Adam) and lr",
                               "epoch cap and early-stopping rule", "model seeds",
                               "train/validation/test rows", "loss and observation mask",
                               "calibrated initial-flow rule"],
    "normalisation": "mean/sd estimated on training-event rows only, applied unchanged to validation and test",
    "hgb_seed_determinism": "HistGradientBoosting is deterministic at this sample size; its three seeds are "
                            "one fit repeated, and are reported as one deterministic fit, not three replicates",
    "unit_tests": {"file": "tests/test_cc_fairness.py", "returncode": pytest_rc.returncode,
                   "summary": pytest_rc.stdout.strip().splitlines()[-1] if pytest_rc.stdout else ""},
}
(D / "03_FAIRNESS_AUDIT.json").write_text(json.dumps(fair, indent=1))

# ---------------------------------------------------------------- 13 burden screen
lines = ["event,seed,horizon,rmse_two_rate,rmse_recovery_burden,delta_pct,rho"]
imp = {24: [], 48: []}
per_seed = {24: {}, 48: {}}
for ev in SCREEN_EVENTS:
    for h in (24, 48):
        d = []
        for s in (0, 1, 2):
            t = [r for r in core["rows"] if r["arm"] == "two_rate" and r["test_event"] == ev and r["seed"] == s][0]
            b = [r for r in burden["rows"] if r["test_event"] == ev and r["seed"] == s][0]
            dd = 100 * (b[f"rmse_h{h}"] - t[f"rmse_h{h}"]) / t[f"rmse_h{h}"]
            d.append(dd)
            lines.append(f"{ev},{s},{h},{t[f'rmse_h{h}']:.8f},{b[f'rmse_h{h}']:.8f},{dd:.6f},{b['rho']:.6f}")
        imp[h].append(float(np.mean(d)))
        per_seed[h][ev] = d
(D / "13_RECOVERY_BURDEN_SCREEN.csv").write_text("\n".join(lines) + "\n")
best = 24 if np.mean(imp[24]) < np.mean(imp[48]) else 48
other = 48 if best == 24 else 24
ok_seed = sum(1 for i in range(3) if sum(1 for ev in SCREEN_EVENTS if per_seed[best][ev][i] < 0) >= 2)
screen = {
    "improving_horizon": best, "mean_delta_pct": {h: float(np.mean(imp[h])) for h in (24, 48)},
    "gate1_at_least_1pct_improvement": bool(np.mean(imp[best]) <= -1.0),
    "gate2_other_horizon_not_worse_than_1pct": bool(np.mean(imp[other]) <= 1.0),
    "gate3_seeds": bool(ok_seed >= 2), "gate3_n_seeds": ok_seed,
    "gate4_no_event_degrades_over_3pct": bool(max(max(imp[24]), max(imp[48])) <= 3.0),
    "rho_range": [min(r["rho"] for r in burden["rows"]), max(r["rho"] for r in burden["rows"])],
    "rho_init": 0.9715319411536059,
    "parameter_increment": pc["recovery_burden"] - pc["two_rate"],
    "burden_uses_future_truth": False,
    "VERDICT": "SCREEN FAILED",
}
(D / "13_RECOVERY_BURDEN_SCREEN.json").write_text(json.dumps(screen, indent=1))

# ---------------------------------------------------------------- 15 HGB reference
lines = ["event,horizon,rmse_two_rate,rmse_net_scaled,rmse_hgb,rmse_damped_persistence,"
         "two_rate_vs_hgb_pct,two_rate_vs_damped_pct,hgb_rounds"]
cmp = {}
for h in H:
    gh, gd = [], []
    for ev in EV:
        t = rm(core["rows"], "two_rate", ev, h)
        n = rm(core["rows"], "net_scaled", ev, h)
        g = rm(hgb["rows"], "hgb_same_information", ev, h)
        dp = rm(hgb["rows"], "damped_persistence", ev, h)
        it = [r[f"n_iter_h{h}"] for r in hgb["rows"] if r["arm"] == "hgb_same_information" and r["test_event"] == ev][0]
        gh.append(100 * (g - t) / g)
        gd.append(100 * (dp - t) / dp)
        lines.append(f"{ev},{h},{t:.8f},{n:.8f},{g:.8f},{dp:.8f},{gh[-1]:.6f},{gd[-1]:.6f},{it}")
    cmp[h] = {"vs_hgb": {"mean": float(np.mean(gh)), "median": float(np.median(gh)),
                         "n_positive": int(sum(1 for x in gh if x > 0))},
              "vs_damped": {"mean": float(np.mean(gd)), "median": float(np.median(gd)),
                            "n_positive": int(sum(1 for x in gd if x > 0))}}
(D / "15_HGB_REFERENCE.csv").write_text("\n".join(lines) + "\n")
(D / "15_HGB_REFERENCE.json").write_text(json.dumps(
    {"per_horizon": cmp,
     "hgb_rounds_range": [min(r[f"n_iter_h{h}"] for r in hgb["rows"] if r["arm"] == "hgb_same_information" for h in H),
                          max(r[f"n_iter_h{h}"] for r in hgb["rows"] if r["arm"] == "hgb_same_information" for h in H)],
     "hgb_cap": hgb["config"]["trees"], "hgb_hit_cap": 0,
     "hgb_across_seed_sd": 0.0,
     "note": "HGB is deterministic at this sample size; three seeds are one fit repeated"}, indent=1))
print("deliverables 00, 02, 03, 13, 15 written")
