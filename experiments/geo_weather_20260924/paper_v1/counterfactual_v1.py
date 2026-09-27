"""Geography counterfactuals for the paper's Figure 3 (development tranche, event-held-out folds).

Every trained fold model (runs/geo_weather_20260924/<label>/fold0k/final.pt) is rebuilt and re-run on its own held-out
county-events with one input changed:

  AsymODE + GCRK    own descriptors (reproduces outer.npz) against the fitting counties' mean descriptors
                    (standardised geography set to 0).

Seeds that have all five folds are averaged in forecast space. No weights change and nothing is refit.
Writes runs/geo_weather_20260924/paper_v1/counterfactual_<name>.npz (system, fips, seeds, P_<variant> [U, 144]).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "2")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(EXP))
from asymode import gcrk_train as G  # noqa: E402
from screen import load  # noqa: E402

RUNS = ROOT / "runs" / "geo_weather_20260924"
OUT = RUNS / "paper_v1"
SEEDS = (0, 1, 2, 3, 4)
MODELS = {"gcrk": dict(label="v1_gcrk_s{}", arm="GCRK+Cin")}


def rebuild(F: dict, arm: str, state: dict) -> torch.nn.Module:
    model = G.AsymODE(F["xu"].shape[-1], F["xr"].shape[-1], F["xo"].shape[-1])
    model.attach_context_input(G.N_STATIC)
    if arm == "GCRK+Cin":
        model.attach_gcrk(torch.zeros(F["geo"].shape[-1]))      # geo_center and calibration buffers come from state
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def variants(name: str, model, b: dict, F: dict) -> dict:
    out = {"own": model(b)["P"].numpy()}
    bm = dict(b); bm["geo"] = torch.zeros_like(b["geo"])
    out["mean_geo"] = model(bm)["P"].numpy()
    return out


def main(which: list[str]) -> None:
    torch.set_num_threads(2)
    base = load("v1D")
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(base["fips"])
    for name in which:
        spec = MODELS[name]
        F = base
        acc, used = {}, []
        for s in SEEDS:
            folds = [RUNS / spec["label"].format(s) / f"fold{k:02d}" for k in range(1, 6)]
            if not all((f / "DONE.json").exists() for f in folds):
                continue
            per = {}
            for f in folds:
                ck = torch.load(f / "final.pt", weights_only=False)
                assert ck["arm"] == spec["arm"]
                ref = np.load(f / "outer.npz")
                idx = ref["idx"]
                b = G.make_batch(F, idx, ck["stats"])
                res = variants(name, rebuild(F, spec["arm"], ck["model_state"]), b, F)
                err = float(np.abs(res["own"] - ref["P"]).max())
                assert err < 1e-5, f"{f}: rebuilt model does not reproduce its export ({err:.2e})"
                for k, P in res.items():
                    per.setdefault(k, np.full((n, 144), np.nan, np.float32))[idx] = P
            for k, P in per.items():
                assert not np.isnan(P).any()
                acc.setdefault(k, []).append(P)
            used.append(s)
            print(name, "seed", s, "done", flush=True)
        if not used:
            print(name, "no complete seed"); continue
        np.savez_compressed(OUT / f"counterfactual_{name}.npz", system=base["system"], fips=base["fips"],
                            seeds=np.array(used), **{f"P_{k}": np.mean(v, 0).astype(np.float32) for k, v in acc.items()})
        print(name, "seeds", used, "->", OUT / f"counterfactual_{name}.npz", flush=True)


REGIMES = ["tropical", "winter", "synoptic_wind", "convective", "heavy_rain"]
RWORD = {"all": "All", "tropical": "Tropical", "winter": "Winter", "synoptic_wind": "Synoptic", "convective": "Convective",
         "heavy_rain": "Rain"}
PAIRS = {"GCRK": ("gcrk", "P_own", "P_mean_geo")}


def summarize() -> None:
    """Per route and regime: the change of the design-weighted path MSE when the geography enters (own against the
    counterfactual), the mean absolute change of the predicted peak, and the share of county-events whose predicted
    peak moves by more than 0.5 points. Writes results/v1/counterfactual_summary.json and the macros
    paper_v1/generated/numbers_cf.tex."""
    import json
    F = np.load(ROOT / "data" / "interim" / "panel_v1" / "features_v1D.npz")
    reg, w = F["regime"].astype(str), F["w"].astype(float)
    y, m = F["y"].astype(float), F["m"].astype(float)
    res, mac = {}, []
    for key, (name, own, ref) in PAIRS.items():
        f = OUT / f"counterfactual_{name}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        a, b = z[own].astype(float), z[ref].astype(float)
        d = {}
        for r in ["all"] + REGIMES:
            ii = np.arange(len(reg)) if r == "all" else np.where(reg == r)[0]
            ea = (w[ii, None] * m[ii] * (a[ii] - y[ii]) ** 2).sum(); eb = (w[ii, None] * m[ii] * (b[ii] - y[ii]) ** 2).sum()
            dp = a[ii].max(1) - b[ii].max(1)
            d[r] = dict(mse_change=float(ea / eb - 1), mean_abs_peak_change_pp=float(100 * np.abs(dp).mean()),
                        share_peak_moved_gt_half_pp=float((np.abs(dp) > 0.005).mean()))
            mac.append(rf"\newcommand{{\Cf{key}{RWORD[r]}}}{{{100 * d[r]['mse_change']:+.1f}\%}}".replace("-", "$-$").replace("+", "$+$"))
        mac.append(rf"\newcommand{{\Cf{key}Dpeak}}{{{d['all']['mean_abs_peak_change_pp']:.2f}}}")
        mac.append(rf"\newcommand{{\Cf{key}Moved}}{{{100 * d['all']['share_peak_moved_gt_half_pp']:.0f}\%}}")
        mac.append(rf"\newcommand{{\Cf{key}Seeds}}{{{len(z['seeds'])}}}")
        res[key] = dict(model=name, own=own, counterfactual=ref, seeds=[int(s) for s in z["seeds"]], by_regime=d)
    (EXP / "results" / "v1" / "counterfactual_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    (HERE / "generated").mkdir(exist_ok=True)
    (HERE / "generated" / "numbers_cf.tex").write_text("% generated by paper_v1/counterfactual_v1.py; do not edit\n" + "\n".join(mac) + "\n")
    print(json.dumps({k: {r: round(100 * v["mse_change"], 1) for r, v in x["by_regime"].items()} for k, x in res.items()}))


if __name__ == "__main__":
    args = sys.argv[1:] or ["gcrk", "summary"]
    if [a for a in args if a != "summary"]:
        main([a for a in args if a != "summary"])
    if "summary" in args:
        summarize()
