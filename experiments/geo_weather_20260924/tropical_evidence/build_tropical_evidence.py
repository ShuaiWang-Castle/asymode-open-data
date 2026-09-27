"""Tropical evidence: where the host forecasts unseen tropical-cyclone systems better than the all-zero forecast, on the
development tranche of the designed panel (DATASET_DESIGN v1), host W+Cin seed 0, five event folds (held-out systems).
Tables: per system; by stratum (S1 warned, S2 advisory/watch, S3 ring); by national tercile of five geography axes
within the tropical regime; the weather and geography of the tropical county-events. Design-weighted MSE throughout
(weights w of DATASET_DESIGN section 5.4). Development tranche only; single seed; descriptive, not a registered test.
Output: tropical_evidence/tables.json and the markdown tables printed to stdout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
from evaluate_v1 import per_unit, predictions  # noqa: E402

OUT = ROOT / "data" / "interim" / "panel_v1"
AXES = [("canopy", "tree canopy %"), ("coast_km", "distance to coast km"), ("poorly_drained", "poorly drained share"),
        ("relief_sd", "relief (SD of elevation) m"), ("log_cust_density", "log customers per km2")]


def rel(a, b):
    return float(a / b - 1) if b > 0 else float("nan")


def main() -> None:
    z = np.load(OUT / "features_v1D.npz")
    y, m, w = z["y"].astype(float), z["m"].astype(float), z["w"].astype(float)
    reg, sysv, st = z["regime"].astype(str), z["system"].astype(str), z["stratum"].astype(str)
    P = predictions(["v1_host_s0"], len(y))
    sh, N = per_unit(P, y, m); sz, _ = per_unit(np.zeros_like(y), y, m)
    sy = pd.read_parquet(OUT / "systems.parquet").set_index("system")
    ax = pd.read_parquet(OUT / "county_axes.parquet").set_index("fips")
    d = pd.DataFrame(dict(system=sysv, regime=reg, stratum=st, fips=z["fips"].astype(str), w=w, sh=sh, sz=sz, N=N,
                          peak=np.nanmax(np.where(m > 0, y, np.nan), 1)))
    d = d.join(ax, on="fips")
    t = d[d.regime == "tropical"].copy()

    def mse(g):
        den = (g.w * g.N).sum()
        return pd.Series({"county_events": len(g), "zero_MSE": (g.w * g.sz).sum() / den, "host_MSE": (g.w * g.sh).sum() / den,
                          "host_vs_zero": rel((g.w * g.sh).sum(), (g.w * g.sz).sum()), "peak_ge_10pct": float((g.peak >= 0.10).mean())})
    per_sys = t.groupby("system").apply(mse)
    per_sys["storm"] = [sy.at[s, "tc_name"] for s in per_sys.index]
    per_sys["origin"] = [str(sy.at[s, "origin"])[:10] for s in per_sys.index]
    per_sys["region"] = [sy.at[s, "region"] for s in per_sys.index]
    per_sys = per_sys.sort_values("origin")
    by_st = t.groupby("stratum").apply(mse)
    ref = pd.read_parquet(OUT / "county_axes.parquet")
    by_geo = {}
    for a, lab in AXES:
        cuts = np.nanquantile(ref[a], [1 / 3, 2 / 3])
        t["_t"] = np.digitize(t[a], cuts)
        g = t.groupby("_t").apply(mse)
        g.index = [f"lower third (< {cuts[0]:.3g})", f"middle third", f"upper third (> {cuts[1]:.3g})"][:len(g)]
        by_geo[lab] = g
    all_reg = d.groupby("regime").apply(mse)
    res = dict(per_regime=all_reg.round(6).to_dict(orient="index"), per_system=per_sys.round(6).to_dict(orient="index"),
               by_stratum=by_st.round(6).to_dict(orient="index"),
               by_geography={k: v.round(6).to_dict(orient="index") for k, v in by_geo.items()},
               note="host W+Cin seed 0, five event folds of the development tranche; design-weighted MSE; single seed; "
                    "descriptive, not a registered test")
    (HERE / "tables.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    def fmt(df, cols):
        def cell(v):
            if isinstance(v, (float, np.floating)):
                return f"{v:+.1%}" if abs(v) < 5 and v == v and df is not None and False else f"{v:.3g}"
            return str(v)
        head = "| | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
        rows = ["| " + str(i) + " | " + " | ".join(cell(df.at[i, c]) for c in cols) + " |" for i in df.index]
        return head + "\n".join(rows)
    print("## per regime\n", fmt(all_reg, ["county_events", "zero_MSE", "host_MSE", "host_vs_zero"]))
    print("\n## tropical systems\n", fmt(per_sys, ["storm", "origin", "region", "county_events", "zero_MSE", "host_MSE", "host_vs_zero", "peak_ge_10pct"]))
    print("\n## tropical by stratum\n", fmt(by_st, ["county_events", "zero_MSE", "host_MSE", "host_vs_zero"]))
    for k, v in by_geo.items():
        print(f"\n## tropical by {k}\n", fmt(v, ["county_events", "zero_MSE", "host_MSE", "host_vs_zero"]))


if __name__ == "__main__":
    main()
