"""Step 4 of DATASET_DESIGN v1 (section 12) for the development tranche D.
(a) county gates of every D system from pre-window data only (sample_counties.gate_domain); the G3 lookback switch of
    section 6 (30 -> 90 days if G3 removes more than 5% of D's otherwise gated domain counties) is decided here, on D,
    and then applies to every tranche; the gate audit (drop rates by stratum) is written;
(b) one system at a time, so that at most one CONUS ERA5 window is on disk: fetch ERA5, hazard index and county sample
    (sample_counties.sample_one), panels and host inputs, cropped archive, raw files deleted (build_panels.build_one).
Outputs: data/interim/panel_v1/{gated_D.parquet, county_sample_D.parquet, panels/}; data_provenance/{gate_audit_D.json,
panels_v1D.jsonl}."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_frame as BF  # noqa: E402
import build_panels as BPV  # noqa: E402
import gates as GT  # noqa: E402
import sample_counties as SC  # noqa: E402
from frame_common import EXP, INTERIM, OUT  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates-only", action="store_true", help="step (a) only")
    ap.add_argument("--part", default="0/1", help="i/n: step (b) for the D systems whose rank mod n == i")
    ap.add_argument("--combine", action="store_true", help="write county_sample_D.parquet from the per-system files")
    a = ap.parse_args()
    pi, pn = map(int, a.part.split("/"))
    sy = pd.read_parquet(OUT / "systems.parquet").set_index("system")
    sc = pd.read_parquet(OUT / "system_counties.parquet")
    dr = pd.read_parquet(OUT / "draws.parquet")
    D = sorted(dr[dr.tranche == "D"].system)
    st = GT.FixedGates(BF.load_exclusions("data_provenance/operator_exclusions.csv"))
    cmask = GT.CMask()
    f = OUT / "gated_D.parquet"
    if not f.exists():
        g = pd.concat([SC.gate_domain(s, sy, sc, st, cmask) for s in D], ignore_index=True)
        rec = SC.gate_audit(g, "D")
        print("gate audit (30-day lookback):", json.dumps(rec), flush=True)
        if rec["g3_drop_share_of_otherwise_gated"] > 0.05:          # section 6: the lookback switch, decided on D
            GT.LOOKBACK_D = 90
            g = pd.concat([SC.gate_domain(s, sy, sc, st, cmask) for s in D], ignore_index=True)
            rec = SC.gate_audit(g, "D")
            print("gate audit (90-day lookback):", json.dumps(rec), flush=True)
        g.to_parquet(f, index=False)
    if a.gates_only:
        return
    g = pd.read_parquet(f)
    GT.LOOKBACK_D = json.loads((EXP / "data_provenance" / "gate_audit_D.json").read_text())["lookback_days"]
    W = SC.pop_weights()
    w = BPV.area_weights(); nn = BPV.neighbours()
    denom = pd.read_parquet(INTERIM / "eaglei_county_customers_2024.parquet")["customers"]
    parts = []
    cs_dir = OUT / "county_sample_D"
    cs_dir.mkdir(exist_ok=True)
    if a.combine:
        pd.concat([pd.read_parquet(cs_dir / f"{s}.parquet") for s in D], ignore_index=True).to_parquet(
            OUT / "county_sample_D.parquet", index=False)
        return
    for rank, s in enumerate(D):
        if rank % pn != pi:
            continue
        pf = cs_dir / f"{s}.parquet"
        if not pf.exists():
            if not (BPV.PANELS / f"panel216w_{s}.npz").exists():
                BPV.FE.fetch(s, pd.Timestamp(sy.at[s, "window_start"]))
            SC.sample_one(s, g[g.system == s], W).to_parquet(pf, index=False)
        d = pd.read_parquet(pf)
        parts.append(d)
        if not (BPV.PANELS / f"panel216w_{s}.npz").exists():
            rec = BPV.build_one(s, sy, dr, d, w, nn, denom, cmask)
            with open(EXP / "data_provenance" / "panels_v1D.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
    if pn == 1:
        pd.concat(parts, ignore_index=True).to_parquet(OUT / "county_sample_D.parquet", index=False)


if __name__ == "__main__":
    main()
