"""Placebo for the artefact training mask (build_train_mask.py): drop as many forecast hours from the training loss
as the flags do, in the same event and the same outage-level bin (quantiles of y over all observed forecast hours, finer
at the top: 0.1, ..., 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999), chosen at random among unflagged observed hours of any unit (seed 20260925). The placebo removes the same
loss weight at the same outage levels as the flags, without their content.
Output: data/interim/geo_weather/train_mask_<tag>_placebo.npz (m_train)."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / "interim" / "geo_weather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/interim/open_gcrk/features_e3r2.npz")
    ap.add_argument("--tag", default="e3")
    a = ap.parse_args()
    F = np.load(ROOT / a.feat)
    y, m = np.nan_to_num(F["y"]).astype(np.float64), F["m"].astype(np.float32)
    real = np.load(OUT / f"train_mask_{a.tag}.npz")["m_train"]
    flagged = (m > 0) & (real == 0)
    q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]
    edges = np.quantile(y[m > 0], q)                    # finer at the top, where the flagged hours sit
    dec = np.digitize(y, edges)
    rng = np.random.default_rng(20260925)
    ev = F["event"].astype(str)
    plc = m.copy(); short = 0
    evu = np.broadcast_to(ev[:, None], m.shape)
    for e in np.unique(ev):                     # same event and outage decile, any unit: matches count and level
        for d in np.unique(dec[flagged & (evu == e)]):
            need = int((flagged & (evu == e) & (dec == d)).sum())
            pool = np.flatnonzero((m > 0) & ~flagged & (evu == e) & (dec == d))
            if len(pool) < need:
                short += need - len(pool)
            pick = rng.choice(pool, min(need, len(pool)), replace=False)
            plc.flat[pick] = 0.0
    np.savez_compressed(OUT / f"train_mask_{a.tag}_placebo.npz", m_train=plc)
    meta = dict(dropped_real=int(flagged.sum()), dropped_placebo=int(((m > 0) & (plc == 0)).sum()), decile_shortfall=short,
                sum_y2_share_real=float((y ** 2 * flagged).sum() / (y ** 2 * m).sum()),
                sum_y2_share_placebo=float((y ** 2 * ((m > 0) & (plc == 0))).sum() / (y ** 2 * m).sum()))
    (HERE / "data_provenance" / f"train_mask_{a.tag}_placebo.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(meta)


if __name__ == "__main__":
    main()
