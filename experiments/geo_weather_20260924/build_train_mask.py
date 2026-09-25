"""Training weights from EAGLE-I artefact flags (LITERATURE_preprocessing section 6, change 4). The evaluation
targets and masks stay as they are (program.md rule 4); only the training loss drops the flagged hours.

EAGLE-I stores no zero rows, and scraper timeouts cluster in storms (Brelsford et al. 2024), so a storm-time
drop that returns within an hour is more likely a collection artefact than a restoration followed by a new
failure. Flags on the hourly outage fraction y (observed hours only; forecast hours 72..215):
  dip      y_t < 0.5 min(y_t-1, y_t+1) and min(y_t-1, y_t+1) >= 0.005 (a one-hour V)
  spike    y_t > 2 max(y_t-1, y_t+1) + 0.005 (a one-hour inverted V; includes jumps above the moving average)
  plateau  the same non-zero value for >= 96 consecutive observed hours (a stale map: EAGLE-I's QA flags counts
           unchanged for more than four days)
  over     y_t >= 0.999 (the fraction reached the denominator: a denominator error more often than a blackout)
Output: data/interim/geo_weather/train_mask_e3.npz with m_train [U,144] (float32, <= m) and the flag counts;
provenance data_provenance/train_mask_e3.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEAT = ROOT / "data" / "interim" / "open_gcrk" / "features_e3r2.npz"
OUT = ROOT / "data" / "interim" / "geo_weather"
ORIGIN = 72


def flags(y: np.ndarray, obs: np.ndarray) -> dict:
    """y, obs [U, 216] -> boolean flags [U, 216]."""
    yv = np.where(obs, y, np.nan)
    prev, nxt = np.roll(yv, 1, 1), np.roll(yv, -1, 1)
    prev[:, 0] = np.nan; nxt[:, -1] = np.nan
    with np.errstate(invalid="ignore"):
        lo = np.fmin(prev, nxt); hi = np.fmax(prev, nxt)
        both = np.isfinite(prev) & np.isfinite(nxt)
        dip = both & (yv < 0.5 * lo) & (lo >= 0.005)
        spike = both & (yv > 2 * hi + 0.005)
        over = obs & (yv >= 0.999)
    plateau = np.zeros_like(obs)
    for u in range(y.shape[0]):
        v, o = yv[u], obs[u]
        s = 0
        while s < len(v):
            if not o[s] or not (v[s] > 0):
                s += 1; continue
            e = s
            while e + 1 < len(v) and o[e + 1] and v[e + 1] == v[s]:
                e += 1
            if e - s + 1 >= 96:
                plateau[u, s:e + 1] = True
            s = e + 1
    return dict(dip=dip, spike=spike, plateau=plateau, over=over)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(FEAT.relative_to(ROOT)))
    ap.add_argument("--tag", default="e3")
    a = ap.parse_args()
    F = np.load(ROOT / a.feat)
    y, obs, m = F["y_full"].astype(np.float64), F["obs_full"], F["m"].astype(np.float32)
    fl = flags(y, obs)
    bad = np.zeros_like(obs)
    for v in fl.values():
        bad |= v
    m_train = m * (~bad[:, ORIGIN:]).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"train_mask_{a.tag}.npz", m_train=m_train, **{f"flag_{k}": v for k, v in fl.items()})
    ev = F["event"].astype(str)
    n_obs = float(m.sum())
    meta = dict(forecast_hours_observed=int(n_obs), flagged_share=float(1 - m_train.sum() / n_obs),
                by_flag={k: int(v[:, ORIGIN:][m > 0].sum()) for k, v in fl.items()},
                units_with_any_flag=int((bad[:, ORIGIN:] & (m > 0)).any(1).sum()),
                flagged_share_by_event={e: round(float(1 - m_train[ev == e].sum() / max(m[ev == e].sum(), 1)), 5)
                                        for e in sorted(set(ev))},
                # squared-error weight of the flagged hours under a zero forecast (how much of the target
                # variance they carry)
                flagged_share_of_sum_y2=float(((np.nan_to_num(y[:, ORIGIN:]) ** 2) * m * bad[:, ORIGIN:]).sum() / ((np.nan_to_num(y[:, ORIGIN:]) ** 2) * m).sum()))
    (HERE / "data_provenance" / f"train_mask_{a.tag}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
