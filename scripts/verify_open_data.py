"""Verify a clone of the open-data branch.

Two checks, both offline:

1. every file listed in `data/SHA256SUMS.txt` is present and unmodified;
2. the published panels reproduce the archived onset audit
   (`results/panel_onset_audit.json`) exactly, under the audited definition:
   15-minute resolution, lead-in = every step before the storm day, counties with
   no observed lead-in dropped, interruption threshold 0.01.

    python scripts/verify_open_data.py
"""
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def check_sums() -> int:
    sums = ROOT / "data/SHA256SUMS.txt"
    if not sums.exists():
        print("data/SHA256SUMS.txt not present, skipping the checksum check\n")
        return 0
    bad, n = [], 0
    for line in sums.read_text().splitlines():
        h, _, rel = line.partition("  ")
        if not rel:
            continue
        n += 1
        f = ROOT / rel
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != h:
            bad.append(rel)
    print(f"checksums: {n} files, " + ("all match" if not bad else f"{len(bad)} MISMATCH e.g. {bad[:3]}"))
    return len(bad)


def check_onset() -> int:
    audit = ROOT / "results/panel_onset_audit.json"
    if not audit.exists():
        print("results/panel_onset_audit.json not on this branch, skipping the onset check")
        return 0
    days = json.loads(audit.read_text())["days"]
    print(f"\n{'day':<13}{'counties':>9}{'interrupted':>12}{'published':>11}{'archived':>10}")
    ok = n = 0
    for r in days:
        day = r["event_day"]
        f = ROOT / f"data/interim/panel_{day}.npz"
        if not f.exists():
            continue
        n += 1
        z = np.load(f, allow_pickle=True)
        y, obs = z["y"], z["observed"]
        ts = pd.Index(pd.to_datetime([str(t) for t in z["ts"]]))
        lead = ts < pd.Timestamp(day)
        with np.errstate(all="ignore"):
            pre = np.nanmedian(np.where(obs[:, lead], y[:, lead], np.nan), axis=1)
            ever = np.nanmax(np.where(obs, y, np.nan), axis=1)
        interrupted = np.nan_to_num(ever, nan=-1) >= 0.01
        v = pre[interrupted]
        v = v[np.isfinite(v)]
        share = float((v <= 0).sum() / max(len(v), 1))
        same = abs(share - r["frac_typ_zero"]) < 1e-9
        ok += same
        print(f"{day:<13}{len(z['fips']):>9}{int(interrupted.sum()):>12}"
              f"{share * 100:>10.1f}%{r['frac_typ_zero'] * 100:>9.1f}%{'  ok' if same else '  MISMATCH'}")
    print(f"\n{ok}/{n} panels reproduce the archived onset statistic bit for bit")
    return n - ok


if __name__ == "__main__":
    raise SystemExit(1 if (check_sums() + check_onset()) else 0)
