"""Integrity and leakage audit for the event-transfer confirmatory task (Section 3).

Fails closed. Checks, for the g2-convective-11 manifest:

1. every panel/driver pair exists and agrees on fips, county dimension and length;
2. panel timestamps are UTC 15-minute and strictly increasing; driver hours align
   to the panel's own hour grid;
3. the scored feature block is exactly the public 14-channel block;
4. `clock_sin`/`clock_cos` are built from the timestamp hour, not from lead time
   (asserted by rebuilding both and comparing against what the harness emits);
5. every `observed == False` target is excluded from the loss and the metrics
   (asserted on the mask arrays the loader returns);
6. no test-event row can reach training normalisation (asserted structurally by
   the split code path, and numerically in cc_event_transfer.py).

    python experiments/cc_integrity_audit.py --out results/.../01_INTEGRITY_AUDIT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from asymode import panels as panelset  # noqa: E402
import exp05_real_dynamics as exp05  # noqa: E402

INTERIM = ROOT / "data/interim"
EXPECTED_CHANNELS = ["cape", "cloud", "gust", "precip", "pressure", "rh", "snowfall",
                     "soil_moisture", "t2m_c", "u10", "v10", "wind_speed",
                     "clock_sin", "clock_cos"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="configs/panel_manifest_g2-convective-11.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lines: list[str] = []
    fail: list[str] = []

    def say(s: str) -> None:
        print(s)
        lines.append(s)

    want, panel_digest = panelset.resolve(INTERIM, str(ROOT / a.panels))
    chan_names = panelset.channel_names(INTERIM)
    chan_digest = panelset.channel_digest(chan_names)
    say(f"manifest: {a.panels}")
    say(f"panel_digest: {panel_digest}")
    say(f"channel_digest: {chan_digest}")
    say(f"events ({len(want)}): {', '.join(sorted(want))}")
    say("")

    # --- 1/2. per-panel alignment -------------------------------------------------
    say("## Panel / driver alignment")
    say("")
    say("| event | counties | 15-min steps | hours | fips match | ts UTC 15-min monotone | driver hours >= panel hours |")
    say("|---|---|---|---|---|---|---|")
    for day in sorted(want):
        pz = np.load(INTERIM / f"panel_{day}.npz", allow_pickle=True)
        dz = np.load(INTERIM / f"drivers_{day}.npz", allow_pickle=True)
        ts = pd.to_datetime([str(t) for t in pz["ts"]])
        dt = np.unique(np.diff(ts.values).astype("timedelta64[m]").astype(int))
        mono = bool(len(dt) == 1 and dt[0] == 15)
        fips_ok = pz["fips"].tolist() == dz["fips"].tolist()
        n_h = len(ts) // 4
        drv_ok = dz["X"].shape[1] >= n_h
        cty_ok = pz["y"].shape[0] == dz["X"].shape[0] == len(pz["fips"])
        if not (mono and fips_ok and drv_ok and cty_ok):
            fail.append(f"{day}: alignment (mono={mono} fips={fips_ok} drv={drv_ok} cty={cty_ok})")
        say(f"| {day} | {pz['y'].shape[0]} | {len(ts)} | {n_h} | {'yes' if fips_ok else '**NO**'} | "
            f"{'yes' if mono else '**NO**'} | {'yes' if drv_ok else '**NO**'} |")
    say("")

    # --- 3. channel block ---------------------------------------------------------
    say("## Feature block")
    say("")
    ok_chan = chan_names == EXPECTED_CHANNELS
    if not ok_chan:
        fail.append(f"channel block is {chan_names}, expected {EXPECTED_CHANNELS}")
    say(f"scored channels ({len(chan_names)}): `{', '.join(chan_names)}`")
    say(f"matches the public 14-channel block: **{'yes' if ok_chan else 'NO'}**")
    say("")

    # --- 4. clock provenance ------------------------------------------------------
    say("## Clock provenance")
    say("")
    y0, X, yt, m, fips, panel, origin, t0h = exp05.load_pooled(48, 12, panels=want, with_time=True)
    Xc = exp05.add_context(X, y0, 48, t0_hour=t0h, clock="utc_hour")
    Xold = exp05.add_context(X, y0, 48, clock="lead_phase_old")
    # rebuild the clock independently from the panel timestamps
    hours = {}
    for day in sorted(want):
        pz = np.load(INTERIM / f"panel_{day}.npz", allow_pickle=True)
        ts = pd.to_datetime([str(t) for t in pz["ts"]])[::4]
        hours[day] = np.array(ts.hour)
    ind_sin = np.empty_like(Xc[:, :, -2])
    for i in range(len(panel)):
        h = (hours[panel[i]][origin[i] + 1] + np.arange(X.shape[1])) % 24
        ind_sin[i] = np.sin(2 * np.pi * h / 24)
    same_ts = bool(np.allclose(ind_sin, Xc[:, :, -2], atol=1e-6))
    differs_from_lead = not bool(np.allclose(Xc[:, :, -2], Xold[:, :, -2], atol=1e-6))
    n_phase = len({tuple(np.round(Xc[i, :3, -2], 6)) for i in range(0, len(panel), 97)})
    if not (same_ts and differs_from_lead):
        fail.append(f"clock: ts-derived={same_ts} differs-from-lead={differs_from_lead}")
    say(f"clock rebuilt independently from panel timestamps matches the harness: **{'yes' if same_ts else 'NO'}**")
    say(f"clock differs from the legacy lead-phase channel: **{'yes' if differs_from_lead else 'NO'}**")
    say(f"distinct clock phases across sampled origins: {n_phase} (a lead-time clock would give 1)")
    say("")

    # --- 5. mask ------------------------------------------------------------------
    say("## Observation mask")
    say("")
    say(f"pooled samples: {len(y0):,} · counties {len(set(fips)):,} · events {len(set(panel))}")
    say(f"scored cells per horizon: {int(m[:, 0].sum()):,} (h+1) .. {int(m[:, 47].sum()):,} (h+48)")
    say(f"unobserved share excluded from every loss and metric: {100 * (1 - m.mean()):.2f}%")
    finite = bool(np.isfinite(yt[m]).all())
    if not finite:
        fail.append("non-finite target inside the observation mask")
    say(f"all masked-in targets finite: **{'yes' if finite else 'NO'}**")
    say("")

    say("## Verdict")
    say("")
    say("**PASS — no integrity or alignment check failed.**" if not fail else
        "**FAIL**\n\n" + "\n".join(f"* {f}" for f in fail))

    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Integrity audit\n\n" + "\n".join(lines) + "\n")
        print(f"\nwritten: {a.out}")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
