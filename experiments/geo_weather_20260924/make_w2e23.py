"""H2b inputs (PREREG_W2 amendment 6): W2e restricted to unit_ok = True (the 23 events kept by the HRRR coverage rule),
with event-grouped splits (five folds, events in date order, fold = rank mod 5), and the pop / hrrr_coarse hazard
features for the same units (missing HRRR hours as zeros)."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GW = ROOT / "data" / "interim" / "geo_weather"
F = np.load(GW / "features_w2e.npz")
hc = np.load(GW / "eih_w2e_hrrr_coarse.npz")
ok = hc["unit_ok"].astype(bool)
assert np.array_equal(hc["fips"], F["fips"]) and np.array_equal(hc["event"], F["event"])
sub = {k: F[k][ok] if F[k].shape[:1] == ok.shape else F[k] for k in F.files}
np.savez_compressed(GW / "features_w2e23.npz", **sub)
for v in ("pop", "hrrr_coarse"):
    z = np.load(GW / f"eih_w2e_{v}.npz")
    phi = np.nan_to_num(z["phi"][ok].astype(np.float32), nan=0.0).astype(np.float16)
    np.savez(GW / f"eih_w2e23_{v}.npz", phi=phi, names=z["names"], fips=z["fips"][ok], event=z["event"][ok])
ev = sub["event"].astype(str); allu = np.arange(len(ev)); events = sorted(set(ev))
event = {}
for k in range(5):
    out_ev = [e for i, e in enumerate(events) if i % 5 == k]
    held, dev = allu[np.isin(ev, out_ev)], allu[~np.isin(ev, out_ev)]
    event[str(k + 1)] = dict(outer=held.tolist(), dev=dev.tolist(), outer_events=out_ev)
(HERE / "splits_w2e23.json").write_text(json.dumps(dict(n_units=len(ev), event=event, main=event)) + "\n")
print("units", len(ev), "events", len(events), {k: (v["outer_events"], len(v["outer"])) for k, v in event.items()})
