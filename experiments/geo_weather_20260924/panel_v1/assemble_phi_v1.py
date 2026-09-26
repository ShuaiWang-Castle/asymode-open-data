"""Hazard pathway inputs of the designed panel: the hazard dictionary v2 (build_hazard_v2.py) of one weather source,
in the unit order of features_v1<tranche>.npz, forecast hours only, as the hazard arms of screen.py read them
(F['phi'] [U, 144, K]; screen.py --phi <variant> loads data/interim/geo_weather/eih_<PANEL><variant>.npz).
Output: data/interim/geo_weather/eih_v1D_<source>2.npz (phi float16, names, fips, event).
usage: python assemble_phi_v1.py --source era5 [--tranche D]"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from frame_common import EXP, INTERIM, OUT, PREFIX_H

SRC = OUT / "hazard_v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["era5", "hrrr"], required=True)
    ap.add_argument("--tranche", default="D")
    a = ap.parse_args()
    F = np.load(OUT / f"features_v1{a.tranche}.npz")
    sysv, fips = F["system"].astype(str), F["fips"].astype(str)
    names, phi = None, None
    for s in sorted(set(sysv)):
        z = np.load(SRC / f"{a.source}_{s}.npz")
        nm = list(z["names"].astype(str))
        if names is None:
            names = nm
            phi = np.zeros((len(sysv), 216 - PREFIX_H, len(names)), np.float16)
        assert nm == names, f"{s}: feature names differ"
        pos = {f: i for i, f in enumerate(z["fips"].astype(str))}
        idx = np.where(sysv == s)[0]
        phi[idx] = z["X"][[pos[f] for f in fips[idx]], PREFIX_H:, :]
    X = phi.astype(np.float32)
    assert np.isfinite(X).all(), "non-finite hazard features"
    assert (X >= -1e-3).all(), "hazard features must be non-negative (competing-hazard entry)"
    dst = INTERIM / "geo_weather" / f"eih_v1{a.tranche}_{a.source}2.npz"
    np.savez(dst, phi=phi, names=np.array(names), fips=fips, event=sysv)
    rec = dict(file=str(dst.relative_to(INTERIM.parents[1])), sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
               units=int(len(sysv)), features=names)
    (EXP / "data_provenance" / f"eih_v1{a.tranche}_{a.source}2.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec)[:400])


if __name__ == "__main__":
    main()
