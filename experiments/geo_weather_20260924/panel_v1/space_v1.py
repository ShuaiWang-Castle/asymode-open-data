"""Neighbour table of the designed panel for the spatio-temporal GCRK (asymode.gcrk._ResponseST).

Neighbours of a county-event are the other sampled county-events of the same weather system (event folds hold out whole
systems, so a county and its neighbours are always on the same side of a split), at most K = 8, the nearest by
great-circle distance between Census 2020 internal points, within 150 km.

  wd[i, k]      exp(-distance / 50 km), static
  up[i, t, k]   upwind share at hour t: the wind at the pair's midpoint (the mean of the two counties' ERA5 10 m winds)
                blowing from neighbour k towards county i, cos(angle) clipped at 0 and weighted by wd, normalised over k,
                times min(1, wind speed at i / 10 m/s); 0 when no neighbour is upwind
Only static geography and the weather of the window enter; no outcome. Writes
data/interim/panel_v1/space_v1D.npz (nbr [N, K] int32 with -1 for none, wd [N, K], up [N, 216, K] float16, dist_km)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FEAT = ROOT / "data" / "interim" / "panel_v1" / "features_v1D.npz"
GAZ = ROOT / "data" / "raw" / "census" / "2020_Gaz_counties_national.txt"
OUT = ROOT / "data" / "interim" / "panel_v1" / "space_v1D.npz"
K, RMAX_KM, SCALE_KM, V_REF = 8, 150.0, 50.0, 10.0
R_EARTH = 6371.0


def main() -> None:
    z = np.load(FEAT)
    fips, sysv = z["fips"].astype(str), z["system"].astype(str)
    names = list(z["weather_channels"].astype(str))
    xu = z["xu"]
    u10, v10 = xu[:, :, names.index("u10")].astype(np.float64), xu[:, :, names.index("v10")].astype(np.float64)
    gz = pd.read_csv(GAZ, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    gz.columns = [c.strip() for c in gz.columns]
    pos = {g.zfill(5): (la, lo) for g, la, lo in zip(gz.GEOID, gz.INTPTLAT, gz.INTPTLONG)}
    missing = sorted({f for f in fips if f not in pos})
    assert not missing, f"no internal point for {missing[:5]}"
    lat = np.radians([pos[f][0] for f in fips]); lon = np.radians([pos[f][1] for f in fips])
    N, T = len(fips), xu.shape[1]
    nbr = np.full((N, K), -1, np.int32); wd = np.zeros((N, K), np.float32); dist = np.full((N, K), np.nan, np.float32)
    east = np.zeros((N, K)); north = np.zeros((N, K))              # unit vector from the neighbour towards county i
    for S in np.unique(sysv):
        ii = np.where(sysv == S)[0]
        la, lo = lat[ii], lon[ii]
        dla = la[:, None] - la[None, :]; dlo = lo[:, None] - lo[None, :]
        h = np.sin(dla / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin(dlo / 2) ** 2
        D = 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        np.fill_diagonal(D, np.inf)
        for a, i in enumerate(ii):
            order = [b for b in np.argsort(D[a]) if D[a, b] <= RMAX_KM][:K]
            for k, b in enumerate(order):
                nbr[i, k], dist[i, k] = ii[b], D[a, b]
                wd[i, k] = np.exp(-D[a, b] / SCALE_KM)
                x = R_EARTH * np.cos((la[a] + la[b]) / 2) * (lo[a] - lo[b]); y = R_EARTH * (la[a] - la[b])
                r = np.hypot(x, y)
                east[i, k], north[i, k] = x / r, y / r
    ok = nbr >= 0
    j = np.where(ok, nbr, 0)
    up = np.zeros((N, T, K), np.float32)
    for k in range(K):
        jk = j[:, k]
        ue = 0.5 * (u10 + u10[jk]); vn = 0.5 * (v10 + v10[jk])      # midpoint wind [N, T]
        sp = np.hypot(ue, vn)
        cos = (ue * east[:, k, None] + vn * north[:, k, None]) / np.maximum(sp, 1e-6)
        up[:, :, k] = np.where(ok[:, k, None], wd[:, k, None] * np.clip(cos, 0, None), 0.0)
    tot = up.sum(-1, keepdims=True)
    speed = np.hypot(u10, v10)
    up = np.where(tot > 0, up / np.maximum(tot, 1e-12), 0.0) * np.minimum(1.0, speed / V_REF)[:, :, None]
    np.savez_compressed(OUT, fips=fips, system=sysv, nbr=nbr, wd=wd, up=up.astype(np.float16), dist_km=dist,
                        params=np.array([K, RMAX_KM, SCALE_KM, V_REF]))
    n_nb = ok.sum(1)
    print(f"{N} county-events; neighbours per county-event: mean {n_nb.mean():.2f}, none {np.mean(n_nb == 0):.3f}, "
          f"all {K} {np.mean(n_nb == K):.3f}; median distance {np.nanmedian(dist):.0f} km; "
          f"upwind share > 0 in {np.mean(up.sum(-1) > 0):.3f} of hours; mean upwind row sum {up.sum(-1).mean():.3f}")


if __name__ == "__main__":
    main()
