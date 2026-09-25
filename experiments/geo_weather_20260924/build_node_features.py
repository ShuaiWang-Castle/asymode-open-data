"""Node features for the local fading-memory operator (DESIGN v1): fixed physical downscaling D, fixed
non-negative calm-zero forcing features, and a fixed log-spaced memory bank. No learned parameter here.

D (constants from the literature, contrib/MECHANISMS_structure.md section 0):
  T_k  = T_cell - Gamma(month) dz_k / 1000, Gamma = monthly near-surface lapse rate (Kunkel 1989, as tabulated
         by Liston & Elder 2006), dz_k = node elevation - ERA5 model orography of the cell
  Td_k = mean of (Td_cell - Gamma_d(month) dz_k / 1000, Gamma_d = lambda(month) c / b, Liston & Elder 2006) and
         (dew point of T_k at the cell's relative humidity); Td_k <= T_k
  Tw_k = Stull (2011) wet-bulb temperature of (T_k, RH_k)
  snow fraction = sigmoid((1.0 - Tw_k) / 0.7)  (T50 1.0 C: Jennings et al. 2018; scale 0.7 C: Dai 2008)
  gust, precipitation, soil water, CAPE: the node's cell mixture, not downscaled
Trigger features (instantaneous, zero in calm): gust exceedances [G - 10]+, [G - 15]+, [G - 20]+ (per 5 m/s),
  liquid precipitation, precipitation x five hat functions of Tw centred at -3, -1.5, 0, 1.5, 3 C (half-width
  1.5 C), convective log1p(CAPE / 1000 x precipitation).
Load features: 1, soil water (per 0.3 m3/m3), and exact exponential filters (unit gain, tau = 3, 12, 48, 192 h,
  run from the first prefix hour) of liquid precipitation, the five precipitation-by-Tw features and [G - 15]+.
Susceptibility attributes (static per node): dz / 100 m, canopy, canopy x leaf-on (May-Oct), forest,
  developed, poorly drained, tpi / 10 m, slope / 10 deg.

Variants (REVIEW_formal F2): quad (nodes under their own cell mixture), pooled (the county's nodes under the
population-weighted county weather), mean (one node at the county mean), other (another county's nodes from
the same relief stratum under this county's pooled weather; seeded permutation within quintiles of the
customer-weighted spread of dz).
Output: data/interim/geo_weather/node_feat_<variant>.npz: trig [U,K,144,NT], load [U,K,144,NL] (float16),
attr [U,K,NA], rho [U,K]; unit order of features_e3r2.npz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / "interim" / "geo_weather"
GAMMA = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7])        # K/km, Jan..Dec
LAMBDA = np.array([.41, .42, .40, .39, .38, .36, .33, .33, .36, .37, .40, .40])       # 1/km, Jan..Dec
B_M, C_M = 17.502, 240.97
GAMMA_D = LAMBDA * C_M / B_M                                                          # K/km
TAUS = (3.0, 12.0, 48.0, 192.0)
HATS = (-3.0, -1.5, 0.0, 1.5, 3.0)
ORIGIN, T = 72, 216
TRIG = ["gust_x10", "gust_x15", "gust_x20", "rain"] + [f"p_tw{c:+.1f}" for c in HATS] + ["convective"]
FILT = ["rain"] + [f"p_tw{c:+.1f}" for c in HATS] + ["gust_x15"]
LOAD = ["one", "soil"] + [f"{f}_tau{int(t)}" for t in TAUS for f in FILT]
ATTR = ["dz100", "canopy", "canopy_leafon", "forest", "developed", "wet", "tpi10", "slope10"]
SEED = 20260925


def wet_bulb(t, rh):
    rh = np.clip(rh, 1.0, 100.0)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def dew_from_rh(t, rh):
    a, b = 17.625, 243.04
    g = np.log(np.clip(rh, 1e-3, 100.0) / 100.0) + a * t / (b + t)
    return b * g / (a - g)


def rh_from(t, td):
    a, b = 17.625, 243.04
    return 100.0 * np.exp(a * td / (b + td) - a * t / (b + t))


def exp_filter(x: np.ndarray, tau: float) -> np.ndarray:
    """Unit-gain exact exponential filter along the last axis: m_t = e m_{t-1} + (1 - e) x_t, m_{-1} = 0."""
    e = np.exp(-1.0 / tau)
    out = np.empty_like(x)
    acc = np.zeros(x.shape[:-1], x.dtype)
    for s in range(x.shape[-1]):
        acc = e * acc + (1 - e) * x[..., s]
        out[..., s] = acc
    return out


def features(w: np.ndarray, ch: list[str], attr: np.ndarray, an: list[str], month: np.ndarray):
    """w [U,K,T,C] physical units; attr [U,K,A]; month [U] 1..12 -> trig [U,K,144,NT], load [U,K,144,NL], g [U,K,NA]."""
    c = {n: w[..., ch.index(n)].astype(np.float32) for n in ("t2m_c", "d2m_c", "rh", "precip", "gust", "soil_moisture", "cape")}
    a = {n: attr[..., an.index(n)].astype(np.float32) for n in an}
    gam, gamd = GAMMA[month - 1][:, None, None], GAMMA_D[month - 1][:, None, None]
    dzk = a["dz"][..., None] / 1000.0
    t = c["t2m_c"] - gam * dzk
    td = 0.5 * (c["d2m_c"] - gamd * dzk) + 0.5 * dew_from_rh(t, c["rh"])
    td = np.minimum(td, t)
    tw = wet_bulb(t, rh_from(t, td))
    p = np.clip(c["precip"], 0, None)
    f_snow = 1.0 / (1.0 + np.exp(-(1.0 - tw) / 0.7))
    rain = p * (1 - f_snow)
    g = np.clip(c["gust"], 0, None)
    base = {"gust_x10": np.clip(g - 10, 0, None) / 5, "gust_x15": np.clip(g - 15, 0, None) / 5,
            "gust_x20": np.clip(g - 20, 0, None) / 5, "rain": rain,
            "convective": np.log1p(np.clip(c["cape"], 0, None) / 1000.0 * p)}
    for h in HATS:
        base[f"p_tw{h:+.1f}"] = p * np.clip(1 - np.abs(tw - h) / 1.5, 0, None)
    o = slice(ORIGIN, T)
    trig = np.stack([base[n][..., o] for n in TRIG], -1)
    load = [np.ones_like(t[..., o]), np.clip(c["soil_moisture"], 0, None)[..., o] / 0.3]
    for tau in TAUS:
        for n in FILT:
            load.append(exp_filter(base[n], tau)[..., o])
    load = np.stack(load, -1)
    leaf = ((month >= 5) & (month <= 10)).astype(np.float32)[:, None]
    gg = np.stack([a["dz"] / 100.0, a["canopy"] / 100.0, a["canopy"] / 100.0 * leaf, a["forest"], a["developed"],
                   a["wet"], a["tpi"] / 10.0, a["slope"] / 10.0], -1)
    return trig.astype(np.float16), load.astype(np.float16), gg.astype(np.float32)


def relief_strata(attr, rho, an, q=5):
    dz = attr[..., an.index("dz")]
    m = (rho * dz).sum(1)
    sd = np.sqrt((rho * (dz - m[:, None]) ** 2).sum(1))
    edges = np.quantile(sd, np.linspace(0, 1, q + 1)[1:-1])
    return np.digitize(sd, edges), sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["quad", "pooled", "mean", "other"])
    a = ap.parse_args()
    N = np.load(OUT / "node_weather_e3.npz")
    ch, an = list(N["channels"].astype(str)), list(N["attr_names"].astype(str))
    fips, ev = N["fips"].astype(str), N["event"].astype(str)
    month = np.array([int(e[5:7]) for e in ev])
    attr, rho = N["node_attr"].astype(np.float32), N["node_rho"].astype(np.float32)
    meta = {}
    for v in a.variants:
        if v == "quad":
            w, at, r = N["node_weather"], attr, rho
        elif v == "pooled":
            w, at, r = np.broadcast_to(N["pop_weather"][:, None], N["node_weather"].shape), attr, rho
        elif v == "mean":
            w = N["pop_weather"][:, None]
            at = (rho[..., None] * attr).sum(1, keepdims=True) / rho.sum(1, keepdims=True)[..., None]
            r = np.ones((len(fips), 1), np.float32)
        elif v == "other":
            counties, first = np.unique(fips, return_index=True)
            strat, _ = relief_strata(attr[first], rho[first], an)
            rng = np.random.default_rng(SEED)
            donor = {}
            for s in np.unique(strat):
                cs = counties[strat == s]
                perm = rng.permutation(len(cs))
                while len(cs) > 1 and (perm == np.arange(len(cs))).any():     # no county keeps its own nodes
                    perm = rng.permutation(len(cs))
                donor.update(dict(zip(cs, cs[perm])))
            fi = dict(zip(counties, first))
            src = np.array([fi[donor[f]] for f in fips])
            w, at, r = np.broadcast_to(N["pop_weather"][:, None], N["node_weather"].shape), attr[src], rho[src]
            meta["other_same_county_share"] = float(np.mean([donor[f] == f for f in counties]))
        else:
            raise ValueError(v)
        trig, load, gg = [], [], []
        for s in range(0, len(fips), 500):                                   # chunks keep memory bounded
            tr, lo, g = features(np.asarray(w[s:s + 500], np.float32), ch, at[s:s + 500], an, month[s:s + 500])
            trig.append(tr); load.append(lo); gg.append(g)
        trig, load, gg = np.concatenate(trig), np.concatenate(load), np.concatenate(gg)
        np.savez(OUT / f"node_feat_{v}.npz", trig=trig, load=load, attr=gg, rho=r.astype(np.float32),
                 trig_names=np.array(TRIG), load_names=np.array(LOAD), attr_names=np.array(ATTR), fips=fips, event=ev)
        meta[v] = dict(shape_trig=list(trig.shape), shape_load=list(load.shape),
                       trig_active_share=np.round((trig.astype(np.float32) > 1e-3).mean((0, 1, 2)), 4).tolist())
        print(v, meta[v], flush=True)
    (HERE / "data_provenance" / "node_feat.json").write_text(json.dumps(dict(
        trig=TRIG, load=LOAD, attr=ATTR, taus=TAUS, hats=HATS, gamma_k_per_km=GAMMA.tolist(),
        gamma_d_k_per_km=np.round(GAMMA_D, 3).tolist(), **meta), indent=1) + "\n")


if __name__ == "__main__":
    main()
