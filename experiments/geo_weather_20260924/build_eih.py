"""Exposure-integrated hazard features (DESIGN v1): a fixed dictionary of hazard-gated local forcing features,
computed per node with fixed physical downscaling, multiplied by static modulators, integrated over the
county's exposure measure, and passed through a fixed memory bank. Nothing here is learned.

  Phi_i,(b,a,tau)(t) = F_tau * sum_k w_ik m_a(g_k) psi_b(D(W_c(k)(t), g_k))

psi_b (zero in calm): gust ramps [G - 10]+, [G - 15]+, [G - 20]+ (per 5 m/s); liquid precipitation (mm/h);
  precipitation x hat functions of the node wet-bulb temperature at -3, -1.5, 0, 1.5, 3 C (half-width 1.5 C);
  convective log1p(CAPE / 1000 x precipitation).
D: T_k = T_c - Gamma(month) dz_k; Td_k = mean of the dew-point lapse Gamma_d(month) and constant relative humidity;
  Tw_k Stull (2011); snow fraction sigmoid((1.0 - Tw) / 0.7) (build_node_features.py for the sources).
m_a: 1, canopy, canopy x leaf-on (May-Oct), poorly drained share.
F_tau: identity and unit-gain exponential filters with tau = 3, 12, 48 h, run from the first prefix hour
  (linear and time-invariant, so they commute with the exposure integral and are applied after it).

Variants (the recoverability ladder):
  area    nodes = cells, weights = area, no downscaling (the host's current weighting)
  pop     nodes = cells, weights = population, no downscaling
  quad    nodes = cell x dz band, weights = population, downscaling by the band's dz
  mean    one node: population-weighted county weather and attributes, downscaled by the mean dz
  pooled  the county's bands (pooled over cells) under the population-weighted county weather
  other   another county's bands (same relief stratum, seeded) under this county's population-weighted weather
  quadn   quad with each modulator normalised to mean one over the county's exposure (within-county only)
Output: data/interim/geo_weather/eih_<variant>.npz: phi [U,144,J] float16, names; unit order of features_e3r2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "open_gcrk_20260919"))
sys.path.insert(0, str(HERE))
import build_panel216_r2 as P2  # noqa: E402
from build_node_features import GAMMA, GAMMA_D, HATS, dew_from_rh, rh_from, wet_bulb  # noqa: E402

OUT = ROOT / "data" / "interim" / "geo_weather"
FEAT = ROOT / "data" / "interim" / "open_gcrk" / "features_e3r2.npz"
ORIGIN, T = 72, 216
PSI = ["gust_x10", "gust_x15", "gust_x20", "rain"] + [f"p_tw{c:+.1f}" for c in HATS] + ["convective"]
MOD = ["one", "canopy", "canopy_leafon", "wet"]
TAUS = (0, 3, 12, 48)
NAMES = [f"{b}*{a}@{t}" for t in TAUS for a in MOD for b in PSI]
VARIANTS = ("area", "pop", "quad", "mean", "pooled", "other", "quadn")
SEED = 20260925


def psi(t, td_c, rh_c, p, g, cape, dz, month):
    """Node forcing features. t, td_c, rh_c, p, g, cape [T, n]; dz [n] (m); month scalar -> [T, n, B]."""
    gam, gamd = GAMMA[month - 1], GAMMA_D[month - 1]
    tk = t - gam * dz / 1000.0
    td = 0.5 * (td_c - gamd * dz / 1000.0) + 0.5 * dew_from_rh(tk, rh_c)
    td = np.minimum(td, tk)
    tw = wet_bulb(tk, rh_from(tk, td))
    p = np.clip(p, 0, None)
    f_snow = 1.0 / (1.0 + np.exp(-(1.0 - tw) / 0.7))
    g = np.clip(g, 0, None)
    out = [np.clip(g - 10, 0, None) / 5, np.clip(g - 15, 0, None) / 5, np.clip(g - 20, 0, None) / 5, p * (1 - f_snow)]
    out += [p * np.clip(1 - np.abs(tw - h) / 1.5, 0, None) for h in HATS]
    out.append(np.log1p(np.clip(cape, 0, None) / 1000.0 * p))
    return np.stack(out, -1)


def filt(x: np.ndarray, tau: float) -> np.ndarray:
    if tau == 0:
        return x
    e = np.exp(-1.0 / tau)
    out = np.empty_like(x); acc = np.zeros(x.shape[1:], x.dtype)
    for s in range(x.shape[0]):
        acc = e * acc + (1 - e) * x[s]
        out[s] = acc
    return out


def modulators(nd: pd.DataFrame, leaf: float) -> np.ndarray:
    can = nd["canopy"].to_numpy() / 100.0
    return np.stack([np.ones(len(nd)), can, can * leaf, nd["wet"].to_numpy()], -1)


def integrate(ps: np.ndarray, w: np.ndarray, m: np.ndarray) -> np.ndarray:
    """ps [T, n, B], w [n], m [n, A] -> memory bank [144, J]."""
    x = np.einsum("n,tnb,na->tab", w, ps, m)                    # [T, B, A]
    x = x.reshape(T, -1)                                          # b fastest, then a
    return np.concatenate([filt(x, tau)[ORIGIN:] for tau in TAUS], -1)


def relief_donors(nodes: pd.DataFrame, counties: np.ndarray) -> dict:
    g = nodes.groupby("fips")
    mu = g.apply(lambda d: np.average(d.dz, weights=d.w_pop))
    sd = g.apply(lambda d: float(np.sqrt(np.average((d.dz - np.average(d.dz, weights=d.w_pop)) ** 2, weights=d.w_pop))))
    sd = sd.reindex(counties)
    strat = np.digitize(sd.to_numpy(), np.quantile(sd.to_numpy(), [0.2, 0.4, 0.6, 0.8]))
    rng = np.random.default_rng(SEED)
    donor = {}
    for s in np.unique(strat):
        cs = counties[strat == s]
        perm = rng.permutation(len(cs))
        while len(cs) > 1 and (perm == np.arange(len(cs))).any():
            perm = rng.permutation(len(cs))
        donor.update(dict(zip(cs, cs[perm])))
    return donor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    F = np.load(FEAT)
    fips_u, ev_u = F["fips"].astype(str), F["event"].astype(str)
    nodes = pd.read_parquet(OUT / "nodes_cs.parquet")
    by = {f: d.reset_index(drop=True) for f, d in nodes.groupby("fips")}
    counties = np.array(sorted(set(fips_u)))
    assert set(counties) <= set(by), "counties without nodes"
    donor = relief_donors(nodes[nodes.fips.isin(set(counties))], counties)
    sel = {e["event"]: e for e in json.loads((ROOT / "experiments/open_gcrk_20260919/selected_events_e3.json").read_text())["events"]}
    phi = {v: np.zeros((len(fips_u), T - ORIGIN, len(NAMES)), np.float16) for v in a.variants}
    for ev in sorted(set(ev_u)):
        t0 = pd.Timestamp(sel[ev]["window_start_utc"])
        times, ch, _ = P2.fields_r2(ev, t0)
        assert (pd.DatetimeIndex(times) == pd.date_range(t0, periods=T, freq="h")).all(), ev
        a_, b_ = 17.625, 243.04
        gm = np.log(np.clip(ch["rh"], 1e-3, None) / 100.0) + a_ * ch["t2m_c"] / (b_ + ch["t2m_c"])
        ch["d2m_c"] = b_ * gm / (a_ - gm)
        month = int(ev[5:7]); leaf = float(5 <= month <= 10)
        fields = {k: np.asarray(ch[k], np.float32) for k in ("t2m_c", "d2m_c", "rh", "precip", "gust", "cape")}
        for u in np.where(ev_u == ev)[0]:
            nd = by[fips_u[u]]
            cell = nd.groupby(["i", "j"], as_index=False).agg(w_pop=("w_pop", "sum"), w_area=("w_area", "sum"))
            wc = {k: v[:, cell.i.to_numpy(), cell.j.to_numpy()] for k, v in fields.items()}       # [T, cells]
            pw = cell.w_pop.to_numpy()
            county = {k: (v * pw).sum(1, keepdims=True) for k, v in wc.items()}                   # pop-mean weather
            ci = pd.MultiIndex.from_frame(nd[["i", "j"]]).map({(r.i, r.j): n for n, r in cell.iterrows()}.get).to_numpy()
            wn = {k: v[:, ci] for k, v in wc.items()}                                              # node = its cell
            m_nodes = modulators(nd, leaf)
            # cell-level attributes (band-merged): weighted means inside the cell
            def cell_attr(wcol):
                ww = nd[wcol].to_numpy()
                agg = pd.DataFrame(dict(c=ci, w=ww, **{k: nd[k].to_numpy() * ww for k in ("canopy", "wet", "dz")}))
                s = agg.groupby("c").sum()
                return pd.DataFrame({k: s[k] / s["w"] for k in ("canopy", "wet", "dz")}), s["w"].to_numpy()
            for v in a.variants:
                if v in ("area", "pop"):
                    att, w = cell_attr("w_area" if v == "area" else "w_pop")
                    ps = psi(wc["t2m_c"], wc["d2m_c"], wc["rh"], wc["precip"], wc["gust"], wc["cape"],
                             np.zeros(len(att)), month)
                    phi[v][u] = integrate(ps, w / w.sum(), modulators(att, leaf))
                elif v in ("quad", "quadn"):
                    w = nd.w_pop.to_numpy()
                    ps = psi(wn["t2m_c"], wn["d2m_c"], wn["rh"], wn["precip"], wn["gust"], wn["cape"], nd.dz.to_numpy(), month)
                    m = m_nodes
                    if v == "quadn":
                        m = m / np.clip((w[:, None] * m).sum(0, keepdims=True), 1e-6, None)
                    phi[v][u] = integrate(ps, w, m)
                elif v == "mean":
                    w = nd.w_pop.to_numpy()
                    mean = pd.DataFrame({k: [np.average(nd[k], weights=w)] for k in ("canopy", "wet", "dz")})
                    ps = psi(county["t2m_c"], county["d2m_c"], county["rh"], county["precip"], county["gust"],
                             county["cape"], mean.dz.to_numpy(), month)
                    phi[v][u] = integrate(ps, np.ones(1), modulators(mean, leaf))
                elif v in ("pooled", "other"):
                    src = nd if v == "pooled" else by[donor[fips_u[u]]]
                    bands = src.assign(wdz=src.dz * src.w_pop, wcan=src.canopy * src.w_pop, wwet=src.wet * src.w_pop)
                    bands = bands.groupby("band").agg(w=("w_pop", "sum"), wdz=("wdz", "sum"), wcan=("wcan", "sum"),
                                                      wwet=("wwet", "sum"))
                    att = pd.DataFrame(dict(dz=bands.wdz / bands.w, canopy=bands.wcan / bands.w, wet=bands.wwet / bands.w))
                    n = len(att)
                    rep = {k: np.repeat(val, n, 1) for k, val in county.items()}
                    ps = psi(rep["t2m_c"], rep["d2m_c"], rep["rh"], rep["precip"], rep["gust"], rep["cape"],
                             att.dz.to_numpy(), month)
                    w = bands.w.to_numpy()
                    phi[v][u] = integrate(ps, w / w.sum(), modulators(att, leaf))
        print(ev, int((ev_u == ev).sum()), "units", flush=True)
    for v in a.variants:
        np.savez(OUT / f"eih_{v}.npz", phi=phi[v], names=np.array(NAMES), fips=fips_u, event=ev_u)
    meta = dict(names=NAMES, variants=a.variants, donor_self_share=float(np.mean([donor[c] == c for c in counties])),
                active_share={v: np.round((phi[v].astype(np.float32) > 1e-4).mean((0, 1))[:len(PSI) * len(MOD)], 3).tolist()
                              for v in a.variants})
    (HERE / "data_provenance" / "eih.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("saved", {v: phi[v].shape for v in a.variants})


if __name__ == "__main__":
    main()
