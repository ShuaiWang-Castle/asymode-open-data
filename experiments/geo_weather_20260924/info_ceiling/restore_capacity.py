"""Restoration-capacity probes (notes/RESTORATION_CAPACITY_PROBE_SCOPE_20261002_ZH.md). Public D only; no training of
the neural models; no existing file is modified.

  python restore_capacity.py size   observed restoration after the peak by outage size and regime; the host's and the
                                    dose-kernel arm's restoration rate at the observed peak (descriptive)
  python restore_capacity.py sim    post-hoc simulation on the host's held-out rates (seed 0): first-order restoration
                                    r p replaced by the capacity-limited flow r p / (1 + kappa r p) (descriptive)
  python restore_capacity.py geo    registered test R1: does geography predict the restoration of large outages?
  python restore_capacity.py simregion  post-hoc simulation: the host's restoration slowed by the regional burden (oracle and forecast)
  python restore_capacity.py predictable  how predictable is the regional burden from ERA5 before the storm (event folds)
  python restore_capacity.py rolling  registered test R3: rolling-origin restoration forecast with the regional burden observed at the origin
  python restore_capacity.py region mechanism test: is restoration slower when the neighbouring counties (150 km) are also out?
"""
from __future__ import annotations

import json
import sys

import numpy as np

from geo_evidence import cluster_draws, fold_of_unit
from probe import BOOT_SEED, EVENT_PARAMS, EVENT_ROUNDS, FEAT, OUT, REGIMES, RES, ROOT, load_static
from score_dose import collect

BINS = [(.005, .02), (.02, .05), (.05, .10), (.10, .25), (.25, 1.01)]
KAPPAS = [0, 5, 10, 20, 40, 80, 160, 320]


def wq(v, ww, q=.5):
    o = np.argsort(v); c = np.cumsum(ww[o]) / ww.sum()
    return float(v[o][min(np.searchsorted(c, q), len(v) - 1)])


def base():
    s = load_static()
    y, m, w = s["y"].astype(float), s["m"].astype(float), s["w"].astype(float)
    reg, peak = s["regime"].astype(str), s["peak"].astype(float)
    yo = np.where(m > 0, y, -1.0); tpk = yo.argmax(1)
    ok = (tpk + 24 <= 143) & (m[np.arange(len(y)), np.minimum(tpk + 24, 143)] > 0)
    return s, y, m, w, reg, peak, tpk, ok


def size() -> None:
    s, y, m, w, reg, peak, tpk, ok = base(); n = len(y)
    host = collect("v1_host_s0", n); dkv = collect("v1_dkv_s0", n)
    out = {"bins": BINS, "note": "R24 = outage fraction 24 h after the observed peak over the peak (design-weighted median); "
           "implied hourly rate = 1 - R24^(1/24), floor of R24 1e-4; model rates are medians at the observed peak hour", "by_regime": {}}
    for r in REGIMES:
        rows = []
        for lo, hi in BINS:
            f = ok & (reg == r) & (peak >= lo) & (peak < hi)
            if f.sum() < 8:
                rows.append(dict(n=int(f.sum()))); continue
            r24 = y[f, tpk[f] + 24] / peak[f]; med = wq(r24, w[f])
            rows.append(dict(n=int(f.sum()), R24=med, R24_unweighted=float(np.median(r24)), share_R24_above_0p2=float(np.average(r24 > .2, weights=w[f])),
                             implied_rate=float(1 - max(med, 1e-4) ** (1 / 24)), host_rate=float(np.median(host[2][f, tpk[f]])),
                             dkv_rate=float(np.median(dkv[2][f, tpk[f]])) if len(dkv[3]) == 5 else None))
        out["by_regime"][r] = rows
        print(f"{r:14s} " + " | ".join(f"n {q['n']:3d} R24 {q['R24']:.2f} (>0.2: {q['share_R24_above_0p2']:.2f}) rate {q['implied_rate']:.3f} host {q['host_rate']:.3f}" if "R24" in q else f"n {q['n']:3d}" for q in rows))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "restore_size.json").write_text(json.dumps(out, indent=1) + "\n")


def sim() -> None:
    s, y, m, w, reg, peak, tpk, ok = base(); n = len(y)
    y0 = np.load(FEAT, allow_pickle=False)["y0"].astype(float)
    P0, U, R, _, _ = collect("v1_host_s0", n)
    S = peak >= .10; A = np.ones(n, bool); cnt = m.sum(1); fold = fold_of_unit(n)

    def roll(kappa):
        p = y0.copy(); out = np.empty_like(U)
        for t in range(144):
            p = np.clip(p + U[:, t] * (1 - p) - R[:, t] * p / (1 + kappa * R[:, t] * p), 0, 1); out[:, t] = p
        return out

    def rmse(P, sub):
        return float(np.sqrt((w * (m * (P - y) ** 2).sum(1))[sub].sum() / (w * cnt)[sub].sum()))

    def row(P):
        pk = P.max(1)
        return dict(S=1 - rmse(P, S) / rmse(P0, S), all=1 - rmse(P, A) / rmse(P0, A), nonS=1 - rmse(P, ~S) / rmse(P0, ~S),
                    false_peaks=int(((pk >= .1) & ~S).sum()),
                    S_by_regime={r: 1 - rmse(P, S & (reg == r)) / rmse(P0, S & (reg == r)) for r in REGIMES},
                    all_by_regime={r: 1 - rmse(P, reg == r) / rmse(P0, reg == r) for r in REGIMES})

    sims = {k: roll(float(k)) for k in KAPPAS}
    out = {"reproduction_max_abs_diff": float(np.abs(sims[0] - P0).max()), "origin_stock_share_ge_1pct": dict(S=float((y0[S] >= .01).mean()), nonS=float((y0[~S] >= .01).mean())),
           "S_peak_hour_after_origin_median": {r: float(np.median(tpk[S & (reg == r)])) for r in REGIMES},
           "global": {str(k): row(sims[k]) for k in KAPPAS}}
    Pg, Pr, cg, cr = np.empty_like(P0), np.empty_like(P0), {}, {}
    for k in range(1, 6):
        cg[k] = min(KAPPAS, key=lambda q: rmse(sims[q], fold != k)); Pg[fold == k] = sims[cg[k]][fold == k]
        for r in REGIMES:
            cr[f"{k}:{r}"] = min(KAPPAS, key=lambda q: rmse(sims[q], (fold != k) & (reg == r)))
            Pr[(fold == k) & (reg == r)] = sims[cr[f"{k}:{r}"]][(fold == k) & (reg == r)]
    out["cross_fitted_global"] = dict(chosen=cg, **row(Pg)); out["cross_fitted_by_regime"] = dict(chosen=cr, **row(Pr))
    (RES / "restore_saturation_sim.json").write_text(json.dumps(out, indent=1) + "\n")
    pc = lambda d: f"S {100 * d['S']:+.2f}% all {100 * d['all']:+.2f}% nonS {100 * d['nonS']:+.2f}% FP {d['false_peaks']}"  # noqa: E731
    for k in KAPPAS:
        print(f"kappa {k:4d}: {pc(out['global'][str(k)])}")
    print("cross-fitted global:", cg, pc(out["cross_fitted_global"]))
    print("cross-fitted by regime:", pc(out["cross_fitted_by_regime"]), {r: f"{100 * v:+.1f}%" for r, v in out["cross_fitted_by_regime"]["S_by_regime"].items()},
          "all:", {r: f"{100 * v:+.1f}%" for r, v in out["cross_fitted_by_regime"]["all_by_regime"].items()})


def geo() -> None:
    import lightgbm as lgb
    from scipy.stats import spearmanr
    s, y, m, w, reg, peak, tpk, ok = base(); n = len(y)
    fam, sysv = s["family"].astype(str), s["system"].astype(str)
    Wh = np.load(OUT / "Wh.npy"); wc = list(s["Whcols"].astype(str))
    zv = np.load(ROOT / "data" / "interim" / "panel_v1" / "vuln_v1D.npz")
    fold = fold_of_unit(n)
    out = {}
    for pop, sel in (("A_tropical_winter", ok & (peak >= .10) & np.isin(reg, ["tropical", "winter"])), ("B_all_regimes", ok & (peak >= .10))):
        ix = np.where(sel)[0]
        seg_ok = np.array([m[u, t + 1:t + 25].sum() >= 12 for u, t in zip(ix, tpk[ix])]); ix = ix[seg_ok]
        a24 = np.clip(np.array([(m[u, t + 1:t + 25] * y[u, t + 1:t + 25]).sum() / m[u, t + 1:t + 25].sum() for u, t in zip(ix, tpk[ix])]) / peak[ix], 0, 1.5)
        r24 = np.clip(y[ix, tpk[ix] + 24] / peak[ix], 0, 1.5)
        post = []
        for u, t in zip(ix, tpk[ix]):
            seg = Wh[u, t:t + 25]
            post.append([seg[:, wc.index("gust")].max(), seg[:, wc.index("wind_speed")].max(), seg[:, wc.index("gust_excess_energy")].max(),
                         seg[:, wc.index("precip")].sum(), seg[:, wc.index("snowfall")].sum(), seg[:, wc.index("t2m_c")].min()])
        Pb = np.column_stack([np.log(peak[ix]), tpk[ix], np.array(post, np.float32)]).astype(np.float32)
        blocks = {"P": Pb, "B": s["B"][ix], "C": s["C"][ix], "G": s["G"][ix], "Gp": s["Gp"][ix], "H": s["H"][ix], "Hp": s["Hp"][ix],
                  "V": zv["z"][ix].astype(np.float32), "Vp": zv["z_perm"][ix].astype(np.float32)}
        variants = {"BW": ["P", "B"], "BG": ["P", "B", "G"], "BGp": ["P", "B", "Gp"], "BH": ["P", "B", "H"], "BHp": ["P", "B", "Hp"],
                    "BV": ["P", "B", "V"], "BVp": ["P", "B", "Vp"], "BC": ["P", "B", "C"]}
        f, ws = fold[ix], w[ix]
        keys = sorted(set(zip(reg[ix], fam[ix]))); gid = {kk: q for q, kk in enumerate(keys)}
        g = np.array([gid[kk] for kk in zip(reg[ix], fam[ix])]); members = [np.where(g == q)[0] for q in range(len(keys))]
        draws = [np.concatenate([members[q] for q in d]) for d in cluster_draws(keys, np.random.default_rng(BOOT_SEED))]
        res = dict(units=int(len(ix)), systems=int(len(np.unique(sysv[ix]))), families=len(keys), by_regime={r: int((reg[ix] == r).sum()) for r in REGIMES})
        for tname, tgt in (("A24", a24), ("R24", r24)):
            preds = {}
            for name, bl in variants.items():
                X = np.concatenate([blocks[q] for q in bl], 1); P = np.full(len(ix), np.nan)
                for k in range(1, 6):
                    tr, te = f != k, f == k
                    if te.sum() == 0:
                        continue
                    P[te] = lgb.train(EVENT_PARAMS, lgb.Dataset(X[tr], tgt[tr], weight=ws[tr]), EVENT_ROUNDS).predict(X[te])
                preds[name] = P

            def rmse(P, j):
                return float(np.sqrt(np.average((P[j] - tgt[j]) ** 2, weights=ws[j])))

            al = np.arange(len(ix)); mu = float(np.average(tgt, weights=ws)); var = float(np.average((tgt - mu) ** 2, weights=ws))
            d = dict(mean=mu, sd=float(np.sqrt(var)), r2={k: float(1 - rmse(P, al) ** 2 / var) for k, P in preds.items()}, differences={})
            for a, b in (("BG", "BGp"), ("BG", "BW"), ("BH", "BHp"), ("BH", "BW"), ("BV", "BVp"), ("BV", "BW"), ("BC", "BW")):
                bs = [1 - rmse(preds[a], j) / rmse(preds[b], j) for j in draws]
                d["differences"][f"{a} vs {b}"] = dict(point=float(1 - rmse(preds[a], al) / rmse(preds[b], al)), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
            res[tname] = d
            print(pop, tname, f"n {len(ix)} mean {mu:.3f} sd {np.sqrt(var):.3f} | R2", {k: round(v, 3) for k, v in d["r2"].items()})
            print("   ", {k: f"{100 * v['point']:+.2f}% [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}]" for k, v in d["differences"].items()})
        # descriptive: within-system partial Spearman of A24 with six county descriptors, controlling for log peak
        gc, cc = list(s["Gcols"].astype(str)), list(s["Ccols"].astype(str))
        desc = {"canopy_mean": s["G"][ix, gc.index("geo_canopy_mean")], "forest_frac": s["G"][ix, gc.index("geo_forest_frac")],
                "developed_frac": s["G"][ix, gc.index("geo_developed_frac")], "relief": s["G"][ix, gc.index("geo_relief_p95_p5")],
                "log_pop_density": s["C"][ix, cc.index("log_pop_density")], "rucc": s["C"][ix, cc.index("rucc")]}
        tab = {}
        for name, v in desc.items():
            rho = []
            for q in np.unique(sysv[ix]):
                j = np.where(sysv[ix] == q)[0]
                if len(j) < 8 or np.std(v[j]) == 0 or np.std(a24[j]) == 0:
                    continue
                rk = lambda x: np.argsort(np.argsort(x)).astype(float)  # noqa: E731
                lp = rk(np.log(peak[ix][j])); ra = rk(a24[j]); rv = rk(v[j].astype(float))
                ea = ra - np.polyval(np.polyfit(lp, ra, 1), lp); ev = rv - np.polyval(np.polyfit(lp, rv, 1), lp)
                rho.append(float(np.corrcoef(ea, ev)[0, 1]))
            rho = np.array(rho)
            tab[name] = dict(systems=int(len(rho)), mean=float(np.nanmean(rho)), positive=int((rho > 0).sum()), negative=int((rho < 0).sum()))
        res["within_system_partial_spearman_A24"] = tab
        print("    within-system partial Spearman (A24):", {k: f"{v['mean']:+.3f} ({v['positive']}+/{v['negative']}-)" for k, v in tab.items()})
        out[pop] = res
    (RES / "restore_geo.json").write_text(json.dumps(out, indent=1) + "\n")

def region_table():
    """Customers out in all counties within 150 km of each county-event (own county included and excluded), hourly over
    the forecast window, from the full EAGLE-I county table (hourly mean of the quarter-hour rows, absent rows = 0)."""
    import pandas as pd
    f = OUT / "regional_burden_150km.npz"
    if f.exists():
        return dict(np.load(f))
    z = np.load(FEAT, allow_pickle=False)
    fips, sysv = z["fips"].astype(str), z["system"].astype(str)
    origin = pd.to_datetime(z["origin"].astype(str))
    gz = pd.read_csv(ROOT / "data" / "raw" / "census" / "2020_Gaz_counties_national.txt", sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    gz.columns = [c.strip() for c in gz.columns]
    gz["GEOID"] = gz.GEOID.str.zfill(5)
    cust = pd.read_parquet(ROOT / "data" / "interim" / "eaglei_county_customers_2024.parquet")["customers"]
    cust.index = cust.index.astype(str).str.zfill(5)
    gz = gz[gz.GEOID.isin(cust.index)].reset_index(drop=True)
    col = {g: j for j, g in enumerate(gz.GEOID)}
    la, lo = np.radians(gz.INTPTLAT.to_numpy()), np.radians(gz.INTPTLONG.to_numpy())
    cu = cust.reindex(gz.GEOID).to_numpy(float)
    n = len(fips); out_in, out_ex, base_in, n_hit = np.zeros((n, 144), np.float32), np.zeros((n, 144), np.float32), np.zeros(n), np.zeros((n, 144), np.float32)
    own = np.zeros((n, 144), np.float32)
    for q in np.unique(sysv):
        ii = np.where(sysv == q)[0]; t0 = origin[ii[0]]; t1 = t0 + pd.Timedelta(hours=144)
        parts = [pd.read_parquet(ROOT / "data" / "interim" / f"eaglei_outages_{yy}.parquet", columns=["fips", "ts", "customers_out"],
                                 filters=[("ts", ">=", t0), ("ts", "<", t1)]) for yy in sorted({t0.year, (t1 - pd.Timedelta(minutes=1)).year})]
        df = pd.concat(parts, ignore_index=True)
        df = df[(df.ts >= t0) & (df.ts < t1)]
        df["j"] = df.fips.astype(str).str.zfill(5).map(col); df = df.dropna(subset=["j", "customers_out"])
        df["h"] = ((df.ts - t0) / pd.Timedelta(hours=1)).astype(int)
        g = df.groupby(["j", "h"]).customers_out.sum() / 4.0          # hourly mean of the quarter-hour rows, absent rows = 0
        M = np.zeros((len(gz), 144)); M[g.index.get_level_values(0).astype(int), g.index.get_level_values(1)] = g.to_numpy(float)
        frac = M / np.maximum(cu[:, None], 1.0)
        for i in ii:
            c = col.get(fips[i])
            if c is None:
                continue
            d = 2 * 6371.0 * np.arcsin(np.sqrt(np.sin((la - la[c]) / 2) ** 2 + np.cos(la[c]) * np.cos(la) * np.sin((lo - lo[c]) / 2) ** 2))
            nb = d <= 150.0
            out_in[i] = M[nb].sum(0); own[i] = M[c]; out_ex[i] = out_in[i] - M[c]; base_in[i] = cu[nb].sum()
            nbx = nb.copy(); nbx[c] = False
            n_hit[i] = (frac[nbx] >= .05).sum(0)
        print(q, len(ii), f"rows {len(df)}", flush=True)
    r = dict(out_in=out_in, out_ex=out_ex, own=own, base_in=base_in, n_hit=n_hit)
    np.savez_compressed(f, **r)
    return r


RADII = (50, 150, 300, 600)


def region_table_multi():
    """As region_table for several radii and from 24 h before the origin (168 hourly columns, column 24 = origin hour):
    customers out of the other counties within each radius, their customer base, and the county's own customers out."""
    import pandas as pd
    f = OUT / "regional_burden_multi.npz"
    if f.exists():
        return dict(np.load(f))
    z = np.load(FEAT, allow_pickle=False)
    fips, sysv = z["fips"].astype(str), z["system"].astype(str)
    origin = pd.to_datetime(z["origin"].astype(str))
    gz = pd.read_csv(ROOT / "data" / "raw" / "census" / "2020_Gaz_counties_national.txt", sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    gz.columns = [c.strip() for c in gz.columns]
    gz["GEOID"] = gz.GEOID.str.zfill(5)
    cust = pd.read_parquet(ROOT / "data" / "interim" / "eaglei_county_customers_2024.parquet")["customers"]
    cust.index = cust.index.astype(str).str.zfill(5)
    gz = gz[gz.GEOID.isin(cust.index)].reset_index(drop=True)
    col = {g: j for j, g in enumerate(gz.GEOID)}
    la, lo = np.radians(gz.INTPTLAT.to_numpy()), np.radians(gz.INTPTLONG.to_numpy())
    cu = cust.reindex(gz.GEOID).to_numpy(float)
    n = len(fips); T = 168
    r = {f"out_ex_{q}": np.zeros((n, T), np.float32) for q in RADII}
    r.update({f"base_ex_{q}": np.zeros(n) for q in RADII}); r["own"] = np.zeros((n, T), np.float32)
    for q in np.unique(sysv):
        ii = np.where(sysv == q)[0]; t0 = origin[ii[0]] - pd.Timedelta(hours=24); t1 = t0 + pd.Timedelta(hours=T)
        parts = [pd.read_parquet(ROOT / "data" / "interim" / f"eaglei_outages_{yy}.parquet", columns=["fips", "ts", "customers_out"],
                                 filters=[("ts", ">=", t0), ("ts", "<", t1)]) for yy in sorted({t0.year, (t1 - pd.Timedelta(minutes=1)).year})]
        df = pd.concat(parts, ignore_index=True)
        df = df[(df.ts >= t0) & (df.ts < t1)]
        df["j"] = df.fips.astype(str).str.zfill(5).map(col); df = df.dropna(subset=["j", "customers_out"])
        df["h"] = ((df.ts - t0) / pd.Timedelta(hours=1)).astype(int)
        g = df.groupby(["j", "h"]).customers_out.sum() / 4.0
        M = np.zeros((len(gz), T)); M[g.index.get_level_values(0).astype(int), g.index.get_level_values(1)] = g.to_numpy(float)
        for i in ii:
            c = col.get(fips[i])
            if c is None:
                continue
            d = 2 * 6371.0 * np.arcsin(np.sqrt(np.sin((la - la[c]) / 2) ** 2 + np.cos(la[c]) * np.cos(la) * np.sin((lo - lo[c]) / 2) ** 2))
            r["own"][i] = M[c]
            for rad in RADII:
                nb = d <= rad; nb[c] = False
                r[f"out_ex_{rad}"][i] = M[nb].sum(0); r[f"base_ex_{rad}"][i] = cu[nb].sum()
        print(q, len(ii), flush=True)
    np.savez_compressed(f, **r)
    return r


def region() -> None:
    import lightgbm as lgb
    s, y, m, w, reg, peak, tpk, ok = base(); n = len(y)
    z = np.load(FEAT, allow_pickle=False); cust = z["cust"].astype(float)
    fam, sysv = s["family"].astype(str), s["system"].astype(str)
    r = region_table()
    fr_own = r["own"] / np.maximum(cust[:, None], 1.0)
    cc = [np.corrcoef(fr_own[i][m[i] > 0], y[i][m[i] > 0])[0, 1] for i in np.where(peak >= .05)[0] if (m[i] > 0).sum() > 50]
    check = dict(own_fraction_vs_panel_y_median_corr=float(np.nanmedian(cc)), median_abs_diff_at_peak=float(np.median(np.abs(fr_own[np.arange(n), tpk] - peak)[peak >= .05])))
    print("alignment check:", check)
    base_ex = np.maximum(r["base_in"] - cust, 1.0)
    rf_ex = r["out_ex"] / base_ex[:, None]                                    # outage fraction of the neighbours within 150 km
    a = np.arange(n)
    RF = rf_ex[a, tpk]; RN = r["out_ex"][a, tpk]; NH = r["n_hit"][a, tpk]
    RFmax = rf_ex.max(1); RNmax = r["out_ex"].max(1)
    Wh = np.load(OUT / "Wh.npy"); wc = list(s["Whcols"].astype(str))
    out = {"alignment_check": check}
    sel0 = ok & np.array([m[u, t + 1:min(t + 25, 144)].sum() >= 12 for u, t in zip(a, tpk)])
    a24 = np.zeros(n); r24 = np.zeros(n)
    for u in np.where(sel0)[0]:
        t = tpk[u]; a24[u] = (m[u, t + 1:t + 25] * y[u, t + 1:t + 25]).sum() / m[u, t + 1:t + 25].sum() / peak[u]; r24[u] = y[u, t + 24] / peak[u]
    a24, r24 = np.clip(a24, 0, 1.5), np.clip(r24, 0, 1.5)
    # descriptive table: own peak x neighbours' outage fraction at the peak hour
    pb = [(.02, .05), (.05, .10), (.10, .25), (.25, 1.01)]; rb = [(0, .005), (.005, .02), (.02, .05), (.05, 1.01)]
    tab = {}
    print("A24 (weighted mean; n) by own peak (rows) x neighbours' outage fraction within 150 km at the peak hour (columns)", rb)
    for lo, hi in pb:
        row = []
        for l2, h2 in rb:
            f = sel0 & (peak >= lo) & (peak < hi) & (RF >= l2) & (RF < h2)
            row.append(dict(n=int(f.sum()), A24=float(np.average(a24[f], weights=w[f])) if f.sum() >= 8 else None, R24=float(np.average(r24[f], weights=w[f])) if f.sum() >= 8 else None))
        tab[f"{lo}-{hi}"] = row
        print(f"  peak [{lo:.2f},{hi:.2f}): " + " | ".join(f"n {q['n']:3d} A24 {q['A24']:.2f} R24 {q['R24']:.2f}" if q["A24"] is not None else f"n {q['n']:3d}   -   " for q in row))
    out["A24_by_own_peak_and_neighbour_fraction"] = dict(peak_bins=pb, neighbour_fraction_bins=rb, table=tab)
    print("neighbours' outage fraction at the peak hour of S units, weighted median by regime:", {q: round(wq(RF[sel0 & (peak >= .10) & (reg == q)], w[sel0 & (peak >= .10) & (reg == q)]), 4) for q in REGIMES})
    out["neighbour_fraction_S_median_by_regime"] = {q: wq(RF[sel0 & (peak >= .10) & (reg == q)], w[sel0 & (peak >= .10) & (reg == q)]) for q in REGIMES}
    fold = fold_of_unit(n)
    for pop, sel in (("S_all_regimes", sel0 & (peak >= .10)), ("peak_ge_2pct", sel0 & (peak >= .02))):
        ix = np.where(sel)[0]
        post = []
        for u, t in zip(ix, tpk[ix]):
            seg = Wh[u, t:t + 25]
            post.append([seg[:, wc.index("gust")].max(), seg[:, wc.index("wind_speed")].max(), seg[:, wc.index("gust_excess_energy")].max(),
                         seg[:, wc.index("precip")].sum(), seg[:, wc.index("snowfall")].sum(), seg[:, wc.index("t2m_c")].min()])
        Pb = np.column_stack([np.log(peak[ix]), tpk[ix], np.array(post, np.float32)]).astype(np.float32)
        bcols = list(s["Bcols"].astype(str)); regc = [j for j, c in enumerate(bcols) if c.startswith("reg_")]; other = [j for j in range(len(bcols)) if j not in regc]
        Rb = np.column_stack([np.log1p(RN[ix]), RF[ix], NH[ix], np.log1p(RNmax[ix]), RFmax[ix], np.log(base_ex[ix])]).astype(np.float32)
        Rp = Rb[np.random.default_rng(BOOT_SEED).permutation(len(ix))]
        blocks = {"P": Pb, "Bo": s["B"][ix][:, other], "Reg": s["B"][ix][:, regc], "R": Rb, "Rp": Rp}
        variants = {"P": ["P", "Bo"], "P+regime": ["P", "Bo", "Reg"], "P+region": ["P", "Bo", "R"], "P+region_perm": ["P", "Bo", "Rp"], "P+regime+region": ["P", "Bo", "Reg", "R"]}
        f, ws = fold[ix], w[ix]
        keys = sorted(set(zip(reg[ix], fam[ix]))); gid = {kk: q for q, kk in enumerate(keys)}
        g = np.array([gid[kk] for kk in zip(reg[ix], fam[ix])]); members = [np.where(g == q)[0] for q in range(len(keys))]
        draws = [np.concatenate([members[q] for q in d]) for d in cluster_draws(keys, np.random.default_rng(BOOT_SEED))]
        res = dict(units=int(len(ix)))
        for tname, tgt in (("A24", a24[ix]), ("R24", r24[ix])):
            preds = {}
            for name, bl in variants.items():
                X = np.concatenate([blocks[q] for q in bl], 1); P = np.full(len(ix), np.nan)
                for k in range(1, 6):
                    tr, te = f != k, f == k
                    P[te] = lgb.train(EVENT_PARAMS, lgb.Dataset(X[tr], tgt[tr], weight=ws[tr]), EVENT_ROUNDS).predict(X[te])
                preds[name] = P

            def rmse(P, j):
                return float(np.sqrt(np.average((P[j] - tgt[j]) ** 2, weights=ws[j])))

            al = np.arange(len(ix)); mu = float(np.average(tgt, weights=ws)); var = float(np.average((tgt - mu) ** 2, weights=ws))
            d = dict(mean=mu, r2={k: float(1 - rmse(P, al) ** 2 / var) for k, P in preds.items()}, differences={})
            for x, b in (("P+region", "P+region_perm"), ("P+region", "P"), ("P+regime", "P"), ("P+regime+region", "P+regime"), ("P+regime+region", "P+region")):
                bs = [1 - rmse(preds[x], j) / rmse(preds[b], j) for j in draws]
                d["differences"][f"{x} vs {b}"] = dict(point=float(1 - rmse(preds[x], al) / rmse(preds[b], al)), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
            res[tname] = d
            print(pop, tname, f"n {len(ix)} | R2", {k: round(v, 3) for k, v in d["r2"].items()})
            print("   ", {k: f"{100 * v['point']:+.2f}% [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}]" for k, v in d["differences"].items()})
        out[pop] = res
    (RES / "restore_region.json").write_text(json.dumps(out, indent=1) + "\n")

def sampled_neighbours(radius_km=150.0):
    """Row-normalised sparse matrix over the sampled county-events: the other sampled county-events of the same system
    within the radius and the unit itself, weighted by customers over the county selection probability."""
    import pandas as pd
    from scipy import sparse
    z = np.load(FEAT, allow_pickle=False)
    fips, sysv, cust, pic = z["fips"].astype(str), z["system"].astype(str), z["cust"].astype(float), z["pi_c"].astype(float)
    gz = pd.read_csv(ROOT / "data" / "raw" / "census" / "2020_Gaz_counties_national.txt", sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    gz.columns = [c.strip() for c in gz.columns]
    pos = {g.zfill(5): (a, b) for g, a, b in zip(gz.GEOID, gz.INTPTLAT, gz.INTPTLONG)}
    la = np.radians([pos[f][0] for f in fips]); lo = np.radians([pos[f][1] for f in fips])
    rows, cols, vals = [], [], []
    for q in np.unique(sysv):
        ii = np.where(sysv == q)[0]
        d = 2 * 6371.0 * np.arcsin(np.sqrt(np.sin((la[ii][:, None] - la[ii][None, :]) / 2) ** 2 + np.cos(la[ii][:, None]) * np.cos(la[ii][None, :]) * np.sin((lo[ii][:, None] - lo[ii][None, :]) / 2) ** 2))
        a, b = np.where(d <= radius_km)
        rows.append(ii[a]); cols.append(ii[b]); vals.append((cust / pic)[ii[b]])
    W = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(fips), len(fips)))
    return sparse.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W


def simregion() -> None:
    """Post-hoc simulation on the host's held-out rates (seed 0): restoration slowed by the regional outage burden,
    r / (1 + kappa g(burden)). oracle: the observed outage fraction of all other counties within 150 km in the same hour
    (not available to a forecast); forecast: the model's own stock of the sampled county-events within 150 km in the
    previous hour (coupled recursion, uses nothing but the host's rates)."""
    from scipy.stats import spearmanr
    s, y, m, w, reg, peak, tpk, ok = base(); n = len(y)
    z = np.load(FEAT, allow_pickle=False); y0 = z["y0"].astype(float); cust = z["cust"].astype(float)
    P0, U, R, _, _ = collect("v1_host_s0", n)
    r = region_table(); rf = r["out_ex"] / np.maximum(r["base_in"] - cust, 1.0)[:, None]
    W = sampled_neighbours()
    S = peak >= .10; A = np.ones(n, bool); cnt = m.sum(1); fold = fold_of_unit(n); a = np.arange(n)
    obs_nb = W @ np.where(m > 0, y, 0.0); pred_nb = W @ P0
    rf_in = r["out_in"] / np.maximum(r["base_in"], 1.0)[:, None]
    chk = dict(sampled_vs_full_region_spearman_at_peak=float(spearmanr(obs_nb[a, tpk][peak >= .02], rf_in[a, tpk][peak >= .02]).statistic),
               host_predicted_vs_observed_region_spearman_at_peak=float(spearmanr(pred_nb[a, tpk][peak >= .02], rf_in[a, tpk][peak >= .02]).statistic),
               host_predicted_vs_observed_region_spearman_at_peak_S=float(spearmanr(pred_nb[a, tpk][S], rf_in[a, tpk][S]).statistic),
               window_max_spearman_all_units=float(spearmanr(pred_nb.max(1), rf_in.max(1)).statistic),
               median_ratio_predicted_over_observed_at_peak_S=float(np.median(pred_nb[a, tpk][S] / np.maximum(rf_in[a, tpk][S], 1e-6))))
    print("regional burden, host prediction vs observation:", chk)

    def roll(kappa, mode, g):
        p = y0.copy(); out = np.empty_like(U)
        for t in range(144):
            b = rf[:, t] if mode == "oracle" else W @ p
            p = np.clip(p + U[:, t] * (1 - p) - R[:, t] * p / (1 + kappa * g(b)), 0, 1); out[:, t] = p
        return out

    def rmse(P, sub):
        return float(np.sqrt((w * (m * (P - y) ** 2).sum(1))[sub].sum() / (w * cnt)[sub].sum()))

    def row(P):
        pk = P.max(1)
        return dict(S=1 - rmse(P, S) / rmse(P0, S), all=1 - rmse(P, A) / rmse(P0, A), nonS=1 - rmse(P, ~S) / rmse(P0, ~S), false_peaks=int(((pk >= .1) & ~S).sum()),
                    S_by_regime={q: 1 - rmse(P, S & (reg == q)) / rmse(P0, S & (reg == q)) for q in REGIMES},
                    all_by_regime={q: 1 - rmse(P, reg == q) / rmse(P0, reg == q) for q in REGIMES})

    pc = lambda d: f"S {100 * d['S']:+.2f}% all {100 * d['all']:+.2f}% nonS {100 * d['nonS']:+.2f}% FP {d['false_peaks']}"  # noqa: E731
    out = {"burden_check": chk}
    grids = {"sqrt": [0, 2, 5, 10, 20, 40, 80], "linear": [0, 10, 30, 100, 300, 1000]}
    for mode in ("oracle", "forecast"):
        for gname, g in (("sqrt", np.sqrt), ("linear", lambda b: b)):
            sims = {k: roll(float(k), mode, g) for k in grids[gname]}
            res = {"grid": {str(k): row(P) for k, P in sims.items()}}
            Pc, ch = np.empty_like(P0), {}
            for k in range(1, 6):
                ch[k] = min(grids[gname], key=lambda q: rmse(sims[q], fold != k)); Pc[fold == k] = sims[ch[k]][fold == k]
            res["cross_fitted"] = dict(chosen=ch, **row(Pc)); out[f"{mode}_{gname}"] = res
            print(f"== {mode} / {gname}")
            for k in grids[gname]:
                print(f"   kappa {k:5d}: {pc(res['grid'][str(k)])}")
            cf = res["cross_fitted"]
            print(f"   cross-fitted {ch}: {pc(cf)} | S by regime", {q: f"{100 * v:+.1f}%" for q, v in cf["S_by_regime"].items()}, "| all by regime", {q: f"{100 * v:+.1f}%" for q, v in cf["all_by_regime"].items()})
    (RES / "restore_region_sim.json").write_text(json.dumps(out, indent=1) + "\n")

def predictable() -> None:
    """How predictable is the regional burden before the storm? Event-level LightGBM under the event folds: the window
    maximum of the outage fraction of all counties within 150 km (own county included) against the county's own peak, from
    the county's own ERA5 summaries (M1 blocks) and from those plus their average over the sampled neighbours."""
    import lightgbm as lgb
    from scipy.stats import spearmanr
    s, y, m, w, reg, peak, tpk, ok = base(); n = len(y)
    r = region_table(); W = sampled_neighbours()
    rf_in = r["out_in"] / np.maximum(r["base_in"], 1.0)[:, None]
    tg = {"own_peak": np.log(peak + 1e-3), "regional_peak": np.log(rf_in.max(1) + 1e-3)}
    Xo = np.concatenate([s["B"], s["We"]], 1).astype(np.float32); Xn = np.concatenate([Xo, (W @ s["We"].astype(float)).astype(np.float32)], 1)
    fold = fold_of_unit(n); sysv = s["system"].astype(str); out = {}
    for tn, t in tg.items():
        for xn, X in (("own_weather", Xo), ("own+neighbour_weather", Xn)):
            P = np.empty(n)
            for k in range(1, 6):
                P[fold == k] = lgb.train(EVENT_PARAMS, lgb.Dataset(X[fold != k], t[fold != k], weight=w[fold != k]), EVENT_ROUNDS).predict(X[fold == k])
            mu = np.average(t, weights=w); r2 = 1 - np.average((P - t) ** 2, weights=w) / np.average((t - mu) ** 2, weights=w)
            sm = np.array([t[sysv == q].mean() for q in np.unique(sysv)]); pm = np.array([P[sysv == q].mean() for q in np.unique(sysv)])
            out[f"{tn}|{xn}"] = dict(r2=float(r2), spearman=float(spearmanr(P, t).statistic), system_mean_spearman=float(spearmanr(pm, sm).statistic),
                                     severe_recall_top10pct=float(np.mean(P[t >= np.quantile(t, .9)] >= np.quantile(P, .9))))
            print(tn, xn, {k: round(v, 3) for k, v in out[f"{tn}|{xn}"].items()}, flush=True)
    (RES / "restore_region_predictability.json").write_text(json.dumps(out, indent=1) + "\n")

def rolling(rad=None) -> None:
    """Registered tests R3 and R3b: rolling-origin restoration forecast from held-out rates (seed 0). With a radius
    (km, one of RADII) the burden comes from region_table_multi and only the main variants are run (robustness)."""
    s, y, m, w, reg, peak, tpk, ok = base(); n = len(y)
    z = np.load(FEAT, allow_pickle=False); cust = z["cust"].astype(float)
    fam = s["family"].astype(str)
    _, U, R, _, _ = collect("v1_host_s0", n)
    _, Ud, Rd, fd, _ = collect("v1_dkv_s0", n)
    if rad is None:
        r = region_table(); rf = r["out_ex"] / np.maximum(r["base_in"] - cust, 1.0)[:, None]
    else:
        r = region_table_multi(); rf = r[f"out_ex_{rad}"][:, 24:] / np.maximum(r[f"base_ex_{rad}"], 1.0)[:, None]
    fold = fold_of_unit(n); H = 48; origins = [24, 48, 72, 96]
    rng = np.random.default_rng(BOOT_SEED)
    cases = []                                                    # (units, origin, initial stock, burden, permuted burden)
    for d in origins:
        ii = np.where(m[:, d - 1] > 0)[0]
        cases.append((ii, d, y[ii, d - 1], rf[ii, d - 1], rf[ii, d - 1][rng.permutation(len(ii))]))

    def sse(kind, par, only=None):
        """per-unit (sum of squared error, cells) pooled over the origins; all cases and cases with initial stock >= 1%"""
        a = np.zeros((2, n)); c = np.zeros((2, n))
        Uq, Rq = (Ud, Rd) if kind == "D" else (U, R)
        for ii, d, p0, b, bp in cases:
            if only is not None and d != only:
                continue
            p = p0.copy(); e = np.zeros(len(ii))
            for t in range(d, d + H):
                if kind == "persist":
                    q = p0
                else:
                    if kind in ("GR", "GRp"):
                        den = (1 + par[0]) * (1 + par[1] * (b if kind == "GR" else bp))
                    elif kind in ("LR", "LRp"):
                        den = (1 + par[0] * Rq[ii, t] * p) * (1 + par[1] * (b if kind == "LR" else bp))
                    else:
                        den = {"H": 1.0, "D": 1.0, "G": 1 + par[0], "L": 1 + par[0] * Rq[ii, t] * p}[kind]
                    p = np.clip(p + Uq[ii, t] * (1 - p) - Rq[ii, t] * p / den, 0, 1); q = p
                e += m[ii, t] * (q - y[ii, t]) ** 2
            k = m[ii, d:d + H].sum(1)
            a[0, ii] += e; c[0, ii] += k
            act = p0 >= .01
            a[1, ii[act]] += e[act]; c[1, ii[act]] += k[act]
        return a, c

    grids = {"H": [(0,)], "persist": [(0,)], "G": [(k,) for k in (0, .25, .5, 1, 2, 4, 8)], "L": [(k,) for k in (0, 5, 10, 20, 40, 80, 160, 320)],
             "GR": [(a, b) for a in (0, .5, 1, 2, 4) for b in (0, 10, 30, 100, 300)], "LR": [(a, b) for a in (0, 20, 40, 80, 160) for b in (0, 10, 30, 100, 300)]}
    grids["GRp"] = grids["GR"]; grids["LRp"] = grids["LR"]
    if len(fd) == 5:
        grids["D"] = [(0,)]
    if rad is not None:
        grids = {k: grids[k] for k in ("H", "G", "L", "GR", "LR")}
    cnt = None; cf = {}; chosen = {}; by_origin = {}
    for kind, g in grids.items():
        res = {par: sse(kind, par) for par in g}
        cnt = res[g[0]][1]
        A = np.zeros((2, n)); ch = {}
        for k in range(1, 6):
            tr = fold != k
            best = min(g, key=lambda par: (w * res[par][0][0])[tr].sum())
            ch[k] = best; A[:, fold == k] = res[best][0][:, fold == k]
        cf[kind] = A; chosen[kind] = {k: list(v) for k, v in ch.items()}
        if kind in ("H", "GR"):
            for d in origins:
                Ad = np.zeros(n)
                for k in range(1, 6):
                    Ad[fold == k] = sse(kind, ch[k], only=d)[0][0][fold == k]
                by_origin[(kind, d)] = Ad
        print(kind, "chosen", chosen[kind], flush=True)
    keys = sorted(set(zip(reg, fam))); gid = {kk: j for j, kk in enumerate(keys)}; g = np.array([gid[kk] for kk in zip(reg, fam)])
    counts = np.zeros((2000, len(keys)))
    for q in REGIMES:
        js = np.array([gid[kk] for kk in keys if kk[0] == q]); counts[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), size=2000)

    def rel(a, b, sub=None):
        sub = np.ones(n, bool) if sub is None else sub
        def gs(v):
            o = np.zeros(len(keys)); np.add.at(o, g[sub], (w * v)[sub]); return o
        qa, qb = counts @ gs(a), counts @ gs(b); okb = qb > 0
        d = 1 - np.sqrt(qa[okb] / qb[okb])
        return dict(point=float(1 - np.sqrt((w * a)[sub].sum() / (w * b)[sub].sum())), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])

    out = {"origins": origins, "horizon_h": H, "cases": int(sum(len(c[0]) for c in cases)), "active_cases": int(sum((c[2] >= .01).sum() for c in cases)), "chosen": chosen, "rmse": {}, "comparisons": {}}
    for kind in grids:
        out["rmse"][kind] = dict(all=float(np.sqrt((w * cf[kind][0]).sum() / (w * cnt[0]).sum())), active=float(np.sqrt((w * cf[kind][1]).sum() / (w * cnt[1]).sum())))
    f = lambda x: f"{100 * x['point']:+.2f}% [{100 * x['ci95'][0]:+.2f}, {100 * x['ci95'][1]:+.2f}]"  # noqa: E731
    pairs = [("GR", "H"), ("GR", "G"), ("GR", "GRp"), ("G", "H"), ("L", "H"), ("GR", "L"), ("LR", "L"), ("LR", "LRp"), ("LR", "H"), ("LR", "GR"), ("H", "persist"), ("LR", "persist")]
    if "D" in grids:
        pairs += [("D", "H"), ("GR", "D"), ("LR", "D")]
    pairs = [q for q in pairs if q[0] in grids and q[1] in grids]
    for a, b in pairs:
        d = dict(all=rel(cf[a][0], cf[b][0]), active=rel(cf[a][1], cf[b][1]),
                 all_by_regime={q: rel(cf[a][0], cf[b][0], reg == q)["point"] for q in REGIMES}, S_units=rel(cf[a][0], cf[b][0], peak >= .10))
        out["comparisons"][f"{a} vs {b}"] = d
        print(f"{a} vs {b}: all {f(d['all'])} | active {f(d['active'])} | S units {f(d['S_units'])} | by regime", {q: f"{100 * v:+.1f}%" for q, v in d["all_by_regime"].items()})
    out["GR_vs_H_by_origin"] = {str(d): rel(by_origin[("GR", d)], by_origin[("H", d)]) for d in origins}
    print("GR vs H by origin:", {k: f(v) for k, v in out["GR_vs_H_by_origin"].items()})
    print("rmse", {k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in out["rmse"].items()}, "cases", out["cases"], "active", out["active_cases"])
    (RES / ("restore_rolling.json" if rad is None else f"restore_rolling_r{rad}.json")).write_text(json.dumps(out, indent=1) + "\n")


def radius() -> None:
    """Robustness to the radius of the regional burden: the R2 learner with at-peak-hour features only, and the rolling test."""
    import lightgbm as lgb
    s, y, m, w, reg, peak, tpk, ok = base(); n = len(y); a = np.arange(n)
    r = region_table_multi(); fold = fold_of_unit(n)
    sel0 = ok & np.array([m[u, t + 1:min(t + 25, 144)].sum() >= 12 for u, t in zip(a, tpk)]) & (peak >= .10)
    ix = np.where(sel0)[0]
    r24 = np.clip(y[ix, tpk[ix] + 24] / peak[ix], 0, 1.5)
    a24 = np.clip(np.array([(m[u, t + 1:t + 25] * y[u, t + 1:t + 25]).sum() / m[u, t + 1:t + 25].sum() for u, t in zip(ix, tpk[ix])]) / peak[ix], 0, 1.5)
    flat = float(np.mean([np.ptp(y[u, t + 1:t + 13]) == 0 for u, t in zip(ix, tpk[ix])]))
    Wh = np.load(OUT / "Wh.npy"); wc = list(s["Whcols"].astype(str))
    post = []
    for u, t in zip(ix, tpk[ix]):
        seg = Wh[u, t:t + 25]
        post.append([seg[:, wc.index("gust")].max(), seg[:, wc.index("wind_speed")].max(), seg[:, wc.index("gust_excess_energy")].max(),
                     seg[:, wc.index("precip")].sum(), seg[:, wc.index("snowfall")].sum(), seg[:, wc.index("t2m_c")].min()])
    bcols = list(s["Bcols"].astype(str)); other = [j for j, c in enumerate(bcols) if not c.startswith("reg_")]
    Pb = np.column_stack([np.log(peak[ix]), tpk[ix], np.array(post, np.float32), s["B"][ix][:, other]]).astype(np.float32)
    ws, f = w[ix], fold[ix]
    out = {"S_units": int(len(ix)), "share_flat_12h_after_peak": flat, "r2": {}}

    def fit(X, tgt):
        P = np.empty(len(ix))
        for k in range(1, 6):
            P[f == k] = lgb.train(EVENT_PARAMS, lgb.Dataset(X[f != k], tgt[f != k], weight=ws[f != k]), EVENT_ROUNDS).predict(X[f == k])
        mu = np.average(tgt, weights=ws)
        return float(1 - np.average((P - tgt) ** 2, weights=ws) / np.average((tgt - mu) ** 2, weights=ws))

    out["r2"]["no_region"] = dict(A24=fit(Pb, a24), R24=fit(Pb, r24))
    for rad in RADII:
        o, b = r[f"out_ex_{rad}"][ix, tpk[ix] + 24], np.maximum(r[f"base_ex_{rad}"][ix], 1.0)
        X = np.column_stack([Pb, np.log1p(o), o / b, np.log(b)]).astype(np.float32)
        out["r2"][str(rad)] = dict(A24=fit(X, a24), R24=fit(X, r24))
    print("flat share", round(flat, 4), "| out-of-event R2 with the burden at the peak hour only:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out["r2"].items()}, flush=True)
    (RES / "restore_radius.json").write_text(json.dumps(out, indent=1) + "\n")
    for rad in RADII:
        print(f"== rolling test with radius {rad} km", flush=True); rolling(rad)

def export() -> None:
    """Ring burdens for the restoration kernel: data/interim/panel_v1/burden_v1D.npz, rings [N, 168, 3] = outage fraction
    of the other counties at 0-50, 50-150 and 150-300 km, hourly from 24 h before the origin (col_origin = 24)."""
    z = np.load(FEAT, allow_pickle=False); r = region_table_multi()
    edges = (50, 150, 300); rings = []
    for j, e in enumerate(edges):
        o = r[f"out_ex_{e}"].astype(np.float64) - (r[f"out_ex_{edges[j - 1]}"] if j else 0.0)
        b = r[f"base_ex_{e}"] - (r[f"base_ex_{edges[j - 1]}"] if j else 0.0)
        rings.append(np.where(b[:, None] > 0, o / np.maximum(b[:, None], 1.0), 0.0))
    rings = np.clip(np.stack(rings, -1), 0.0, 1.0).astype(np.float32)
    f = ROOT / "data" / "interim" / "panel_v1" / "burden_v1D.npz"
    np.savez_compressed(f, fips=z["fips"], system=z["system"], rings=rings, ring_km=np.array(edges), col_origin=np.array(24))
    print(f.name, rings.shape, "mean ring fractions at the origin hour - 1:", rings[:, 23].mean(0).round(5).tolist(), "at +47 h:", rings[:, 71].mean(0).round(5).tolist())


if __name__ == "__main__":
    {"size": size, "sim": sim, "geo": geo, "region": region, "simregion": simregion, "predictable": predictable, "rolling": rolling, "multi": region_table_multi, "radius": radius, "export": export}[sys.argv[1]]()
