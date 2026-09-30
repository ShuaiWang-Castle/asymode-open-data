"""Information-ceiling probe (notes/INFO_CEILING_PROBE_SCOPE_20260930_ZH.md): how far can a flexible non-neural model,
given nested sets of public information, lower the error on the severe county-events S under the same event folds as
the kernels? No neural training; no existing source, checkpoint or result is modified; public D only; sealed C and
paper_v1/ are not read.

  python probe.py build        feature tables (cached in runs/geo_weather_20260924/info_ceiling_20260930/)
  python probe.py event        event-level LightGBM probes (peak magnitude and S membership)
  python probe.py traj [names] county-hour LightGBM trajectory probes (all variants by default)
  python probe.py score        metrics against the host, family-cluster intervals -> results/v1/info_ceiling/
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "3")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
FEAT = ROOT / "data" / "interim" / "panel_v1" / "features_v1D.npz"
HRRR = ROOT / "data" / "interim" / "geo_weather" / "eih_v1D_hrrr2.npz"
NODES = ROOT / "data" / "interim" / "panel_v1" / "nodes_geo_D.parquet"
CUST = ROOT / "data" / "interim" / "eaglei_county_customers_2024.parquet"
EAGLEI = [ROOT / "data" / "interim" / f"eaglei_outages_{y}.parquet" for y in (2014, 2015, 2016, 2017)]
SPLITS = EXP / "splits_v1D.json"
RUNS = ROOT / "runs" / "geo_weather_20260924"
OUT = RUNS / "info_ceiling_20260930"
RES = EXP / "results" / "v1" / "info_ceiling"
SEED, PERM_SEED, BOOT_SEED, B = 20260930, 20260930, 20260924, 2000
THREADS = 3
REGIMES = ["tropical", "winter", "synoptic_wind", "convective", "heavy_rain"]
VARIANTS = {"M0": ["B"], "M1": ["B", "W"], "M2": ["B", "W", "G"], "M2p": ["B", "W", "Gp"], "M3": ["B", "W", "G", "C"],
            "M4": ["B", "W", "G", "C", "H"], "M4p": ["B", "W", "G", "C", "Hp"], "M5": ["B", "W", "G", "C", "H", "R"],
            "M1H": ["B", "W", "H"], "M1R": ["B", "W", "R"]}
TRAJ_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED,
                   num_threads=THREADS, verbose=-1)
EVENT_PARAMS = dict(TRAJ_PARAMS, min_data_in_leaf=40)
TRAJ_ROUNDS, EVENT_ROUNDS = 400, 300


# ----------------------------------------------------------------------------------------------- static blocks
def historical_susceptibility(fips_needed: np.ndarray) -> pd.DataFrame:
    """County outage record in EAGLE-I 2014-2017 (before the frame starts on 2018-07-12): daily maxima of customers out
    over the 2024 modelled customers; counts of days above 1/5/10/25%, the largest days, and the coverage."""
    f = OUT / "hist_2014_2017.parquet"
    if f.exists():
        return pd.read_parquet(f)
    cust = pd.read_parquet(CUST)["customers"]
    need = set(fips_needed.tolist())
    parts = []
    for p in EAGLEI:
        d = pd.read_parquet(p, columns=["fips", "ts", "customers_out"])
        d = d[d["fips"].isin(need)]
        d["day"] = d["ts"].dt.floor("D")
        g = d.groupby(["fips", "day"], observed=True)["customers_out"].max().reset_index()
        g["year"] = g["day"].dt.year
        parts.append(g)
        print("history", p.name, len(d), "rows ->", len(g), "county-days", flush=True)
        del d
    g = pd.concat(parts, ignore_index=True)
    g["frac"] = (g["customers_out"].astype(float) / g["fips"].map(cust).astype(float)).clip(upper=1.0)
    rows = []
    for fp, x in g.groupby("fips", observed=True):
        v = np.sort(x["frac"].to_numpy(dtype=float))[::-1]
        yrs = x["year"].nunique()
        rows.append(dict(fips=fp, h_days=len(v), h_years=yrs, h_n1=int((v >= .01).sum()), h_n5=int((v >= .05).sum()),
                         h_n10=int((v >= .10).sum()), h_n25=int((v >= .25).sum()), h_n10_rate=float((v >= .10).sum() / max(yrs, 1)),
                         h_max=float(v[0]), h_top5=float(v[:5].mean()), h_q99=float(np.quantile(v, .99))))
    h = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    h.to_parquet(f)
    return h


def exposure_geography() -> pd.DataFrame:
    """Population-weighted descriptors at the 3 km nodes: canopy, share of people under dense canopy, poorly drained
    share, share of people on poorly drained ground, elevation and its spread among the population."""
    n = pd.read_parquet(NODES)
    n["w"] = n["w_pop"] / n.groupby("fips")["w_pop"].transform("sum")
    def agg(x):
        w = x["w"].to_numpy(); z = x["z"].to_numpy()
        mz = float((w * z).sum())
        return pd.Series(dict(ge_canopy=float((w * x["canopy"]).sum()), ge_canopy_hi=float((w * (x["canopy"] >= 50)).sum()),
                              ge_wet=float((w * x["wet"]).sum()), ge_wet_hi=float((w * (x["wet"] >= .25)).sum()),
                              ge_z=mz, ge_z_sd=float(np.sqrt(max((w * (z - mz) ** 2).sum(), 0.0)))))
    return n.groupby("fips").apply(agg).reset_index()


def county_permutation(fips: np.ndarray) -> dict:
    counties = np.unique(fips)
    return dict(zip(counties, np.random.default_rng(PERM_SEED).permutation(counties)))


# ----------------------------------------------------------------------------------------------- build
def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(FEAT)
    N = len(z["fips"])
    fips, reg = z["fips"].astype(str), z["regime"].astype(str)
    rn = list(z["recovery_features"].astype(str)); dn = list(z["damage_features"].astype(str))
    xu, xr = z["xu"], z["xr"]
    # B: origin state, customers, regime, season
    month = pd.to_datetime(pd.Series(z["origin"].astype(str))).dt.month.to_numpy()
    Bcols = ["y0"] + [f"prefix_{c}" for c in ("p_max_prefix", "p_mean_66_71", "p_trend_65_71", "prefix_active_share")] + \
            ["log_cust"] + [f"reg_{r}" for r in REGIMES] + ["month_sin", "month_cos"]
    Bm = np.column_stack([z["y0"]] + [xr[:, 72, rn.index(c)] for c in ("p_max_prefix", "p_mean_66_71", "p_trend_65_71",
                                                                         "prefix_active_share")] +
                         [xr[:, 72, rn.index("log_cust")]] + [(reg == r).astype(float) for r in REGIMES] +
                         [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]).astype(np.float32)
    # G: geo40 + exposure at the nodes; Gp: county-level donor permutation of both
    ge = exposure_geography().set_index("fips")
    gecols = list(ge.columns)
    Ge = ge.reindex(fips).to_numpy(dtype=np.float32)
    Gm = np.concatenate([z["geo"].astype(np.float32), Ge], 1)
    Gcols = [f"geo_{c}" for c in z["geo_features"].astype(str)] + gecols
    donor = county_permutation(fips)
    first = {f: i for i, f in reversed(list(enumerate(fips)))}
    Gp = np.stack([Gm[first[donor[f]]] for f in fips])
    # C: county context without the customer count (already in B)
    Ccols = ["rucc", "log_pop_density", "coop_share", "log1p_n_utilities", "log1p_saidi"]
    Cm = np.column_stack([xr[:, 72, rn.index(c)] for c in Ccols]).astype(np.float32)
    # H: history 2014-2017; Hp: the same county-level permutation
    h = historical_susceptibility(np.unique(fips)).set_index("fips")
    Hcols = list(h.columns)
    Hm = h.reindex(fips).to_numpy(dtype=np.float32)
    Hp = np.stack([Hm[first[donor[f]]] for f in fips])
    # W hourly: the 42 damage inputs, 8 neighbour summaries, cumulative window quantities, hours since the origin
    win = slice(72, 216)
    xw = xu[:, win, :].astype(np.float32)
    nb = xr[:, win, rn.index("nbr_gust_mean"):rn.index("nbr_soil_moisture_max") + 1].astype(np.float32)
    cum = []
    for c, op in (("gust", "max"), ("precip", "sum"), ("snowfall", "sum"), ("gust_excess_energy", "max"),
                  ("near_freeze", "sum"), ("snow_ice_load", "max"), ("wind_speed", "max"), ("wet_wind", "sum")):
        v = xw[:, :, dn.index(c)]
        cum.append(np.maximum.accumulate(v, 1) if op == "max" else np.cumsum(v, 1))
    lead = np.broadcast_to(np.arange(144, dtype=np.float32), (N, 144))
    Wh = np.concatenate([xw, nb, np.stack(cum, -1), lead[..., None]], -1)
    Wcols = dn + [rn[i] for i in range(rn.index("nbr_gust_mean"), rn.index("nbr_soil_moisture_max") + 1)] + \
            ["cum_max_gust", "cum_sum_precip", "cum_sum_snowfall", "cum_max_gust_excess", "cum_sum_near_freeze",
             "cum_max_snow_ice_load", "cum_max_wind", "cum_sum_wet_wind", "lead"]
    # R hourly: HRRR weather-only hazards (no canopy or drainage products) and four running maxima
    zh = np.load(HRRR)
    assert np.array_equal(zh["fips"].astype(str), fips) and np.array_equal(zh["event"].astype(str), z["event"].astype(str))
    names = list(zh["names"].astype(str))
    keep = [i for i, n in enumerate(names) if not (n.startswith("canopy*") or n.startswith("drain*"))]
    ph = zh["phi"][:, :, keep].astype(np.float32)
    runmax = [np.maximum.accumulate(ph[:, :, keep.index(names.index(c))], 1) for c in ("g_exc_max", "ice_max", "wetsnow_max", "uh_max")]
    Rh = np.concatenate([ph, np.stack(runmax, -1)], -1)
    Rcols = [names[i] for i in keep] + ["cum_g_exc_max", "cum_ice_max", "cum_wetsnow_max", "cum_uh_max"]
    # event-level summaries of the hourly blocks
    def summ(a, cols, ops=("max", "mean")):
        out, oc = [], []
        for op in ops:
            out.append((np.nanmax if op == "max" else np.nanmean)(a, axis=1)); oc += [f"{c}_{op}" for c in cols]
        return np.concatenate(out, 1).astype(np.float32), oc
    We, Wecols = summ(Wh[:, :, :-1], Wcols[:-1])
    Re, Recols = summ(Rh, Rcols)
    y = z["y"].astype(np.float32); m = z["m"].astype(np.float32)
    yo = np.where(z["obs_full"][:, win], z["y_full"][:, win], np.nan)
    peak = np.nanmax(np.where(np.isnan(yo), -1.0, yo), 1).astype(np.float32)
    np.savez(OUT / "static.npz", B=Bm, G=Gm, Gp=Gp, C=Cm, H=Hm, Hp=Hp, We=We, Re=Re, peak=peak, y=y, m=m,
             w=z["w"].astype(np.float32), fips=fips, regime=reg, family=z["family"].astype(str), system=z["system"].astype(str),
             Bcols=np.array(Bcols), Gcols=np.array(Gcols), Ccols=np.array(Ccols), Hcols=np.array(Hcols),
             Wecols=np.array(Wecols), Recols=np.array(Recols), Whcols=np.array(Wcols), Rhcols=np.array(Rcols))
    np.save(OUT / "Wh.npy", Wh.astype(np.float32)); np.save(OUT / "Rh.npy", Rh.astype(np.float32))
    cov = dict(units=N, hist_counties=int(np.isfinite(Hm[:, 0]).sum()), hist_missing_units=int(np.isnan(Hm[:, 0]).sum()),
               exposure_missing_units=int(np.isnan(Ge[:, 0]).sum()), W_hourly=len(Wcols), R_hourly=len(Rcols),
               S_units=int((peak >= .10).sum()))
    (OUT / "build_manifest.json").write_text(json.dumps(cov, indent=1) + "\n")
    print("built", cov, flush=True)


# ----------------------------------------------------------------------------------------------- probes
def load_static():
    s = np.load(OUT / "static.npz", allow_pickle=False)
    return {k: s[k] for k in s.files}


def folds():
    sp = json.loads(SPLITS.read_text())
    return [(k, np.array(sp["event"][str(k)]["dev"]), np.array(sp["event"][str(k)]["outer"])) for k in range(1, 6)]


def event_matrix(s, blocks):
    parts = []
    for b in blocks:
        parts.append({"B": s["B"], "W": s["We"], "G": s["G"], "Gp": s["Gp"], "C": s["C"], "H": s["H"], "Hp": s["Hp"],
                      "R": s["Re"]}[b])
    return np.concatenate(parts, 1)


def run_event() -> None:
    import lightgbm as lgb
    s = load_static()
    peak, w = s["peak"].astype(float), s["w"].astype(float)
    target = np.log(np.clip(peak, 0, None) + 1e-3)
    sev = (peak >= .10).astype(float)
    res = {}
    for name, blocks in VARIANTS.items():
        X = event_matrix(s, blocks)
        pk, pr = np.full(len(peak), np.nan), np.full(len(peak), np.nan)
        for k, dev, outer in folds():
            ok = np.isfinite(target[dev]) & (peak[dev] >= 0)
            reg = lgb.train(EVENT_PARAMS, lgb.Dataset(X[dev][ok], target[dev][ok], weight=w[dev][ok]), EVENT_ROUNDS)
            pk[outer] = np.clip(np.exp(reg.predict(X[outer])) - 1e-3, 0, 1)
            clf = lgb.train(dict(EVENT_PARAMS, objective="binary"), lgb.Dataset(X[dev][ok], sev[dev][ok], weight=w[dev][ok]), EVENT_ROUNDS)
            pr[outer] = clf.predict(X[outer])
        res[name] = dict(peak=pk, prob=pr)
        print("event", name, X.shape, flush=True)
    np.savez(OUT / "event_oof.npz", **{f"{n}_peak": v["peak"] for n, v in res.items()}, **{f"{n}_prob": v["prob"] for n, v in res.items()})


def traj_rows(s, Wh, Rh, blocks, units, even_only):
    hrs = np.arange(0, 144, 2) if even_only else np.arange(144)
    ui, ti = np.repeat(units, len(hrs)), np.tile(hrs, len(units))
    parts = []
    for b in blocks:
        if b == "W":
            parts.append(Wh[ui, ti])
        elif b == "R":
            parts.append(Rh[ui, ti])
        else:
            parts.append(s[b][ui])
    return np.concatenate(parts, 1), ui, ti


def run_traj(names: list[str]) -> None:
    import lightgbm as lgb
    s = load_static()
    Wh, Rh = np.load(OUT / "Wh.npy", mmap_mode="r"), np.load(OUT / "Rh.npy", mmap_mode="r")
    Wh, Rh = np.asarray(Wh), np.asarray(Rh)
    y, m, w = s["y"], s["m"], s["w"]
    for name in names:
        f = OUT / f"traj_{name}.npy"
        if f.exists():
            print("traj", name, "exists", flush=True); continue
        P = np.full(y.shape, np.nan, np.float32)
        t0 = time.time()
        for k, dev, outer in folds():
            X, ui, ti = traj_rows(s, Wh, Rh, VARIANTS[name], dev, True)
            ok = m[ui, ti] > 0
            ds = lgb.Dataset(X[ok], y[ui, ti][ok], weight=w[ui][ok], free_raw_data=True)
            bst = lgb.train(TRAJ_PARAMS, ds, TRAJ_ROUNDS)
            del X, ds
            Xo, uo, to = traj_rows(s, Wh, Rh, VARIANTS[name], outer, False)
            P[uo, to] = np.clip(bst.predict(Xo), 0, 1)
            del Xo
            print(f"traj {name} fold {k} done ({time.time() - t0:.0f} s)", flush=True)
        assert not np.isnan(P).any()
        np.save(f, P)


# ----------------------------------------------------------------------------------------------- scoring
def host_paths(n):
    sys.path.insert(0, str(EXP / "paper_v1")); sys.path.insert(0, str(EXP))
    from evaluate_paper import rollout
    per = [rollout(f"v1_host_s{s}", n) for s in range(5)]
    return np.mean(per, 0), per[0]


def score() -> None:
    s = load_static()
    y, m, w = s["y"].astype(float), s["m"].astype(float), s["w"].astype(float)
    reg, fam, peak = s["regime"], s["family"], s["peak"].astype(float)
    n = len(y); S = peak >= .10; nonS = ~S
    host5, host0 = host_paths(n)
    paths = {"host5": host5, "host0": host0}
    for name in VARIANTS:
        f = OUT / f"traj_{name}.npy"
        if f.exists():
            paths[name] = np.load(f).astype(float)
    e = {k: np.where(m > 0, P - y, 0.0) for k, P in paths.items()}
    sse = {k: (m * v ** 2).sum(1) for k, v in e.items()}
    cnt = m.sum(1)
    keys = sorted({(r, f) for r, f in zip(reg, fam)}); gid = {kk: j for j, kk in enumerate(keys)}
    g = np.array([gid[(r, f)] for r, f in zip(reg, fam)])
    rng = np.random.default_rng(BOOT_SEED); counts = np.zeros((B, len(keys)))
    for r in REGIMES:
        js = np.array([gid[kk] for kk in keys if kk[0] == r])
        counts[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), size=B)

    def gsum(v, sub):
        out = np.zeros(len(keys)); np.add.at(out, g[sub], (w * v)[sub]); return out

    def rmse(k, sub):
        return float(np.sqrt((w * sse[k])[sub].sum() / (w * cnt)[sub].sum()))

    def rel(a, b, sub):
        """improvement 1 - RMSE(a)/RMSE(b) on the subset, with its family-cluster 95% interval."""
        qa, qb, nn = counts @ gsum(sse[a], sub), counts @ gsum(sse[b], sub), counts @ gsum(cnt, sub)
        d = 1 - np.sqrt(qa / nn) / np.sqrt(qb / nn)
        return dict(point=1 - rmse(a, sub) / rmse(b, sub), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])

    out = {"S_units": int(S.sum()), "nonS_units": int(nonS.sum()), "metric": "full 144 h design-weighted RMSE; improvement = 1 - RMSE/RMSE_ref"}
    rows = {}
    for k in paths:
        pk = paths[k].max(1)
        rows[k] = dict(S_rmse=rmse(k, S), all_rmse=rmse(k, np.ones(n, bool)), nonS_rmse=rmse(k, nonS),
                       false_peaks=int(((pk >= .10) & nonS).sum()),
                       false_peak_weight_share=float(w[(pk >= .10) & nonS].sum() / w[nonS].sum()),
                       S_median_peak_ratio=float(np.median(pk[S] / peak[S])),
                       S_share_half_peak=float(np.mean(pk[S] >= .5 * peak[S])))
        if k not in ("host5", "host0"):
            rows[k]["S_vs_host5"] = rel(k, "host5", S); rows[k]["S_vs_host0"] = rel(k, "host0", S)
            rows[k]["all_vs_host5"] = rel(k, "host5", np.ones(n, bool)); rows[k]["nonS_vs_host5"] = rel(k, "host5", nonS)
            rows[k]["S_by_regime_vs_host5"] = {r: 1 - rmse(k, S & (reg == r)) / rmse("host5", S & (reg == r)) for r in REGIMES}
    out["trajectory"] = rows
    pairs = [("M1", "M0"), ("M2", "M1"), ("M2", "M2p"), ("M3", "M2"), ("M4", "M3"), ("M4", "M4p"), ("M5", "M4"),
             ("M1H", "M1"), ("M1R", "M1")]
    out["increments_S"] = {f"{a} vs {b}": rel(a, b, S) for a, b in pairs if a in paths and b in paths}
    out["increments_all"] = {f"{a} vs {b}": rel(a, b, np.ones(n, bool)) for a, b in pairs if a in paths and b in paths}
    # event level
    ev = OUT / "event_oof.npz"
    if ev.exists():
        z = np.load(ev)
        from sklearn.metrics import average_precision_score, roc_auc_score
        evt = {}
        hp = host5.max(1)
        evt["host5"] = dict(auc=float(roc_auc_score(S, hp, sample_weight=w)), ap=float(average_precision_score(S, hp, sample_weight=w)),
                            S_peak_rmse=float(np.sqrt(np.average((hp[S] - peak[S]) ** 2, weights=w[S]))),
                            S_median_peak_ratio=float(np.median(hp[S] / peak[S])))
        for name in VARIANTS:
            pk, pr = z[f"{name}_peak"], z[f"{name}_prob"]
            evt[name] = dict(auc=float(roc_auc_score(S, pr, sample_weight=w)), ap=float(average_precision_score(S, pr, sample_weight=w)),
                             auc_from_peak=float(roc_auc_score(S, pk, sample_weight=w)),
                             S_peak_rmse=float(np.sqrt(np.average((pk[S] - peak[S]) ** 2, weights=w[S]))),
                             S_median_peak_ratio=float(np.median(pk[S] / peak[S])),
                             false_severe=int(((pk >= .10) & nonS).sum()))
        out["event"] = evt
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "info_ceiling_scores.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if not isinstance(vv, dict)}
                      for k, v in rows.items()}, indent=1))
    for k, v in rows.items():
        if "S_vs_host5" in v:
            x = v["S_vs_host5"]; print(f"{k}: S vs host5 {100 * x['point']:+.2f}% [{100 * x['ci95'][0]:+.2f}, {100 * x['ci95'][1]:+.2f}]")
    for k, v in out["increments_S"].items():
        print(f"S increment {k}: {100 * v['point']:+.2f}% [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}]")
    if "event" in out:
        for k, v in out["event"].items():
            print("event", k, {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()})


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "event":
        run_event()
    elif cmd == "traj":
        run_traj(sys.argv[2:] or list(VARIANTS))
    elif cmd == "score":
        score()
