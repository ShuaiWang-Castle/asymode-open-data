"""Information probe, round 2 (notes/INFO_CEILING_PROBE2_SCOPE_20260930_ZH.md): revealed vulnerability V (hazard-
conditioned outage history from NOAA Storm Events and EAGLE-I 2014-2017) and within-county exposure co-location X (HRRR
node products against the product of county means), each against its null. Same learner, folds, weights, metrics
and intervals as probe.py, which is not modified.

  python probe2.py build | event | traj [names] | score
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from probe import (BOOT_SEED, B, CUST, EAGLEI, EVENT_PARAMS, EVENT_ROUNDS, HRRR, NODES, OUT, REGIMES, RES, ROOT,
                   TRAJ_PARAMS, TRAJ_ROUNDS, county_permutation, folds, host_paths, load_static)

STORM = sorted((ROOT / "data" / "raw" / "storm_events_2014_2017").glob("StormEvents_details-*_d201[4-7]_*.csv.gz"))
ZONES = ROOT / "data" / "raw" / "nws" / "zone_county.txt"
CATS = {"wind": ("Thunderstorm Wind", "High Wind", "Strong Wind", "Tornado"),
        "winter": ("Winter Storm", "Ice Storm", "Heavy Snow", "Blizzard", "Lake-Effect Snow"),
        "tropical": ("Tropical Storm", "Hurricane", "Hurricane (Typhoon)", "Tropical Depression"),
        "rain": ("Heavy Rain", "Flash Flood", "Flood")}
MATCH = {"tropical": "tropical", "winter": "winter", "synoptic_wind": "wind", "convective": "wind", "heavy_rain": "rain"}
ALPHA = 5.0
VARIANTS2 = {"M4V": ["B", "W", "G", "C", "H", "V"], "M4Vp": ["B", "W", "G", "C", "H", "Vp"],
             "M1HV": ["B", "W", "H", "V"], "M1HVp": ["B", "W", "H", "Vp"],
             "M1RX": ["B", "W", "R", "X"], "M1RXp": ["B", "W", "R", "Xp"],
             "M5X": ["B", "W", "G", "C", "H", "R", "X"], "M5Xp": ["B", "W", "G", "C", "H", "R", "Xp"]}


# ------------------------------------------------------------------------------------------ vulnerability
def storm_hazard_days() -> pd.DataFrame:
    zc = pd.read_csv(ZONES, sep="|", header=None, dtype=str)
    zc = zc[zc[6].str.fullmatch(r"\d{5}", na=False)]
    abbr = dict(zip(zc[6].str[:2], zc[0]))
    zmap = zc.groupby([zc[0], zc[1].astype(int)])[6].apply(list).to_dict()
    cat_of = {t: c for c, ts in CATS.items() for t in ts}
    out = []
    for f in STORM:
        d = pd.read_csv(f, usecols=["BEGIN_DATE_TIME", "CZ_TIMEZONE", "CZ_TYPE", "STATE_FIPS", "CZ_FIPS", "EVENT_TYPE"], low_memory=False)
        d["cat"] = d["EVENT_TYPE"].map(cat_of)
        d = d.dropna(subset=["cat", "STATE_FIPS", "CZ_FIPS"])
        local = pd.to_datetime(d["BEGIN_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce")
        off = d["CZ_TIMEZONE"].astype(str).str.extract(r"([+-]?\d+)\s*$")[0].astype(float).fillna(0.0)
        d["day"] = (local - pd.to_timedelta(off, unit="h")).dt.floor("D")
        d = d.dropna(subset=["day"])
        d = d.assign(st=d["STATE_FIPS"].astype(int).map("{:02d}".format), cz=d["CZ_FIPS"].astype(int))
        cty = d[d["CZ_TYPE"] == "C"]
        cty = cty.assign(fips=cty["st"] + cty["cz"].map("{:03d}".format))
        zon = d[d["CZ_TYPE"] == "Z"]
        zon = zon.assign(key=list(zip(zon["st"].map(abbr), zon["cz"])))
        zon = zon.assign(fips=zon["key"].map(zmap)).explode("fips").dropna(subset=["fips"])
        out.append(pd.concat([cty[["fips", "day", "cat"]], zon[["fips", "day", "cat"]]]))
        print("storm events", f.name[:45], len(d), "->", len(out[-1]), flush=True)
    return pd.concat(out).drop_duplicates()


def eaglei_daily(need: set):
    """Daily maximum customers out (UTC days) with its timestamp, per county, 2014-2017; and the yearly coverage."""
    parts = []
    for p in EAGLEI:
        d = pd.read_parquet(p, columns=["fips", "ts", "customers_out"])
        d = d[d["fips"].isin(need)].dropna(subset=["customers_out"])
        d["day"] = d["ts"].dt.floor("D")
        i = d.groupby(["fips", "day"], observed=True)["customers_out"].idxmax()
        g = d.loc[i.values, ["fips", "day", "ts", "customers_out"]]
        parts.append(g)
        del d
    g = pd.concat(parts, ignore_index=True)
    g["fips"] = g["fips"].astype(str)
    cust = pd.read_parquet(CUST)["customers"]
    g["frac"] = (g["customers_out"].astype(float) / g["fips"].map(cust).astype(float)).clip(upper=1.0)
    g["year"] = g["day"].dt.year
    return g


def restoration_hours(top: pd.DataFrame) -> pd.Series:
    """For each (fips, peak ts, peak count), hours within 72 h after the peak with customers out >= half the peak."""
    res = {}
    by_year = top.groupby(top["ts"].dt.year)
    for y, ev in by_year:
        p = [q for q in EAGLEI if str(y) in q.name]
        if not p:
            continue
        d = pd.read_parquet(p[0], columns=["fips", "ts", "customers_out"])
        d = d[d["fips"].isin(set(ev["fips"]))].dropna(subset=["customers_out"]).sort_values(["fips", "ts"])
        d["fips"] = d["fips"].astype(str)
        d = d.reset_index(drop=True)
        starts = d.groupby("fips").indices
        ts = d["ts"].to_numpy(); co = d["customers_out"].to_numpy(dtype=float)
        for f, t0, pk in ev[["fips", "ts", "customers_out"]].itertuples(index=False):
            ii = starts.get(f)
            if ii is None:
                continue
            tt = ts[ii]
            lo, hi = np.searchsorted(tt, np.datetime64(t0)), np.searchsorted(tt, np.datetime64(t0) + np.timedelta64(72, "h"))
            res[(f, t0)] = 0.25 * float((co[ii][lo:hi] >= 0.5 * float(pk)).sum())
        del d
    return pd.Series(res)


def vulnerability(fips_units: np.ndarray, regime: np.ndarray):
    f = OUT / "vuln_2014_2017.parquet"
    need = set(np.unique(fips_units).tolist())
    if f.exists():
        V = pd.read_parquet(f)
    else:
        haz = storm_hazard_days()
        haz = haz[haz["fips"].isin(need)]
        daily = eaglei_daily(need)
        covered = set(zip(daily["fips"], daily["year"]))
        frac = daily.set_index(["fips", "day"])["frac"]
        rows = []
        for (fp, day, cat) in haz.itertuples(index=False):
            if (fp, day.year) not in covered:
                continue
            v = max(frac.get((fp, day), 0.0), frac.get((fp, day + pd.Timedelta(days=1)), 0.0))
            rows.append((fp, cat, v))
        resp = pd.DataFrame(rows, columns=["fips", "cat", "resp"])
        glob = resp.groupby("cat")["resp"].agg(ge1=lambda x: (x >= .01).mean(), ge5=lambda x: (x >= .05).mean(), mean="mean")
        agg = resp.groupby(["fips", "cat"])["resp"].agg(n="size", k1=lambda x: (x >= .01).sum(), k5=lambda x: (x >= .05).sum(), s="sum")
        cols = {}
        for c in CATS:
            a = agg.xs(c, level="cat") if c in agg.index.get_level_values("cat") else pd.DataFrame(columns=["n", "k1", "k5", "s"])
            g0 = glob.loc[c] if c in glob.index else pd.Series(dict(ge1=0.0, ge5=0.0, mean=0.0))
            cols[f"v_{c}_n"] = a["n"]
            cols[f"v_{c}_ge1"] = (a["k1"] + ALPHA * g0["ge1"]) / (a["n"] + ALPHA)
            cols[f"v_{c}_ge5"] = (a["k5"] + ALPHA * g0["ge5"]) / (a["n"] + ALPHA)
            cols[f"v_{c}_mean"] = (a["s"] + ALPHA * g0["mean"]) / (a["n"] + ALPHA)
        V = pd.DataFrame(cols)
        # counties with EAGLE-I coverage but no hazard day of a category get the prior (n = 0)
        cov_counties = sorted({fp for fp, _ in covered})
        V = V.reindex(cov_counties)
        for c in CATS:
            g0 = glob.loc[c] if c in glob.index else pd.Series(dict(ge1=0.0, ge5=0.0, mean=0.0))
            V[f"v_{c}_n"] = V[f"v_{c}_n"].fillna(0.0)
            for q, key in (("ge1", "ge1"), ("ge5", "ge5"), ("mean", "mean")):
                V[f"v_{c}_{q}"] = V[f"v_{c}_{q}"].fillna(g0[key])
        # restoration: the three largest outage days (>= 1%) per county
        top = daily[daily["frac"] >= .01].sort_values("frac", ascending=False).groupby("fips").head(3)
        hrs = restoration_hours(top)
        top = top.assign(hrs=[hrs.get((f, t), np.nan) for f, t in zip(top["fips"], top["ts"])])
        V["v_restore_h"] = top.groupby("fips")["hrs"].median().reindex(V.index)
        V.index.name = "fips"
        V = V.reset_index()
        V.to_parquet(f)
        print("vulnerability", V.shape, "hazard days", len(resp), flush=True)
    V = V.set_index("fips")
    base = V.reindex(fips_units)
    match = np.column_stack([np.array([base[f"v_{MATCH[r]}_{q}"].iloc[i] for i, r in enumerate(regime)], dtype=float)
                             for q in ("n", "ge1", "ge5", "mean")])
    M = np.concatenate([base.to_numpy(dtype=float), match], 1).astype(np.float32)
    cols = list(V.columns) + ["v_match_n", "v_match_ge1", "v_match_ge5", "v_match_mean"]
    return M, cols


# ------------------------------------------------------------------------------------------ exposure co-location
def colocation(fips: np.ndarray):
    zh = np.load(HRRR)
    names = list(zh["names"].astype(str))
    ph = zh["phi"].astype(np.float32)
    X = ph[:, :, [names.index(c) for c in ("canopy*g_exc", "canopy*ice_wind", "canopy*snow_wind", "drain*wet_wind")]]
    n = pd.read_parquet(NODES)
    n["w"] = n["w_pop"] / n.groupby("fips")["w_pop"].transform("sum")
    can = (n["w"] * n["canopy"] / 100.0).groupby(n["fips"]).sum()
    wet = (n["w"] * n["wet"]).groupby(n["fips"]).sum()
    c, wt = can.reindex(fips).to_numpy(np.float32), wet.reindex(fips).to_numpy(np.float32)
    Xp = np.stack([c[:, None] * ph[:, :, names.index("g_exc_mean")], c[:, None] * ph[:, :, names.index("ice_wind_mean")],
                   c[:, None] * ph[:, :, names.index("snow_wind_mean")], wt[:, None] * ph[:, :, names.index("wet_wind_mean")]], -1)
    return X, Xp


def build() -> None:
    s = load_static()
    fips, reg = s["fips"].astype(str), s["regime"].astype(str)
    V, vcols = vulnerability(fips, reg)
    donor = county_permutation(fips)
    first = {f: i for i, f in reversed(list(enumerate(fips)))}
    Vp = np.stack([V[first[donor[f]]] for f in fips])
    X, Xp = colocation(fips)
    summ = lambda a: np.concatenate([np.nanmax(a, 1), np.nanmean(a, 1)], 1).astype(np.float32)  # noqa: E731
    np.savez(OUT / "static2.npz", V=V, Vp=Vp, Xe=summ(X), Xpe=summ(Xp), Vcols=np.array(vcols))
    np.save(OUT / "Xh.npy", X.astype(np.float32)); np.save(OUT / "Xph.npy", Xp.astype(np.float32))
    man = dict(units=len(fips), V_features=len(vcols), V_missing_units=int(np.isnan(V[:, 1]).sum()),
               restore_missing_units=int(np.isnan(V[:, vcols.index("v_restore_h")]).sum()),
               colocation_corr=[float(np.corrcoef(X[..., k].ravel(), Xp[..., k].ravel())[0, 1]) for k in range(4)])
    (OUT / "build2_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print("built round 2", man, flush=True)


# ------------------------------------------------------------------------------------------ probes
def blocks_static(s, s2):
    return {"B": s["B"], "G": s["G"], "Gp": s["Gp"], "C": s["C"], "H": s["H"], "Hp": s["Hp"], "V": s2["V"], "Vp": s2["Vp"]}


def run_event() -> None:
    import lightgbm as lgb
    s = load_static(); s2 = dict(np.load(OUT / "static2.npz"))
    st = blocks_static(s, s2)
    ev = {"W": s["We"], "R": s["Re"], "X": s2["Xe"], "Xp": s2["Xpe"]}
    peak, w = s["peak"].astype(float), s["w"].astype(float)
    target = np.log(np.clip(peak, 0, None) + 1e-3); sev = (peak >= .10).astype(float)
    res = {}
    for name, blocks in VARIANTS2.items():
        X = np.concatenate([st[b] if b in st else ev[b] for b in blocks], 1)
        pk, pr = np.full(len(peak), np.nan), np.full(len(peak), np.nan)
        for k, dev, outer in folds():
            ok = peak[dev] >= 0
            reg = lgb.train(EVENT_PARAMS, lgb.Dataset(X[dev][ok], target[dev][ok], weight=w[dev][ok]), EVENT_ROUNDS)
            pk[outer] = np.clip(np.exp(reg.predict(X[outer])) - 1e-3, 0, 1)
            clf = lgb.train(dict(EVENT_PARAMS, objective="binary"), lgb.Dataset(X[dev][ok], sev[dev][ok], weight=w[dev][ok]), EVENT_ROUNDS)
            pr[outer] = clf.predict(X[outer])
        res[f"{name}_peak"], res[f"{name}_prob"] = pk, pr
        print("event2", name, X.shape, flush=True)
    np.savez(OUT / "event2_oof.npz", **res)


def run_traj(names) -> None:
    import lightgbm as lgb
    s = load_static(); s2 = dict(np.load(OUT / "static2.npz"))
    st = blocks_static(s, s2)
    hourly = {"W": np.load(OUT / "Wh.npy"), "R": np.load(OUT / "Rh.npy"), "X": np.load(OUT / "Xh.npy"), "Xp": np.load(OUT / "Xph.npy")}
    y, m, w = s["y"], s["m"], s["w"]

    def rows(blocks, units, even):
        hrs = np.arange(0, 144, 2) if even else np.arange(144)
        ui, ti = np.repeat(units, len(hrs)), np.tile(hrs, len(units))
        return np.concatenate([hourly[b][ui, ti] if b in hourly else st[b][ui] for b in blocks], 1), ui, ti

    for name in names:
        f = OUT / f"traj_{name}.npy"
        if f.exists():
            print("traj", name, "exists", flush=True); continue
        P = np.full(y.shape, np.nan, np.float32); t0 = time.time()
        for k, dev, outer in folds():
            X, ui, ti = rows(VARIANTS2[name], dev, True)
            ok = m[ui, ti] > 0
            bst = lgb.train(TRAJ_PARAMS, lgb.Dataset(X[ok], y[ui, ti][ok], weight=w[ui][ok]), TRAJ_ROUNDS)
            del X
            Xo, uo, to = rows(VARIANTS2[name], outer, False)
            P[uo, to] = np.clip(bst.predict(Xo), 0, 1)
            print(f"traj {name} fold {k} done ({time.time() - t0:.0f} s)", flush=True)
        assert not np.isnan(P).any()
        np.save(f, P)


# ------------------------------------------------------------------------------------------ scoring
def score() -> None:
    from sklearn.metrics import roc_auc_score
    s = load_static()
    y, m, w = s["y"].astype(float), s["m"].astype(float), s["w"].astype(float)
    reg, fam, peak = s["regime"], s["family"], s["peak"].astype(float)
    n = len(y); S = peak >= .10; nonS = ~S; allu = np.ones(n, bool)
    host5, _ = host_paths(n)
    paths = {"host5": host5}
    for name in ["M1", "M1H", "M1R", "M4", "M5"] + list(VARIANTS2):
        f = OUT / f"traj_{name}.npy"
        if f.exists():
            paths[name] = np.load(f).astype(float)
    sse = {k: (m * np.where(m > 0, P - y, 0) ** 2).sum(1) for k, P in paths.items()}
    cnt = m.sum(1)
    keys = sorted({(r, f) for r, f in zip(reg, fam)}); gid = {kk: j for j, kk in enumerate(keys)}
    g = np.array([gid[(r, f)] for r, f in zip(reg, fam)])
    rng = np.random.default_rng(BOOT_SEED); counts = np.zeros((B, len(keys)))
    for r in REGIMES:
        js = np.array([gid[kk] for kk in keys if kk[0] == r]); counts[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), size=B)

    def gs(v, sub, wt=None):
        wt = w if wt is None else wt
        o = np.zeros(len(keys)); np.add.at(o, g[sub], (wt * v)[sub]); return o

    def rmse(k, sub):
        return float(np.sqrt((w * sse[k])[sub].sum() / (w * cnt)[sub].sum()))

    def rel(a, b, sub):
        qa, qb, nn = counts @ gs(sse[a], sub), counts @ gs(sse[b], sub), counts @ gs(cnt, sub)
        d = 1 - np.sqrt(qa / nn) / np.sqrt(qb / nn)
        return dict(point=1 - rmse(a, sub) / rmse(b, sub), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])

    out = {"trajectory": {}, "increments": {}, "event_risk": {}}
    for k in paths:
        pk = paths[k].max(1)
        out["trajectory"][k] = dict(S_rmse=rmse(k, S), all_rmse=rmse(k, allu), nonS_rmse=rmse(k, nonS),
                                    false_peaks=int(((pk >= .10) & nonS).sum()), S_median_peak_ratio=float(np.median(pk[S] / peak[S])))
        if k != "host5":
            out["trajectory"][k]["S_vs_host5"] = rel(k, "host5", S); out["trajectory"][k]["all_vs_host5"] = rel(k, "host5", allu)
    pairs = [("M4V", "M4Vp"), ("M4V", "M4"), ("M1HV", "M1HVp"), ("M1HV", "M1H"), ("M1RX", "M1RXp"), ("M1RX", "M1R"),
             ("M5X", "M5Xp"), ("M5X", "M5")]
    for a, b in pairs:
        if a in paths and b in paths:
            out["increments"][f"{a} vs {b}"] = dict(S=rel(a, b, S), all=rel(a, b, allu), nonS=rel(a, b, nonS))
    ev1, ev2 = np.load(OUT / "event_oof.npz"), (np.load(OUT / "event2_oof.npz") if (OUT / "event2_oof.npz").exists() else None)
    prob = {k: ev1[f"{k}_prob"] for k in ("M1H", "M4")}
    if ev2 is not None:
        prob.update({k: ev2[f"{k}_prob"] for k in VARIANTS2})
    Sf = S.astype(float)

    def brier(k): return (np.clip(prob[k], 1e-4, 1 - 1e-4) - Sf) ** 2
    for a, b in (("M4V", "M4Vp"), ("M4V", "M4"), ("M1HV", "M1HVp"), ("M1HV", "M1H")):
        if a in prob and b in prob:
            res = {}
            for tag, wt in (("weighted", w), ("unweighted", np.ones(n))):
                la, lb = counts @ gs(brier(a), allu, wt), counts @ gs(brier(b), allu, wt)
                d = 1 - la / lb
                res[f"brier_skill_{tag}"] = dict(point=float(1 - (brier(a) * wt).sum() / (brier(b) * wt).sum()),
                                                 ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])
                res[f"auc_{tag}"] = [float(roc_auc_score(S, prob[a], sample_weight=wt)), float(roc_auc_score(S, prob[b], sample_weight=wt))]
            out["event_risk"][f"{a} vs {b}"] = res
    (RES / "info_probe2_scores.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["trajectory"].items():
        extra = f"  S vs host {100 * v['S_vs_host5']['point']:+.2f}%" if "S_vs_host5" in v else ""
        print(f"{k:6s} S {v['S_rmse']:.5f} all {v['all_rmse']:.5f} nonS {v['nonS_rmse']:.5f} FP {v['false_peaks']:3d}{extra}")
    for k, v in out["increments"].items():
        print(k, {kk: f"{100 * vv['point']:+.2f}% [{100 * vv['ci95'][0]:+.2f}, {100 * vv['ci95'][1]:+.2f}]" for kk, vv in v.items()})
    for k, v in out["event_risk"].items():
        print(k, {kk: (f"{100 * vv['point']:+.2f}% [{100 * vv['ci95'][0]:+.2f}, {100 * vv['ci95'][1]:+.2f}]" if isinstance(vv, dict)
                       else [round(x, 3) for x in vv]) for kk, vv in v.items()})


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"build": build, "event": run_event, "score": score}.get(cmd, lambda: run_traj(sys.argv[2:] or list(VARIANTS2)))()
