"""The DATASET_DESIGN v1 frame (section 3): parent weather systems from NWS VTEC events and HURDAT2 tracks, typed,
timed, gated and stratified, before any draw.

No outage value is read. EAGLE-I enters only through the national set of collection-run timestamps (gate S-b, a
property of the calendar) and the 2024 modelled customers; the coverage history enters G2.

Outputs (data/interim/panel_v1/):
  events.parquet           one row per VTEC event of the damage phenomena (W) and of the S2 products (A, Y)
  exposure.parquet         (key, fips, a)
  systems.parquet          one row per system, with gates S-a..S-d and strata
  system_counties.parquet  (system, fips, stratum S1/S2/S3, label, labels)
  frame_counts.csv         N_h per regime x period x region x season class, systems passing every gate
usage: python build_frame.py [--exclusions data_provenance/operator_exclusions.csv]
"""
from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import exposure as X
from frame_common import (A_WARNED, CT_SWITCH, LINK_CAP_H, TROPICAL_PHENOM, coverage_max, COLLECTION_SHARE, COMPOUND_SHARE, CORE_MONTHS, EXP, FRAME_END, FRAME_START,
                          HORIZON_H, HRRR_SHARE, INTERIM, LEAD_H, LINK_GAP_H, MIN_COVERAGE, MIN_CUSTOMERS, OUT,
                          P2_START, POLYGON_PHENOM, PREFIX_H, PRIORITY, RAW, REGIME_OF, RING_KM, S2_PRODUCTS,
                          SEGMENT_H, TC_RADIUS_KM, W_PHENOM, adjacency, counties, coverage_min, coverage_year,
                          haversine_km)

YEARS = range(2018, 2026)
H = pd.Timedelta(hours=1)


# ---------------------------------------------------------------- VTEC events and exposure

def load_vtec() -> pd.DataFrame:
    cols = ["WFO", "ISSUED", "EXPIRED", "INIT_ISS", "PHENOM", "GTYPE", "SIG", "ETN", "STATUS", "NWS_UGC"]
    out = []
    for y in YEARS:
        d = pd.read_csv(RAW / "nws" / "vtec" / f"wwa_{y}01010000_{y}12312359.csv.gz", dtype=str, usecols=cols)
        ps = list(zip(d.PHENOM, d.SIG))
        keep = np.array([(p in W_PHENOM and s == "W") or (p, s) in S2_PRODUCTS for p, s in ps])
        out.append(d[keep])
    d = pd.concat(out, ignore_index=True)
    for c in ("ISSUED", "EXPIRED", "INIT_ISS"):
        d[c] = pd.to_datetime(d[c], format="%Y-%m-%d %H:%M")
    d["key"] = (d.WFO + "|" + d.PHENOM + "|" + d.SIG + "|" + d.ETN.astype(int).astype(str) + "|"
                + d.INIT_ISS.dt.year.astype(str))
    return d.drop_duplicates(["key", "GTYPE", "NWS_UGC", "ISSUED", "EXPIRED"])


def events_table(v: pd.DataFrame) -> pd.DataFrame:
    g = v.groupby("key")
    e = pd.DataFrame({"phenom": g.PHENOM.first(), "sig": g.SIG.first(), "init": g.INIT_ISS.min(),
                      "issued": g.ISSUED.min(), "end": g.EXPIRED.max()})
    e["begin"] = e[["issued", "init"]].max(axis=1)        # section 3.6: max(ISSUED, INIT_ISS)
    e["end"] = e[["end", "init"]].max(axis=1)
    return e.reset_index()


def exposure_table(v: pd.DataFrame, ev: pd.DataFrame, cty: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    st_fips = dict(zip(cty.state, cty.state_fips))
    fips_ok = set(cty.fips)
    poly = X.polygon_shares(YEARS)
    has_poly = set(poly.key)
    use_poly = ev.phenom.isin(POLYGON_PHENOM) & ev.key.isin(has_poly)
    parts = [poly[poly.key.isin(set(ev.key[use_poly]))]]
    rows = v[(v.GTYPE == "C") & ~v.key.isin(set(ev.key[use_poly])) & v.NWS_UGC.notna()][["key", "NWS_UGC", "INIT_ISS"]]
    rows = rows.drop_duplicates(["key", "NWS_UGC"])
    cnty = rows[rows.NWS_UGC.str[2] == "C"]
    f = cnty.NWS_UGC.str[:2].map(st_fips) + cnty.NWS_UGC.str[3:6]
    parts.append(pd.DataFrame({"key": cnty.key.to_numpy(), "fips": f.to_numpy(), "a": 1.0}))
    zon = rows[rows.NWS_UGC.str[2] == "Z"].copy()
    # zone geometry valid at issuance: the IEM UGC snapshot of the month of INIT_ISS
    idx = pd.read_parquet(RAW / "nws" / "ugc" / "ugc_snapshot_index.parquet")
    idx = idx[idx.ugc.str[2] == "Z"]
    zon["snapshot"] = zon.INIT_ISS.dt.to_period("M").dt.to_timestamp()
    zon = zon.merge(idx, left_on=["snapshot", "NWS_UGC"], right_on=["snapshot", "ugc"], how="left")
    zs = X.zone_shares()
    hit = zon[zon.vhash.notna()].merge(zs, on="vhash")[["key", "fips", "a"]]
    miss = zon[zon.vhash.isna()]
    zc = pd.read_csv(RAW / "nws" / "zone_county.txt", sep="|", header=None, dtype=str,
                     names=["state", "zone", "cwa", "name", "ugc", "county", "fips", "tz", "fe", "lat", "lon"])
    zc["ugc"] = zc.state + "Z" + zc.zone.str.zfill(3)
    fb = miss.merge(zc[["ugc", "fips"]], left_on="NWS_UGC", right_on="ugc")[["key", "fips"]].assign(a=1.0)
    parts += [hit, fb]
    x = pd.concat(parts, ignore_index=True)
    x = x[x.fips.isin(fips_ok)]
    x = x.groupby(["key", "fips"], as_index=False)["a"].sum()
    x["a"] = x["a"].clip(upper=1.0)
    stats = dict(polygon_events=int(use_poly.sum()), zone_rows=int(len(zon)), zone_rows_matched=int(zon.vhash.notna().sum()),
                 zone_rows_fallback=int(len(miss)), zone_rows_fallback_mapped=int(fb.key.nunique()))
    return x, stats


# ---------------------------------------------------------------- tropical cyclones

def read_hurdat() -> pd.DataFrame:
    rows = []
    for f in sorted((RAW / "nws" / "hurdat2").glob("hurdat2-*.txt")):
        sid = name = None
        for line in f.read_text().splitlines():
            p = [s.strip() for s in line.split(",")]
            if len(p) >= 3 and p[0][:2].isalpha() and len(p[0]) == 8:
                sid, name = p[0], p[1]
                continue
            if p[0][:4] < "2018":            # only the frame's years are parsed (an old 1969 line lacks a comma)
                continue
            t = pd.Timestamp(f"{p[0]} {p[1][:2]}:{p[1][2:]}")
            lat = float(p[4][:-1]) * (1 if p[4][-1] == "N" else -1)
            lon = float(p[5][:-1]) * (-1 if p[5][-1] == "W" else 1)
            rows.append((sid, name, t, p[3], lat, lon))
    d = pd.DataFrame(rows, columns=["sid", "name", "t", "status", "lat", "lon"])
    return d[(d.t >= "2018-06-01") & (d.t < "2026-01-01")]


def tc_systems(hd: pd.DataFrame, cty: pd.DataFrame) -> list[dict]:
    out = []
    clat, clon = cty.lat.to_numpy(), cty.lon.to_numpy()
    for sid, g in hd.groupby("sid"):
        g = g.drop_duplicates("t").sort_values("t")
        hrs = pd.date_range(g.t.iloc[0], g.t.iloc[-1], freq="h")
        tt = g.t.astype("int64").to_numpy()
        lat = np.interp(hrs.astype("int64"), tt, g.lat.to_numpy())
        lon = np.interp(hrs.astype("int64"), tt, g.lon.to_numpy())
        dist = haversine_km(lat[:, None], lon[:, None], clat[None, :], clon[None, :])     # (hours, counties)
        near = dist.min(1) <= TC_RADIUS_KM
        if not near.any():
            continue
        i0, i1 = np.where(near)[0][[0, -1]]
        dom = cty.fips.to_numpy()[(dist[i0:i1 + 1] <= TC_RADIUS_KM).any(0)]
        out.append(dict(sid=sid, name=g.name.iloc[0], span_start=hrs[i0], span_end=hrs[i1], domain=set(dom)))
    return out


# ---------------------------------------------------------------- linking into families and systems

class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def link(ev: pd.DataFrame, s1: dict[str, set], nb: dict[str, set]) -> np.ndarray:
    """Section 3.4 (2): two events are linked when their time intervals are within 12 h and their S1 county sets
    overlap or are adjacent. Candidate pairs through 3 h (county, time) bins, then the exact gap is checked."""
    keys = ev.key.to_list()
    init = ev.init.to_numpy("datetime64[h]").astype(np.int64)
    end = np.minimum(ev.end.to_numpy("datetime64[h]").astype(np.int64), init + LINK_CAP_H)   # amendment 2 M1
    B = 3
    cells = defaultdict(list)
    for i, k in enumerate(keys):
        for c in s1[k]:
            for b in range(init[i] // B, end[i] // B + 1):
                cells[(c, b)].append(i)
    dsu = DSU(len(keys))
    for i, k in enumerate(keys):
        dil = set(s1[k])
        for c in s1[k]:
            dil |= nb.get(c, set())
        lo, hi = (init[i] - LINK_GAP_H) // B, (end[i] + LINK_GAP_H) // B
        seen = set()
        for c in dil:
            for b in range(lo, hi + 1):
                for j in cells.get((c, b), ()):
                    if j != i and j not in seen:
                        seen.add(j)
                        gap = max(init[i], init[j]) - min(end[i], end[j])
                        if gap <= LINK_GAP_H:
                            dsu.union(i, j)
    return np.array([dsu.find(i) for i in range(len(keys))])


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclusions", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cty = counties()
    fips_all = set(cty.fips)
    nb = adjacency(fips_all)
    cust = dict(zip(cty.fips, cty.customers))
    reg_of = dict(zip(cty.fips, cty.region))
    st_of = dict(zip(cty.fips, cty.state))
    cov = coverage_max()             # amendment 2 S4

    v = load_vtec()
    ev = events_table(v)
    expo, xstats = exposure_table(v, ev, cty)
    print("events", len(ev), "exposure rows", len(expo), xstats, flush=True)
    ev.to_parquet(OUT / "events.parquet", index=False)
    expo.to_parquet(OUT / "exposure.parquet", index=False)
    ev = ev.set_index("key", drop=False)
    a_of = defaultdict(dict)
    for k, f, x in zip(expo.key, expo.fips, expo.a):
        a_of[k][f] = x
    isW = (ev.sig == "W") & ev.phenom.isin(W_PHENOM)
    s1 = {k: {f for f, x in a_of[k].items() if x >= A_WARNED} for k in ev.key[isW]}
    wkeys = [k for k in ev.key[isW] if s1[k]]

    # tropical systems first (section 3.4 (1), amendment 1 (1))
    hd = read_hurdat()
    names = dict(zip(hd.sid, hd.name))
    tcs = tc_systems(hd, cty)
    by_sid = {tc["sid"]: ti for ti, tc in enumerate(tcs)}
    owner = {}
    # a tropical W event carries its storm in its event number: ETN = 1000 + n (Atlantic storm n), 2000 + n (eastern
    # Pacific); it joins that storm's system. A storm with such events but no HURDAT2 span (a potential tropical cyclone
    # never named, or a track that stays beyond 300 km) forms a TC system from its events alone.
    extra = {}
    for k in wkeys:
        if ev.at[k, "phenom"] in TROPICAL_PHENOM:
            etn, yr = int(k.split("|")[3]), k.split("|")[4]
            if 1000 <= etn < 3000:
                sid = f"{'AL' if etn < 2000 else 'EP'}{etn % 1000:02d}{yr}"
                if sid not in by_sid:
                    extra.setdefault(sid, []).append(k)
                    by_sid[sid] = None
                owner[k] = (sid, 1.0)
    w_init = ev.loc[wkeys, "init"].to_numpy()
    for ti, tc in enumerate(tcs):
        # other phenomena: INIT_ISS in [span start - 24 h, span end + 12 h], at least half of the counties in the domain
        lo = np.datetime64(tc["span_start"] - pd.Timedelta(hours=24))
        hi = np.datetime64(tc["span_end"] + pd.Timedelta(hours=12))
        for i in np.where((w_init >= lo) & (w_init <= hi))[0]:
            k = wkeys[i]
            if k in owner and owner[k][1] == 1.0:
                continue
            cs = set(a_of[k])
            share = len(cs & tc["domain"]) / len(cs)
            if share >= 0.5 and share > owner.get(k, (None, 0.0))[1]:
                owner[k] = (tc["sid"], share)
    systems = []
    for tc in tcs:
        ks = [k for k, (t, _) in owner.items() if t == tc["sid"]]
        if ks:
            systems.append(dict(kind="tc", tc_sid=tc["sid"], tc_name=tc["name"], family=f"TC_{tc['sid']}", keys=ks,
                                onset=tc["span_start"], tc_domain=tc["domain"]))
    for sid, ks in extra.items():
        ks = [k for k, (t, _) in owner.items() if t == sid]
        systems.append(dict(kind="tc", tc_sid=sid, tc_name=names.get(sid, "(no best track)") + " (no span)",
                            family=f"TC_{sid}", keys=ks,
                            onset=ev.loc[ks, "begin"].min(), tc_domain=set().union(*[set(a_of[k]) for k in ks])))
    allw = ev.loc[wkeys].reset_index(drop=True)
    allw["family"] = link(allw, s1, nb)
    tc_fam = {}                      # amendment 2 M2: a graph family linked to a TC system's event joins its family
    for fam, g in allw.groupby("family"):
        tcs_in = [owner[k][0] for k in g.key if k in owner]
        if tcs_in:
            tc_fam[fam] = "TC_" + max(set(tcs_in), key=tcs_in.count)
    rest = allw[~allw.key.isin(set(owner))]
    for fam, g in rest.groupby("family"):
        g = g.sort_values("init")
        seg = ((g.init - g.init.iloc[0]) // pd.Timedelta(hours=SEGMENT_H)).astype(int)
        for sgi, gg in g.groupby(seg):
            systems.append(dict(kind="nontc", tc_sid="", tc_name="", family=tc_fam.get(fam, f"F{fam}"), segment=int(sgi),
                                keys=gg.key.to_list(), onset=gg.begin.min()))
    print("systems", len(systems), "tc", sum(s["kind"] == "tc" for s in systems), flush=True)

    # per-county product index for S2 / S3 (section 5.1)
    byc = defaultdict(list)          # fips -> sorted list of (init, end, key, kind)
    e_init, e_end = ev["init"].to_dict(), ev["end"].to_dict()
    e_sig, e_ph = ev["sig"].to_dict(), ev["phenom"].to_dict()
    for k, x in a_of.items():
        kind = "W" if (e_sig[k] == "W" and e_ph[k] in W_PHENOM) else "AY"
        if kind == "W" and not s1.get(k):
            kind = "W0"                  # a W event with no warned county acts like an advisory (S2 only)
        for f in x:
            byc[f].append((e_init[k], e_end[k], k, kind))
    for f in byc:
        byc[f].sort()
    inits = {f: [r[0] for r in L] for f, L in byc.items()}
    max_dur = min((ev["end"] - ev["init"]).max(), pd.Timedelta(days=14))
    clat = dict(zip(cty.fips, cty.lat)); clon = dict(zip(cty.fips, cty.lon))
    F_arr = cty.fips.to_numpy(); LAT = cty.lat.to_numpy(); LON = cty.lon.to_numpy()

    def products_at(f, t0, t1):
        L, I = byc.get(f, []), inits.get(f, [])
        i, j = bisect.bisect_left(I, t0 - max_dur), bisect.bisect_left(I, t1)
        return [r for r in L[i:j] if r[1] > t0]

    hrrr = pd.read_parquet(OUT / "hrrr_hours.parquet").set_index("valid")["present"]
    coll = collection_hours()
    excl = load_exclusions(a.exclusions) if a.exclusions else None

    srows, crows = [], []
    for si, s in enumerate(systems):
        ks = s["keys"]
        lab = defaultdict(set)
        touched = set()
        for k in ks:
            for f, x in a_of[k].items():
                if x >= A_WARNED:
                    lab[f].add("tropical" if s["kind"] == "tc" else REGIME_OF[e_ph[k]])
                else:
                    touched.add(f)
        S1 = set(lab)
        label = {f: ("tropical" if s["kind"] == "tc" else min(L, key=PRIORITY.get)) for f, L in lab.items()}
        by_reg = defaultdict(float)
        for f in S1:
            by_reg[label[f]] += cust.get(f, 0.0)
        tot = sum(by_reg.values())
        order = sorted(by_reg, key=lambda r: (-by_reg[r], PRIORITY[r]))
        regime = order[0]
        # amendment 2 S2: share of S1 customers in counties whose label vector holds another regime
        other = sum(cust.get(f, 0.0) for f in S1 if lab[f] - {regime})
        compound = s["kind"] != "tc" and tot > 0 and other / tot >= COMPOUND_SHARE
        onset = pd.Timestamp(s["onset"])
        t_s = onset.floor("h") - pd.Timedelta(hours=LEAD_H)
        w0, w1 = t_s - pd.Timedelta(hours=PREFIX_H), t_s + pd.Timedelta(hours=HORIZON_H)
        # ring R(s): counties within 200 km of an S1 centroid (plus the TC domain)
        s1l = np.array([clat[f] for f in S1]); s1o = np.array([clon[f] for f in S1])
        dmin = np.full(len(F_arr), np.inf)
        for la, lo in zip(s1l, s1o):
            dmin = np.minimum(dmin, haversine_km(la, lo, LAT, LON))
        R = set(F_arr[dmin <= RING_KM]) | (s.get("tc_domain") or set())
        S2, S3 = set(), set()
        for f in R - S1:
            prods = products_at(f, t_s, w1)
            own_touch = f in touched
            ay = any(p[3] in ("AY", "W0") for p in prods)
            w_any = any(p[3] == "W" for p in prods)
            if own_touch or ay:
                S2.add(f)
            elif not w_any:
                S3.add(f)
        # gates
        hrs = pd.date_range(w0, w1 - H, freq="h")
        s_a = (w0 >= FRAME_START) and (w1 <= FRAME_END)
        s_b = float(coll.reindex(hrs, fill_value=False).mean()) >= COLLECTION_SHARE
        s_c = float(hrrr.reindex(hrs, fill_value=False).mean()) >= HRRR_SHARE
        pre = pd.date_range(w0, t_s - H, freq="h")
        cp = coll.reindex(pre, fill_value=False)
        s_e = bool(cp.iloc[-1]) and float(cp.mean()) >= COLLECTION_SHARE      # amendment 2 S6
        cy = coverage_year(t_s.year)

        def g12(f):
            return cust.get(f, 0.0) >= MIN_CUSTOMERS and cov.get((st_of[f], cy), 0.0) >= MIN_COVERAGE

        def excluded(f):
            if f.startswith("09") and w1 > CT_SWITCH:      # amendment 1 (2): Connecticut after the code switch
                return True
            return excl is not None and excl(f, t_s, w1)

        s1_ok = [f for f in S1 if g12(f) and not excluded(f)]
        s_d = len(s1_ok) > 0
        mcust = sum(cust.get(f, 0.0) for f in s1_ok)
        reg_c = defaultdict(float)
        for f in S1:
            reg_c[reg_of[f]] += cust.get(f, 0.0)
        region = max(reg_c, key=reg_c.get) if reg_c else ""
        season = "core" if t_s.month in CORE_MONTHS[regime] else "shoulder"
        sid = f"S{si:05d}"
        srows.append(dict(system=sid, kind=s["kind"], family=s["family"], segment=s.get("segment", 0), tc_sid=s["tc_sid"],
                          tc_name=s["tc_name"], regime=regime, compound=bool(compound),
                          regime_shares=json.dumps({r: round(by_reg[r] / tot, 4) for r in order}) if tot else "{}",
                          n_events=len(ks), onset=onset, origin=t_s, window_start=w0, window_end=w1,
                          period="P1" if t_s < P2_START else "P2", region=region, season=season,
                          n_S1=len(S1), n_S2=len(S2), n_S3=len(S3), n_S1_gated=len(s1_ok), S1_customers=tot,
                          M_base=mcust, M=mcust * (2 if season == "shoulder" else 1) * (2 if compound else 1),
                          S_a=s_a, S_b=s_b, S_c=s_c, S_d=s_d, S_e=s_e, exclusions_applied=excl is not None,
                          keys=json.dumps(ks)))
        for f in S1:
            crows.append((sid, f, "S1", label[f], ",".join(sorted(lab[f]))))
        crows += [(sid, f, "S2", "", "") for f in S2] + [(sid, f, "S3", "", "") for f in S3]
        if si % 2000 == 0:
            print("system", si, "of", len(systems), flush=True)
    sy = pd.DataFrame(srows)
    sy["eligible"] = sy.S_a & sy.S_b & sy.S_c & sy.S_d & sy.S_e
    sy.to_parquet(OUT / "systems.parquet", index=False)
    pd.DataFrame(crows, columns=["system", "fips", "stratum", "label", "labels"]).to_parquet(
        OUT / "system_counties.parquet", index=False)
    el = sy[sy.eligible]
    # amendment 2 S4: the state-years whose minimum coverage is below 0.8 while the maximum passes; their counties face the
    # pre-window in-data check at the county-sample step
    cmin = coverage_min()
    sv = sorted((st, y) for (st, y), v in cov.items() if v >= MIN_COVERAGE and cmin.get((st, y), 0.0) < MIN_COVERAGE)
    pd.DataFrame(sv, columns=["state", "coverage_year"]).to_csv(OUT / "frame_state_years_min_below.csv", index=False)
    nh = el.groupby(["regime", "period", "region", "season"]).size().rename("N_h").reset_index()
    nh.to_csv(OUT / "frame_counts.csv", index=False)
    print("gates:", {g: int(sy[g].sum()) for g in ("S_a", "S_b", "S_c", "S_d", "S_e", "eligible")}, "of", len(sy))
    print(el.groupby("regime").agg(n=("system", "size"), compound=("compound", "sum"),
                                   shoulder=("season", lambda x: int((x == "shoulder").sum())),
                                   median_S1=("n_S1", "median"), median_S2=("n_S2", "median"),
                                   median_S3=("n_S3", "median")).to_string())


def collection_hours() -> pd.Series:
    """Hours with at least one national collection run (a timestamp with >= 5 reporting counties,
    panel.collection_timestamps), from the EAGLE-I timestamps alone (section 3.7, S-b)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    parts = []
    for y in range(2018, 2026):
        t = pq.read_table(INTERIM / f"eaglei_outages_{y}.parquet", columns=["ts"]).column("ts")
        vc = pc.value_counts(t)
        s = pd.Series(vc.field("counts").to_numpy(), index=pd.DatetimeIndex(vc.field("values").to_numpy()))
        parts.append(s)
    n = pd.concat(parts)
    ran = n[n >= 5].index.floor("h").unique()
    full = pd.date_range("2018-01-01", "2026-01-01", freq="h", inclusive="left")
    return pd.Series(full.isin(ran), index=full)


def load_exclusions(path: str):
    """County-window exclusions of section 7: [start - 12 h, end + 48 h] overlapping the forecast window."""
    d = pd.read_csv(EXP / path, dtype=str)
    d = d[d.county_fips.notna() & (d.county_fips != "")]
    d["s"] = pd.to_datetime(d.start_utc, utc=True).dt.tz_localize(None) - pd.Timedelta(hours=12)
    d["e"] = pd.to_datetime(d.end_utc, utc=True).dt.tz_localize(None) + pd.Timedelta(hours=48)
    by = defaultdict(list)
    for f, s, e in zip(d.county_fips.str.zfill(5), d.s, d.e):
        by[f].append((s, e))

    def excluded(f, t0, t1):
        return any(s < t1 and e > t0 for s, e in by.get(f, ()))
    return excluded


if __name__ == "__main__":
    main()
