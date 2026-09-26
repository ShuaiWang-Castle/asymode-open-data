"""DATASET_DESIGN v1 draws (section 4, amendment 2): the sealed confirmation tranche C first, then development D, by
systematic PPS within each regime on the size measure M, the frame sorted by period, region and origin (implicit
stratification); units with pi >= 1 are taken with certainty, iteratively. D is audited for geography, season and
compound coverage and supplemented (amendment 2 S3); D's inclusion probabilities are the inclusion frequencies of
10,000 replays of the D draw, the audit and the supplements with the realised C fixed. C is never supplemented.

Reads the committed frame (systems.parquet, system_counties.parquet), county_axes.parquet, the operator exclusions and
data_provenance/used_windows.json; no outcome. Runs once.

Outputs: data/interim/panel_v1/draws.parquet; data_provenance/draws.json; data_provenance/frame_v1/tranches.parquet."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from frame_common import (CT_SWITCH, EXP, MIN_COVERAGE, MIN_CUSTOMERS, OUT, REGIMES, counties, coverage_max,
                          coverage_year)

SEED_C, SEED_D, SEED_REPLAY = 20260926, 20260927, 20260928   # section 4.4; the replay seed replaces the supplement seed
N_C, N_D = 20, 15                                             # section 4.5
BLOCK = pd.Timedelta(days=16)                                 # sections 4.2 and 4.4
AXES = ["relief_sd", "canopy", "poorly_drained", "log_cust_density", "coast_km"]
MIN_TERCILE, MIN_SHOULDER, MIN_COMPOUND, N_SUPP, REPLAYS = 0.15, 0.25, 5, 3, 10_000


def pps_systematic(size: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Systematic PPS of n units from sizes in frame order. Returns (selected mask, first-order pi)."""
    size = size.astype(float)
    pi = np.zeros(len(size))
    cert = np.zeros(len(size), bool)
    n_left = min(n, len(size))
    while n_left > 0:
        rem = ~cert
        p = n_left * size[rem] / size[rem].sum()
        big = p >= 1.0
        if not big.any():
            pi[rem] = p
            break
        idx = np.where(rem)[0][big]
        cert[idx] = True
        n_left -= len(idx)
    pi[cert] = 1.0
    sel = cert.copy()
    if n_left > 0:
        rem = np.where(~cert)[0]
        cum = np.concatenate([[0.0], np.cumsum(pi[rem])])
        u = rng.uniform(0.0, 1.0)
        hits = np.floor(cum[1:] - u) - np.floor(cum[:-1] - u) > 0
        sel[rem[hits]] = True
    return sel, pi


class Audit:
    """Amendment 2 S3: per D-frame system, the counts of its audit counties (S1-S3 passing G1-G2 by the maximum, not
    operator-excluded) in each national tercile of each axis; arrays per regime frame for fast replays."""

    def __init__(self, frames: dict, sc: pd.DataFrame, excluded):
        c = counties().set_index("fips")
        cov = coverage_max()
        ax = pd.read_parquet(OUT / "county_axes.parquet").set_index("fips")
        ref = c[(c.customers >= MIN_CUSTOMERS).to_numpy() & np.array([cov.get((st, 2022), 0) >= MIN_COVERAGE
                                                                      for st in c.state])].index
        self.cuts = {a: np.nanquantile(ax.loc[ax.index.isin(ref), a], [1 / 3, 2 / 3]) for a in AXES}
        terc = pd.DataFrame({a: np.where(ax[a].notna(), np.digitize(ax[a].to_numpy(), self.cuts[a]), -1) for a in AXES},
                            index=ax.index)
        cust, state = c.customers.to_dict(), c.state.to_dict()
        self.F = {}
        allsys = pd.concat(frames.values())
        info = allsys.set_index("system")[["origin", "window_end"]]
        g = sc[sc.system.isin(set(allsys.system))]
        cnt = {}
        for sid, gg in g.groupby("system"):
            o, w1 = pd.Timestamp(info.at[sid, "origin"]), pd.Timestamp(info.at[sid, "window_end"])
            cy = coverage_year(o.year)
            fs = [f for f in gg.fips if cust.get(f, 0) >= MIN_CUSTOMERS and cov.get((state.get(f, ""), cy), 0) >= MIN_COVERAGE
                  and not excluded(f, o, w1) and not (f.startswith("09") and w1 > CT_SWITCH)]
            t = terc.reindex(fs).to_numpy()
            m = np.zeros((len(AXES), 3))
            for k in range(len(AXES)):
                v = t[:, k]
                v = v[np.isfinite(v) & (v >= 0)].astype(int)
                m[k] = np.bincount(v, minlength=3)[:3]
            cnt[sid] = m
        for r, F in frames.items():
            C_ = np.stack([cnt.get(x, np.zeros((len(AXES), 3))) for x in F.system]) if len(F) else np.zeros((0, len(AXES), 3))
            tot = C_.sum(2, keepdims=True)
            self.F[r] = dict(sys=F.system.to_numpy(), M=F.M.to_numpy(float), cnt=C_,
                             cell=(tot > 0) & (C_ >= 0.5 * tot),                      # (N, K, 3)
                             shoulder=(F.season == "shoulder").to_numpy(), compound=F.compound.to_numpy(bool))

    @staticmethod
    def shares(cnt_sel: np.ndarray) -> np.ndarray:
        m = cnt_sel.sum(0)
        tot = m.sum(1, keepdims=True)
        return np.where(tot > 0, m / np.maximum(tot, 1), 1.0)


def _supp(A, sel, mask, rng):
    idx = np.where(mask & ~sel)[0]
    if len(idx):
        s_, _ = pps_systematic(A["M"][idx], min(N_SUPP, len(idx)), rng)
        sel[idx[s_]] = True
    return len(idx)


def draw_D(audit, rng, log=None):
    """The D draw, the audit and the supplements (amendment 2 S3 (iii)); returns the set of D systems."""
    sel = {}
    for r in REGIMES:
        A = audit.F[r]
        s_, _ = pps_systematic(A["M"], min(N_D, len(A["M"])), rng)
        sel[r] = s_
    for r in REGIMES:
        A = audit.F[r]
        if not len(A["M"]):
            continue
        for k, a in enumerate(AXES):
            sh = audit.shares(A["cnt"][sel[r]])[k]
            if (sh >= MIN_TERCILE).all():
                continue
            t = int(np.argmin(sh))
            n_cell = _supp(A, sel[r], A["cell"][:, k, t], rng)
            if log is not None:
                log.append(dict(regime=r, condition=a, tercile=t, share_before=round(float(sh[t]), 4), cell=n_cell,
                                share_after=round(float(audit.shares(A["cnt"][sel[r]])[k][t]), 4)))
        before = float(A["shoulder"][sel[r]].mean()) if sel[r].any() else 0.0
        if before < MIN_SHOULDER:
            n_cell = _supp(A, sel[r], A["shoulder"], rng)
            if log is not None:
                log.append(dict(regime=r, condition="season", share_before=round(before, 4), cell=n_cell,
                                share_after=round(float(A["shoulder"][sel[r]].mean()), 4)))
    n_comp = sum(int(audit.F[r]["compound"][sel[r]].sum()) for r in REGIMES)
    if n_comp < MIN_COMPOUND:
        r = max(REGIMES, key=lambda q: int(audit.F[q]["compound"].sum()))
        A = audit.F[r]
        n_cell = _supp(A, sel[r], A["compound"], rng)
        if log is not None:
            log.append(dict(regime=r, condition="compound", count_before=n_comp, cell=n_cell,
                            count_after=sum(int(audit.F[q]["compound"][sel[q]].sum()) for q in REGIMES)))
    return set().union(*[set(audit.F[r]["sys"][sel[r]]) for r in REGIMES])


def main() -> None:
    import build_frame as BF
    sy = pd.read_parquet(OUT / "systems.parquet")
    sc = pd.read_parquet(OUT / "system_counties.parquet")
    frame_hash = hashlib.sha256((OUT / "systems.parquet").read_bytes()).hexdigest()
    counties_hash = hashlib.sha256((OUT / "system_counties.parquet").read_bytes()).hexdigest()
    excluded = BF.load_exclusions("data_provenance/operator_exclusions.csv")
    el = sy[sy.eligible].copy().reset_index(drop=True)
    dom = sc.groupby("system").fips.apply(set).to_dict()
    # section 4.2: used systems
    used = json.loads((EXP / "data_provenance" / "used_windows.json").read_text())
    uw = [(pd.Timestamp(w["window_start"]), set(w["fips"])) for w in used]
    el["used"] = [any(abs(w0 - s0) <= BLOCK and dom[s] & F for s0, F in uw)
                  for s, w0 in zip(el.system, el.window_start)]
    key = ["period", "region", "origin"]
    rngC, rngD, rngR = (np.random.default_rng(x) for x in (SEED_C, SEED_D, SEED_REPLAY))
    el["tranche"], el["pi"], el["pi_C"], el["pi_D"], el["blocked_by_C"] = "U", 0.0, 0.0, 0.0, False
    alloc = {}
    for r in REGIMES:                                          # C first (section 4.4 (3))
        F = el[(el.regime == r) & ~el.used].sort_values(key, kind="stable")
        n = min(N_C, int(np.floor(0.55 * len(F)))) if r == "tropical" else N_C
        sel, pi = pps_systematic(F.M.to_numpy(), n, rngC)
        el.loc[F.index[sel], "tranche"] = "C"
        el.loc[F.index, "pi_C"] = pi
        el.loc[F.index[sel], "pi"] = pi[sel]
        g = el[el.regime == r]
        alloc[r] = {"C": int(sel.sum()), "C_frame": int(len(F)), "frame": int(len(g)),
                    "used_share_systems": round(float(g.used.mean()), 4),
                    "used_share_M": round(float(g.M[g.used].sum() / g.M.sum()), 4)}
    C = el[el.tranche == "C"]
    fam_C = set(C.family)
    by_day = defaultdict(list)
    for s, w0 in zip(C.system, C.window_start):
        by_day[w0.normalize()].append((w0, dom[s]))

    def blocked(s, fam, w0):                                   # section 4.4 (4)
        if fam in fam_C:
            return True
        for d in range(-16, 17):
            for c0, F in by_day.get((w0 + pd.Timedelta(days=d)).normalize(), ()):
                if abs(c0 - w0) <= BLOCK and dom[s] & F:
                    return True
        return False

    el["blocked_by_C"] = [t != "C" and blocked(s, f, w0) for s, f, w0, t in
                          zip(el.system, el.family, el.window_start, el.tranche)]
    frames = {r: el[(el.regime == r) & (el.tranche == "U") & ~el.blocked_by_C].sort_values(key, kind="stable")
              for r in REGIMES}
    audit = Audit(frames, sc, excluded)
    log = []
    Dset = draw_D(audit, rngD, log)                            # the registered D draw
    el.loc[el.system.isin(Dset), "tranche"] = "D"
    hits = defaultdict(int)                                    # amendment 2 S3 (iv): pi by replays
    for i in range(REPLAYS):
        for s_ in draw_D(audit, rngR):
            hits[s_] += 1
    el["pi_D"] = el.system.map(lambda s: hits.get(s, 0) / REPLAYS)
    el.loc[el.tranche == "D", "pi"] = el.loc[el.tranche == "D", "pi_D"]
    for r in REGIMES:
        g = el[(el.regime == r) & (el.tranche == "D")]
        alloc[r].update({"D": int(len(g)), "D_frame": int(len(frames[r])), "D_used": int(g.used.sum()),
                         "D_min_pi": round(float(g.pi.min()), 5) if len(g) else None})
    assert (el.loc[el.tranche == "D", "pi"] > 0).all()
    out = el[["system", "regime", "family", "origin", "window_start", "tranche", "pi", "pi_C", "pi_D", "used",
              "blocked_by_C", "M", "season", "compound", "region", "period"]]
    out.to_parquet(OUT / "draws.parquet", index=False)
    fv = EXP / "data_provenance" / "frame_v1"
    fv.mkdir(parents=True, exist_ok=True)
    out[out.tranche != "U"].to_parquet(fv / "tranches.parquet", index=False)
    rec = dict(frame_sha256=frame_hash, system_counties_sha256=counties_hash,
               seeds={"C": SEED_C, "D": SEED_D, "replay": SEED_REPLAY}, replays=REPLAYS, allocation=alloc,
               audit_cuts={a: [float(x) for x in audit.cuts[a]] for a in AXES}, audit_log=log,
               n_eligible=int(len(el)), n_used=int(el.used.sum()), n_blocked=int(el.blocked_by_C.sum()),
               tranches_sha256=hashlib.sha256((fv / "tranches.parquet").read_bytes()).hexdigest())
    (EXP / "data_provenance" / "draws.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
