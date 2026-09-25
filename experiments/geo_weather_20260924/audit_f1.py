"""F1 information audit, parts (c), (d), (e) of contrib/REVIEW_formal.md section 4. No model is trained.

Inputs are the exposure-integrated features of build_eih.py: data/interim/geo_weather/eih_<v>.npz for
v in area, pop, quad, mean, pooled, other, quadn, each with phi [U, 144, F] (hours 72..215), names [F]
("psi*mod@tau"), fips, event; unit order = data/interim/open_gcrk/features_e3r2.npz. The base is the held-out
prediction of W+Cin (main design, seed 0).

Contrasts (higher information minus lower information):
  C1 quad - pop       sub-grid (bands with downscaling against population-weighted cells)
  C2 pop - area       exposure weighting
  C3 pooled - mean    L1g: the county's band distribution under county weather against one node
  C4 quad - pooled    co-location of bands with their own cell's weather
  C5 pooled - mean, controlling for other - mean: own band distribution beyond a same-stratum donor's

  (c) alignment  Unit level: the base's window-mean residual and the window-mean contrast, both residualised on
                 event x state x relief-tercile blocks, the base's window-mean prediction and log customers (the
                 contrast also on the lower variant's own intensity, and for C5 on the donor contrast). Effect size =
                 part correlation; test = cluster-robust score t (clusters event x state) with a Rademacher multiplier
                 bootstrap for the max over features (Westfall-Young), Bonferroni over the five contrasts. Its
                 family-wise false-positive rate is measured on synthetic residual fields (homoscedastic, and
                 heteroscedastic with the residual's own magnitudes) with the residual's spatial dependence; the
                 registered rule picks the null. Secondary: the same at hour level (unit- and lead-demeaned cells),
                 with an exact sign-flip over events as the conservative check.
  (d) power      For each within-modulator regressor (quadn[psi*mod@tau] - quad[psi*one@tau] = Cov_rho(psi, mod/mean)),
                 and for each ladder contrast C1-C4, the regressor and its nuisance (the lower variant; for modulators
                 quad[psi*one@tau]) are propagated through the base's own linearised rollout; cluster-robust standard
                 error under the base's residual noise; minimum detectable effect, its pooled-RMSE worth, power at 2%.
  (e) ceiling    open_gcrk RESULTS section 16 (D3) regressor, same settings, folds and targets, with each variant's
                 summaries added; within-block permutation null for the quadrature arm.

The base panel, splits and base runs follow open_gcrk common.py (OPEN_GCRK_ROUND, default e3r2); another panel
needs its features_<round>.npz, splits_<round>.json and base runs in that layout, plus --eih-prefix.

Run (from the repository root, one process):
    .venv/bin/python experiments/geo_weather_20260924/audit_f1.py                # registered settings
    .venv/bin/python experiments/geo_weather_20260924/audit_f1.py --quick        # smoke settings, not for reading
Writes experiments/geo_weather_20260924/results/F1/: F1_<parts>.json and .md (F1_cde.* for a full run),
alignment.csv, alignment_hour.csv, mde.csv, ceiling.csv, ceiling_contrasts.csv.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("OPEN_GCRK_ROUND", "e3r2")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "open_gcrk_20260919"))
import common as C  # noqa: E402

EIH = ROOT / "data" / "interim" / "geo_weather"
AREA_WEIGHTS = ROOT / "data" / "interim" / "open_gcrk" / "era5_county_weights_conus.parquet"
OUT = HERE / "results" / "F1"
VARIANTS = ("area", "pop", "quad", "mean", "pooled", "other", "quadn")
CONTRASTS = [("C1 quad-pop (sub-grid)", "quad", "pop", None),
             ("C2 pop-area (exposure weighting)", "pop", "area", None),
             ("C3 pooled-mean (L1g marginal)", "pooled", "mean", None),
             ("C4 quad-pooled (co-location)", "quad", "pooled", None),
             ("C5 pooled-mean | other-mean (own beyond donor)", "pooled", "mean", "other")]
T = 144

REG = dict(
    version="interface v2 (exposure-integrated eih_<v>.npz), fixed 2026-09-25 before any F1 number existed",
    base=dict(design="main", arm="W+Cin", seed=0),
    sign=+1, sign_meaning="higher-information variant above the lower one goes with under-prediction (y - P > 0)",
    revision="2026-09-25 00:45, before any F1 number existed: the within-block permutation null of the first draft is "
             "invalid when |residual| and |contrast| share a scale (a pure random-sign contrast came out 'significant' on "
             "synthetic inputs, at unit and hour level); replaced by cluster-robust score statistics with a multiplier "
             "bootstrap, and the calibration now includes heteroscedastic synthetic fields",
    unit_statistic="effect size: part correlation of residual (on blocks, base window-mean prediction, log customers) and "
                   "contrast (same + lower variant's window-mean intensity [+ donor contrast for C5]); test: cluster-robust "
                   "score t, clusters event x state",
    blocks="event x state x relief tercile (relief_p95_p5; cut points over counties)",
    family="Westfall-Young max-T over features within a contrast (Rademacher multiplier bootstrap of cluster scores), "
           "Bonferroni over the five contrasts",
    n_mult=4000, n_syn=1000, n_mult_syn=999,
    syn_field="per event: Gaussian field over county centroids, covariance c exp(-d / l) + (1 - c) I, (c, l) fitted to the "
              "correlogram of the event-demeaned residual; four variants: homoscedastic l, 2 l; heteroscedastic "
              "|residual| x field, l, 2 l",
    fpr_rule="if the family-wise false-positive rate at 0.05 of the multiplier max-T test, measured on the synthetic "
             "fields, exceeds 0.10 in any variant for a contrast, that contrast is read with the synthetic-field max-T "
             "p-values of the variant with the largest rate; otherwise with the multiplier max-T p-values",
    hour_level=dict(demeaning="unit then lead, observed cells", test="cluster-robust score t (event x state) with "
                    "multiplier max-T; conservative check: exact sign-flip over events (12 clusters), max-T"),
    min_eff_clusters=20,
    revision_2="2026-09-25 01:00, after the first run of (c, d) on the wind panel: (d) standard errors are the larger of "
               "the event x state and county cluster-robust errors (they disagreed by up to 74x for sparse regressors), "
               "and every statistic reports its effective number of clusters, (sum w)^2 / sum w^2 over clusters with w "
               "the regressor's squared mass; a (c) pass and the (d) kill rule use only regressors with at least 20 "
               "effective clusters in both clusterings. No verdict of that run changes under either version.",
    power=dict(alpha=0.05, power=0.80, z_sum=float(norm.ppf(0.975) + norm.ppf(0.80)), resolution=0.02,
               clusters="larger of event x state and county cluster-robust standard errors",
               entry="additive hazard u' = 1 - (1 - u) exp(-dt x), dt = 1 h, linearised through the base's own rollout "
                     "p_s = p_{s-1} + u_s (1 - p_{s-1}) - r_s p_{s-1}",
               regressors="within modulators quadn[psi*mod@tau] - quad[psi*one@tau] (nuisance quad[psi*one@tau]); "
                          "ladder contrasts C1-C4 (nuisance: the lower variant)"),
    ceiling=dict(settings="open_gcrk review2_checks.d3: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, "
                          "max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, "
                          "random_state=0)", targets="window peak and mean of observed hours, raw and log(t + 0.002)",
                 summaries="each variant: max and mean over the 144 forecast hours", null="quad summaries permuted "
                 "within event x state blocks", n_perm=20, n_boot=2000),
    kill=dict(c="no (contrast, feature) passes at 0.05 after max-T and Bonferroni, in the registered direction, under "
                "the null the rule selects",
              d="the pooled-RMSE gain at the minimum detectable effect exceeds the 2% resolution for every within-"
                "modulator regressor",
              e="X0 + quad summaries stays inside its within-block permutation null for every target and scale of the "
                "main design"),
)
QUICK = dict(n_mult=499, n_syn=100, n_mult_syn=199, ceiling_n_perm=2, n_boot=200)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- inputs
CTX_NAMES = ["log_cust", "rucc", "log_pop_density", "coop_share", "log1p_n_utilities", "log1p_saidi", "p71",
             "p_max_prefix", "p_mean_66_71", "p_trend_65_71", "prefix_active_share"]


def load_base(path: Path):
    Z = np.load(path, allow_pickle=False)
    F = {k: Z[k] for k in ("y", "m", "y0", "cust", "fips", "event", "geo", "geo_features")}
    rf = [str(x) for x in Z["recovery_features"]]
    if rf[14:25] != CTX_NAMES or not all(x.startswith("nbr_") for x in rf[25:33]):
        raise SystemExit(f"{path.name}: recovery-feature layout differs from the one the D3 unit features assume")
    xu, xr = Z["xu"], Z["xr"]
    parts = []
    for a in range(0, len(xu), 1000):                     # identical to review2_checks.unit_features, in row chunks
        u64, r64 = xu[a:a + 1000].astype(np.float64), xr[a:a + 1000].astype(np.float64)
        fw = np.concatenate([u64[:, 72:].max(1), u64[:, 72:].mean(1), u64[:, :72].mean(1)], 1)
        parts.append(np.concatenate([fw, r64[:, 72:, 25:33].max(1), r64[:, 0, 14:25]], 1))
    F["X0"] = np.concatenate(parts, 0)
    del xu, xr, Z
    gc.collect()
    return F


def _npz_memmap(path: Path, member: str):
    """Read-only memmap of an uncompressed .npz member (None if the member is compressed)."""
    import struct
    import zipfile
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            return None
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        head = fh.read(30)
        n_name, n_extra = struct.unpack("<HH", head[26:30])
        fh.seek(info.header_offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(fh)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(fh)
        offset = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


def load_phi(path: Path, F: dict):
    """phi as a read-only memmap when the file is uncompressed (keeps resident memory small), else loaded."""
    z = np.load(path, allow_pickle=False)
    if not (np.array_equal(z["fips"].astype(str), F["fips"].astype(str))
            and np.array_equal(z["event"].astype(str), F["event"].astype(str))):
        raise SystemExit(f"{path.name}: unit order differs from features_e3r2.npz")
    names = [str(x) for x in z["names"]]
    phi = _npz_memmap(path, "phi")
    if phi is None:
        phi = z["phi"]
    if phi.shape[:2] != (len(F["y"]), T) or phi.shape[2] != len(names):
        raise SystemExit(f"{path.name}: phi has shape {phi.shape}")
    return phi, names


def window_mean(X: np.ndarray, m: np.ndarray) -> np.ndarray:
    w = m.astype(np.float64)
    if X.ndim == 3:
        w = w[:, :, None]
    return (np.asarray(X, np.float64) * w).sum(1) / np.maximum(w.sum(1), 1e-12)


def summaries(phi: np.ndarray, m: np.ndarray, chunk: int = 16):
    """Observed-window mean [U,F] (float64) and max / mean over all hours [U,F] (float32); non-finite -> 0."""
    U, _, Fn = phi.shape
    wm, mx, mn = np.empty((U, Fn)), np.empty((U, Fn), np.float32), np.empty((U, Fn), np.float32)
    bad = 0
    for a in range(0, Fn, chunk):
        X = phi[:, :, a:a + chunk].astype(np.float32)
        fin = np.isfinite(X)
        bad += int((~fin).sum())
        X = np.where(fin, X, 0.0)
        wm[:, a:a + chunk] = window_mean(X, m)
        mx[:, a:a + chunk], mn[:, a:a + chunk] = X.max(1), X.mean(1)
    return wm, mx, mn, bad


def county_centroids(fips: np.ndarray):
    d = pd.read_parquet(AREA_WEIGHTS)
    d["lat"], d["lon"] = 50.0 - 0.25 * d["i"], -125.0 + 0.25 * d["j"]     # 0.25 deg grid from 50 N, 125 W
    d["w"] = d["w"] / d.groupby("fips")["w"].transform("sum")
    c = d.assign(la=d.lat * d.w, lo=d.lon * d.w).groupby("fips")[["la", "lo"]].sum().reindex(pd.Index(fips))
    if c.isna().any().any():
        raise SystemExit("centroid missing for some counties")
    return c["la"].to_numpy(), c["lo"].to_numpy()


def es_lab(F: dict) -> np.ndarray:
    fips, ev = F["fips"].astype(str), F["event"].astype(str)
    return np.char.add(np.char.add(ev, "|"), np.array([f[:2] for f in fips]))


def unit_blocks(F: dict):
    fips, ev = F["fips"].astype(str), F["event"].astype(str)
    st = np.array([f[:2] for f in fips])
    gnames = [str(x) for x in F["geo_features"]]
    if "relief_p95_p5" not in gnames:
        log("relief_p95_p5 missing: fixed-effect blocks are event x state only")
        return es_lab(F), es_lab(F)
    rel = F["geo"][:, gnames.index("relief_p95_p5")].astype(np.float64)
    first = pd.Series(np.arange(len(fips))).groupby(fips).first().to_numpy()
    cuts = np.nanquantile(rel[first], [1 / 3, 2 / 3])
    terc = np.where(np.isnan(rel), 3, np.searchsorted(cuts, rel, side="right"))
    lab = np.char.add(np.char.add(np.char.add(ev, "|"), st), np.char.add("|", terc.astype(str)))
    es = np.char.add(np.char.add(ev, "|"), st)
    return lab, es


# ------------------------------------------------------------------------------------------ small algebra
def demean(X: np.ndarray, bid: np.ndarray) -> np.ndarray:
    X2 = np.asarray(X, np.float64).reshape(len(X), -1)
    cnt = np.bincount(bid).astype(np.float64)
    means = np.stack([np.bincount(bid, X2[:, c], len(cnt)) for c in range(X2.shape[1])], 1) / cnt[:, None]
    return (X2 - means[bid]).reshape(np.shape(X))


def resid(V: np.ndarray, Z: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(Z, V, rcond=None)
    return V - Z @ beta


def correlogram(r, ev, lat, lon):
    edges = np.array([0, 25, 50, 75, 100, 150, 200, 300, 400, 600])
    num, cnt = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    var = float((r ** 2).mean())
    for e in np.unique(ev):
        i = np.where(ev == e)[0]
        d = _dist(lat[i], lon[i])
        iu = np.triu_indices(len(i), 1)
        dd, pp = d[iu], np.outer(r[i], r[i])[iu]
        b = np.searchsorted(edges, dd, side="right") - 1
        k = (b >= 0) & (b < len(num))
        num += np.bincount(b[k], pp[k], len(num))
        cnt += np.bincount(b[k], minlength=len(num))
    rho_b = num / np.maximum(cnt, 1) / var
    mid = 0.5 * (edges[1:] + edges[:-1])
    best = (np.inf, 0.0, 50.0)
    for ell in np.geomspace(5, 2000, 400):
        e_b = np.exp(-mid / ell)
        c = float(np.clip((cnt * rho_b * e_b).sum() / max((cnt * e_b ** 2).sum(), 1e-12), 0, 1))
        sse = float((cnt * (rho_b - c * e_b) ** 2).sum())
        if sse < best[0]:
            best = (sse, c, float(ell))
    return dict(bins_km=mid.tolist(), rho=rho_b.tolist(), pairs=cnt.tolist(), c=best[1], ell_km=best[2])


def _dist(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dy = la[:, None] - la[None, :]
    dx = (lo[:, None] - lo[None, :]) * np.cos(0.5 * (la[:, None] + la[None, :]))
    return 6371.0 * np.sqrt(dx ** 2 + dy ** 2)


def synthetic_fields(r, ev, lat, lon, c, ell, n, rng):
    out = np.empty((len(r), n), np.float32)
    for e in np.unique(ev):
        i = np.where(ev == e)[0]
        S = c * np.exp(-_dist(lat[i], lon[i]) / ell) + (1 - c) * np.eye(len(i))
        Lc = np.linalg.cholesky(S + 1e-9 * np.eye(len(i)))
        out[i] = (r[i].std() * (Lc @ rng.standard_normal((len(i), n)))).astype(np.float32)
    return out


# -------------------------------------------------------------------------------------- (c) unit alignment
def cluster_matrix(cid: np.ndarray):
    from scipy import sparse
    G = int(cid.max() + 1)
    return sparse.csr_matrix((np.ones(len(cid)), (cid, np.arange(len(cid)))), shape=(G, len(cid)))


def eff_clusters(W: np.ndarray) -> np.ndarray:
    """(sum_c w_c)^2 / sum_c w_c^2 per column of cluster masses W [G, F]."""
    tot, sq = W.sum(0), (W ** 2).sum(0)
    return np.where(sq > 0, tot ** 2 / np.where(sq > 0, sq, 1.0), 0.0)


def score_t(Sc: np.ndarray) -> np.ndarray:
    """Cluster-robust score t per column from cluster sums Sc [G, F]."""
    den = np.sqrt((Sc ** 2).sum(0))
    return np.where(den > 0, Sc.sum(0) / np.where(den > 0, den, 1.0), np.nan)


def sign_flips(E: int, rng) -> np.ndarray:
    """All 2^E sign patterns over E clusters (a random 65,536 of them when E > 16)."""
    if E <= 16:
        return (((np.arange(2 ** E)[:, None] >> np.arange(E)[None, :]) & 1) * 2 - 1).astype(np.float64)
    return rng.choice(np.array([-1.0, 1.0]), size=(65536, E))


def flip_p(Se: np.ndarray, t: np.ndarray, sg: int, flips: np.ndarray):
    """Exact sign-flip p-values over clusters (single and max-T) from cluster sums Se [E, F]."""
    ok = np.isfinite(t)
    den = np.sqrt((Se[:, ok] ** 2).sum(0))
    Tb = sg * (flips @ Se[:, ok]) / np.where(den > 0, den, 1.0)
    p1, pm = np.full(len(t), np.nan), np.full(len(t), np.nan)
    p1[ok] = (Tb >= sg * t[ok][None, :]).mean(0)
    pm[ok] = (Tb.max(1)[None, :] >= sg * t[ok][:, None]).mean(1)
    return p1, pm


def mult_maxT(Sc: np.ndarray, t: np.ndarray, sg: int, B: int, rng):
    """Westfall-Young max-T p-values and single-test p-values from a Rademacher multiplier bootstrap of cluster scores."""
    ok = np.isfinite(t)
    den = np.sqrt((Sc[:, ok] ** 2).sum(0))
    mx, pj = np.empty(B), np.zeros(int(ok.sum()))
    for s in range(0, B, 500):
        W = rng.choice(np.array([-1.0, 1.0]), size=(min(B, s + 500) - s, Sc.shape[0]))
        Tb = sg * (W @ Sc[:, ok]) / den
        mx[s:s + len(W)] = Tb.max(1)
        pj += (Tb >= sg * t[ok][None, :]).sum(0)
    p_single, p_max = np.full(len(t), np.nan), np.full(len(t), np.nan)
    p_single[ok] = (1 + pj) / (B + 1)
    p_max[ok] = (1 + (mx[None, :] >= sg * t[ok][:, None]).sum(1)) / (B + 1)
    return p_single, p_max, float(np.quantile(mx, 0.95))


def audit_c(F, P, m, S, names, reg, rng, contrasts=None, only=None):
    contrasts = CONTRASTS if contrasts is None else contrasts
    jj = np.arange(len(names)) if only is None else np.array([names.index(x) for x in only])
    names = [names[j] for j in jj]
    y = F["y"].astype(np.float64)
    keep = m.sum(1) > 0
    lab, es = unit_blocks(F)
    bid = np.unique(lab[keep], return_inverse=True)[1]
    cid = np.unique(es[keep], return_inverse=True)[1]
    M = cluster_matrix(cid)
    Mc = cluster_matrix(np.unique(F["fips"].astype(str)[keep], return_inverse=True)[1])
    ev = F["event"].astype(str)[keep]
    eid = np.unique(ev, return_inverse=True)[1]
    Me = cluster_matrix(eid)
    flips = sign_flips(int(eid.max() + 1), rng)
    lat, lon = county_centroids(F["fips"].astype(str)[keep])
    r = window_mean(y - P, m)[keep]
    Zc = demean(np.column_stack([window_mean(P, m)[keep], np.log(np.maximum(F["cust"].astype(np.float64), 1))[keep]]), bid)
    rr = resid(demean(r, bid), Zc)
    r_ev = r - pd.Series(r).groupby(ev).transform("mean").to_numpy()
    cg = correlogram(r_ev, ev, lat, lon)
    syn = {}
    for tag, mult, het in (("homo l", 1.0, False), ("homo 2l", 2.0, False), ("hetero l", 1.0, True), ("hetero 2l", 2.0, True)):
        R = synthetic_fields(r_ev, ev, lat, lon, cg["c"], mult * cg["ell_km"], reg["n_syn"], rng).astype(np.float64)
        if het:                                   # keep each unit's own residual magnitude, random spatially dependent sign
            R = np.abs(r_ev)[:, None] * R                 # (t statistics are scale-free)
        syn[tag] = resid(demean(R, bid), Zc)
    rows, fam = [], {}
    sg = reg["sign"]
    for cname, hi, lo, ctrl in contrasts:
        wm = {v: S[v][0][keep][:, jj] for v in (hi, lo) + ((ctrl,) if ctrl else ())}
        Dm = np.zeros((len(r), len(names)))
        ok = np.zeros(len(names), bool)
        for j in range(len(names)):
            Zj = [Zc, demean(wm[lo][:, j], bid)[:, None]]
            if ctrl:
                Zj.append(demean(wm[ctrl][:, j] - wm[lo][:, j], bid)[:, None])
            raw = wm[hi][:, j] - wm[lo][:, j]
            D = resid(demean(raw, bid), np.concatenate(Zj, 1))
            if np.linalg.norm(D) > 1e-9 * max(np.linalg.norm(raw), 1e-300):
                Dm[:, j], ok[j] = D, True
        Dn = Dm / np.where(ok, np.linalg.norm(Dm, axis=0), 1.0)
        pc = np.where(ok, (rr @ Dn) / np.linalg.norm(rr), np.nan)                 # part correlation (effect size)
        Sc = np.asarray(M @ (rr[:, None] * Dm))                                   # [G, F] cluster scores
        ge_es = eff_clusters(np.asarray(M @ (Dm ** 2)))
        ge_cty = eff_clusters(np.asarray(Mc @ (Dm ** 2)))
        ge_ev = eff_clusters(np.asarray(Me @ (Dm ** 2)))
        t = np.where(ok, score_t(Sc), np.nan)
        p_single, p_mult, crit = mult_maxT(Sc, t, sg, reg["n_mult"], rng)
        Se = np.asarray(Me @ (rr[:, None] * Dm))
        t_ev = np.where(ok, score_t(Se), np.nan)
        pe_single, pe_max = flip_p(Se, t_ev, sg, flips)
        fwer, p_syn = {}, {}
        for tag, Rs in syn.items():
            ts = np.full((Rs.shape[1], len(names)), np.nan)
            for j in np.where(ok)[0]:
                ts[:, j] = score_t(np.asarray(M @ (Rs * Dm[:, [j]])))
            mxs = np.nanmax(sg * ts, 1)
            fwer[tag] = float((mxs >= crit).mean())
            p_syn[tag] = np.where(ok, (1 + (mxs[None, :] >= sg * t[:, None]).sum(1)) / (len(mxs) + 1), np.nan)
        worst = max(fwer, key=fwer.get)
        use_syn = fwer[worst] > 0.10
        p_use = p_syn[worst] if use_syn else p_mult
        fam[cname] = dict(features_tested=int(ok.sum()), clusters=int(Sc.shape[0]), mult_maxT_crit95=crit, fwer_syn=fwer,
                          null_used=f"synthetic field ({worst})" if use_syn else "cluster multiplier")
        for j, nm in enumerate(names):
            rows.append(dict(contrast=cname, feature=nm, part_corr=pc[j], t=t[j], eff_clusters_es=ge_es[j],
                             eff_clusters_county=ge_cty[j], eff_clusters_event=ge_ev[j], p_single=p_single[j],
                             p_maxT_mult=p_mult[j], p_maxT_syn=p_syn[worst][j], p_maxT=p_use[j],
                             p_final=min(1.0, len(contrasts) * p_use[j]) if np.isfinite(p_use[j]) else np.nan,
                             t_event=t_ev[j], p_single_event_flip=pe_single[j], p_maxT_event_flip=pe_max[j],
                             p_final_event_flip=min(1.0, len(contrasts) * pe_max[j]) if np.isfinite(pe_max[j]) else np.nan))
        log(f"(c) {cname}: {int(ok.sum())} features, null {fam[cname]['null_used']}, FWER syn "
            + ", ".join(f"{k} {v:.3f}" for k, v in fwer.items()))
    A = pd.DataFrame(rows)
    A["eligible"] = np.minimum(A["eff_clusters_es"], A["eff_clusters_county"]) >= reg["min_eff_clusters"]
    if reg.get("min_eff_events") is not None:
        A["eligible"] &= A["eff_clusters_event"] >= reg["min_eff_events"]
    A["pass"] = (A["p_final"] < 0.05) & (sg * A["t"] > 0) & A["eligible"]
    if reg.get("require_event_flip"):
        A["pass"] &= A["p_final_event_flip"] < 0.05
    info = dict(n_units=int(keep.sum()), n_blocks=int(bid.max() + 1), singleton_blocks=int((np.bincount(bid) == 1).sum()),
                n_clusters=int(cid.max() + 1), n_events=int(eid.max() + 1), correlogram=cg, families=fam,
                killed=bool(not A["pass"].any()))
    return A, info


# ------------------------------------------------------- single pre-registered test (power first, then the test)
def unit_residual(F, P, m, scale: str):
    """Window-mean residual and window-mean prediction on the chosen scale (log: log(x + 0.002), as the D3 targets)."""
    y = F["y"].astype(np.float64)
    if scale == "log":
        return window_mean(np.log(y + 0.002) - np.log(P + 0.002), m), window_mean(np.log(P + 0.002), m)
    return window_mean(y - P, m), window_mean(P, m)


def single_h1a(F, P, m, S, names, feat, contrast, reg, rng, target, out: Path, scale="raw", diagnostic=False):
    """One registered (contrast, feature) test. Order: eligibility (design only) -> power at the target part correlation
    (noise from the residual's marginal properties only: per-event spread, correlogram, magnitudes; never its alignment
    with the feature) -> written to disk -> the test itself, only if power >= 0.8."""
    cname, hi, lo, ctrl = contrast
    j = names.index(feat)
    sg = reg["sign"]
    keep = m.sum(1) > 0
    lab, es = unit_blocks(F)
    bid = np.unique(lab[keep], return_inverse=True)[1]
    cid = np.unique(es[keep], return_inverse=True)[1]
    fid = np.unique(F["fips"].astype(str)[keep], return_inverse=True)[1]
    ev = F["event"].astype(str)[keep]
    eid = np.unique(ev, return_inverse=True)[1]
    M, Mc, Me = cluster_matrix(cid), cluster_matrix(fid), cluster_matrix(eid)
    flips = sign_flips(int(eid.max() + 1), rng)
    r_all, pbar = unit_residual(F, P, m, scale)
    Zc = demean(np.column_stack([pbar[keep], np.log(np.maximum(F["cust"].astype(np.float64), 1))[keep]]), bid)
    wm = {v: S[v][0][keep][:, j] for v in (hi, lo) + ((ctrl,) if ctrl else ())}
    Zj = [Zc, demean(wm[lo], bid)[:, None]]
    if ctrl:
        Zj.append(demean(wm[ctrl] - wm[lo], bid)[:, None])
    raw = wm[hi] - wm[lo]
    D = resid(demean(raw, bid), np.concatenate(Zj, 1))
    geff = {k: float(eff_clusters(np.asarray(Mx @ (D ** 2))[:, None])[0]) for k, Mx in
            (("event_state", M), ("county", Mc), ("event", Me))}
    res = dict(test=f"{cname.split()[0]}:{feat}", contrast=cname, feature=feat, sign=sg, n_units=int(keep.sum()),
               n_events=int(eid.max() + 1), n_event_state=int(cid.max() + 1), eff_clusters=geff,
               nonzero_event_state=int((np.bincount(cid, np.abs(raw)) > 0).sum()),
               nonzero_events=int((np.bincount(eid, np.abs(raw)) > 0).sum()), min_eff_clusters=reg["min_eff_clusters"],
               min_eff_events=reg.get("min_eff_events"), require_event_flip=bool(reg.get("require_event_flip")),
               residual_scale=scale, diagnostic_power=bool(diagnostic))
    res["eligible"] = bool(geff["event_state"] >= reg["min_eff_clusters"] and geff["county"] >= reg["min_eff_clusters"]
                           and (reg.get("min_eff_events") is None or geff["event"] >= reg["min_eff_events"]))
    if not res["eligible"] and not diagnostic:
        res["verdict"] = "not testable (eligibility, decided from the design before any residual was read)"
        (out / "H1a_power.json").write_text(json.dumps(res, indent=1) + "\n")
        return res
    # noise model: the residual's marginal properties only
    r = r_all[keep]
    r_ev = r - pd.Series(r).groupby(ev).transform("mean").to_numpy()
    lat, lon = county_centroids(F["fips"].astype(str)[keep])
    cg = correlogram(r_ev, ev, lat, lon)
    nD, B0 = np.linalg.norm(D), reg["n_mult_syn"]
    W0 = rng.choice(np.array([-1.0, 1.0]), size=(B0, M.shape[0]))

    def stats(R):
        """Per column of R (residual draws, already residualised): t, multiplier p (own draws), exact event-flip p."""
        Sc, Se = np.asarray(M @ (R * D[:, None])), np.asarray(Me @ (R * D[:, None]))
        t, te = score_t(Sc), score_t(Se)
        den = np.sqrt((Sc ** 2).sum(0))
        pm = (1 + (sg * (W0 @ Sc) / den >= sg * t[None, :]).sum(0)) / (B0 + 1)
        dene = np.sqrt((Se ** 2).sum(0))
        pf = (sg * (flips @ Se) / np.where(dene > 0, dene, 1.0) >= sg * te[None, :]).mean(0)
        return t, pm, pf

    h0, h1, fpr, powr = {}, {}, {}, {}
    for tag, mult, het in (("homo l", 1.0, False), ("homo 2l", 2.0, False), ("hetero l", 1.0, True), ("hetero 2l", 2.0, True)):
        R = synthetic_fields(r_ev, ev, lat, lon, cg["c"], mult * cg["ell_km"], reg["n_syn"], rng).astype(np.float64)
        if het:
            R = np.abs(r_ev)[:, None] * R
        R0 = resid(demean(R, bid), Zc)
        h0[tag] = stats(R0)
        fpr[tag] = float((h0[tag][1] < 0.05).mean())
        rho = float(target)
        beta = rho * np.linalg.norm(R0, axis=0) / (nD * np.sqrt(1 - rho ** 2))
        h1[tag] = stats(R0 + D[:, None] * beta[None, :])
    worst = max(fpr, key=fpr.get)
    use_syn = fpr[worst] > 0.10
    t0w = sg * h0[worst][0]
    for tag in h1:
        t1, pm1, pf1 = h1[tag]
        ps1 = (1 + (t0w[None, :] >= sg * t1[:, None]).sum(1)) / (len(t0w) + 1)
        pu = ps1 if use_syn else pm1
        ok = (pu < 0.05) & (sg * t1 > 0)
        if reg.get("require_event_flip"):
            ok &= pf1 < 0.05
        powr[tag] = float(ok.mean())
    res.update(correlogram=cg, fpr_single_test=fpr, fpr_joint_with_event_flip={k: float(((v[1] < .05) & (v[2] < .05)).mean())
               for k, v in h0.items()}, null_used=f"synthetic field ({worst})" if use_syn else "cluster multiplier",
               power_target_part_corr=float(target), power=powr, power_min=float(min(powr.values())),
               n_syn=int(reg["n_syn"]), n_mult_per_draw=int(B0))
    res["informative"] = bool(res["power_min"] >= 0.8)
    if not res["eligible"]:
        res["verdict"] = "diagnostic power only: not eligible, so the test was not computed"
        (out / "H1a_power.json").write_text(json.dumps(res, indent=1) + "\n")
        return res
    if not res["informative"]:
        res["verdict"] = "uninformative (power below 0.8): the test was not computed"
        (out / "H1a_power.json").write_text(json.dumps(res, indent=1) + "\n")
        return res
    (out / "H1a_power.json").write_text(json.dumps(res, indent=1) + "\n")          # on disk before the test is computed
    log(f"power {res['power_min']:.3f} (target part corr {target}); computing the registered test")
    rr = resid(demean(r, bid), Zc)
    Sc, Se = np.asarray(M @ (rr * D))[:, None], np.asarray(Me @ (rr * D))[:, None]
    t, te = score_t(Sc)[0], score_t(Se)[0]
    Wb = rng.choice(np.array([-1.0, 1.0]), size=(reg["n_mult"], M.shape[0]))
    p_mult = float((1 + (sg * (Wb @ Sc[:, 0]) / np.sqrt((Sc ** 2).sum()) >= sg * t).sum()) / (reg["n_mult"] + 1))
    p_syn = float((1 + (t0w >= sg * t).sum()) / (len(t0w) + 1))
    p_flip = float((sg * (flips @ Se[:, 0]) / np.sqrt((Se ** 2).sum()) >= sg * te).mean())
    p_used = p_syn if use_syn else p_mult
    passed = bool(p_used < 0.05 and sg * t > 0 and (p_flip < 0.05 or not reg.get("require_event_flip")))
    res.update(part_corr=float(rr @ D / (np.linalg.norm(rr) * nD)), t_event_state=float(t), t_event=float(te),
               p_multiplier=p_mult, p_synthetic=p_syn, p_used=p_used, p_event_flip=p_flip, passed=passed,
               verdict="pass" if passed else "fail (power >= 0.8)")
    return res


# ------------------------------------------------------------------------ (c) hour level and (d) power
def propagate(x, a_c, b_c):
    """z_s = a_s z_{s-1} + b_s x_s, z_{-1} = 0 (x [U,T])."""
    out = np.empty(x.shape)
    acc = np.zeros(x.shape[0])
    for s in range(x.shape[1]):
        acc = a_c[:, s] * acc + b_c[:, s] * x[:, s]
        out[:, s] = acc
    return out


class Hourly:
    """Everything that needs hourly arrays of two variants at a time."""

    def __init__(self, F, P, u, r, m, reg, rng):
        self.m = m.astype(bool)
        self.reg = reg
        y = F["y"].astype(np.float64)
        self.ev = F["event"].astype(str)
        self.e_dm = self._dm(y - P)[self.m]                               # unit- and lead-demeaned residual cells
        self.ne = np.linalg.norm(self.e_dm)
        cells_ev = np.repeat(self.ev[:, None], T, 1)[self.m]
        self.cid_ev = np.unique(cells_ev, return_inverse=True)[1]         # event clusters (exact sign flip)
        E = int(self.cid_ev.max() + 1)
        self.flips = (((np.arange(2 ** E)[:, None] >> np.arange(E)[None, :]) & 1) * 2 - 1).astype(np.float64) \
            if E <= 16 else rng.choice(np.array([-1.0, 1.0]), size=(65536, E))
        p_prev = np.concatenate([F["y0"].astype(np.float64)[:, None], P[:, :-1]], 1)
        self.prop = (1.0 - u - r, (1.0 - p_prev) * (1.0 - u))
        self.e_c = (y - P)[self.m]
        self.see = float((self.e_c ** 2).sum())
        fips = F["fips"].astype(str)
        es = np.char.add(np.char.add(self.ev, "|"), np.array([f[:2] for f in fips]))
        self.cl = {"event_state": np.unique(np.repeat(es[:, None], T, 1)[self.m], return_inverse=True)[1],
                   "county": np.unique(np.repeat(fips[:, None], T, 1)[self.m], return_inverse=True)[1]}
        self.M_es = cluster_matrix(self.cl["event_state"])
        self.M_ev = cluster_matrix(self.cid_ev)

    def _dm(self, X):
        mb = self.m
        X = np.where(mb, X, 0.0)
        X = X - np.where(mb, (X.sum(1) / np.maximum(mb.sum(1), 1))[:, None], 0.0)
        return X - np.where(mb, (X.sum(0) / np.maximum(mb.sum(0), 1))[None, :], 0.0)

    def hour_scores(self, D):
        """Part correlation and cluster sums (event x state, event) of residual x contrast over demeaned cells."""
        Dd = self._dm(D)[self.m]
        nd = np.linalg.norm(Dd)
        if nd < 1e-12 or self.ne < 1e-12:
            return np.nan, None, None
        prod = self.e_dm * Dd
        self.last_geff = float(eff_clusters(np.asarray(self.M_es @ (Dd ** 2))[:, None])[0])
        return float(prod.sum() / (self.ne * nd)), np.asarray(self.M_es @ prod), np.asarray(self.M_ev @ prod)

    def power(self, x, nuis):
        """Within/ladder coefficient: propagated regressor x with propagated nuisance; cluster-robust SE."""
        rp = self.reg["power"]
        xt, nt = propagate(x, *self.prop)[self.m], propagate(nuis, *self.prop)[self.m]
        N = len(xt)
        X = np.column_stack([np.ones(N), nt, xt])
        b01, *_ = np.linalg.lstsq(X[:, :2], xt, rcond=None)
        xp2 = (xt - X[:, :2] @ b01) ** 2
        sxx = float(xp2.sum())
        row = dict(sd_regressor=float(x[self.m].std()))
        for cname, cid in self.cl.items():
            row[f"eff_clusters_{cname}"] = float(eff_clusters(np.bincount(cid, xp2)[:, None])[0])
        if not np.isfinite(sxx) or sxx <= 1e-24 * max(float((xt ** 2).sum()), 1e-300):
            return dict(row, se=np.nan, mde=np.nan, gain_at_mde=np.nan)
        inv = np.linalg.pinv(X.T @ X)
        th = inv @ (X.T @ self.e_c)
        res = self.e_c - X @ th
        for cname, cid in self.cl.items():
            Sg = np.column_stack([np.bincount(cid, X[:, c] * res) for c in range(3)])
            G = Sg.shape[0]
            V = inv @ (Sg.T @ Sg) @ inv * (G / (G - 1)) * ((N - 1) / (N - 3))
            row[f"se_{cname}"] = float(np.sqrt(max(V[2, 2], 0.0)))
        se = max(row["se_event_state"], row["se_county"])
        mde = rp["z_sum"] * se
        th2 = float(np.sqrt((1 - (1 - rp["resolution"]) ** 2) * self.see / sxx))
        return dict(row, theta_hat=float(th[2]), se=se, t=float(th[2] / se) if se > 0 else np.nan, mde=mde,
                    gain_at_mde=float(1 - np.sqrt(max(0.0, 1 - mde ** 2 * sxx / self.see))), theta_for_2pct=th2,
                    power_at_2pct=float(norm.cdf(th2 / se - norm.ppf(0.975)) + norm.cdf(-th2 / se - norm.ppf(0.975))))


def _hour_one(H, cname, nm, a_, b_, feats, pcs, S_es, S_ev, drows):
    pc, s_es, s_ev = H.hour_scores(a_ - b_)
    if s_es is not None:
        feats.append((nm, H.last_geff))
        pcs.append(pc)
        S_es.append(s_es)
        S_ev.append(s_ev)
    drows.append(dict(kind="ladder", contrast=cname, feature=nm, **H.power(a_ - b_, b_)))


def audit_hourly(F, P, u, r, m, reg, rng, paths, names, contrasts=None, only=None):
    contrasts = CONTRASTS if contrasts is None else contrasts
    H = Hourly(F, P, u, r, m, reg, rng)
    sg = reg["sign"]
    hrows, drows = [], []
    for cname, hi, lo, ctrl in contrasts:
        if ctrl:
            continue
        log(f"(c-hour, d) {cname}")
        A, na = load_phi(paths[hi], F)
        B, nb = load_phi(paths[lo], F)
        jb = {n_: i for i, n_ in enumerate(nb)}
        feats, pcs, S_es, S_ev = [], [], [], []
        pairs = [(ja, jb[nm], nm) for ja, nm in enumerate(na) if nm in jb and (only is None or nm in only)]
        for q0 in range(0, len(pairs), 16):
            chunk = pairs[q0:q0 + 16]
            Ac = np.asarray(A[:, :, [c[0] for c in chunk]], np.float32)
            Bc = np.asarray(B[:, :, [c[1] for c in chunk]], np.float32)
            for k, (_, _, nm) in enumerate(chunk):
                a_, b_ = np.nan_to_num(Ac[:, :, k].astype(np.float64)), np.nan_to_num(Bc[:, :, k].astype(np.float64))
                _hour_one(H, cname, nm, a_, b_, feats, pcs, S_es, S_ev, drows)
            del Ac, Bc
        if feats:
            S1, S2 = np.stack(S_es, 1), np.stack(S_ev, 1)
            t1, t2 = score_t(S1), score_t(S2)
            _, p1, _ = mult_maxT(S1, t1, sg, reg["n_mult"], rng)
            d2 = np.sqrt((S2 ** 2).sum(0))
            mx2 = (sg * (H.flips @ S2) / np.where(d2 > 0, d2, 1.0)).max(1)            # exact sign flip over events
            p2 = (1 + (mx2[None, :] >= sg * t2[:, None]).sum(1)) / (len(mx2) + 1)
            for k, (nm, geff) in enumerate(feats):
                hrows.append(dict(contrast=cname, feature=nm, eff_clusters_es=geff, part_corr=pcs[k], t_event_state=t1[k],
                                  p_maxT=p1[k], t_event=t2[k], p_maxT_event_flip=p2[k]))
        del A, B
        gc.collect()
    log("(d) within modulators (quadn against quad)")
    A, na = load_phi(paths["quadn"], F)
    B, nb = load_phi(paths["quad"], F)
    jb = {n_: i for i, n_ in enumerate(nb)}
    todo = []
    for ja, nm in enumerate(na):
        try:
            psi, rest = nm.split("*", 1)
            mod, tau = rest.split("@", 1)
        except ValueError:
            continue
        ref = f"{psi}*one@{tau}"
        if mod != "one" and ref in jb and (only is None or nm in only):
            todo.append((ja, jb[ref], nm, mod))
    for q0 in range(0, len(todo), 16):
        chunk = todo[q0:q0 + 16]
        Ac = np.asarray(A[:, :, [c[0] for c in chunk]], np.float32)
        Bc = np.asarray(B[:, :, [c[1] for c in chunk]], np.float32)
        for k, (_, _, nm, mod) in enumerate(chunk):
            a_, b_ = np.nan_to_num(Ac[:, :, k].astype(np.float64)), np.nan_to_num(Bc[:, :, k].astype(np.float64))
            drows.append(dict(kind="within", contrast=f"quadn-quad [{mod}]", feature=nm, modulator=mod,
                              **H.power(a_ - b_, b_)))
        del Ac, Bc
    del A, B
    gc.collect()
    Hd = pd.DataFrame(hrows)
    if len(Hd):
        k = max(1, sum(1 for c in contrasts if not c[3]))
        Hd["p_final"] = np.minimum(1.0, k * Hd["p_maxT"])
        Hd["p_final_event_flip"] = np.minimum(1.0, k * Hd["p_maxT_event_flip"])
    return Hd, pd.DataFrame(drows)


# -------------------------------------------------------------------------------------------- (e) ceiling
def unit_targets(F):
    m = F["m"].astype(bool)
    y = F["y"].astype(np.float64)
    return np.where(m, y, -np.inf).max(1), np.where(m, y, 0).sum(1) / np.maximum(m.sum(1), 1)


def audit_e(F, S, designs, reg, rng, sp, check_d3):
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = len(F["y"])
    fips, ev = F["fips"].astype(str), F["event"].astype(str)
    es = np.char.add(np.char.add(ev, "|"), np.array([f[:2] for f in fips]))
    blocks = [np.where(es == b)[0] for b in np.unique(es)]
    X0, G40 = F["X0"], F["geo"].astype(np.float64)
    summ = {v: np.concatenate([S[v][1], S[v][2]], 1).astype(np.float64) for v in S}
    arms = {"A0": ("X0 (D3 without geography)", X0), "A1": ("X0 + 40 descriptors (D3 with geography)", np.c_[X0, G40])}
    for k, v in enumerate(VARIANTS):
        arms[f"V{k}"] = (f"X0 + {v}", np.c_[X0, summ[v]])
    key = {v: f"V{k}" for k, v in enumerate(VARIANTS)}
    pairs = [("A0", key["quad"]), (key["area"], key["pop"]), (key["pop"], key["quad"]), (key["mean"], key["pooled"]),
             (key["pooled"], key["quad"]), (key["other"], key["quad"]), (key["other"], key["pooled"]),
             ("A0", key["area"])]
    peak, wmean = unit_targets(F)
    u_c, inv_c = np.unique(fips, return_inverse=True)
    wts = rng.multinomial(len(u_c), np.full(len(u_c), 1 / len(u_c)), size=reg["n_boot"])

    def fit_predict(X, t, folds, lg):
        tt = np.log(t + 0.002) if lg else t
        pred = np.full(n, np.nan)
        for spec in folds.values():
            dev, out = np.array(spec["dev"]), np.array(spec["outer"])
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                                l2_regularization=1.0, early_stopping=False, random_state=0)
            mdl.fit(X[dev], tt[dev])
            pred[out] = mdl.predict(X[out])
        return pred

    def score(pred, t, lg):
        tt = np.log(t + 0.002) if lg else t
        se = (pred - tt) ** 2
        return float(np.sqrt(se.mean())), float(1 - se.sum() / ((tt - tt.mean()) ** 2).sum()), se

    rows, prs = [], []
    for design in designs:
        folds = sp[design]
        for tname, t in (("peak", peak), ("mean", wmean)):
            for lg in (False, True):
                res = {a: score(fit_predict(X, t, folds, lg), t, lg) for a, (_, X) in arms.items()}
                null = []
                for _ in range(reg["ceiling_n_perm"]):
                    perm = np.arange(n)
                    for bi in blocks:
                        perm[bi] = rng.permutation(bi)
                    null.append(score(fit_predict(np.c_[X0, summ["quad"][perm]], t, folds, lg), t, lg)[0])
                null = np.array(null)
                for a, (label, _) in arms.items():
                    row = dict(design=design, target=tname, scale="log" if lg else "raw", arm=a, label=label,
                               rmse=res[a][0], r2=res[a][1])
                    if a == key["quad"]:
                        row.update(null_rmse_median=float(np.median(null)), null_rmse_p05=float(np.quantile(null, .05)),
                                   beats_null=bool(res[a][0] < np.quantile(null, .05)))
                    rows.append(row)
                for lo_a, hi_a in pairs:
                    s0, s1 = np.bincount(inv_c, res[lo_a][2]), np.bincount(inv_c, res[hi_a][2])
                    boot = np.sqrt((wts @ s1) / (wts @ s0)) - 1
                    prs.append(dict(design=design, target=tname, scale="log" if lg else "raw",
                                    comparison=f"{arms[hi_a][0]}  vs  {arms[lo_a][0]}", rel_change=res[hi_a][0] / res[lo_a][0] - 1,
                                    ci_lo=float(np.quantile(boot, .025)), ci_hi=float(np.quantile(boot, .975))))
                log(f"(e) {design} {tname} {'log' if lg else 'raw'} done")
    E, Pr = pd.DataFrame(rows), pd.DataFrame(prs)
    chk = {}
    ref_file = C.RESULTS / "review2_information.csv"
    if check_d3 and ref_file.exists():
        R = pd.read_csv(ref_file)
        for _, rw in R.iterrows():
            sub = E[(E.design == rw.design) & (E.target == rw.target) & (E.scale == rw.scale)]
            if len(sub):
                chk[f"{rw.design}/{rw.target}/{rw.scale}"] = dict(
                    d3_no_geo=float(rw.rmse_no_geo), A0=float(sub[sub.arm == "A0"].rmse.iloc[0]),
                    d3_geo=float(rw.rmse_geo), A1=float(sub[sub.arm == "A1"].rmse.iloc[0]))
    q = E[(E.design == "main") & (E.arm == key["quad"])]
    return E, Pr, dict(d3_reproduction=chk, killed=bool(len(q) and not q.beats_null.any()))


# ------------------------------------------------------------------------------------------------ report
def write_md(path, meta, A, info_c, Hd, Dd, E, Pr, info_e, reg):
    fmt = lambda x, f=".4f": "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, f)  # noqa: E731
    L = ["# F1 information audit — parts (c), (d), (e)", "",
         f"Generated {meta['generated']} by `audit_f1.py` (git {meta['git']}). Panel `{meta['panel']}`, splits "
         f"`{meta['splits']}`, inputs `{meta['eih_prefix']}<v>.npz`: {meta['n_units']} county-events, {meta['n_features']} "
         f"features per variant; variants {', '.join(meta['variants'])}. No model was trained; the base is the held-out "
         f"prediction of {meta['base']}. Settings (REG, with its dated revisions) are copied into the JSON next to this "
         "file. Parts (a) contrast and (b) redundancy are computed separately.", ""]
    if meta.get("quick"):
        L += ["**QUICK RUN — smoke settings, not for reading.**", ""]
    if meta.get("nonfinite"):
        L += [f"Non-finite values replaced by 0: {meta['nonfinite']}.", ""]
    if A is not None:
        L += ["## (c) Alignment with the base's held-out residual", "",
              f"{info_c['n_units']} county-events, {info_c['n_blocks']} fixed-effect blocks ({info_c['singleton_blocks']} singletons), "
              f"{info_c['n_clusters']} event x state clusters. Residual correlogram: c = {info_c['correlogram']['c']:.2f}, "
              f"l = {info_c['correlogram']['ell_km']:.0f} km. Direction {reg['sign']:+d}: {reg['sign_meaning']}.", "",
              f"A feature is eligible for a pass only with at least {reg['min_eff_clusters']} effective clusters in both "
              "clusterings; the best eligible feature is shown.", "",
              "| contrast | features (eligible) | FWER on synthetic fields (homo l / 2l, hetero l / 2l) | null used | best feature | part corr | t | p max-T | p final |",
              "|---|---:|---|---|---|---:|---:|---:|---:|"]
        for cname, fam in info_c["families"].items():
            sub = A[(A.contrast == cname) & A.eligible].sort_values("p_maxT")
            b = sub.iloc[0] if len(sub) else None
            fw = " / ".join(f"{v:.3f}" for v in fam["fwer_syn"].values())
            ne = int(A[(A.contrast == cname) & A.eligible].shape[0])
            L.append(f"| {cname} | {fam['features_tested']} ({ne}) | {fw} | {fam['null_used']} | {'' if b is None else b.feature} | "
                     f"{'' if b is None else fmt(b.part_corr, '+.3f')} | {'' if b is None else fmt(b.t, '+.2f')} | "
                     f"{'' if b is None else fmt(b.p_maxT)} | {'' if b is None else fmt(b.p_final)} |")
        if meta.get("single"):
            r1 = A.iloc[0]
            L += ["", f"**Single pre-registered test `{meta['single']}`** (one-sided, direction {reg['sign']:+d}; no max-T, no "
                  "Bonferroni):", "",
                  "| part corr | t (event x state) | p multiplier | p synthetic | p used | t (event) | p exact event flip | "
                  "eff. clusters event x state / county / event | eligible | pass |", "|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
                  f"| {fmt(r1.part_corr, '+.4f')} | {fmt(r1.t, '+.2f')} | {fmt(r1.p_single)} | {fmt(r1.p_maxT_syn)} | "
                  f"{fmt(r1.p_final)} | {fmt(r1.t_event, '+.2f')} | {fmt(r1.p_single_event_flip)} | "
                  f"{r1.eff_clusters_es:.1f} / {r1.eff_clusters_county:.1f} / {r1.eff_clusters_event:.1f} | "
                  f"{'yes' if r1.eligible else 'no'} | {'yes' if r1['pass'] else 'no'} |",
                  "", f"Eligibility: ≥ {reg['min_eff_clusters']} effective clusters at event x state and county level"
                  + (f", ≥ {meta['min_eff_events']} effective events" if meta.get("min_eff_events") else "")
                  + ("; a pass also needs the exact event sign-flip p < 0.05" if meta.get("require_event_flip") else "") + "."]
        L += ["", f"**(c) verdict: {'KILL' if info_c['killed'] else 'not killed'}** — rule: {reg['kill']['c']}.", ""]
        if Hd is not None and len(Hd):
            L += ["Hour level (secondary; unit- and lead-demeaned cells; Bonferroni over four contrasts; best feature with at "
                  f"least {reg['min_eff_clusters']} effective event x state clusters):", "",
                  "| contrast | best feature | eff. clusters | part corr | t (event x state) | p final | t (event) | p final, exact event flip |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
            for cname, sub in Hd.groupby("contrast", sort=False):
                sub = sub[sub.eff_clusters_es >= reg["min_eff_clusters"]]
                if not len(sub):
                    continue
                b = sub.sort_values("p_maxT").iloc[0]
                L.append(f"| {cname} | {b.feature} | {b.eff_clusters_es:.0f} | {b.part_corr:+.4f} | {b.t_event_state:+.2f} | "
                         f"{b.p_final:.4f} | {b.t_event:+.2f} | {b.p_final_event_flip:.4f} |")
            L.append("")
    if Dd is not None and len(Dd):
        rp = reg["power"]
        L += ["## (d) Power: minimum detectable effect of each regressor", "",
              f"MDE = {rp['z_sum']:.2f} x the larger of the event x state and county cluster-robust SEs. `gain at MDE` = pooled-RMSE "
              "reduction if an effect of exactly that size existed and were captured; `power at 2%` = power against the effect that "
              f"would move pooled RMSE by 2%. Only regressors with at least {reg['min_eff_clusters']} effective clusters in both "
              "clusterings enter the table and the verdict. t is descriptive only (not lead- or unit-demeaned).", "",
              "| kind | contrast | regressors (eligible / all) | gain at MDE: min / median / max | share with gain at MDE <= 2% | power at 2%: median | largest |t| (descr.) |",
              "|---|---|---:|---|---:|---:|---:|"]
        Dd = Dd.copy()
        Dd["eligible"] = (np.minimum(Dd.get("eff_clusters_event_state", np.nan), Dd.get("eff_clusters_county", np.nan))
                          >= reg["min_eff_clusters"])
        for (kind, cname), sub in Dd.groupby(["kind", "contrast"], sort=False):
            ok = sub.dropna(subset=["gain_at_mde"])
            ok = ok[ok.eligible]
            if not len(ok):
                L.append(f"| {kind} | {cname} | 0 / {len(sub)} | | | | |")
                continue
            L.append(f"| {kind} | {cname} | {len(ok)} / {len(sub)} | {100 * ok.gain_at_mde.min():.3f}% / "
                     f"{100 * ok.gain_at_mde.median():.3f}% / {100 * ok.gain_at_mde.max():.3f}% | "
                     f"{(ok.gain_at_mde <= rp['resolution']).mean():.2f} | {ok.power_at_2pct.median():.2f} | "
                     f"{ok.t.abs().max():.2f} |")
        w = Dd[(Dd.kind == "within") & Dd.eligible].dropna(subset=["gain_at_mde"])
        killed_d = None if len(w) == 0 else bool((w.gain_at_mde > rp["resolution"]).all())
        verdict_d = "undetermined (no eligible regressor)" if killed_d is None else ("KILL" if killed_d else "not killed")
        L += ["", f"**(d) verdict: {verdict_d}** — rule: {reg['kill']['d']}. An MDE is read against "
              "the effect sizes part (a) makes plausible.", ""]
        meta["killed_d"] = killed_d
    if E is not None and len(E):
        L += ["## (e) Information ceiling (D3 regressor with each variant's summaries)", "",
              "| design | target | scale | arm | RMSE | R2 | beats null |", "|---|---|---|---|---:|---:|---|"]
        for _, r_ in E.iterrows():
            bn = "" if pd.isna(r_.get("beats_null", np.nan)) else ("yes" if r_.beats_null else "no")
            L.append(f"| {r_.design} | {r_.target} | {r_.scale} | {r_.label} | {r_.rmse:.5f} | {r_.r2:.3f} | {bn} |")
        L += ["", "| design | target | scale | comparison | change | 95% county interval |", "|---|---|---|---|---:|---|"]
        for _, r_ in Pr.iterrows():
            L.append(f"| {r_.design} | {r_.target} | {r_.scale} | {r_.comparison} | {100 * r_.rel_change:+.2f}% | "
                     f"[{100 * r_.ci_lo:+.2f}%, {100 * r_.ci_hi:+.2f}%] |")
        if info_e["d3_reproduction"]:
            dmax = max(max(abs(v["d3_no_geo"] - v["A0"]), abs(v["d3_geo"] - v["A1"])) for v in info_e["d3_reproduction"].values())
            L += ["", f"D3 reproduction (A0, A1 against open_gcrk `results/e3r2/review2_information.csv`): max |difference| {dmax:.2e}."]
        L += ["", f"**(e) verdict: {'KILL' if info_e['killed'] else 'not killed'}** — rule: {reg['kill']['e']}.", ""]
    L += ["## Notes", "",
          "* (c) The review's donor null is realised by the `other` variant (a same-relief-stratum donor's bands under the "
          "county's own weather): C5 asks whether the county's own band distribution aligns beyond the donor's.",
          "* (d) Rates are propagated with the base's own u and r; no restoration time constant is assumed.", ""]
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eih-dir", type=Path, default=EIH)
    ap.add_argument("--eih-prefix", default="eih_", help="file prefix, e.g. eih_w1_ for the winter panel")
    ap.add_argument("--features", type=Path, default=None, help="panel features npz (default: open_gcrk layout)")
    ap.add_argument("--splits", type=Path, default=None, help="splits json (default: open_gcrk layout)")
    ap.add_argument("--base", default=None, help="per-fold base outer.npz pattern with {fold}, relative to the repository "
                    "root (default: open_gcrk layout, W+Cin, main, seed 0); must hold idx, P and, for (d), u and r")
    ap.add_argument("--design", default="main", help="splits design of the base's outer folds")
    ap.add_argument("--single", default=None, help='one pre-registered test "C<k>:<feature>" (no max-T, no Bonferroni)')
    ap.add_argument("--min-eff-events", type=float, default=None, help="also require this many effective event clusters")
    ap.add_argument("--require-event-flip", action="store_true", help="a pass also needs exact event sign-flip p < 0.05")
    ap.add_argument("--scale", choices=("raw", "log"), default="raw", help="residual scale of the single test")
    ap.add_argument("--diagnostic-power", action="store_true", help="compute power even when ineligible (never the test)")
    ap.add_argument("--power-target", type=float, default=None, help="with --single: power check at this part correlation "
                    "first; the test is computed only if power >= 0.8 (writes H1a_power.json, then H1a.json and H1a.md)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--parts", default="cde", help="subset of c, d, e (hour-level alignment runs with d)")
    ap.add_argument("--designs", default="main", help="(e): main[,event]")
    ap.add_argument("--quick", action="store_true", help="smoke settings, not for reading")
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()
    reg = json.loads(json.dumps(REG))
    reg.update(ceiling_n_perm=REG["ceiling"]["n_perm"], n_boot=REG["ceiling"]["n_boot"])
    if args.quick:
        reg.update(QUICK)
    reg.update(min_eff_events=args.min_eff_events, require_event_flip=bool(args.require_event_flip))
    contrasts, only = CONTRASTS, None
    if args.single:
        ck, feat = args.single.split(":", 1)
        contrasts = [c for c in CONTRASTS if c[0].split()[0] == ck]
        if not contrasts:
            raise SystemExit(f"--single: no contrast {ck}")
        only = [feat]
    rng = np.random.default_rng(args.seed)
    need = VARIANTS
    if args.single:
        c0 = [c for c in CONTRASTS if c[0].split()[0] == args.single.split(":", 1)[0]]
        need = tuple(v for v in VARIANTS if c0 and v in (c0[0][1], c0[0][2], c0[0][3]))
    paths = {v: args.eih_dir / f"{args.eih_prefix}{v}.npz" for v in need}
    miss = [p.name for p in paths.values() if not p.exists()]
    if miss:
        raise SystemExit(f"missing input files: {miss}")
    feat_path = args.features if args.features is not None else C.FEATURES
    feat_path = feat_path if feat_path.is_absolute() else ROOT / feat_path
    sp_path = args.splits if args.splits is not None else C.SPLITS_FILE
    sp_path = sp_path if sp_path.is_absolute() else ROOT / sp_path
    sp = json.loads(sp_path.read_text())
    default_panel = args.features is None and args.splits is None and args.base is None
    base_desc = args.base or f"open_gcrk {C.ROUND} W+Cin, main design, seed 0"
    log(f"panel {feat_path.name}; splits {sp_path.name}; base {base_desc}")
    t0 = time.time()
    F = load_base(feat_path)
    n = len(F["y"])
    if args.base is None:
        b = REG["base"]
        P, u, r = (C.collect(b["design"], b["arm"], b["seed"], n, key=k) for k in ("P", "u", "r"))
    else:
        P, u, r = (np.full((n, T), np.nan) for _ in range(3))
        for f, spec in sp[args.design].items():
            z = np.load(ROOT / args.base.format(fold=int(f)), allow_pickle=False)
            if not np.array_equal(np.sort(z["idx"]), np.sort(np.array(spec["outer"]))):
                raise SystemExit(f"base fold {f}: idx differs from the outer units of the splits")
            P[z["idx"]] = z["P"]
            for key, arr in (("u", u), ("r", r)):
                if key in z.files:
                    arr[z["idx"]] = z[key]
        if not np.isfinite(P).all():
            raise SystemExit("base predictions incomplete")
        if not (np.isfinite(u).all() and np.isfinite(r).all()):
            u = r = None
    if P is None:
        raise SystemExit("base predictions incomplete")
    if "d" in args.parts and (u is None or r is None):
        log("the base files hold no u and r: part (d) is skipped (it propagates through the base's own rates)")
        args.parts = args.parts.replace("d", "")
    m = F["m"].astype(bool)
    log(f"base loaded: {n} units")
    S, names, bad = {}, None, {}
    for v in paths:
        phi, nm = load_phi(paths[v], F)
        if names is None:
            names = nm
        elif nm != names:
            raise SystemExit(f"{paths[v].name}: feature names differ from the first input's")
        wm, mx, mn, nb = summaries(phi, m)
        S[v] = (wm, mx, mn)
        if nb:
            bad[v] = nb
        del phi
        gc.collect()
        log(f"summaries {v}")
    meta = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), panel=feat_path.name, splits=sp_path.name, base=base_desc,
                variants_loaded=list(paths),
                single=args.single, min_eff_events=args.min_eff_events, require_event_flip=bool(args.require_event_flip),
                eih_prefix=args.eih_prefix, variants=list(VARIANTS), n_units=int(n),
                n_features=len(names), features=names, nonfinite=bad, quick=bool(args.quick), seed=args.seed,
                inputs={p.name: sha256(p) for p in paths.values()},
                git=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                   text=True).stdout.strip())
    if args.single and args.power_target is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        res = single_h1a(F, P, m, S, names, only[0], contrasts[0], reg, rng, args.power_target, args.out,
                         scale=args.scale, diagnostic=args.diagnostic_power)
        res.update(meta=meta, registered=REG, runtime_s=round(time.time() - t0, 1))
        (args.out / "H1a.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        L = [f"# Registered single test {res['test']}", "", f"Generated {meta['generated']} by `audit_f1.py` (git {meta['git']}). "
             f"Panel `{meta['panel']}`, base {meta['base']}, residual scale {res['residual_scale']}. Order of operations: "
             "eligibility from the design, then power "
             f"(written to `H1a_power.json` before the test), then the test.", "",
             f"* Units {res['n_units']}, events {res['n_events']}, event x state clusters {res['n_event_state']}; nonzero contrast "
             f"in {res['nonzero_event_state']} event x state blocks and {res['nonzero_events']} events.",
             "* Effective clusters: event x state {event_state:.1f}, county {county:.1f}, event {event:.1f}; ".format(**res["eff_clusters"])
             + f"required ≥ {res['min_eff_clusters']} / ≥ {res['min_eff_clusters']} / ≥ {res['min_eff_events']}: "
             f"**{'eligible' if res['eligible'] else 'not eligible'}**."]
        if "power" in res:
            L += [f"* Single-test false-positive rate on synthetic fields: "
                  + ", ".join(f"{k} {v:.3f}" for k, v in res["fpr_single_test"].items()) + f"; null used: {res['null_used']}.",
                  f"* Power at part correlation {res['power_target_part_corr']}: "
                  + ", ".join(f"{k} {v:.3f}" for k, v in res["power"].items()) + f"; minimum {res['power_min']:.3f} "
                  f"(**{'informative' if res['informative'] else 'uninformative'}**)."]
        if "p_used" in res:
            L += [f"* Test: part corr {res['part_corr']:+.4f}, t (event x state) {res['t_event_state']:+.2f}, p multiplier "
                  f"{res['p_multiplier']:.4f}, p synthetic {res['p_synthetic']:.4f}, p used {res['p_used']:.4f}; t (event) "
                  f"{res['t_event']:+.2f}, exact event-flip p {res['p_event_flip']:.4f}."]
        L += ["", f"**Verdict: {res['verdict']}.**", ""]
        (args.out / "H1a.md").write_text("\n".join(L) + "\n")
        log(f"H1a: {res['verdict']}")
        return
    A = info_c = Hd = Dd = E = Pr = None
    info_e = dict(d3_reproduction={}, killed=None)
    if "c" in args.parts:
        if only and only[0] not in names:
            raise SystemExit(f"--single: feature {only[0]} not in the inputs")
        A, info_c = audit_c(F, P, m, S, names, reg, rng, contrasts, only)
    if "d" in args.parts:
        Hd, Dd = audit_hourly(F, P, u, r, m, reg, rng, paths, names, contrasts, only)
    if "e" in args.parts:
        E, Pr, info_e = audit_e(F, S, [d.strip() for d in args.designs.split(",")], reg, rng, sp, default_panel)
    args.out.mkdir(parents=True, exist_ok=True)
    for df, name in ((A, "alignment.csv"), (Hd, "alignment_hour.csv"), (Dd, "mde.csv"), (E, "ceiling.csv"),
                     (Pr, "ceiling_contrasts.csv")):
        if df is not None:
            df.to_csv(args.out / name, index=False)
    tag = "".join(sorted(set(args.parts) & set("cde"))) + ("_single" if args.single else "")
    write_md(args.out / f"F1_{tag}.md", meta, A, info_c, Hd, Dd, E, Pr, info_e, REG)

    def recs(d):
        return None if d is None else json.loads(d.to_json(orient="records"))
    (args.out / f"F1_{tag}.json").write_text(json.dumps(dict(
        meta=meta, registered=REG, run_settings={k: reg[k] for k in ("n_mult", "n_syn", "ceiling_n_perm", "n_boot")},
        c=None if A is None else dict(info_c, table=recs(A)), c_hour=recs(Hd), d=recs(Dd),
        e=None if E is None else dict(info_e, table=recs(E), contrasts=recs(Pr)), runtime_s=round(time.time() - t0, 1)),
        indent=1, default=float) + "\n")
    log(f"wrote {args.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
