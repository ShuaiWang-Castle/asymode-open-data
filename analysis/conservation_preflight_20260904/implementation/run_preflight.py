"""GitHub-only conservation and design preflight — main runner.

Zero neural training. No model repair. No campaign. No outcome-driven selection.

Inputs come exclusively from two pinned clean clones passed on the command line.
Outputs are written only under
``analysis/conservation_preflight_20260904/results/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import preflight_data as D
import preflight_lib as L
from preflight_lib import (CAP_R, CAP_U, CLOSURE_TOL, EVENT_ROW_CAP, K_BALANCE,
                           LOCAL_K, LOCAL_QUERIES, PCA_DIMS, constant_fit)

SEED = 0
MANIFEST_DIGEST = "db286b4960a4"
FOLD_DIGEST = "beb00a6762ba"
DATA_SHA = "8dd47c5ccd829611f27b69a3d64c274a0a24c400"
AUDITED_BASE = "d6555015cbe1c2b67f5197c725a8c8a785109b51"

ID_COLS = ["event", "county", "t_cur", "physical_hour", "y", "y_next", "delta",
           "one_customer_fraction"]


def sha12(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()[:12]


def git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def verify_integrity(code: Path, data: Path, log: list) -> dict:
    """Every provenance gate. Any failure raises and the run stops."""
    out = {}
    out["code_head"] = git(code, "rev-parse", "HEAD")
    out["code_branch"] = git(code, "rev-parse", "--abbrev-ref", "HEAD")
    out["data_head"] = git(data, "rev-parse", "HEAD")
    out["code_status_porcelain"] = git(code, "status", "--porcelain")
    out["audited_base"] = AUDITED_BASE

    if out["data_head"] != DATA_SHA:
        raise SystemExit(f"BLOCKED_data_commit_mismatch {out['data_head']}")
    subprocess.run(["git", "-C", str(code), "merge-base", "--is-ancestor",
                    AUDITED_BASE, "HEAD"], check=True)
    diff = git(code, "diff", "--name-only", f"{AUDITED_BASE}..HEAD",
               "--", "src", "experiments", "configs")
    out["base_to_head_src_experiments_configs_diff"] = diff
    if diff:
        raise SystemExit("BLOCKED_audited_base_diff_nonempty")

    # SHA-256 manifest of every public file
    res = subprocess.run(["shasum", "-a", "256", "-c", "data/SHA256SUMS.txt"],
                         cwd=str(data), capture_output=True, text=True)
    lines = [l for l in res.stdout.splitlines() if l.strip()]
    ok = sum(1 for l in lines if l.endswith(": OK"))
    bad = [l for l in lines if not l.endswith(": OK")]
    out["checksum_files_checked"] = len(lines)
    out["checksum_files_ok"] = ok
    out["checksum_failures"] = bad
    out["checksum_output_sha256"] = hashlib.sha256(res.stdout.encode()).hexdigest()
    out["checksum_command"] = "shasum -a 256 -c data/SHA256SUMS.txt"
    if res.returncode != 0 or bad or ok != len(lines):
        raise SystemExit("BLOCKED_public_checksum_failure")

    man = json.loads((data / "configs/panel_manifest_g3-all-26.json").read_text())
    if man["digest"] != MANIFEST_DIGEST:
        raise SystemExit("BLOCKED_manifest_digest_mismatch")
    if len(man["panels"]) != 26 or len(set(man["panels"])) != 26:
        raise SystemExit("BLOCKED_manifest_panel_count")
    recomputed = hashlib.sha256("|".join(sorted(man["panels"])).encode()).hexdigest()[:12]
    if recomputed != MANIFEST_DIGEST:
        raise SystemExit("BLOCKED_manifest_digest_not_reproducible")
    out["manifest_digest"] = man["digest"]
    out["manifest_digest_recomputed"] = recomputed
    out["n_panels"] = len(man["panels"])

    fmap = json.loads((code / "analysis/gpt_rescue_20260904/cc_v2/"
                              "event_folds_v2.json").read_text())
    body = {k: v for k, v in fmap.items() if k != "digest"}
    if fmap["digest"] != FOLD_DIGEST or sha12(body) != FOLD_DIGEST:
        raise SystemExit("BLOCKED_fold_digest_mismatch")
    out["fold_digest"] = fmap["digest"]
    out["fold_digest_recomputed"] = sha12(body)

    log.append(f"integrity: {ok}/{len(lines)} public files verified; "
               f"manifest {man['digest']}; folds {fmap['digest']}")
    return out, man, fmap


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def build_all(data: Path, panels: list[str], log: list):
    interim = data / "data/interim"
    events_df = pd.read_parquet(interim / "storm_events_county.parquet",
                                columns=["fips", "t_begin_utc", "t_end_utc"])
    fam = pd.read_parquet(interim / "event_days_stratified.parquet")
    fam["day"] = pd.to_datetime(fam["day"]).dt.strftime("%Y-%m-%d")
    family = dict(zip(fam["day"], fam["dominant"]))

    frames, windows, channels = [], {}, None
    for e in panels:
        p = D.load_event(interim, e)
        channels = channels or p.channels
        df = D.build_transitions(p)
        n_states = min(p.y_hourly.shape[1], p.X.shape[1])
        fp = D.noaa_hourly_footprint(events_df, p.fips, p.ts_hourly, n_states)
        comp = D.tie_break_composite(p, n_states)
        win = D.active_window(fp, comp, n_states)
        win.update(event=e, n_states=int(n_states), n_counties=int(len(p.fips)),
                   family=family.get(e, "unknown"),
                   observed_hour_share=float(p.obs_hourly.mean()))
        windows[e] = win
        df["in_active48"] = False
        if win["available"]:
            sel = (df["t_cur"] >= win["t_start"]) & (df["t_cur"] <= win["t_end"])
            df.loc[sel, "in_active48"] = True
        df["family"] = win["family"]
        frames.append(df)
        log.append(f"  {e} fam={win['family']:11s} n={len(df):7d} "
                   f"peak={win['peak']} fp={win['peak_footprint']:.3f} "
                   f"a48={'yes' if win['available'] else 'NO:'+win['reason']}")
    allrows = pd.concat(frames, ignore_index=True)
    return allrows, windows, channels


def design_rows(allrows: pd.DataFrame, design: str) -> pd.DataFrame:
    return allrows if design == "full" else allrows[allrows["in_active48"]]


def fit_frame(df: pd.DataFrame, w=None, allow_gamma=True):
    return constant_fit(df["y"].to_numpy(), df["delta"].to_numpy(), w,
                        df["one_customer_fraction"].to_numpy(),
                        df["county"].to_numpy(), allow_gamma=allow_gamma)


# --------------------------------------------------------------------------
# local geometry
# --------------------------------------------------------------------------

def deterministic_pca(Z: np.ndarray, dims: int):
    """Economy SVD with a fixed sign convention, so the basis is reproducible."""
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt[:dims].copy()
    for i in range(V.shape[0]):
        j = int(np.argmax(np.abs(V[i])))
        if V[i, j] < 0:
            V[i] *= -1.0
    return V, S[:dims]


def knn_indices(P: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    """k nearest rows of P to q, ties broken deterministically by row index."""
    d = ((P - q) ** 2).sum(axis=1)
    n = d.shape[0]
    if n <= k:
        return np.lexsort((np.arange(n), d))
    part = np.argpartition(d, k)[: k + 1]
    kth = d[part].max()
    cand = np.flatnonzero(d <= kth)
    if cand.size == k:
        return cand
    sel = np.lexsort((cand, d[cand]))[:k]
    return cand[sel]


def local_metrics(allrows, fmap, channels, design, log):
    feats = D.feature_columns(channels)
    rows = design_rows(allrows, design)
    out = []
    for fold in sorted(fmap["folds"], key=int):
        held = set(fmap["folds"][fold])
        src = rows[~rows["event"].isin(held)]
        if src.empty:
            log.append(f"  local fold {fold} {design}: no source rows")
            continue
        capped = pd.concat(
            [D.deterministic_subsample(g, EVENT_ROW_CAP, salt=f"cap:{design}")
             for _, g in src.groupby("event", sort=True)],
            ignore_index=True)
        assert not set(capped["event"]) & held, "held-out event leaked into source"

        X = capped[feats].to_numpy(dtype=np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        Z = (X - mean) / std
        V, _ = deterministic_pca(Z, PCA_DIMS)
        P = Z @ V.T

        keys = list(zip(capped["event"], capped["county"], capped["physical_hour"]))
        h = np.array([L.stable_hash(*k, salt=f"query:{design}") for k in keys],
                     dtype=np.uint64)
        order = np.lexsort((np.arange(len(capped)), h))
        n_q = min(LOCAL_QUERIES, len(capped))
        queries = np.sort(order[:n_q])

        yv = capped["y"].to_numpy()
        dv = capped["delta"].to_numpy()
        ocf = capped["one_customer_fraction"].to_numpy()
        cty = capped["county"].to_numpy()
        evv = capped["event"].to_numpy()

        t0 = time.time()
        for qi in queries:
            idx = knn_indices(P, P[qi], LOCAL_K)
            assert idx.size == LOCAL_K or len(capped) < LOCAL_K
            assert not set(evv[idx]) & held, "held-out event entered a neighbourhood"
            f = constant_fit(yv[idx], dv[idx], None, ocf[idx], cty[idx],
                             allow_gamma=True)
            rec = f.as_dict()
            rec.update(fold=int(fold), design=design, query_event=str(evv[qi]),
                       query_county=str(cty[qi]),
                       query_physical_hour=int(capped["physical_hour"].iloc[qi]),
                       query_y=float(yv[qi]), k=LOCAL_K,
                       n_source_rows=int(len(capped)),
                       n_source_events=int(capped["event"].nunique()))
            out.append(rec)
        log.append(f"  local fold {fold} {design}: {n_q} cells, "
                   f"{len(capped)} source rows, {capped['event'].nunique()} events, "
                   f"{time.time()-t0:.1f}s")
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    code, data = Path(a.code).resolve(), Path(a.data).resolve()
    results = code / "analysis/conservation_preflight_20260904/results"
    results.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    t_start = time.time()

    prov, man, fmap = verify_integrity(code, data, log)
    panels = list(man["panels"])

    log.append("transition construction:")
    allrows, windows, channels = build_all(data, panels, log)
    log.append(f"total legal adjacent observed transitions: {len(allrows)}")

    # ---------------- event table
    ev_rows = []
    for e in panels:
        for design in D.DESIGNS:
            w = windows[e]
            sub = design_rows(allrows[allrows["event"] == e], design)
            base = dict(event=e, design=design, family=w["family"],
                        n_counties_panel=w["n_counties"],
                        observed_hour_share=w["observed_hour_share"],
                        active48_available=bool(w["available"]),
                        active48_reason=w["reason"],
                        active48_peak_hour=w["peak"],
                        active48_peak_footprint=w["peak_footprint"],
                        active48_t_start=w["t_start"], active48_t_end=w["t_end"])
            if design == "active48" and not w["available"]:
                ev_rows.append({**base, "n": 0, "status": "unavailable"})
                continue
            f = fit_frame(sub, None, allow_gamma=True)
            ev_rows.append({**base, **f.as_dict(), "status": "ok"})
    ev = pd.DataFrame(ev_rows)

    # ---------------- fold table
    fd_rows = []
    for fold in sorted(fmap["folds"], key=int):
        held = set(fmap["folds"][fold])
        for design in D.DESIGNS:
            rows = design_rows(allrows, design)
            src = rows[~rows["event"].isin(held)]
            if src.empty:
                continue
            n_ev = src["event"].nunique()
            for scheme in ("row_pooled", "equal_event"):
                if scheme == "row_pooled":
                    w, gamma_ok = None, True
                else:
                    cnt = src.groupby("event")["y"].transform("size").to_numpy()
                    w = 1.0 / (n_ev * cnt)
                    gamma_ok = False
                f = fit_frame(src, w, allow_gamma=gamma_ok)
                fd_rows.append(dict(fold=int(fold), design=design, weighting=scheme,
                                    n_source_events=int(n_ev),
                                    held_out_events=",".join(sorted(held)),
                                    gamma_defined=gamma_ok, **f.as_dict()))
    fd = pd.DataFrame(fd_rows)

    # ---------------- local table
    log.append("local k=200 geometry:")
    loc_rows = []
    for design in D.DESIGNS:
        loc_rows += local_metrics(allrows, fmap, channels, design, log)
    loc = pd.DataFrame(loc_rows)

    ev.to_csv(results / "EVENT_CONSERVATION_METRICS.csv", index=False)
    fd.to_csv(results / "FOLD_CONSERVATION_METRICS.csv", index=False)
    loc.to_csv(results / "LOCAL_GAMMA_METRICS.csv", index=False)

    prov.update(
        seed=SEED, python=sys.version.split()[0], platform=platform.platform(),
        packages=dict(numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__,
                      pyarrow=__import__("pyarrow").__version__),
        parameters=dict(cap_U=CAP_U, cap_R=CAP_R, closure_tol=CLOSURE_TOL,
                        K_balance=K_BALANCE, local_k=LOCAL_K,
                        local_queries=LOCAL_QUERIES, event_row_cap=EVENT_ROW_CAP,
                        pca_dims=PCA_DIMS, active_half=D.ACTIVE_HALF,
                        designs=list(D.DESIGNS), tie_channels=list(D.TIE_CHANNELS),
                        feature_columns=D.feature_columns(channels)),
        commands=[
            "git clone --branch open-audit-20260904 <repo> code",
            f"git clone <repo> data && git -C data checkout --detach {DATA_SHA}",
            "shasum -a 256 -c data/SHA256SUMS.txt",
            "python -m pytest -q test_preflight.py",
            "python run_preflight.py --code <code> --data <data>",
        ],
        n_transitions_total=int(len(allrows)),
        active48_unavailable=[e for e in panels if not windows[e]["available"]],
        windows={e: {k: (None if isinstance(v, float) and np.isnan(v) else v)
                     for k, v in windows[e].items()} for e in panels},
        runtime_seconds=round(time.time() - t_start, 1),
        log=log,
    )
    (results / "RUN_PROVENANCE.json").write_text(json.dumps(prov, indent=2, default=str))
    print("\n".join(log))
    print(f"\nwrote 3 CSVs + RUN_PROVENANCE.json to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
