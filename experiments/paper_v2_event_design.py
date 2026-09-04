"""Task 1 of the V2 pilot: freeze the data design before any model is imported.

Builds, for every panel in the 26-event development cohort:

  * an event design table with no model-result column;
  * the three outcome-blind event-centred origin anchors;
  * a five-fold event assignment balanced on exogenous descriptors only;
  * an audit of the origin rule, including every rounding and every duplicate drop.

Outcome-blind means what it says. Outage summaries are computed for the table --
they describe the task and the strata -- but the fold objective and the anchor
rule read only family, year, county footprint, observation coverage, and
exogenous weather/geography. No outage severity, residual, prior gain or target
peak enters either.

Run before importing the model:

    PYTHONPATH=src:experiments ./.venv/bin/python experiments/paper_v2_event_design.py
"""
from __future__ import annotations

import hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from asymode import panels as panelset                      # noqa: E402
from asymode.evalproto import to_hourly                      # noqa: E402
from exp06_by_family import family_of_day                    # noqa: E402

INTERIM = ROOT / "data/interim"
OUT = ROOT / "analysis/gpt_rescue_20260904/cc_v2"
MANIFEST = ROOT / "configs/panel_manifest_g3-all-26.json"
PAST_H, FUT_H = 24, 24          # required past context and future target, hours
N_FOLDS = 5


def sha12(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()[:12]


def panel_frame(day: str):
    pz = np.load(INTERIM / f"panel_{day}.npz", allow_pickle=True)
    dz = np.load(INTERIM / f"drivers_{day}.npz", allow_pickle=True)
    yh, oh = to_hourly(pz["y"], pz["observed"])
    ts = pd.to_datetime([str(t) for t in dz["ts"]])
    n = min(yh.shape[1], dz["X"].shape[1], len(ts))
    return dict(day=day, y=np.nan_to_num(yh[:, :n]), obs=oh[:, :n],
                X=dz["X"][:, :n], ts=ts[:n], fips=np.array(pz["fips"], dtype=str),
                denom=np.asarray(pz["denominator"], dtype=float),
                channels=[str(c) for c in dz["channels"]])


def noaa_interval(day: str, fips: np.ndarray, ts: pd.DatetimeIndex, ev: pd.DataFrame):
    """First begin, last end and midpoint of the NOAA events inside this panel.

    Restricted to the panel's own counties and its own time window, so the anchors
    are a property of the event as this panel observes it. `county_event_days` is
    NOT used: it carries only a calendar day and cannot place an anchor on an hour.
    """
    lo, hi = ts[0], ts[-1]
    m = (ev["fips"].isin(set(fips.tolist()))
         & (ev["t_begin_utc"] >= lo) & (ev["t_begin_utc"] <= hi))
    sel = ev.loc[m]
    if sel.empty:
        return None
    t0 = sel["t_begin_utc"].min()
    t1 = sel["t_end_utc"].max()
    if pd.isna(t1) or t1 < t0:
        t1 = sel["t_begin_utc"].max()
    # Clip the interval to the panel window. Some NOAA rows -- floods, long winter
    # advisories -- carry an end time weeks after their begin, which would put the
    # "last event" anchor and the midpoint outside the panel entirely and make the
    # reported interval longer than the panel itself. The anchors are defined by
    # the event *as this panel observes it*, so the window is the bound.
    t0 = max(t0, lo)
    t1 = min(t1, hi)
    if t1 < t0:
        t1 = t0
    return t0, t1, t0 + (t1 - t0) / 2, int(len(sel))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    want, panel_digest = panelset.resolve(INTERIM, str(MANIFEST))
    fam = family_of_day()
    ev = pd.read_parquet(INTERIM / "storm_events_county.parquet")
    ev["t_begin_utc"] = pd.to_datetime(ev["t_begin_utc"])
    ev["t_end_utc"] = pd.to_datetime(ev["t_end_utc"])
    stat = pd.read_parquet(INTERIM / "county_statics.parquet").set_index("fips")

    rows, audit, anchors_all = [], [], {}
    for day in want:
        P = panel_frame(day)
        n_h = P["y"].shape[1]
        lo_i, hi_i = PAST_H, n_h - FUT_H - 1
        valid = list(range(lo_i, hi_i + 1))
        iv = noaa_interval(day, P["fips"], P["ts"], ev)
        if iv is None:
            audit.append(dict(event=day, status="NO NOAA EVENTS IN WINDOW"))
            continue
        t0, t1, tm, n_ev = iv
        raw = {"pre": t0 - pd.Timedelta(hours=6), "mid": tm,
               "post": t1 + pd.Timedelta(hours=6)}
        chosen, notes = [], []
        for name, t in raw.items():
            k = int(np.argmin(np.abs(P["ts"] - t)))          # nearest hourly grid point
            k_cl = int(np.clip(k, lo_i, hi_i))               # then into the legal range
            notes.append(dict(anchor=name, requested=str(t), nearest_index=k,
                              nearest_time=str(P["ts"][k]), clipped_to=k_cl,
                              clipped=bool(k_cl != k), legal_range=[lo_i, hi_i]))
            chosen.append(k_cl)
        # de-duplicate; a duplicate is dropped, never replaced by searching the curve
        uniq, seen, dropped = [], set(), []
        for k, nt in zip(chosen, notes):
            if k in seen:
                dropped.append(nt["anchor"])
            else:
                seen.add(k); uniq.append(k)
        anchors_all[day] = uniq
        audit.append(dict(event=day, status="ok", n_noaa_rows=n_ev,
                          noaa_begin=str(t0), noaa_end=str(t1), noaa_mid=str(tm),
                          panel_start=str(P["ts"][0]), panel_end=str(P["ts"][-1]),
                          legal_origin_range=[lo_i, hi_i], n_valid_origins=len(valid),
                          anchors=notes, kept_indices=uniq, dropped_duplicates=dropped))

        y0 = P["y"][:, uniq]                      # state at the retained origins
        o0 = P["obs"][:, uniq]
        v = y0[o0]
        fut = P["y"][:, [min(k + FUT_H, n_h - 1) for k in uniq]]
        onset = float(np.mean((y0 < 1e-4) & (fut > 0.02))) if y0.size else float("nan")
        gust = P["X"][:, :, P["channels"].index("gust")]
        prec = P["X"][:, :, P["channels"].index("precip")]
        st = stat.reindex(P["fips"])
        rows.append(dict(
            event=day, family=fam.get(day, "convective"), year=int(day[:4]),
            n_counties=int(len(P["fips"])),
            observation_coverage=float(P["obs"].mean()),
            noaa_begin=str(t0), noaa_end=str(t1),
            noaa_duration_h=float((t1 - t0).total_seconds() / 3600),
            n_valid_origins=len(valid), n_anchor_origins=len(uniq),
            zero_origin_share=float(np.mean(v == 0)) if v.size else float("nan"),
            near_zero_share=float(np.mean((v > 0) & (v <= 0.01))) if v.size else float("nan"),
            interior_share=float(np.mean(v > 0.01)) if v.size else float("nan"),
            future_onset_share=onset,
            median_outage=float(np.median(P["y"][P["obs"]])),
            p90_outage=float(np.quantile(P["y"][P["obs"]], 0.90)),
            gust_p90=float(np.quantile(gust, 0.90)), gust_max=float(gust.max()),
            precip_total_mean=float(prec.sum(1).mean()),
            log_cust_mean=float(np.nanmean(st["log_cust"].values)),
            log_pop_density_mean=float(np.nanmean(st["log_pop_density"].values)),
            lat_mean=float(np.nanmean(st["lat"].values)),
            lon_mean=float(np.nanmean(st["lon"].values)),
            denominator_median=float(np.median(P["denom"])),
            denominator_total=float(np.sum(P["denom"])),
        ))
        print(f"  {day}  {rows[-1]['family']:<11} counties {rows[-1]['n_counties']:>4} "
              f"origins {len(uniq)}  NOAA {t0.date()} {int(rows[-1]['noaa_duration_h']):>3}h", flush=True)

    df = pd.DataFrame(rows).sort_values("event").reset_index(drop=True)

    # ---- five folds, balanced greedily on EXOGENOUS descriptors only ----------
    EXO = ["n_counties", "observation_coverage", "gust_p90", "precip_total_mean",
           "log_cust_mean", "log_pop_density_mean", "lat_mean", "lon_mean",
           "denominator_total", "year"]
    Z = df[EXO].to_numpy(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    order = sorted(range(len(df)), key=lambda i: (df.family[i], -float(np.linalg.norm(Z[i]))))
    folds = {f: [] for f in range(N_FOLDS)}
    cent = {f: np.zeros(Z.shape[1]) for f in range(N_FOLDS)}
    for i in order:                                  # greedy: least-loaded, then closest centroid
        cand = sorted(folds, key=lambda f: (len(folds[f]),
                                            float(np.linalg.norm(Z[i] - (cent[f] / max(len(folds[f]), 1))))))
        f = cand[0]
        folds[f].append(df.event[i]); cent[f] = cent[f] + Z[i]
    fold_map = {"n_folds": N_FOLDS, "balanced_on": EXO,
                "explicitly_excluded": ["outage severity", "prior model gain",
                                        "residuals", "target peak", "zero share",
                                        "future onset share"],
                "folds": {str(f): sorted(v) for f, v in folds.items()},
                "panel_digest": panel_digest,
                "anchors": {k: [int(x) for x in v] for k, v in anchors_all.items()}}
    fold_map["digest"] = sha12(fold_map)

    df.to_csv(OUT / "event_design_table.csv", index=False)
    (OUT / "event_folds_v2.json").write_text(json.dumps(fold_map, indent=1))
    print(f"\nfold digest {fold_map['digest']}   panel digest {panel_digest}")
    for f, v in fold_map["folds"].items():
        fams = [df.set_index('event').family[e] for e in v]
        print(f"  fold {f}: {len(v)} events  {sorted(set(fams))}")
    json.dump(audit, open(OUT / "_origin_audit.json", "w"), indent=1, default=str)
    print(f"\nwrote event_design_table.csv, event_folds_v2.json, _origin_audit.json")


if __name__ == "__main__":
    main()
