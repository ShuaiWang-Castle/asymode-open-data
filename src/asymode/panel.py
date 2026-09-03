"""Turn sparse outage records into a dense, honestly-labelled county panel.

The publisher drops every row whose outage count is zero. About three quarters of
the (county, timestamp) grid is therefore absent, and an absent cell means one of
two very different things: the county genuinely had no customers out, or the
county was not being observed. Collapsing that distinction would fabricate the
exact quantity this project is about -- a county sitting at y = 0 before a storm
starts is the onset case, and a county whose scraper was down looks identical.

The rule used here, stated so it can be argued with:

  1. A timestamp is a *collection run* if any county anywhere has a record at it.
     The programme polls all covered utilities on the same 15-minute cadence, so
     a timestamp with national records is one where collection happened.
  2. A county is *in service* on a day if it has at least one record within a
     window of +/- `service_days` around that day. A county that never reports
     across a fortnight is treated as unobserved, not as quiet.
  3. A cell that is missing, at a collection-run timestamp, in a county that is in
     service, is a true zero.
  4. Everything else is left missing and carried as an explicit mask. It is never
     imputed and never silently dropped.

The mask travels with the panel so every downstream loss, metric and split can
exclude unobserved cells rather than scoring the model against a guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def collection_timestamps(df: pd.DataFrame, min_counties: int = 5) -> pd.DatetimeIndex:
    """Timestamps at which collection demonstrably ran."""
    n = df.groupby("ts")["fips"].size()
    return pd.DatetimeIndex(n[n >= min_counties].index)


def in_service(df: pd.DataFrame, service_days: int = 7) -> pd.DataFrame:
    """(fips, day) -> whether the county reported anything nearby in time."""
    d = df.assign(day=df["ts"].dt.floor("D"))
    seen = d.groupby(["fips", "day"]).size().rename("n").reset_index()
    wide = seen.pivot(index="day", columns="fips", values="n").notna()
    full = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"),
                        fill_value=False)
    w = 2 * service_days + 1
    near = full.rolling(w, center=True, min_periods=1).max().astype(bool)
    return near


def build_panel(df: pd.DataFrame, t0, t1, fips: list[str] | None = None,
                freq: str = "15min", service_days: int = 7,
                min_counties: int = 5) -> dict:
    """Dense (county x time) arrays of outage counts and an observation mask.

    Returns `fips`, `ts`, `counts` (float, NaN where unobserved) and `observed`
    (bool). No normalisation happens here -- the denominator is a separate,
    separately-sourced decision, see docs/DATA_CARD.md.
    """
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    win = df[(df["ts"] >= t0) & (df["ts"] <= t1)]
    if fips is None:
        fips = sorted(win["fips"].unique())
    fips = list(fips)
    ts = pd.date_range(t0, t1, freq=freq)

    ran = collection_timestamps(df[(df["ts"] >= t0 - pd.Timedelta(days=1)) &
                                   (df["ts"] <= t1 + pd.Timedelta(days=1))],
                                min_counties=min_counties)
    ts_ran = pd.Index(ts).isin(ran)

    svc = in_service(df[(df["ts"] >= t0 - pd.Timedelta(days=service_days + 1)) &
                        (df["ts"] <= t1 + pd.Timedelta(days=service_days + 1))],
                     service_days=service_days)
    svc = svc.reindex(columns=fips, fill_value=False)
    day_of = pd.Index(ts).floor("D")
    svc_ct = svc.reindex(index=day_of, fill_value=False).to_numpy().T   # (C, T)

    fi = {f: i for i, f in enumerate(fips)}
    ti = {t: i for i, t in enumerate(ts)}
    counts = np.zeros((len(fips), len(ts)), dtype=np.float32)
    w = win[win["fips"].isin(fi) & win["ts"].isin(ti)]
    counts[w["fips"].map(fi).to_numpy(), w["ts"].map(ti).to_numpy()] = \
        w["customers_out"].astype("float32").to_numpy()

    observed = svc_ct & ts_ran[None, :]
    counts[~observed] = np.nan
    return {"fips": fips, "ts": ts, "counts": counts, "observed": observed,
            "ts_ran": ts_ran, "in_service": svc_ct}


def attach_denominator(panel: dict, denom: pd.Series, name: str) -> dict:
    """Divide counts by a per-county customer total to get the target fraction.

    `denom` is indexed by FIPS. Counties without one are dropped, loudly: a
    missing denominator is missing information, not a licence to guess.
    """
    d = pd.Series(denom).reindex(panel["fips"]).astype("float64")
    keep = d.notna().to_numpy() & (d.to_numpy() > 0)
    if not keep.all():
        dropped = [f for f, k in zip(panel["fips"], keep) if not k]
        print(f"  attach_denominator({name}): dropping {len(dropped)} counties "
              f"with no denominator, e.g. {dropped[:5]}")
    out = {k: v for k, v in panel.items()}
    out["fips"] = [f for f, k in zip(panel["fips"], keep) if k]
    for k in ("counts", "observed", "in_service"):
        out[k] = panel[k][keep]
    out["denominator"] = d[keep].to_numpy()
    out["denominator_source"] = name
    y = out["counts"] / out["denominator"][:, None]
    out["y"] = np.clip(y, 0.0, 1.0)
    out["y_over_one"] = float(np.nansum(y > 1.0))
    return out
