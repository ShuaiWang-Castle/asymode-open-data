"""Build a county panel around a storm day and measure how often onset happens.

The onset question is the whole reason this project exists, and until now it has
only been answered on synthetic trajectories. This script answers it on public
observations: of the counties that a storm eventually interrupts, how many were
sitting at exactly zero when it arrived?

The denominator is provisional -- see docs/DATA_CARD.md. It is built from the
publisher's own state customer totals apportioned by county population, and it is
labelled `provisional_state_pop_share` everywhere it appears so no result can be
quoted without that caveat travelling with it.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.panel import build_panel, attach_denominator   # noqa: E402

INTERIM = ROOT / "data" / "interim"
STATE_ABBR = {  # coverage_history uses abbreviations; the outage files spell states out
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR",
}


def provisional_denominator(year: int) -> tuple[pd.Series, pd.DataFrame]:
    """State customer totals apportioned to counties by 2020 population share."""
    cov = pd.read_parquet(INTERIM / "eaglei_coverage_history.parquet")
    cov["yr"] = pd.to_datetime(cov["year"], format="%m/%d/%y").dt.year
    cov = cov[cov["yr"] == year][["state", "total_customers", "max_pct_covered"]]

    rucc = pd.read_csv(ROOT / "data/raw/census/rucc2023.csv", encoding="latin-1")
    pop = (rucc[rucc["Attribute"] == "Population_2020"]
           .assign(fips=lambda d: d["FIPS"].astype(int).astype(str).str.zfill(5),
                   pop=lambda d: pd.to_numeric(d["Value"], errors="coerce"))
           [["fips", "State", "pop"]].rename(columns={"State": "state"}))
    pop["state_pop"] = pop.groupby("state")["pop"].transform("sum")
    m = pop.merge(cov, on="state", how="inner")
    m["customers"] = m["total_customers"] * m["pop"] / m["state_pop"]
    return m.set_index("fips")["customers"], m.set_index("fips")[["state", "max_pct_covered"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-day", default="2022-06-17")
    ap.add_argument("--before", type=int, default=2, help="days of lead-in")
    ap.add_argument("--after", type=int, default=5, help="days of follow-through")
    ap.add_argument("--min-coverage", type=float, default=0.70)
    ap.add_argument("--onset-threshold", type=float, default=0.01,
                    help="fraction out that counts as 'interrupted'")
    ap.add_argument("--event-days", nargs="+", default=None,
                    help="audit several event days; overrides --event-day")
    ap.add_argument("--out", default="results/panel_onset_audit.json")
    a = ap.parse_args()

    days = [pd.Timestamp(d) for d in (a.event_days or [a.event_day])]
    rows = []
    for day in days:
        r = audit_day(day, a)
        if r is not None:
            rows.append(r)
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "days": rows}, indent=2))

    print("\n=== across event days ===")
    print(f"{'day':<12}{'counties':>9}{'interrupted':>13}{'median y_pre = 0':>19}{'< 1e-4':>10}{'< 1e-3':>10}")
    for r in rows:
        print(f"{r['event_day']:<12}{r['n_counties']:>9}{r['n_interrupted']:>13}"
              f"{r['frac_typ_zero']*100:>17.1f}%{r['frac_lt_1e4']*100:>9.1f}%{r['frac_lt_1e3']*100:>9.1f}%")
    f = lambda k: np.mean([r[k] for r in rows])
    print(f"{'MEAN':<12}{'':>9}{'':>13}{f('frac_typ_zero')*100:>17.1f}%"
          f"{f('frac_lt_1e4')*100:>9.1f}%{f('frac_lt_1e3')*100:>9.1f}%")
    print(f"\nwritten: {a.out}")


def audit_day(day, a) -> dict:
    """Audit one storm day. The reported statistic is the *typical* pre-storm
    state (median over the lead-in), not its maximum. A county that is dark for
    two days apart from one ten-minute blip is at zero for modelling purposes;
    scoring it by the maximum would call that county 'already out' and hide the
    onset case entirely."""
    year = day.year
    t0, t1 = day - pd.Timedelta(days=a.before), day + pd.Timedelta(days=a.after)
    df = pd.read_parquet(INTERIM / f"eaglei_outages_{year}.parquet")
    ced = pd.read_parquet(INTERIM / "county_event_days.parquet")
    hit = sorted(ced[ced["day"] == day]["fips"].unique())
    denom, meta = provisional_denominator(year)
    ok_cov = meta[meta["max_pct_covered"] >= a.min_coverage].index
    fips = [f for f in hit if f in set(denom.index) and f in set(ok_cov)]
    if len(fips) < 20:
        print(f"{day.date()}: only {len(fips)} usable counties, skipping")
        return None

    p = build_panel(df, t0, t1, fips=fips)
    p = attach_denominator(p, denom, "provisional_state_pop_share")
    y, obs = p["y"], p["observed"]
    lead = pd.Index(p["ts"]) < day
    with np.errstate(all="ignore"):
        ever = np.nanmax(np.where(obs, y, np.nan), axis=1)
        pre = np.where(obs[:, lead], y[:, lead], np.nan)
        pre_med = np.nanmedian(pre, axis=1)
    interrupted = np.nan_to_num(ever, nan=-1) >= a.onset_threshold
    v = pre_med[interrupted]
    v = v[np.isfinite(v)]
    n = max(len(v), 1)
    r = {
        "event_day": str(day.date()), "window": [str(t0), str(t1)],
        "n_counties": len(p["fips"]), "n_steps": len(p["ts"]),
        "observed_frac": float(obs.mean()),
        "denominator": p["denominator_source"],
        "n_interrupted": int(interrupted.sum()),
        "frac_typ_zero": float((v <= 0).sum() / n),
        "frac_lt_1e5": float((v <= 1e-5).sum() / n),
        "frac_lt_1e4": float((v <= 1e-4).sum() / n),
        "frac_lt_1e3": float((v <= 1e-3).sum() / n),
        "suppression_p10": float(1.0 / np.percentile(v[v > 0], 90)) if (v > 0).any() else None,
        "suppression_p25": float(1.0 / np.percentile(v[v > 0], 75)) if (v > 0).any() else None,
        "peak_y_median": float(np.nanmedian(ever[interrupted])),
    }
    print(f"{day.date()}: {r['n_counties']} counties, {r['n_interrupted']} interrupted, "
          f"{100*r['frac_typ_zero']:.1f}% typically at zero, obs {100*r['observed_frac']:.1f}%")
    np.savez_compressed(INTERIM / f"panel_{day.date()}.npz",
                        y=y, observed=obs, denominator=p["denominator"],
                        fips=np.array(p["fips"]), ts=np.array(p["ts"].astype(str)))
    return r


if __name__ == "__main__":
    main()
