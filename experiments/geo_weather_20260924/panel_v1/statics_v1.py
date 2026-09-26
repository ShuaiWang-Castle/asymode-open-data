"""County statics of the host on the designed panel (DATASET_DESIGN v1, amendment 2 B1): the table of
scripts/build_county_statics.py with SAIDI and SAIFI taken from EIA-861 2017 (the last year before the frame; the 2023
values are computed from 2023 interruptions, inside the frame), and rows for Connecticut's eight 2020 counties.

* SAIDI / SAIFI: the first SAIDI and SAIFI columns of Reliability_2017 (IEEE, with major event days, as the 2023
  build takes the first columns), utility mean, spread to the counties of the utility's 2017 service territory, county
  mean; the same construction as the 2023 build.
* A county without a 2017 SAIDI (its utilities did not report reliability in 2017) takes its state's median 2017
  county value, else the national median (flag saidi_imputed).
* Connecticut: log_cust from the EAGLE-I 2024 modelled customers; n_utilities and coop_share by the 2023 build's rule
  (service territory names match the 2020 counties); rucc and log_pop_density crosswalked from the 2022 planning
  regions with the population of the 3 km nodes (rucc rounded).
Output: data/interim/panel_v1/county_statics_v1.parquet"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from build_county_statics import read_861  # noqa: E402
from frame_common import INTERIM, OUT, RAW, counties  # noqa: E402

NORM = r"\b(COUNTY|PARISH|BOROUGH|CENSUS AREA|CITY AND|MUNICIPIO)\b"


def names() -> pd.DataFrame:
    frames = []
    for f in ("2020_Gaz_counties_national.txt", "2023_Gaz_counties_national.txt"):
        g = pd.read_csv(RAW / "census" / f, sep="\t", dtype=str, encoding="latin-1")
        g.columns = [c.strip() for c in g.columns]
        frames.append(g[["GEOID", "NAME", "USPS"]])
    g = pd.concat(frames).drop_duplicates("GEOID")

    def norm(s):
        return s.astype(str).str.upper().str.replace(NORM, "", regex=True).str.replace(r"[^A-Z]", "", regex=True)
    g["key"] = g.USPS.str.upper() + "|" + norm(g.NAME)
    return g, norm


def territory(year: int):
    p = RAW / "eia" / ("861_2017" if year == 2017 else "861") / f"Service_Territory_{year}.xlsx"
    t = read_861(p)
    u = [c for c in t.columns if "Utility Number" in c][0]
    s = [c for c in t.columns if c.strip() == "State"][0]
    c = [c for c in t.columns if "County" in c][0]
    g, norm = names()
    t["key"] = t[s].astype(str).str.upper() + "|" + norm(t[c])
    return t.merge(g[["GEOID", "key"]], on="key", how="inner").rename(columns={u: "u"})


def reliability_2017() -> pd.DataFrame:
    rel = read_861(RAW / "eia" / "861_2017" / "Reliability_2017.xlsx")
    ru = [c for c in rel.columns if "Utility Number" in c][0]
    sa = [c for c in rel.columns if c.upper().startswith("SAIDI")][0]
    sf = [c for c in rel.columns if c.upper().startswith("SAIFI")][0]
    r = rel[[ru, sa, sf]].copy()
    r.columns = ["u", "saidi", "saifi"]
    for c in ("saidi", "saifi"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.groupby("u").mean().reset_index()
    t = territory(2017)[["GEOID", "u"]].merge(r, on="u", how="left")
    return t.groupby("GEOID")[["saidi", "saifi"]].mean().rename_axis("fips")


def main() -> None:
    st = pd.read_parquet(INTERIM / "county_statics.parquet").set_index("fips")
    rel = reliability_2017()
    st["saidi_2023"], st["saifi_2023"] = st.saidi, st.saifi
    st["saidi"], st["saifi"] = rel.saidi.reindex(st.index), rel.saifi.reindex(st.index)
    ct = [f for f in counties().fips if f.startswith("09") and f not in st.index]
    # Connecticut 2020 counties: population-weighted crosswalk from the planning regions
    import geopandas as gpd
    n = pd.read_parquet(OUT / "nodes3k.parquet")
    n = n[n.fips.isin(ct)]
    pr = gpd.read_file(RAW / "census" / "cb_county" / "cb_2023_us_county_500k.shp")
    pr = pr[pr.STATEFP == "09"].to_crs(4326)
    pts = gpd.GeoDataFrame(n, geometry=gpd.points_from_xy(n.lon, n.lat), crs=4326)
    j = gpd.sjoin(pts, pr[["GEOID", "geometry"]], predicate="within")
    w = j.groupby(["fips", "GEOID"])["pop"].sum().rename("w").reset_index()
    w["w"] = w.w / w.groupby("fips").w.transform("sum")
    t23 = territory(2023)
    sales = read_861(RAW / "eia" / "861" / "Sales_Ult_Cust_2023.xlsx")
    su = [c for c in sales.columns if "Utility Number" in c][0]
    so = [c for c in sales.columns if "Ownership" in c]
    own = sales[[su] + so].drop_duplicates(su).set_index(su)[so[0]]
    t23["own"] = t23.u.map(own)
    agg = t23.groupby("GEOID").agg(n_utilities=("u", "nunique"),
                                   coop_share=("own", lambda s: float(s.astype(str).str.contains("Cooperative", case=False).mean())))
    cust = pd.read_parquet(INTERIM / "eaglei_county_customers_2024.parquet")["customers"]
    g = pd.read_csv(RAW / "census" / "2020_Gaz_counties_national.txt", sep="\t", dtype=str)
    g.columns = [c.strip() for c in g.columns]
    g = g.set_index("GEOID")
    rows = []
    for f in ct:
        ww = w[w.fips == f]
        src = st.reindex(ww.GEOID)
        row = dict(fips=f)
        for c in ("rucc", "log_pop_density"):
            v = float(np.nansum(src[c].to_numpy() * ww.w.to_numpy()))
            row[c] = float(np.rint(v)) if c == "rucc" else v
        row["n_utilities"] = agg.n_utilities.get(f, np.nan)
        row["coop_share"] = agg.coop_share.get(f, np.nan)
        row["saidi"], row["saifi"] = rel.saidi.get(f, np.nan), rel.saifi.get(f, np.nan)
        row["log_cust"] = float(np.log(max(cust.get(f, 1.0), 1.0)))
        row["log_area"] = float(np.log(float(g.at[f, "ALAND"]) / 1e6))
        row["lat"], row["lon"] = float(g.at[f, "INTPTLAT"]), float(g.at[f, "INTPTLONG"])
        rows.append(row)
    ctd = pd.DataFrame(rows).set_index("fips")
    out = pd.concat([st, ctd])
    out.index.name = "fips"
    # a county whose 2017 SAIDI is missing (utilities that did not report reliability in 2017) takes the median of its
    # state's 2017 county values, else the national median; flagged
    stc = out.index.str[:2]
    med = out.saidi.groupby(stc).median()
    out["saidi_imputed"] = out.saidi.isna()
    out["saidi"] = out.saidi.fillna(pd.Series(stc.map(med), index=out.index)).fillna(out.saidi.median())
    out.reset_index().to_parquet(OUT / "county_statics_v1.parquet", index=False)
    c = counties()
    print("rows", len(out), "CONUS 2020 counties covered", c.fips.isin(out.index).sum(), "of", len(c))
    print("SAIDI 2017 imputed among CONUS counties:", int(out.reindex(c.fips).saidi_imputed.sum()),
          "; still missing:", int(out.reindex(c.fips).saidi.isna().sum()),
          "; 2023 missing:", int(out.reindex(c.fips).saidi_2023.isna().sum()))
    print("corr(log1p SAIDI 2017, 2023):", round(np.log1p(out.saidi).corr(np.log1p(out.saidi_2023)), 3))
    print(ctd[["rucc", "log_pop_density", "n_utilities", "coop_share", "saidi", "log_cust"]].round(3).to_string())


if __name__ == "__main__":
    main()
