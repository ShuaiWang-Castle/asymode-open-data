"""County-level static covariates, assembled from primary public sources.

Used by the gate under H-C. Every column is traceable to a named public file;
nothing is carried in from elsewhere. Counties missing a column are reported and
the column is filled with the median rather than dropped, because dropping a
county here would silently change the panel's county set.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.features import load_adjacency   # noqa: E402

RAW, INTERIM = ROOT / "data/raw", ROOT / "data/interim"


def read_861(path: Path) -> pd.DataFrame:
    """Read an EIA-861 sheet, finding the header row rather than assuming it.

    The forms use a different number of banner rows per file and per year, and a
    hard-coded skiprows silently produces a frame of Unnamed columns instead of
    failing. Search for the row that names the utility key.
    """
    for skip in range(0, 6):
        d = pd.read_excel(path, skiprows=skip, nrows=3)
        cols = [str(c).strip() for c in d.columns]
        if any("Utility Number" in c for c in cols):
            out = pd.read_excel(path, skiprows=skip)
            out.columns = [str(c).strip() for c in out.columns]
            return out
    raise ValueError(f"no header row with 'Utility Number' found in {path.name}")


def main():
    # --- Census Gazetteer: land area and centroid ---
    gz = pd.read_csv(RAW / "census/2023_Gaz_counties_national.txt", sep="\t",
                     dtype={"GEOID": str}, encoding="latin-1")
    gz.columns = [c.strip() for c in gz.columns]
    g = gz[["GEOID", "ALAND", "INTPTLAT", "INTPTLONG"]].rename(
        columns={"GEOID": "fips", "ALAND": "aland_m2",
                 "INTPTLAT": "lat", "INTPTLONG": "lon"})
    g["fips"] = g["fips"].str.zfill(5)
    g["log_area"] = np.log(g["aland_m2"].clip(lower=1) / 1e6)

    # --- USDA ERS: rural-urban continuum and 2020 population ---
    ru = pd.read_csv(RAW / "census/rucc2023.csv", encoding="latin-1")
    ru["fips"] = ru["FIPS"].astype(int).astype(str).str.zfill(5)
    piv = ru.pivot_table(index="fips", columns="Attribute", values="Value",
                         aggfunc="first")
    st = pd.DataFrame({
        "rucc": pd.to_numeric(piv.get("RUCC_2023"), errors="coerce"),
        "pop2020": pd.to_numeric(piv.get("Population_2020"), errors="coerce"),
    }).reset_index()

    d = g.merge(st, on="fips", how="left")
    d["log_pop"] = np.log(d["pop2020"].clip(lower=1))
    d["log_pop_density"] = d["log_pop"] - d["log_area"]

    # --- adjacency degree ---
    adj = load_adjacency(RAW / "census/county_adjacency2023.txt")
    d["n_neighbours"] = d["fips"].map(lambda f: len(adj.get(f, [])))

    # --- EIA-861: utilities serving each county, and ownership mix ---
    terr = read_861(RAW / "eia/861/Service_Territory_2023.xlsx")
    ucol = [c for c in terr.columns if "Utility Number" in c][0]
    scol = [c for c in terr.columns if c.strip() == "State"][0]
    ccol = [c for c in terr.columns if "County" in c][0]

    sales = read_861(RAW / "eia/861/Sales_Ult_Cust_2023.xlsx")
    su = [c for c in sales.columns if "Utility Number" in c][0]
    so = [c for c in sales.columns if "Ownership" in c]
    own = (sales[[su] + so].drop_duplicates(su).set_index(su)[so[0]]
           if so else pd.Series(dtype=object))

    # County names in EIA are text; join through the Gazetteer's names.
    gname = gz[["GEOID", "NAME", "USPS"]].copy()
    gname["GEOID"] = gname["GEOID"].str.zfill(5)
    norm = lambda s: (s.astype(str).str.upper()
                      .str.replace(r"\b(COUNTY|PARISH|BOROUGH|CENSUS AREA|CITY AND|MUNICIPIO)\b",
                                   "", regex=True)
                      .str.replace(r"[^A-Z]", "", regex=True))
    gname["key"] = gname["USPS"].str.upper() + "|" + norm(gname["NAME"])
    terr["key"] = terr[scol].astype(str).str.upper() + "|" + norm(terr[ccol])
    t = terr.merge(gname[["GEOID", "key"]], on="key", how="inner")
    t["own"] = t[ucol].map(own)
    agg = t.groupby("GEOID").agg(
        n_utilities=(ucol, "nunique"),
        coop_share=("own", lambda s: float((s.astype(str)
                                            .str.contains("Cooperative", case=False)).mean())),
    ).reset_index().rename(columns={"GEOID": "fips"})
    d = d.merge(agg, on="fips", how="left")

    # --- EIA-861 Reliability: utility-level SAIDI/SAIFI spread to served counties ---
    rel = read_861(RAW / "eia/861/Reliability_2023.xlsx")
    ru2 = [c for c in rel.columns if "Utility Number" in c][0]
    sa = [c for c in rel.columns if c.upper().startswith("SAIDI")]
    sf = [c for c in rel.columns if c.upper().startswith("SAIFI")]
    if sa:
        r = rel[[ru2] + sa[:1] + sf[:1]].copy()
        r.columns = ["u", "saidi", "saifi"]
        for c in ("saidi", "saifi"):
            r[c] = pd.to_numeric(r[c], errors="coerce")
        r = r.groupby("u").mean().reset_index()
        tj = t[["GEOID", ucol]].rename(columns={ucol: "u"}).merge(r, on="u", how="left")
        rr = tj.groupby("GEOID")[["saidi", "saifi"]].mean().reset_index()
        rr = rr.rename(columns={"GEOID": "fips"})
        d = d.merge(rr, on="fips", how="left")
        # A utility serving many counties spreads one number across all of them.
        # This is an approximation and is only ever used as a static prior.

    # --- customer density, from the published county customer totals ---
    cust = INTERIM / "eaglei_county_customers_2024.parquet"
    if cust.exists():
        cc = pd.read_parquet(cust).reset_index()
        cc.columns = ["fips", "customers"]
        d = d.merge(cc, on="fips", how="left")
        d["log_cust"] = np.log(d["customers"].clip(lower=1))
        d["log_cust_density"] = d["log_cust"] - d["log_area"]

    cols = ["fips", "log_area", "rucc", "log_pop", "log_pop_density", "n_neighbours",
            "n_utilities", "coop_share", "saidi", "saifi", "log_cust",
            "log_cust_density", "lat", "lon"]
    cols = [c for c in cols if c in d.columns]
    out = d[cols].copy()

    print(f"{len(out):,} counties")
    print(f"{'column':<20}{'missing':>10}{'median':>14}")
    for c in cols[1:]:
        miss = int(out[c].isna().sum())
        print(f"{c:<20}{miss:>10}{np.nanmedian(pd.to_numeric(out[c], errors='coerce')):>14.3f}")
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(out[c].median())

    dst = INTERIM / "county_statics.parquet"
    out.to_parquet(dst, index=False)
    print(f"\nwritten: {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
