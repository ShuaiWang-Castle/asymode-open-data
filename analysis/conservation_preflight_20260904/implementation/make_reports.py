"""Generate PREFLIGHT_SUMMARY.json and PREFLIGHT_REPORT.md from the three CSVs.

Every table is produced programmatically from the full tables. No event, family,
window or quantile is hand-picked, and no scientific promotion/rejection label is
assigned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def f(x, p=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{p}g}"


def quant(s: pd.Series) -> dict:
    s = s.dropna()
    return {f"p{int(q*100)}": (float(s.quantile(q)) if len(s) else None) for q in QS}


def md_table(df: pd.DataFrame, floatfmt=4) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f(v, floatfmt))
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(code: Path, pytest_out: Path) -> int:
    res = code / "analysis/conservation_preflight_20260904/results"
    ev = pd.read_csv(res / "EVENT_CONSERVATION_METRICS.csv")
    fd = pd.read_csv(res / "FOLD_CONSERVATION_METRICS.csv")
    loc = pd.read_csv(res / "LOCAL_GAMMA_METRICS.csv")
    prov = json.loads((res / "RUN_PROVENANCE.json").read_text())

    fam = dict(zip(ev["event"], ev["family"]))
    loc["family"] = loc["query_event"].map(fam)
    ok = ev[ev["status"] == "ok"].copy()
    ok["band_width"] = ok["balanced_hi"] - ok["balanced_lo"]
    ok["band_in_customer_units"] = ok["band_width"] / ok["median_one_customer_fraction"]

    S: dict = {}
    S["provenance"] = {k: prov[k] for k in
                       ["code_head", "code_branch", "data_head", "audited_base",
                        "manifest_digest", "fold_digest", "checksum_files_checked",
                        "checksum_files_ok", "checksum_output_sha256",
                        "code_status_porcelain",
                        "base_to_head_src_experiments_configs_diff"]}
    S["parameters"] = prov["parameters"]
    S["packages"] = prov["packages"]
    S["seed"] = prov["seed"]
    S["n_transitions_total"] = prov["n_transitions_total"]
    S["active48_unavailable"] = prov["active48_unavailable"]
    S["pytest"] = {"output_tail": pytest_out.read_text().strip().splitlines()[-1]
                   if pytest_out.exists() else "not found"}

    # ---- exact algebra achieved on real data
    S["algebra_checks"] = {
        "max_abs_identity_residual_unconstrained": float(max(
            ev["identity_residual_unc"].abs().max(),
            fd["identity_residual_unc"].abs().max(),
            loc["identity_residual_unc"].abs().max())),
        "max_abs_AB_minus_C2_minus_v": float(max(
            ev["abc_residual"].abs().max(), fd["abc_residual"].abs().max(),
            loc["abc_residual"].abs().max())),
        "boundary_rows_local": int((loc["boundary_status"] != "interior").sum()),
        "boundary_rows_local_breaking_box_identity":
            int((loc.loc[loc["boundary_status"] != "interior",
                         "identity_residual_box"].abs() > 1e-6).sum()),
    }

    # ---- closure / interiority
    S["closure"] = {}
    for d, g in ev[ev["status"] == "ok"].groupby("design"):
        S["closure"][f"event_{d}"] = dict(
            n_rows=int(len(g)), closure_pass=int(g["closure_pass"].sum()),
            interior_unc=int(g["interior_unc"].sum()),
            median_closure_ratio=float(g["closure_ratio"].median()),
            median_mu=float(g["mu"].median()),
            median_abs_mean_delta=float(g["mean_delta"].abs().median()))
    for (d, w), g in fd.groupby(["design", "weighting"]):
        S["closure"][f"fold_{d}_{w}"] = dict(
            n_rows=int(len(g)), closure_pass=int(g["closure_pass"].sum()),
            interior_unc=int(g["interior_unc"].sum()),
            median_closure_ratio=float(g["closure_ratio"].median()),
            median_mu=float(g["mu"].median()))
    for d, g in loc.groupby("design"):
        S["closure"][f"local_{d}"] = dict(
            n_rows=int(len(g)), closure_pass_frac=float(g["closure_pass"].mean()),
            interior_frac=float(g["interior_unc"].mean()),
            boundary_frac=float((g["boundary_status"] != "interior").mean()),
            median_closure_ratio=float(g["closure_ratio"].replace(
                [np.inf, -np.inf], np.nan).median()))

    # ---- fold rate ratio vs closed-window prediction
    rr = []
    for _, r in fd[fd["weighting"] == "row_pooled"].iterrows():
        pred = r["mu"] / (1 - r["mu"])
        got = r["U_unc"] / r["R_unc"] if r["R_unc"] != 0 else np.nan
        rr.append(dict(fold=int(r["fold"]), design=r["design"],
                       U_over_R=float(got), mu_over_1_minus_mu=float(pred),
                       ratio=float(got / pred) if pred else None,
                       closure_pass=bool(r["closure_pass"])))
    S["fold_rate_ratio_vs_closed_window"] = rr

    # ---- local Gamma
    S["local_gamma"] = {}
    for d, g in loc.groupby("design"):
        S["local_gamma"][d] = dict(
            n=int(len(g)), quantiles=quant(g["Gamma_plugin"]),
            frac_above_1=float((g["Gamma_plugin"] > 1).mean()),
            frac_above_4=float((g["Gamma_plugin"] > 4).mean()),
            frac_exactly_zero=float((g["Gamma_plugin"] == 0).mean()),
            median=float(g["Gamma_plugin"].median()),
            frac_c_common_zero=float((g["c_common"] == 0).mean()),
            median_sigma=float(g["sigma"].median()),
            near_closure_populated_frac=float(g["Gamma_near_closure"].notna().mean()))
    a, fl = S["local_gamma"]["active48"], S["local_gamma"]["full"]
    S["local_gamma"]["active_over_full"] = dict(
        median_ratio=(None if fl["median"] == 0 else a["median"] / fl["median"]),
        median_ratio_note=("undefined: both medians are exactly 0 because the "
                           "local box fit puts one rate at 0 in the majority of "
                           "cells" if fl["median"] == 0 else ""),
        frac_above_1_ratio=(a["frac_above_1"] / fl["frac_above_1"]
                            if fl["frac_above_1"] else None),
        quantile_ratios={k: (a["quantiles"][k] / fl["quantiles"][k]
                             if fl["quantiles"][k] else None)
                         for k in a["quantiles"]})

    # ---- event-level full vs active-48
    piv = ok.pivot_table(index="event", columns="design",
                         values=["mu", "Gamma_plugin", "n", "c_common", "sigma"])
    both = piv.dropna()
    S["event_full_vs_active48"] = dict(
        n_events_both=int(len(both)),
        median_mu_ratio=float((both[("mu", "active48")]
                               / both[("mu", "full")]).median()),
        median_gamma_ratio=float((both[("Gamma_plugin", "active48")]
                                  / both[("Gamma_plugin", "full")]).median()),
        median_n_ratio=float((both[("n", "active48")] / both[("n", "full")]).median()),
        median_gamma_full=float(both[("Gamma_plugin", "full")].median()),
        median_gamma_active48=float(both[("Gamma_plugin", "active48")].median()))

    # ---- treatment scales
    S["treatment_scale"] = {}
    for d, g in ok.groupby("design"):
        with np.errstate(invalid="ignore", divide="ignore"):
            cr = (g["c_common"] / g["R_box"].replace(0, np.nan))
        S["treatment_scale"][f"event_{d}"] = dict(
            median_c=float(g["c_common"].median()),
            median_c_over_R=float(cr.median()),
            median_rms_delivered=float(g["rms_delivered_treatment"].median()),
            median_sigma=float(g["sigma"].median()),
            median_rms_delivered_over_sigma=float(
                (g["rms_delivered_treatment"] / g["sigma"]).median()))
    for d, g in loc.groupby("design"):
        with np.errstate(invalid="ignore", divide="ignore"):
            cr = (g["c_common"] / g["R_box"].replace(0, np.nan))
        S["treatment_scale"][f"local_{d}"] = dict(
            median_c=float(g["c_common"].median()),
            median_c_over_R=float(cr.median()),
            median_rms_delivered=float(g["rms_delivered_treatment"].median()),
            median_sigma=float(g["sigma"].median()),
            frac_c_zero=float((g["c_common"] == 0).mean()),
            p90_rms_delivered_over_sigma=float(
                (g["rms_delivered_treatment"] / g["sigma"]).quantile(0.9)))

    # ---- resolution geometry
    S["balanced_flow_vs_resolution"] = {}
    for d, g in ok.groupby("design"):
        b = g["band_in_customer_units"].replace([np.inf, -np.inf], np.nan).dropna()
        S["balanced_flow_vs_resolution"][d] = dict(
            n_events=int(len(g)),
            median_band_width=float(g["band_width"].median()),
            median_one_customer_fraction=float(g["median_one_customer_fraction"].median()),
            band_in_customer_units=dict(min=float(b.min()), median=float(b.median()),
                                        max=float(b.max())),
            n_events_band_narrower_than_one_customer=int((b < 1).sum()),
            median_balanced_share=float(g["balanced_share"].median()))

    # ---- families
    S["families"] = {}
    for d, g in loc.groupby("design"):
        S["families"][f"local_{d}"] = {
            k: dict(n=int(len(v)), median=float(v["Gamma_plugin"].median()),
                    p90=float(v["Gamma_plugin"].quantile(0.9)),
                    frac_above_1=float((v["Gamma_plugin"] > 1).mean()),
                    frac_c_zero=float((v["c_common"] == 0).mean()))
            for k, v in g.groupby("family")}
    S["families"]["event_full"] = {
        k: dict(n_events=int(len(v)), median_mu=float(v["mu"].median()),
                median_gamma=float(v["Gamma_plugin"].median()),
                closure_pass_frac=float(v["closure_pass"].mean()))
        for k, v in ok[ok["design"] == "full"].groupby("family")}

    (res / "PREFLIGHT_SUMMARY.json").write_text(json.dumps(S, indent=2, default=str))

    # ================================================================ report
    L = []
    A = L.append
    A("# Conservation and design preflight — generated tables\n")
    A("Machine-generated from the three CSV tables. No event, family, window or "
      "quantile is hand-selected. This file states measurements only; it assigns "
      "no scientific promotion or rejection label.\n")
    A(f"- code HEAD `{prov['code_head']}` (`{prov['code_branch']}`)")
    A(f"- public-data HEAD `{prov['data_head']}`")
    A(f"- audited base `{prov['audited_base']}`")
    A(f"- manifest digest `{prov['manifest_digest']}`, fold digest `{prov['fold_digest']}`")
    A(f"- public files verified `{prov['checksum_files_ok']}/{prov['checksum_files_checked']}`")
    A(f"- legal adjacent observed transitions: **{prov['n_transitions_total']:,}**")
    A(f"- active-48 unavailable for: {prov['active48_unavailable'] or 'none'}\n")

    A("## 1. Exact algebra achieved on the real tables\n")
    A(f"- max |U(1-mu) - R mu - mean_delta| over event/fold/local unconstrained "
      f"fits: **{f(S['algebra_checks']['max_abs_identity_residual_unconstrained'],3)}**")
    A(f"- max |A*B - C^2 - v|: **{f(S['algebra_checks']['max_abs_AB_minus_C2_minus_v'],3)}**")
    A(f"- local rows on a rate boundary: "
      f"{S['algebra_checks']['boundary_rows_local']}, of which "
      f"{S['algebra_checks']['boundary_rows_local_breaking_box_identity']} have a "
      f"box-fit identity residual above 1e-6 — expected, since a constrained "
      f"optimum obeys KKT inequalities, not the unconstrained normal equations.\n")

    A("## 2. Event-level conservation, both fixed designs\n")
    cols = ["event", "family", "design", "n", "mu", "v", "mean_delta",
            "closure_ratio", "closure_pass", "interior_unc", "U_box", "R_box",
            "boundary_status", "sigma", "c_common", "Gamma_plugin",
            "Gamma_near_closure", "Gamma_cap"]
    A(md_table(ev[cols].sort_values(["event", "design"])))
    A("")
    A("Closure and interiority counts:\n")
    rows = [dict(scope=k, **v) for k, v in S["closure"].items()]
    A(md_table(pd.DataFrame(rows)))
    A("")

    A("## 3. Source-fold constant fits, both weightings\n")
    cols = ["fold", "design", "weighting", "n", "n_source_events", "mu", "v",
            "mean_delta", "closure_ratio", "closure_pass", "interior_unc",
            "U_unc", "R_unc", "boundary_status", "sigma", "c_common",
            "identity_residual_unc", "n_eff_kish", "Gamma_plugin", "gamma_defined"]
    A(md_table(fd[cols]))
    A("")
    A("Fitted rate ratio against the closed-window prediction `mu/(1-mu)`:\n")
    A(md_table(pd.DataFrame(S["fold_rate_ratio_vs_closed_window"])))
    A("")

    A("## 4. Local k=200 design geometry\n")
    rows = []
    for d in ("full", "active48"):
        g = S["local_gamma"][d]
        rows.append(dict(design=d, n=g["n"], **{k: v for k, v in g["quantiles"].items()},
                         frac_gt_1=g["frac_above_1"], frac_gt_4=g["frac_above_4"],
                         frac_eq_0=g["frac_exactly_zero"],
                         frac_c_zero=g["frac_c_common_zero"],
                         closure_pass=g["near_closure_populated_frac"]))
    A(md_table(pd.DataFrame(rows)))
    A("")
    r = S["local_gamma"]["active_over_full"]
    A(f"- active-48 / full median plug-in Gamma ratio: "
      f"**{'undefined' if r['median_ratio'] is None else f(r['median_ratio'])}**"
      + (f" — {r['median_ratio_note']}" if r["median_ratio_note"] else ""))
    A(f"- active-48 / full ratio of the fraction above one: "
      f"**{f(r['frac_above_1_ratio'])}**")
    A("- upper-quantile ratios: " + ", ".join(
        f"{k} {f(v)}" for k, v in r["quantile_ratios"].items() if v is not None))
    A("")
    A("Per fold and design:\n")
    t = (loc.groupby(["design", "fold"])
            .agg(n=("Gamma_plugin", "size"),
                 median=("Gamma_plugin", "median"),
                 p90=("Gamma_plugin", lambda s: s.quantile(0.9)),
                 frac_gt_1=("Gamma_plugin", lambda s: float((s > 1).mean())),
                 frac_c_zero=("c_common", lambda s: float((s == 0).mean())),
                 closure_pass=("closure_pass", "mean"),
                 interior=("interior_unc", "mean"),
                 n_source_rows=("n_source_rows", "max"))
            .reset_index())
    A(md_table(t))
    A("")

    A("## 5. Event-level full versus active-48\n")
    e = S["event_full_vs_active48"]
    A(f"- events with both designs available: {e['n_events_both']}")
    A(f"- median `mu` ratio active-48/full: **{f(e['median_mu_ratio'])}**")
    A(f"- median row-count ratio active-48/full: **{f(e['median_n_ratio'])}**")
    A(f"- median plug-in Gamma, full: {f(e['median_gamma_full'])}; "
      f"active-48: {f(e['median_gamma_active48'])}")
    A(f"- median plug-in Gamma ratio active-48/full: **{f(e['median_gamma_ratio'])}**\n")

    A("## 6. Common rate and delivered transition treatment\n")
    A(md_table(pd.DataFrame([dict(scope=k, **v)
                             for k, v in S["treatment_scale"].items()])))
    A("")

    A("## 7. Balanced-flow interval versus county reporting resolution\n")
    A(md_table(pd.DataFrame([dict(design=k, **{kk: vv for kk, vv in v.items()
                                               if kk != "band_in_customer_units"},
                                  **{f"band_cust_{kk}": vv for kk, vv
                                     in v["band_in_customer_units"].items()})
                             for k, v in S["balanced_flow_vs_resolution"].items()])))
    A("")
    A("Per event (K=2 band expressed in units of one reporting customer):\n")
    A(md_table(ok[["event", "family", "design", "balanced_lo", "balanced_hi",
                   "band_width", "median_one_customer_fraction",
                   "band_in_customer_units", "balanced_share"]]
               .sort_values(["design", "event"])))
    A("")

    A("## 8. Family-level summaries\n")
    A("Event level, full design:\n")
    A(md_table(pd.DataFrame([dict(family=k, **v)
                             for k, v in S["families"]["event_full"].items()])))
    A("")
    for d in ("full", "active48"):
        A(f"Local k=200, {d} design:\n")
        A(md_table(pd.DataFrame([dict(family=k, **v)
                                 for k, v in S["families"][f"local_{d}"].items()])))
        A("")

    (res / "PREFLIGHT_REPORT.md").write_text("\n".join(L) + "\n")
    print(f"wrote PREFLIGHT_SUMMARY.json and PREFLIGHT_REPORT.md to {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2])))
