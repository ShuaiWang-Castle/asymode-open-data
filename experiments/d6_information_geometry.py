"""D-6: local information geometry of the panels (C1) and where the state
dispersion comes from (C2). Zero training. Interpretation fixed in
docs/THEORY_PLAN.md (D-6) and docs/PREREGISTRATION_long_horizon.md.

For every forecast-origin cell (x = drivers at the first forecast step, y = state
at the origin) the conditional design Gram Q(x) = E[phi(Y) phi(Y)^T | X = x] is
estimated over a driver-space neighbourhood of x, and reported as

    mu_y, var_y, lambda_min(Q), lambda_max(Q), condition number, N_local,
    N_local * var_y, A = E[(1-y)^2], B = E[y^2], variance_ratio_R_over_U = A / B,

together with the law-of-total-variance split of var_y by county (C2) and the
cross-sectional variance among rows of the same storm hour. All normalisation
and the PCA are fitted on the TRAINING rows of each fold of the pinned outer
split; every cell is evaluated once per fold in which it is a training row.

Neighbourhoods (predeclared sensitivity grid): k-nearest neighbours in the
training-fitted 5-component PCA space for k in {50, 200, 800}, and an 8 x 8
quantile grid on the first two components. Neighbourhoods with effective size
< 30 or var_y < 1e-8 are flagged and excluded from aggregates.

    python experiments/d6_information_geometry.py --panels configs/panel_manifest_g2-convective-11.json
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from asymode import panels as panelset, information as I, splits, schema   # noqa: E402
import exp05_real_dynamics as exp05                                        # noqa: E402
from exp06_by_family import family_of_day                                  # noqa: E402

INTERIM = ROOT / "data/interim"
K_GRID = (50, 200, 800)
MIN_N, MIN_VAR = 30, 1e-8


def local_stats(y, counties, hour_ids):
    """Statistics of one neighbourhood: Gram identities, LOTV by county, same-hour spread."""
    s = I.gram_stats(y)
    n = len(y)
    # law of total variance by county
    _, inv = np.unique(counties, return_inverse=True)
    cnt = np.bincount(inv); means = np.bincount(inv, weights=y) / cnt
    within = float(np.sum((y - means[inv]) ** 2) / n)
    between = float(np.sum(cnt * (means - y.mean()) ** 2) / n)
    # cross-sectional spread: variance among rows sharing a storm hour, averaged over hours with >= 2 rows
    _, hinv = np.unique(hour_ids, return_inverse=True)
    hcnt = np.bincount(hinv)
    xs = [np.var(y[hinv == h]) for h in np.where(hcnt >= 2)[0]]
    return dict(mu=s["mu"], var=s["var"], lam_min=s["lam_min"], lam_max=s["lam_max"], cond=s["cond"],
                n=n, n_var=n * s["var"], A=s["A"], B=s["B"],
                var_ratio_R_over_U=(s["A"] / s["B"]) if s["B"] > 0 else float("inf"),
                within_county=within, between_county=between,
                cross_share=(between / s["var"]) if s["var"] > 0 else float("nan"),
                n_counties=int(len(cnt)), same_hour_var=float(np.mean(xs)) if xs else float("nan"),
                n_same_hour_groups=int(len(xs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default=None)
    ap.add_argument("--horizon", type=int, default=48); ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--outer-split-seed", type=int, default=0)
    ap.add_argument("--split-unit", choices=["event", "county"], default="event")
    ap.add_argument("--max-cells", type=int, default=4000, help="cells evaluated per fold (deterministic subsample)")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default="results/d6_information_geometry.json")
    a = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    t_launch = time.time(); source = panelset.source_version(ROOT)
    want, panel_digest = panelset.resolve(INTERIM, a.panels)
    y0, X, yt, m, fips, panel, origin, t0h = exp05.load_pooled(a.horizon, a.stride, panels=want, with_time=True)
    x = X[:, 0, :].astype(np.float64)                     # drivers at the first forecast step
    y = y0.astype(np.float64)
    fam = family_of_day(); famv = np.array([fam.get(p, "convective") for p in panel])
    hour_id = np.array([f"{p}|{o}" for p, o in zip(panel, origin)])
    assign, split_map, split_digest, _ = exp05.outer_assignment(fips, panel, a.split_unit, a.k, a.outer_split_seed)
    print(f"{len(y):,} origin cells · {len(set(panel))} events [{panel_digest}] · split {a.split_unit} {split_digest}")

    cells = []          # one record per (fold, neighbourhood rule, cell)
    rng = np.random.default_rng(a.outer_split_seed)
    for f in range(a.k):
        tr = np.where(assign != f)[0]
        mu_x, sd_x = x[tr].mean(0), x[tr].std(0) + 1e-9
        pca = PCA(n_components=5, random_state=0).fit((x[tr] - mu_x) / sd_x)
        z = pca.transform((x[tr] - mu_x) / sd_x)
        sub = rng.choice(len(tr), size=min(a.max_cells, len(tr)), replace=False)
        for k in K_GRID:
            nn = NearestNeighbors(n_neighbors=min(k, len(tr))).fit(z)
            _, idx = nn.kneighbors(z[sub])
            for i, row in zip(sub, idx):
                r = tr[row]
                st = local_stats(y[r], fips[r], hour_id[r])
                st.update(rule=f"knn{k}", fold=f, cell=int(tr[i]), event=str(panel[tr[i]]), family=str(famv[tr[i]]),
                          flagged=bool(st["n"] < MIN_N or st["var"] < MIN_VAR))
                cells.append(st)
        # quantile grid on PC1 x PC2
        q1, q2 = np.quantile(z[:, 0], np.linspace(0, 1, 9)), np.quantile(z[:, 1], np.linspace(0, 1, 9))
        b1 = np.clip(np.searchsorted(q1, z[:, 0], side="right") - 1, 0, 7)
        b2 = np.clip(np.searchsorted(q2, z[:, 1], side="right") - 1, 0, 7)
        bid = b1 * 8 + b2
        for b in np.unique(bid):
            r = tr[bid == b]
            st = local_stats(y[r], fips[r], hour_id[r])
            st.update(rule="grid8x8", fold=f, cell=int(b), event=None, family=None,
                      flagged=bool(st["n"] < MIN_N or st["var"] < MIN_VAR))
            # a bin's event/family mix is recorded as its dominant member
            ev, cnt = np.unique(panel[r], return_counts=True); st["event"] = str(ev[np.argmax(cnt)])
            fa, cnt = np.unique(famv[r], return_counts=True); st["family"] = str(fa[np.argmax(cnt)])
            cells.append(st)
        print(f"  fold {f}: {len(sub)} cells x {len(K_GRID)} knn rules + {len(np.unique(bid))} grid bins", flush=True)

    # ---- aggregation with individual events shown ----
    def agg(rows, key):
        out = {}
        for g in sorted({r[key] for r in rows}):
            v = [r for r in rows if r[key] == g and not r["flagged"]]
            if not v: out[g] = None; continue
            def med(name): return float(np.median([r[name] for r in v]))
            out[g] = dict(n_cells=len(v), median_var=med("var"), median_lam_min=med("lam_min"),
                          median_n_var=med("n_var"), median_var_ratio_R_over_U=med("var_ratio_R_over_U"),
                          median_cross_share=float(np.nanmedian([r["cross_share"] for r in v])),
                          median_same_hour_var=float(np.nanmedian([r["same_hour_var"] for r in v])),
                          frac_flagged=float(np.mean([r["flagged"] for r in rows if r[key] == g])))
        return out
    summary = {}
    for rule in [f"knn{k}" for k in K_GRID] + ["grid8x8"]:
        rows = [r for r in cells if r["rule"] == rule]
        summary[rule] = {"by_family": agg(rows, "family"), "by_event": agg(rows, "event"),
                         "overall": agg([dict(r, all="all") for r in rows], "all")["all"]}
    # event-cluster bootstrap of the cross-county share (knn200)
    rows = [r for r in cells if r["rule"] == "knn200" and not r["flagged"] and np.isfinite(r["cross_share"])]
    ev = np.array([r["event"] for r in rows]); cs = np.array([r["cross_share"] for r in rows])
    events = np.unique(ev); ev_mean = np.array([cs[ev == e].mean() for e in events])
    bs = [ev_mean[rng.integers(0, len(events), len(events))].mean() for _ in range(a.B)]
    summary["cross_share_knn200"] = {"per_event": {str(e): float(v) for e, v in zip(events, ev_mean)},
                                     "mean_over_events": float(ev_mean.mean()),
                                     "event_cluster_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    # ordering stability of families across rules (by median lam_min and by median n_var)
    order = {}
    for rule in summary:
        if rule.startswith("knn") or rule == "grid8x8":
            bf = {k: v for k, v in summary[rule]["by_family"].items() if v}
            order[rule] = {"by_lam_min": sorted(bf, key=lambda g: -bf[g]["median_lam_min"]),
                           "by_n_var": sorted(bf, key=lambda g: -bf[g]["median_n_var"])}
    summary["family_order"] = order

    cfg = dict(vars(a)); cfg.update(panel_digest=panel_digest, panels=sorted(set(panel.tolist())),
                                    channel_digest=panelset.channel_digest(panelset.channel_names(INTERIM)),
                                    source=source, outer_split_digest=split_digest, k_grid=list(K_GRID),
                                    min_n=MIN_N, min_var=MIN_VAR, wall_time_s=round(time.time() - t_launch, 1),
                                    note="diagnostic; zero training; cells = forecast origins; PCA/standardisation fitted on training rows per fold")
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    # estimability of the county decomposition: rows per county inside neighbourhoods
    facts = {}
    for rule in sorted({c["rule"] for c in cells}):
        v = [c for c in cells if c["rule"] == rule and not c["flagged"]]
        rpc = [c["n"] / c["n_counties"] for c in v if c["n_counties"]]
        facts[rule] = {"n_cells": len(v), "median_rows_per_county": float(np.median(rpc)) if rpc else None,
                       "p90_rows_per_county": float(np.quantile(rpc, 0.9)) if rpc else None,
                       "frac_cells_all_distinct_counties": float(np.mean([c["n_counties"] == c["n"] for c in v])) if v else None}
    summary["rows_per_county_in_neighbourhood"] = facts
    out.write_text(json.dumps({"config": cfg, "summary": summary}, indent=1))
    import gzip
    side = out.with_name(out.stem.replace("d6_information_geometry", "d6_cells") + ".json.gz")
    with gzip.open(side, "wt") as fh:      # per-cell records: large, gitignored sidecar
        json.dump([{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in cells], fh)
    print(f"\nwritten: {a.out}")
    for rule in [f"knn{k}" for k in K_GRID] + ["grid8x8"]:
        print(f"\n[{rule}] family: n_cells | median var | median lam_min | median N*var | median Var(R)/Var(U) | cross-county share")
        for g, v in summary[rule]["by_family"].items():
            if v: print(f"  {g:<11} {v['n_cells']:>6} | {v['median_var']:.2e} | {v['median_lam_min']:.2e} | {v['median_n_var']:.3f} | {v['median_var_ratio_R_over_U']:.1f} | {v['median_cross_share']:.2f}")
    c = summary["cross_share_knn200"]
    print(f"\ncross-county share of var(y|x) (knn200): mean over events {c['mean_over_events']:.2f}, event-cluster 95% CI [{c['event_cluster_ci95'][0]:.2f}, {c['event_cluster_ci95'][1]:.2f}]")
    print("family order:", json.dumps(order))


if __name__ == "__main__":
    main()
