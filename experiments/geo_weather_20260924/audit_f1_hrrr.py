"""EXPLORATORY, not registered: F1 parts (c) and (d) for the HRRR contrasts C6 = hrrr - pop (default),
C7 = hrrr - hrrr_coarse (resolution) and C8 = hrrr_coarse - pop (source and definition), --contrasts C6,C7,C8.

Reuses audit_f1.py (frozen at 239fd7d for the registered W2 test) without modifying it: same unit-level alignment
test (part correlation; cluster-robust score t with a multiplier max-T over the 160 features; synthetic-field
calibration and the registered null rule), same hour-level statistic and same power regressions. Only the contrast
differs.

C6 compares the population-weighted (county, HRRR 3-km cell) nodes, with HRRR weather and HRRR-terrain lapse rates,
against the population-weighted ERA5 cells. It therefore mixes resolution with source and definition differences
(HRRR t-1 f01 forecasts against reanalysis; instantaneous surface gust against ERA5's hourly maximum). An alignment
says the HRRR-based hazard carries information the host misses; it does not by itself say that the information is
sub-grid structure.

--single "C8:<feature>" with --power-target runs ONE pre-registered test through the frozen audit_f1.single_h1a
(eligibility from the design, then power written to disk, then the test only if power >= 0.8), for contrasts that
the frozen script does not define (C6-C8). Missing hours: a unit-hour enters the window means only if it is observed
and both variants are finite there; the same mask is used for the residual and for the contrast. Units without any
such hour, units flagged False in an optional `unit_ok` array of the higher variant's file, and events listed in
--exclude-events are left out; all exclusions are reported. The frozen function writes its power record as
H1a_power.json; the wrapper adds H2a.json and H2a.md next to it.

Run from the repository root, one process, e.g. (wind panel, default base):
    .venv/bin/python experiments/geo_weather_20260924/audit_f1_hrrr.py --out experiments/geo_weather_20260924/results/F1_hrrr
and for W1:
    .venv/bin/python experiments/geo_weather_20260924/audit_f1_hrrr.py --features data/interim/geo_weather/features_w1.npz \
        --splits experiments/geo_weather_20260924/splits_w1.json --base "runs/geo_weather_20260924/w1_base/fold{fold:02d}/outer.npz" \
        --eih-prefix eih_w1_ --out experiments/geo_weather_20260924/results/F1_w1_hrrr
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_f1 as A  # noqa: E402  (sets the thread and round defaults)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXTRA = {"C6": ("C6 hrrr-pop (km-scale, exploratory)", "hrrr", "pop", None),
         "C7": ("C7 hrrr-hrrr_coarse (resolution, exploratory)", "hrrr", "hrrr_coarse", None),
         "C8": ("C8 hrrr_coarse-pop (source and definition, exploratory)", "hrrr_coarse", "pop", None)}
BANNER = ("**EXPLORATORY — not registered.** Contrasts {c}, run with the frozen `audit_f1.py` functions through "
          "`audit_f1_hrrr.py`; each contrast is its own max-T family over the features (no correction across contrasts). "
          "C6 = hrrr - pop mixes resolution with source and definition differences (HRRR t-1 f01 against reanalysis; "
          "instantaneous against hourly-maximum gust); C7 = hrrr - hrrr_coarse isolates resolution and C8 = "
          "hrrr_coarse - pop isolates source and definition (hrrr_coarse = HRRR averaged onto the ERA5 cells, then the "
          "pop construction).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--base", default=None, help="per-fold outer.npz pattern with {fold} (default: open_gcrk e3r2 W+Cin)")
    ap.add_argument("--design", default="main")
    ap.add_argument("--eih-dir", type=Path, default=A.EIH)
    ap.add_argument("--eih-prefix", default="eih_")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--quick", action="store_true", help="smoke settings, not for reading")
    ap.add_argument("--contrasts", default="C6", help="comma list of C6, C7, C8")
    ap.add_argument("--single", default=None, help='one pre-registered test "C<k>:<feature>" (k in 6, 7, 8)')
    ap.add_argument("--min-eff-events", type=float, default=None)
    ap.add_argument("--require-event-flip", action="store_true")
    ap.add_argument("--power-target", type=float, default=None)
    ap.add_argument("--exclude-events", default="", help="comma list of events left out (single mode)")
    args = ap.parse_args()
    if args.single:
        if args.power_target is None:
            raise SystemExit("--single needs --power-target")
        args.contrasts = args.single.split(":", 1)[0]
    keys = [k.strip() for k in args.contrasts.split(",") if k.strip()]
    if not keys or any(k not in EXTRA for k in keys):
        raise SystemExit(f"--contrasts must be a comma list of {list(EXTRA)}")
    cons = [EXTRA[k] for k in keys]
    tag = "_".join(keys)
    reg = json.loads(json.dumps(A.REG))
    if args.quick:
        reg.update(A.QUICK)
    reg.update(min_eff_events=args.min_eff_events if args.single else None,
               require_event_flip=bool(args.require_event_flip) if args.single else False)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    C = A.C
    feat_path = args.features if args.features is not None else C.FEATURES
    feat_path = feat_path if feat_path.is_absolute() else A.ROOT / feat_path
    sp_path = args.splits if args.splits is not None else C.SPLITS_FILE
    sp_path = sp_path if sp_path.is_absolute() else A.ROOT / sp_path
    sp = json.loads(sp_path.read_text())
    F = A.load_base(feat_path)
    n = len(F["y"])
    if args.base is None:
        b = A.REG["base"]
        P, u, r = (C.collect(b["design"], b["arm"], b["seed"], n, key=k) for k in ("P", "u", "r"))
        base_desc = f"open_gcrk {C.ROUND} W+Cin, main design, seed 0"
    else:
        P, u, r = (np.full((n, A.T), np.nan) for _ in range(3))
        for f, spec in sp[args.design].items():
            z = np.load(A.ROOT / args.base.format(fold=int(f)), allow_pickle=False)
            if not np.array_equal(np.sort(z["idx"]), np.sort(np.array(spec["outer"]))):
                raise SystemExit(f"base fold {f}: idx differs from the outer units of the splits")
            P[z["idx"]] = z["P"]
            for key, arr in (("u", u), ("r", r)):
                if key in z.files:
                    arr[z["idx"]] = z[key]
        base_desc = args.base
        if not (np.isfinite(u).all() and np.isfinite(r).all()):
            u = r = None
    if P is None or not np.isfinite(P).all():
        raise SystemExit("base predictions incomplete")
    m = F["m"].astype(bool)
    A.log(f"panel {feat_path.name}; base {base_desc}")
    if args.single:
        return run_single(args, reg, rng, F, P, m, base_desc, feat_path, sp_path, t0)
    need = []
    for c in cons:
        for v in (c[1], c[2]):
            if v not in need:
                need.append(v)
    paths = {v: args.eih_dir / f"{args.eih_prefix}{v}.npz" for v in need}
    miss = [p.name for p in paths.values() if not p.exists()]
    if miss:
        raise SystemExit(f"missing input files: {miss}")
    S, names, bad = {}, None, {}
    for v, pth in paths.items():
        phi, nm = A.load_phi(pth, F)
        if names is None:
            names = nm
        elif nm != names:
            raise SystemExit(f"{pth.name}: feature names differ from {paths[need[0]].name}")
        wm, mx, mn, nb = A.summaries(phi, m)
        S[v] = (wm, mx, mn)
        if nb:
            bad[v] = nb
        del phi
        gc.collect()
    parts, info_c = [], None
    for c in cons:
        a_, i_ = A.audit_c(F, P, m, S, names, reg, rng, contrasts=[c])
        parts.append(a_)
        if info_c is None:
            info_c = i_
        else:
            info_c["families"].update(i_["families"])
    Acsv = pd.concat(parts, ignore_index=True)
    info_c["killed"] = bool(not Acsv["pass"].any())
    Hd = Dd = None
    if u is not None:
        H = A.Hourly(F, P, u, r, m, reg, rng)
        sg = reg["sign"]
        hrows, drows = [], []
        for cname, vh, vl, _ in cons:
            hi, na = A.load_phi(paths[vh], F)
            lo, nb_ = A.load_phi(paths[vl], F)
            feats, pcs, S_es, S_ev = [], [], [], []
            for q0 in range(0, len(na), 16):
                js = list(range(q0, min(q0 + 16, len(na))))
                Ac = np.asarray(hi[:, :, js], np.float32)
                Bc = np.asarray(lo[:, :, js], np.float32)
                for k, jj in enumerate(js):
                    a_ = np.nan_to_num(Ac[:, :, k].astype(np.float64))
                    b_ = np.nan_to_num(Bc[:, :, k].astype(np.float64))
                    A._hour_one(H, cname, na[jj], a_, b_, feats, pcs, S_es, S_ev, drows)
                del Ac, Bc
            if feats:
                S1, S2 = np.stack(S_es, 1), np.stack(S_ev, 1)
                t1, t2 = A.score_t(S1), A.score_t(S2)
                _, p1, _ = A.mult_maxT(S1, t1, sg, reg["n_mult"], rng)
                d2 = np.sqrt((S2 ** 2).sum(0))
                mx2 = (sg * (H.flips @ S2) / np.where(d2 > 0, d2, 1.0)).max(1)
                p2 = (1 + (mx2[None, :] >= sg * t2[:, None]).sum(1)) / (len(mx2) + 1)
                for k, (nm, geff) in enumerate(feats):
                    hrows.append(dict(contrast=cname, feature=nm, eff_clusters_es=geff, part_corr=pcs[k],
                                      t_event_state=t1[k], p_maxT=p1[k], t_event=t2[k], p_maxT_event_flip=p2[k]))
            del hi, lo
            gc.collect()
        Hd = pd.DataFrame(hrows)
        if len(Hd):
            Hd["p_final"] = Hd["p_maxT"]
            Hd["p_final_event_flip"] = Hd["p_maxT_event_flip"]
        Dd = pd.DataFrame(drows)
    args.out.mkdir(parents=True, exist_ok=True)
    Acsv.to_csv(args.out / f"alignment_{tag}.csv", index=False)
    if Hd is not None:
        Hd.to_csv(args.out / f"alignment_hour_{tag}.csv", index=False)
        Dd.to_csv(args.out / f"mde_{tag}.csv", index=False)
    meta = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), panel=feat_path.name, splits=sp_path.name, base=base_desc,
                eih_prefix=args.eih_prefix, variants=list(paths), n_units=int(n), n_features=len(names), features=names,
                nonfinite=bad, quick=bool(args.quick), seed=args.seed, exploratory=True, contrasts=keys,
                inputs={p.name: A.sha256(p) for p in paths.values()}, audit_f1_sha256=A.sha256(Path(A.__file__)),
                git=subprocess.run(["git", "-C", str(A.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                   text=True).stdout.strip())
    md = args.out / f"F1_{tag}.md"
    A.write_md(md, meta, Acsv, info_c, Hd, Dd, None, None, dict(d3_reproduction={}, killed=None), A.REG)
    text = md.read_text()
    lines = [ln for ln in text.splitlines() if not ln.startswith("**(d) verdict")]
    lines.insert(1, "")
    lines.insert(2, BANNER.format(c=", ".join(keys)))
    md.write_text("\n".join(lines) + "\n")

    def recs(d):
        return None if d is None else json.loads(d.to_json(orient="records"))
    (args.out / f"F1_{tag}.json").write_text(json.dumps(dict(
        meta=meta, exploratory=BANNER.format(c=", ".join(keys)), run_settings={k: reg[k] for k in ("n_mult", "n_syn")},
        c=dict(info_c, table=recs(Acsv)), c_hour=recs(Hd), d=recs(Dd), runtime_s=round(time.time() - t0, 1)),
        indent=1, default=float) + "\n")
    A.log(f"wrote {args.out}")


def run_single(args, reg, rng, F, P, m, base_desc, feat_path, sp_path, t0):
    key, feat = args.single.split(":", 1)
    if key not in EXTRA:
        raise SystemExit(f"--single: contrast must be one of {list(EXTRA)}")
    con = EXTRA[key]
    _, vh, vl, _ = con
    paths = {v: args.eih_dir / f"{args.eih_prefix}{v}.npz" for v in (vh, vl)}
    miss = [p.name for p in paths.values() if not p.exists()]
    if miss:
        raise SystemExit(f"missing input files: {miss}")
    phi_h, nh = A.load_phi(paths[vh], F)
    phi_l, nl = A.load_phi(paths[vl], F)
    if nh != nl:
        raise SystemExit(f"{paths[vh].name} and {paths[vl].name}: feature names differ")
    if feat not in nh:
        raise SystemExit(f"--single: feature {feat} not in the inputs")
    j = nh.index(feat)
    x_h = np.asarray(phi_h[:, :, j], np.float64)
    x_l = np.asarray(phi_l[:, :, j], np.float64)
    del phi_h, phi_l
    ev = F["event"].astype(str)
    n = len(ev)
    ok_file = np.ones(n, bool)
    zh = np.load(paths[vh], allow_pickle=False)
    if "unit_ok" in zh.files:
        ok_file = zh["unit_ok"].astype(bool)
        if ok_file.shape != (n,):
            raise SystemExit("unit_ok has the wrong shape")
    excl = {e.strip() for e in args.exclude_events.split(",") if e.strip()}
    unknown = sorted(excl - set(ev))
    if unknown:
        raise SystemExit(f"--exclude-events: unknown events {unknown}")
    ok_unit = ok_file & ~np.isin(ev, sorted(excl))
    fin = np.isfinite(x_h) & np.isfinite(x_l)
    m2 = m & fin & ok_unit[:, None]
    dropped_no_hours = int(((m & ok_unit[:, None]).sum(1) > 0).sum() - (m2.sum(1) > 0).sum())
    miss_share = {e: float(1 - (m2[ev == e].sum() / max((m & ok_unit[:, None])[ev == e].sum(), 1)))
                  for e in sorted(set(ev[ok_unit]))}
    S = {vh: (A.window_mean(np.nan_to_num(x_h), m2)[:, None], None, None),
         vl: (A.window_mean(np.nan_to_num(x_l), m2)[:, None], None, None)}
    args.out.mkdir(parents=True, exist_ok=True)
    res = A.single_h1a(F, P, m2, S, [feat], feat, con, reg, rng, args.power_target, args.out)
    exclusions = dict(unit_ok_false=int((~ok_file).sum()), excluded_events=sorted(excl),
                      excluded_event_units=int(np.isin(ev, sorted(excl)).sum()),
                      units_without_valid_hours=dropped_no_hours,
                      missing_share_of_observed_hours_by_event=miss_share,
                      rule="a unit-hour enters only if observed and both variants are finite; the same mask for the "
                           "residual and the contrast")
    meta = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), panel=feat_path.name, splits=sp_path.name, base=base_desc,
                eih_prefix=args.eih_prefix, single=args.single, power_target=args.power_target,
                min_eff_events=args.min_eff_events, require_event_flip=bool(args.require_event_flip), seed=args.seed,
                quick=bool(args.quick), inputs={pth.name: A.sha256(pth) for pth in paths.values()},
                audit_f1_sha256=A.sha256(Path(A.__file__)), wrapper_sha256=A.sha256(Path(__file__)),
                git=subprocess.run(["git", "-C", str(A.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                   text=True).stdout.strip())
    out = dict(res, exclusions=exclusions, meta=meta, runtime_s=round(time.time() - t0, 1))
    (args.out / "H2a.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    L = [f"# Registered single test {res['test']} (through audit_f1_hrrr.py)", "",
         f"Generated {meta['generated']} (git {meta['git']}; audit_f1.py sha {meta['audit_f1_sha256'][:12]}…, wrapper sha "
         f"{meta['wrapper_sha256'][:12]}…). Panel `{meta['panel']}`, base {meta['base']}. Order: eligibility from the design, "
         "then power (written by the frozen function to `H1a_power.json` before the test), then the test.", "",
         f"* Units {res['n_units']}, events {res['n_events']}, event x state clusters {res['n_event_state']}; nonzero contrast in "
         f"{res['nonzero_event_state']} blocks and {res['nonzero_events']} events.",
         f"* Exclusions: unit_ok false {exclusions['unit_ok_false']}, excluded events {exclusions['excluded_events'] or 'none'} "
         f"({exclusions['excluded_event_units']} units), units without a valid hour {exclusions['units_without_valid_hours']}; "
         "largest missing share of observed hours in a kept event "
         f"{max(miss_share.values()) if miss_share else 0:.3f}.",
         "* Effective clusters: event x state {event_state:.1f}, county {county:.1f}, event {event:.1f}; ".format(**res["eff_clusters"])
         + f"required ≥ {res['min_eff_clusters']} / ≥ {res['min_eff_clusters']} / ≥ {res['min_eff_events']}: "
         f"**{'eligible' if res['eligible'] else 'not eligible'}**."]
    if "power" in res:
        L += ["* Single-test false-positive rate on synthetic fields: " + ", ".join(f"{k} {v:.3f}" for k, v in res["fpr_single_test"].items())
              + f"; null used: {res['null_used']}.",
              f"* Power at part correlation {res['power_target_part_corr']}: " + ", ".join(f"{k} {v:.3f}" for k, v in res["power"].items())
              + f"; minimum {res['power_min']:.3f} (**{'informative' if res['informative'] else 'uninformative'}**)."]
    if "p_used" in res:
        L += [f"* Test: part corr {res['part_corr']:+.4f}, t (event x state) {res['t_event_state']:+.2f}, p multiplier "
              f"{res['p_multiplier']:.4f}, p synthetic {res['p_synthetic']:.4f}, p used {res['p_used']:.4f}; t (event) "
              f"{res['t_event']:+.2f}, exact event-flip p {res['p_event_flip']:.4f}."]
    L += ["", f"**Verdict: {res['verdict']}.**", ""]
    (args.out / "H2a.md").write_text("\n".join(L) + "\n")
    A.log(f"single {args.single}: {res['verdict']}")


if __name__ == "__main__":
    main()
