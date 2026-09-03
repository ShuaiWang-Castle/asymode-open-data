"""D-7: shape of the public target — zero inflation, extreme-county dominance,
wave structure, and what the cumulative path features can express. Zero training.

Registered as H-I in docs/PREREGISTRATION_external_priors.md. Every rule below is
fixed here before the script was run:

* scored support = the protocol's origins (stride 12, >= 24 h history) and
  horizons 1/6/24/48, observed cells only;
* a county-event is INTERRUPTED if its hourly peak y reaches THRESH = 0.01;
* a WAVE is a maximal run of hours with y >= THRESH; two runs separated by fewer
  than GAP_H = 6 hours below THRESH are one wave (so a brief dip does not create
  a spurious second wave);
* concentration is reported as the share of sum(y^2) held by the top 1 / 3 / 10
  counties of an event and as a Gini coefficient over counties;
* the feature check uses features.hazard_path exactly as the experiments do.

    python experiments/d7_target_shape.py --panels configs/panel_manifest_g3-all-26.json
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from asymode import panels as panelset                                    # noqa: E402
from asymode.evalproto import to_hourly                                   # noqa: E402
from asymode.features import hazard_path                                  # noqa: E402
import exp05_real_dynamics as exp05                                       # noqa: E402
from exp06_by_family import family_of_day                                 # noqa: E402

INTERIM = ROOT / "data/interim"
THRESH, GAP_H = 0.01, 6


def gini(v):
    v = np.sort(np.asarray(v, dtype=float))
    if v.sum() <= 0:
        return float("nan")
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


def waves(y, obs):
    """Wave spans [start, end] of one county-event's hourly series."""
    hot = (y >= THRESH) & obs
    if not hot.any():
        return []
    idx = np.where(hot)[0]
    spans = []
    s = p = idx[0]
    for i in idx[1:]:
        if i - p - 1 >= GAP_H:
            spans.append((s, p)); s = i
        p = i
    spans.append((s, p))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default=None)
    ap.add_argument("--horizon", type=int, default=48); ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--out", default="results/d7_target_shape.json")
    a = ap.parse_args()
    t0 = time.time(); source = panelset.source_version(ROOT)
    want, panel_digest = panelset.resolve(INTERIM, a.panels)
    fam = family_of_day()

    # ---------- 1. scored support: zero inflation by horizon ----------
    y0, X, yt, m, fips, panel, origin = exp05.load_pooled(a.horizon, a.stride, panels=want)
    zero = {}
    for h in (1, 6, 24, 48):
        v = yt[:, h - 1][m[:, h - 1]]
        zero[h] = dict(n=int(v.size), exact_zero=float((v == 0).mean()),
                       le_1e4=float((v <= 1e-4).mean()), le_1e3=float((v <= 1e-3).mean()),
                       mean=float(v.mean()), p99=float(np.quantile(v, 0.99)), max=float(v.max()))
    # squared energy of the target in the two halves of the forecast window
    half = {}
    for lo, hi, name in ((0, 24, "h+1..24"), (24, 48, "h+25..48")):
        sl = yt[:, lo:hi]; ms = m[:, lo:hi]
        half[name] = float((sl[ms] ** 2).sum())
    tot = sum(half.values())
    half = {k: dict(sq_energy=v, share=v / tot) for k, v in half.items()}

    # ---------- 2..4. per-panel: concentration, daily profile, waves ----------
    per_panel, wave_rows, feat_rows = [], [], []
    for day in sorted(want):
        pz = np.load(INTERIM / f"panel_{day}.npz", allow_pickle=True)
        dz = np.load(INTERIM / f"drivers_{day}.npz", allow_pickle=True)
        yh, oh = to_hourly(pz["y"], pz["observed"])
        yh = np.nan_to_num(yh)
        f_ = pz["fips"]; T = yh.shape[1]
        ts = np.array([str(t) for t in pz["ts"][::4]])[:T]
        Xh = dz["X"][:, :T, :]; chan = [str(c) for c in dz["channels"]]
        hp, hp_names = hazard_path(Xh, chan)
        gust = Xh[..., chan.index("gust")]
        # concentration of squared target energy across counties
        e = (yh ** 2 * oh).sum(1)
        order = np.argsort(e)[::-1]; tot_e = e.sum()
        sh = lambda k: float(e[order[:k]].sum() / tot_e) if tot_e > 0 else float("nan")
        # exact Lorenz curve of squared energy over counties, on a fixed 41-point grid
        cum = np.cumsum(e[order]) / tot_e if tot_e > 0 else np.zeros(len(e))
        frac = (np.arange(len(e)) + 1) / len(e)
        lg = np.linspace(0.0, 1.0, 41)
        lorenz = np.interp(lg, np.concatenate([[0.0], frac]), np.concatenate([[0.0], cum])).tolist()
        # daily squared energy relative to the storm day (panel starts 2 days before)
        days = np.array([int(t[8:10]) for t in ts])
        day_idx = (np.arange(T) // 24) - 2
        prof = {int(d): float((yh[:, day_idx == d] ** 2 * oh[:, day_idx == d]).sum()) for d in np.unique(day_idx)}
        s = sum(prof.values()) or 1.0
        # waves per county
        n_int = n_multi = 0; ratios = []
        for i in range(len(f_)):
            sp = waves(yh[i], oh[i])
            if not sp or yh[i].max() < THRESH:
                continue
            n_int += 1
            peaks = [float(yh[i, s0:s1 + 1].max()) for s0, s1 in sp]
            if len(sp) >= 2:
                n_multi += 1
                ratios.append(peaks[1] / peaks[0] if peaks[0] > 0 else np.nan)
                # feature coordinate at the onset of wave 2
                t2 = sp[1][0]
                own_max = float(gust[i, sp[1][0]:sp[1][1] + 1].max())
                feat_rows.append(dict(panel=day, family=fam.get(day, "convective"),
                                      running_max_at_w2=float(hp[i, t2, hp_names.index("path_gust_max")]),
                                      own_gust_max_w2=own_max,
                                      hours_since_peak_at_w2=float(hp[i, t2, hp_names.index("path_hours_since_peak")]),
                                      cum_gust_at_w2=float(hp[i, t2, hp_names.index("path_gust_sum")]),
                                      hours_between_waves=int(sp[1][0] - sp[0][1]),
                                      peak1=peaks[0], peak2=peaks[1]))
            wave_rows.append(dict(panel=day, family=fam.get(day, "convective"), n_waves=len(sp),
                                  peaks=peaks, first_share=float(peaks[0] / max(peaks)) if peaks else np.nan))
        per_panel.append(dict(panel=day, family=fam.get(day, "convective"), n_counties=int(len(f_)),
                              n_interrupted=n_int, n_multiwave=n_multi,
                              multiwave_share=float(n_multi / n_int) if n_int else float("nan"),
                              median_wave2_over_wave1=float(np.nanmedian(ratios)) if ratios else float("nan"),
                              top1_share=sh(1), top3_share=sh(3), top10_share=sh(10), gini=gini(e),
                              sq_energy=float(tot_e), lorenz_grid=lg.tolist(), lorenz=lorenz,
                              daily_share={str(k): v / s for k, v in sorted(prof.items())}))
        print(f"  {day} {fam.get(day,'convective'):<11} interrupted {n_int:>4} · multiwave {n_multi:>4} "
              f"({per_panel[-1]['multiwave_share']*100 if n_int else 0:.0f}%) · top3 {sh(3)*100:.0f}% · gini {gini(e):.2f}", flush=True)

    # ---------- 5. concentration of a fitted arm's squared error (legacy exports, descriptive) ----------
    err_conc = {}
    for arm in ("susceptible", "trees_matched"):
        p = ROOT / f"results/oof_{arm}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        hs = z["horizons"].tolist(); out = {}
        for hi, h in enumerate(hs):
            msk = z["mask"][:, hi].astype(bool)
            se = ((z["pred"][:, :, hi].mean(0) - z["y"][:, hi]) ** 2)[msk]
            cf = z["fips"][msk]
            u, inv = np.unique(cf, return_inverse=True)
            per = np.bincount(inv, weights=se)
            o = np.argsort(per)[::-1]; tt = per.sum()
            out[h] = dict(n_counties=int(len(u)), top1=float(per[o[:1]].sum() / tt),
                          top3=float(per[o[:3]].sum() / tt), top10=float(per[o[:10]].sum() / tt),
                          gini=gini(per))
        err_conc[arm] = out

    cfg = dict(vars(a)); cfg.update(panel_digest=panel_digest, panels=sorted(want), source=source,
                                    thresh=THRESH, gap_h=GAP_H, wall_time_s=round(time.time() - t0, 1),
                                    note="descriptive; zero training; error concentration uses the legacy county-split exports")
    res = dict(config=cfg, zero_by_horizon=zero, forecast_window_halves=half, per_panel=per_panel,
               feature_coordinate_at_wave2=feat_rows, waves=wave_rows, error_concentration=err_conc)
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))

    print("\n=== scored support, zero inflation ===")
    for h, v in zero.items():
        print(f"  h+{h:<3} n={v['n']:>7,}  y=0 {v['exact_zero']*100:5.1f}%  y<=1e-4 {v['le_1e4']*100:5.1f}%  "
              f"y<=1e-3 {v['le_1e3']*100:5.1f}%  mean {v['mean']:.5f}  p99 {v['p99']:.4f}  max {v['max']:.3f}")
    print(f"\n=== forecast-window squared energy: {json.dumps({k: round(v['share'], 3) for k, v in half.items()})}")
    mw = [p for p in per_panel if np.isfinite(p["multiwave_share"])]
    print(f"\n=== waves: multiwave share median {np.median([p['multiwave_share'] for p in mw])*100:.0f}% "
          f"(range {min(p['multiwave_share'] for p in mw)*100:.0f}-{max(p['multiwave_share'] for p in mw)*100:.0f}%) · "
          f"wave2/wave1 peak median {np.nanmedian([p['median_wave2_over_wave1'] for p in mw]):.3f}")
    print(f"=== concentration: top-3 county share of squared energy, median over panels "
          f"{np.median([p['top3_share'] for p in per_panel])*100:.0f}%; gini median {np.median([p['gini'] for p in per_panel]):.2f}")
    if feat_rows:
        r = feat_rows
        pin = np.mean([x["running_max_at_w2"] > x["own_gust_max_w2"] + 1e-6 for x in r])
        print(f"=== at second-wave onset ({len(r)} county-events): running gust max already above this wave's own max "
              f"in {pin*100:.0f}% of cases; median hours_since_peak {np.median([x['hours_since_peak_at_w2'] for x in r]):.0f} h; "
              f"median gap between waves {np.median([x['hours_between_waves'] for x in r]):.0f} h")
    for arm, v in err_conc.items():
        print(f"=== squared-error concentration, {arm}: " + " · ".join(
            f"h+{h} top3 {d['top3']*100:.0f}% top10 {d['top10']*100:.0f}% gini {d['gini']:.2f}" for h, d in v.items()))
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
