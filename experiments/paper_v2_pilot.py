"""V2 implementation pilot: nine optimization jobs on three outcome-blind events.

Reads the frozen design written by `paper_v2_event_design.py` and never recomputes
it. For each pilot test event the source events are every event outside that test
event's fold; counties inside each source event are hashed 80/20 into fit and
validation groups, and checkpointing uses the equal-event mean validation objective
over all source events, so no single validation storm controls stopping.

Nine jobs: three events x (one two-flow start + two fixed one-flow branch starts).
The one-flow branch is chosen on source-event validation only; the test event never
sees a start selected for it.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from asymode import panels as panelset                                  # noqa: E402
from asymode.evalproto import to_hourly                                  # noqa: E402
from asymode.features import load_adjacency                              # noqa: E402
from asymode_paper import features as F                                  # noqa: E402
from asymode_paper.asymmetric_flows import (AsymmetricFlows, CAP_U_MAIN,  # noqa: E402
                                            CAP_U_BKG, CAP_R)
from asymode_paper.initialization import (fit_two_flow, fit_u_ray,        # noqa: E402
                                          fit_r_ray, modular_init)
from asymode_paper import trainer as TR                                   # noqa: E402

INTERIM = ROOT / "data/interim"
OUT = ROOT / "analysis/gpt_rescue_20260904/cc_v2"
HORIZON = 24
LOOKBACK = 24


def county_group(fips: str, event: str) -> str:
    h = hashlib.sha256(f"paper_v2:{event}:{fips}".encode()).hexdigest()
    return "val" if int(h[:8], 16) % 5 == 0 else "fit"      # fixed 80/20


def load_event(day: str, anchors: list[int], stat: pd.DataFrame, adj: dict):
    pz = np.load(INTERIM / f"panel_{day}.npz", allow_pickle=True)
    dz = np.load(INTERIM / f"drivers_{day}.npz", allow_pickle=True)
    yh, oh = to_hourly(pz["y"], pz["observed"])
    ts = pd.to_datetime([str(t) for t in dz["ts"]])
    n = min(yh.shape[1], dz["X"].shape[1], len(ts))
    yh, oh, X, ts = np.nan_to_num(yh[:, :n]), oh[:, :n], dz["X"][:, :n], ts[:n]
    fips = np.array(pz["fips"], dtype=str)
    ch = [str(c) for c in dz["channels"]]
    S = stat.reindex(fips)[F.STATIC].to_numpy(float)
    S = np.nan_to_num(S, nan=np.nanmedian(S, axis=0))
    A = F.row_normalised_adjacency(fips, adj)
    grp = np.array([county_group(f, day) for f in fips])
    blocks = {}
    for k in anchors:
        hod = np.array([t.hour for t in ts[k + 1:k + 1 + HORIZON]], dtype=float)
        xu, xo, xr = F.build_blocks(X, ch, S, yh[:, max(k - LOOKBACK, 0):k],
                                    oh[:, max(k - LOOKBACK, 0):k], k, HORIZON, hod, A)
        blocks[k] = dict(xu=xu, xo=xo, xr=xr, y0=yh[:, k],
                         yt=yh[:, k + 1:k + 1 + HORIZON],
                         m=oh[:, k + 1:k + 1 + HORIZON] & oh[:, k][:, None])
    return dict(day=day, fips=fips, grp=grp, blocks=blocks, anchors=anchors,
                d_u=blocks[anchors[0]]["xu"].shape[-1],
                d_o=blocks[anchors[0]]["xo"].shape[-1],
                d_r=blocks[anchors[0]]["xr"].shape[-1])


def pack(ev, which: str, mu=None, sd=None):
    """Teacher-forced transitions (deduped by county+hour) and rollout tensors."""
    keep = np.where(ev["grp"] == which)[0] if which in ("fit", "val") else np.arange(len(ev["fips"]))
    seen, ty, tdy, txu, txo, txr = set(), [], [], [], [], []
    ry0, rxu, rxo, rxr, ryt, rm = [], [], [], [], [], []
    for k in ev["anchors"]:
        b = ev["blocks"][k]
        y_prev = np.concatenate([b["y0"][:, None], b["yt"][:, :-1]], axis=1)
        obs_pair = b["m"]
        for t in range(HORIZON):
            for ci in keep:
                key = (int(ci), k + 1 + t)
                if key in seen or not obs_pair[ci, t]:
                    continue
                seen.add(key)
                ty.append(y_prev[ci, t]); tdy.append(b["yt"][ci, t] - y_prev[ci, t])
                txu.append(b["xu"][ci, t]); txo.append(b["xo"][ci, t]); txr.append(b["xr"][ci, t])
        ry0.append(b["y0"][keep]); rxu.append(b["xu"][keep]); rxo.append(b["xo"][keep])
        rxr.append(b["xr"][keep]); ryt.append(b["yt"][keep]); rm.append(b["m"][keep])
    d = dict(tf_y=np.array(ty), tf_dy=np.array(tdy), tf_xu=np.array(txu),
             tf_xo=np.array(txo), tf_xr=np.array(txr),
             roll_y0=ry0, roll_xu=rxu, roll_xo=rxo, roll_xr=rxr, roll_yt=ryt, roll_m=rm)
    if mu is not None:
        for a, b_ in (("tf_xu", 0), ("tf_xo", 1), ("tf_xr", 2)):
            d[a] = (d[a] - mu[b_]) / sd[b_]
        for lst, b_ in ((d["roll_xu"], 0), (d["roll_xo"], 1), (d["roll_xr"], 2)):
            for i in range(len(lst)):
                lst[i] = (lst[i] - mu[b_]) / sd[b_]
    return d


def strata(y0):
    return {"full": np.ones_like(y0, bool), "zero": y0 == 0,
            "near_zero": (y0 > 0) & (y0 <= 0.01), "interior": y0 > 0.01}


def eval_arm(model, ev, mu, sd):
    """Teacher-forced one-step MSE, 24 h path MSE and h+24 MSE, by origin stratum."""
    d = pack(ev, "all", mu, sd)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    out = {}
    with torch.no_grad():
        p = model.step_from_state(T(d["tf_y"]), T(d["tf_xu"]), T(d["tf_xo"]), T(d["tf_xr"])).numpy()
    tgt = d["tf_y"] + d["tf_dy"]
    for s, m in strata(d["tf_y"]).items():
        out[f"tf_mse_{s}"] = float(np.mean((p[m] - tgt[m]) ** 2)) if m.sum() else float("nan")
        out[f"tf_n_{s}"] = int(m.sum())
    py, ty, my, y0s = [], [], [], []
    with torch.no_grad():
        for i in range(len(d["roll_y0"])):
            pr, _ = model(T(d["roll_y0"][i]), T(d["roll_xu"][i]), T(d["roll_xo"][i]), T(d["roll_xr"][i]))
            py.append(pr.numpy()); ty.append(d["roll_yt"][i]); my.append(d["roll_m"][i])
            y0s.append(d["roll_y0"][i])
    py, ty, my, y0s = map(np.concatenate, (py, ty, my, y0s))
    for s, m in strata(y0s).items():
        mm = my[m]
        out[f"path_mse_{s}"] = float(((py[m] - ty[m]) ** 2 * mm).sum() / max(mm.sum(), 1))
        h = my[m][:, -1]
        out[f"h24_mse_{s}"] = float((((py[m][:, -1] - ty[m][:, -1]) ** 2) * h).sum() / max(h.sum(), 1))
        out[f"path_n_{s}"] = int(mm.sum())
    return out


class Constant:
    """Deterministic baselines that share the evaluation path with the networks."""
    def __init__(self, kind, u=0.0, r=0.0, rho=1.0):
        self.kind, self.u, self.r, self.rho, self.clamp_events = kind, u, r, rho, 0
    def step_from_state(self, y, *a):
        if self.kind == "zero":
            return torch.zeros_like(y)
        if self.kind == "damped":
            return self.rho * y
        return y + self.u * (1 - y) - self.r * y
    def __call__(self, y0, xu, xo, xr, collect=False):
        y, out = y0, []
        for _ in range(xu.shape[1]):
            y = torch.clamp(self.step_from_state(y), 0, 1)
            out.append(y)
        return torch.stack(out, 1), []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--events", nargs="*", default=None, help="override pilot events")
    ap.add_argument("--repeat-first", action="store_true", help="reproducibility check")
    a = ap.parse_args()
    t_launch = time.time()

    design = pd.read_csv(OUT / "event_design_table.csv")
    folds = json.load(open(OUT / "event_folds_v2.json"))
    anchors = {k: list(v) for k, v in folds["anchors"].items()}
    fold_of = {e: int(f) for f, v in folds["folds"].items() for e in v}
    stat = pd.read_parquet(INTERIM / "county_statics.parquet").set_index("fips")
    adj = load_adjacency(ROOT / "data/raw/census/county_adjacency2023.txt")

    # ---- outcome-blind pilot selection: exogenous medoid within family -------
    EXO = ["n_counties", "observation_coverage", "gust_p90", "precip_total_mean",
           "log_cust_mean", "log_pop_density_mean", "lat_mean", "lon_mean"]
    Z = design[EXO].to_numpy(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    def medoid(mask):
        idx = np.where(mask)[0]
        D = np.linalg.norm(Z[idx][:, None, :] - Z[idx][None, :, :], axis=-1)
        return design.event[idx[int(np.argmin(D.sum(1)))]]
    if a.events:
        pilot = list(a.events)
    else:
        pilot = [medoid(design.family == "convective"), medoid(design.family == "winter"),
                 medoid(design.family.isin(["tropical", "wind"]))]
    print(f"pilot events (exogenous medoids, no outage information used): {pilot}", flush=True)

    diag, results, effects = TR.Diag(), [], []
    for test_event in pilot:
        src = [e for e in design.event if fold_of[e] != fold_of[test_event]]
        print(f"\n=== test {test_event} (fold {fold_of[test_event]}) · {len(src)} source events ===", flush=True)
        EV = {e: load_event(e, anchors[e], stat, adj) for e in src + [test_event]}
        d_u, d_o, d_r = (EV[src[0]][k] for k in ("d_u", "d_o", "d_r"))

        raw = {e: pack(EV[e], "fit") for e in src}
        cat = lambda k: np.concatenate([raw[e][k] for e in src])
        mu = [cat("tf_xu").mean(0), cat("tf_xo").mean(0), cat("tf_xr").mean(0)]
        sd = [cat("tf_xu").std(0) + 1e-6, cat("tf_xo").std(0) + 1e-6, cat("tf_xr").std(0) + 1e-6]
        fitd = {e: pack(EV[e], "fit", mu, sd) for e in src}
        vald = {e: pack(EV[e], "val", mu, sd) for e in src}

        # ---- exact constant class on the pooled fit transitions --------------
        y = np.concatenate([fitd[e]["tf_y"] for e in src])
        dy = np.concatenate([fitd[e]["tf_dy"] for e in src])
        w = np.ones_like(y)
        U0, R0 = fit_two_flow(y, dy, w, CAP_U_MAIN + CAP_U_BKG, CAP_R)
        a_ray = fit_u_ray(y, dy, w, CAP_U_MAIN + CAP_U_BKG)
        b_ray = fit_r_ray(y, dy, w, CAP_R)
        print(f"  constants  two-flow U0={U0:.6g} R0={R0:.6g} | U-ray a={a_ray:.6g} | R-ray b={b_ray:.6g}",
              flush=True)

        T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32)
        def val_fn(model):
            tot = 0.0
            with torch.no_grad():
                for e in src:
                    d = vald[e]
                    n = 0.0; s = 0.0
                    for i in range(len(d["roll_y0"])):
                        pr, _ = model(T(d["roll_y0"][i]), T(d["roll_xu"][i]),
                                      T(d["roll_xo"][i]), T(d["roll_xr"][i]))
                        mm = T(d["roll_m"][i].astype(np.float32))
                        s += float(((pr - T(d["roll_yt"][i])) ** 2 * mm).sum()); n += float(mm.sum())
                    tot += s / max(n, 1.0)
            return tot / len(src)          # equal-event mean; no single storm decides

        jobs = [("asym_two_flow", "two_flow_start", U0, R0),
                ("asym_one_flow", "interruption_ray_start", a_ray, 0.0),
                ("asym_one_flow", "restoration_ray_start", 0.0, b_ray)]
        per_arm = {}
        for arm, start, u_init, r_init in jobs:
            label = f"{test_event}|{arm}|{start}"
            spec = modular_init(u_init, r_init, CAP_U_MAIN, CAP_U_BKG, CAP_R)
            m = AsymmetricFlows(d_u, d_o, d_r, arm)
            m.apply_modular_init(spec)
            u_chk, r_chk = m.constant_flows()
            init_err = max(abs(u_chk - u_init), abs(r_chk - r_init))
            m0 = AsymmetricFlows(d_u, d_o, d_r, arm); m0.apply_modular_init(spec)
            base0 = eval_arm(m0, EV[test_event], mu, sd)
            rec = TR.train(m, arm, src, fitd, val_fn, a.seed, diag, label)
            ev_res = eval_arm(m, EV[test_event], mu, sd)
            print(f"  {arm:<14} {start:<24} init_err={init_err:.2e} "
                  f"sel={rec['selected_stage']}@{rec['selected_update']} "
                  f"val={rec['selected_validation']:.6e} path24={ev_res['path_mse_full']:.6e} "
                  f"({rec['wall_s']}s)", flush=True)
            results.append(dict(test_event=test_event, arm=arm, start=start,
                                init_flow_error=init_err, u_init=u_init, r_init=r_init,
                                **rec, **{f"test_{k}": v for k, v in ev_res.items()},
                                **{f"update0_{k}": v for k, v in base0.items()}))
            per_arm.setdefault(arm, []).append(results[-1])

        two = per_arm["asym_two_flow"][0]
        one = min(per_arm["asym_one_flow"], key=lambda r: r["selected_validation"])
        cst = {}
        for name, mdl in (("constant_two_flow", Constant("two", U0, R0)),
                          ("constant_one_flow", Constant("two", a_ray, 0.0) if a_ray >= b_ray
                           else Constant("two", 0.0, b_ray)),
                          ("damped_persistence", Constant("damped", rho=float(np.clip(
                              np.sum(y * (y + dy)) / max(np.sum(y * y), 1e-12), 0, 1)))),
                          ("all_zero", Constant("zero"))):
            cst[name] = eval_arm(mdl, EV[test_event], mu, sd)
        row = dict(test_event=test_event, selected_one_flow_start=one["start"])
        for s in ("full", "zero", "near_zero", "interior"):
            for met in ("tf_mse", "path_mse", "h24_mse"):
                a_, b_ = two[f"test_{met}_{s}"], one[f"test_{met}_{s}"]
                row[f"two_{met}_{s}"] = a_; row[f"one_{met}_{s}"] = b_
                row[f"d_{met}_{s}"] = b_ - a_
                row[f"rel_{met}_{s}"] = 100 * (b_ - a_) / b_ if b_ else float("nan")
            for name in cst:
                row[f"{name}_path_mse_{s}"] = cst[name][f"path_mse_{s}"]
                row[f"{name}_tf_mse_{s}"] = cst[name][f"tf_mse_{s}"]
        row["two_vs_own_update0_path_full"] = two["update0_path_mse_full"] - two["test_path_mse_full"]
        row["one_vs_own_update0_path_full"] = one["update0_path_mse_full"] - one["test_path_mse_full"]
        row["two_vs_constant_two_flow_path_full"] = cst["constant_two_flow"]["path_mse_full"] - two["test_path_mse_full"]
        row["one_vs_constant_two_flow_path_full"] = cst["constant_two_flow"]["path_mse_full"] - one["test_path_mse_full"]
        effects.append(row)
        print(f"  -> one-flow start selected on source validation: {one['start']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_json(OUT / "pilot_results.json", orient="records", indent=1)
    pd.DataFrame(effects).to_csv(OUT / "pilot_event_effects.csv", index=False)
    pd.DataFrame(diag.rows).to_csv(OUT / "pilot_training_diagnostics.csv", index=False)
    json.dump(dict(pilot_events=pilot, seed=a.seed,
                   panel_digest=folds["panel_digest"], fold_digest=folds["digest"],
                   caps=dict(C_U_main=CAP_U_MAIN, C_U_background=CAP_U_BKG, C_R=CAP_R),
                   budget=dict(stage_a=TR.STAGE_A_UPDATES, stage_b=TR.STAGE_B_UPDATES,
                               val_every=TR.VAL_EVERY, stage_b_min=TR.STAGE_B_MIN,
                               patience_checks=TR.PATIENCE_CHECKS, lr=TR.LR),
                   wall_time_s=round(time.time() - t_launch, 1)),
              open(OUT / "pilot_run_config.json", "w"), indent=1)
    print(f"\nwrote pilot_results.json, pilot_event_effects.csv, pilot_training_diagnostics.csv")


if __name__ == "__main__":
    main()
