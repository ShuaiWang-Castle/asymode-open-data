"""Quick screen (program.md): outer folds 1-2 of the county-grouped design of the twelve-event panel, a fixed
number of training steps on all development units, seed 0, then the held-out counties' open-loop rollouts.

  python screen.py --label base_e3r2 --data e3r2 --arm W+Cin
  python screen.py --label pop_v3p   --data v3p  --arm W+Cin
Every arm of the same seed shares the host initialisation (paired init). Outputs (not in git):
runs/geo_weather_20260924/<label>/fold0<k>/{outer.npz, DONE.json}; evaluate with evaluate_screen.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "2")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode import gcrk_train as G  # noqa: E402

RUNS = ROOT / "runs" / "geo_weather_20260924"
SPLITS = ROOT / "experiments" / "open_gcrk_20260919" / "splits_e3r2.json"
DATA = {"e3r2": ROOT / "data" / "interim" / "open_gcrk" / "features_e3r2.npz",
        "v3p": ROOT / "data" / "interim" / "geo_weather" / "features_v3p.npz",
        "w1": ROOT / "data" / "interim" / "geo_weather" / "features_w1.npz",
        "w2d": ROOT / "data" / "interim" / "geo_weather" / "features_w2d.npz",
        "w2e": ROOT / "data" / "interim" / "geo_weather" / "features_w2e.npz",
        "w2e23": ROOT / "data" / "interim" / "geo_weather" / "features_w2e23.npz",
        "v1D": ROOT / "data" / "interim" / "panel_v1" / "features_v1D.npz"}
SPLIT_FILES = {"e3r2": SPLITS, "v3p": SPLITS, "w1": HERE / "splits_w1.json", "w2d": HERE / "splits_w2d.json",
               "w2e": HERE / "splits_w2e.json", "w2e23": HERE / "splits_w2e23.json", "v1D": HERE / "splits_v1D.json"}
PANEL = {"e3r2": "", "v3p": "", "w1": "w1_", "w2d": "w2d_", "w2e": "w2e_", "w2e23": "w2e23_", "v1D": "v1D_"}   # prefix of the eih_ and train_mask files


def load(data: str) -> dict:
    z = np.load(DATA[data])
    return {k: z[k] for k in z.files}


def attach_phi(F: dict, variant: str, keep: str | None = None) -> dict:
    """Exposure-integrated hazard features eih_<variant>.npz (build_eih.py) as F['phi']; `keep` = a regex on the
    feature names to use a subset."""
    import re
    z = np.load(ROOT / "data" / "interim" / "geo_weather" / f"eih_{variant}.npz")
    assert np.array_equal(z["fips"], F["fips"]) and np.array_equal(z["event"], F["event"])
    names = z["names"].astype(str)
    cols = np.arange(len(names)) if keep is None else np.array([i for i, n in enumerate(names) if re.search(keep, n)])
    F = dict(F); F["phi"] = z["phi"][..., cols]; F["phi_names"] = names[cols]
    return F


HIST = ["p71", "p_max_prefix", "p_mean_66_71", "p_trend_65_71", "prefix_active_share"]
RING_KM = (50, 150, 300)


def hist_features(y: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """The five outage-history inputs of x^R from the 72 observed hours before the origin (as open_gcrk build_features)."""
    import pandas as pd
    v = np.where(obs, y, np.nan)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        p71 = v[:, 71]
        pmax = np.nanmax(v, 1)
        pm6 = np.nanmean(v[:, 66:72], 1)
        p65 = pd.DataFrame(v).T.ffill().T.to_numpy()[:, 65]
        trend = p71 - np.where(np.isnan(p65), p71, p65)
        active = np.nanmean((v > 0.005).astype(float) + 0 * v, 1)
    return np.stack([p71, pmax, pm6, trend, active], 1)


def shift_origin(F: dict, d: int) -> dict:
    """Later forecast origin (RESTORATION_KERNEL_DESIGN): hour 72 + d of the window becomes the origin. The hourly inputs
    move d hours to the left (the last d hours repeat the final hour and are masked), the stock at the new origin and
    the five outage-history inputs are recomputed from the 72 observed hours before it; a county-event whose stock is
    not observed in the hour before the new origin is masked. d = 0 returns the panel unchanged (checked)."""
    names = [str(c) for c in F["recovery_features"]]
    cols = [names.index(c) for c in HIST]
    y, obs = F["y_full"].astype(np.float64), F["obs_full"].astype(bool)
    hist = hist_features(y[:, d:d + 72], obs[:, d:d + 72])
    if d == 0:
        ref = F["xr"][:, 0, cols].astype(np.float64)
        assert np.allclose(np.nan_to_num(hist, nan=-9.0), np.nan_to_num(ref, nan=-9.0), atol=1e-6), "history inputs not reproduced"
        return F
    G = dict(F)
    for k in ("xu", "xr", "xo"):
        G[k] = np.concatenate([F[k][:, d:], np.repeat(F[k][:, -1:], d, 1)], 1)
    G["xr"][:, :, cols] = hist[:, None, :].astype(np.float32)
    ok = obs[:, 71 + d]
    G["y0"] = np.where(ok, y[:, 71 + d], 0.0).astype(np.float32)
    pad = np.zeros((len(y), d), np.float32)
    G["y"] = np.concatenate([F["y"][:, d:], pad], 1)
    G["m"] = (np.concatenate([F["m"][:, d:], pad], 1) * ok[:, None]).astype(F["m"].dtype)
    assert not any(k in F for k in ("phi", "space", "m_train")), "hourly side inputs are not shifted"
    return G


def attach_burden(F: dict, data: str, d: int, mode: str) -> dict:
    """Outage fraction of the other counties in three distance rings in the hour before the origin
    (info_ceiling/restore_capacity.py export). perm: rows permuted over the county-events (null)."""
    z = np.load(ROOT / "data" / "interim" / "panel_v1" / f"burden_{data}.npz")
    assert np.array_equal(z["fips"], F["fips"]) and np.array_equal(z["system"], F["system"]) and tuple(z["ring_km"]) == RING_KM
    b = z["rings"][:, int(z["col_origin"]) + d - 1].astype(np.float32)
    if mode == "perm":
        b = b[np.random.default_rng(20261002).permutation(len(b))]
    F = dict(F); F["burden"] = b
    return F


def design_weighted(F: dict, dev: np.ndarray) -> dict:
    """DATASET_DESIGN v1 section 1 and amendment 2 S7: the training loss weighs county-event i of regime r by
    w_i / Z_r, Z_r = the design-weighted all-zero SSE of r over the training units (regimes with Z_r = 0 left out),
    scaled so the mean weight over observed training cells is 1."""
    w, reg = F["w"].astype(float), F["regime"].astype(str)
    m, y = F["m"].astype(float), F["y"].astype(float)
    u = np.zeros(len(w))
    for r in sorted(set(reg[dev])):
        ii = dev[reg[dev] == r]
        Z = float((w[ii, None] * m[ii] * y[ii] ** 2).sum())
        if Z > 0:
            u[ii] = w[ii] / Z
    scale = m[dev].sum() / max((m[dev] * u[dev, None]).sum(), 1e-300)
    Ff = dict(F)
    Ff["m_train"] = (m * (u * scale)[:, None]).astype(np.float32)
    return Ff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--data", default="e3r2", choices=sorted(DATA))
    ap.add_argument("--arm", default="W+Cin")
    ap.add_argument("--phi", default=None, help="exposure-integrated hazard variant (arm W+Cin+H)")
    ap.add_argument("--keep", default=None, help="regex on hazard feature names")
    ap.add_argument("--train-mask", action="store_true", help="drop EAGLE-I artefact-flagged hours from the training loss")
    ap.add_argument("--mask-placebo", action="store_true", help="with --train-mask: the matched random placebo mask")
    ap.add_argument("--ctx-geo", nargs="*", default=None, help="geographic descriptors added to the county context")
    ap.add_argument("--xu-phi", default=None, help="regex of eih features appended to the damage inputs (hours 72+)")
    ap.add_argument("--xu-phi-variant", default="area")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--design", default="main", choices=["main", "event"], help="county-grouped or event-grouped folds")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--design-weights", action="store_true", help="DATASET_DESIGN v1 loss (design weights, regime-normalised)")
    ap.add_argument("--vuln", default=None, choices=["real", "perm"], help="county vulnerability vector for the DKV arms (perm: county-permuted null)")
    ap.add_argument("--origin-shift", type=int, default=0, help="hours by which the forecast origin is moved later (RESTORATION_KERNEL_DESIGN)")
    ap.add_argument("--burden", default=None, choices=["real", "perm"], help="regional outage burden before the origin for the spatial restoration arms (perm: null)")
    ap.add_argument("--burden-input", default=None, choices=["real", "perm"], help="control: the burden as three plain recovery inputs (zero initial weights), no kernel")
    ap.add_argument("--lr-recovery", type=float, default=None, help="control: learning rate of the recovery network (default: the host's 3e-4)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.lr_recovery is not None:
        G.LR_RECOVERY = a.lr_recovery
    F = load(a.data)
    if a.origin_shift:
        F = shift_origin(F, a.origin_shift)
    if a.burden:
        F = attach_burden(F, a.data, a.origin_shift, a.burden)
    if a.burden_input:               # control: the same burden as three plain recovery inputs, log1p(100 b), no kernel
        Fb = attach_burden(F, a.data, a.origin_shift, a.burden_input)
        extra = np.log1p(100.0 * Fb["burden"])[:, None, :].repeat(F["xr"].shape[1], 1).astype(np.float32)
        F = dict(F); F["xr"] = np.concatenate([F["xr"], extra], -1); F["xr_extra"] = extra.shape[-1]
    if a.phi:
        F = attach_phi(F, PANEL[a.data] + a.phi, a.keep)
    if a.vuln:                       # county vulnerability vector (info_ceiling/vuln_prior.py)
        zv = np.load(ROOT / "data" / "interim" / "panel_v1" / f"vuln_{a.data}.npz")
        assert np.array_equal(zv["fips"], F["fips"]) and np.array_equal(zv["system"], F["system"])
        F = dict(F); F["vuln"] = zv["z" if a.vuln == "real" else "z_perm"].astype(np.float32)
    if a.arm == "STGCRK+Cin":        # neighbour table of the spatio-temporal GCRK (panel_v1/space_v1.py)
        zs = np.load(ROOT / "data" / "interim" / "panel_v1" / f"space_{a.data}.npz")
        assert np.array_equal(zs["fips"], F["fips"]) and np.array_equal(zs["system"], F["system"])
        F = dict(F); F["space"] = dict(nbr=zs["nbr"], wd=zs["wd"], up=zs["up"])
    if a.ctx_geo:
        gi = [list(F["geo_features"].astype(str)).index(g) for g in a.ctx_geo]
        F = dict(F); F["ctx_extra"] = F["geo"][:, gi].astype(np.float32)
    if a.xu_phi:
        import re
        z = np.load(ROOT / "data" / "interim" / "geo_weather" / f"eih_{PANEL[a.data]}{a.xu_phi_variant}.npz")
        assert np.array_equal(z["fips"], F["fips"]) and np.array_equal(z["event"], F["event"])
        names = z["names"].astype(str); cols = [i for i, n in enumerate(names) if re.search(a.xu_phi, n)]
        extra = np.zeros(F["xu"].shape[:2] + (len(cols),), np.float32)
        extra[:, 72:] = z["phi"][..., cols].astype(np.float32)
        F = dict(F); F["xu"] = np.concatenate([F["xu"], extra], -1); F["xu_extra"] = len(cols)
        print("appended to xu:", list(names[cols]), flush=True)
    if a.train_mask:
        tm = ("train_mask_e3" if PANEL[a.data] == "" else f"train_mask_{PANEL[a.data].rstrip('_')}") + \
             ("_placebo" if a.mask_placebo else "") + ".npz"
        F = dict(F); F["m_train"] = np.load(ROOT / "data" / "interim" / "geo_weather" / tm)["m_train"]
        assert F["m_train"].shape == F["m"].shape and (F["m_train"] <= F["m"]).all()
    sp = json.loads(SPLIT_FILES[a.data].read_text())
    assert sp["n_units"] == len(F["fips"])
    for k in a.folds:
        out = RUNS / a.label / f"fold{k:02d}"
        if (out / "DONE.json").exists():
            continue
        # I20 never reuses a partial directory, including an existing empty one.
        out.mkdir(parents=True, exist_ok=a.arm != "CRK+Cin")
        dev, held = np.array(sp[a.design][str(k)]["dev"]), np.array(sp[a.design][str(k)]["outer"])
        if a.arm == "CRK+Cin" and len(np.unique(F["fips"][dev])) < 32:
            raise ValueError("I20 real training requires at least 32 unique fitting counties")
        t0 = time.time()
        log = lambda s: print(f"[{a.label} f{k}] {s}", flush=True)  # noqa: E731
        Ff = design_weighted(F, dev) if a.design_weights else F
        e = G.refit(Ff, dev, a.steps, a.seed, a.arm, log=log)
        res = G.export(e, Ff, held)
        np.savez_compressed(out / "outer.npz", **res)
        torch.save(dict(model_state=e.model.state_dict(), stats=e.stats, arm=a.arm, steps=a.steps), out / "final.pt")
        extra = {}
        if e.model.dose is not None:
            dk = e.model.dose
            extra = dict(dose_beta=dk.beta.detach().tolist(), dose_eta=dk.eta.detach().tolist(), dose_tau=dk.tau().detach().tolist(),
                         dose_theta_u=dk.theta_u.detach().tolist(), dose_theta_r=dk.theta_r.detach().tolist(), vuln=a.vuln)
            if dk.d_vuln:
                extra.update(dose_gamma_u=dk.gamma_u.detach().tolist(), dose_gamma_r=dk.gamma_r.detach().tolist())
        if e.model.rest is not None:
            rk = e.model.rest
            extra = dict(rest_kappa_l=float(rk.L_SCALE * rk.theta_l.detach()), rest_kappa_b=(rk.B_SCALE * rk.theta_b.detach()).tolist(), burden=a.burden)
        extra["origin_shift"] = a.origin_shift
        extra["lr_recovery"] = G.LR_RECOVERY
        extra["burden_input"] = a.burden_input
        if a.arm == "CRK+Cin":
            extra.update(kernel_trace=e.training_trace, microbatch=e.microbatch_size)
        if e.model.haz_beta is not None:
            beta = e.model.haz_beta.detach().numpy()
            top = np.argsort(-beta)[:12]
            extra = dict(beta_nonzero=int((beta > 0).sum()), beta_top={str(F["phi_names"][i]): float(beta[i]) for i in top})
        (out / "DONE.json").write_text(json.dumps(dict(label=a.label, data=a.data, arm=a.arm, phi=a.phi, keep=a.keep, train_mask=a.train_mask, mask_placebo=a.mask_placebo, ctx_geo=a.ctx_geo, xu_phi=a.xu_phi, design=a.design, design_weights=a.design_weights, fold=k,
                                                       steps=a.steps, seed=a.seed, fit_loss=e.last_loss,
                                                       seconds=round(time.time() - t0, 1), **extra), indent=1) + "\n")
        log(f"done in {time.time() - t0:.0f} s, fit loss {e.last_loss:.4e}")


if __name__ == "__main__":
    main()
