"""A6.3 planted-effect worlds (PREREG Amendment 6): semi-synthetic outage paths on the real inputs.

Inputs, initial states p_71, masks and folds are those of the five-event panel with the round-2
inputs (features_r2.npz, splits_r2.json). The forecast-window paths are generated from a trained GCRK
network (R2 screen, seed 0, fold 1) used as the ground truth, kernel opening tanh(alpha) = 0.9:

  T0  shared kernel: every county gets the neutral geographic code
  TB  geography-conditioned memory: true descriptors; the geography-to-code matrix U and the
      code-to-damping / code-to-gain maps Vl, Vg are replaced by seeded random matrices (so the
      learner does not start aligned with the truth), scaled by one factor kappa
  TA  T0 plus a level term on the damage logit: a seeded random direction in descriptor space,
      scaled by s_A

Unobserved heterogeneity: the damage rate is multiplied by exp(N(-s^2/2, s^2)) per unit and per
event x state block; outages are then quantised to whole customers, which produces exact zeros.
(sigma_unit, sigma_block) are chosen on a grid so that, in T0, the truth's RMSE relative to the
all-zero forecast and the unit-level R2 of its window mean are close to the real data's
(0.85 and 0.30). kappa and s_A are chosen so that the planted gap -- RMSE of the neutral-geography
truth relative to the full truth, on the noisy paths -- is about 6%.

Writes data/interim/open_gcrk/features_syn{T0,TB,TA}.npz (y replaced; mu = the truth's mean path,
mu_neutral = the neutral-geography truth's mean path), splits_syn*.json and results/planted_worlds.json.
"""
from __future__ import annotations

import json
import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["OPEN_GCRK_ROUND"] = "r2"
import common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
from asymode.asym_host import AsymODE, U_CAP, BKG_CAP  # noqa: E402
from asymode.gcrk_train import make_batch  # noqa: E402

torch.set_num_threads(1)
TRUTH_CELL = ("main", 0, "1", "GCRK")
TARGET_RATIO, TARGET_R2, TARGET_GAP = 0.85, 0.30, 0.06
N_MC, SEED = 200, 20260921
OUT = C.ROOT / "data" / "interim" / "open_gcrk"


def truth_model(F):
    snap = torch.load(C.cell(*TRUTH_CELL) / "final.pt", weights_only=False)
    m = AsymODE(F["xu"].shape[-1], F["xr"].shape[-1], F["xo"].shape[-1])
    m.attach_gcrk(torch.zeros(F["geo"].shape[-1]))
    m.load_state_dict(snap["model_state"]); m.eval()
    with torch.no_grad():
        m.kernel.alpha.fill_(float(np.arctanh(0.9)))
        m.kernel.training_step.fill_(10_000)
    return m, snap["stats"]


def rates(model, b, geo=None, level=None):
    bb = dict(b)
    if geo is not None:
        bb["geo"] = geo
    with torch.no_grad():
        o = model(bb)
        raw = o["raw_logit"]
        if level is not None:                      # level term added to the raw damage logit, then the host's own path
            from asymode.asym_host import rate_inertia
            logit = rate_inertia((raw + level[:, None]).contiguous(), o["forget"].contiguous())
            u = torch.clamp(o["gate"] * U_CAP * torch.sigmoid(logit) + o["background"], 0.0, U_CAP + BKG_CAP)
        else:
            u = o["u"]
    return u.numpy().astype(np.float64), o["r"].numpy().astype(np.float64)


def rollout(u, r, y0, mult):
    p = y0.copy(); out = np.empty_like(u)
    uu = np.clip(u * mult[:, None], 0.0, U_CAP + BKG_CAP)
    for s in range(u.shape[1]):
        p = np.clip(p + uu[:, s] * (1 - p) - r[:, s] * p, 0, 1); out[:, s] = p
    return out


def multipliers(rng, n, block_inv, n_blocks, su, sb):
    mu = np.exp(rng.normal(-su ** 2 / 2, su, n)) if su > 0 else np.ones(n)
    mb = np.exp(rng.normal(-sb ** 2 / 2, sb, n_blocks))[block_inv] if sb > 0 else np.ones(n)
    return mu * mb


def mean_path(u, r, y0, block_inv, n_blocks, su, sb, cust, seed):
    rng = np.random.default_rng(seed); acc = np.zeros_like(u)
    for _ in range(N_MC):
        acc += quantise(rollout(u, r, y0, multipliers(rng, len(y0), block_inv, n_blocks, su, sb)), cust)
    return acc / N_MC


def quantise(p, cust):
    c = np.maximum(cust, 1.0)[:, None]
    return np.round(p * c) / c


def rmse(a, b, m):
    return float(np.sqrt(((a - b)[m] ** 2).mean()))


def unit_r2(pred, y, m):
    a = np.where(m, pred, 0).sum(1) / np.maximum(m.sum(1), 1); t = np.where(m, y, 0).sum(1) / np.maximum(m.sum(1), 1)
    return float(1 - ((a - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def main():
    F = C.load_features(); n = len(F["y"]); m = F["m"].astype(bool)
    y0 = F["y0"].astype(np.float64); cust = F["cust"].astype(np.float64)
    ev = F["event"].astype(str); st = np.array([f[:2] for f in F["fips"].astype(str)])
    _, block_inv = np.unique(np.char.add(np.char.add(ev, "|"), st), return_inverse=True); n_blocks = block_inv.max() + 1
    model, stats = truth_model(F)
    b = make_batch(F, np.arange(n), stats)
    k = model.kernel
    neutral = (3.0 * torch.atanh(k.geo_center.clamp(-0.999, 0.999)))[None].repeat(n, 1)
    u0, r0 = rates(model, b, geo=neutral)

    # ---- noise calibration on T0
    best = None
    for su in (1.0, 1.5, 2.0, 2.5):
        for sb in (0.5, 1.0, 1.5):
            rng = np.random.default_rng(SEED)
            y = quantise(rollout(u0, r0, y0, multipliers(rng, n, block_inv, n_blocks, su, sb)), cust)
            mu = mean_path(u0, r0, y0, block_inv, n_blocks, su, sb, cust, SEED + 1)
            ratio = rmse(mu, y, m) / rmse(np.zeros_like(y), y, m); r2 = unit_r2(mu, y, m)
            score = (ratio - TARGET_RATIO) ** 2 + (0.5 * (r2 - TARGET_R2)) ** 2
            print(f"sigma_unit {su} sigma_block {sb}: truth/zero {ratio:.3f} unit R2 {r2:.3f} zero share {float((y[m] == 0).mean()):.3f}", flush=True)
            if best is None or score < best[0]:
                best = (score, su, sb, ratio, r2)
    _, su, sb, ratio, r2 = best
    print(f"chosen sigma_unit {su} sigma_block {sb} (truth/zero {ratio:.3f}, unit R2 {r2:.3f})", flush=True)

    # ---- planted truths
    gen = torch.Generator().manual_seed(777)
    d, r_, G = k.Vl.shape[0], k.Vl.shape[1], k.U.shape[1]
    U_t = torch.randn(r_, G, generator=gen) / np.sqrt(G)
    Vl_t = torch.randn(d, r_, generator=gen); Vg_t = torch.randn(d, r_, generator=gen)
    w_A = torch.randn(G, generator=gen); w_A = w_A / w_A.norm()
    keep = {n_: getattr(k, n_).detach().clone() for n_ in ("U", "Vl", "Vg")}

    def world_tb(kappa):
        with torch.no_grad():
            k.U.copy_(U_t); k.Vl.copy_(kappa * Vl_t); k.Vg.copy_(0.5 * kappa * Vg_t)
        out = rates(model, b)
        with torch.no_grad():
            for n_, v in keep.items():
                getattr(k, n_).copy_(v)
        return out

    def gap(u1, r1):
        rng = np.random.default_rng(SEED)
        y = quantise(rollout(u1, r1, y0, multipliers(rng, n, block_inv, n_blocks, su, sb)), cust)
        mu = mean_path(u1, r1, y0, block_inv, n_blocks, su, sb, cust, SEED + 1)
        mu_n = mean_path(u0, r0, y0, block_inv, n_blocks, su, sb, cust, SEED + 1)
        return rmse(mu_n, y, m) / rmse(mu, y, m) - 1, y, mu, mu_n

    def tune(make, grid):
        res = []
        for v in grid:
            g_, *_ = gap(*make(v)); res.append((abs(g_ - TARGET_GAP), v, g_)); print(f"   scale {v}: planted gap {g_:+.4f}", flush=True)
        return min(res)[1]

    print("TB: tuning kappa", flush=True)
    kappa = tune(world_tb, (2.0, 4.0, 6.0, 9.0, 13.0))
    g_std = b["geo"]
    print("TA: tuning s_A", flush=True)
    s_A = tune(lambda s: rates(model, b, geo=neutral, level=s * (g_std @ w_A)), (0.1, 0.2, 0.35, 0.5, 0.8))

    worlds = {"T0": (u0, r0), "TB": world_tb(kappa), "TA": rates(model, b, geo=neutral, level=s_A * (g_std @ w_A))}
    report = dict(sigma_unit=su, sigma_block=sb, kappa=kappa, s_A=s_A, truth_cell="/".join(map(str, TRUTH_CELL)), n_mc=N_MC, worlds={})
    base = {k_: F[k_] for k_ in F}
    for name, (u1, r1) in worlds.items():
        g_, y, mu, mu_n = gap(u1, r1)
        lam_info = {}
        arr = dict(base); arr["y"] = y.astype(np.float32); arr["mu"] = mu.astype(np.float32); arr["mu_neutral"] = mu_n.astype(np.float32)
        yf = np.array(F["y_full"], copy=True); yf[:, 72:] = y; arr["y_full"] = yf.astype(np.float32)
        np.savez_compressed(OUT / f"features_syn{name}.npz", **arr)
        sp = json.loads((C.HERE / "splits_r2.json").read_text())
        (C.HERE / f"splits_syn{name}.json").write_text(json.dumps(dict(n_units=sp["n_units"], main=sp["main"])) + "\n")
        report["worlds"][name] = dict(planted_gap=g_, rmse_zero=rmse(np.zeros_like(y), y, m), rmse_truth=rmse(mu, y, m),
                                      rmse_neutral_truth=rmse(mu_n, y, m), neutral_vs_truth_mean_path=rmse(mu_n, mu, m),
                                      unit_r2_truth=unit_r2(mu, y, m), zero_share=float((y[m] == 0).mean()),
                                      mean_outage=float(y[m].mean()), real_mean_outage=float(F["y"][m].mean()), **lam_info)
        print(name, json.dumps(report["worlds"][name]), flush=True)
    (C.HERE / "results" / "planted_worlds.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
