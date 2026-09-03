"""S-2: does rate recovery track the exact local information bound? (C4)

Held fixed across the sweep: the true rate functions, the driver paths, the
noise variance, the total sample size, the relaxation scale, the optimiser and
its budget. The ONLY manipulated quantity is the conditional state dispersion at
shared drivers: every driver path is replicated r times with initial states
drawn from U(0, s), s in S_GRID. Because the same path is shared by its r
replicates, every (path, step) is a group of r cells with identical drivers --
the fixed-x design of Theorem 2 -- whose realised state spread we measure.

Two layers, reported per level of s:
  oracle  -- local OLS on each shared-driver group, on the exact fixed-design
             model D_i = phi(y_i)^T beta + eps_i (noise on the increment only,
             noiseless states in the design); its squared error is compared to
             the exact Gauss-Markov / CRLB value sigma^2 (A^T A)^{-1}.
  neural  -- the two-rate model fitted by rollout MSE on the replicated dataset
             (observation noise on the states, as in the real pipeline); rate
             recovery RMSE on the driver grid, against the mean local information
             N_group * var(y | group) and the mean CRLB total sigma^2 tr((A^T A)^{-1}).

Negative control: at fixed s = S_CTRL the true rate magnitudes are scaled by
{0.5, 2} (cap_u, cap_r), which changes relaxation and signal scale but not the
manipulated quantity; the two sweeps must separate in the (information, error)
plane for the main reading to stand.

Interpretation fixed in docs/THEORY_PLAN.md (S-2): if the neural recovery error
tracks the bound within a constant factor across the sweep, Prop. 2 is the
predictive statement; if the error floor is set elsewhere, the theory is
descriptive and the paper's claim is limited to Prop. 1.
"""
from __future__ import annotations

import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode import information as I, panels as panelset                      # noqa: E402
from asymode.synthetic import TrueRates, make_drivers, rollout_truth           # noqa: E402
from asymode.dynamics import TwoRateConfig, TwoRateODE, InflowForm, calibrate_init  # noqa: E402
from asymode.fit import FitConfig, train, rate_recovery, recovery_grid         # noqa: E402

S_GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
S_CTRL = 0.1
CTRL_SCALES = (0.5, 2.0)


def build(n_paths, T, r, s, rates, sigma, seed):
    rng = np.random.default_rng(seed)
    drv = make_drivers(n_paths, T, rng, pulse_scale=1.0)              # (n_paths, T, 3), fixed by seed
    drivers = np.repeat(drv, r, axis=0)                                # replicate paths
    y0 = rng.uniform(0.0, s, size=n_paths * r) if s > 0 else np.zeros(n_paths * r)
    y = rollout_truth(drivers, y0, rates)                              # noiseless states (n, T)
    ynoisy = np.clip(y + rng.normal(scale=sigma, size=y.shape), 0, 1) if sigma > 0 else y
    return drivers, y0, y, ynoisy, rng


def oracle_layer(drivers, y0, y, rates, sigma, r, rng):
    """Per shared-driver group (path, t): exact fixed-design OLS vs its CRLB."""
    n, T = y.shape
    states = np.concatenate([y0[:, None], y[:, :-1]], axis=1)          # y_t entering step t
    U = rates.u(drivers); R = rates.r(drivers)                          # (n, T)
    groups = n // r
    se_u, se_r, crlb_u, crlb_r, nvar, lam_min, gap, tr_crlb = [], [], [], [], [], [], [], []
    for g in range(groups):
        rows = slice(g * r, (g + 1) * r)
        for t in range(T):
            ys = states[rows, t]; u, rr = U[g * r, t], R[g * r, t]
            A = I.phi(ys); G = A.T @ A
            if np.linalg.det(G) <= 1e-14:
                continue
            eps = rng.normal(scale=sigma, size=r)
            D = A @ np.array([u, rr]) + eps
            b = np.linalg.solve(G, A.T @ D)
            cov = sigma ** 2 * np.linalg.inv(G)
            se_u.append((b[0] - u) ** 2); se_r.append((b[1] - rr) ** 2)
            crlb_u.append(cov[0, 0]); crlb_r.append(cov[1, 1]); tr_crlb.append(np.trace(cov))
            st = I.gram_stats(ys); nvar.append(r * st["var"]); lam_min.append(st["lam_min"])
            gap.append(I.one_rate_gap(u, rr, ys))
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return dict(n_groups=len(se_u), mse_u=f(se_u), mse_r=f(se_r), crlb_u=f(crlb_u), crlb_r=f(crlb_r),
                ratio_u=f(se_u) / f(crlb_u) if crlb_u else float("nan"),
                ratio_r=f(se_r) / f(crlb_r) if crlb_r else float("nan"),
                mean_n_var=f(nvar), median_n_var=float(np.median(nvar)) if nvar else float("nan"),
                mean_lam_min=f(lam_min), mean_one_rate_gap=f(gap), mean_crlb_total=f(tr_crlb),
                frac_groups_identifiable=len(se_u) / (groups * T))


def neural_layer(drivers, y0, ynoisy, rates, seed, epochs, patience):
    torch.manual_seed(seed)
    ytr = np.concatenate([y0[:, None], ynoisy], axis=1)
    u0, r0 = calibrate_init(ytr, np.ones_like(ytr, dtype=bool), InflowForm.SUSCEPTIBLE)
    cfg = TwoRateConfig(d_in=drivers.shape[-1], cap_u=rates.cap_u * 1.5, cap_r=rates.cap_r * 1.5,
                        hidden_u=32, hidden_r=32, inflow=InflowForm.SUSCEPTIBLE, u_init=u0, r_init=r0)
    model = TwoRateODE(cfg)
    fc = FitConfig(epochs=epochs, patience=patience, seed=seed, batch=128, lr=3e-3)
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    model, hist = train(model, t(y0), t(drivers), t(ynoisy), fc)
    grid = recovery_grid(drivers, n=2000, seed=seed)
    rec = rate_recovery(model, grid, rates)
    rec.update(epochs_run=hist["epochs_run"], best_val=hist["best_val"], hit_cap=hist["epochs_run"] >= epochs)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=64); ap.add_argument("--replicates", type=int, default=8)
    ap.add_argument("--T", type=int, default=96); ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=250); ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--driver-seed", type=int, default=7)
    ap.add_argument("--out", default="results/s2_crlb_tracking.json")
    a = ap.parse_args()
    t_launch = time.time(); source = panelset.source_version(ROOT)
    base = TrueRates()
    rows = []

    def run_cell(kind, s, scale, seed):
        rates = replace(base, cap_u=base.cap_u * scale, cap_r=base.cap_r * scale)
        drivers, y0, y, ynoisy, rng = build(a.n_paths, a.T, a.replicates, s, rates, a.sigma, a.driver_seed)
        # realised dispersion at shared drivers (noiseless states) -- the manipulated quantity, measured
        st = np.concatenate([y0[:, None], y[:, :-1]], 1).reshape(a.n_paths, a.replicates, a.T)
        realised = float(np.mean(np.var(st, axis=1)))
        orc = oracle_layer(drivers, y0, y, rates, a.sigma, a.replicates, np.random.default_rng(100 + seed))
        t0 = time.time()
        nn = neural_layer(drivers, y0, ynoisy, rates, seed, a.epochs, a.patience)
        row = dict(kind=kind, s=s, rate_scale=scale, seed=seed, realised_var_shared=realised,
                   oracle=orc, neural=nn, wall_s=round(time.time() - t0, 1))
        rows.append(row)
        print(f"  {kind:<8} s={s:<5} scale={scale:<4} seed={seed}  N*var {orc['mean_n_var']:.4f}  "
              f"oracle mse/crlb u {orc['ratio_u']:.2f} r {orc['ratio_r']:.2f}  "
              f"neural rmse_u {nn['rmse_u']:.4f} rmse_r {nn['rmse_r']:.4f} corr {nn['err_corr']:+.2f}  "
              f"{nn['epochs_run']}ep {row['wall_s']}s", flush=True)

    for s in S_GRID:
        for seed in a.seeds:
            run_cell("sweep", s, 1.0, seed)
    for sc in CTRL_SCALES:
        for seed in a.seeds:
            run_cell("control", S_CTRL, sc, seed)

    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a)); cfg.update(source=source, s_grid=list(S_GRID), s_ctrl=S_CTRL, ctrl_scales=list(CTRL_SCALES),
                                    true_rates=vars(base), wall_time_s=round(time.time() - t_launch, 1),
                                    note="synthetic; known truth; oracle layer uses noise on the increment only")
    out.write_text(json.dumps({"config": cfg, "rows": rows}, indent=1))
    print(f"\nwritten: {a.out}")
    # slope of log neural rate MSE against log mean N*var over the sweep (seed-averaged)
    sw = [r for r in rows if r["kind"] == "sweep" and r["oracle"]["mean_n_var"] > 0]
    xs = np.log([r["oracle"]["mean_n_var"] for r in sw]); ys = np.log([r["neural"]["rmse_r"] ** 2 for r in sw])
    if len(set(xs.round(6))) > 1:
        print(f"log-log slope of neural MSE(R) vs N*var over the sweep: {np.polyfit(xs, ys, 1)[0]:+.2f} (CRLB predicts -1)")


if __name__ == "__main__":
    main()
