"""EXP01 -- can the two rate functions be recovered, and when can they not?

Ground truth is known in closed form, so this measures recovery rather than
inferring it. The sweep moves trajectories along the axis the identifiability
argument names: how much the state moves while the fit is asked to split the
interruption and restoration rates. Prediction, stated before running:

  H1  Trajectory fit stays good across the whole sweep. Fitting y well is not
      evidence that the rates were recovered.
  H2  Recovery is non-monotone in forcing. Weak forcing leaves the state pinned
      near 0, where the restoration term is multiplied by y ~ 0 and r is
      unidentified. Very strong forcing pins the state near 1, where the
      interruption term is multiplied by (1 - y) ~ 0 and u is unidentified.
  H3  In the poorly identified regimes the two rate errors are positively
      correlated: the fit slides along the ridge where an inflated interruption
      rate is paid for by an inflated restoration rate.

Kill conditions, fixed before running. H1 dies if trajectory RMSE varies by more
than 3x across the sweep. H2 dies if the best-recovery forcing level is at either
end of the sweep for a majority of seeds. H3 dies if the sign of the error
correlation is not consistent across all seeds in the worst-identified level.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asymode.dynamics import TwoRateODE, TwoRateConfig, InflowForm   # noqa: E402
from asymode.synthetic import make_dataset, TrueRates                # noqa: E402
from asymode.fit import FitConfig, train, rate_recovery, recovery_grid, traj_rmse  # noqa: E402

TRUE = TrueRates()


def one_run(pulse_scale: float, seed: int, n: int, T: int, epochs: int) -> dict:
    ds = make_dataset(n=n, T=T, seed=1000 + seed, pulse_scale=pulse_scale,
                      y0_mode="zero", rates=TRUE, n_pulses=(1, 3))
    y0, drivers, y = ds.tensors()
    # Both rates see all three channels: the model must discover on its own that
    # hazard drives interruption and daylight drives restoration.
    cfg = TwoRateConfig(d_in=3, cap_u=TRUE.cap_u, cap_r=TRUE.cap_r,
                        hidden_u=32, hidden_r=32, inflow=InflowForm.SUSCEPTIBLE)
    model = TwoRateODE(cfg)
    t0 = time.time()
    model, info = train(model, y0, drivers, y,
                        FitConfig(epochs=epochs, seed=seed, patience=50))
    grid = recovery_grid(ds.drivers, n=4000, seed=seed)
    rec = rate_recovery(model, grid, TRUE)
    return {
        "pulse_scale": pulse_scale, "seed": seed,
        "state_spread": ds.state_spread,
        "y_mean": float(np.mean(ds.y)), "y_p95": float(np.percentile(ds.y, 95)),
        "frac_near_zero": float(np.mean(ds.y < 0.01)),
        "frac_near_one": float(np.mean(ds.y > 0.99)),
        "traj_rmse": traj_rmse(model, y0, drivers, y),
        "best_val": info["best_val"], "epochs_run": info["epochs_run"],
        "wall_s": round(time.time() - t0, 1),
        **rec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.15, 0.3, 0.6, 1.2, 2.4, 4.8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--out", default=str(ROOT / "results" / "exp01_identifiability.json"))
    a = ap.parse_args()

    rows = []
    for ps in a.scales:
        for sd in a.seeds:
            r = one_run(ps, sd, a.n, a.T, a.epochs)
            rows.append(r)
            print(f"scale {ps:<5} seed {sd} | spread {r['state_spread']:.3f} "
                  f"| traj {r['traj_rmse']:.4f} | nrmse_u {r['nrmse_u']:.3f} "
                  f"nrmse_r {r['nrmse_r']:.3f} | errcorr {r['err_corr']:+.2f} "
                  f"| {r['wall_s']}s", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": "exp01_identifiability",
               "true_rates": TRUE.__dict__,
               "config": vars(a), "rows": rows}
    out.write_text(json.dumps(payload, indent=2))

    print("\n=== aggregate over seeds (mean +/- std) ===")
    print(f"{'scale':>6} {'spread':>7} {'traj_rmse':>18} {'nrmse_u':>16} {'nrmse_r':>16} {'errcorr':>14}")
    for ps in a.scales:
        g = [r for r in rows if r["pulse_scale"] == ps]
        f = lambda k: (np.mean([x[k] for x in g]), np.std([x[k] for x in g]))
        sp, tr, nu, nr, ec = f("state_spread"), f("traj_rmse"), f("nrmse_u"), f("nrmse_r"), f("err_corr")
        print(f"{ps:>6} {sp[0]:>7.3f} {tr[0]:>10.4f}+-{tr[1]:<6.4f} "
              f"{nu[0]:>8.3f}+-{nu[1]:<6.3f} {nr[0]:>8.3f}+-{nr[1]:<6.3f} {ec[0]:>+8.2f}+-{ec[1]:<5.2f}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
