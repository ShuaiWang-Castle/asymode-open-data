"""EXP09 -- identifiability along the state axis, with both degenerate ends reachable.

Second design. The hypotheses and kill conditions are in
`docs/PREREGISTRATION_exp01_h2.md` and are not restated here.

EXP01 swept forcing amplitude and reported that the saturated regime never
appeared. That was a property of its generator, not of the dynamics: the update
`y <- y(1 - u - r) + u` contracts toward `u/(u + r)`, so the attainable band is
fixed by the rate constants and forcing only moves the state inside it. For the
constants EXP01 used the band was [0.074, 0.944], and both of the regimes its H2
named lie outside it. No amount of forcing could have reached either.

So the axis here is the equilibrium itself. For each target `y*` the rates are
set to `u0 = S*y*` and `r0 = S*(1 - y*)` with `S` fixed, which puts the
equilibrium exactly at `y*` while holding the relaxation rate constant across the
sweep -- points differ in where the state sits, not in how fast it settles there.
Each cap is twice its operating rate, placing each sigmoid at its midpoint where
it is best conditioned, and the intercepts follow in closed form.

Because `u0` and `r0` necessarily change size as `y*` moves, recovery could track
rate magnitude rather than state position. `--control` runs the registered
control for that: `y*` pinned at 0.5 while `S` sweeps the same range of values
`u0` takes in the main sweep. The pre-registration makes the control a gate --
if it reproduces the main sweep's shape, the main sweep says nothing about state
position, whatever it shows.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asymode.dynamics import TwoRateODE, TwoRateConfig, InflowForm   # noqa: E402
from asymode.synthetic import make_dataset, make_drivers, TrueRates  # noqa: E402
from asymode.fit import (FitConfig, train, rate_recovery,            # noqa: E402
                         recovery_grid, traj_rmse)

A_U, A_R = 3.0, 1.5          # slopes are held fixed; only the scales move


def rates_for(y_star: float, S: float, h_med: float) -> TrueRates:
    """Generator constants that put the equilibrium at `y_star`.

    `u0/(u0 + r0) = y_star` by construction and `u0 + r0 = S` at every sweep
    point. Capping at twice the operating rate puts each sigmoid at its midpoint,
    so the intercepts are the closed forms below and nothing is tuned.
    """
    u0, r0 = S * y_star, S * (1.0 - y_star)
    return TrueRates(cap_u=2 * u0, cap_r=2 * r0,
                     a_u=A_U, b_u=-A_U * h_med, a_r=A_R, b_r=0.0)


def one_run(y_star: float, S: float, h_med: float, seed: int,
            n: int, T: int, epochs: int, burn: int) -> dict:
    true = rates_for(y_star, S, h_med)
    # Simulate a burn-in and fit on what follows. Starting every trajectory at
    # y = 0 would spend much of the window in transit -- the relaxation time is
    # 1/S, so reaching y* = 0.99 from zero takes roughly ln(100)/S steps -- and
    # that transit is exactly the well-identified interior the sweep is trying to
    # get away from at the ends. Fitting through it would make the degenerate
    # regimes look more identifiable than they are, which is the direction of
    # error that matters here.
    ds = make_dataset(n=n, T=burn + T, seed=1000 + seed, pulse_scale=1.0,
                      y0_mode="zero", rates=true, n_pulses=(1, 3))
    y_np, d_np = ds.y[:, burn:], ds.drivers[:, burn:]
    y0_np = ds.y[:, burn - 1]
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    y0, drivers, y = t(y0_np), t(d_np), t(y_np)
    # Both rates see all three channels: the model must discover on its own which
    # channel drives which rate. The caps are the generator's, so the fit is not
    # also being asked to find the scale.
    cfg = TwoRateConfig(d_in=3, cap_u=true.cap_u, cap_r=true.cap_r,
                        hidden_u=32, hidden_r=32, inflow=InflowForm.SUSCEPTIBLE)
    model = TwoRateODE(cfg)
    t0 = time.time()
    model, info = train(model, y0, drivers, y,
                        FitConfig(epochs=epochs, seed=seed, patience=50))
    rec = rate_recovery(model, recovery_grid(d_np, n=4000, seed=seed), true)
    return {
        "y_star": y_star, "S": S, "seed": seed,
        "cap_u": true.cap_u, "cap_r": true.cap_r,
        "y_mean": float(np.mean(y_np)),
        "frac_near_zero": float(np.mean(y_np < 0.01)),
        "frac_near_one": float(np.mean(y_np > 0.99)),
        "traj_rmse": traj_rmse(model, y0, drivers, y),
        "best_val": info["best_val"], "epochs_run": info["epochs_run"],
        "wall_s": round(time.time() - t0, 1),
        **rec,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.01, 0.03, 0.08, 0.20, 0.40, 0.60, 0.80, 0.92, 0.97, 0.99])
    ap.add_argument("--S", type=float, default=0.10)
    ap.add_argument("--control", action="store_true",
                    help="registered control: y* pinned at 0.5, S swept instead")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--burn", type=int, default=64,
                    help="steps simulated but not fitted, so the fit sees the "
                         "target regime rather than the approach to it")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--out", default="results/exp09_identifiability_state.json")
    a = ap.parse_args()

    # The hazard median sets b_u so the interruption sigmoid sits at its midpoint
    # on typical forcing. Taken from the same driver process every point uses.
    h_med = float(np.median(make_drivers(256, a.T, np.random.default_rng(0),
                                         pulse_scale=1.0)[..., 0]))

    if a.control:
        # Sweep S over the same values u0 takes in the main sweep, at y* = 0.5.
        # At y* = 0.5, u0 = S/2, so S = 2 * u0 recovers each main-sweep magnitude.
        points = [(0.5, 2.0 * a.S * ys) for ys in a.targets]
    else:
        points = [(ys, a.S) for ys in a.targets]

    rows = []
    for y_star, S in points:
        for sd in a.seeds:
            r = one_run(y_star, S, h_med, sd, a.n, a.T, a.epochs, a.burn)
            rows.append(r)
            print(f"y* {y_star:<5} S {S:<6.4f} seed {sd} | mean y {r['y_mean']:.4f} "
                  f"| <.01 {r['frac_near_zero']:.3f} >.99 {r['frac_near_one']:.3f} "
                  f"| traj {r['traj_rmse']:.4f} | nrmse_u {r['nrmse_u']:.3f} "
                  f"nrmse_r {r['nrmse_r']:.3f} | errcorr {r['err_corr']:+.2f} "
                  f"| {r['wall_s']}s", flush=True)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a)); cfg["out"] = a.out; cfg["h_med"] = h_med
    cfg["preregistration"] = "docs/PREREGISTRATION_exp01_h2.md"
    out.write_text(json.dumps({"experiment": "exp09_identifiability_state",
                               "config": cfg, "rows": rows}, indent=2))

    print(f"\n=== aggregate over seeds (mean +/- std) ==="
          f"{'  [CONTROL: y* pinned at 0.5]' if a.control else ''}")
    print(f"{'y*':>6}{'S':>8}{'mean y':>9}{'<.01':>7}{'>.99':>7}"
          f"{'traj_rmse':>17}{'nrmse_u':>15}{'nrmse_r':>15}{'errcorr':>13}")
    for y_star, S in points:
        g = [r for r in rows if r["y_star"] == y_star and r["S"] == S]
        f = lambda k: (np.mean([x[k] for x in g]), np.std([x[k] for x in g]))
        ym, fz, fo = f("y_mean")[0], f("frac_near_zero")[0], f("frac_near_one")[0]
        tr, nu, nr, ec = f("traj_rmse"), f("nrmse_u"), f("nrmse_r"), f("err_corr")
        print(f"{y_star:>6.2f}{S:>8.4f}{ym:>9.4f}{fz:>7.3f}{fo:>7.3f}"
              f"{tr[0]:>10.4f}+-{tr[1]:<5.4f}{nu[0]:>8.3f}+-{nu[1]:<5.3f}"
              f"{nr[0]:>8.3f}+-{nr[1]:<5.3f}{ec[0]:>+8.2f}+-{ec[1]:<4.2f}")

    # The registered quantities, stated so the kill conditions can be applied
    # from the printout rather than re-derived by hand.
    joint = {}
    for y_star, S in points:
        g = [r for r in rows if r["y_star"] == y_star and r["S"] == S]
        joint[(y_star, S)] = np.mean([max(r["nrmse_u"], r["nrmse_r"]) for r in g])
    keys = list(joint)
    best = min(keys, key=lambda k: joint[k])
    lo, hi = keys[0], keys[-1]
    d_lo = np.mean([r["nrmse_r"] - r["nrmse_u"] for r in rows
                    if (r["y_star"], r["S"]) == lo])
    d_hi = np.mean([r["nrmse_r"] - r["nrmse_u"] for r in rows
                    if (r["y_star"], r["S"]) == hi])
    print(f"\njoint error min at {best}, ratio max/min "
          f"{max(joint.values())/max(min(joint.values()), 1e-12):.2f}x "
          f"(H2a needs the min in the interior and the ratio above 2x)")
    print(f"nrmse_r - nrmse_u: {d_lo:+.3f} at the low end, {d_hi:+.3f} at the high end "
          f"(H2b needs a sign flip, not a magnitude difference)")
    tj = [r["traj_rmse"] for r in rows]
    print(f"traj_rmse spread {max(tj)/max(min(tj), 1e-12):.2f}x "
          f"(H1 dies above 3x)")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
