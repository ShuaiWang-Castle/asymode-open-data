"""EXP02 -- onset: what the epidemic form cannot do, and what it costs.

An outage begins in a county that currently has none. Any dynamics whose inflow
is proportional to the present outage fraction is, at y = 0, identically zero:
u * y * (1 - y) = 0. Such a model cannot start an outage at all. This is algebra,
not an empirical finding, so the experiment is built to measure the *cost* under
a fair comparison rather than to re-derive the algebra.

Three arms, identical drivers, identical loss, identical capacity, identical
seeds:
  susceptible        u * (1 - y)              -- this work
  transmission       u * y * (1 - y)          -- epidemic form as usually written
  transmission_seed  u * (y + eps) * (1 - y)  -- steelman: eps >= 0 learnable, so
                                                 the arm CAN ignite from zero

The steelman is the arm that matters. It removes the algebraic objection and
replaces it with a measurable tension: eps must be large enough to ignite from
zero, but a large eps swamps the y-dependence that makes the form epidemic, so
the arm is pushed toward the susceptible form it was meant to contrast with. The
fitted eps is therefore reported as a headline number, not a nuisance parameter.

Ground truth uses kappa = 1.5 by default: the served pool enters as (1 - y)^1.5,
which NONE of the three arms implements. All three are misspecified, so the
comparison cannot be dismissed as the proposed form grading its own homework.
kappa = 1.0 is also run, as the well-specified reference point.

What this experiment does NOT establish: that onset-from-zero is common in real
county outage data. That is an empirical claim about the world and must be
settled on public observations, not here. Results are graded accordingly.

Kill conditions, fixed before running. The onset claim dies if, on kappa = 1.5,
the susceptible arm fails to beat transmission_seed on onset trajectories in any
of the three seeds, or if the fitted eps is not consistently pushed far above its
initialisation.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asymode.dynamics import TwoRateODE, TwoRateConfig, InflowForm   # noqa: E402
from asymode.synthetic import make_dataset, TrueRates                # noqa: E402
from asymode.fit import FitConfig, train                             # noqa: E402

TRUE = TrueRates()
ARMS = [InflowForm.SUSCEPTIBLE, InflowForm.TRANSMISSION, InflowForm.TRANSMISSION_SEED]


def split_rmse(model, y0, drivers, y_true, onset_mask):
    with torch.no_grad():
        pred = model(y0, drivers)
        se = (pred - y_true) ** 2
        rm = lambda m: float(torch.sqrt(se[m].mean())) if m.any() else float("nan")
        return {"rmse_all": float(torch.sqrt(se.mean())),
                "rmse_onset": rm(onset_mask),
                "rmse_started": rm(~onset_mask)}


def one_run(arm: InflowForm, seed: int, kappa: float, n: int, T: int, epochs: int) -> dict:
    ds = make_dataset(n=n, T=T, seed=2000 + seed, pulse_scale=1.2,
                      y0_mode="mixed", rates=TRUE, kappa=kappa, n_pulses=(1, 3))
    y0, drivers, y = ds.tensors()
    onset = torch.tensor(ds.y0 == 0.0)
    cfg = TwoRateConfig(d_in=3, cap_u=TRUE.cap_u, cap_r=TRUE.cap_r,
                        hidden_u=32, hidden_r=32, inflow=arm)
    model = TwoRateODE(cfg)
    t0 = time.time()
    model, info = train(model, y0, drivers, y, FitConfig(epochs=epochs, seed=seed, patience=50))
    out = {"arm": arm.value, "seed": seed, "kappa": kappa,
           "n_onset": int(onset.sum()), "n_started": int((~onset).sum()),
           "best_val": info["best_val"], "epochs_run": info["epochs_run"],
           "wall_s": round(time.time() - t0, 1),
           **split_rmse(model, y0, drivers, y, onset)}
    s = model.seed
    out["fitted_seed_eps"] = float(s.detach()) if s is not None else None
    # How far the onset trajectories actually rose in the truth, so the reported
    # error can be read against the signal it is failing to track.
    out["truth_onset_peak_mean"] = float(np.mean(ds.y[ds.y0 == 0.0].max(axis=1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", type=float, nargs="+", default=[1.0, 1.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--out", default=str(Path("results") / "exp02_onset.json"))
    a = ap.parse_args()

    rows = []
    for kap in a.kappas:
        for arm in ARMS:
            for sd in a.seeds:
                r = one_run(arm, sd, kap, a.n, a.T, a.epochs)
                rows.append(r)
                eps = "-" if r["fitted_seed_eps"] is None else f"{r['fitted_seed_eps']:.4f}"
                print(f"kappa {kap} {arm.value:<18} seed {sd} | all {r['rmse_all']:.4f} "
                      f"| onset {r['rmse_onset']:.4f} | started {r['rmse_started']:.4f} "
                      f"| eps {eps} | {r['wall_s']}s", flush=True)

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Config is archived with a repo-relative path so results never carry the
    # absolute location of the checkout.
    cfg = dict(vars(a)); cfg["out"] = str(out.relative_to(ROOT))
    out.write_text(json.dumps({"experiment": "exp02_onset", "true_rates": TRUE.__dict__,
                               "config": cfg, "rows": rows}, indent=2))

    print("\n=== aggregate over seeds (mean +/- std) ===")
    for kap in a.kappas:
        print(f"\n-- kappa = {kap} (truth uses (1-y)^{kap}) --")
        print(f"{'arm':<20}{'rmse_all':>18}{'rmse_onset':>18}{'rmse_started':>18}{'fitted_eps':>14}")
        for arm in ARMS:
            g = [r for r in rows if r["arm"] == arm.value and r["kappa"] == kap]
            f = lambda k: (np.mean([x[k] for x in g]), np.std([x[k] for x in g]))
            al, on, st = f("rmse_all"), f("rmse_onset"), f("rmse_started")
            e = [x["fitted_seed_eps"] for x in g if x["fitted_seed_eps"] is not None]
            es = f"{np.mean(e):.4f}+-{np.std(e):.4f}" if e else "-"
            print(f"{arm.value:<20}{al[0]:>10.4f}+-{al[1]:<6.4f}{on[0]:>10.4f}+-{on[1]:<6.4f}"
                  f"{st[0]:>10.4f}+-{st[1]:<6.4f}{es:>14}")
    print(f"\nwritten: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
