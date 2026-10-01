"""Training loop and recovery diagnostics for the two-rate dynamics."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch

from .dynamics import TwoRateODE, TwoRateConfig, rollout_mse


@dataclass
class FitConfig:
    epochs: int = 300
    batch: int = 128
    lr: float = 3e-3
    weight_decay: float = 0.0
    val_frac: float = 0.2
    patience: int = 60
    seed: int = 0
    device: str = "cpu"
    log_every: int = 0        # 0 = silent


def _split(n: int, val_frac: float, seed: int):
    g = np.random.default_rng(seed)
    idx = g.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


def train(model: TwoRateODE, y0, drivers, y_true, fc: FitConfig):
    """Fit by rollout MSE on the state. Returns (model, history)."""
    torch.manual_seed(fc.seed)
    dev = torch.device(fc.device)
    model = model.to(dev)
    y0, drivers, y_true = y0.to(dev), drivers.to(dev), y_true.to(dev)

    tr, va = _split(y0.shape[0], fc.val_frac, fc.seed)
    tr_t = torch.tensor(tr, dtype=torch.long, device=dev)
    va_t = torch.tensor(va, dtype=torch.long, device=dev)

    opt = torch.optim.Adam(model.parameters(), lr=fc.lr, weight_decay=fc.weight_decay)
    best, best_state, bad = float("inf"), None, 0
    hist = []
    for ep in range(fc.epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        tot, nb = 0.0, 0
        for s in range(0, len(perm), fc.batch):
            b = perm[s:s + fc.batch]
            opt.zero_grad()
            loss = rollout_mse(model, y0[b], drivers[b], y_true[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss); nb += 1
        model.eval()
        with torch.no_grad():
            vl = float(rollout_mse(model, y0[va_t], drivers[va_t], y_true[va_t]))
        hist.append({"epoch": ep, "train": tot / max(nb, 1), "val": vl})
        if fc.log_every and ep % fc.log_every == 0:
            print(f"  ep {ep:4d}  train {tot/max(nb,1):.3e}  val {vl:.3e}")
        if vl < best - 1e-9:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= fc.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": hist, "best_val": best, "epochs_run": len(hist),
                   "fit_config": asdict(fc)}


def recovery_grid(drivers: np.ndarray, n: int = 4000, seed: int = 0) -> np.ndarray:
    """Driver points drawn from the observed driver distribution.

    Recovery is scored where the data actually lives; scoring on a uniform box
    would report error in regions the fit was never asked about.
    """
    g = np.random.default_rng(seed)
    flat = drivers.reshape(-1, drivers.shape[-1])
    return flat[g.choice(len(flat), size=min(n, len(flat)), replace=False)]


def rate_recovery(model: TwoRateODE, grid: np.ndarray, true_rates, y_ref: float = 0.0):
    """Compare fitted rate functions to the truth on a driver grid.

    Also returns the correlation between the two signed errors. The
    identifiability argument predicts that when the state barely moves, the two
    rates trade off along a ridge: an overestimated interruption rate is paid for
    by an overestimated restoration rate, so the errors become *positively*
    correlated even while the trajectory fit stays good.
    """
    dev = next(model.parameters()).device
    gt = torch.tensor(grid, dtype=torch.float32, device=dev)
    u_hat, r_hat = model.rate_curves(gt, y_ref=y_ref)
    u_hat = u_hat.cpu().numpy(); r_hat = r_hat.cpu().numpy()
    u_true = np.asarray(true_rates.u(grid), dtype=np.float64)
    r_true = np.asarray(true_rates.r(grid), dtype=np.float64)
    eu, er = u_hat - u_true, r_hat - r_true
    denom = (np.std(eu) * np.std(er))
    return {
        "rmse_u": float(np.sqrt(np.mean(eu ** 2))),
        "rmse_r": float(np.sqrt(np.mean(er ** 2))),
        "nrmse_u": float(np.sqrt(np.mean(eu ** 2)) / (np.std(u_true) + 1e-12)),
        "nrmse_r": float(np.sqrt(np.mean(er ** 2)) / (np.std(r_true) + 1e-12)),
        "bias_u": float(np.mean(eu)),
        "bias_r": float(np.mean(er)),
        "err_corr": float(np.mean(eu * er) - np.mean(eu) * np.mean(er)) / denom if denom > 1e-12 else float("nan"),
        "corr_u": float(np.corrcoef(u_hat, u_true)[0, 1]) if np.std(u_hat) > 1e-12 else float("nan"),
        "corr_r": float(np.corrcoef(r_hat, r_true)[0, 1]) if np.std(r_hat) > 1e-12 else float("nan"),
    }


def traj_rmse(model: TwoRateODE, y0, drivers, y_true) -> float:
    dev = next(model.parameters()).device
    with torch.no_grad():
        pred = model(y0.to(dev), drivers.to(dev))
        return float(torch.sqrt(torch.mean((pred - y_true.to(dev)) ** 2)))
