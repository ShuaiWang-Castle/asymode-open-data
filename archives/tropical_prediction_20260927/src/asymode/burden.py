"""Recovery-burden extension: one scalar memory state, fed only to the restoration rate.

The gated candidate of the event-transfer task, and nothing else:

    B_{t+1} = rho * B_t + (1 - rho) * y_hat_t
    R_t     = R_theta(x_t, B_t)
    U_t     = U_theta(x_t)

`rho` is a single globally learned scalar constrained to [0.80, 0.999] by a
sigmoid on an unconstrained parameter, initialised at a 24-hour half-life
(rho = 0.5 ** (1/24) = 0.9715).

`B_0` is computed from *observed pre-origin history only*, by running the same
recursion over the lookback window; after the origin the burden is updated from
the model's own prediction, never from future truth. That is the whole point of
the gate in Section 11.1.5, so it is enforced structurally: `forward` never sees
the target.

Everything else -- the rate parameterisation, caps, initialisation rule, loss --
is the two-rate model's, unchanged. The only parameter increment is the extra
input column on the restoration network plus the single `rho`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .dynamics import RateNet, TwoRateConfig

__all__ = ["BurdenConfig", "RecoveryBurdenODE", "RHO_LO", "RHO_HI", "RHO_INIT"]

RHO_LO, RHO_HI = 0.80, 0.999
RHO_INIT = float(0.5 ** (1.0 / 24.0))       # 24-hour half-life


@dataclass
class BurdenConfig(TwoRateConfig):
    """Same fields as the two-rate config; the restoration net gets one extra input."""


class RecoveryBurdenODE(nn.Module):
    def __init__(self, cfg: BurdenConfig):
        super().__init__()
        self.cfg = cfg
        self.u_net = RateNet(cfg.d_in, cap=cfg.cap_u, hidden=cfg.hidden_u)
        self.r_net = RateNet(cfg.d_in + 1, cap=cfg.cap_r, hidden=cfg.hidden_r)
        # rho = RHO_LO + (RHO_HI - RHO_LO) * sigmoid(rho_raw)
        z = (RHO_INIT - RHO_LO) / (RHO_HI - RHO_LO)
        self.rho_raw = nn.Parameter(torch.tensor(float(np.log(z / (1 - z)))))
        # identical initialisation rule to the two-rate model: the head bias is set
        # so the initial rate equals the calibrated flow.
        for net, init in ((self.u_net, cfg.u_init), (self.r_net, cfg.r_init)):
            if init is None:
                continue
            q = min(max(float(init) / net.cap, 1e-9), 1 - 1e-9)
            with torch.no_grad():
                net.net[-1].bias.fill_(float(np.log(q / (1 - q))))

    @property
    def rho(self) -> torch.Tensor:
        return RHO_LO + (RHO_HI - RHO_LO) * torch.sigmoid(self.rho_raw)

    @property
    def seed(self):
        return None

    def burden_from_history(self, hist: torch.Tensor, hist_mask: torch.Tensor | None = None) -> torch.Tensor:
        """B_0 from the observed pre-origin window, by the same recursion.

        `hist` is (B, L) observed states ending at the origin. Unobserved entries,
        if a mask is given, leave the burden unchanged rather than injecting a zero.
        """
        rho = self.rho
        b = hist[:, 0].clone()
        for k in range(1, hist.shape[1]):
            upd = rho * b + (1.0 - rho) * hist[:, k]
            b = upd if hist_mask is None else torch.where(hist_mask[:, k], upd, b)
        return b

    def forward(self, y0: torch.Tensor, drivers: torch.Tensor,
                b0: torch.Tensor | None = None) -> torch.Tensor:
        """y0 (B,), drivers (B, T, d_in), b0 (B,) from observed history. Returns (B, T)."""
        lo, hi = self.cfg.clip_state
        rho = self.rho
        y = y0
        b = y0.clone() if b0 is None else b0
        out = []
        for t in range(drivers.shape[1]):
            x = drivers[:, t]
            u = self.u_net(x)
            r = self.r_net(torch.cat([x, b.unsqueeze(-1)], dim=-1))
            y_next = torch.clamp(y + u * (1.0 - y) - r * y, lo, hi)
            b = rho * b + (1.0 - rho) * y          # burden lags the *predicted* state
            y = y_next
            out.append(y)
        return torch.stack(out, dim=1)
