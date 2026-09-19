"""AsymODE host: two process-specific rate networks driving one outage stock.

    p_t = clip(p_{t-1} + u_t (1 - p_{t-1}) - r_t p_{t-1}, 0, 1),   t = 72..215,
    p_71 = observed.

Damage (served population): a two-hidden-layer MLP of width 32 on the damage
inputs x^U (weather and causal weather summaries only), a learned logit smoother
lbar_t = f_t lbar_{t-1} + (1 - f_t) l_t with f_t = sigmoid(w.x^U_t + b), an
occurrence gate pi_t on instantaneous hazard composites, and a small background
rate:  u_t = clip(0.5 pi_t sigmoid(lbar_t) + 0.015 sigmoid(bg(x^U_t)), 0, 0.515).

Recovery (outaged population): an independent two-hidden-layer MLP of width 16,
recomputed every hour: r_t = 0.5 sigmoid(F_R(x^R_t)).

W is this host as is. GCRK replaces the damage MLP's second linear layer with
`asymode.gcrk.GCRKLayer`, reusing the same W2/b2 parameters; nothing else
changes. Rates do not read the stock, so all hourly rates are computed in one
vectorised pass and only the two scalar recursions (logit smoother, stock) are
sequential; both have hand-written adjoints.
"""
from __future__ import annotations

import torch
from torch import nn

from .gcrk import GCRKLayer

ORIGIN = 72          # first forecast hour; p_71 is the last observed value
U_CAP, R_CAP, BKG_CAP = 0.5, 0.5, 0.015


# --------------------------------------------------------------- scalar recursions
class _Inertia(torch.autograd.Function):
    """y_0 = x_0; y_s = g_s y_{s-1} + (1 - g_s) x_s   (rows independent)."""

    @staticmethod
    def forward(ctx, x, g):
        y = torch.empty_like(x)
        y[:, 0] = x[:, 0]
        for s in range(1, x.shape[1]):
            y[:, s] = g[:, s] * y[:, s - 1] + (1.0 - g[:, s]) * x[:, s]
        ctx.save_for_backward(x, g, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, g, y = ctx.saved_tensors
        gx = torch.empty_like(x)
        gg = torch.zeros_like(g)
        acc = gy[:, -1].clone()
        for s in range(x.shape[1] - 1, 0, -1):
            gx[:, s] = acc * (1.0 - g[:, s])
            gg[:, s] = acc * (y[:, s - 1] - x[:, s])
            acc = gy[:, s - 1] + acc * g[:, s]
        gx[:, 0] = acc
        return gx, gg


class _Stock(torch.autograd.Function):
    """p_s = clip(p_{s-1} + u_s (1 - p_{s-1}) - r_s p_{s-1}, 0, 1), p_{-1} = y0."""

    @staticmethod
    def forward(ctx, u, r, y0):
        y = torch.empty_like(u)
        p = y0
        for s in range(u.shape[1]):
            p = (p + u[:, s] * (1.0 - p) - r[:, s] * p).clamp(0.0, 1.0)
            y[:, s] = p
        ctx.save_for_backward(u, r, y0, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        u, r, y0, y = ctx.saved_tensors
        gu = torch.empty_like(u)
        gr = torch.empty_like(r)
        acc = torch.zeros_like(y0)
        for s in range(u.shape[1] - 1, -1, -1):
            prev = y0 if s == 0 else y[:, s - 1]
            pre = prev + u[:, s] * (1.0 - prev) - r[:, s] * prev
            acc = acc + gy[:, s]
            acc = torch.where((pre < 0.0) | (pre > 1.0), torch.zeros_like(acc), acc)
            gu[:, s] = acc * (1.0 - prev)
            gr[:, s] = -acc * prev
            acc = acc * (1.0 - u[:, s] - r[:, s])
        return gu, gr, acc


rate_inertia = _Inertia.apply
stock_path = _Stock.apply


def rate_inertia_loop(x, g):
    out = [x[:, 0]]
    for s in range(1, x.shape[1]):
        out.append(g[:, s] * out[-1] + (1.0 - g[:, s]) * x[:, s])
    return torch.stack(out, 1)


def stock_path_loop(u, r, y0):
    p, out = y0, []
    for s in range(u.shape[1]):
        p = (p + u[:, s] * (1.0 - p) - r[:, s] * p).clamp(0.0, 1.0)
        out.append(p)
    return torch.stack(out, 1)


# ------------------------------------------------------------------------ networks
def mlp(d_in: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                         nn.ReLU(), nn.Linear(hidden, 1))


class AsymODE(nn.Module):
    def __init__(self, d_u: int, d_r: int, d_occ: int, hidden_u: int = 32, hidden_r: int = 16,
                 u_bias_init: float = -2.0, occ_bias: float = 0.0, bkg_bias: float = -5.0,
                 smoother_bias: float = -3.0):
        super().__init__()
        # creation order: recovery, damage, smoother gate, occurrence gate, background
        self.recovery = mlp(d_r, hidden_r)
        self.damage = mlp(d_u, hidden_u)
        self.smoother = nn.Linear(d_u, 1)
        self.occurrence = nn.Linear(d_occ, 1)
        self.background = nn.Linear(d_u, 1)
        with torch.no_grad():
            self.damage[-1].bias.fill_(u_bias_init)
            self.smoother.weight.zero_()
            self.smoother.bias.fill_(smoother_bias)
            self.occurrence.bias.fill_(occ_bias)
            self.background.bias.fill_(bkg_bias)

    # ------------------------------------------------------------------ kernel slot
    @property
    def kernel(self) -> GCRKLayer | None:
        layer = self.damage[2]
        return layer if isinstance(layer, GCRKLayer) else None

    def attach_gcrk(self, center: torch.Tensor, private_seed: int = 1729) -> GCRKLayer:
        if self.kernel is not None:
            raise RuntimeError("kernel already attached")
        self.damage[2] = GCRKLayer(self.damage[2], center, private_seed)
        return self.damage[2]

    def parameter_groups(self):
        """(host incl. kernel, recovery) -- the two Adam learning-rate groups."""
        rec = [p for n, p in self.named_parameters() if n.startswith("recovery.")]
        host = [p for n, p in self.named_parameters() if not n.startswith("recovery.")]
        return host, rec

    def hidden(self, xu: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.damage[0](xu))

    # --------------------------------------------------------------------- forward
    def forward(self, b: dict, exit_open: bool = True, diagnostics: bool = False) -> dict:
        """b: xu [B,216,d_u], xr [B,216,d_r], xo [B,216,d_occ], geo [B,G], y0 [B]."""
        xu = b["xu"]
        h = self.hidden(xu)
        layer = self.damage[2]
        kd = {}
        if isinstance(layer, GCRKLayer):
            ans = layer(h, b["geo"], diagnostics=diagnostics, exit_open=exit_open)
            a2, kd = ans if diagnostics else (ans, {})
        else:
            a2 = layer(h)
        raw = self.damage[4](torch.relu(a2)).squeeze(-1)[:, ORIGIN:]
        forget = torch.sigmoid(self.smoother(xu[:, ORIGIN:])).squeeze(-1)
        logit = rate_inertia(raw.contiguous(), forget.contiguous())
        r = R_CAP * torch.sigmoid(self.recovery(b["xr"][:, ORIGIN:])).squeeze(-1)
        gate = torch.sigmoid(self.occurrence(b["xo"][:, ORIGIN:])).squeeze(-1)
        bkg = BKG_CAP * torch.sigmoid(self.background(xu[:, ORIGIN:])).squeeze(-1)
        cond = U_CAP * torch.sigmoid(logit)
        u = torch.clamp(gate * cond + bkg, 0.0, U_CAP + BKG_CAP)
        p = stock_path(u.contiguous(), r.contiguous(), b["y0"].contiguous())
        out = dict(P=p, u=u, r=r, gate=gate, background=bkg, conditional=cond,
                   raw_logit=raw, logit=logit, forget=forget)
        if diagnostics:
            out.update(h1=h, a2=a2, **{"kernel_" + k: v for k, v in kd.items()})
        return out


def masked_se(P: torch.Tensor, y: torch.Tensor, m: torch.Tensor):
    """Per-unit (sum of squared error, number of observed cells)."""
    se = (P - y).square() * m
    return se.sum(1), m.sum(1)


def trajectory_loss(P: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Masked MSE of the outage fraction over the rollout hours 72..215."""
    s, n = masked_se(P, y, m)
    return s.sum() / n.sum().clamp_min(1.0)
