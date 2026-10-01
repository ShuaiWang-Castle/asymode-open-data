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
        self.level, self.level_source, self.ctx_in = None, None, None
        self.mech, self.mech_in = None, None
        self.haz_beta, self.haz_a, self.haz_b = None, None, None
        self.haz_signed = False           # True: signed coefficients, a logit shift of the damage rate
        with torch.no_grad():
            self.damage[-1].bias.fill_(u_bias_init)
            self.smoother.weight.zero_()
            self.smoother.bias.fill_(smoother_bias)
            self.occurrence.bias.fill_(occ_bias)
            self.background.bias.fill_(bkg_bias)

    def expand_damage_inputs(self, k: int):
        """Append k damage inputs with zero weights in every layer that reads x^U (first damage layer, smoother,
        background), so a model built with the base's input width keeps its initialisation and equals the base at
        step 0 (paired initialisation)."""
        for name in ("smoother", "background"):
            old = getattr(self, name)
            new = nn.Linear(old.in_features + k, old.out_features)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :old.in_features] = old.weight; new.bias.copy_(old.bias)
            setattr(self, name, new)
        old = self.damage[0]
        new = nn.Linear(old.in_features + k, old.out_features)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, :old.in_features] = old.weight; new.bias.copy_(old.bias)
        self.damage[0] = new

    # ------------------------------------------------- county-level slot (Amendment 5)
    def attach_level(self, d_ctx: int, source: str = "ctx"):
        """One linear term on the damage logit, constant over the window and zero at the start.

        `source` names the batch entry it reads ('ctx' = the county context variables, 'geo' = the
        geographic descriptors). Zero weight and bias make the arm identical to W at step 0.
        """
        if getattr(self, "level", None) is not None:
            raise RuntimeError("level term already attached")
        self.level_source = source
        self.level = nn.Linear(d_ctx, 1)
        with torch.no_grad():
            self.level.weight.zero_()
            self.level.bias.zero_()
        return self.level

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

    def attach_context_input(self, d_ctx: int):
        """County context into the first damage layer (PREREG Amendment 7): h = ReLU(W x + A c + b),
        A zero at the start, so the arm is W at step 0."""
        if getattr(self, "ctx_in", None) is not None:
            raise RuntimeError("context input already attached")
        self.ctx_in = nn.Linear(d_ctx, self.damage[0].out_features, bias=False)
        with torch.no_grad():
            self.ctx_in.weight.zero_()
        return self.ctx_in

    def attach_mechanisms(self, module: nn.Module, d_m: int):
        """Local geo-weather mechanism intensities into the first damage layer (geo_weather_20260924 DESIGN):
        h = ReLU(W x + A c + M z + b), z the standardised county intensities of `module`, M zero at the
        start, so the arm equals its base at step 0. `set_mechanism_scale` fixes the standardisation."""
        if getattr(self, "mech", None) is not None:
            raise RuntimeError("mechanisms already attached")
        self.mech = module
        self.mech_in = nn.Linear(d_m, self.damage[0].out_features, bias=False)
        with torch.no_grad():
            self.mech_in.weight.zero_()
        self.register_buffer("mech_mu", torch.zeros(d_m))
        self.register_buffer("mech_sd", torch.ones(d_m))
        return self.mech

    @torch.no_grad()
    def set_mechanism_scale(self, b: dict):
        lam = self.mech(b["nw"], b["na"], b["nr"])[:, ORIGIN:]
        self.mech_mu.copy_(lam.mean((0, 1)))
        self.mech_sd.copy_(lam.std((0, 1)) + 1e-6)

    def mechanism_inputs(self, b: dict) -> torch.Tensor:
        z = (self.mech(b["nw"], b["na"], b["nr"]) - self.mech_mu) / self.mech_sd
        z = torch.cat([torch.zeros_like(z[:, :ORIGIN]), z[:, ORIGIN:]], 1)    # only the rollout hours are read
        return z.clamp(-10.0, 10.0)

    def attach_hazard(self, d_phi: int):
        """Exposure-integrated hazard features as a competing hazard (geo_weather_20260924 DESIGN v1):
        u = cap (1 - (1 - u_host / cap) exp(-beta . phi_t)), beta >= 0 (projected after every step), zero at the
        start, so the arm equals its base at step 0 and d u / d beta = (cap - u_host) phi_t there."""
        if getattr(self, "haz_beta", None) is not None:
            raise RuntimeError("hazard already attached")
        self.haz_beta = nn.Parameter(torch.zeros(d_phi))
        return self.haz_beta

    def attach_hazard_slots(self, trig_idx, load_idx, slots: int):
        """Load x trigger slots on top of the linear hazard (DESIGN v1 section 4): slot r adds
        (a_r . phi_trig)(b_r . [1, phi_load]) with a_r, b_r >= 0; a = 0 and b = (1, 0, ..., 0) at the start, so the
        arm equals its base at step 0 and d/d a_r = phi_trig there (no saddle)."""
        assert self.haz_beta is not None, "attach_hazard first"
        self.register_buffer("haz_trig", torch.as_tensor(trig_idx, dtype=torch.long))
        self.register_buffer("haz_load", torch.as_tensor(load_idx, dtype=torch.long))
        self.haz_a = nn.Parameter(torch.zeros(slots, len(trig_idx)))
        b = torch.zeros(slots, 1 + len(load_idx)); b[:, 0] = 1.0
        self.haz_b = nn.Parameter(b)
        return self.haz_a, self.haz_b

    def hazard_params(self):
        return [p for p in (self.haz_beta, self.haz_a, self.haz_b) if p is not None]

    def hazard(self, phi: torch.Tensor) -> torch.Tensor:
        lam = phi @ self.haz_beta
        if self.haz_a is not None:
            trig = phi[..., self.haz_trig] @ self.haz_a.T                                  # [B, T, R]
            load = torch.cat([torch.ones_like(phi[..., :1]), phi[..., self.haz_load]], -1) @ self.haz_b.T
            lam = lam + (trig * load).sum(-1)
        return lam

    def hidden(self, xu: torch.Tensor, ctx: torch.Tensor | None = None, mech: torch.Tensor | None = None) -> torch.Tensor:
        a1 = self.damage[0](xu)
        if self.ctx_in is not None:
            a1 = a1 + self.ctx_in(ctx)[:, None, :]
        if self.mech_in is not None:
            a1 = a1 + self.mech_in(mech)
        return torch.relu(a1)

    # --------------------------------------------------------------------- forward
    def forward(self, b: dict, exit_open: bool = True, diagnostics: bool = False) -> dict:
        """b: xu [B,216,d_u], xr [B,216,d_r], xo [B,216,d_occ], geo [B,G], y0 [B]."""
        xu = b["xu"]
        mz = self.mechanism_inputs(b) if self.mech is not None else None
        h = self.hidden(xu, b.get("ctx"), mz)
        layer = self.damage[2]
        kd = {}
        if isinstance(layer, GCRKLayer):
            ans = layer(h, b["geo"], diagnostics=diagnostics, exit_open=exit_open)
            a2, kd = ans if diagnostics else (ans, {})
        else:
            a2 = layer(h)
        raw = self.damage[4](torch.relu(a2)).squeeze(-1)[:, ORIGIN:]
        if self.level is not None:
            raw = raw + self.level(b[self.level_source])
        forget = torch.sigmoid(self.smoother(xu[:, ORIGIN:])).squeeze(-1)
        logit = rate_inertia(raw.contiguous(), forget.contiguous())
        r = R_CAP * torch.sigmoid(self.recovery(b["xr"][:, ORIGIN:])).squeeze(-1)
        gate = torch.sigmoid(self.occurrence(b["xo"][:, ORIGIN:])).squeeze(-1)
        bkg = BKG_CAP * torch.sigmoid(self.background(xu[:, ORIGIN:])).squeeze(-1)
        cond = U_CAP * torch.sigmoid(logit)
        u = torch.clamp(gate * cond + bkg, 0.0, U_CAP + BKG_CAP)
        if self.haz_beta is not None:
            cap = U_CAP + BKG_CAP
            lam = self.hazard(b["phi"])                                          # [B, 144]
            if self.haz_signed:           # no clip on the coefficients: raise or lower the damage rate in logit space
                q = (u / cap).clamp(1e-6, 1 - 1e-6)
                u = cap * torch.sigmoid(torch.logit(q) + lam)
            else:
                u = u - (cap - u) * torch.expm1(-lam)                          # = cap - (cap - u) exp(-lam), lam >= 0
        p = stock_path(u.contiguous(), r.contiguous(), b["y0"].contiguous())
        out = dict(P=p, u=u, r=r, gate=gate, background=bkg, conditional=cond,
                   raw_logit=raw, logit=logit, forget=forget)
        if self.haz_beta is not None:
            out["hazard"] = lam
        if self.level is not None:
            out["level"] = self.level(b[self.level_source]).squeeze(-1)
        if mz is not None:
            out["mech_z"] = mz
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
