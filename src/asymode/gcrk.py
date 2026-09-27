"""Geography-Conditioned Response Kernel (GCRK).

GCRK sits inside the damage network, on its first hidden representation h_{i,t}.
It keeps one bounded response state per county and adds it to h before the
network's own second linear layer:

    b_t  = h_0                                   t = 0
         = mean(h_0 .. h_{min(t,72)-1})          t >= 1   (fixed from hour 72 on)
    v_t  = h_t - b_t
    q_t  = v_t / sqrt(c^2 + |v_t|^2)
    d_t  = sigmoid(4 (|v_t| / c - theta)) q_t                  |d_t| < 1
    z_i  = tanh(U g~_i)                                        4-dim geographic code
    lam_i = lam_lo + (lam_hi - lam_lo) sigmoid(l0 + V_l z_i)   coordinatewise dissipation
    a_i  = (1/2) w / sqrt(1 + |w|^2),  w = a0 + V_a z_i        |a_i| < 1/2
    om_i = exp(log 2 tanh(g0 + V_g z_i))                       readout gain in (1/2, 2)
    nu_i = min_j lam_ij
    [I + D_i - nu_i S_i(d_t)] e_t = e_{t-1} + nu_i d_t,   S(d) = a d^T - d a^T,  e_{-1} = 0
    a2_t = W2 (h_t + beta_s c Om_i e_t) + b2,   beta_s = min(1, s/200) tanh(alpha)

c is the median of |v_t| over the fitting counties (hours 1..215), floored at
1e-4, and theta is their 80th percentile divided by c. Both are buffers refreshed
from the current fitting-county hidden sequence, never optimised and never
updated from held-out counties. During training the whole response contribution
is dropped with probability 0.2 (no inverse-probability rescaling).

The linear solve is diagonal plus rank two, so each county-hour costs O(d). The
backward pass is the exact adjoint of the recurrence, written out below and
checked against autograd through the plain loop in tests/.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

PREFIX_HOURS = 72
CODE_DIM = 4
MEMORY_MIN_H, MEMORY_MAX_H, MEMORY_INIT_H = 2.0, 96.0, 16.0
INTERACTION_BOUND = 0.5
GAIN_LOG_BOUND = math.log(2.0)
GATE_QUANTILE, GATE_SLOPE = 0.8, 4.0
SCALE_FLOOR = 1e-4
DROP_PATH = 0.2
WARMUP_STEPS = 200
CALIBRATION_EVERY = 10
KAPPA_MAX = 0.5          # spatio-temporal GCRK: each of the static and the upwind neighbour shares is at most 1/2


def prefix_reference(h: torch.Tensor, prefix: int = PREFIX_HOURS) -> torch.Tensor:
    """Strictly-past expanding mean of h, frozen once `prefix` hours have been seen.

    b_0 = h_0; b_t = mean(h_0..h_{t-1}) for 1 <= t <= prefix; b_t = b_prefix after.
    """
    if h.ndim != 3 or h.shape[1] == 0:
        raise ValueError("expected a nonempty [B, T, D] hidden sequence")
    T = h.shape[1]
    n = min(T, prefix)
    cs = h[:, :n].cumsum(1)
    idx = torch.arange(T, device=h.device).clamp(1, n) - 1
    den = torch.arange(T, device=h.device, dtype=h.dtype).clamp(1, n).view(1, -1, 1)
    return cs[:, idx] / den


# ----------------------------------------------------------------------------- solver
def _solve(b: torch.Tensor, f: torch.Tensor, invd: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Solve (H - a f^T + f a^T) x = b for every row, H = diag(1/invd).

    Shapes [B, D]. Writing s_f = f.x and s_a = a.x reduces the system to 2 x 2;
    its determinant 1 + (a'H^-1 a)(f'H^-1 f) - (a'H^-1 f)^2 is >= 1.
    """
    y = b * invd
    u = a * invd
    v = f * invd
    aa = (a * u).sum(-1, keepdim=True)
    ff = (f * v).sum(-1, keepdim=True)
    af = (a * v).sum(-1, keepdim=True)
    fy = (f * y).sum(-1, keepdim=True)
    ay = (a * y).sum(-1, keepdim=True)
    det = 1.0 + aa * ff - af * af
    s_f = ((1.0 + af) * fy - ff * ay) / det
    s_a = (aa * fy + (1.0 - af) * ay) / det
    return y + u * s_f - v * s_a


def response_step(prev: torch.Tensor, f: torch.Tensor, lam: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """One hour of (I + D - S(f)) e = prev + f, with S(f) = a f^T - f a^T."""
    return _solve(prev + f, f, 1.0 / (1.0 + lam), a)


def response_sequence_loop(f: torch.Tensor, lam: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Plain recurrence, differentiated by autograd. Reference for tests."""
    state = torch.zeros_like(f[:, 0])
    out = []
    for t in range(f.shape[1]):
        state = response_step(state, f[:, t], lam, a)
        out.append(state)
    return torch.stack(out, 1)


class _Response(torch.autograd.Function):
    """Same recurrence with a hand-written adjoint (no per-hour autograd graph).

    For e_t = M_t^{-1}(e_{t-1} + f_t), M_t = H - a f_t^T + f_t a^T and upstream
    gradient G_t, the adjoint z_t = M_t^{-T}(G_t + z_{t+1}) gives
        dL/df_t  = (1 - a.e_t) z_t + (a.z_t) e_t
        dL/dlam  = -sum_t z_t * e_t
        dL/da    = sum_t (f_t.e_t) z_t - (z_t.f_t) e_t .
    M_t^T has the same form with a -> -a, so the forward solver is reused.
    """

    @staticmethod
    def forward(ctx, f, lam, a):
        invd = 1.0 / (1.0 + lam)
        state = torch.zeros_like(f[:, 0])
        out = torch.empty_like(f)
        for t in range(f.shape[1]):
            state = _solve(state + f[:, t], f[:, t], invd, a)
            out[:, t] = state
        ctx.save_for_backward(f, lam, a, out)
        return out

    @staticmethod
    def backward(ctx, grad):
        f, lam, a, e = ctx.saved_tensors
        invd = 1.0 / (1.0 + lam)
        gf = torch.empty_like(f)
        glam = torch.zeros_like(lam)
        ga = torch.zeros_like(a)
        adj = torch.zeros_like(f[:, 0])
        grad = grad.contiguous()
        for t in range(f.shape[1] - 1, -1, -1):
            ft, et = f[:, t], e[:, t]
            z = _solve(grad[:, t] + adj, ft, invd, -a)
            ae = (a * et).sum(-1, keepdim=True)
            az = (a * z).sum(-1, keepdim=True)
            fe = (ft * et).sum(-1, keepdim=True)
            zf = (z * ft).sum(-1, keepdim=True)
            gf[:, t] = (1.0 - ae) * z + az * et
            glam -= z * et
            ga += fe * z - zf * et
            adj = z
        return gf, glam, ga


def response_sequence(f: torch.Tensor, lam: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Response state for a whole causal forcing sequence f [B, T, D]."""
    if f.ndim != 3 or f.shape[1] == 0:
        raise ValueError("expected nonempty forcing of shape [B, T, D]")
    if lam.shape != (f.shape[0], f.shape[2]) or a.shape != lam.shape:
        raise ValueError("dissipation and interaction must have shape [B, D]")
    return _Response.apply(f.contiguous(), lam.contiguous(), a.contiguous())


# ------------------------------------------------------------- spatio-temporal coupling
# The states of neighbouring counties are mixed before each hour's solve:
#     e_t = M_t^{-1} (P_t e_{t-1} + f_t),   (P_t e)_i = (1 - sum_k c_ik,t) e_i + sum_k c_ik,t e_nbr(i,k),
# with c_ik,t = cw_ik + cu_ik,t >= 0 and sum_k c_ik,t <= 1. P_t is a Markov mixing of neighbour states, so the largest
# county state norm cannot grow through it and GCRK's unit bound on the state still holds. cw is the static,
# symmetric-in-spirit share (distance x geography similarity); cu the directional share from upwind neighbours.
def _mix(prev: torch.Tensor, c: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
    """prev [B, D], c [B, k] >= 0 with row sums <= 1, nbr [B, k] row indices into prev."""
    return prev * (1.0 - c.sum(-1, keepdim=True)) + (c.unsqueeze(-1) * prev[nbr]).sum(1)


def response_sequence_st_loop(f, lam, a, cw, cu, nbr):
    """Plain spatio-temporal recurrence, differentiated by autograd. Reference for tests."""
    invd = 1.0 / (1.0 + lam)
    state = torch.zeros_like(f[:, 0])
    out = []
    for t in range(f.shape[1]):
        state = _solve(_mix(state, cw + cu[:, t], nbr) + f[:, t], f[:, t], invd, a)
        out.append(state)
    return torch.stack(out, 1)


class _Pattern:
    """Fixed sparsity of the neighbour mixing, P and P^T as CSR with per-hour values (the pattern is built once)."""

    def __init__(self, nbr: torch.Tensor):
        B, k = nbr.shape
        self.B, self.k = B, k
        self.flat = nbr.reshape(-1)
        rows = torch.arange(B, device=nbr.device).repeat_interleave(k)
        self.crow = torch.arange(0, B * k + 1, k, device=nbr.device)
        self.order = torch.argsort(nbr, dim=1)                                   # columns sorted within each row
        self.col = torch.gather(nbr, 1, self.order).reshape(-1)
        key = self.flat * B + rows                                                # transposed rows = neighbours
        self.tperm = torch.argsort(key)
        self.tcol = rows[self.tperm]
        counts = torch.bincount(self.flat, minlength=B)
        self.tcrow = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    def P(self, c: torch.Tensor) -> torch.Tensor:
        return torch.sparse_csr_tensor(self.crow, self.col, torch.gather(c, 1, self.order).reshape(-1), (self.B, self.B))

    def PT(self, c: torch.Tensor) -> torch.Tensor:
        return torch.sparse_csr_tensor(self.tcrow, self.tcol, c.reshape(-1)[self.tperm], (self.B, self.B))


class _ResponseST(torch.autograd.Function):
    """The coupled recurrence with a hand-written adjoint. With z_t = M_t^{-T}(G_t + P_{t+1}^T z_{t+1}), the
    gradients of f, lam and a are those of _Response, and dL/dc_ik,t = z_i,t . (e_nbr(i,k),t-1 - e_i,t-1)."""

    @staticmethod
    def forward(ctx, f, lam, a, cw, cu, nbr):
        invd = 1.0 / (1.0 + lam)
        pat = _Pattern(nbr)
        state = torch.zeros_like(f[:, 0])
        out = torch.empty_like(f)
        for t in range(f.shape[1]):
            c = cw + cu[:, t]
            mixed = state * (1.0 - c.sum(-1, keepdim=True)) + pat.P(c) @ state
            state = _solve(mixed + f[:, t], f[:, t], invd, a)
            out[:, t] = state
        ctx.save_for_backward(f, lam, a, cw, cu, nbr, out)
        return out

    @staticmethod
    def backward(ctx, grad):
        f, lam, a, cw, cu, nbr, e = ctx.saved_tensors
        invd = 1.0 / (1.0 + lam)
        pat = _Pattern(nbr)
        B, k, D = nbr.shape[0], nbr.shape[1], f.shape[-1]
        gf = torch.empty_like(f)
        glam = torch.zeros_like(lam)
        ga = torch.zeros_like(a)
        gcu = torch.zeros_like(cu)
        adj = torch.zeros_like(f[:, 0])
        grad = grad.contiguous()
        for t in range(f.shape[1] - 1, -1, -1):
            ft, et = f[:, t], e[:, t]
            z = _solve(grad[:, t] + adj, ft, invd, -a)
            ae = (a * et).sum(-1, keepdim=True)
            az = (a * z).sum(-1, keepdim=True)
            fe = (ft * et).sum(-1, keepdim=True)
            zf = (z * ft).sum(-1, keepdim=True)
            gf[:, t] = (1.0 - ae) * z + az * et
            glam -= z * et
            ga += fe * z - zf * et
            c = cw + cu[:, t]
            if t > 0:
                prev = e[:, t - 1]
                pn = prev.index_select(0, pat.flat).view(B, k, D)
                gcu[:, t] = torch.bmm(pn, z.unsqueeze(-1)).squeeze(-1) - (z * prev).sum(-1, keepdim=True)
            adj = z * (1.0 - c.sum(-1, keepdim=True)) + pat.PT(c) @ z
        return gf, glam, ga, gcu.sum(1), gcu, None


def response_sequence_st(f, lam, a, cw, cu, nbr):
    """Spatio-temporal response states. f [B, T, D]; lam, a [B, D]; cw [B, k]; cu [B, T, k]; nbr [B, k] (long)."""
    B, T, D = f.shape
    if cw.shape != nbr.shape or cu.shape != (B, T, nbr.shape[1]) or nbr.shape[0] != B:
        raise ValueError("coupling must be cw [B, k], cu [B, T, k] and nbr [B, k]")
    return _ResponseST.apply(f.contiguous(), lam.contiguous(), a.contiguous(), cw.contiguous(), cu.contiguous(),
                             nbr.contiguous())


# ------------------------------------------------------------------------------ layer
class GCRKLayer(nn.Module):
    """Replaces the damage network's second linear layer, reusing its W2 and b2.

    `center` is the fitting-county mean of tanh(g/3) for the standardised
    geography g; it centres the geographic code so that an average county sits at
    the origin. `private_seed` fixes the kernel's own initialisation and its
    drop-path stream independently of the host's seed.
    """

    def __init__(self, original: nn.Linear, center: torch.Tensor, private_seed: int = 1729):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError("GCRKLayer wraps an existing nn.Linear")
        self.weight, self.bias = original.weight, original.bias
        self.in_features, self.out_features = original.in_features, original.out_features
        d, r, G = self.in_features, CODE_DIM, int(center.numel())
        w = original.weight
        self.register_buffer("geo_center", center.detach().clone().to(w.dtype))
        self.register_buffer("scale", w.new_tensor(1.0))
        self.register_buffer("threshold", w.new_tensor(1.5))
        self.register_buffer("calibration_step", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))
        # initial values are drawn and scaled in float32, then cast, whatever the host dtype
        gen = torch.Generator().manual_seed(int(private_seed))
        self.U = nn.Parameter((torch.randn(r, G, generator=gen) / math.sqrt(G)).to(w.dtype))
        self.Vl = nn.Parameter(w.new_zeros(d, r))
        self.Va = nn.Parameter(w.new_zeros(d, r))
        self.Vg = nn.Parameter(w.new_zeros(d, r))
        self.lambda_min = math.expm1(1.0 / MEMORY_MAX_H)
        self.lambda_max = math.expm1(1.0 / MEMORY_MIN_H)
        p = (math.expm1(1.0 / MEMORY_INIT_H) - self.lambda_min) / (self.lambda_max - self.lambda_min)
        self.l0 = nn.Parameter(torch.full((d,), math.log(p / (1.0 - p))).to(w.dtype))
        a0 = torch.randn(d, generator=gen)
        self.a0 = nn.Parameter((INTERACTION_BOUND * a0 / a0.norm()).to(w.dtype))
        self.g0 = nn.Parameter(w.new_zeros(()))
        self.alpha = nn.Parameter(w.new_zeros(()))
        self.bounded_opening = True       # beta = tanh(alpha); False: beta = alpha (no bound on the opening)
        self._drop = torch.Generator().manual_seed(int(private_seed) + 1)
        self.last_mask = 1.0
        self.last_calibration: dict = {}
        self.space_on = False

    def attach_space(self):
        """Spatio-temporal GCRK: couple the response states of neighbouring counties (_ResponseST). Three scalars,
        all zero at the start, so the layer equals GCRK at step 0: kappa_s, the static share taken from the
        neighbours (weights exp(-distance / 50 km) x exp(-gamma |code_i - code_j|^2), normalised), kappa_a, the share
        taken from upwind neighbours (weights from the data, scaled by wind speed), and gamma >= 0, how strongly
        geographic similarity selects the neighbours. project_() keeps them in their ranges after every step."""
        w = self.weight
        self.kappa_s = nn.Parameter(w.new_zeros(()))
        self.kappa_a = nn.Parameter(w.new_zeros(()))
        self.geo_sim = nn.Parameter(w.new_zeros(()))
        self.space_on = True

    @torch.no_grad()
    def project_(self):
        if self.space_on:
            self.kappa_s.clamp_(0.0, KAPPA_MAX); self.kappa_a.clamp_(0.0, KAPPA_MAX); self.geo_sim.clamp_(min=0.0)

    def code_of(self, g: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(g / 3.0) - self.geo_center
        z = z / torch.sqrt(0.1 ** 2 + z.square().sum(-1, keepdim=True))
        return torch.tanh(F.linear(z, self.U))

    def coupling(self, code: torch.Tensor, space: dict):
        """(cw [B, k], cu [B, T, k]) from the batch's neighbour table space = {nbr, wd, up}."""
        nbr, wd, up = space["nbr"], space["wd"], space["up"]
        sim = torch.exp(-self.geo_sim.clamp(min=0.0) * (code.unsqueeze(1) - code[nbr]).square().sum(-1))
        w = wd * sim
        wn = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
        return self.kappa_s.clamp(0.0, KAPPA_MAX) * wn, self.kappa_a.clamp(0.0, KAPPA_MAX) * up

    # geography -> (dissipation, interaction, readout gain)
    def condition(self, g: torch.Tensor):
        code = self.code_of(g)
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(self.l0 + F.linear(code, self.Vl))
        w = self.a0 + F.linear(code, self.Va)
        a = INTERACTION_BOUND * w / torch.sqrt(1.0 + w.square().sum(-1, keepdim=True))
        gain = torch.exp(GAIN_LOG_BOUND * torch.tanh(self.g0 + F.linear(code, self.Vg)))
        return lam, a, gain

    @torch.no_grad()
    def calibrate_(self, h: torch.Tensor, step: int) -> dict:
        """Refresh c and theta from fitting-county hidden sequences only."""
        n = torch.linalg.vector_norm(h - prefix_reference(h), dim=-1)
        values = n[:, 1:].reshape(-1)          # hour 0 has zero departure by construction
        c = values.median().clamp_min(SCALE_FLOOR)
        theta = torch.quantile(values, GATE_QUANTILE) / c
        self.scale.copy_(c)
        self.threshold.copy_(theta)
        self.calibration_step.fill_(int(step))
        self.last_calibration = dict(step=int(step), scale=float(c), theta=float(theta),
                                     n_counties=int(h.shape[0]), n_values=int(values.numel()))
        return dict(self.last_calibration)

    def ramp(self) -> float:
        return min(1.0, float(self.training_step) / WARMUP_STEPS)

    def forward(self, h: torch.Tensor, g: torch.Tensor, diagnostics: bool = False,
                exit_open: bool = True, space: dict | None = None):
        """h: [B, T, d] first hidden sequence; g: [B, G] standardised geography; space: the batch's neighbour table
        (spatio-temporal GCRK only).

        exit_open=False evaluates the same network with the response contribution
        removed, which is the configuration drop-path exposes during training.
        """
        if int(self.calibration_step) < 0:
            raise RuntimeError("calibrate_ on fitting counties before the first forward")
        ref = prefix_reference(h)
        v = h - ref
        n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        q = v / torch.sqrt(self.scale.square() + n.square())
        gate = torch.sigmoid(GATE_SLOPE * (n / self.scale - self.threshold))
        d = gate * q
        lam, a, gain = self.condition(g)
        nu = lam.amin(dim=-1, keepdim=True)
        if self.space_on:
            if space is None:
                raise RuntimeError("spatio-temporal GCRK needs the batch's neighbour table")
            cw, cu = self.coupling(self.code_of(g), space)
            state = response_sequence_st(nu[:, None] * d, lam, a, cw, cu, space["nbr"])
        else:
            state = response_sequence(nu[:, None] * d, lam, a)
        beta = torch.tanh(self.alpha) if self.bounded_opening else self.alpha
        mask = 1.0
        if self.training:
            mask = float(torch.rand((), generator=self._drop) >= DROP_PATH)
        self.last_mask = mask
        effect = self.ramp() * beta * self.scale * gain[:, None] * state
        out = F.linear(h + (mask if exit_open else 0.0) * effect, self.weight, self.bias)
        if diagnostics:
            return out, dict(reference=ref, departure=v, q=q, gate=gate.squeeze(-1), deposit=d,
                             state=state, damping=lam, nu=nu, interaction=a, coordinate_gain=gain,
                             beta=beta, scale=self.scale, threshold=self.threshold,
                             effect=effect, ramp=h.new_tensor(self.ramp()),
                             **({"kappa_s": self.kappa_s.detach(), "kappa_a": self.kappa_a.detach(),
                                 "geo_sim": self.geo_sim.detach()} if self.space_on else {}))
        return out

    def n_new_parameters(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if n not in ("weight", "bias"))
