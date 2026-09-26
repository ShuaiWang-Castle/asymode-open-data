"""Phase-2 control for B4': a two-flow recursion whose rates may read the recursive state.

ASYM_STATE keeps the two-flow form y[k+1] = y[k] + u (1 - y[k]) - r y[k] with (u, r, stay) = softmax(D, R, 0), but the
damage and recovery networks read (c, x[k], y[k] / scale). With state-reading rates the recursion can represent any
interior bounded one-step map (u = T, r = 1 - T), so comparing ASYM, ASYM_STATE and NET separates the effect of
restricting state access from the effect of the two-flow parameterisation. The public NET and ASYM classes are
imported unchanged; parameter counts are matched with the same rule (within 1% of the target).
"""
from __future__ import annotations
import math
import torch
from torch import nn
import models_v2 as MV  # public module, imported read-only


class AsymStateModel(nn.Module):
    def __init__(self, context_dim, clock_dim, horizon, structure, target, state_scale, source_mean, eps=MV.ASYM_INIT_EPS):
        super().__init__()
        self.kind, self.structure, self.horizon, self.clock_dim = 'ASYM_STATE', structure, horizon, clock_dim
        din = context_dim + clock_dim + 1
        self.width_damage, self.width_recovery = MV.match_split(din, MV.ASYM_RATIOS[structure], target)
        self.damage = MV.ResidualMLPn(din, self.width_damage, 1, 2)
        self.recovery = MV.ResidualMLPn(din, self.width_recovery, 1, 2)
        self.register_buffer('state_scale', torch.tensor(float(state_scale)))
        pi = min(max(float(source_mean), 1e-4), 1 - 1e-4)
        u, r = eps * pi, eps * (1 - pi); stay = 1.0 - u - r
        with torch.no_grad():
            self.damage.body[-1].bias.fill_(math.log(u / stay))
            self.recovery.body[-1].bias.fill_(math.log(r / stay))

    def widths(self): return {'damage': self.width_damage, 'recovery': self.width_recovery}

    def forward(self, c, y0, known_clock):
        if c.ndim != 2 or y0.shape != (c.shape[0],) or known_clock.shape != (c.shape[0], self.horizon, self.clock_dim):
            raise ValueError('expected c[B,D], y0[B], known_clock[B,H,clock_dim]')
        y, out = y0, []
        for k in range(self.horizon):
            x = torch.cat([c, known_clock[:, k], y[:, None] / self.state_scale], dim=-1)
            logits = torch.cat([self.damage(x), self.recovery(x), torch.zeros_like(x[..., :1])], dim=-1)
            w = torch.softmax(logits, dim=-1)
            y = y + w[..., 0] * (1 - y) - w[..., 1] * y
            out.append(y)
        return torch.stack(out, dim=1)


def build_model_ext(kind, structure, context_dim, clock_dim, horizon, target, state_scale, source_mean):
    if kind == 'ASYM_STATE':
        return AsymStateModel(context_dim, clock_dim, horizon, structure, target, state_scale, source_mean)
    return MV.build_model(kind, structure, context_dim, clock_dim, horizon, target, state_scale, source_mean)
