"""The competition-informed asymmetric scaffold, and the one thing that varies.

Both arms compute identical nonnegative proposals from identical modules:

    U_tilde = g(x_occ) * C_U * sigmoid(held_logit(x_u)) + C_bkg * sigmoid(bkg(x_u))
    R_tilde = C_R * sigmoid(w_r . x_r + b_r)          recomputed every 8 steps

and differ only at the output, where s = U_tilde - R_tilde:

    two_flow : U = U_tilde,   R = R_tilde
    one_flow : U = relu(s),   R = relu(-s)

so the one-flow arm removes exactly c = min(U_tilde, R_tilde) and nothing else. It
keeps the same interruption ensemble, hold, occurrence gate, background path,
recovery GLM, inputs and parameter budget. That is what makes the comparison a test
of flow separation rather than of capacity or of feature extraction.

State preservation is structural, not enforced. With C_U_main + C_bkg = 0.265 and
C_R = 0.25 the flows satisfy U + R <= 0.515 < 1, so

    Y_next = Y + U(1-Y) - R Y

maps [0,1] into itself in exact arithmetic. The clamp in `forward` is a fail-closed
assertion that counts activations; it is not part of the model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

CAP_U_MAIN = 0.25
CAP_U_BKG = 0.015
CAP_R = 0.25
RECOVERY_EVERY = 8          # forecast steps between recovery recomputations
assert CAP_U_MAIN + CAP_U_BKG + CAP_R < 1.0


def _mlp(d_in: int, hidden: int = 32) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))


class AsymmetricFlows(nn.Module):
    def __init__(self, d_u: int, d_occ: int, d_r: int, arm: str):
        super().__init__()
        assert arm in ("asym_two_flow", "asym_one_flow")
        self.arm = arm
        self.head_a = _mlp(d_u)          # two interruption magnitude heads,
        self.head_b = _mlp(d_u)          # averaged in logit space
        self.hold = nn.Linear(d_u, 1)    # learned first-order hold gate
        self.occ = nn.Linear(d_occ, 1)   # occurrence gate, separate narrow input
        self.bkg = nn.Linear(d_u, 1)     # independent low-capacity background path
        self.rec = nn.Linear(d_r, 1)     # recovery GLM
        self.clamp_events = 0

    # ---------------- proposals, shared by both arms ------------------------
    def proposals(self, x_u_t, x_occ_t, x_r_t, held_prev, r_prev, step):
        raw = 0.5 * (self.head_a(x_u_t) + self.head_b(x_u_t)).squeeze(-1)
        q = torch.sigmoid(self.hold(x_u_t).squeeze(-1))
        held = raw if held_prev is None else q * held_prev + (1.0 - q) * raw
        g = torch.sigmoid(self.occ(x_occ_t).squeeze(-1))
        u_tilde = (g * CAP_U_MAIN * torch.sigmoid(held)
                   + CAP_U_BKG * torch.sigmoid(self.bkg(x_u_t).squeeze(-1)))
        if step % RECOVERY_EVERY == 0 or r_prev is None:
            r_tilde = CAP_R * torch.sigmoid(self.rec(x_r_t).squeeze(-1))
        else:
            r_tilde = r_prev                       # held between recomputations
        return u_tilde, r_tilde, held, g, q

    @staticmethod
    def collapse(arm: str, u_tilde, r_tilde):
        if arm == "asym_two_flow":
            return u_tilde, r_tilde
        s = u_tilde - r_tilde
        return torch.relu(s), torch.relu(-s)

    # ---------------- one teacher-forced transition -------------------------
    def step_from_state(self, y, x_u_t, x_occ_t, x_r_t):
        u_t, r_t, _, _, _ = self.proposals(x_u_t, x_occ_t, x_r_t, None, None, 0)
        U, R = self.collapse(self.arm, u_t, r_t)
        return y + U * (1.0 - y) - R * y

    # ---------------- open-loop rollout -------------------------------------
    def forward(self, y0, x_u, x_occ, x_r, collect=False):
        y = y0
        held = r_prev = None
        out, diag = [], []
        for t in range(x_u.shape[1]):
            u_t, r_t, held, g, q = self.proposals(
                x_u[:, t], x_occ[:, t], x_r[:, t], held, r_prev, t)
            r_prev = r_t
            U, R = self.collapse(self.arm, u_t, r_t)
            y_next = y + U * (1.0 - y) - R * y
            bad = ((y_next < -1e-9) | (y_next > 1 + 1e-9))
            if bool(bad.any()):
                self.clamp_events += int(bad.sum())
            y = torch.clamp(y_next, 0.0, 1.0)      # fail-closed assertion only
            out.append(y)
            if collect:
                diag.append(dict(u=float(u_t.mean()), r=float(r_t.mean()),
                                 gate=float(g.mean()), hold=float(q.mean()),
                                 U=float(U.mean()), R=float(R.mean()),
                                 both_active=float(((u_t > 0) & (r_t > 0)).float().mean()),
                                 common=float(torch.minimum(u_t, r_t).mean())))
        return torch.stack(out, 1), diag

    # ---------------- exact constant-class initialisation -------------------
    def apply_modular_init(self, spec: dict):
        with torch.no_grad():
            for m in (self.head_a, self.head_b):
                for lin in [l for l in m if isinstance(l, nn.Linear)]:
                    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
                [l for l in m if isinstance(l, nn.Linear)][-1].bias.fill_(spec["raw_u_bias"])
            for lin, b in ((self.hold, spec["hold_bias"]), (self.occ, 0.0),
                           (self.bkg, spec["background_bias"]), (self.rec, spec["recovery_bias"])):
                nn.init.zeros_(lin.weight); lin.bias.fill_(b)

    def constant_flows(self, device="cpu"):
        """The (U_tilde, R_tilde) this network emits when every input is zero."""
        with torch.no_grad():
            z_u = torch.zeros(1, self.hold.in_features, device=device)
            z_o = torch.zeros(1, self.occ.in_features, device=device)
            z_r = torch.zeros(1, self.rec.in_features, device=device)
            u, r, _, _, _ = self.proposals(z_u, z_o, z_r, None, None, 0)
        return float(u), float(r)
