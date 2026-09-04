"""Two-stage, update-budgeted training with update 0 as a checkpoint candidate.

Budgets are counted in optimizer updates, not epochs, because the competition
evidence makes "it was not trained enough" an unacceptable explanation for a null.

Stage A pretrains the conditional transition on unique adjacent hourly pairs under
teacher forcing. Stage B fine-tunes the 24-hour open-loop rollout from the fixed
event-centred origins. Both arms receive identical optimiser, learning rate, batch
size, clipping, budget, sampler and validation rule; only the output collapse
differs.

Active-state oversampling is importance corrected. Minibatches are drawn half from
Y<=0.01 and half from Y>0.01 where both pools exist, but each sampled transition is
reweighted by the inverse of its sampling probability and renormalised within the
event, so the optimised objective remains the natural equal-event transition risk
rather than an equal-stratum risk.
"""
from __future__ import annotations

import copy, time
from dataclasses import dataclass, field

import numpy as np
import torch

LR = 3e-3
BATCH_EVENTS = 4                 # source events per update
BATCH_ROWS = 256                 # transitions per event per Stage-A update
CLIP = 5.0
STAGE_A_UPDATES = 1600
STAGE_B_UPDATES = 3200
VAL_EVERY = 200
STAGE_B_MIN = 1600
PATIENCE_CHECKS = 6
ACTIVE_THRESHOLD = 0.01


@dataclass
class Diag:
    rows: list = field(default_factory=list)

    def add(self, **kw):
        self.rows.append(kw)


def _grad_norms(model) -> dict:
    out = {}
    for name, mod in (("head_a", model.head_a), ("head_b", model.head_b),
                      ("hold", model.hold), ("occ", model.occ),
                      ("bkg", model.bkg), ("rec", model.rec)):
        s = 0.0
        for p in mod.parameters():
            if p.grad is not None:
                s += float(p.grad.detach().pow(2).sum())
        out[f"gn_{name}"] = s ** 0.5
    return out


def stage_a_batch(rng, events, ev_data, n_events, n_rows):
    """Event-balanced, state-stratified, importance-corrected transition batch."""
    picks = rng.choice(len(events), size=min(n_events, len(events)), replace=False)
    Y, DY, XU, XO, XR, W, fell_back = [], [], [], [], [], [], 0
    for i in picks:
        d = ev_data[events[i]]
        y, dy = d["tf_y"], d["tf_dy"]
        act = np.where(y > ACTIVE_THRESHOLD)[0]
        qui = np.where(y <= ACTIVE_THRESHOLD)[0]
        half = n_rows // 2
        if len(act) == 0 or len(qui) == 0:          # transparent fallback
            fell_back += 1
            idx = rng.choice(len(y), size=min(n_rows, len(y)), replace=False)
            w = np.ones(len(idx))
        else:
            ia = rng.choice(act, size=min(half, len(act)), replace=len(act) < half)
            iq = rng.choice(qui, size=min(n_rows - half, len(qui)),
                            replace=len(qui) < n_rows - half)
            idx = np.concatenate([ia, iq])
            # inverse sampling probability: stratum z drawn with prob p_z per draw,
            # while its natural share is n_z / n. weight = natural / sampling.
            n = len(y)
            p_a = (len(ia) / len(idx)) / max(len(act), 1)
            p_q = (len(iq) / len(idx)) / max(len(qui), 1)
            w = np.where(y[idx] > ACTIVE_THRESHOLD, (1.0 / n) / p_a, (1.0 / n) / p_q)
        w = w / w.sum()                              # renormalise within the event
        Y.append(y[idx]); DY.append(dy[idx]); W.append(w)
        XU.append(d["tf_xu"][idx]); XO.append(d["tf_xo"][idx]); XR.append(d["tf_xr"][idx])
    return (np.concatenate(Y), np.concatenate(DY), np.concatenate(XU),
            np.concatenate(XO), np.concatenate(XR),
            np.concatenate([w / len(picks) for w in W]), fell_back)


def _tf_loss(model, y, dy, xu, xo, xr, w):
    pred = model.step_from_state(y, xu, xo, xr)
    return (w * (pred - (y + dy)) ** 2).sum()


def _roll_loss(model, y0, xu, xo, xr, yt, m):
    pred, _ = model(y0, xu, xo, xr)
    return ((pred - yt) ** 2 * m).sum() / m.sum().clamp_min(1.0)


def train(model, arm, events, ev_data, val_fn, seed, diag: Diag, label: str):
    """Stage A then Stage B. Returns the selected state dict and a record."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)

    best = val_fn(model)                       # update 0 IS a checkpoint candidate
    best_state, best_update, best_stage = copy.deepcopy(model.state_dict()), 0, "update0"
    diag.add(job=label, stage="A", update=0, validation=best, selected=True)
    t0 = time.time()
    seen = 0
    fallbacks = 0

    for u in range(1, STAGE_A_UPDATES + 1):
        y, dy, xu, xo, xr, w, fb = stage_a_batch(rng, events, ev_data, BATCH_EVENTS, BATCH_ROWS)
        fallbacks += fb; seen += len(y)
        opt.zero_grad(set_to_none=True)
        _tf_loss(model, T(y), T(dy), T(xu), T(xo), T(xr), T(w)).backward()
        gn = _grad_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if u % VAL_EVERY == 0:
            v = val_fn(model)
            sel = v < best - 1e-12
            if sel:
                best, best_state, best_update, best_stage = v, copy.deepcopy(model.state_dict()), u, "A"
            diag.add(job=label, stage="A", update=u, validation=v, selected=sel,
                     examples=seen, **gn)

    model.load_state_dict(best_state)          # Stage B starts from the Stage-A pick
    optB = torch.optim.Adam(model.parameters(), lr=LR)
    bestB, bestB_state, bestB_update = val_fn(model), copy.deepcopy(model.state_dict()), 0
    bad = 0
    for u in range(1, STAGE_B_UPDATES + 1):
        picks = rng.choice(len(events), size=min(BATCH_EVENTS, len(events)), replace=False)
        optB.zero_grad(set_to_none=True)
        tot = 0.0
        for i in picks:
            d = ev_data[events[i]]
            k = int(rng.integers(len(d["roll_y0"])))
            loss = _roll_loss(model, T(d["roll_y0"][k]), T(d["roll_xu"][k]),
                              T(d["roll_xo"][k]), T(d["roll_xr"][k]),
                              T(d["roll_yt"][k]), T(d["roll_m"][k].astype(np.float32)))
            (loss / len(picks)).backward()
            tot += float(loss.detach())
        gn = _grad_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optB.step()
        if u % VAL_EVERY == 0:
            v = val_fn(model)
            sel = v < bestB - 1e-12
            if sel:
                bestB, bestB_state, bestB_update, bad = v, copy.deepcopy(model.state_dict()), u, 0
            else:
                bad += 1
            diag.add(job=label, stage="B", update=u, validation=v, selected=sel,
                     train_loss=tot / len(picks), **gn)
            if u >= STAGE_B_MIN and bad >= PATIENCE_CHECKS:
                break
    if bestB <= best:
        model.load_state_dict(bestB_state)
        sel_stage, sel_update, sel_val = ("B", bestB_update, bestB) if bestB_update else ("update0", 0, bestB)
    else:
        model.load_state_dict(best_state)
        sel_stage, sel_update, sel_val = best_stage, best_update, best
    return dict(selected_stage=sel_stage, selected_update=sel_update,
                selected_validation=sel_val, stage_a_best=best,
                stage_b_best=bestB, examples_stage_a=seen,
                stratum_fallbacks=fallbacks, wall_s=round(time.time() - t0, 1),
                clamp_events=int(model.clamp_events))
