"""Training protocol for W and GCRK: pooled INNER early stopping, REFIT from scratch.

  * Adam, two groups: damage (and kernel) at 3e-3, recovery at 3e-4.
  * INNER: three county-grouped inner folds are trained in lockstep; every 10 steps
    each is evaluated on its held-out inner counties and the pooled trajectory loss
    decides. The search runs at least 400 steps, stops 200 steps after the best
    evaluation, and never beyond 1600 steps. Step 0 is evaluated too.
  * REFIT: a fresh model with the same initialisation seed is trained on all
    development units for exactly the selected number of steps t*, then OUTER is
    exported in evaluation mode.
  * GCRK only: the kernel's calibration buffers are refreshed from the current
    fitting-set hidden sequences every 10 steps and before every evaluation, the
    response contribution opens linearly over 200 steps, and a training-only
    drop-path removes it with probability 0.2 (asymode.gcrk).
  * Training loss = selection loss = masked MSE of the outage fraction over the
    rollout hours 72..215 (observed cells only).
Standardisation statistics are computed on each model's own fitting units.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from .asym_host import AsymODE, masked_se, trajectory_loss
from .geo_mech import MECH, LocalMechanisms

POLICY = dict(min_steps=400, patience_steps=200, max_steps=1600, eval_every=10)
LR_HOST, LR_RECOVERY = 3e-3, 3e-4
N_WEATHER = 14          # leading columns of x^U and x^R that are raw weather (not clipped)
N_STATIC = 6            # x^R columns after the weather block: county background (not clipped)
CLIP = 5.0


class Rule:
    def __init__(self, min_steps=400, patience_steps=200, max_steps=1600, eval_every=10):
        self.policy = dict(min_steps=min_steps, patience_steps=patience_steps, max_steps=max_steps,
                           eval_every=eval_every)
        self.best, self.best_step, self.step = float("inf"), 0, -eval_every

    def observe(self, step: int, value: float) -> bool:
        assert step == self.step + self.policy["eval_every"] and np.isfinite(value)
        self.step = step
        if value < self.best:
            self.best, self.best_step = float(value), int(step)
        p = self.policy
        return step >= p["max_steps"] or (step >= p["min_steps"] and step - self.best_step >= p["patience_steps"])


def _moments(a: np.ndarray):
    flat = a.reshape(-1, a.shape[-1]).astype(np.float64)
    return np.nanmean(flat, 0), np.nanstd(flat, 0) + 1e-6


PERMUTATION_SEED = 20260920
# local geo-weather mechanism arms (geo_weather_20260924): base + LocalMechanisms(**kwargs) into the first damage
# layer. The node inputs (nw, na, nr) come from F; their variants (county mean, placebo) are built by the runner.
HAZARD_ARMS = ("W+Cin+H", "W+Cin+H2", "W+Cin+Hs")   # EIH features F["phi"] [U,144,J]; H2 slots; Hs signed
HAZARD_SLOTS = 4
LR_HAZARD = 0.1 * 3e-3           # slow clock for beta (REVIEW_formal 2.7), fixed from the first run
MECH_ARMS = {"W+Cin+M": dict(), "W+Cin+M0": dict(), "W+Cin+MP": dict(), "W+Cin+Mw": dict(),
             "W+Cin+M-dz": dict(use_dz=False), "W+Cin+M-can": dict(use_canopy=False)}


def regime_onehot(F: dict, k: int) -> np.ndarray:
    """One-hot geographic regime of every unit (PREREG Amendment 8): k-means (seeded, 20 restarts) on the
    standardised descriptors of all panel counties, missing values at the column median. Static covariates
    only; no outage enters."""
    from sklearn.cluster import KMeans
    fips = np.asarray(F["fips"]).astype(str)
    counties, first = np.unique(fips, return_index=True)
    g = np.asarray(F["geo"], dtype=np.float64)[first]
    g = np.where(np.isnan(g), np.nanmedian(g, 0), g)
    z = (g - g.mean(0)) / (g.std(0) + 1e-9)
    lab = KMeans(n_clusters=k, n_init=20, random_state=PERMUTATION_SEED).fit_predict(z)
    of = dict(zip(counties, lab))
    return np.eye(k, dtype=np.float32)[[of[c] for c in fips]]


def geo_variant(F: dict, arm: str) -> dict:
    """Geography controls (PREREG Amendment 6): the features with the geographic input replaced.

    GCRK-S  every unit gets the same vector (zeros; standardisation maps it to the neutral code), so
            one set of kernel parameters serves all counties.
    GCRK-P  counties exchange descriptor vectors by one fixed seeded permutation; a county keeps its
            donor's vector in all its events.
    Any other arm returns F unchanged.
    """
    if arm not in ("GCRK-S", "GCRK-P", "GCRK-K8"):
        return F
    F = dict(F)
    geo = np.array(F["geo"], dtype=np.float32, copy=True)
    if arm == "GCRK-S":
        geo[:] = 0.0
    elif arm == "GCRK-K8":
        geo = regime_onehot(F, 8)
    else:
        fips = np.asarray(F["fips"]).astype(str)
        counties = np.unique(fips)
        first = {c: int(np.where(fips == c)[0][0]) for c in counties}
        donor = dict(zip(counties, np.random.default_rng(PERMUTATION_SEED).permutation(counties)))
        geo = np.stack([F["geo"][first[donor[c]]] for c in fips]).astype(np.float32)
    F["geo"] = geo
    return F


def fit_stats(F: dict, idx: np.ndarray) -> dict:
    """Standardisation statistics from the fitting units only."""
    st = {}
    for k in ("xu", "xr", "xo"):
        st[k] = _moments(F[k][idx])
    st["geo"] = _moments(F["geo"][idx])
    if "ctx_extra" in F:             # extra static county context (e.g. canopy), standardised on the fitting units
        st["ctx_extra"] = _moments(F["ctx_extra"][idx])
    if "phi" in F:                   # per-feature scale: the fitting units' 99th percentile of the active hours
        ph = F["phi"][idx].astype(np.float32).reshape(-1, F["phi"].shape[-1])
        q = np.array([np.percentile(c[c > 0], 99) if (c > 0).any() else 1.0 for c in ph.T])
        st["phi_scale"] = (1.0 / np.maximum(q, 1e-6)).astype(np.float32)
    return st


def _std(a, mom, noclip: int):
    z = (a - mom[0]) / mom[1]
    z = np.nan_to_num(z, nan=0.0)
    if noclip < z.shape[-1]:
        z[..., noclip:] = np.clip(z[..., noclip:], -CLIP, CLIP)
    return z.astype(np.float32)


def make_batch(F: dict, idx: np.ndarray, st: dict, nodes: bool = False) -> dict:
    idx = np.asarray(idx)
    b = dict(xu=_std(F["xu"][idx], st["xu"], N_WEATHER),
             xr=_std(F["xr"][idx], st["xr"], N_WEATHER + N_STATIC),
             xo=_std(F["xo"][idx], st["xo"], 0),
             geo=_std(F["geo"][idx], st["geo"], 0))
    b["ctx"] = b["xr"][:, 0, N_WEATHER:N_WEATHER + N_STATIC]        # county context, constant in the window
    if "ctx_extra" in st:
        b["ctx"] = np.concatenate([b["ctx"], _std(F["ctx_extra"][idx], st["ctx_extra"], 0)], -1)
    b = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}
    b["y0"] = torch.from_numpy(np.ascontiguousarray(F["y0"][idx].astype(np.float32)))
    b["y"] = torch.from_numpy(np.ascontiguousarray(F["y"][idx].astype(np.float32)))
    b["m"] = torch.from_numpy(np.ascontiguousarray(F["m"][idx].astype(np.float32)))
    if "m_train" in F:               # training weights (artefact-flagged hours dropped); evaluation keeps b["m"]
        b["m_train"] = torch.from_numpy(np.ascontiguousarray(F["m_train"][idx].astype(np.float32)))
    if "phi_scale" in st:
        b["phi"] = torch.from_numpy(np.ascontiguousarray(F["phi"][idx].astype(np.float32) * st["phi_scale"]))
    if nodes:                                      # physical units, no standardisation (geo_mech reads them as is)
        for k in ("nw", "na", "nr"):
            b[k] = torch.from_numpy(np.ascontiguousarray(F[k][idx].astype(np.float32)))
    b["idx"] = idx
    return b


class Engine:
    """One model on one fitting set (optionally with a validation set)."""

    def __init__(self, F: dict, fit_idx, val_idx, seed: int, arm: str, private_seed: int = 1729):
        assert arm in ("W", "GCRK", "GCRK-S", "GCRK-P", "GCRK-K8", "GCRK-slow", "GCRK-open", "GCRK+Cin-open",
                       "W+C", "W+G", "W+Cin", "GCRK+Cin",
                       *MECH_ARMS, *HAZARD_ARMS)
        assert (arm in HAZARD_ARMS) == ("phi" in F), "hazard arms need F['phi'] and only they may have it"
        self.arm, self.seed, self.step = arm, int(seed), 0
        self.nodes = arm in MECH_ARMS
        self.fit_idx = np.sort(np.asarray(fit_idx))
        self.val_idx = None if val_idx is None else np.sort(np.asarray(val_idx))
        self.stats = fit_stats(F, self.fit_idx)
        self.fit = make_batch(F, self.fit_idx, self.stats, self.nodes)
        self.val = None if self.val_idx is None else make_batch(F, self.val_idx, self.stats, self.nodes)
        torch.manual_seed(self.seed)
        k_extra = int(F.get("xu_extra", 0))          # appended damage inputs get zero weights (paired init)
        self.model = AsymODE(F["xu"].shape[-1] - k_extra, F["xr"].shape[-1], F["xo"].shape[-1])
        if k_extra:
            self.model.expand_damage_inputs(k_extra)
        if arm in ("W+Cin", "GCRK+Cin", "GCRK+Cin-open") or arm in MECH_ARMS or arm in HAZARD_ARMS:
            self.model.attach_context_input(N_STATIC + (F["ctx_extra"].shape[-1] if "ctx_extra" in F else 0))
        if arm in HAZARD_ARMS:
            self.model.attach_hazard(F["phi"].shape[-1])
            self.model.haz_signed = arm == "W+Cin+Hs"
        if arm == "W+Cin+H2":
            names = [str(n) for n in F["phi_names"]]
            trig = [i for i, n in enumerate(names) if n.endswith("@0")]
            load = [i for i, n in enumerate(names) if "*one@" in n and not n.endswith("@0")]
            self.model.attach_hazard_slots(trig, load, HAZARD_SLOTS)
        if arm in MECH_ARMS:
            self.model.attach_mechanisms(LocalMechanisms(**MECH_ARMS[arm]), len(MECH))
            self.model.set_mechanism_scale(self.fit)
        if arm in ("GCRK", "GCRK-S", "GCRK-P", "GCRK-K8", "GCRK-slow", "GCRK+Cin", "GCRK-open", "GCRK+Cin-open"):
            self.model.attach_gcrk(torch.tanh(self.fit["geo"] / 3.0).mean(0), private_seed)
            if arm.endswith("-open"):     # PI 2026-09-26: no bound on the kernel's opening (beta = alpha)
                self.model.kernel.bounded_opening = False
        elif arm == "W+C":
            self.model.attach_level(N_STATIC, "ctx")
        elif arm == "W+G":
            self.model.attach_level(F["geo"].shape[-1], "geo")
        host, rec = self.model.parameter_groups()
        groups = [dict(params=host, lr=LR_HOST), dict(params=rec, lr=LR_RECOVERY)]
        if arm in HAZARD_ARMS:
            hp = self.model.hazard_params(); ids = {id(p) for p in hp}
            groups = [dict(params=[p for p in host if id(p) not in ids], lr=LR_HOST), dict(params=hp, lr=LR_HAZARD),
                      dict(params=rec, lr=LR_RECOVERY)]
        if arm == "GCRK-slow":        # PREREG Amendment 9: the conditioning maps take one tenth of the host's step
            k = self.model.kernel
            maps = [k.U, k.Vl, k.Va, k.Vg]
            ids = {id(p) for p in maps}
            groups = [dict(params=[p for p in host if id(p) not in ids], lr=LR_HOST), dict(params=maps, lr=0.1 * LR_HOST),
                      dict(params=rec, lr=LR_RECOVERY)]
        self.opt = torch.optim.Adam(groups, lr=LR_HOST)
        self.last_loss = float("nan")
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        k = self.model.kernel
        if k is None:
            return None
        k.training_step.fill_(self.step)
        return k.calibrate_(self.model.hidden(self.fit["xu"], self.fit["ctx"]), self.step)

    def train_step(self) -> float:
        k = self.model.kernel
        if k is not None:
            if self.step % 10 == 0 and int(k.calibration_step) != self.step:
                self.refresh()
            k.training_step.fill_(self.step)
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        out = self.model(self.fit)
        loss = trajectory_loss(out["P"], self.fit["y"], self.fit.get("m_train", self.fit["m"]))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        for n, p in self.model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError(f"nonfinite gradient {n}")
        self.opt.step()
        if self.model.haz_beta is not None and not self.model.haz_signed:
            with torch.no_grad():
                for p in self.model.hazard_params():
                    p.clamp_(min=0.0)
        self.step += 1
        if k is not None:
            k.training_step.fill_(self.step)
        self.last_loss = float(loss.detach())
        return self.last_loss

    @torch.no_grad()
    def evaluate(self):
        """Per-unit (squared-error sum, observed cells) on the validation units."""
        assert self.val is not None, "a REFIT model has no validation set"
        self.refresh()
        self.model.eval()
        out = self.model(self.val)
        s, n = masked_se(out["P"], self.val["y"], self.val["m"])
        return s.double().numpy(), n.double().numpy()

    def snapshot(self) -> dict:
        k = self.model.kernel
        return dict(model_state=copy.deepcopy(self.model.state_dict()), step=self.step, arm=self.arm, seed=self.seed,
                    fit_idx=self.fit_idx, stats=self.stats,
                    kernel_rng=None if k is None else k._drop.get_state().clone())


def select_steps(F: dict, inner: list[tuple], seed: int, arm: str, log=None) -> dict:
    """Pooled INNER early stopping over three lockstep inner folds."""
    engines = [Engine(F, fit, val, seed, arm) for fit, val in inner]
    rule, trace = Rule(**POLICY), []
    while True:
        t = engines[0].step
        res = [e.evaluate() for e in engines]
        pooled = float(sum(s.sum() for s, _ in res) / sum(n.sum() for _, n in res))
        stop = rule.observe(t, pooled)
        row = dict(step=t, pooled_inner_mse=pooled, best_step=rule.best_step)
        for j, (e, (s, n)) in enumerate(zip(engines, res), 1):
            row[f"inner{j}_mse"] = float(s.sum() / n.sum())
            row[f"inner{j}_fit_loss"] = e.last_loss
            k = e.model.kernel
            if k is not None:
                row[f"inner{j}_beta"] = float(torch.tanh(k.alpha).detach())
                row[f"inner{j}_scale"] = float(k.scale)
                row[f"inner{j}_theta"] = float(k.threshold)
        trace.append(row)
        if log is not None and t % 100 == 0:
            log(f"INNER {arm} seed={seed} step={t} pooled={pooled:.6e} best={rule.best_step}")
        if stop:
            break
        for e in engines:
            for _ in range(POLICY["eval_every"]):
                e.train_step()
    return dict(best_step=rule.best_step, best_inner_mse=rule.best, stop_step=rule.step, trace=trace)


def refit(F: dict, dev_idx, steps: int, seed: int, arm: str, log=None) -> Engine:
    e = Engine(F, dev_idx, None, seed, arm)
    for t in range(1, steps + 1):
        e.train_step()
        if t % 10 == 0:
            e.refresh()
        if log is not None and t % 200 == 0:
            log(f"REFIT {arm} seed={seed} step={t}/{steps}")
    e.refresh()
    e.model.eval()
    return e


@torch.no_grad()
def export(e: Engine, F: dict, idx) -> dict:
    """Open-loop OUTER rollouts from the observed p_71 (kernel open, and closed for GCRK)."""
    b = make_batch(F, idx, e.stats, e.nodes)
    e.model.eval()
    out = e.model(b, diagnostics=False)
    res = dict(idx=np.asarray(idx), P=out["P"].numpy(), u=out["u"].numpy(), r=out["r"].numpy(),
               raw_logit=out["raw_logit"].numpy())
    if e.model.kernel is not None:
        off = e.model(b, exit_open=False)
        res.update(P_closed=off["P"].numpy(), raw_logit_closed=off["raw_logit"].numpy())
    return res
