"""EXP08 -- the input and capacity asymmetry axes, one at a time.

EXP05 varied the dynamical axis with inputs and capacity held identical. This
experiment does the complement: the inflow form is pinned to the susceptible
form in every arm, and only the *input* or the *capacity* axis moves. A
difference between arms here is attributable to that axis and to nothing else.

Hypotheses and kill conditions are in `docs/PREREGISTRATION_asymmetry.md`; they
are not restated here, because restating them invites drift. What this file adds
is the design needed to make them decidable:

**Capacity (H-D) is run as a 2x2, not as a single comparison.** The registered
claim is that the *restoration* rate specifically tolerates reduced capacity. An
arm that only shrinks the restoration network cannot separate that from "this
model does not need capacity anywhere", so the mirror arm -- shrink the
interruption network instead -- is run alongside, and the floor arm shrinks both.
Without the mirror, a null result on one side is uninterpretable.

**Input (H-A3) is run as a three-way removal.** The registered claim for ambient
meteorology is deliberately a *negative* case: it should help both rates. So the
family is removed from the interruption side, from the restoration side, and from
both, against an arm that keeps it everywhere. Confirmation requires both
one-sided removals to hurt; if only one hurts, the family is asymmetric and H-A
is weakened rather than supported, which is the outcome the pre-registration
commits to reporting.

Not testable here, for want of the covariates rather than for want of a design:
H-A1 and H-B need pre-origin outage level and clearance rate, H-A2 needs
neighbouring-county aggregates, and H-C needs county statics and hazard
composites. The driver block currently carries raw meteorology only. Those arms
are absent from this file, not silently folded into another arm.

Sample construction, folds, mask and horizons are imported from EXP05 rather than
reimplemented, so "the same samples" is guaranteed by construction instead of by
inspection. The panel set is pinned by manifest and its digest is written into the
result file: a comparison across different sample sets is not a comparison, and
that is a property worth making machine-checkable rather than remembered.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.dynamics import (TwoRateODE, TwoRateConfig, InflowForm,   # noqa: E402
                              calibrate_init)
from asymode.evalproto import make_folds, inner_split                 # noqa: E402
from asymode import panels as panelset                              # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exp05", Path(__file__).resolve().parent / "exp05_real_dynamics.py")
exp05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp05)
load_pooled, add_context = exp05.load_pooled, exp05.add_context

INTERIM = ROOT / "data" / "interim"

# Channel families over the driver block, by name. Resolved against the
# `channels` array in the driver files, never by position: the driver builder is
# free to reorder or extend, and an index list would silently mean something else
# the next time it does.
#
# The split follows the pre-registration's own wording -- "ambient meteorological
# fields *beyond the core hazard variables*". Soil moisture sits in `hazard`: a
# saturated root zone is a damage pathway, not an ambient field, and it is the
# dominant one for wind-thrown trees. The placement matters more than it looks,
# because H-A3 is registered as a *negative* case -- the claim is that not every
# family is asymmetric. Putting a variable with a known damage mechanism into the
# ambient family would let that family test asymmetric for a reason that has
# nothing to do with the hypothesis, and under the registered wording that
# outcome weakens H-A rather than strengthening it.
FAMILIES: dict[str, tuple[str, ...]] = {
    "hazard":  ("cape", "gust", "precip", "snowfall", "soil_moisture",
                "u10", "v10", "wind_speed"),
    "ambient": ("cloud", "pressure", "rh", "t2m_c"),
    "clock":   ("clock_sin", "clock_cos"),
}


def check_families(names: list[str]) -> None:
    """Every channel must belong to exactly one family.

    Reviewing an ablation starts with checking what it actually removed.

    A channel in no family is not neutral: an arm that names the families it
    keeps drops it silently along with the family under test, so the ablation
    removes more than it claims and the result is attributed to the wrong thing.
    That is not hypothetical -- the wind components arrived in a driver rebuild,
    landed in no family, and turned an ablation billed as "remove ambient
    meteorology" into "remove ambient meteorology and two wind channels". The
    contaminated run reversed the sign of the finding.

    So this raises rather than warns. A blocked run is cheap; a mis-scoped
    ablation that looks like a result is not.
    """
    claimed: dict[str, list[str]] = {}
    for fam, chans in FAMILIES.items():
        for c in chans:
            claimed.setdefault(c, []).append(fam)
    orphan = [c for c in names if c not in claimed]
    dupes = {c: f for c, f in claimed.items() if len(f) > 1}
    missing = [c for c in claimed if c not in names]
    if orphan or dupes or missing:
        raise SystemExit(
            "channel families do not partition the driver block:\n"
            + (f"  in no family: {orphan}\n" if orphan else "")
            + (f"  in more than one: {dupes}\n" if dupes else "")
            + (f"  named by a family but absent from the drivers: {missing}\n" if missing else "")
            + "  assign every channel before running an input ablation.")


@dataclass(frozen=True)
class Arm:
    """One configuration of the two rate networks.

    `fam_u` / `fam_r` are family names, not indices. `None` means every channel,
    which is how an axis is switched off: pass the same value to both sides.
    """
    name: str
    axis: str                      # which registered hypothesis the arm serves
    fam_u: tuple[str, ...] | None
    fam_r: tuple[str, ...] | None
    hidden_u: int
    hidden_r: int
    note: str = ""
    fam_gate: tuple[str, ...] | None = None   # None -> no gate on the interruption rate
    # Defined but not run by default. The width profile is what decides H-C, and
    # reading its shape on proxy covariates before building the registered ones
    # would tell whoever builds them what shape to expect. Name them explicitly
    # with --arms to run them anyway.
    deferred: bool = False


ALL_FAM = ("hazard", "ambient", "clock")

ARMS: list[Arm] = [
    # --- control: both axes off. Identical to the EXP05 susceptible arm. ------
    Arm("control", "none", ALL_FAM, ALL_FAM, 32, 32,
        "both rates read everything at equal capacity"),

    # --- capacity axis (H-D), inputs held identical across all four -----------
    Arm("cap_r_glm",    "H-D", ALL_FAM, ALL_FAM, 32, 0,
        "registered direction: restoration degenerates to a logistic GLM"),
    Arm("cap_u_glm",    "H-D", ALL_FAM, ALL_FAM, 0, 32,
        "mirror. without it a null on the restoration side is uninterpretable"),
    Arm("cap_both_glm", "H-D", ALL_FAM, ALL_FAM, 0, 0,
        "floor: how much of the fit survives with no hidden layer anywhere"),

    # --- input axis (H-A3), capacity held identical across all four -----------
    Arm("in_ambient_u_only", "H-A3", ALL_FAM, ("hazard", "clock"), 32, 32,
        "ambient removed from restoration"),
    Arm("in_ambient_r_only", "H-A3", ("hazard", "clock"), ALL_FAM, 32, 32,
        "ambient removed from interruption"),
    Arm("in_ambient_none",   "H-A3", ("hazard", "clock"), ("hazard", "clock"), 32, 32,
        "ambient removed from both"),

    # --- gate input width (H-C), PILOT ONLY ----------------------------------
    # These cannot decide H-C. The registered hypothesis is that the gate wants
    # county identity and hazard composites and not raw weather; none of those
    # three families exists in the driver block yet, so what is swept here is
    # width over the covariates that do exist. Graded [C] by construction: the
    # registered criteria are evaluated once, on the registered covariates, and
    # running a proxy first and the real test later would be exactly the forking
    # path the pre-registration is the control for.
    #
    # Only the two endpoints run by default -- enough to show the gate trains and
    # to measure its collapse rate. The four interior points, including the
    # equal-width pair that separates content from width, are defined here and
    # deferred: the profile is run once, on the registered covariates.
    Arm("gate_clock",          "H-C-pilot", ALL_FAM, ALL_FAM, 32, 32,
        "width 2", fam_gate=("clock",)),
    Arm("gate_hazard",         "H-C-pilot", ALL_FAM, ALL_FAM, 32, 32,
        "width 5", fam_gate=("hazard",), deferred=True),
    Arm("gate_ambient",        "H-C-pilot", ALL_FAM, ALL_FAM, 32, 32,
        "width 5, content control against gate_hazard", fam_gate=("ambient",), deferred=True),
    Arm("gate_hazard_clock",   "H-C-pilot", ALL_FAM, ALL_FAM, 32, 32,
        "width 7", fam_gate=("hazard", "clock"), deferred=True),
    Arm("gate_hazard_ambient", "H-C-pilot", ALL_FAM, ALL_FAM, 32, 32,
        "width 10", fam_gate=("hazard", "ambient"), deferred=True),
    Arm("gate_all",            "H-C-pilot", ALL_FAM, ALL_FAM, 32, 32,
        "width 12, the suspected failure mode: gate and pulse on the same inputs",
        fam_gate=ALL_FAM),
]


def resolve(fams: tuple[str, ...] | None, names: list[str]) -> list[int] | None:
    """Family names -> channel indices. `None` (every channel) stays `None`.

    Every named channel must exist. A family that silently resolves to fewer
    channels than intended would turn an input-asymmetry result into an artefact
    of a typo, so this raises instead.
    """
    if fams is None or set(fams) == set(FAMILIES):
        return None
    want: list[str] = []
    for f in fams:
        if f not in FAMILIES:
            raise KeyError(f"unknown family {f!r}; known: {sorted(FAMILIES)}")
        want += list(FAMILIES[f])
    missing = [c for c in want if c not in names]
    if missing:
        raise KeyError(f"channels absent from the driver block: {missing}")
    return sorted(names.index(c) for c in want)


def side_evidence(y0: np.ndarray, yt: np.ndarray, m: np.ndarray) -> dict:
    """How much evidence each rate is identified from, on a given sample set.

    The state equation exposes the rates only through

        dy_t = u_t (1 - y_t) - r_t y_t

    so a one-step cell carries information about `u` in proportion to (1 - y_t)
    and about `r` in proportion to y_t. A cell at y_t = 0 carries *no* information
    about the restoration rate at all -- not little, none -- and that is a
    property of the dynamics, not of the units the target happens to be in.
    Restoration is observable only where there is something left to restore.

    Two summaries per side, because they answer different questions:

      * `lev`  = sum of squared leverage, in units of fully-informative cells.
                 The Fisher information a scalar rate would accumulate.
      * `ess`  = the same weights read as a Kish effective sample size,
                 (sum w^2)^2 / sum w^4 -- how many cells actually carry it, which
                 `lev` cannot see because a handful of extreme cells can carry it.

    Reported whatever it says, per the registered commitment. This measures the
    leverage the *data* offers; it explains why a capacity asymmetry might be
    warranted and does not on its own test H-D. The test is the kill condition.
    """
    ent = np.concatenate([y0[:, None], yt[:, :-1]], axis=1)   # state entering each step
    mm = m.astype(bool)
    w = {"u": (1.0 - ent)[mm], "r": ent[mm]}
    out: dict[str, float] = {"n_cells": int(mm.sum())}
    for side, ww in w.items():
        i = ww ** 2
        s1, s2 = float(i.sum()), float((i ** 2).sum())
        out[f"lev_{side}"] = s1
        out[f"ess_{side}"] = (s1 * s1 / s2) if s2 > 0 else 0.0
    pos = ent[mm] > 0
    # The decomposition that separates "rarely identifiable" from "weakly
    # identifiable when it is": lev_r = P(y>0) * E[y^2 | y>0] * n_cells.
    out["p_identifiable_r"] = float(pos.mean())
    out["mean_sq_given_pos_r"] = float((ent[mm][pos] ** 2).mean()) if pos.any() else 0.0
    return out


def run_arm(arm: Arm, tr, te, data, args, seed: int, names: list[str],
            fips, fold_id: int) -> dict:
    """Train one arm. Mirrors the EXP05 recipe exactly except for the two axes."""
    y0, X, yt, m = data
    torch.manual_seed(seed); np.random.seed(seed)
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6      # training folds only
    Xn = ((X - mu) / sd).astype(np.float32)

    t = lambda a: torch.tensor(a)
    ytr = np.concatenate([y0[tr][:, None], yt[tr]], axis=1)
    mtr = np.concatenate([np.ones((len(tr), 1), bool), m[tr]], axis=1)
    # Every arm here shares the inflow form, so every arm gets the same initial
    # flows. The capacity and input axes must not be entangled with the
    # initialisation the way the dynamical axis necessarily was.
    u0, r0 = calibrate_init(ytr, mtr, InflowForm.SUSCEPTIBLE)
    cfg = TwoRateConfig(d_in=Xn.shape[-1], cap_u=args.cap_u, cap_r=args.cap_r,
                        hidden_u=arm.hidden_u, hidden_r=arm.hidden_r,
                        inflow=InflowForm.SUSCEPTIBLE,
                        idx_u=resolve(arm.fam_u, names), idx_r=resolve(arm.fam_r, names),
                        u_init=u0, r_init=r0,
                        gate_u=arm.fam_gate is not None,
                        idx_gate=resolve(arm.fam_gate, names))
    model = TwoRateODE(cfg)
    # The rate each arm actually starts at, before any training. Arms are
    # calibrated to the same initial flow, but they do not have identical
    # parameters -- the networks differ in shape and are drawn independently -- so
    # this is the residual initialisation difference that structure has to be
    # judged against. Recorded rather than argued about.
    with torch.no_grad():
        probe = torch.tensor(Xn[np.asarray(tr)[:2048], 0])
        pu, pr = model.rates(probe, torch.tensor(y0[np.asarray(tr)[:2048]]))
        init_u_mean, init_r_mean = float(pu.mean()), float(pr.mean())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Early stopping holds out counties, not rows: see `inner_split`. A row-random
    # split leaves the same counties on both sides and cannot see the failure the
    # outer folds exist to measure.
    fi, vi = inner_split(fips[tr], seed=seed, fold=fold_id)
    tr_arr = np.asarray(tr)
    fit, va = tr_arr[fi], tr_arr[vi]
    Y0, XX, YT, MM = t(y0), t(Xn), t(yt), t(m.astype(np.float32))

    def loss_on(ix):
        pred = model(Y0[ix], XX[ix])
        se = (pred - YT[ix]) ** 2 * MM[ix]
        return se.sum() / MM[ix].sum().clamp_min(1.0)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(fit)
        for s in range(0, len(perm), args.batch):
            b = torch.tensor(perm[s:s + args.batch], dtype=torch.long)
            opt.zero_grad(); l = loss_on(b); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_on(torch.tensor(va, dtype=torch.long)))
        if vl < best - 1e-10:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        ti = torch.tensor(te, dtype=torch.long)
        pred = model(Y0[ti], XX[ti]).numpy()
    out: dict = {}
    for h in args.horizons:
        e = (pred[:, h - 1] - yt[te][:, h - 1])[m[te][:, h - 1]]
        out[f"rmse_h{h}"] = float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")
        out[f"n_h{h}"] = int(e.size)
    # Collapse screen. A degenerate arm that predicts a constant can post a
    # respectable RMSE against a target that is zero half the time, and it must
    # not be read as a fit. Recorded per arm rather than checked once.
    out["pred_sd"] = float(pred.std())
    out["pred_max"] = float(pred.max())
    out["frac_pred_below_1e6"] = float((pred < 1e-6).mean())
    out["val_loss"] = best
    out["u_init"], out["r_init"] = u0, r0
    out["init_u_mean"], out["init_r_mean"] = init_u_mean, init_r_mean
    out["n_param_u"] = sum(p.numel() for p in model.phi_u.parameters())
    out["n_param_r"] = (0 if model.phi_r is None else
                        sum(p.numel() for p in model.phi_r.parameters()))
    if arm.fam_gate is not None:
        # Measured on the test rollout, not over training. A collapsed gate is not
        # a worse arm, it is a void one: the pulse branch stops receiving gradient
        # once the gate reaches zero and cannot reopen, so the arm's error is not
        # evidence about the structure. Recorded so the distinction is made from
        # the file rather than from memory.
        model.phi_u.reset_gate_stats()
        with torch.no_grad():
            model(Y0[ti], XX[ti])
        g = model.phi_u.gate_stats()
        out.update(g)
        out["gate_width"] = len(resolve(arm.fam_gate, names) or names)
        out["void"] = bool(g["frac_gate_closed"] > 0.99)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24, 48])
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--cap-u", type=float, default=0.25)
    ap.add_argument("--cap-r", type=float, default=0.25)
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arm names")
    ap.add_argument("--panels", default=None,
                    help="panel set: omit for the manifest, 'auto' to pool what "
                         "is on disk (exploration only), or a path to a JSON file")
    ap.add_argument("--out", default="results/exp08_architecture.json")
    a = ap.parse_args()

    want, digest = panelset.resolve(INTERIM, a.panels)
    y0, X, yt, m, fips, panel = load_pooled(a.horizon, a.stride, panels=want)
    X = add_context(X, y0, a.horizon)
    names = panelset.channel_names(INTERIM)
    assert len(names) == X.shape[-1], (len(names), X.shape[-1])
    check_families(names)

    panels = sorted(set(panel.tolist()))
    print(f"pooled samples {len(y0):,} over {len(panels)} panels [{digest}], "
          f"{len(set(fips)):,} counties, {X.shape[-1]} channels, horizon {a.horizon} h")
    print(f"observed targets: {m.mean()*100:.1f}%   mean y0 {y0.mean():.5f}")

    arms = ([x for x in ARMS if x.name in a.arms] if a.arms
            else [x for x in ARMS if not x.deferred])
    if a.arms:
        unknown = set(a.arms) - {x.name for x in ARMS}
        if unknown:
            raise SystemExit(f"unknown arms: {sorted(unknown)}")

    rows: list[dict] = []
    evidence: list[dict] = []
    for seed in a.seeds:
        fold = make_folds(sorted(set(fips)), k=a.k, seed=seed)
        fmap = {f: fo for f, fo in zip(sorted(set(fips)), fold)}
        assign = np.array([fmap[f] for f in fips])
        for f in range(a.k):
            te = np.where(assign == f)[0]; tr = np.where(assign != f)[0]
            evidence.append({"seed": seed, "fold": f, "split": "train",
                             **side_evidence(y0[tr], yt[tr], m[tr])})
            evidence.append({"seed": seed, "fold": f, "split": "test",
                             **side_evidence(y0[te], yt[te], m[te])})
            for arm in arms:
                t0 = time.time()
                r = run_arm(arm, tr, te, (y0, X, yt, m), a, seed, names, fips, f)
                wall = round(time.time() - t0, 1)
                rows.append({"arm": arm.name, "axis": arm.axis, "seed": seed,
                             "fold": f, "n_test": len(te), "wall_s": wall, **r})
                print(f"  seed {seed} fold {f} {arm.name:<20} "
                      + " ".join(f"h{h}={r[f'rmse_h{h}']:.5f}" for h in a.horizons)
                      + f"  {wall}s", flush=True)

    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a)); cfg["out"] = a.out
    cfg["panels"] = panels
    cfg["panel_digest"] = digest
    cfg["channels"] = names
    cfg["channel_digest"] = panelset.channel_digest(names)
    cfg["source"] = panelset.source_version(ROOT)
    cfg["families"] = {k: list(v) for k, v in FAMILIES.items()}
    cfg["arms"] = [{"name": x.name, "axis": x.axis, "fam_u": x.fam_u, "fam_r": x.fam_r,
                    "hidden_u": x.hidden_u, "hidden_r": x.hidden_r,
                    "fam_gate": x.fam_gate, "note": x.note} for x in arms]
    out.write_text(json.dumps({"config": cfg, "rows": rows,
                               "evidence": evidence}, indent=2))

    print(f"\n=== pooled over {a.k} folds x {len(a.seeds)} seeds ===")
    print(f"{'arm':<22}{'axis':<7}" + "".join(f"{'RMSE h+'+str(h):>19}" for h in a.horizons))
    for arm in arms:
        g = [r for r in rows if r["arm"] == arm.name]
        line = f"{arm.name:<22}{arm.axis:<7}"
        for h in a.horizons:
            v = [r[f"rmse_h{h}"] for r in g if np.isfinite(r[f"rmse_h{h}"])]
            line += f"{np.mean(v):>13.5f}±{np.std(v):<5.5f}"
        print(line)

    gated = [r for r in rows if "gate_width" in r]
    if gated:
        print(f"\n=== gate profile (H-C pilot, [C]) ===")
        print(f"{'arm':<22}{'width':>6}{'gate mean':>11}{'closed':>9}{'open':>8}{'void':>7}")
        for arm in arms:
            g = [r for r in gated if r["arm"] == arm.name]
            if not g:
                continue
            print(f"{arm.name:<22}{g[0]['gate_width']:>6}"
                  f"{np.mean([r['gate_mean'] for r in g]):>11.4f}"
                  f"{np.mean([r['frac_gate_closed'] for r in g]):>9.4f}"
                  f"{np.mean([r['frac_gate_open'] for r in g]):>8.4f}"
                  f"{sum(r['void'] for r in g):>4}/{len(g)}")

    e = [r for r in evidence if r["split"] == "train"]
    print(f"\n=== identification leverage, training folds (mean over {len(e)}) ===")
    print(f"  interruption   lev {np.mean([x['lev_u'] for x in e]):12,.0f}   "
          f"ESS {np.mean([x['ess_u'] for x in e]):12,.0f}")
    print(f"  restoration    lev {np.mean([x['lev_r'] for x in e]):12,.0f}   "
          f"ESS {np.mean([x['ess_r'] for x in e]):12,.0f}")
    print(f"  ratio          lev {np.mean([x['lev_u'] for x in e])/max(np.mean([x['lev_r'] for x in e]),1e-9):8,.0f}:1   "
          f"ESS {np.mean([x['ess_u'] for x in e])/max(np.mean([x['ess_r'] for x in e]),1e-9):8,.0f}:1")
    print(f"  P(restoration identifiable) {np.mean([x['p_identifiable_r'] for x in e]):.4f}")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
