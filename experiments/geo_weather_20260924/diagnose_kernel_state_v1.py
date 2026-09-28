"""Frozen seed-0 kernel diagnostics, development tranche only; no fitting or paper writes.

Run with a single CPU thread at low priority. Complete systems are batched together so
the sampled graph loses no edges. Detailed arrays/rows go under ignored runs/; the
summary goes to results/v1/kernel_state_diagnostics_s0.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch
from torch.nn import functional as TF

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from asymode import gcrk_train as G
from asymode.asym_host import rate_inertia, stock_path
from screen import load

RUNS = ROOT / "runs" / "geo_weather_20260924"
DETAIL = RUNS / "kernel_state_diagnostics_s0"
REGIMES = ["tropical", "winter", "synoptic_wind", "convective", "heavy_rain"]


def distribution(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    qs = np.quantile(x, [.1, .5, .9, .99])
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std()),
                p10=float(qs[0]), p50=float(qs[1]), p90=float(qs[2]), p99=float(qs[3]), max=float(x.max()))


def effective_rank(x, center=True, weights=None):
    x = np.asarray(x, dtype=np.float64)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    if center:
        x = x - (w[:, None] * x).sum(0)
    s = np.linalg.svd(x * np.sqrt(w[:, None]), compute_uv=False)
    p = s * s / max(float(s @ s), 1e-300)
    positive = p > 0
    return dict(singular_values=s.tolist(), variance_share=p.tolist(),
                entropy_rank=float(np.exp(-(p[positive] * np.log(p[positive])).sum())),
                participation_rank=float(1 / max(float(p @ p), 1e-300)))


def confounding(x, reg, weights=None):
    x = np.asarray(x, dtype=np.float64)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mean = (w[:, None] * x).sum(0)
    total = (w[:, None] * (x - mean) ** 2).sum(0)
    between = np.zeros(x.shape[1])
    residual = x.copy()
    for r in np.unique(reg):
        ii = reg == r
        mu = (w[ii, None] * x[ii]).sum(0) / w[ii].sum()
        between += w[ii].sum() * (mu - mean) ** 2
        residual[ii] -= mu
    return dict(between_fraction=float(between.sum() / max(total.sum(), 1e-300)),
                between_fraction_by_coordinate=(between / np.maximum(total, 1e-300)).tolist(),
                total_trace=float(total.sum()), within_trace=float((total - between).sum()),
                within_effective_rank=effective_rank(residual, weights=w))


def rebuild(F, ck):
    model = G.AsymODE(F["xu"].shape[-1], F["xr"].shape[-1], F["xo"].shape[-1])
    model.attach_context_input(G.N_STATIC)
    k = model.attach_gcrk(torch.zeros(F["geo"].shape[-1]))
    if ck["arm"] == "STGCRK+Cin":
        k.attach_space()
    model.load_state_dict(ck["model_state"])
    return model.eval()


def system_batches(F, idx, limit=512):
    groups = [idx[F["system"][idx] == s] for s in np.unique(F["system"][idx])]
    current = []
    for group in groups:
        if current and sum(len(x) for x in current) + len(group) > limit:
            yield np.concatenate(current)
            current = []
        current.append(group)
    if current:
        yield np.concatenate(current)


def downstream(model, b, a2):
    raw = model.damage[4](torch.relu(a2)).squeeze(-1)[:, 72:]
    forget = torch.sigmoid(model.smoother(b["xu"][:, 72:])).squeeze(-1)
    logit = rate_inertia(raw.contiguous(), forget.contiguous())
    cond = .5 * torch.sigmoid(logit)
    occurrence = torch.sigmoid(model.occurrence(b["xo"][:, 72:])).squeeze(-1)
    background = .015 * torch.sigmoid(model.background(b["xu"][:, 72:])).squeeze(-1)
    u = (occurrence * cond + background).clamp(0, .515)
    r = .5 * torch.sigmoid(model.recovery(b["xr"][:, 72:])).squeeze(-1)
    return dict(P=stock_path(u.contiguous(), r.contiguous(), b["y0"].contiguous()), u=u, raw_logit=raw,
                logit=logit, conditional=cond)


def intervention(model, b, out, name, temporal_alpha=None, fit_mean_code=None):
    k = model.kernel
    kd = None
    if name == "closed":
        a2 = TF.linear(out["h1"], k.weight, k.bias)
    elif name in ("zero_geo", "neutral_code", "fit_mean_code"):
        g = torch.zeros_like(b["geo"]) if name == "zero_geo" else b["geo"]
        if name in ("neutral_code", "fit_mean_code"):
            code = b["geo"].new_zeros((1, k.U.shape[0])) if name == "neutral_code" else fit_mean_code
            with patch.object(k, "code_of", lambda g: code.expand(len(g), -1)):
                a2 = k(out["h1"], g, space=b.get("space"))
        else:
            a2 = k(out["h1"], g, space=b.get("space"))
    elif name == "coupling_off":
        saved = k.kappa_s.clone(), k.kappa_a.clone()
        k.kappa_s.zero_(); k.kappa_a.zero_()
        try:
            a2, kd = k(out["h1"], b["geo"], space=b["space"], diagnostics=True)
        finally:
            k.kappa_s.copy_(saved[0]); k.kappa_a.copy_(saved[1])
    elif name.startswith("gamma_"):
        saved = k.geo_sim.clone()
        k.geo_sim.fill_(float(name.split("_")[1]))
        try:
            a2 = k(out["h1"], b["geo"], space=b["space"])
        finally:
            k.geo_sim.copy_(saved)
    elif name == "temporal_beta":
        ratio = torch.tanh(temporal_alpha) / torch.tanh(k.alpha)
        a2 = TF.linear(out["h1"] + ratio * out["kernel_effect"], k.weight, k.bias)
    else:
        raise ValueError(name)
    result = downstream(model, b, a2)
    if kd is not None:
        result["state"] = kd["state"]
        result["effect"] = kd["effect"]
    return result


def compare(P, Q, F, idx, bootstrap=True):
    """Frozen variant P versus original Q, over all 144 forecast hours (72..215)."""
    y, m, w = F["y"][idx].astype(float), F["m"][idx].astype(float), F["w"][idx].astype(float)
    reg, fam = F["regime"][idx].astype(str), F["family"][idx].astype(str)
    s1 = (m * (P.astype(float) - y) ** 2).sum(1) * w
    s0 = (m * (Q.astype(float) - y) ** 2).sum(1) * w
    by = {r: float(s1[reg == r].sum() / s0[reg == r].sum() - 1) for r in REGIMES if (reg == r).any()}
    res = dict(n=len(idx), pooled_mse_change=float(s1.sum() / s0.sum() - 1),
               balanced_mse_change=float(np.mean(list(by.values()))), by_regime=by,
               mean_abs_path_change_pp=float(100 * np.abs(P - Q).mean()),
               mean_abs_peak_change_pp=float(100 * np.abs(P.max(1) - Q.max(1)).mean()),
               share_peak_change_gt_half_pp=float((np.abs(P.max(1) - Q.max(1)) > .005).mean()))
    if bootstrap:
        rng = np.random.default_rng(20260924)
        num, den, changes = np.zeros(2000), np.zeros(2000), []
        for r in REGIMES:
            keys = np.unique(fam[reg == r])
            if not len(keys):
                continue
            a = np.array([s1[(reg == r) & (fam == f)].sum() for f in keys])
            b = np.array([s0[(reg == r) & (fam == f)].sum() for f in keys])
            count = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=2000)
            na, nb = count @ a, count @ b
            num += na; den += nb; changes.append(na / nb - 1)
        res["pooled_mse_ci95"] = np.quantile(num / den - 1, [.025, .975]).tolist()
        res["balanced_mse_ci95"] = np.quantile(np.mean(changes, axis=0), [.025, .975]).tolist()
    return res


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))
    F = load("v1D")
    splits = json.loads((HERE / "splits_v1D.json").read_text())["event"]
    sp = np.load(ROOT / "data/interim/panel_v1/space_v1D.npz")
    assert np.array_equal(sp["fips"], F["fips"]) and np.array_equal(sp["system"], F["system"])
    space = {key: sp[key] for key in ("nbr", "wd", "up")}
    n = len(F["y"])
    DETAIL.mkdir(parents=True, exist_ok=True)
    report = dict(meta=dict(seed=0, tranche="D", folds=args.folds, county_events=n,
                           thread_count=1, nice=os.getpriority(os.PRIO_PROCESS, 0),
                           inference_only=True, intervals="2000 family-cluster draws within regime; seed 20260924",
                           metric_window="all 144 forecast hours 72..215; compare_kernels +1 omits hour 72",
                           distributions="unweighted descriptive unless explicitly labelled design_weighted",
                           neutral_code="exact z=0; standardized g=0 generally has nonzero z due to geo_center",
                           fit_mean_code="unweighted empirical mean code of fitting county-events, using that fold's statistics",
                           effective_rank="entropy and participation of normalized squared singular values",
                           interventions="frozen parameters, not refits and not causal training explanations"), models={})
    for arm, label in (("GCRK", "v1_gcrk_s0"), ("STGCRK", "v1_stgcrk_s0")):
        FF = dict(F)
        if arm == "STGCRK":
            FF["space"] = space
        metrics = {}
        predictions = {name: np.full((n, 144), np.nan, np.float32) for name in
                       ["own", "closed", "zero_geo", "neutral_code", "fit_mean_code"] + (["coupling_off"] if arm == "STGCRK" else [])}
        fold_ids = np.zeros(n, int)
        folds = {}
        fold1 = {}
        for fold in args.folds:
            path = RUNS / label / f"fold{fold:02d}"
            ck = torch.load(path / "final.pt", weights_only=False)
            model = rebuild(F, ck); k = model.kernel
            ref = np.load(path / "outer.npz"); idx = ref["idx"]
            assert set(idx) == set(splits[str(fold)]["outer"])
            fit_idx = np.array(splits[str(fold)]["dev"])
            fit_g = torch.from_numpy(G._std(F["geo"][fit_idx], ck["stats"]["geo"], 0))
            fit_mean_code = k.code_of(fit_g).mean(0, keepdim=True)
            row = np.full(n, -1, int); row[idx] = np.arange(len(idx))
            fold_ids[idx] = fold
            g = torch.from_numpy(G._std(F["geo"][idx], ck["stats"]["geo"], 0))
            z = k.code_of(g).numpy(); lam, interaction, gain = [x.numpy() for x in k.condition(g)]
            zero_z = k.code_of(torch.zeros_like(g[:1])).numpy()[0]
            w = F["w"][idx]
            fd = dict(n=len(idx), systems=len(np.unique(F["system"][idx])),
                      beta=float(torch.tanh(k.alpha)), beta_derivative=float(1 - torch.tanh(k.alpha) ** 2),
                      scale=float(k.scale), threshold=float(k.threshold), ramp=k.ramp(),
                      U=effective_rank(k.U.detach().numpy(), center=False), z=effective_rank(z),
                      z_design_weighted=effective_rank(z, weights=w),
                      z_mean=z.mean(0).tolist(), z_std=z.std(0).tolist(),
                      z_class_confounding=confounding(z, F["regime"][idx]),
                      z_class_confounding_design_weighted=confounding(z, F["regime"][idx], w),
                      zero_geo_code=zero_z.tolist(), zero_geo_code_norm=float(np.linalg.norm(zero_z)),
                      fitting_mean_code=fit_mean_code[0].tolist(), fitting_mean_code_norm=float(fit_mean_code.norm()),
                      code_abs=distribution(np.abs(z)),
                      code_saturated_gt_095=float((np.abs(z) > .95).mean()),
                      lambda_distribution=distribution(lam),
                      memory_hours=distribution(1 / np.log1p(lam)),
                      interaction_norm=distribution(np.linalg.norm(interaction, axis=1)), gain_distribution=distribution(gain),
                      map_frobenius={name: float(getattr(k, name).norm()) for name in ("U", "Vl", "Va", "Vg")},
                      condition_class_between_fraction={name: confounding(value, F["regime"][idx])["between_fraction"]
                                                        for name, value in (("lambda", lam), ("a", interaction), ("Omega", gain))},
                      condition_coordinate_county_sd={name: distribution(value.std(0))
                                                      for name, value in (("lambda", lam), ("a", interaction), ("Omega", gain))},
                      reconstructed_max_abs_error=0., closed_max_abs_error=0., lost_graph_edges=0)
            with patch.object(k, "code_of", lambda gg: gg.new_zeros((len(gg), k.U.shape[0]))):
                neutral = [x.numpy() for x in k.condition(g)]
            fd["condition_abs_delta_from_neutral"] = {name: distribution(np.abs(value - base)) for name, value, base in
                                                       zip(("lambda", "a", "Omega"), (lam, interaction, gain), neutral)}
            zero = [x.numpy() for x in k.condition(torch.zeros_like(g))]
            fd["condition_abs_delta_from_zero_geo"] = {name: distribution(np.abs(value - base)) for name, value, base in
                                                        zip(("lambda", "a", "Omega"), (lam, interaction, gain), zero)}
            if arm == "STGCRK":
                fd.update({name: float(getattr(k, name)) for name in ("kappa_s", "kappa_a", "geo_sim")})
                neighbor_count = (space["nbr"][idx] >= 0).sum(1)
                upwind = space["up"][idx, 72:].astype(float).sum(-1)
                fd["graph"] = dict(neighbor_count=distribution(neighbor_count),
                                   isolated_county_event_share=float((neighbor_count == 0).mean()),
                                   full_eight_neighbor_share=float((neighbor_count == 8).mean()),
                                   forecast_upwind_row_sum=distribution(upwind),
                                   no_upwind_forecast_hour_share=float((upwind == 0).mean()))
            temporal_alpha = torch.load(RUNS / "v1_gcrk_s0" / f"fold{fold:02d}/final.pt", weights_only=False)["model_state"]["damage.2.alpha"]
            for ii in system_batches(F, idx):
                if arm == "STGCRK":
                    nb = space["nbr"][ii]; valid = nb >= 0
                    lost = int((valid & ~np.isin(nb, ii)).sum())
                    assert lost == 0, "incomplete-system batch loses graph edges"
                b = G.make_batch(FF, ii, ck["stats"])
                out = model(b, diagnostics=True)
                assert torch.allclose(downstream(model, b, out["a2"])["P"], out["P"], atol=1e-7, rtol=0)
                p = out["P"].numpy(); predictions["own"][ii] = p
                fd["reconstructed_max_abs_error"] = max(fd["reconstructed_max_abs_error"], float(np.abs(p - ref["P"][row[ii]]).max()))
                closed = intervention(model, b, out, "closed")
                predictions["closed"][ii] = closed["P"].numpy()
                fd["closed_max_abs_error"] = max(fd["closed_max_abs_error"], float(np.abs(closed["P"].numpy() - ref["P_closed"][row[ii]]).max()))
                for name in predictions:
                    if name not in ("own", "closed"):
                        variant = intervention(model, b, out, name, fit_mean_code=fit_mean_code)
                        predictions[name][ii] = variant["P"].numpy()
                        if name == "coupling_off":
                            coupling_off_state, coupling_off_effect = variant["state"], variant["effect"]
                if arm == "STGCRK" and fold == 1:
                    for name in ("gamma_0", "gamma_1", "gamma_10", "temporal_beta"):
                        fold1.setdefault(name, np.full((n, 144), np.nan, np.float32))[ii] = intervention(model, b, out, name, temporal_alpha)["P"].numpy()
                effect = out["kernel_effect"][:, 72:]
                h = out["h1"][:, 72:]
                a2 = out["a2"][:, 72:]
                a2base = TF.linear(h, k.weight, k.bias)
                projected = TF.linear(effect, k.weight)
                direction = (model.damage[4].weight * (a2 > 0)) @ k.weight
                cosine = (effect * direction).sum(-1) / (effect.norm(dim=-1) * direction.norm(dim=-1)).clamp_min(1e-12)
                vals = dict(state_norm=out["kernel_state"].norm(dim=-1), kernel_gate=out["kernel_gate"],
                            deposit_norm=out["kernel_deposit"].norm(dim=-1),
                            hidden_effect_norm=effect.norm(dim=-1),
                            projected_effect_norm=projected.norm(dim=-1),
                            projected_effect_ratio=projected.norm(dim=-1) / a2base.norm(dim=-1).clamp_min(1e-12),
                            readout_cosine=cosine, abs_readout_cosine=cosine.abs(),
                            raw_logit_delta=out["raw_logit"] - closed["raw_logit"],
                            smoothed_logit_delta=out["logit"] - closed["logit"],
                            conditional=out["conditional"], occurrence_gate=out["gate"],
                            damage_rate_delta=out["u"] - closed["u"], prediction_delta=out["P"] - closed["P"])
                if arm == "STGCRK":
                    vals["uncoupled_state_norm"] = coupling_off_state.norm(dim=-1)
                    vals["state_change_from_coupling_norm"] = (out["kernel_state"] - coupling_off_state).norm(dim=-1)
                    vals["uncoupled_projected_effect_norm"] = TF.linear(coupling_off_effect[:, 72:], k.weight).norm(dim=-1)
                    code = k.code_of(b["geo"])
                    cw, cu = k.coupling(code, b["space"])
                    vals["mixing_share"] = (cw[:, None] + cu).sum(-1)
                    fd.setdefault("neighbor_code_distance_samples", []).extend(
                        (code[:, None] - code[b["space"]["nbr"]]).square().sum(-1)[b["space"]["wd"] > 0].tolist())
                for name, tensor in vals.items():
                    value = tensor.numpy()
                    metrics.setdefault(name, np.full((n, value.shape[1]), np.nan, np.float32))[ii] = value
            assert fd["reconstructed_max_abs_error"] < 1e-5 and fd["closed_max_abs_error"] < 1e-5
            if arm == "STGCRK":
                fd["neighbor_code_squared_distance"] = distribution(fd.pop("neighbor_code_distance_samples"))
            folds[str(fold)] = fd
            print(arm, "fold", fold, "reproduced", fd["reconstructed_max_abs_error"], "beta", fd["beta"], flush=True)
        active = np.where(fold_ids > 0)[0]
        assert len(np.unique(active)) == len(active)
        assert all(np.isfinite(v[active]).all() for v in metrics.values())
        assert metrics["state_norm"][active].max() <= 1 + 1e-5
        groups = {"all": active, **{f"fold{k}": np.where(fold_ids == k)[0] for k in args.folds},
                  **{r: active[F["regime"][active] == r] for r in REGIMES}}
        # Every scalar is summarised on the forecast hours; the state/gate prefix is reported separately.
        summaries = {group: {name: distribution(v[ii, -144:]) for name, v in metrics.items()} for group, ii in groups.items()}
        hourly = {}
        for start, stop in ((0, 24), (24, 72), (72, 96), (96, 144), (144, 192), (192, 216)):
            hourly[f"{start}:{stop}"] = {name: distribution(v[active, start:stop] if v.shape[1] == 216 else
                                                           v[active, max(0, start - 72):max(0, stop - 72)])
                                              for name, v in metrics.items() if v.shape[1] == 216 or stop > 72}
        county_rows = []
        for fips in np.unique(F["fips"][active]):
            ii = active[F["fips"][active] == fips]
            county_rows.append(dict(fips=str(fips), n_events=len(ii), **{name: float(v[ii, -144:].mean()) for name, v in metrics.items()}))
        with (DETAIL / f"{arm}_county_means.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(county_rows[0])); writer.writeheader(); writer.writerows(county_rows)
        with (DETAIL / f"{arm}_hour_means.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=["hour", *metrics]); writer.writeheader()
            for hour in range(216):
                writer.writerow(dict(hour=hour, **{name: float(v[active, hour if v.shape[1] == 216 else hour - 72].mean())
                                                   if v.shape[1] == 216 or hour >= 72 else "" for name, v in metrics.items()}))
        np.savez_compressed(DETAIL / f"{arm}_arrays.npz", idx=active, fips=F["fips"], system=F["system"],
                            fold=fold_ids, **metrics, **{f"P_{key}": v for key, v in predictions.items()})
        result = dict(folds=folds, distributions=summaries, by_hour_block=hourly,
                      county_mean_distributions={name: distribution([r[name] for r in county_rows]) for name in metrics},
                      counterfactuals={name: compare(p[active], predictions["own"][active], F, active) for name, p in predictions.items() if name != "own"},
                      exact_neutral_vs_zero_geo=compare(predictions["neutral_code"][active], predictions["zero_geo"][active], F, active))
        result["saturation"] = dict(
            share_state_norm_above_09=float((metrics["state_norm"][active, -144:] > .9).mean()),
            share_kernel_gate_below_001=float((metrics["kernel_gate"][active, -144:] < .01).mean()),
            share_kernel_gate_above_099=float((metrics["kernel_gate"][active, -144:] > .99).mean()),
            share_conditional_below_0005=float((metrics["conditional"][active] < .0005).mean()),
            share_conditional_above_0495=float((metrics["conditional"][active] > .495).mean()),
            share_occurrence_gate_below_001=float((metrics["occurrence_gate"][active] < .01).mean()))
        if fold1:
            ii = np.where(fold_ids == 1)[0]
            result["fold1_interventions"] = {name: compare(p[ii], predictions["own"][ii], F, ii) for name, p in fold1.items()}
            result["fold1_interventions"]["coupling_off"] = compare(predictions["coupling_off"][ii], predictions["own"][ii], F, ii)
            gamma = folds["1"]["geo_sim"]
            g0 = result["fold1_interventions"]["gamma_0"]
            result["fold1_gamma_finite_difference"] = dict(gamma=gamma,
                relative_pooled_mse_slope_from_zero=-g0["pooled_mse_change"] / max(gamma, 1e-30),
                relative_balanced_mse_slope_from_zero=-g0["balanced_mse_change"] / max(gamma, 1e-30),
                note="One-sided finite difference on held-out outcomes, no backward/optimization; tiny differences may be float32 noise.")
        report["models"][arm] = result
    target = HERE / "results/v1/kernel_state_diagnostics_s0.json"
    target.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print("wrote", target.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
