"""D04 statistical low-rank varying coefficients; no real-data work on import.

Prediction = N bN + X bX + row_sum((X Ug) * (G Vg))
             + row_sum((X Uc) * (C Vc)) + partially pooled county intercept.
X, N, G, C and y must already use TRAINING-ONLY scaling/bases. County intercepts
are estimated only from the supplied training outcomes; unseen counties get zero.
No event effects, split construction, outcome scaling, or causal identification
is supplied here. A low-rank fit is nonconvex: deterministic starts and the
reported local optimization diagnostics do not establish global optimality.

The objective is sum(normalized_w * residual**2) + lambda_county * sum(a**2)
 + ridge * (||bX||^2 + ||penalized_bN||^2
            + 0.5*(||Ug||^2+||Vg||^2+||Uc||^2+||Vc||^2)).
lambda_county = county_ridge / number_of_training_counties; default county_ridge
is 0.1. The first nonzero constant nuisance column is unpenalized, or specify
config['intercept_index']. All other nuisance coefficients are penalized.
Geographic/context factors have symmetric penalties; their factor coordinates
are not individually identified. Compare their product/response functions.

Each L-BFGS closure first profiles county intercepts in no-grad chunks, then
backpropagates chunkwise with those intercepts held fixed. This is the exact
profile-objective gradient (envelope theorem), up to float32 roundoff. It avoids
the full row-wise G x X matrix and retains no full-sample autograd graph.
Run this file with --self-test for synthetic checks; no panel is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


@dataclass
class D04Model:
    beta_n: np.ndarray
    beta_x: np.ndarray
    u_geo: np.ndarray
    v_geo: np.ndarray
    u_context: np.ndarray
    v_context: np.ndarray
    county_levels: np.ndarray
    county_intercepts: np.ndarray
    diagnostics: dict[str, Any]
    config: dict[str, Any]


def _matrix(a, name):
    if isinstance(a, torch.Tensor):
        if a.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        t = a.detach().to(dtype=torch.float32)
    else:
        t = torch.from_numpy(np.asarray(a, dtype=np.float32))
    if t.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    return t


def _county_codes(county):
    a = np.asarray(county).astype(str)
    if a.ndim != 1:
        raise ValueError("county must be one dimensional")
    return np.unique(a, return_inverse=True)


def fit_model(X, N, G, C, y, w, county, config, init_seed=0):
    """Fit one start; return a portable D04Model with convergence diagnostics.

    config: geo_rank (0/2/4), context_rank (default 2), ridge (.01/.1 or another
    nonnegative value), max_iter (default 60), batch_size (default 8192),
    county_ridge (.1), tolerance_grad (1e-5), tolerance_change (1e-9),
    history_size (10), intercept_index (optional), threads (default 2).
    Two starts are two explicit calls. Input arrays are never modified.
    """
    cfg = dict(config) if isinstance(config, Mapping) else vars(config).copy()
    torch.set_num_threads(int(cfg.get("threads", 2)))
    tx, tn, tg, tc = [_matrix(a, name) for a, name in
                     ((X, "X"), (N, "N"), (G, "G"), (C, "C"))]
    n, p = tx.shape
    if n == 0 or any(a.shape[0] != n for a in (tn, tg, tc)):
        raise ValueError("nonempty matrices must have the same row count")
    ty = torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(-1)
    tw = torch.as_tensor(np.asarray(w, dtype=np.float32)).reshape(-1)
    levels, codes = _county_codes(county)
    if len(ty) != n or len(tw) != n or len(codes) != n:
        raise ValueError("y, w and county must match the design rows")
    batch = int(cfg.get("batch_size", 8192))
    if batch <= 0:
        raise ValueError("batch_size must be positive")
    for a in (tx, tn, tg, tc):
        if any(not bool(torch.isfinite(a[s:s + batch]).all()) for s in range(0, n, batch)):
            raise ValueError("design matrices must be finite")
    if not bool(torch.isfinite(ty).all() and torch.isfinite(tw).all()):
        raise ValueError("y and weights must be finite")
    if bool((tw < 0).any()) or float(tw.sum()) <= 0:
        raise ValueError("weights must be nonnegative with a positive sum")
    tw = tw / tw.sum()
    ci = torch.from_numpy(codes.astype(np.int64, copy=False))
    nc = len(levels)
    cw = torch.zeros(nc, dtype=torch.float32).scatter_add_(0, ci, tw)
    ridge = float(cfg.get("ridge", .01))
    county_lambda = float(cfg.get("county_ridge", .1)) / nc
    if ridge < 0 or county_lambda <= 0:
        raise ValueError("ridge must be nonnegative and county_ridge positive")
    rg = int(cfg.get("geo_rank", 2))
    rc = int(cfg.get("context_rank", 2)) if tc.shape[1] else 0
    if rg < 0 or rc < 0 or (rg and not tg.shape[1]):
        raise ValueError("invalid interaction ranks")
    q, gd, cd = tn.shape[1], tg.shape[1], tc.shape[1]
    intercept = cfg.get("intercept_index", None)
    if intercept is None and q:
        lo, hi = tn.amin(dim=0), tn.amax(dim=0)
        candidates = torch.nonzero((hi - lo <= 1e-7) & (lo.abs() > 1e-7)).flatten()
        intercept = int(candidates[0]) if len(candidates) else None
    if intercept is not None and not 0 <= int(intercept) < q:
        raise ValueError("intercept_index outside nuisance columns")
    if intercept is not None:
        intercept = int(intercept)
        if (float(tn[:, intercept].max() - tn[:, intercept].min()) > 1e-7
                or abs(float(tn[0, intercept])) <= 1e-7):
            raise ValueError("unpenalized intercept column must be nonzero and constant")
    pmask = torch.ones(q, dtype=torch.float32)
    if intercept is not None:
        pmask[intercept] = 0
    gen = torch.Generator(device="cpu").manual_seed(int(init_seed))
    beta_n = torch.nn.Parameter(torch.zeros(q))
    beta_x = torch.nn.Parameter(torch.zeros(p))

    def factor(dim, rank):
        return torch.nn.Parameter(.05 * torch.randn(dim, rank, generator=gen)
                                  / np.sqrt(max(1, dim)))

    ug, vg, uc, vc = factor(p, rg), factor(gd, rg), factor(p, rc), factor(cd, rc)
    parameters = [a for a in (beta_n, beta_x, ug, vg, uc, vc) if a.numel()]
    if not parameters:
        raise ValueError("at least one nonempty model parameter is required")
    with torch.no_grad():
        if intercept is not None:
            beta_n[intercept] = (tw * ty).sum() / tn[0, intercept]

    def fixed(s, e):
        pred = tn[s:e] @ beta_n + tx[s:e] @ beta_x
        if rg:
            pred = pred + ((tx[s:e] @ ug) * (tg[s:e] @ vg)).sum(dim=1)
        if rc:
            pred = pred + ((tx[s:e] @ uc) * (tc[s:e] @ vc)).sum(dim=1)
        return pred

    @torch.no_grad()
    def profile_county():
        residual_sum = torch.zeros(nc, dtype=torch.float32)
        for s in range(0, n, batch):
            e = min(n, s + batch)
            residual_sum.scatter_add_(0, ci[s:e], tw[s:e] * (ty[s:e] - fixed(s, e)))
        return residual_sum / (cw + county_lambda)

    def penalty():
        return ridge * ((beta_n.square() * pmask).sum() + beta_x.square().sum()
                        + .5 * sum(a.square().sum() for a in (ug, vg, uc, vc)))

    evaluations = 0
    losses = []

    def closure():
        nonlocal evaluations
        for a in parameters:
            a.grad = None
        county_a = profile_county()
        total = float(county_lambda * county_a.square().sum())
        for s in range(0, n, batch):
            e = min(n, s + batch)
            residual = ty[s:e] - fixed(s, e) - county_a[ci[s:e]]
            chunk_loss = (tw[s:e] * residual.square()).sum()
            chunk_loss.backward()
            total += float(chunk_loss.detach())
        pen = penalty()
        pen.backward()
        total += float(pen.detach())
        if not np.isfinite(total) or any(not bool(torch.isfinite(a.grad).all()) for a in parameters):
            raise FloatingPointError("nonfinite profiled objective or gradient")
        evaluations += 1
        losses.append(total)
        return torch.tensor(total, dtype=torch.float32)

    initial_loss = float(closure())
    max_iter = int(cfg.get("max_iter", 60))
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    tol_grad = float(cfg.get("tolerance_grad", 1e-5))
    tol_change = float(cfg.get("tolerance_change", 1e-9))
    optimizer = torch.optim.LBFGS(
        parameters, lr=1., max_iter=max_iter, max_eval=max(1, 2 * max_iter),
        tolerance_grad=tol_grad, tolerance_change=tol_change,
        history_size=int(cfg.get("history_size", 10)), line_search_fn="strong_wolfe")
    optimizer.step(closure)
    final_loss = float(closure())
    final_grad = torch.cat([a.grad.reshape(-1) for a in parameters])
    grad_inf = float(final_grad.abs().max())
    grad_l2 = float(torch.linalg.vector_norm(final_grad))
    county_a = profile_county()
    state = optimizer.state[parameters[0]]
    n_iter = int(state.get("n_iter", 0))
    local_stop = n_iter < max_iter
    convergence = "gradient_tolerance" if grad_inf <= tol_grad else (
        "optimizer_stopped_before_budget" if local_stop else "iteration_budget")

    def arr(a):
        return a.detach().cpu().numpy().copy()

    diag = dict(initial_loss=initial_loss, loss=final_loss, gradnorm=grad_l2,
                grad_max_abs=grad_inf, iterations=n_iter, evaluations=evaluations,
                optimizer_func_evals=int(state.get("func_evals", 0)),
                convergence=convergence, gradient_converged=grad_inf <= tol_grad,
                finitefit=True, objective_improved=final_loss < initial_loss,
                train_rows=n, train_counties=nc, county_lambda=county_lambda,
                geo_rank=rg, context_rank=rc, ridge=ridge, init_seed=int(init_seed),
                batch_size=batch, unpenalized_intercept_index=intercept,
                parameter_count=sum(a.numel() for a in parameters),
                final_county_penalty=float(county_lambda * county_a.square().sum()),
                objective_history=losses)
    return D04Model(arr(beta_n), arr(beta_x), arr(ug), arr(vg), arr(uc), arr(vc),
                    levels.copy(), arr(county_a), diag, cfg)


def _county_values(model, county):
    labels = np.asarray(county).astype(str)
    if labels.ndim != 1:
        raise ValueError("county must be one dimensional")
    idx = np.searchsorted(model.county_levels, labels)
    safe = np.minimum(idx, len(model.county_levels) - 1)
    known = (idx < len(model.county_levels)) & (model.county_levels[safe] == labels)
    return np.where(known, model.county_intercepts[safe], 0).astype(np.float32)


def explain_components(model, X, N, G, C, county):
    """Return numpy nuisance/shared/context/geo/county vectors; new county=0."""
    x, n, g, c = [np.asarray(a, dtype=np.float32) for a in (X, N, G, C)]
    rows = x.shape[0]
    if any(a.ndim != 2 or a.shape[0] != rows for a in (x, n, g, c)):
        raise ValueError("prediction matrices must have matching rows")
    if len(county) != rows:
        raise ValueError("county must match prediction rows")
    dims = (len(model.beta_x), len(model.beta_n), model.v_geo.shape[0], model.v_context.shape[0])
    if tuple(a.shape[1] for a in (x, n, g, c)) != dims:
        raise ValueError("prediction columns do not match fitted model")
    return dict(nuisance=n @ model.beta_n, shared=x @ model.beta_x,
                context=((x @ model.u_context) * (c @ model.v_context)).sum(axis=1),
                geo=((x @ model.u_geo) * (g @ model.v_geo)).sum(axis=1),
                county=_county_values(model, county))


def predict(model, X, N, G, C, county):
    """Predict a caller-selected batch. The caller can stream larger arrays."""
    components = explain_components(model, X, N, G, C, county)
    return sum(components.values())


def self_test():
    """Synthetic response recovery and profiling checks; never accesses D."""
    rng = np.random.default_rng(123)
    nc, per = 72, 16
    county = np.repeat(np.arange(nc), per).astype(str)
    g = np.repeat(rng.normal(size=(nc, 4)), per, axis=0).astype(np.float32)
    c = np.repeat(rng.normal(size=(nc, 2)), per, axis=0).astype(np.float32)
    x = rng.normal(size=(nc * per, 6)).astype(np.float32)
    n = np.ones((len(x), 1), dtype=np.float32)
    w = np.ones(len(x), dtype=np.float32)
    train, test = np.arange(54 * per), np.arange(54 * per, nc * per)
    base = dict(ridge=.01, max_iter=60, batch_size=127, tolerance_grad=1e-5)
    results = {}
    for name, y in (("zero_geo", .7 * x[:, 0] - .35 * x[:, 2]),
                    ("pure_geo_interaction", 1.3 * x[:, 0] * g[:, 0])):
        scores = {}
        fitted = {}
        for rank in (0, 2):
            model = fit_model(x[train], n[train], g[train], c[train], y[train],
                              w[train], county[train], dict(base, geo_rank=rank), 11)
            pred = predict(model, x[test], n[test], g[test], c[test], county[test])
            comp = explain_components(model, x[test], n[test], g[test], c[test], county[test])
            assert np.all(comp["county"] == 0), "unseen county acquired intercept"
            assert np.isfinite(pred).all() and model.diagnostics["finitefit"]
            assert model.diagnostics["objective_improved"]
            np.testing.assert_allclose(pred, sum(comp.values()), rtol=1e-6, atol=1e-6)
            scores[rank] = float(np.mean((pred - y[test]) ** 2))
            fitted[rank] = model
        if name == "pure_geo_interaction":
            assert scores[2] < .2 * scores[0], scores
        else:
            assert scores[2] < .01 and scores[2] < scores[0] + .005, scores
            assert np.linalg.norm(fitted[2].v_geo @ fitted[2].u_geo.T) < .1
        results[name] = dict(mse_rank0=scores[0], mse_rank2=scores[2],
                             geo_iterations=fitted[2].diagnostics["iterations"])
    # Independent start and direct closed-form county intercept check.
    alt = fit_model(x[train], n[train], g[train], c[train], y[train], w[train],
                    county[train], dict(base, geo_rank=2), 19)
    alt_pred = predict(alt, x[test], n[test], g[test], c[test], county[test])
    alt_mse = float(np.mean((alt_pred - y[test]) ** 2))
    assert alt_mse < .001
    pieces = explain_components(alt, x[train], n[train], g[train], c[train], county[train])
    fixed_pred = sum(v for k, v in pieces.items() if k != "county")
    for label, observed_a in zip(alt.county_levels, alt.county_intercepts):
        use = county[train] == label
        normalized_w = w[train] / w[train].sum()
        expected_a = ((normalized_w[use] * (y[train][use] - fixed_pred[use])).sum()
                      / (normalized_w[use].sum() + alt.diagnostics["county_lambda"]))
        np.testing.assert_allclose(observed_a, expected_a, rtol=1e-4, atol=2e-6)
    results["second_start_interaction_mse"] = alt_mse
    # Compare the envelope gradient with an independently differentiated profile.
    xx = torch.tensor([[1., .2], [2., -.3], [.5, .8], [-1., .1]])
    yy, ww = torch.tensor([.2, 1., -.1, .4]), torch.tensor([.1, .2, .3, .4])
    ii = torch.tensor([0, 0, 1, 1]); lam = .05
    bb = torch.tensor([.3, -.2], requires_grad=True)
    cw = torch.zeros(2).scatter_add_(0, ii, ww)
    a = torch.zeros(2).scatter_add_(0, ii, ww * (yy - xx @ bb)) / (cw + lam)
    objective = (ww * (yy - xx @ bb - a[ii]).square()).sum() + lam * a.square().sum()
    exact, = torch.autograd.grad(objective, bb)
    envelope = -2 * xx.T @ (ww * (yy - xx @ bb.detach() - a.detach()[ii]))
    torch.testing.assert_close(exact, envelope, rtol=1e-5, atol=1e-7)
    results["profile_gradient_max_error"] = float((exact - envelope).abs().max())
    return results


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        parser.print_help()
