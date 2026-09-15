#!/usr/bin/env python3
"""Event-level data features for the NET/ASYM choice, following the paper's conditional-risk comparison.

Post hoc and descriptive: written after the affected-county CV results were seen, on the PI decision of
2026-09-15 (event level; report the solvable-benchmark estimates and the readable features). No model
prediction enters any feature; model results are joined only afterwards to check the decision.

Replicates. Within one event, the affected county-events scored in the CV test are grouped by forecast origin t.
Counties sharing an origin see the same storm timing and serve as approximate replicates of the conditional
response. Their local weather and history still differ and they share common shocks, so they are not the
independent identical-input events of the paper's Assumption 1; every estimate below is a descriptive plug-in.

Benchmark at each origin (the paper's solvable comparison with a nonzero common origin):
  forcing  x_k = share of the origin's counties under damaging weather at hour t+k+1 (gust, precipitation or
           snowfall at or above the 90th percentile of its positive values over all affected county-hours;
           inputs only);
  dual     g_{k+1} = u x_k + (1 - u x_k - r) g_k                 (u, r)
  net      g_{k+1} = u x_k + (1 - u x_k - r) g_k + b g_k^2       (u, r, b)
  Both are fitted by full-path least squares to the cross-county mean path from the mean origin g_0.
  At the dual fit, J = [dg/du, dg/dr], t = dg/db and q = (I - P_J) t is the net model's extra response direction.
  County deviations are measured from the dual reference started at each county's own origin, which is exact for
  the affine dual recursion: E_i = Y_i - (g + (y0_i - g_0) c) with c = dg/dg_0.
  S = {q' mean(E)}^2 / q'q,  nu^2 = q' Cov(E) q / q'q,  sigma_x^2 = tr Cov(E) / H,  n = counties at the origin.
Cross-fitting (Appendix F): the dual reference and q come from a random half of the counties and are applied to
  the other half, Z_i = q'E_i / |q|; S_hat = mean(Z)^2 - var(Z)/n_half and nu_hat^2 = var(Z); 20 random splits,
  both directions.
Event level: Delta_hat = mean over origins of (nu_hat^2 / n - S_hat) / H; positive favours ASYM, as in Theorem 1.
  Lambda_hat = mean(n S_hat) / mean(nu_hat^2); Lambda_hat < 1 favours ASYM. A second version replaces n by the
  number of independent training events of the event's outer fold. Readable features: sigma_x and n.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
W = HERE.parent
sys.path.insert(0, str(HERE))
import cv_data as CD   # noqa: E402

UD = CD.UD
H, SPLITS, MIN_COUNTIES, RNG_SEED = 24, 20, 10, 20260915
FORCING_CHANNELS = ('gust', 'precip', 'snowfall')
STARTS = ((0.05, 0.05), (0.20, 0.10), (0.02, 0.30))


def simulate(u, r, b, x, g0):
    """Path g_1..g_H with its sensitivities to (u, r, b) and to the origin g_0."""
    g = np.empty(H + 1); g[0] = g0
    J = np.zeros((H + 1, 3)); c = np.ones(H + 1)
    for k in range(H):
        a = 1.0 - u * x[k] - r + 2.0 * b * g[k]
        J[k + 1, 0] = a * J[k, 0] + x[k] * (1.0 - g[k])
        J[k + 1, 1] = a * J[k, 1] - g[k]
        J[k + 1, 2] = a * J[k, 2] + g[k] ** 2
        c[k + 1] = a * c[k]
        g[k + 1] = u * x[k] + (1.0 - u * x[k] - r) * g[k] + b * g[k] ** 2
    return g[1:], J[1:], c[1:]


def fit(mu, x, g0, net, starts=STARTS):
    npar = 3 if net else 2
    lb = [0.0, 0.0] + ([-20.0] if net else [])
    ub = [1.0, 1.0] + ([20.0] if net else [])

    def fun(p):
        return simulate(p[0], p[1], p[2] if net else 0.0, x, g0)[0] - mu

    def jac(p):
        return simulate(p[0], p[1], p[2] if net else 0.0, x, g0)[1][:, :npar]

    best = None
    for s in starts:
        p0 = np.array([min(max(s[0], 1e-6), 1 - 1e-6), min(max(s[1], 1e-6), 1 - 1e-6)] + ([0.0] if net else []))
        res = least_squares(fun, p0, jac=jac, bounds=(lb, ub), method='trf')
        if best is None or res.cost < best.cost:
            best = res
    return best.x, float(2 * best.cost / H)


def extra_direction(u, r, x, g0):
    g, J, c = simulate(u, r, 0.0, x, g0)
    JA, t = J[:, :2], J[:, 2]
    q = t - JA @ (np.linalg.pinv(JA) @ t)
    return g, c, q


def origin_features(Y, y0, x, rng):
    n = len(y0); mu = Y.mean(axis=0); g0 = float(y0.mean())
    (u, r), mse_dual = fit(mu, x, g0, net=False)
    (_, _, b), mse_net = fit(mu, x, g0, net=True)
    g, c, q = extra_direction(u, r, x, g0)
    E = Y - (g[None, :] + (y0 - g0)[:, None] * c[None, :])
    Sig = np.cov(E, rowvar=False)
    out = {'g0': g0, 'forcing_mean': float(x.mean()), 'u': float(u), 'r': float(r), 'b_net': float(b),
           'mse_dual_fit_to_mean': mse_dual, 'mse_net_fit_to_mean': mse_net, 'sigma2': float(np.trace(Sig) / H),
           'mean_path_peak': float(mu.max())}
    qq = float(q @ q)
    if qq < 1e-24:
        out.update(S_plug=np.nan, nu2_plug=np.nan, S_hat=np.nan, nu2_hat=np.nan)
        return out
    out['S_plug'] = float((q @ E.mean(axis=0)) ** 2 / qq)
    out['nu2_plug'] = float(q @ Sig @ q / qq)
    S_list, v_list = [], []
    for _ in range(SPLITS):
        perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
        for fs, es in ((A, B), (B, A)):
            g0f = float(y0[fs].mean())
            (uf, rf), _ = fit(Y[fs].mean(axis=0), x, g0f, net=False, starts=((u, r),))
            gf, cf, qf = extra_direction(uf, rf, x, g0f)
            nq = float(np.sqrt(qf @ qf))
            if nq < 1e-12:
                continue
            Z = (Y[es] - (gf[None, :] + (y0[es] - g0f)[:, None] * cf[None, :])) @ qf / nq
            v = float(Z.var(ddof=1))
            S_list.append(float(Z.mean() ** 2 - v / len(es))); v_list.append(v)
    out['S_hat'] = float(np.mean(S_list)) if S_list else np.nan
    out['nu2_hat'] = float(np.mean(v_list)) if v_list else np.nan
    return out


def figures(E):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    F = W / 'figures'; F.mkdir(parents=True, exist_ok=True)
    col = np.where(E.realized_winner == 'ASYM', '#d62728', '#1f77b4')

    def label(ax):
        for xx, yy, name in zip(ax.collections[0].get_offsets()[:, 0], ax.collections[0].get_offsets()[:, 1], E.event):
            ax.annotate(name[2:], (xx, yy), fontsize=6, xytext=(3, 2), textcoords='offset points')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, tag, lab in ((axes[0], 'counties', 'n = counties per origin'), (axes[1], 'train_events', 'n = training events of the fold')):
        ax.scatter(E[f'delta_hat_{tag}'], E.delta_cv, c=col, s=40); label(ax)
        ax.set_xscale('symlog', linthresh=1e-7); ax.set_yscale('symlog', linthresh=1e-5)
        ax.axvline(0, color='k', lw=.8); ax.axhline(0, color='k', lw=.8)
        ax.set_xlabel(f'benchmark Delta_hat, positive favours ASYM ({lab})'); ax.set_ylabel('CV test Delta (affected), positive favours ASYM')
    axes[0].set_title('Paper benchmark estimate vs realized neural Delta (red = ASYM won, blue = NET won)', fontsize=9)
    fig.tight_layout(); fig.savefig(F / 'event_features_decision.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].scatter(E.sigma_x_pp, E.delta_cv, c=col, s=40); label(axes[0])
    axes[0].set_yscale('symlog', linthresh=1e-5); axes[0].axhline(0, color='k', lw=.8)
    axes[0].set_xlabel('conditional path dispersion sigma_x (percentage points)'); axes[0].set_ylabel('CV test Delta (affected)')
    axes[1].scatter(E.signal_to_noise_per_replicate, E.delta_cv, c=col, s=40); label(axes[1])
    axes[1].set_xscale('symlog', linthresh=1e-3); axes[1].set_yscale('symlog', linthresh=1e-5)
    axes[1].axhline(0, color='k', lw=.8); axes[1].axvline(0, color='k', lw=.8)
    axes[1].set_xlabel('S_hat / nu_hat^2 (signal to noise per replicate)'); axes[1].set_ylabel('CV test Delta (affected)')
    fig.tight_layout(); fig.savefig(F / 'event_features_readable.png', dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=str(W.parents[1])); a = ap.parse_args()
    lock = json.loads((W / 'locks/FOLDS.json').read_text())
    idx = pd.read_csv(W / 'locks/CV_WINDOW_INDEX.csv.gz',
                      dtype={'fips': str, 'event': str, 'origin_utc': str, 'type': str, 'type_group': str})
    c = UD.load_corpus(Path(a.root))
    aff = idx[idx.affected].copy()
    rows = np.unique(aff.row.to_numpy())
    thresholds, dmg = {}, np.zeros(c.Y.shape, dtype=bool)
    for name in FORCING_CHANNELS:
        v = c.X[rows][:, :, UD.RAW_CHANNELS.index(name)].astype(np.float64)
        pos = v[np.isfinite(v) & (v > 0)]
        thresholds[name] = float(np.quantile(pos, 0.9))
        dmg[rows] = dmg[rows] | (v >= thresholds[name])
    fold_of = {e: f['fold'] for f in lock['folds'] for e in f['events']}
    n_train = {e: sum(len(f['events']) for f in lock['folds'] if f['fold'] in lock['roles'][str(fold_of[e])]['train_folds'])
               for e in fold_of}
    steps = np.arange(1, H + 1); recs = []
    for ei, (e, ge) in enumerate(aff.groupby('event', sort=True)):
        for t, gt in ge.groupby('t', sort=True):
            r = gt.row.to_numpy(); t = int(t)
            base = {'event': e, 't': t, 'origin_utc': gt.origin_utc.iloc[0], 'n': int(len(r))}
            if len(r) < MIN_COUNTIES:
                recs.append({**base, 'status': 'NA_FEW_COUNTIES'}); continue
            Y = c.Y[r[:, None], t + steps[None, :]].astype(np.float64)
            y0 = c.Y[r, t].astype(np.float64)
            x = dmg[r[:, None], t + steps[None, :]].mean(axis=0).astype(np.float64)
            recs.append({**base, 'status': 'OK', **origin_features(Y, y0, x, np.random.default_rng([RNG_SEED, ei, t]))})
        print(f'{e}: {ge.t.nunique()} origins', flush=True)
    O = pd.DataFrame(recs)
    ok = O[(O.status == 'OK') & O.S_hat.notna()].copy()
    ok['delta_hat_t'] = (ok.nu2_hat / ok.n - ok.S_hat) / H
    P = pd.read_csv(W / 'results/CV_EVENT_PAIRED.csv'); P = P[P.population == 'affected'].set_index('event')
    ev = []
    for e, g in ok.groupby('event'):
        nt = n_train[e]; p = P.loc[e]
        ev.append({'event': e, 'type': lock['event_types'][e], 'fold': fold_of[e], 'origins_used': int(len(g)),
                   'n_counties_mean': float(g.n.mean()), 'n_train_events': int(nt),
                   'sigma_x_pp': float(100 * np.sqrt(g.sigma2.mean())), 'S_hat': float(g.S_hat.mean()), 'nu2_hat': float(g.nu2_hat.mean()),
                   'signal_to_noise_per_replicate': float(g.S_hat.mean() / g.nu2_hat.mean()),
                   'lambda_hat_counties': float((g.n * g.S_hat).mean() / g.nu2_hat.mean()),
                   'delta_hat_counties': float(g.delta_hat_t.mean()),
                   'lambda_hat_train_events': float(nt * g.S_hat.mean() / g.nu2_hat.mean()),
                   'delta_hat_train_events': float(((g.nu2_hat / nt - g.S_hat) / H).mean()),
                   'lambda_plug_counties': float((g.n * g.S_plug).mean() / g.nu2_plug.mean()),
                   'b_net_median': float(g.b_net.median()), 'mean_origin_state': float(g.g0.mean()),
                   'forcing_mean': float(g.forcing_mean.mean()), 'mean_path_peak_max': float(g.mean_path_peak.max()),
                   'delta_cv': float(p.delta_seed_mean), 'seeds_delta_positive': int(p.seeds_delta_positive),
                   'NET_path_mse': float(p.NET_path_mse), 'ASYM_path_mse': float(p.ASYM_path_mse)})
    E = pd.DataFrame(ev)
    E['realized_winner'] = np.where(E.delta_cv > 0, 'ASYM', 'NET')
    for tag in ('counties', 'train_events'):
        E[f'rule_{tag}'] = np.where(E[f'delta_hat_{tag}'] > 0, 'ASYM', 'NET')

    def risk(choice):
        return float(np.where(choice == 'ASYM', E.ASYM_path_mse, E.NET_path_mse).mean())

    decision = {'always_NET_event_equal_mse': float(E.NET_path_mse.mean()), 'always_ASYM_event_equal_mse': float(E.ASYM_path_mse.mean()),
                'better_of_two_per_event_mse': float(np.minimum(E.NET_path_mse, E.ASYM_path_mse).mean())}
    for tag in ('counties', 'train_events'):
        decision[f'rule_n_{tag}'] = {'event_equal_mse': risk(E[f'rule_{tag}']),
                                     'share_matching_realized_winner': float((E[f'rule_{tag}'] == E.realized_winner).mean()),
                                     'share_choosing_ASYM': float((E[f'rule_{tag}'] == 'ASYM').mean())}
    E['delta_cv_relative'] = E.delta_cv / E.NET_path_mse
    assoc = {}
    for col in ('delta_hat_counties', 'delta_hat_train_events', 'lambda_hat_counties', 'signal_to_noise_per_replicate',
                'S_hat', 'nu2_hat', 'sigma_x_pp', 'n_counties_mean', 'b_net_median', 'mean_origin_state', 'mean_path_peak_max'):
        rho, pv = spearmanr(E[col], E.delta_cv); rho_rel, _ = spearmanr(E[col], E.delta_cv_relative)
        assoc[col] = {'spearman_with_delta_cv': float(rho), 'descriptive_p': float(pv), 'spearman_with_relative_delta_cv': float(rho_rel)}
    R = W / 'results'
    O.to_csv(R / 'EVENT_FEATURES_BY_ORIGIN.csv', index=False)
    E.to_csv(R / 'EVENT_FEATURES.csv', index=False)
    out = {'status': 'POST_HOC_DESCRIPTIVE', 'written_after_cv_results_were_seen': True,
           'delta_convention': 'positive favours ASYM for both Delta_hat and the CV test Delta',
           'forcing_thresholds_90pct_of_positive_values': thresholds, 'H': H, 'splits': SPLITS, 'min_counties': MIN_COUNTIES,
           'origins_total': int(len(O)), 'origins_used': int(len(ok)), 'events': int(len(E)),
           'decision': decision, 'spearman_association_over_events': assoc,
           'caveat': 'counties within an event share shocks and differ in local weather; plug-in estimates, not the paper\'s '
                     'unbiased estimator under Assumption 1; decision rule not validated as an automatic selector'}
    (R / 'EVENT_FEATURES_SUMMARY.json').write_text(json.dumps(out, indent=1) + '\n')
    figures(E)
    print(json.dumps({'decision': decision, 'assoc_delta_hat_counties': assoc['delta_hat_counties'],
                      'assoc_sigma_x': assoc['sigma_x_pp']}, indent=1))


if __name__ == '__main__':
    main()
