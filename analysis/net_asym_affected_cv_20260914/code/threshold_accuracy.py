#!/usr/bin/env python3
"""How accurate is the paper's decision threshold Lambda = 1 on these data? Post hoc, descriptive.

PI question (2026-09-15): the share of events assigned to each model does not matter; what matters is whether the
threshold is right. With n = the number of independent training events of each event's outer fold (the paper's n),
the paper predicts that NET is better exactly when Lambda = n S / nu^2 > 1. This script locates the estimated Lambda at
which the realized neural preference flips, with uncertainty from resampling overlap components, at two resolutions:
- event level: the 26 events of results/EVENT_FEATURES.csv; realized Delta from the CV test (affected);
- origin level: every (event, origin) of results/EVENT_FEATURES_BY_ORIGIN.csv; realized Delta recomputed from the saved
  prediction shards (affected windows, seed mean), with Lambda_t = n_train S_hat_t / nu_hat_t^2.

Two boundary estimates:
- MSE-optimal threshold: the tau that minimises the event-equal MSE of "NET if Lambda > tau";
- logistic boundary: the Lambda at which P(NET better) = 1/2 on log Lambda. The slope is lightly ridge-penalised
  against separation; the intercept is not penalised, so the boundary is not pulled towards 1.
Origins whose bias-corrected signal is not positive (S_hat <= 0) have Lambda_t = 0; the theory assigns them to ASYM.
They enter the MSE-optimal analysis and are left out of the logistic fit. The MSE-optimal procedure is also applied to
the mean outage level at the origins.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
W = HERE.parent
KINDS = ('NET', 'ASYM')
FINAL_SEEDS = (9201, 9202, 9203, 9204, 9205)
B_EVENT, B_ORIGIN, RNG_SEED = 4000, 1000, 20260916


def optimal_threshold(values, net_mse, asym_mse, weights):
    order = np.argsort(values, kind='mergesort')
    v, n, a, w = values[order], net_mse[order], asym_mse[order], weights[order]
    loss = np.r_[0.0, np.cumsum(w * a)] + np.r_[np.cumsum((w * n)[::-1])[::-1], 0.0]
    loss = np.where(np.r_[True, v[1:] != v[:-1], True], loss, np.inf)
    ks = np.flatnonzero(loss <= loss.min() * (1 + 1e-12))
    k = int(ks[len(ks) // 2])
    if k == 0:
        tau = -np.inf
    elif k == len(v):
        tau = np.inf
    else:
        tau = float(np.sqrt(v[k - 1] * v[k])) if v[k - 1] > 0 else float(0.5 * (v[k - 1] + v[k]))
    return float(tau), float(loss[k] / w.sum())


def rule_loss(values, net_mse, asym_mse, weights, tau):
    return float(np.sum(weights * np.where(values > tau, net_mse, asym_mse)) / weights.sum())


def logistic_boundary(x, y, w):
    xm = float(np.average(x, weights=w)); xc = x - xm

    def nll(p):
        z = p[0] + p[1] * xc
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)) + 1e-3 * p[1] ** 2)

    def grad(p):
        z = p[0] + p[1] * xc; r = w * (1 / (1 + np.exp(-z)) - y)
        return np.array([r.sum(), (r * xc).sum() + 2e-3 * p[1]])

    a, b = minimize(nll, np.zeros(2), jac=grad, method='BFGS').x
    return (float(np.exp(xm - a / b)) if b > 0 else float('nan')), float(b)


def ci(samples, log=True):
    s = np.asarray([x for x in samples if np.isfinite(x) and (x > 0 or not log)])
    if len(s) == 0:
        return {'p2.5': None, 'p50': None, 'p97.5': None, 'finite_share': 0.0}
    q = np.exp(np.percentile(np.log(s), [2.5, 50, 97.5])) if log else np.percentile(s, [2.5, 50, 97.5])
    return {'p2.5': float(q[0]), 'p50': float(q[1]), 'p97.5': float(q[2]), 'finite_share': float(len(s) / len(samples))}


def origin_realized(lock):
    fold_of = {e: f['fold'] for f in lock['folds'] for e in f['events']}
    out = []
    for e, k in sorted(fold_of.items()):
        T = np.load(W / f'predictions/truth/{e}.npz'); truth = T['truth'].astype(np.float64)
        per = {}
        for kind in KINDS:
            s = []
            for seed in FINAL_SEEDS:
                z = np.load(W / f'predictions/fold{k}/{kind}/seed{seed}/{e}.npz')
                if not (np.array_equal(z['row'], T['row']) and np.array_equal(z['t'], T['t'])):
                    raise SystemExit(f'window keys differ: {e} {kind} {seed}')
                s.append(((z['pred'].astype(np.float64) - truth) ** 2).mean(axis=1))
            per[kind] = np.mean(s, axis=0)
        df = pd.DataFrame({'t': T['t'], 'affected': T['affected'].astype(bool), 'NET': per['NET'], 'ASYM': per['ASYM']})
        g = df[df.affected].groupby('t')[['NET', 'ASYM']].mean().reset_index(); g['event'] = e
        out.append(g)
    return pd.concat(out, ignore_index=True)


def boot_weights(units, comp_of, rng):
    comps = np.array(sorted(set(comp_of.values())))
    draw = rng.choice(comps, size=len(comps), replace=True)
    mult = pd.Series(draw).value_counts()
    return np.array([mult.get(comp_of[u], 0) for u in units], dtype=float)


def main():
    lock = json.loads((W / 'locks/FOLDS.json').read_text())
    comp_of = {e: i for i, c in enumerate(lock['components']) for e in c['events']}
    rng = np.random.default_rng(RNG_SEED)
    E = pd.read_csv(W / 'results/EVENT_FEATURES.csv')
    lam_e = E.lambda_hat_train_events.to_numpy(); y_e = (E.realized_winner == 'NET').to_numpy().astype(float)
    net_e, asym_e = E.NET_path_mse.to_numpy(), E.ASYM_path_mse.to_numpy(); lvl = E.mean_origin_state.to_numpy()
    w1 = np.ones(len(E))
    tau_e, loss_e = optimal_threshold(lam_e, net_e, asym_e, w1)
    lb_e, slope_e = logistic_boundary(np.log(lam_e), y_e, w1)
    tau_l, loss_l = optimal_threshold(lvl, net_e, asym_e, w1)
    bt, bl, bv = [], [], []
    for _ in range(B_EVENT):
        wb = boot_weights(E.event, comp_of, rng)
        m = wb > 0
        bt.append(optimal_threshold(lam_e[m], net_e[m], asym_e[m], wb[m])[0])
        bl.append(logistic_boundary(np.log(lam_e[m]), y_e[m], wb[m])[0])
        bv.append(optimal_threshold(lvl[m], net_e[m], asym_e[m], wb[m])[0])
    n_train_mean = float(E.n_train_events.mean())
    event = {'units': int(len(E)), 'n_train_events_mean': n_train_mean,
             'mse_optimal_lambda_threshold': {'point': tau_e, 'bootstrap': ci(bt), 'share_above_1': float(np.mean(np.array(bt) > 1)),
                                              'implied_n_eff': n_train_mean / tau_e if tau_e > 0 else None},
             'logistic_boundary_lambda': {'point': lb_e, 'slope': slope_e, 'bootstrap': ci(bl),
                                          'share_above_1': float(np.mean(np.nan_to_num(np.array(bl), nan=-1) > 1))},
             'event_equal_mse': {'rule_lambda_gt_1': rule_loss(lam_e, net_e, asym_e, w1, 1.0), 'rule_at_mse_optimal_tau_in_sample': loss_e,
                                 'always_NET': float(net_e.mean()), 'always_ASYM': float(asym_e.mean()),
                                 'better_of_two': float(np.minimum(net_e, asym_e).mean())},
             'mean_outage_level_threshold': {'point_pct': 100 * tau_l, 'bootstrap_pct': {k: (None if v is None else 100 * v) if k != 'finite_share' else v
                                                                                       for k, v in ci(bv).items()},
                                             'in_sample_event_equal_mse': loss_l}}

    O = pd.read_csv(W / 'results/EVENT_FEATURES_BY_ORIGIN.csv'); O = O[O.status == 'OK']
    Rz = origin_realized(lock)
    M = O.merge(Rz, on=['event', 't'], how='inner').merge(E[['event', 'n_train_events', 'delta_cv']], on='event')
    M['lam'] = np.where(M.S_hat > 0, M.n_train_events * M.S_hat / M.nu2_hat, 0.0)
    M['net_better'] = (M.NET < M.ASYM).astype(float)
    M['w'] = 1.0 / M.groupby('event').t.transform('size')
    chk = M.groupby('event').apply(lambda g: float((g.NET - g.ASYM).mean())).rename('origin_equal_delta').reset_index().merge(E[['event', 'delta_cv']], on='event')
    lam_o, net_o, asym_o, w_o = M.lam.to_numpy(), M.NET.to_numpy(), M.ASYM.to_numpy(), M.w.to_numpy()
    pos = lam_o > 0
    tau_o, loss_o = optimal_threshold(lam_o, net_o, asym_o, w_o)
    lb_o, slope_o = logistic_boundary(np.log(lam_o[pos]), M.net_better.to_numpy()[pos], w_o[pos])
    bt, bl = [], []
    for _ in range(B_ORIGIN):
        wb = boot_weights(M.event, comp_of, rng) * w_o
        m = wb > 0; mp = m & pos
        bt.append(optimal_threshold(lam_o[m], net_o[m], asym_o[m], wb[m])[0])
        bl.append(logistic_boundary(np.log(lam_o[mp]), M.net_better.to_numpy()[mp], wb[mp])[0])
    zero = M[~pos]
    bins = pd.qcut(np.log(M.loc[pos, 'lam']), 8)
    B = M.loc[pos].assign(bin=bins).groupby('bin', observed=True).apply(
        lambda g: pd.Series({'lambda_median': float(g.lam.median()), 'origins': int(len(g)),
                             'share_NET_better_event_weighted': float(np.average(g.net_better, weights=g.w)),
                             'mean_relative_delta': float(np.average((g.NET - g.ASYM) / g.NET, weights=g.w))})).reset_index(drop=True)
    B = pd.concat([pd.DataFrame([{'lambda_median': 0.0, 'origins': int(len(zero)),
                                  'share_NET_better_event_weighted': float(np.average(zero.net_better, weights=zero.w)) if len(zero) else np.nan,
                                  'mean_relative_delta': float(np.average((zero.NET - zero.ASYM) / zero.NET, weights=zero.w)) if len(zero) else np.nan}]), B],
                  ignore_index=True)
    origin = {'units': int(len(M)), 'origins_with_nonpositive_signal': int((~pos).sum()),
              'check_spearman_origin_equal_vs_cv_delta': float(chk[['origin_equal_delta', 'delta_cv']].corr(method='spearman').iloc[0, 1]),
              'mse_optimal_lambda_threshold': {'point': tau_o, 'bootstrap': ci(bt), 'share_above_1': float(np.mean(np.array(bt) > 1))},
              'logistic_boundary_lambda': {'point': lb_o, 'slope': slope_o, 'bootstrap': ci(bl),
                                           'share_above_1': float(np.mean(np.nan_to_num(np.array(bl), nan=-1) > 1))},
              'event_equal_mse': {'rule_lambda_gt_1': rule_loss(lam_o, net_o, asym_o, w_o, 1.0), 'rule_at_mse_optimal_tau_in_sample': loss_o,
                                  'always_NET': rule_loss(lam_o, net_o, asym_o, w_o, -np.inf), 'always_ASYM': rule_loss(lam_o, net_o, asym_o, w_o, np.inf),
                                  'better_of_two': float(np.sum(w_o * np.minimum(net_o, asym_o)) / w_o.sum())},
              'binned': B.to_dict('records')}
    out = {'status': 'POST_HOC_DESCRIPTIVE', 'question': 'is Lambda = 1 (n = independent training events) where the neural preference flips?',
           'bootstrap_unit': 'overlap components', 'event_level': event, 'origin_level': origin}
    (W / 'results/THRESHOLD_ACCURACY.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    M.to_csv(W / 'results/THRESHOLD_ORIGIN_LEVEL.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    col = np.where(y_e > 0, '#1f77b4', '#d62728')
    ax[0].scatter(lam_e, E.delta_cv, c=col, s=40)
    for xx, yy, name in zip(lam_e, E.delta_cv, E.event):
        ax[0].annotate(name[2:], (xx, yy), fontsize=6, xytext=(3, 2), textcoords='offset points')
    c_e = event['mse_optimal_lambda_threshold']['bootstrap']
    ax[0].axvspan(c_e['p2.5'], c_e['p97.5'], color='0.85', zorder=0)
    ax[0].axvline(tau_e, color='k', ls='--', lw=1, label=f'MSE-optimal threshold {tau_e:.2f}')
    ax[0].axvline(1.0, color='#2ca02c', lw=1.5, label='paper threshold Lambda = 1')
    ax[0].set_xscale('log'); ax[0].set_yscale('symlog', linthresh=1e-5); ax[0].axhline(0, color='k', lw=.8)
    ax[0].set_xlabel('Lambda_hat = n_train S_hat / nu_hat^2 (event)'); ax[0].set_ylabel('CV test Delta (affected), positive favours ASYM')
    ax[0].set_title('Event level (blue = NET better, red = ASYM better; band = bootstrap 95% of threshold)', fontsize=8)
    ax[0].legend(fontsize=7)
    xb = B[B.lambda_median > 0]
    ax[1].plot(xb.lambda_median, xb.share_NET_better_event_weighted, 'o-', color='k', label='share of origins where NET is better')
    xs = np.exp(np.linspace(np.log(M.loc[pos, 'lam'].min()), np.log(M.loc[pos, 'lam'].max()), 200))
    ax[1].axvline(1.0, color='#2ca02c', lw=1.5, label='paper threshold Lambda = 1')
    if np.isfinite(lb_o):
        c_o = origin['logistic_boundary_lambda']['bootstrap']
        if c_o['p2.5'] is not None:
            ax[1].axvspan(c_o['p2.5'], c_o['p97.5'], color='0.85', zorder=0)
        ax[1].axvline(lb_o, color='k', ls='--', lw=1, label=f'logistic 50% boundary {lb_o:.2f}')
    ax[1].axhline(0.5, color='0.5', lw=.8); ax[1].set_xscale('log'); ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('Lambda_hat_t (origin, octile bins; origins with S_hat <= 0 not shown)'); ax[1].set_ylabel('share NET better (event-weighted)')
    ax[1].set_title('Origin level', fontsize=8); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(W / 'figures/threshold_accuracy.png', dpi=150); plt.close(fig)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
