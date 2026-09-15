#!/usr/bin/env python3
"""Compare the estimated Lambda between times where NET was better and times where ASYM was better. Post hoc.

PI suggestion (2026-09-15): compare the mean and median of Lambda over the forecast origins where NET was better with
those where ASYM was better. If the paper's threshold is right, NET-better origins should sit above Lambda = 1 and
ASYM-better origins below it.

Inputs:
- results/THRESHOLD_ORIGIN_LEVEL.csv, per event and origin: Lambda_t = n_train S_hat_t / nu_hat_t^2 (0 when
  S_hat_t <= 0) and the seed-mean affected path MSE of both models;
- results/EVENT_FEATURES.csv, the event level.

Per group: units, mean, median, quartiles, share above 1, share without a positive signal. Between groups: difference
and ratio of medians and of means, and the rank probability P(Lambda of a NET-better time > Lambda of an ASYM-better
time). Also the prevalence-free separating threshold, which maximises TPR - FPR for "NET if Lambda > tau" (Youden), and
the same quantities at tau = 1, plus the separation (share above tau among NET-better times minus among ASYM-better
times) on a fixed grid of tau. Every 95% interval comes from resampling overlap components.
Sensitivities: origins with a positive signal only; origins where the two models differ by at least 5% of NET's MSE.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
W = HERE.parent
B, RNG_SEED, CLEAR = 2000, 20260917, 0.05
GRID = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)


def summary(lam):
    lam = np.asarray(lam, float)
    return {'units': int(len(lam)), 'mean': float(lam.mean()), 'median': float(np.median(lam)),
            'q25': float(np.percentile(lam, 25)), 'q75': float(np.percentile(lam, 75)),
            'share_above_1': float((lam > 1).mean()), 'share_without_positive_signal': float((lam <= 0).mean())}


def youden(lam, nb):
    a, b = np.sort(lam[nb]), np.sort(lam[~nb])
    v = np.unique(lam)
    cands = np.r_[-np.inf, (v[:-1] + v[1:]) / 2, np.inf]
    tpr = 1 - np.searchsorted(a, cands, side='right') / len(a)
    fpr = 1 - np.searchsorted(b, cands, side='right') / len(b)
    J = tpr - fpr
    ks = np.flatnonzero(J >= J.max() - 1e-12); k = int(ks[len(ks) // 2])
    return float(cands[k]), float(J[k]), float(tpr[k]), float(fpr[k])


def scalars(lam, nb):
    a, b = lam[nb], lam[~nb]
    auc = mannwhitneyu(a, b, alternative='two-sided').statistic / (len(a) * len(b))
    tau, J, tpr, fpr = youden(lam, nb)
    return {'youden_threshold': tau, 'youden_J': J, 'TPR_at_youden': tpr, 'FPR_at_youden': fpr,
            'J_at_1': float((a > 1).mean() - (b > 1).mean()), 'TPR_at_1': float((a > 1).mean()), 'FPR_at_1': float((b > 1).mean()),
            'median_NET_better': float(np.median(a)), 'median_ASYM_better': float(np.median(b)),
            'mean_NET_better': float(a.mean()), 'mean_ASYM_better': float(b.mean()),
            'median_difference': float(np.median(a) - np.median(b)), 'mean_difference': float(a.mean() - b.mean()),
            'P_NET_better_time_has_larger_lambda': float(auc)}


def analyse(df, comp_idx, n_comp, rng):
    lam, nb = df.lam.to_numpy(float), df.net_better.to_numpy(bool)
    point = scalars(lam, nb)
    curve = {f'{t:g}': float((lam[nb] > t).mean() - (lam[~nb] > t).mean()) for t in GRID}
    boots = []
    for _ in range(B):
        counts = np.bincount(rng.integers(0, n_comp, n_comp), minlength=n_comp)
        idx = np.repeat(np.arange(len(df)), counts[comp_idx])
        if nb[idx].all() or (~nb[idx]).all():
            continue
        boots.append(scalars(lam[idx], nb[idx]))
    Bt = pd.DataFrame(boots)
    return {'NET_better': summary(lam[nb]), 'ASYM_better': summary(lam[~nb]), 'point': point, 'separation_at_fixed_thresholds': curve,
            'bootstrap_95': {k: [float(Bt[k].quantile(.025)), float(Bt[k].quantile(.975))] for k in Bt.columns},
            'bootstrap_draws_used': int(len(Bt))}


def main():
    lock = json.loads((W / 'locks/FOLDS.json').read_text())
    comp_of = {e: i for i, c in enumerate(lock['components']) for e in c['events']}
    n_comp = len(lock['components'])
    rng = np.random.default_rng(RNG_SEED)
    M = pd.read_csv(W / 'results/THRESHOLD_ORIGIN_LEVEL.csv')
    M['net_better'] = M.NET < M.ASYM
    M['relative_gap'] = (M.NET - M.ASYM).abs() / M.NET
    E = pd.read_csv(W / 'results/EVENT_FEATURES.csv')
    E = E.assign(lam=E.lambda_hat_train_events, net_better=E.realized_winner == 'NET')
    sets = {'origins_all': M, 'origins_positive_signal': M[M.lam > 0], 'origins_gap_at_least_5pct': M[M.relative_gap >= CLEAR],
            'events': E}
    res = {'status': 'POST_HOC_DESCRIPTIVE', 'lambda': 'n_train S_hat / nu_hat^2 (paper n = independent training events); 0 when S_hat <= 0',
           'paper_prediction': 'NET better above Lambda = 1, ASYM better below', 'bootstrap_unit': 'overlap components', 'draws': B}
    for name, df in sets.items():
        df = df.reset_index(drop=True)
        res[name] = analyse(df, df.event.map(comp_of).to_numpy(), n_comp, rng)
    (W / 'results/LAMBDA_GROUP_COMPARISON.json').write_text(json.dumps(res, indent=1) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    floor = 1e-3
    for grp, color, lab in ((True, '#1f77b4', 'NET better'), (False, '#d62728', 'ASYM better')):
        raw = M.lam[M.net_better == grp].to_numpy()
        v = np.sort(np.maximum(raw, floor))
        ax[0].step(v, np.arange(1, len(v) + 1) / len(v), where='post', color=color,
                   label=f'{lab}: {len(v)} origins, median {np.median(raw):.2f}, mean {raw.mean():.2f}')
        ax[0].axvline(max(np.median(raw), floor), color=color, ls='--', lw=1)
    ax[0].axvline(1, color='#2ca02c', lw=1.5, label='paper threshold Lambda = 1')
    t0 = res['origins_all']['point']['youden_threshold']
    ax[0].axvline(t0, color='0.35', ls=':', lw=1.5, label=f'best separating cut {t0:.2f} (largest vertical gap between the curves)')
    ax[0].set_xscale('log'); ax[0].set_ylim(0, 1)
    ax[0].set_xlabel('Lambda_hat at the origin (no positive signal drawn at 1e-3)'); ax[0].set_ylabel('cumulative share of origins')
    ax[0].set_title('Origins: distribution of Lambda_hat by realized better model (dashed = group medians)', fontsize=8)
    ax[0].legend(fontsize=7)
    jr = np.random.default_rng(1)
    for i, (grp, color, lab) in enumerate(((True, '#1f77b4', 'NET better'), (False, '#d62728', 'ASYM better'))):
        v = E.lam[E.net_better == grp].to_numpy()
        ax[1].scatter(i + jr.uniform(-.12, .12, len(v)), v, color=color, s=30)
        ax[1].hlines(np.median(v), i - .3, i + .3, color='k', ls='--', lw=1.2)
        ax[1].hlines(v.mean(), i - .3, i + .3, color='k', lw=1.2)
    ax[1].axhline(1, color='#2ca02c', lw=1.5)
    t1 = res['events']['point']['youden_threshold']
    ax[1].axhline(t1, color='0.35', ls=':', lw=1.5)
    ax[1].set_yscale('log'); ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels([f'NET better ({int(E.net_better.sum())} events)', f'ASYM better ({int((~E.net_better).sum())} events)'])
    ax[1].set_ylabel('event Lambda_hat'); ax[1].set_title(f'Events (solid = mean, dashed = median, green = Lambda = 1, dotted = best separating cut {t1:.2f})', fontsize=8)
    fig.tight_layout(); fig.savefig(W / 'figures/lambda_group_comparison.png', dpi=150); plt.close(fig)

    rows = []
    for name in sets:
        r = res[name]; p = r['point']; c = r['bootstrap_95']
        rows.append({'set': name, 'n_NET': r['NET_better']['units'], 'n_ASYM': r['ASYM_better']['units'],
                     'median_NET': p['median_NET_better'], 'median_ASYM': p['median_ASYM_better'],
                     'mean_NET': p['mean_NET_better'], 'mean_ASYM': p['mean_ASYM_better'],
                     'NET_share>1': r['NET_better']['share_above_1'], 'ASYM_share>1': r['ASYM_better']['share_above_1'],
                     'median_diff_CI': f"[{c['median_difference'][0]:.2f}, {c['median_difference'][1]:.2f}]",
                     'mean_diff_CI': f"[{c['mean_difference'][0]:.2f}, {c['mean_difference'][1]:.2f}]",
                     'P(larger)': p['P_NET_better_time_has_larger_lambda'],
                     'P_CI': f"[{c['P_NET_better_time_has_larger_lambda'][0]:.2f}, {c['P_NET_better_time_has_larger_lambda'][1]:.2f}]",
                     'medNET_CI': f"[{c['median_NET_better'][0]:.2f}, {c['median_NET_better'][1]:.2f}]",
                     'medASYM_CI': f"[{c['median_ASYM_better'][0]:.2f}, {c['median_ASYM_better'][1]:.2f}]"})
    pd.set_option('display.width', 260)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    yrows = []
    for name in sets:
        p = res[name]['point']; c = res[name]['bootstrap_95']
        yrows.append({'set': name, 'youden_tau': p['youden_threshold'], 'tau_CI': f"[{c['youden_threshold'][0]:.2f}, {c['youden_threshold'][1]:.2f}]",
                      'J': p['youden_J'], 'TPR': p['TPR_at_youden'], 'FPR': p['FPR_at_youden'],
                      'J_at_1': p['J_at_1'], 'J1_CI': f"[{c['J_at_1'][0]:.2f}, {c['J_at_1'][1]:.2f}]", 'TPR_at_1': p['TPR_at_1'], 'FPR_at_1': p['FPR_at_1'],
                      'J_CI': f"[{c['youden_J'][0]:.2f}, {c['youden_J'][1]:.2f}]"})
    print(pd.DataFrame(yrows).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print('separation (share above tau: NET-better minus ASYM-better)')
    print(pd.DataFrame({n: res[n]['separation_at_fixed_thresholds'] for n in sets}).T.to_string(float_format=lambda x: f'{x:.2f}'))


if __name__ == '__main__':
    main()
