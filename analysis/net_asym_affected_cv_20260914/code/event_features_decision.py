#!/usr/bin/env python3
"""Leave-one-component-out calibration of event-level feature decisions (post hoc, descriptive).

Reads results/EVENT_FEATURES.csv from event_features.py. For each overlap component, a threshold is chosen on the
other events so that applying the rule minimises their event-equal affected MSE; the rule is then applied to the
held-out events.

- Paper feature: the benchmark signal-to-noise per replicate S_hat / nu_hat^2, with the paper's orientation (larger
  favours NET). Its threshold is 1/n_eff, so calibrating it calibrates the unknown effective number of independent
  events.
- Readable features: sigma_x and the mean outage level at the origins, calibrated the same way, with their
  orientation also chosen on the other events.

Also reported: uncalibrated plug-in rules and partial rank associations controlling for outage level. There are 26
events; every number is descriptive.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
W = HERE.parent


def selected_mse(E, choose_net):
    return np.where(choose_net, E.NET_path_mse.to_numpy(), E.ASYM_path_mse.to_numpy())


def rule(values, orientation, tau):
    return values > tau if orientation > 0 else values < tau


def calibrate(train, col, orientations):
    v = np.sort(train[col].unique())
    cands = np.r_[-np.inf, (v[:-1] + v[1:]) / 2, np.inf]
    best = None
    for o in orientations:
        for tau in cands:
            m = selected_mse(train, rule(train[col].to_numpy(), o, tau)).mean()
            key = (round(float(m), 15), 0 if o == orientations[0] else 1)
            if best is None or key < best[0]:
                best = (key, o, tau)
    return best[1], best[2]


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    B = np.c_[np.ones_like(rz), rz]
    ex = rx - B @ np.linalg.lstsq(B, rx, rcond=None)[0]
    ey = ry - B @ np.linalg.lstsq(B, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def level_figure(E, T):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = np.where(E.realized_winner == 'ASYM', '#d62728', '#1f77b4')
    thr = T[T.feature == 'mean_origin_state'].threshold
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = 100 * E.mean_origin_state
    ax.scatter(x, E.delta_cv, c=col, s=40)
    for xx, yy, name in zip(x, E.delta_cv, E.event):
        ax.annotate(name[2:], (xx, yy), fontsize=6, xytext=(3, 2), textcoords='offset points')
    ax.axvspan(100 * thr.min(), 100 * thr.max(), color='0.85', zorder=0)
    ax.axvline(100 * thr.median(), color='k', lw=1, ls='--')
    ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1e-5); ax.axhline(0, color='k', lw=.8)
    ax.set_xlabel('mean outage level at the forecast origins, affected counties (%)')
    ax.set_ylabel('CV test Delta = MSE_NET - MSE_ASYM (affected)')
    ax.set_title('red = ASYM better, blue = NET better; dashed = median leave-one-component-out threshold, band = its range', fontsize=8)
    fig.tight_layout(); fig.savefig(W / 'figures/event_features_level.png', dpi=150); plt.close(fig)


def main():
    E = pd.read_csv(W / 'results/EVENT_FEATURES.csv')
    lock = json.loads((W / 'locks/FOLDS.json').read_text())
    comp_of = {e: i for i, c in enumerate(lock['components']) for e in c['events']}
    E['component'] = E.event.map(comp_of)
    specs = [('signal_to_noise_per_replicate', (+1,), 'paper: S_hat/nu_hat^2, NET if above 1/n_eff'),
             ('sigma_x_pp', (+1, -1), 'readable: conditional path dispersion sigma_x'),
             ('mean_origin_state', (+1, -1), 'readable: mean outage level at the origins (an input)'),
             ('n_counties_mean', (+1, -1), 'readable: replicates per origin n')]
    out_rows, thr_rows = [], []
    choices = E[['event', 'component', 'type', 'realized_winner', 'NET_path_mse', 'ASYM_path_mse', 'delta_cv']].copy()
    for col, ori, label in specs:
        choose = np.zeros(len(E), bool)
        for comp, g in E.groupby('component'):
            o, tau = calibrate(E[E.component != comp], col, ori)
            pos = g.index.to_numpy()
            choose[pos] = rule(E.loc[pos, col].to_numpy(), o, tau)
            thr_rows.append({'feature': col, 'held_out_component': int(comp), 'orientation_net_if': 'above' if o > 0 else 'below',
                             'threshold': float(tau)})
        choices[f'loco_{col}'] = np.where(choose, 'NET', 'ASYM')
        out_rows.append({'feature': col, 'description': label, 'loco_event_equal_mse': float(selected_mse(E, choose).mean()),
                         'share_matching_realized_winner': float((choices[f'loco_{col}'] == E.realized_winner).mean()),
                         'share_choosing_NET': float(choose.mean())})
    T = pd.DataFrame(thr_rows)
    snr_thr = T[(T.feature == 'signal_to_noise_per_replicate') & np.isfinite(T.threshold) & (T.threshold > 0)].threshold
    base = {'always_NET': float(E.NET_path_mse.mean()), 'always_ASYM': float(E.ASYM_path_mse.mean()),
            'better_of_two_per_event': float(np.minimum(E.NET_path_mse, E.ASYM_path_mse).mean())}
    plug = {}
    for tag in ('counties', 'train_events'):
        ch = E[f'delta_hat_{tag}'].to_numpy() <= 0
        plug[f'paper_rule_Delta_hat_n_{tag}'] = {'event_equal_mse': float(selected_mse(E, ch).mean()),
                                                 'share_matching_realized_winner': float((np.where(ch, 'NET', 'ASYM') == E.realized_winner).mean()),
                                                 'share_choosing_NET': float(ch.mean())}
    partial = {f'{col}_given_mean_origin_state': partial_spearman(E[col], E.delta_cv, E.mean_origin_state)
               for col in ('signal_to_noise_per_replicate', 'delta_hat_counties', 'sigma_x_pp', 'S_hat', 'nu2_hat')}
    partial['mean_origin_state_given_signal_to_noise'] = partial_spearman(E.mean_origin_state, E.delta_cv, E.signal_to_noise_per_replicate)
    partial['sigma_x_given_signal_to_noise'] = partial_spearman(E.sigma_x_pp, E.delta_cv, E.signal_to_noise_per_replicate)
    res = {'status': 'POST_HOC_DESCRIPTIVE', 'events': int(len(E)), 'components': int(E.component.nunique()),
           'baselines_event_equal_affected_mse': base, 'uncalibrated_paper_rules': plug,
           'loco_calibrated_rules': out_rows,
           'implied_n_eff_from_loco_snr_threshold': {'median': float(np.median(1 / snr_thr)) if len(snr_thr) else None,
                                                     'min': float((1 / snr_thr).min()) if len(snr_thr) else None,
                                                     'max': float((1 / snr_thr).max()) if len(snr_thr) else None},
           'partial_spearman_with_delta_cv': partial}
    (W / 'results/EVENT_FEATURES_DECISION.json').write_text(json.dumps(res, indent=1) + '\n')
    T.to_csv(W / 'results/EVENT_FEATURES_LOCO_THRESHOLDS.csv', index=False)
    choices.to_csv(W / 'results/EVENT_FEATURES_LOCO_CHOICES.csv', index=False)
    level_figure(E, T)
    print(json.dumps(res, indent=1))
    print(T.groupby(['feature', 'orientation_net_if']).threshold.describe().to_string())


if __name__ == '__main__':
    main()
