#!/usr/bin/env python3
"""Descriptive detail for Task B from the saved paired tables only (no model, no selection).

Delta = MSE_NET - MSE_ASYM; positive favours ASYM. Reused 2019 retrospective evaluation.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); a = ap.parse_args()
R = Path(a.work) / 'results/aneel'
pc = pd.read_csv(R / 'ANEEL_COMPANY_PAIRED_DELTA.csv'); pg = pd.read_csv(R / 'ANEEL_COLLECTION_PAIRED_DELTA.csv')
out = {'delta_convention': 'MSE_NET - MSE_ASYM; positive favours ASYM', 'status': 'descriptive, reused 2019 retrospective', 'variants': {}}
comp_rows = []
for v in ('raw', 'output_clip_01'):
    q = pc[pc.variant == v]
    seeds = sorted(q.seed.unique())
    per_seed = q.groupby('seed').delta_mse_path.mean()            # company-equal Delta per seed
    loso = {int(s): float(per_seed.drop(s).mean()) for s in seeds}
    cm = q.groupby('company').agg(NET=('NET_mse_path', 'mean'), ASYM=('ASYM_mse_path', 'mean'), delta=('delta_mse_path', 'mean'),
                                  seeds_pos=('delta_mse_path', lambda s: int((s > 0).sum())))
    cm['rel_gain_asym_vs_net'] = (cm.NET - cm.ASYM) / cm.NET
    total = cm.delta.sum()
    order = cm.delta.abs().sort_values(ascending=False)
    top3 = list(order.index[:3])
    g = pg[pg.variant == v].groupby(['group', 'company']).delta_mse_path.mean()
    out['variants'][v] = {
        'company_equal_delta_by_seed': {int(s): float(x) for s, x in per_seed.items()},
        'leave_one_seed_out_delta': loso,
        'leave_one_seed_out_all_negative': all(x < 0 for x in loso.values()),
        'companies_delta_positive': int((cm.delta > 0).sum()), 'companies': int(len(cm)),
        'companies_positive_in_all_5_seeds': int((cm.seeds_pos == 5).sum()), 'companies_negative_in_all_5_seeds': int((cm.seeds_pos == 0).sum()),
        'top3_companies_by_abs_delta': [int(c) for c in top3],
        'share_of_summed_company_delta_from_top3': float(cm.delta[top3].sum() / total) if total != 0 else None,
        'spearman_company_NET_mse_vs_delta': float(cm.NET.rank().corr(cm.delta.rank())),
        'spearman_company_NET_mse_vs_relative_gain': float(cm.NET.rank().corr(cm.rel_gain_asym_vs_net.rank())),
        'collections_delta_positive': int((g > 0).sum()), 'collections': int(len(g)),
        'collection_delta_min': float(g.min()), 'collection_delta_max': float(g.max())}
    for c, r in cm.iterrows():
        comp_rows.append({'variant': v, 'company': int(c), 'NET_mse_path': r.NET, 'ASYM_mse_path': r.ASYM,
                          'delta_seed_mean': r.delta, 'seeds_delta_positive': int(r.seeds_pos), 'rel_gain_asym_vs_net': r.rel_gain_asym_vs_net})
pd.DataFrame(comp_rows).sort_values(['variant', 'delta_seed_mean']).to_csv(R / 'ANEEL_COMPANY_DETAIL.csv', index=False)
(R / 'ANEEL_DESCRIPTIVE_DETAIL.json').write_text(json.dumps(out, indent=1))
for v, d in out['variants'].items():
    print(f"[{v}] per-seed Delta {[f'{x:+.2e}' for x in d['company_equal_delta_by_seed'].values()]}")
    print(f"   leave-one-seed-out {[f'{x:+.2e}' for x in d['leave_one_seed_out_delta'].values()]}  all negative: {d['leave_one_seed_out_all_negative']}")
    print(f"   companies + {d['companies_delta_positive']}/{d['companies']} (all 5 seeds +: {d['companies_positive_in_all_5_seeds']}, all 5 seeds -: {d['companies_negative_in_all_5_seeds']})"
          f"  collections + {d['collections_delta_positive']}/{d['collections']}  range [{d['collection_delta_min']:+.2e}, {d['collection_delta_max']:+.2e}]")
    print(f"   top-3 |Delta| companies {d['top3_companies_by_abs_delta']} carry {d['share_of_summed_company_delta_from_top3']:.2f} of summed Delta;"
          f" Spearman(NET mse, Delta)={d['spearman_company_NET_mse_vs_delta']:+.2f}, Spearman(NET mse, rel gain)={d['spearman_company_NET_mse_vs_relative_gain']:+.2f}")
