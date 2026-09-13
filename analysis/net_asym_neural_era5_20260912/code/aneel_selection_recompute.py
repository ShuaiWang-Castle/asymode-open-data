#!/usr/bin/env python3
"""Recompute NET/ASYM final selection from the existing 2018 tables only.

Candidate identity is (model, config_id). Every eligible candidate must carry
exactly two folds x three development seeds. No model is trained and no 2019
column is read.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
for x in ('stage1', 'stage2', 'origins', 'out'): ap.add_argument(f'--{x}', required=True)
a = ap.parse_args()
d1, d2 = pd.read_csv(a.stage1), pd.read_csv(a.stage2)
z = np.load(a.origins); n = {k: len(z[k]) for k in z.files}
FOLDS, DEV = ('fold_A_fit', 'fold_B_fit'), (4101, 4102, 4103)
out, rows = {}, []
for kind in ('NET', 'ASYM'):
    top = list(d1[d1.model == kind].groupby('config_id').best_val_mse_path.mean().sort_values().index[:2])
    d = pd.concat([d1[(d1.model == kind) & d1.config_id.isin(top)],
                   d2[(d2.model == kind) & d2.config_id.isin(top)]], ignore_index=True)
    for cid in top:
        sub = d[d.config_id == cid]
        assert len(sub) == 6, (kind, cid, len(sub))
        assert set(zip(sub.fit, sub.seed)) == {(f, s) for f in FOLDS for s in DEV}, (kind, cid)
        for r in sub.itertuples():
            rows.append({'model': kind, 'config_id': cid, 'fit': r.fit, 'seed': r.seed,
                         'best_val_mse_path': r.best_val_mse_path, 'best_update': r.best_update})
    agg = d.groupby('config_id').best_val_mse_path.mean().sort_values()
    sel = agg.index[0]; ch = d[d.config_id == sel]
    scaled = [r.best_update * n['full_2018'] / n[r.fit] for r in ch.itertuples()]
    steps = int(min(6000, max(250, round(float(np.median(scaled)) / 250) * 250)))
    cfg = {'p32768_lr0.0003': (32768, 3e-4), 'p8192_lr0.001': (8192, 1e-3),
           'p8192_lr0.0003': (8192, 3e-4), 'p32768_lr0.001': (32768, 1e-3),
           'p8192_lr0.003': (8192, 3e-3), 'p32768_lr0.003': (32768, 3e-3)}[sel]
    out[kind] = {'top2': top, 'selected_config': sel, 'parameter_target': cfg[0], 'lr': cfg[1],
                 'final_updates': steps, 'candidate_means_six_records': {k: float(v) for k, v in agg.items()},
                 'scaled_updates': sorted(scaled), 'median_scaled': float(np.median(scaled))}
expected = {'NET': (32768, 3e-4, 3500), 'ASYM': (8192, 1e-3, 1000)}
checks = {k: (out[k]['parameter_target'], out[k]['lr'], out[k]['final_updates']) == expected[k] for k in out}
res = {'task': 'ANEEL targeted selection repair (table-only)', 'candidate_key': '(model, config_id)',
       'support_rule': '2 folds x 3 development seeds per eligible candidate', 'selection': out,
       'matches_verified_repair_config': checks, 'status': 'PASS' if all(checks.values()) else 'FAIL',
       'origin_counts': n,
       'sources': {'stage1_sha256': hashlib.sha256(Path(a.stage1).read_bytes()).hexdigest(),
                   'stage2_sha256': hashlib.sha256(Path(a.stage2).read_bytes()).hexdigest(),
                   'origins_sha256': hashlib.sha256(Path(a.origins).read_bytes()).hexdigest()}}
Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out) / 'ANEEL_SELECTION_LOCK.json').write_text(json.dumps(res, indent=1))
pd.DataFrame(rows).to_csv(Path(a.out) / 'ANEEL_ELIGIBLE_SUPPORT_RECORDS.csv', index=False)
for k, v in out.items():
    print(f"  {k:4s} top2={v['top2']} -> {v['selected_config']} target={v['parameter_target']} "
          f"lr={v['lr']:g} updates={v['final_updates']} (median scaled {v['median_scaled']:.2f})")
print(f"  matches spec 5.3: {checks} -> {res['status']}  origin counts: {n}")
