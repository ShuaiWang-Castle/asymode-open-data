#!/usr/bin/env python3
"""Checkpoint-only replay for the affected-county CV round.

Inference only: builds no optimizer, calls no training function and selects nothing. Each final checkpoint is
loaded with its frozen fold statistics, test predictions are recomputed and compared with the saved shards.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
W = HERE.parent
sys.path.insert(0, str(HERE))
import cv_data as CD                  # noqa: E402
import cv_train as CT                 # noqa: E402
from models_v2 import build_model     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(W.parents[1])); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--max-models', type=int, default=None); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data = CD.CVData(Path(a.root), W / 'locks'); rows = []
    for f in sorted((W / 'checkpoints').glob('fold*/*.pt'))[:a.max_models]:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        if data.fold != ck['fold']:
            data.set_fold(ck['fold'])
        if not (ck['scale'] == data.scale and np.array_equal(np.array(ck['wx_mean']), data.wx_mean)
                and np.array_equal(np.array(ck['wx_std']), data.wx_std)):
            raise SystemExit(f'{f.parent.name}/{f.name}: frozen statistics differ from the rebuilt fold data')
        m = build_model(ck['kind'], ck['structure'], ck['ctx_dim'], ck['step_dim'], ck['horizon'], ck['parameter_target'],
                        ck['scale'], ck['source_mean'])
        m.load_state_dict(ck['state_dict'])
        pe = CT.predict(m, data, 'test'); worst = 0.0
        for e, g in data.df['test'].groupby('event'):
            z = np.load(W / f"predictions/fold{ck['fold']}/{ck['kind']}/seed{ck['seed']}/{e}.npz")
            if not (np.array_equal(z['row'], g.row.to_numpy()) and np.array_equal(z['t'], g.t.to_numpy())):
                raise SystemExit(f'{f.name} {e}: window keys differ')
            worst = max(worst, float(np.abs(pe[g.index.to_numpy()].astype(np.float32) - z['pred']).max()))
        rows.append({'checkpoint': f'{f.parent.name}/{f.name}', 'max_abs_diff_vs_shard': worst})
        print(f'  {f.parent.name}/{f.name:18s} max|replay-shard|={worst:.2e}', flush=True)
    worst = max([r['max_abs_diff_vs_shard'] for r in rows], default=float('nan'))
    res = {'replayed_checkpoints': len(rows), 'worst_max_abs_diff_vs_shard': worst, 'exact': worst == 0.0,
           'note': 'inference only; no optimizer, no training call, no selection', 'rows': rows}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print(f'replayed {len(rows)} checkpoints; worst difference {worst:.2e}')


if __name__ == '__main__':
    main()
