#!/usr/bin/env python3
"""Rebuild the per-fit files that the analysis scripts read, from the packed public results.

python3.11 source/unpack_fits.py b4   ->  results/b4_fits/<key>.npz and <key>.json
Each npz holds best_prediction, best_test_prediction and snapshots (only the final checkpoint, shape (1, 2, 8, 24)),
which is all that b4_analyze.py and b4s_analyze.py use. The json records are the original per-fit records.
"""
from __future__ import annotations
import gzip, json, sys
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parent.parent


def main(batch):
    with np.load(W / f'results/{batch}_predictions.npz') as a:
        arr = {k: a[k] for k in ('best_train', 'best_test', 'final_train', 'final_test')}
        axes = json.loads(str(a['axes']))
    recs = {}
    with gzip.open(W / f'results/{batch}_fit_records.jsonl.gz', 'rt', encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line); recs[r['key']] = r
    out = W / f'results/{batch}_fits'; out.mkdir(parents=True, exist_ok=True)
    n = 0
    if batch == 'b4':
        for idx in np.ndindex(*arr['best_train'].shape[:5]):
            k, gi, ri, ni, rep = idx
            key = f"g{gi}_r{ri}_n{axes['n'][ni]}_rep{rep:02}_{axes['model'][k]}"
            if key not in recs:
                continue
            snaps = np.stack([arr['final_train'][idx], arr['final_test'][idx]])[None]
            np.savez_compressed(out / f'{key}.npz', best_prediction=arr['best_train'][idx], best_test_prediction=arr['best_test'][idx], snapshots=snaps)
            (out / f'{key}.json').write_text(json.dumps(recs[key]) + '\n'); n += 1
    else:
        for idx in np.ndindex(*arr['best_train'].shape[:6]):
            c, k, fi, ri, ni, rep = idx
            key = f"{axes['combo'][c]}_f{axes['feedback_index'][fi]}_r{ri}_n{axes['n'][ni]}_rep{rep:02}_{axes['model'][k]}"
            if key not in recs:
                continue
            snaps = np.stack([arr['final_train'][idx], arr['final_test'][idx]])[None]
            np.savez_compressed(out / f'{key}.npz', best_prediction=arr['best_train'][idx], best_test_prediction=arr['best_test'][idx], snapshots=snaps)
            (out / f'{key}.json').write_text(json.dumps(recs[key]) + '\n'); n += 1
    print(json.dumps({'batch': batch, 'fits_written': n, 'out': str(out.relative_to(W))}))


if __name__ == '__main__':
    main(sys.argv[1])
