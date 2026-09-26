#!/usr/bin/env python3
"""Pack per-fit outputs into compact public files.

For a batch (b4: NET/ASYM/ASYM_STATE on the B4' law; b5: generality run) write
  results/<batch>_predictions.npz  float32 predictions indexed by model, cell and replication:
      best_train, best_test (checkpoint chosen by the original early-stop policy), final_train, final_test (update 3000)
  results/<batch>_fit_records.jsonl.gz  one JSON record per fit (configuration, stopping, validation curve, timing)
and the SHA-256 of every generated data file, so the deterministic generators can be checked.
"""
from __future__ import annotations
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pack(batch):
    if batch == 'b4':
        import b4_data as A
        fits = W / 'results/b4_fits'
        kinds = ['NET', 'ASYM', 'ASYM_STATE']
        axes = {'model': kinds, 'gamma': A.GAMMAS.tolist(), 'rho': A.RHOS.tolist(), 'n': A.NS.tolist(), 'replication': list(range(A.REPS))}
        key = lambda k, i, r, j, rep: f'g{i}_r{r}_n{A.NS[j]}_rep{rep:02}_{k}'
        shape = (len(kinds), len(A.GAMMAS), len(A.RHOS), len(A.NS), A.REPS)
        data_dirs = [A.OUT]
    else:
        import b5_experiment as E
        fits = E.FITS
        combos = [f'{law}{t}' for law, t in E.COMBOS]
        axes = {'combo': combos, 'model': E.KINDS, 'feedback_index': E.FEEDBACK_IDX, 'rho': [0.0, 1.0], 'n': E.NS, 'replication': list(range(E.REPS))}
        key = None
        shape = (len(combos), len(E.KINDS), len(E.FEEDBACK_IDX), 2, len(E.NS), E.REPS)
        import b5_data as BB, b4_data as A
        data_dirs = [BB.OUT_B]
    arr = {name: np.full(shape + (8, 24), np.nan, dtype=np.float32) for name in ('best_train', 'best_test', 'final_train', 'final_test')}
    records = []; missing = 0
    for idx in np.ndindex(*shape):
        if batch == 'b4':
            k = axes['model'][idx[0]]; stem = key(k, idx[1], idx[2], idx[3], idx[4])
        else:
            combo = axes['combo'][idx[0]]; k = axes['model'][idx[1]]
            stem = f"{combo}_f{axes['feedback_index'][idx[2]]}_r{idx[3]}_n{axes['n'][idx[4]]}_rep{idx[5]:02}_{k}"
        f = fits / f'{stem}.npz'; j = fits / f'{stem}.json'
        if not (f.exists() and j.exists()):
            missing += 1; continue
        with np.load(f) as a:
            arr['best_train'][idx] = a['best_prediction']; arr['best_test'][idx] = a['best_test_prediction']
            arr['final_train'][idx] = a['snapshots'][-1, 0]; arr['final_test'][idx] = a['snapshots'][-1, 1]
        records.append(json.loads(j.read_text()))
    out_npz = W / f'results/{batch}_predictions.npz'
    np.savez_compressed(out_npz, **arr, axes=np.array(json.dumps(axes)))
    with gzip.open(W / f'results/{batch}_fit_records.jsonl.gz', 'wt', encoding='utf-8') as fh:
        for r in records:
            fh.write(json.dumps(r) + '\n')
    checks = {}
    for d in data_dirs:
        for p in sorted(Path(d).glob('*')):
            if p.is_file():
                checks[f'{Path(d).name}/{p.name}'] = sha(p)
    (W / f'results/{batch}_data_checksums.json').write_text(json.dumps(checks, indent=1) + '\n')
    print(json.dumps({'batch': batch, 'fits_packed': len(records), 'missing': missing, 'npz_MB': round(out_npz.stat().st_size / 2 ** 20, 1),
                      'data_files_checksummed': len(checks)}))


if __name__ == '__main__':
    pack(sys.argv[1])
