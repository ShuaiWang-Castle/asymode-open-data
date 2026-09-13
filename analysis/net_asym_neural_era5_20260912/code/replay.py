#!/usr/bin/env python3
"""Checkpoint-only replay for all three tasks. Inference only.

Builds no optimizer, calls no training function, performs no selection. Each checkpoint is
loaded with its frozen statistics, predictions are recomputed and compared with the saved
prediction shards, and the headline losses are recomputed from saved truth.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import build_model   # noqa: E402


def replay_us(G, A, limit):
    import us_train as UT
    data = UT.USData(G, A / 'locks'); de = data.df['evaluation']
    te, we = data.truth64('evaluation'), data.weights['evaluation']; rows = []
    for f in sorted((A / 'checkpoints/us_era5').glob('*.pt'))[:limit]:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        if not (ck['scale'] == data.scale and np.array_equal(np.array(ck['wx_mean']), data.wx_mean)
                and np.array_equal(np.array(ck['wx_std']), data.wx_std)):
            raise SystemExit(f'{f.name}: frozen statistics differ from the rebuilt data')
        m = build_model(ck['kind'], ck['ctx_dim'], ck['step_dim'], ck['horizon'], ck['config']['parameter_target'],
                        ck['scale'], ck['source_mean'])
        m.load_state_dict(ck['state_dict'])
        pe = UT.predict(m, data, 'evaluation'); worst = 0.0
        for e, g in de.groupby('event'):
            z = np.load(A / f"predictions/us_era5/{ck['kind']}/seed{ck['seed']}/{e}.npz")
            if not (np.array_equal(z['row'], g.row.to_numpy()) and np.array_equal(z['t'], g.t.to_numpy())):
                raise SystemExit(f'{f.name} {e}: window keys differ')
            worst = max(worst, float(np.abs(pe[g.index.to_numpy()].astype(np.float32) - z['pred']).max()))
        rows.append({'task': 'us_era5', 'checkpoint': f.name, 'max_abs_diff_vs_shard': worst,
                     **{f'eval_{k}': v for k, v in UT.metrics(pe, te, we).items()}})
        print(f"  us {f.name:22s} max|replay-shard|={worst:.2e} path={rows[-1]['eval_path_mse']:.6e}", flush=True)
    return rows


def replay_synthetic(A, limit):
    import synth_train as ST
    ctx, y0, step = ST.scene_inputs(); rows = []
    for f in sorted((A / 'checkpoints/synthetic').rglob('*.pt'))[:limit]:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        m = build_model(ck['kind'], ck['ctx_dim'], ck['step_dim'], ck['horizon'], ck['config']['parameter_target'],
                        ck['scale'], ck['source_mean'])
        m.load_state_dict(ck['state_dict'])
        with torch.no_grad():
            p3 = m(ctx, y0, step).double().numpy()
        tag = f.stem
        z = np.load(A / f"predictions/synthetic/{ck['kind']}/{tag}.npz")
        t = np.load(A / f"predictions/synthetic/truth/law{ck['law']}_g{ck['gamma']:g}_seed{ck['seed']}.npz")
        test = float(((p3[:, None, :] - t['test']) ** 2).mean(axis=(1, 2)).mean())
        rows.append({'task': 'synthetic', 'checkpoint': f'{ck["kind"]}/{f.name}',
                     'max_abs_diff_vs_shard': float(np.abs(p3.astype(np.float32) - z['pred']).max()),
                     'test_path_mse_recomputed': test, 'vs_exact_mean_path_mse': float(((p3 - t['mu0']) ** 2).mean())})
    print(f"  synthetic: {len(rows)} checkpoints, worst max|replay-shard| = "
          f"{max([r['max_abs_diff_vs_shard'] for r in rows], default=float('nan')):.2e}", flush=True)
    return rows


def replay_aneel(G, A, scratch, limit):
    import aneel_driver_repaired as RD
    R1 = G / 'analysis/asymode_trajectory_r1_20260909'; PKG = R1 / '_package/ASYMODE_CC_STANDALONE_20260909'
    man = json.loads((R1 / 'data_resolution/DATA_MANIFEST.json').read_text())
    man['ledger_path'] = str(R1 / 'data_resolution/rebuilt_selected_ledgers.npz')
    rt = Path(scratch) / 'aneel_runtime_manifest_replay.json'; rt.write_text(json.dumps(man))
    ctx = RD.Ctx(rt, PKG / 'code', R1 / 'preflight')
    st = ctx.fit_stats('full_2018'); ev = ctx.eval_tensors(2019, 'reused_2019', st['state_scale']); rows = []
    for f in sorted((A / 'checkpoints/aneel').glob('*.pt'))[:limit]:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        m = ctx.CM.TrajectoryModel(ck['kind'], 168, 24, 5, ck['parameter_target'], ck['state_scale'], ck['source_mean'])
        m.load_state_dict(ck['state_dict'])
        p = ctx.predict(m, ev)
        z = np.load(A / f"predictions/aneel/{ck['kind']}/seed{ck['seed']}.npz")['pred']
        _, _, r = ctx.score(p, ev)
        rows.append({'task': 'aneel', 'checkpoint': f.name, 'max_abs_diff_vs_shard': float(np.abs(p.astype(np.float32) - z).max()),
                     'company_equal_path_mse': r['mse_path']})
        print(f"  aneel {f.name:16s} max|replay-shard|={rows[-1]['max_abs_diff_vs_shard']:.2e} path={r['mse_path']:.6e}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['us', 'synthetic', 'aneel', 'all'], default='all')
    ap.add_argument('--root', required=True); ap.add_argument('--work', required=True)
    ap.add_argument('--scratch', default=None); ap.add_argument('--max-models', type=int, default=None)
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    G, A = Path(a.root).resolve(), Path(a.work).resolve()
    scratch = Path(a.scratch) if a.scratch else A / 'logs'
    rows = []
    if a.task in ('us', 'all'):
        rows += replay_us(G, A, a.max_models)
    if a.task in ('synthetic', 'all'):
        rows += replay_synthetic(A, a.max_models)
    if a.task in ('aneel', 'all'):
        rows += replay_aneel(G, A, scratch, a.max_models)
    worst = max([r['max_abs_diff_vs_shard'] for r in rows], default=float('nan'))
    res = {'replayed_checkpoints': len(rows), 'worst_max_abs_diff_vs_shard': worst, 'exact': worst == 0.0,
           'note': 'inference only; no optimizer, no training call, no selection', 'rows': rows}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"replayed {len(rows)} checkpoints; worst difference {worst:.2e}")


if __name__ == '__main__':
    main()
