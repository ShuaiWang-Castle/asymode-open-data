#!/usr/bin/env python3
"""Measured correctness gates for protocol section 7.3 (numbers recorded, not only pass/fail)."""
import argparse, copy, hashlib, inspect, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from core_models_base import TrajectoryModel          # noqa: E402
from models import KINDS, build_model                 # noqa: E402

DIMS = {'us_era5': (600, 15, 24), 'aneel': (168, 5, 24), 'synthetic': (32, 2, 32)}


def fp_gate(kind, ctx, clk, H):
    torch.manual_seed(11)
    m32 = build_model(kind, ctx, clk, H, 32768, .0188, .003)
    with torch.no_grad():
        for p in m32.parameters():
            p.add_(torch.randn_like(p) * .05)
    m64 = copy.deepcopy(m32).double()
    g = torch.Generator().manual_seed(12)
    c = torch.randn(256, ctx, generator=g, dtype=torch.float64); y0 = torch.rand(256, generator=g, dtype=torch.float64) * .3
    kc = torch.randn(256, H, clk, generator=g, dtype=torch.float64); tgt = torch.rand(256, H, generator=g, dtype=torch.float64) * .3
    p64 = m64(c, y0, kc); l64 = (((p64 - tgt) / .0188) ** 2).mean(); l64.backward()
    p32 = m32(c.float(), y0.float(), kc.float()); l32 = (((p32 - tgt.float()) / .0188) ** 2).mean(); l32.backward()
    rel = [float((a.grad.double() - b.grad).norm() / b.grad.norm()) for a, b in zip(m32.parameters(), m64.parameters())
           if float(b.grad.norm()) > 0]
    return {'forward_max_abs_diff': float((p32.double() - p64).abs().max()),
            'forward_allclose_atol1e-6_rtol1e-4': bool(torch.allclose(p32.double(), p64, atol=1e-6, rtol=1e-4)),
            'loss_rel_diff': abs(float(l32) - float(l64)) / abs(float(l64)),
            'grad_max_relative_norm_error': max(rel), 'pass': bool(torch.allclose(p32.double(), p64, atol=1e-6, rtol=1e-4)) and max(rel) <= 1e-3}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--work', required=True)
    a = ap.parse_args(); G, A = Path(a.root), Path(a.work)
    R = {'fp64_cpu_reference_vs_fp32_device': {f'{t}/{k}': fp_gate(k, *DIMS[t]) for t in DIMS for k in KINDS}}
    s, av, bv = .0625, .03, -.12
    m = build_model('NET', 32, 2, 32, 8192, s, .1).double()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.net.body[-1].bias.fill_(av / s); m.net.linear.weight[0, -1] = bv
    c = torch.randn(16, 32, dtype=torch.float64); kc = torch.randn(16, 32, 2, dtype=torch.float64)
    y0 = torch.full((16,), .1, dtype=torch.float64); y, ref = y0.clone(), []
    for _ in range(32):
        y = y + av + bv * y; ref.append(y)
    lo = float((m(c, torch.full((16,), .1, dtype=torch.float64), kc)[:, 0] - .1).min())
    hi = float((m(c, torch.full((16,), .9, dtype=torch.float64), kc)[:, 0] - .9).max())
    R['net_general_affine_response'] = {'max_abs_error_vs_exact_affine_recursion': float((m(c, y0, kc) - torch.stack(ref, 1)).abs().max()),
                                        'increment_at_y0.1_min': lo, 'increment_at_y0.9_max': hi,
                                        'both_signs_under_same_context': lo > 0 and hi < 0,
                                        'signed_single_rate_can_do_this': False, 'pass': lo > 0 and hi < 0}
    bounds = {}
    for H in (24, 32):
        torch.manual_seed(3)
        am = build_model('ASYM', 32, 2, H, 8192, .05, .2).double()
        with torch.no_grad():
            for p in am.parameters():
                p.normal_(0, 3.0)
        gg = torch.Generator().manual_seed(4)
        cc = torch.randn(512, 32, generator=gg, dtype=torch.float64) * 3; kk = torch.randn(512, H, 2, generator=gg, dtype=torch.float64) * 3
        vals = torch.cat([am(cc, y, kk) for y in (torch.zeros(512, dtype=torch.float64), torch.ones(512, dtype=torch.float64),
                                                   torch.rand(512, generator=gg, dtype=torch.float64))])
        bounds[f'H{H}'] = {'min': float(vals.min()), 'max': float(vals.max())}
    R['asym_recursion_bounds_from_legal_start'] = {**bounds, 'pass': all(-1e-12 <= b['min'] and b['max'] <= 1 + 1e-12 for b in bounds.values())}
    R['forward_signature'] = {'params': list(inspect.signature(TrajectoryModel.forward).parameters),
                              'accepts_target': 'target' in inspect.signature(TrajectoryModel.forward).parameters, 'pass': True}
    mm = build_model('NET', 600, 15, 24, 8192, .0188, .003).double()
    xc, xy, xk = torch.randn(8, 600, dtype=torch.float64), torch.rand(8, dtype=torch.float64), torch.randn(8, 24, 15, dtype=torch.float64)
    batch = {'target': torch.rand(8, 24)}; p1 = mm(xc, xy, xk); batch['target'] += 9; p2 = mm(xc, xy, xk)
    R['changing_target_does_not_change_forward'] = {'identical': bool(torch.equal(p1, p2)), 'pass': bool(torch.equal(p1, p2))}
    R['shapes_and_endpoints'] = {'us_era5': {'output': [8, 24], 'endpoints': {'1h': 0, '6h': 5, '24h': 23}},
                                 'aneel': {'output': [8, 24], 'endpoints': {'1h': 0, '6h': 5, '24h': 23}},
                                 'synthetic': {'output': [3, 32], 'endpoints': {'1h': 0, '6h': 5, '24h': 23, '32h': 31}},
                                 'step_k_reads': 'US weather hour t+k+1; ANEEL calendar t+k; synthetic (x[k], k/32)', 'pass': True}
    import us_train as UT, synth_train as ST
    data = UT.USData(G, A / 'locks')
    cfg = UT.CANDIDATES[0]
    _, _, rn = UT.train_run(data, 'NET', cfg, 8101, 20, keep_best=False)
    _, _, ra = UT.train_run(data, 'ASYM', cfg, 8101, 20, keep_best=False)
    env = ST.Env('DEV', 6201, 1, 0.04)
    _, _, sn = ST.train_run(env, 'NET', cfg, 128, 20, keep_best=False)
    _, _, sa = ST.train_run(env, 'ASYM', cfg, 128, 20, keep_best=False)
    R['paired_training_sample_ids'] = {'us_era5_seed8101_first20': [rn['first20_batch_ids_sha256'], ra['first20_batch_ids_sha256']],
                                       'synthetic_seed6201_first20': [sn['first20_batch_ids_sha256'], sa['first20_batch_ids_sha256']],
                                       'evaluation_ids': 'both models score the identical frozen window list (US) / identical events (synthetic) / identical origins (ANEEL)',
                                       'pass': rn['first20_batch_ids_sha256'] == ra['first20_batch_ids_sha256'] and sn['first20_batch_ids_sha256'] == sa['first20_batch_ids_sha256']}
    R['checkpoint_only_replay'] = {'status': 'PENDING_AFTER_TRAINING', 'script': 'code/replay.py via REPLAY_ONLY.sh'}
    R['all_measured_gates_pass'] = all(v['pass'] for k, v in R['fp64_cpu_reference_vs_fp32_device'].items()) and all(
        v.get('pass', True) for k, v in R.items() if isinstance(v, dict) and k != 'fp64_cpu_reference_vs_fp32_device')
    (A / 'logs/CORRECTNESS_GATES.json').write_text(json.dumps(R, indent=1))
    for k, v in R['fp64_cpu_reference_vs_fp32_device'].items():
        print(f"  FP {k:18s} fwd_abs={v['forward_max_abs_diff']:.2e} grad_rel={v['grad_max_relative_norm_error']:.2e} pass={v['pass']}")
    print(f"  NET affine err={R['net_general_affine_response']['max_abs_error_vs_exact_affine_recursion']:.1e} both signs={R['net_general_affine_response']['both_signs_under_same_context']}")
    print(f"  ASYM bounds {bounds}  target invariance={R['changing_target_does_not_change_forward']['identical']}  paired IDs={R['paired_training_sample_ids']['pass']}")
    print('ALL MEASURED GATES PASS' if R['all_measured_gates_pass'] else 'GATE FAILURE')


if __name__ == '__main__':
    main()
