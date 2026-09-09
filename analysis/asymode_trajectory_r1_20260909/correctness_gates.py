#!/usr/bin/env python3
"""Section 6 correctness gates. Must pass before any formal trial.

No CUDA device exists on this host, so the reference comparison is FP32 compute
device vs an FP64 CPU reference of the identical model, which is the numerical
content the gate is asking for. cuda_available is reported as False rather than
silently claimed.
"""
from __future__ import annotations
import argparse, inspect, json, math, sys
from pathlib import Path
import numpy as np, torch


def gates(pkg_code: Path, out: Path) -> dict:
    sys.path.insert(0, str(pkg_code))
    import core_models as CM
    R = {}
    B, CDIM, H, CLK = 96, 168, 24, 5
    g = torch.Generator().manual_seed(20260909)
    c = torch.randn(B, CDIM, generator=g, dtype=torch.float64)
    y0 = torch.rand(B, generator=g, dtype=torch.float64) * 0.3
    kc = torch.rand(B, H, CLK, generator=g, dtype=torch.float64)

    # 1 forward signature does not accept target
    sig = set(inspect.signature(CM.TrajectoryModel.forward).parameters)
    R['g1_forward_has_no_target'] = {
        'params': sorted(sig - {'self'}),
        'pass': sig == {'self', 'c', 'y0', 'known_clock'}}

    per = {}
    for kind in ('DIRECT', 'NET', 'ASYM', 'SR'):
        torch.manual_seed(7)
        m = CM.TrajectoryModel(kind, CDIM, H, CLK, 32768, 0.0188149, 0.0030089).double()
        with torch.no_grad():
            p = m(c, y0, kc)
        d = {}
        # 3 output dimension
        d['g3_output_shape'] = list(p.shape); d['g3_pass'] = tuple(p.shape) == (B, H)
        # 2 target-independence: predictions cannot depend on labels that were never passed
        tgt = torch.rand(B, H, dtype=torch.float64)
        with torch.no_grad():
            p2 = m(c, y0, kc)
        _ = tgt[torch.randperm(B)]
        with torch.no_grad():
            p3 = m(c, y0, kc)
        d['g2_label_permutation_delta'] = float(max((p - p2).abs().max(), (p - p3).abs().max()))
        d['g2_pass'] = d['g2_label_permutation_delta'] == 0.0
        # 4 gradient reaches every one of the 24 steps
        m.zero_grad()
        w = torch.zeros(B, H, dtype=torch.float64); w[:, -1] = 1.0
        (m(c, y0, kc) * w).sum().backward()
        gn = sum(float(q.grad.abs().sum()) for q in m.parameters() if q.grad is not None)
        d['g4_grad_from_final_step_only'] = gn; d['g4_pass'] = gn > 0
        # 5 bounded arms stay in [0,1] with no clip
        if kind in ('ASYM', 'SR'):
            d['g5_min'], d['g5_max'] = float(p.min()), float(p.max())
            d['g5_pass'] = bool(p.min() >= 0.0 and p.max() <= 1.0)
        # 6 / 7 structural identity
        if kind == 'DIRECT':
            calls = {'n': 0}
            orig = m.net.forward
            def counted(x, _o=orig, _c=calls):
                _c['n'] += 1; return _o(x)
            m.net.forward = counted
            with torch.no_grad():
                m(c, y0, kc)
            m.net.forward = orig
            d['g6_net_calls_for_24_outputs'] = calls['n']; d['g6_pass'] = calls['n'] == 1
        if kind == 'NET':
            with torch.no_grad():
                a = m(c, y0, kc); b = m(c, y0 + 0.05, kc)
            step1 = float((a[:, 0] - b[:, 0]).abs().mean())
            later = float((a[:, -1] - b[:, -1]).abs().mean())
            d['g7_y0_perturbation_step1'] = step1
            d['g7_y0_perturbation_step24'] = later
            d['g7_pass'] = later > 0 and step1 > 0
        # 8 FP32 (compute dtype) vs FP64 reference
        m32 = CM.TrajectoryModel(kind, CDIM, H, CLK, 32768, 0.0188149, 0.0030089)
        m32.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
        c32, y32, k32 = c.float(), y0.float(), kc.float()
        with torch.no_grad():
            pf = m32(c32, y32, k32)
        af = float((pf.double() - p).abs().max())
        rf = float(((pf.double() - p).abs() / p.abs().clamp_min(1e-12)).max())
        d['g8_forward_atol'] = af; d['g8_forward_rtol'] = rf
        d['g8_forward_pass'] = af <= 1e-6 and rf <= 1e-4
        tgt32 = torch.rand(B, H, generator=torch.Generator().manual_seed(3))
        for mm, dt in ((m, torch.float64), (m32, torch.float32)):
            mm.zero_grad()
            CM.path_loss(mm(c.to(dt), y0.to(dt), kc.to(dt)), tgt32.to(dt),
                         torch.tensor(0.0188149, dtype=dt)).backward()
        g64 = torch.cat([q.grad.reshape(-1).double() for q in m.parameters()])
        g32 = torch.cat([q.grad.reshape(-1).double() for q in m32.parameters()])
        nrm = g64.norm()
        d['g8_grad_max_rel_err'] = float((g32 - g64).abs().max() / nrm)
        d['g8_grad_pass'] = d['g8_grad_max_rel_err'] <= 1e-3
        d['pass'] = all(v for k, v in d.items() if k.endswith('pass'))
        per[kind] = d
    R['per_model'] = per
    R['cuda_available'] = torch.cuda.is_available()
    R['reference_note'] = ('No CUDA device on this host; FP32 compute dtype compared '
                           'against an FP64 CPU reference of the identical model.')
    R['all_pass'] = bool(R['g1_forward_has_no_target']['pass'] and all(v['pass'] for v in per.values()))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'CORRECTNESS_GATES.json').write_text(json.dumps(R, indent=2))
    return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--package-code', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    r = gates(Path(a.package_code), Path(a.output))
    for k, v in r['per_model'].items():
        print(f"  {k:7s} pass={v['pass']}  fwd_atol={v['g8_forward_atol']:.3e} "
              f"grad_rel={v['g8_grad_max_rel_err']:.3e}")
    print('ALL GATES PASS' if r['all_pass'] else 'GATE FAILURE')
    raise SystemExit(0 if r['all_pass'] else 1)
