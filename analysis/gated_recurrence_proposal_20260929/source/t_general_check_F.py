#!/usr/bin/env python3
"""Exponent of the gate-feasibility gap F in the occupancy level (THEORY_GENERAL.md, G4 dichotomy).

Autocatalytic process B (a = p + g*y, b = R) with g = .30 > R = .20: the linear part expands, so the gated contraction
class violates R2 at first order and F should scale like ell^2 (the index of the gated class then like ell^1).
Externally forced process A (a = p, b = R - g*y) with g = .05: F comes only from the interaction, like S (ell^4).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
import t_general_check as G  # noqa: E402


def main():
    ells = [.01, .02, .04, .08]; out = {}
    for proc, g in (('B', .30), ('A', .05)):
        rows = [G.index(proc, 512, l, g, with_F=True) for l in ells]
        S = [r['S'] for r in rows]; F = [max(r['F'], 1e-300) for r in rows]; xQ = [r['xQ'] for r in rows]
        tot = [(s + f) / x for s, f, x in zip(S, F, xQ)]
        out[f'{proc}_g{g}'] = {'ell': ells, 'S': S, 'F': F, 'xQ': xQ,
                               'slopes': {'S': G.slope(ells, S), 'F': G.slope(ells, F) if min(F) > 1e-30 else None,
                                          'xQ': G.slope(ells, xQ), '(S+F)/xQ': G.slope(ells, tot)}}
        s = out[f'{proc}_g{g}']['slopes']
        print(f"process {proc}, g {g}: log-slopes in ell  S {s['S']:.2f}  F {s['F'] if s['F'] is None else round(s['F'], 2)}  "
              f"x_Q {s['xQ']:.2f}  (S+F)/x_Q {s['(S+F)/xQ']:.2f};  F/S at ell=.01: {F[0] / S[0]:.3g}")
    (W / 'results/t_general_check_F.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
