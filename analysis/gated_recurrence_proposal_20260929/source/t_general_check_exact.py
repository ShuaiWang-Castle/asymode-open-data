#!/usr/bin/env python3
"""Cleaner exponent check for Theorem G4: designs whose initial counts scale exactly (integers), smaller levels.
Same processes as t_general_check.py (A externally forced with crowding, B autocatalytic in the contraction regime)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
import t_general_check as G  # noqa: E402

def main():
    out = {}
    for proc in ('A', 'B'):
        ells = [.005, .01, .02, .04]            # with M = 2000 the initial counts are 0, 5, 10, 15 times ell/.005
        L = [G.index(proc, 2000, l, .05)['Lambda'] for l in ells]
        Ms = [500, 1000, 2000]                 # at ell = .02: counts 0, 5, 10, 15 times M/500
        LM = [G.index(proc, M, .02, .05)['Lambda'] for M in Ms]
        gs = [.0125, .025, .05]
        Lg = [G.index(proc, 2000, .02, g)['Lambda'] for g in gs]
        local = [float(np.log(L[i + 1] / L[i]) / np.log(2)) for i in range(3)]
        out[proc] = {'ell': ells, 'Lambda_ell': L, 'local_slopes_ell': local, 'slope_M': G.slope(Ms, LM), 'slope_g': G.slope(gs, Lg)}
        print(f"process {proc}: local ell-slopes {[round(s, 3) for s in local]} (pred 3); M-slope {out[proc]['slope_M']:.3f} (pred 1); g-slope {out[proc]['slope_g']:.3f} (pred 2)")
    (W / 'results/t_general_check_exact.json').write_text(json.dumps(out, indent=1) + '\n')

if __name__ == '__main__':
    main()
