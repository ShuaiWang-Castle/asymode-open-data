"""Post-hoc diagnostic (descriptive, added after round 2): is the host's shortfall on tropical severe county-events a
consequence of its training weights? The host minimises the regime-normalised loss (weight w_i / Z_r, Z_r = the
design-weighted all-zero SSE of regime r on the fitting units, screen.design_weighted); the probes used w_i alone.
M1 is refitted with the host's weights (M1z), everything else as probe.py. Writes traj_M1z.npy and
results/v1/info_ceiling/info_ceiling_weighting.json."""
from __future__ import annotations

import json
import time

import numpy as np

from probe import OUT, REGIMES, RES, TRAJ_PARAMS, TRAJ_ROUNDS, VARIANTS, folds, host_paths, load_static, traj_rows


def main() -> None:
    import lightgbm as lgb
    s = load_static()
    y, m, w, reg = s["y"], s["m"], s["w"].astype(float), s["regime"]
    Wh, Rh = np.load(OUT / "Wh.npy"), np.load(OUT / "Rh.npy")
    f = OUT / "traj_M1z.npy"
    if not f.exists():
        P = np.full(y.shape, np.nan, np.float32); t0 = time.time()
        for k, dev, outer in folds():
            u = np.zeros(len(w))
            for r in REGIMES:
                ii = dev[reg[dev] == r]
                Z = float((w[ii, None] * m[ii] * y[ii].astype(float) ** 2).sum())
                if Z > 0:
                    u[ii] = w[ii] / Z
            u *= m[dev].sum() / (m[dev] * u[dev, None]).sum()
            X, ui, ti = traj_rows(s, Wh, Rh, VARIANTS["M1"], dev, True)
            ok = m[ui, ti] > 0
            bst = lgb.train(TRAJ_PARAMS, lgb.Dataset(X[ok], y[ui, ti][ok], weight=u[ui][ok]), TRAJ_ROUNDS)
            Xo, uo, to = traj_rows(s, Wh, Rh, VARIANTS["M1"], outer, False)
            P[uo, to] = np.clip(bst.predict(Xo), 0, 1)
            print(f"M1z fold {k} ({time.time() - t0:.0f} s)", flush=True)
        np.save(f, P)
    yy, mm = y.astype(float), m.astype(float)
    peak = s["peak"].astype(float); S = peak >= .10; nonS = ~S
    host5, _ = host_paths(len(y))
    paths = {"host5": host5, "M1": np.load(OUT / "traj_M1.npy").astype(float), "M1z": np.load(f).astype(float)}

    def rmse(P, sub):
        e = np.where(mm > 0, P - yy, 0)
        return float(np.sqrt((w * (mm * e ** 2).sum(1))[sub].sum() / (w * mm.sum(1))[sub].sum()))

    out = {}
    for k in ("M1", "M1z"):
        out[k] = dict(S_vs_host5=1 - rmse(paths[k], S) / rmse(host5, S), all_vs_host5=1 - rmse(paths[k], np.ones(len(y), bool)) / rmse(host5, np.ones(len(y), bool)),
                      nonS_vs_host5=1 - rmse(paths[k], nonS) / rmse(host5, nonS), false_peaks=int(((paths[k].max(1) >= .10) & nonS).sum()),
                      S_by_regime={r: 1 - rmse(paths[k], S & (reg == r)) / rmse(host5, S & (reg == r)) for r in REGIMES},
                      all_by_regime={r: 1 - rmse(paths[k], reg == r) / rmse(host5, reg == r) for r in REGIMES})
    (RES / "info_ceiling_weighting.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        print(k, f"S {100 * v['S_vs_host5']:+.2f}%  all {100 * v['all_vs_host5']:+.2f}%  nonS {100 * v['nonS_vs_host5']:+.2f}%  FP {v['false_peaks']}")
        print("   S by regime", {r: f"{100 * x:+.1f}%" for r, x in v["S_by_regime"].items()})
        print("   all by regime", {r: f"{100 * x:+.1f}%" for r, x in v["all_by_regime"].items()})


if __name__ == "__main__":
    main()
