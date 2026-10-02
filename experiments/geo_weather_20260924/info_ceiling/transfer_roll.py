"""Transfer of the restoration-kernel screen to other forecast origins without retraining: the host and the kernel
arm trained on the panel shifted by `train` hours are rolled out, held-out folds only, on the panel shifted by each of
the other origins (inputs, origin stock, outage-history inputs and regional burden recomputed for that origin).
Design-weighted RMSE of the kernel arm against the host over the first 48 h; family-cluster bootstrap within regime.
usage: python transfer_roll.py [train] -> results/v1/info_ceiling/roll_transfer_<train>.json"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from geo_evidence import fold_of_unit
from probe import BOOT_SEED, B, EXP, REGIMES, RES, RUNS, load_static

sys.path.insert(0, str(EXP))
ORIGINS = (0, 24, 48, 72, 96)


def main(train: int) -> None:
    import screen
    from asymode import gcrk_train as G
    torch.set_num_threads(2)
    F0 = screen.load("v1D"); s = load_static()
    w, reg, fam, peak = s["w"].astype(float), s["regime"].astype(str), s["family"].astype(str), s["peak"].astype(float)
    n = len(w); fold = fold_of_unit(n)
    sp = json.loads(screen.SPLIT_FILES["v1D"].read_text())
    keys = sorted(set(zip(reg, fam))); gid = {kk: j for j, kk in enumerate(keys)}; g = np.array([gid[kk] for kk in zip(reg, fam)])
    rng = np.random.default_rng(BOOT_SEED); counts = np.zeros((B, len(keys)))
    for r in REGIMES:
        js = np.array([gid[kk] for kk in keys if kk[0] == r]); counts[:, js] = rng.multinomial(len(js), np.full(len(js), 1 / len(js)), size=B)

    def rel(a, b, sub):
        def gs(v):
            o = np.zeros(len(keys)); np.add.at(o, g[sub], (w * v)[sub]); return o
        qa, qb = counts @ gs(a), counts @ gs(b); okb = qb > 0
        d = 1 - np.sqrt(qa[okb] / qb[okb])
        return dict(point=float(1 - np.sqrt((w * a)[sub].sum() / (w * b)[sub].sum())), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])

    arms = {"host": ("W+Cin", False), "rk": ("W+Cin+RK", True)}
    folds = [k for k in range(1, 6) if all((RUNS / f"v1r{train}_{a}_s0" / f"fold{k:02d}" / "final.pt").exists() for a in arms)]
    out = {"trained_on_shift": train, "folds": folds}
    f = lambda x: f"{100 * x['point']:+.2f}% [{100 * x['ci95'][0]:+.2f}, {100 * x['ci95'][1]:+.2f}]"  # noqa: E731
    for d in ORIGINS:
        Fs = screen.shift_origin(F0, d); Fb = screen.attach_burden(Fs, "v1D", d, "real")
        y, mfull, y0 = Fs["y"].astype(float), Fs["m"].astype(float), Fs["y0"].astype(float)
        m = mfull * (np.arange(144) < 48)[None, :]
        sse = {a: np.full(n, np.nan) for a in arms}; full = {a: np.full(n, np.nan) for a in arms}
        for k in folds:
            held = np.array(sp["event"][str(k)]["outer"]); dev = np.array(sp["event"][str(k)]["dev"])
            for a, (arm, burden) in arms.items():
                ck = torch.load(RUNS / f"v1r{train}_{a}_s0" / f"fold{k:02d}" / "final.pt", weights_only=False)
                e = G.Engine(Fb if burden else Fs, dev[:64], None, 0, arm)
                e.model.load_state_dict(ck["model_state"]); e.stats = ck["stats"]
                if e.model.rest is not None:
                    e.model.rest.open = True
                P = G.export(e, Fb if burden else Fs, held)["P"].astype(float)
                sse[a][held] = (m[held] * (P - y[held]) ** 2).sum(1); full[a][held] = (mfull[held] * (P - y[held]) ** 2).sum(1)
        av = ~np.isnan(sse["host"]) & ~np.isnan(sse["rk"]) & (m.sum(1) > 0); act = y0 >= .01
        res = dict(units=int(av.sum()), active_units=int((av & act).sum()), all=rel(sse["rk"], sse["host"], av), active=rel(sse["rk"], sse["host"], av & act),
                   not_active=rel(sse["rk"], sse["host"], av & ~act), S=rel(sse["rk"], sse["host"], av & (peak >= .10)),
                   all_by_regime={r: rel(sse["rk"], sse["host"], av & (reg == r))["point"] for r in REGIMES})
        if d == 0:                      # the registered origin before the storm: the whole 144 h window
            res["full_window"] = dict(all=rel(full["rk"], full["host"], av), S=rel(full["rk"], full["host"], av & (peak >= .10)))
        out[str(d)] = res
        print(f"origin +{d} h ({'trained' if d == train else 'transfer'}): units {res['units']} active {res['active_units']} | first 48 h: all {f(res['all'])} | active {f(res['active'])} | not active {f(res['not_active'])} | S {f(res['S'])} |",
              {r: f"{100 * v:+.1f}%" for r, v in res["all_by_regime"].items()}, ("| 144 h: all " + f(res["full_window"]["all"]) + " S " + f(res["full_window"]["S"])) if d == 0 else "", flush=True)
    (RES / f"roll_transfer_{train}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 48)
