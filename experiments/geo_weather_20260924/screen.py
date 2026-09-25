"""Quick screen (program.md): outer folds 1-2 of the county-grouped design of the twelve-event panel, a fixed
number of training steps on all development units, seed 0, then the held-out counties' open-loop rollouts.

  python screen.py --label base_e3r2 --data e3r2 --arm W+Cin
  python screen.py --label pop_v3p   --data v3p  --arm W+Cin
Every arm of the same seed shares the host initialisation (paired init). Outputs (not in git):
runs/geo_weather_20260924/<label>/fold0<k>/{outer.npz, DONE.json}; evaluate with evaluate_screen.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "2")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode import gcrk_train as G  # noqa: E402

RUNS = ROOT / "runs" / "geo_weather_20260924"
SPLITS = ROOT / "experiments" / "open_gcrk_20260919" / "splits_e3r2.json"
DATA = {"e3r2": ROOT / "data" / "interim" / "open_gcrk" / "features_e3r2.npz",
        "v3p": ROOT / "data" / "interim" / "geo_weather" / "features_v3p.npz",
        "w1": ROOT / "data" / "interim" / "geo_weather" / "features_w1.npz",
        "w2d": ROOT / "data" / "interim" / "geo_weather" / "features_w2d.npz"}
SPLIT_FILES = {"e3r2": SPLITS, "v3p": SPLITS, "w1": HERE / "splits_w1.json", "w2d": HERE / "splits_w2d.json"}
PANEL = {"e3r2": "", "v3p": "", "w1": "w1_", "w2d": "w2d_"}   # prefix of the panel's eih_ and train_mask files


def load(data: str) -> dict:
    z = np.load(DATA[data])
    return {k: z[k] for k in z.files}


def attach_phi(F: dict, variant: str, keep: str | None = None) -> dict:
    """Exposure-integrated hazard features eih_<variant>.npz (build_eih.py) as F['phi']; `keep` = a regex on the
    feature names to use a subset."""
    import re
    z = np.load(ROOT / "data" / "interim" / "geo_weather" / f"eih_{variant}.npz")
    assert np.array_equal(z["fips"], F["fips"]) and np.array_equal(z["event"], F["event"])
    names = z["names"].astype(str)
    cols = np.arange(len(names)) if keep is None else np.array([i for i, n in enumerate(names) if re.search(keep, n)])
    F = dict(F); F["phi"] = z["phi"][..., cols]; F["phi_names"] = names[cols]
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--data", default="e3r2", choices=sorted(DATA))
    ap.add_argument("--arm", default="W+Cin")
    ap.add_argument("--phi", default=None, help="exposure-integrated hazard variant (arm W+Cin+H)")
    ap.add_argument("--keep", default=None, help="regex on hazard feature names")
    ap.add_argument("--train-mask", action="store_true", help="drop EAGLE-I artefact-flagged hours from the training loss")
    ap.add_argument("--mask-placebo", action="store_true", help="with --train-mask: the matched random placebo mask")
    ap.add_argument("--ctx-geo", nargs="*", default=None, help="geographic descriptors added to the county context")
    ap.add_argument("--xu-phi", default=None, help="regex of eih features appended to the damage inputs (hours 72+)")
    ap.add_argument("--xu-phi-variant", default="area")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--design", default="main", choices=["main", "event"], help="county-grouped or event-grouped folds")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    F = load(a.data)
    if a.phi:
        F = attach_phi(F, PANEL[a.data] + a.phi, a.keep)
    if a.ctx_geo:
        gi = [list(F["geo_features"].astype(str)).index(g) for g in a.ctx_geo]
        F = dict(F); F["ctx_extra"] = F["geo"][:, gi].astype(np.float32)
    if a.xu_phi:
        import re
        z = np.load(ROOT / "data" / "interim" / "geo_weather" / f"eih_{PANEL[a.data]}{a.xu_phi_variant}.npz")
        assert np.array_equal(z["fips"], F["fips"]) and np.array_equal(z["event"], F["event"])
        names = z["names"].astype(str); cols = [i for i, n in enumerate(names) if re.search(a.xu_phi, n)]
        extra = np.zeros(F["xu"].shape[:2] + (len(cols),), np.float32)
        extra[:, 72:] = z["phi"][..., cols].astype(np.float32)
        F = dict(F); F["xu"] = np.concatenate([F["xu"], extra], -1); F["xu_extra"] = len(cols)
        print("appended to xu:", list(names[cols]), flush=True)
    if a.train_mask:
        tm = ("train_mask_e3" if PANEL[a.data] == "" else f"train_mask_{PANEL[a.data].rstrip('_')}") + \
             ("_placebo" if a.mask_placebo else "") + ".npz"
        F = dict(F); F["m_train"] = np.load(ROOT / "data" / "interim" / "geo_weather" / tm)["m_train"]
        assert F["m_train"].shape == F["m"].shape and (F["m_train"] <= F["m"]).all()
    sp = json.loads(SPLIT_FILES[a.data].read_text())
    assert sp["n_units"] == len(F["fips"])
    for k in a.folds:
        out = RUNS / a.label / f"fold{k:02d}"
        if (out / "DONE.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        dev, held = np.array(sp[a.design][str(k)]["dev"]), np.array(sp[a.design][str(k)]["outer"])
        t0 = time.time()
        log = lambda s: print(f"[{a.label} f{k}] {s}", flush=True)  # noqa: E731
        e = G.refit(F, dev, a.steps, a.seed, a.arm, log=log)
        res = G.export(e, F, held)
        np.savez_compressed(out / "outer.npz", **res)
        torch.save(dict(model_state=e.model.state_dict(), stats=e.stats, arm=a.arm, steps=a.steps), out / "final.pt")
        extra = {}
        if e.model.haz_beta is not None:
            beta = e.model.haz_beta.detach().numpy()
            top = np.argsort(-beta)[:12]
            extra = dict(beta_nonzero=int((beta > 0).sum()), beta_top={str(F["phi_names"][i]): float(beta[i]) for i in top})
        (out / "DONE.json").write_text(json.dumps(dict(label=a.label, data=a.data, arm=a.arm, phi=a.phi, keep=a.keep, train_mask=a.train_mask, mask_placebo=a.mask_placebo, ctx_geo=a.ctx_geo, xu_phi=a.xu_phi, design=a.design, fold=k,
                                                       steps=a.steps, seed=a.seed, fit_loss=e.last_loss,
                                                       seconds=round(time.time() - t0, 1), **extra), indent=1) + "\n")
        log(f"done in {time.time() - t0:.0f} s, fit loss {e.last_loss:.4e}")


if __name__ == "__main__":
    main()
