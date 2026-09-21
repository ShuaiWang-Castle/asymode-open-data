"""Open-data W vs GCRK campaign: splits, one-cell workers and a local queue.

Designs
  main   county-grouped outer folds (K = 5): every unit (county-event) of a county is
         held out together; inside each outer fold, three county-grouped inner folds
         select the training length (asymode.gcrk_train), then REFIT on all
         development units and export OUTER.
  loeo   leave-one-event-out robustness: OUTER = every unit of one event; the other
         events are the development set (inner folds county-grouped as above).
Arms: W (host) and GCRK (host + kernel). Seeds 0-4 initialise the networks; splits do
not depend on the seed, and W and GCRK of the same (seed, fold) share the host
initialisation.

Fold assignment (fixed, seeded, uses only the observed prefix hours 0-71): counties
are ordered by (first event, state, maximum prefix outage fraction, customers) and
consecutive blocks of K (inner: 3) receive a random permutation of the fold labels.

Usage
  python run.py splits
  python run.py worker --design main --seed 0 --fold 1 --arm GCRK
  python run.py queue --design main --workers 5
Outputs go to runs/open_gcrk_20260919/<design>/seed<s>/fold<ff>/<arm>/ (not in git).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode import gcrk_train as G  # noqa: E402

# OPEN_GCRK_ROUND selects a later round (PREREG Amendment 2); unset = round 1 as pre-registered
ROUND = os.environ.get("OPEN_GCRK_ROUND", "r1")
FEATURES = ROOT / "data" / "interim" / "open_gcrk" / ("features.npz" if ROUND == "r1" else f"features_{ROUND}.npz")
RUNS = ROOT / "runs" / "open_gcrk_20260919" / ("" if ROUND == "r1" else ROUND)
SPLITS = HERE / ("splits.json" if ROUND == "r1" else f"splits_{ROUND}.json")
K_OUTER, K_INNER, SPLIT_SEED = 5, 3, 20260919
SEEDS, ARMS = (0, 1, 2, 3, 4), ("W", "GCRK", "W+C", "W+G", "GCRK-S", "GCRK-P", "GCRK-K8", "W+Cin", "GCRK+Cin")
torch.set_num_threads(1)


def load_features() -> dict:
    z = np.load(FEATURES, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _blocks(order: list[str], k: int, rng) -> dict[str, int]:
    lab = {}
    for s in range(0, len(order), k):
        blk = order[s:s + k]
        perm = rng.permutation(k)[:len(blk)]
        for f, j in zip(blk, perm):
            lab[f] = int(j)
    return lab


def county_order(F: dict, units: np.ndarray) -> list[str]:
    d = pd.DataFrame(dict(fips=F["fips"][units], event=F["event"][units],
                          pre=np.nanmax(np.where(F["obs_full"][units, :72], F["y_full"][units, :72], np.nan), 1),
                          cust=F["cust"][units]))
    g = d.groupby("fips").agg(event=("event", "min"), pre=("pre", "max"), cust=("cust", "max")).reset_index()
    g["state"] = g.fips.str[:2]
    return g.sort_values(["event", "state", "pre", "cust", "fips"]).fips.tolist()


def inner_folds(F: dict, dev_units: np.ndarray, seed: int) -> list[dict]:
    lab = _blocks(county_order(F, dev_units), K_INNER, np.random.default_rng(seed))
    fl = np.array([lab[f] for f in F["fips"][dev_units]])
    return [dict(fit=dev_units[fl != j].tolist(), val=dev_units[fl == j].tolist()) for j in range(K_INNER)]


def splits_event_inner(base: Path = HERE / "splits.json"):
    """Round-2 splits (PREREG Amendment 2 (e)): the round-1 outer folds; main keeps its inner
    folds; every LOEO cell selects t* on inner folds that each leave one development event out."""
    F = load_features()
    ev = F["event"].astype(str)
    sp = json.loads(base.read_text())
    assert sp["n_units"] == len(ev), "round-2 features must keep round 1's units and order"
    for k, spec in sp["loeo"].items():
        dev = np.array(spec["dev"])
        spec["inner"] = [dict(fit=dev[ev[dev] != e].tolist(), val=dev[ev[dev] == e].tolist(), event=e)
                         for e in sorted(set(ev[dev]))]
    sp["loeo_inner"] = "leave one development event out"
    SPLITS.write_text(json.dumps(sp) + "\n")
    print({k: [len(i["val"]) for i in v["inner"]] for k, v in sp["loeo"].items()})


def splits_events12():
    """E3 splits (PREREG Amendment 3): 'main' = county-grouped as round 1 (five outer folds, three
    county-grouped inner folds); 'event' = four outer folds of three events each (events in date
    order, fold = rank mod 4), with three inner folds of three development events each (rank mod 3)."""
    F = load_features()
    n = len(F["fips"]); allu = np.arange(n); ev = F["event"].astype(str)
    outer_lab = _blocks(county_order(F, allu), K_OUTER, np.random.default_rng(SPLIT_SEED))
    of = np.array([outer_lab[f] for f in F["fips"]])
    main = {}
    for k in range(K_OUTER):
        held, dev = allu[of == k], allu[of != k]
        main[str(k + 1)] = dict(outer=held.tolist(), dev=dev.tolist(), inner=inner_folds(F, dev, SPLIT_SEED + 1 + k))
    events = sorted(set(ev))
    event = {}
    for k in range(4):
        out_ev = [e for i, e in enumerate(events) if i % 4 == k]
        dev_ev = [e for e in events if e not in out_ev]
        held, dev = allu[np.isin(ev, out_ev)], allu[~np.isin(ev, out_ev)]
        inner = []
        for j in range(3):
            val_ev = [e for i, e in enumerate(dev_ev) if i % 3 == j]
            inner.append(dict(fit=dev[~np.isin(ev[dev], val_ev)].tolist(), val=dev[np.isin(ev[dev], val_ev)].tolist(),
                              events=val_ev))
        event[str(k + 1)] = dict(events=out_ev, outer=held.tolist(), dev=dev.tolist(), inner=inner)
    for design in (main, event):
        cover = np.zeros(n, int)
        for spec in design.values():
            cover[spec["outer"]] += 1
            assert not set(spec["outer"]) & set(spec["dev"])
            for inn in spec["inner"]:
                assert not set(inn["fit"]) & set(inn["val"]) and set(inn["fit"]) | set(inn["val"]) == set(spec["dev"])
        assert (cover == 1).all()
    for spec in main.values():
        assert not set(F["fips"][spec["outer"]]) & set(F["fips"][spec["dev"]])
    SPLITS.write_text(json.dumps(dict(k_outer=K_OUTER, split_seed=SPLIT_SEED, n_units=n, main=main, event=event)) + "\n")
    print("main outer", [len(v["outer"]) for v in main.values()], "event outer", {k: (v["events"], len(v["outer"]))
                                                                                 for k, v in event.items()})


def make_splits():
    F = load_features()
    n = len(F["fips"])
    allu = np.arange(n)
    outer_lab = _blocks(county_order(F, allu), K_OUTER, np.random.default_rng(SPLIT_SEED))
    of = np.array([outer_lab[f] for f in F["fips"]])
    main = {}
    for k in range(K_OUTER):
        held, dev = allu[of == k], allu[of != k]
        main[str(k + 1)] = dict(outer=held.tolist(), dev=dev.tolist(), inner=inner_folds(F, dev, SPLIT_SEED + 1 + k))
    loeo = {}
    for j, ev in enumerate(sorted(set(F["event"].tolist()))):
        held, dev = allu[F["event"] == ev], allu[F["event"] != ev]
        loeo[str(j + 1)] = dict(event=ev, outer=held.tolist(), dev=dev.tolist(),
                                inner=inner_folds(F, dev, SPLIT_SEED + 101 + j))
    for design in (main, loeo):
        for spec in design.values():
            o, d = set(spec["outer"]), set(spec["dev"])
            assert not o & d and len(o | d) == n
            seen = []
            for inn in spec["inner"]:
                assert not set(inn["fit"]) & set(inn["val"]) and set(inn["fit"]) | set(inn["val"]) == d
                seen += inn["val"]
            assert sorted(seen) == sorted(d)
            if design is main:   # county-grouped: no county on both sides
                assert not set(F["fips"][spec["outer"]]) & set(F["fips"][spec["dev"]])
            for inn in spec["inner"]:
                assert not set(F["fips"][inn["fit"]]) & set(F["fips"][inn["val"]])
    SPLITS.write_text(json.dumps(dict(k_outer=K_OUTER, k_inner=K_INNER, split_seed=SPLIT_SEED,
                                      n_units=n, main=main, loeo=loeo)) + "\n")
    print("outer sizes", [len(v["outer"]) for v in main.values()], "loeo", {v["event"]: len(v["outer"]) for v in loeo.values()})


def cell_dir(design, seed, fold, arm) -> Path:
    return RUNS / design / f"seed{seed}" / f"fold{int(fold):02d}" / arm


def worker(design: str, seed: int, fold: int, arm: str):
    t0 = time.monotonic()
    F = G.geo_variant(load_features(), arm)
    spec = json.loads(SPLITS.read_text())[design][str(fold)]
    d = cell_dir(design, seed, fold, arm)
    if (d / "DONE.json").exists():
        print("exists", d); return
    d.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    inner = [(np.array(i["fit"]), np.array(i["val"])) for i in spec["inner"]]
    sel = G.select_steps(F, inner, seed, arm, log=log)
    pd.DataFrame(sel.pop("trace")).to_csv(d / "selection_trace.csv", index=False)
    e = G.refit(F, np.array(spec["dev"]), sel["best_step"], seed, arm, log=log)
    torch.save(e.snapshot(), d / "final.pt")
    ex = G.export(e, F, np.array(spec["outer"]))
    np.savez_compressed(d / "outer.npz", **ex)
    k = e.model.kernel
    kern = {} if k is None else dict(beta=float(torch.tanh(k.alpha).detach()), scale=float(k.scale),
                                     theta=float(k.threshold), ramp=k.ramp(), new_parameters=k.n_new_parameters())
    if e.model.level is not None:
        kern = dict(level_source=e.model.level_source, level_weight_norm=float(e.model.level.weight.norm()),
                    level_bias=float(e.model.level.bias), new_parameters=int(e.model.level.weight.numel() + 1))
    done = dict(design=design, seed=seed, fold=fold, arm=arm, **sel, n_dev=len(spec["dev"]), n_outer=len(spec["outer"]),
                kernel=kern, seconds=time.monotonic() - t0, completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                torch=torch.__version__)
    (d / "DONE.json").write_text(json.dumps(done, indent=1) + "\n")
    print("DONE", design, seed, fold, arm, sel["best_step"], round(time.monotonic() - t0), flush=True)


def queue(design: str, workers: int, seeds, arms):
    spec = json.loads(SPLITS.read_text())[design]
    tasks = [(s, int(f), a) for s in seeds for f in sorted(spec, key=int) for a in arms
             if not (cell_dir(design, s, f, a) / "DONE.json").exists()]
    # GCRK cells are ~4x slower; start them first so the tail is short
    tasks.sort(key=lambda x: (x[2] != "GCRK", x[0], x[1]))
    logs = RUNS / design / "logs"; logs.mkdir(parents=True, exist_ok=True)
    live = []
    print(f"{len(tasks)} cells pending, {workers} workers", flush=True)
    while tasks or live:
        while tasks and len(live) < workers:
            s, f, a = tasks.pop(0)
            fh = open(logs / f"{a}_s{s}_f{f:02d}.log", "a")
            p = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "worker", "--design", design, "--seed", str(s),
                                  "--fold", str(f), "--arm", a], stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
            live.append((p, fh, s, f, a)); print("START", s, f, a, p.pid, flush=True)
        for item in list(live):
            p, fh, s, f, a = item
            if p.poll() is not None:
                fh.close(); live.remove(item)
                print("FINISH" if p.returncode == 0 else f"FAILED rc={p.returncode}", s, f, a, flush=True)
        time.sleep(3)
    print("QUEUE COMPLETE", design, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["splits", "splits-event-inner", "splits-events12", "worker", "queue"])
    ap.add_argument("--design", choices=["main", "loeo", "event"], default="main")
    ap.add_argument("--seed", type=int); ap.add_argument("--fold", type=int); ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS))); ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    if a.action == "splits":
        make_splits()
    elif a.action == "splits-event-inner":
        splits_event_inner()
    elif a.action == "splits-events12":
        splits_events12()
    elif a.action == "worker":
        worker(a.design, a.seed, a.fold, a.arm)
    else:
        queue(a.design, a.workers, [int(s) for s in a.seeds.split(",")], a.arms.split(","))
