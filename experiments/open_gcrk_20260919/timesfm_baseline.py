"""TimesFM 3.0 zero-shot baseline on the 216-hour panels (no training, local inference).

Official code and weights are read from a local runtime directory (--runtime, holding
source/src and weights/); the weights are used for local academic evaluation only and
are not redistributed. For every county-event unit:
  context     customers out over hours 0-71 (p x customers; the few unobserved hours
              filled forward, then backward)
  covariates  the 14 ERA5 channels over hours 0-215, passed as past-and-future covariates
              (arm WEATHER); none (arm HISTORY)
  forecast    144 hours; the model's point output (its median), clipped to
              [0, customers], divided by customers.
Evaluator settings follow the model's defaults: symmetric averaging, sorted quantiles,
no extra z-normalisation, no padding. Run with a Python that has the TimesFM
dependencies (torch, safetensors, huggingface_hub).
Writes runs/open_gcrk_20260919/timesfm/timesfm_<arm>.npz (event, fips, P, quantiles).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANELS = ROOT / "data" / "interim" / "open_gcrk"
OUT = ROOT / "runs" / "open_gcrk_20260919" / "timesfm"


def units():
    events = json.loads((HERE / "selected_events.json").read_text())["events"]
    ev, fips, ctx, cov, cust = [], [], [], [], []
    for e in events:
        z = np.load(PANELS / f"panel216_{e['event']}.npz")
        y = np.where(z["observed"], z["y"], np.nan)[:, :72].astype(np.float64)
        for i in range(y.shape[0]):
            row = y[i].copy()
            for t in range(1, 72):                      # forward fill
                if np.isnan(row[t]):
                    row[t] = row[t - 1]
            for t in range(70, -1, -1):                 # backward fill of a leading gap
                if np.isnan(row[t]):
                    row[t] = row[t + 1]
            assert np.isfinite(row).all()
            c = float(z["denominator"][i])
            ev.append(e["event"]); fips.append(str(z["fips"][i])); cust.append(c)
            ctx.append((row * c).astype(np.float32))
            cov.append(z["X"][i].T.astype(np.float32))          # (14, 216)
    return np.array(ev), np.array(fips), ctx, cov, np.array(cust)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arms", default="WEATHER,HISTORY")
    a = ap.parse_args()
    sys.path.insert(0, str(a.runtime / "source" / "src"))
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    torch.set_num_threads(1); torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    ev, fips, ctx, cov, cust = units()
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(a.runtime / "weights"), device=a.device,
                                          per_core_batch_size=a.batch, local_files_only=True))
    meta = dict(n_units=len(fips), device=a.device, torch=torch.__version__, context_hours=72, horizon=144,
                point="median", covariates="14 ERA5 channels, hours 0-215", clip="[0, customers]")
    for arm in a.arms.split(","):
        t0 = time.monotonic(); med, qs = [], []
        for lo in range(0, len(fips), a.batch):
            hi = min(lo + a.batch, len(fips))
            outs = list(model.predict_batch(ctx[lo:hi], horizon=144,
                                            past_future_covariates=cov[lo:hi] if arm == "WEATHER" else None,
                                            ts_ids=[f"{ev[j]}_{fips[j]}" for j in range(lo, hi)], return_quantiles=True,
                                            use_symmetric_averaging=True, make_positive=False, sort_quantiles=True,
                                            use_znorm=False, padding_mode="none"))
            for j, o in zip(range(lo, hi), outs):
                assert o.ts_id == f"{ev[j]}_{fips[j]}" and o.forecast.shape == (144,)
                med.append(np.clip(o.forecast, 0, cust[j]) / cust[j])
                qs.append(np.clip(o.quantiles, 0, cust[j]) / cust[j])
            if lo % (a.batch * 50) == 0:
                print(arm, hi, len(fips), round(time.monotonic() - t0), flush=True)
        np.savez_compressed(OUT / f"timesfm_{arm}.npz", event=ev, fips=fips, P=np.array(med, np.float32),
                            quantiles=np.array(qs, np.float32))
        meta[f"{arm}_seconds"] = time.monotonic() - t0
        (OUT / "manifest.json").write_text(json.dumps(meta, indent=1) + "\n")
        print("done", arm, round(meta[f"{arm}_seconds"]), flush=True)


if __name__ == "__main__":
    main()
