"""TimesFM 3.0 zero-shot on the development tranche of the designed panel (no training, local inference), as
open_gcrk_20260919/timesfm_baseline.py: official code and weights from a local runtime directory (--runtime, holding
source/src and weights/), used for local academic evaluation only and not redistributed. For every county-event:
  context     customers out over hours 0-71 (p x customers; unobserved hours filled forward, then backward)
  covariates  the 14 ERA5 channels of the host over hours 0-215 (past and future covariates)
  forecast    144 hours; the model's point output (its median), clipped to [0, customers], divided by customers.
Run with a Python that has torch, safetensors and huggingface_hub.
Writes runs/geo_weather_20260924/timesfm_v1D/timesfm_WEATHER.npz (system, fips, P [U, 144], quantiles)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FEAT = ROOT / "data" / "interim" / "panel_v1" / "features_v1D.npz"
OUT = ROOT / "runs" / "geo_weather_20260924" / "timesfm_v1D"


def units():
    z = np.load(FEAT)                  # every z[key] re-reads the array from the archive: read each once
    y = np.where(z["obs_full"], z["y_full"], np.nan)[:, :72].astype(np.float64)
    xu = z["xu"][:, :, :14]
    cust = z["cust"].astype(np.float64)
    ctx = []
    for i in range(len(y)):
        row = y[i].copy()
        for t in range(1, 72):
            if np.isnan(row[t]):
                row[t] = row[t - 1]
        for t in range(70, -1, -1):
            if np.isnan(row[t]):
                row[t] = row[t + 1]
        row = np.nan_to_num(row, nan=0.0)
        ctx.append((row * cust[i]).astype(np.float32))
    names = list(z["damage_features"].astype(str))[:14]
    assert names == list(z["weather_channels"].astype(str)), names
    cov = [np.ascontiguousarray(xu[i].T, dtype=np.float32) for i in range(len(y))]
    return z["system"].astype(str), z["fips"].astype(str), ctx, cov, cust


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(a.runtime / "source" / "src"))
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    torch.set_num_threads(1); torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    sysv, fips, ctx, cov, cust = units()
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(a.runtime / "weights"), device=a.device,
                                          per_core_batch_size=a.batch, local_files_only=True))
    t0 = time.monotonic(); med, qs = [], []
    for lo in range(0, len(fips), a.batch):
        hi = min(lo + a.batch, len(fips))
        outs = list(model.predict_batch(ctx[lo:hi], horizon=144, past_future_covariates=cov[lo:hi],
                                        ts_ids=[f"{sysv[j]}_{fips[j]}" for j in range(lo, hi)], return_quantiles=True,
                                        use_symmetric_averaging=True, make_positive=False, sort_quantiles=True,
                                        use_znorm=False, padding_mode="none"))
        for j, o in zip(range(lo, hi), outs):
            assert o.ts_id == f"{sysv[j]}_{fips[j]}" and o.forecast.shape == (144,)
            med.append(np.clip(o.forecast, 0, cust[j]) / cust[j])
            qs.append(np.clip(o.quantiles, 0, cust[j]) / cust[j])
        if lo % (a.batch * 100) == 0:
            print(hi, len(fips), round(time.monotonic() - t0), flush=True)
    np.savez_compressed(OUT / "timesfm_WEATHER.npz", system=sysv, fips=fips, P=np.array(med, np.float32),
                        quantiles=np.array(qs, np.float32))
    meta = dict(n_units=len(fips), device=a.device, torch=torch.__version__, context_hours=72, horizon=144,
                point="median", covariates="14 ERA5 channels, hours 0-215", clip="[0, customers]",
                seconds=round(time.monotonic() - t0))
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("done", meta, flush=True)


if __name__ == "__main__":
    main()
