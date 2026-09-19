"""Build the GCRK equivalence fixture from the frozen reference implementation.

The reference is a private, frozen PyTorch implementation of the same layer. This
script reads its source from --reference-dir (files `model.py` with the layer and
`kernel.py` with its response recurrence), executes only the layer definition with
minimal stubs for its host imports, and runs it on SYNTHETIC random inputs. No
data of any kind is involved: hidden sequences, geography, weights and targets are
all drawn from seeded generators below. The saved fixture holds those synthetic
inputs, the reference layer's parameter values and the reference outputs, so
tests/test_gcrk_equivalence.py can check this repository's re-implementation
without the reference present.

Usage:  python tests/fixtures/make_gcrk_fixture.py --reference-dir /path/to/reference
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
B_FIT, B, T, D, G = 9, 4, 216, 32, 34
TRAIN_CALLS = 4


def load_reference(ref: Path):
    """Return (ReferenceLayer class, prefix_reference, CONFIG) from the frozen source."""
    spec = importlib.util.spec_from_file_location("reference_kernel", ref / "kernel.py")
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    src = (ref / "model.py").read_text()
    start = src.index("CONFIG=dict(")
    stop = src.index("# Original host imports")
    body = src[start:stop]
    ns = dict(math=math, torch=torch, nn=nn, F=F, np=np,
              P=types.SimpleNamespace(Layer=nn.Module),
              # the reference's fused CPU solver is not needed: fast=False selects its
              # PyTorch recurrence, which it documents as the numerical reference
              E=types.SimpleNamespace(FR=types.SimpleNamespace(response_sequence=K.response_sequence), K=K))
    exec(compile(body, str(ref / "model.py"), "exec"), ns)
    cls = [v for k, v in ns.items() if isinstance(v, type) and issubclass(v, nn.Module)][0]
    return cls, ns["prefix_reference"], ns["CONFIG"]


def synthetic(dtype):
    """Hidden sequences with a quiet prefix and storm-like bumps, as ReLU features."""
    g = torch.Generator().manual_seed(20260919)
    t = torch.arange(T, dtype=dtype)
    def seq(n):
        base = 0.4 * torch.randn(n, 1, D, generator=g, dtype=dtype)
        noise = 0.15 * torch.randn(n, T, D, generator=g, dtype=dtype)
        c1 = 80 + 30 * torch.rand(n, 1, 1, generator=g, dtype=dtype)
        c2 = 140 + 40 * torch.rand(n, 1, 1, generator=g, dtype=dtype)
        amp = torch.randn(n, 2, D, generator=g, dtype=dtype)
        bump = (amp[:, :1] * torch.exp(-((t.view(1, -1, 1) - c1) / 6.0) ** 2)
                + amp[:, 1:] * torch.exp(-((t.view(1, -1, 1) - c2) / 9.0) ** 2))
        return torch.relu(base + noise + 1.5 * bump)
    h_fit, h = seq(B_FIT), seq(B)
    geo_fit = 1.3 * torch.randn(B_FIT, G, generator=g, dtype=dtype)
    geo = 1.3 * torch.randn(B, G, generator=g, dtype=dtype)
    return h_fit, h, geo_fit, geo, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", type=Path, required=True)
    a = ap.parse_args()
    cls, pref, cfg = load_reference(a.reference_dir)
    out = {}
    for tag, dtype in (("f32", torch.float32), ("f64", torch.float64)):
        h_fit, h, geo_fit, geo, gen = synthetic(dtype)
        torch.manual_seed(7)
        lin = nn.Linear(D, D).to(dtype)
        center = torch.tanh(geo_fit / 3).mean(0)
        ref = cls(lin, center).to(dtype)
        for name in ("U", "a0", "l0"):
            out[f"{tag}_init_{name}"] = getattr(ref, name).detach().numpy().copy()
        # move every kernel parameter away from its initial value (seeded), so each
        # path -- dissipation, interaction, gain, opening -- is exercised
        with torch.no_grad():
            for name, p in ref.named_parameters():
                if name in ("weight", "bias"):
                    continue
                p.add_(0.35 * torch.randn(p.shape, generator=gen, dtype=dtype))
            ref.alpha.fill_(0.8)
        cal = ref.calibrate_(h_fit, 40)
        out[f"{tag}_scale"] = float(ref.scale)
        out[f"{tag}_threshold"] = float(ref.threshold)
        ref.eval()
        for step in (150, 400):
            ref.training_step.fill_(step)
            y, di = ref.compute(h, geo, None, fast=False, diagnostics=True)
            out[f"{tag}_eval{step}_out"] = y.detach().numpy()
            if step == 400:
                for k in ("q", "deposit", "gate", "state", "damping", "interaction", "coordinate_gain"):
                    out[f"{tag}_eval{step}_{k}"] = di[k].detach().numpy()
        # gradients at step 400 in eval mode, of a fixed random projection of the output
        wproj = torch.randn(B, T, D, generator=gen, dtype=dtype)
        hh, gg = h.clone().requires_grad_(True), geo.clone().requires_grad_(True)
        ref.zero_grad()
        (ref.compute(hh, gg, None, fast=False) * wproj).sum().backward()
        out[f"{tag}_wproj"] = wproj.numpy()
        out[f"{tag}_grad_h"] = hh.grad.numpy(); out[f"{tag}_grad_geo"] = gg.grad.numpy()
        for name, p in ref.named_parameters():
            out[f"{tag}_grad_param_{name}"] = p.grad.detach().numpy().copy()
        # training mode: drop-path draws from the layer's private stream
        ref.train(); ref.training_step.fill_(120)
        state0 = ref._rng.get_state()
        masks = []
        for i in range(TRAIN_CALLS):
            y = ref.compute(h, geo, None, fast=False)
            masks.append(ref.last_mask)
            out[f"{tag}_train{i}_out"] = y.detach().numpy()
        out[f"{tag}_train_masks"] = np.array(masks)
        out[f"{tag}_rng_state"] = state0.numpy()
        # inputs and the reference parameters used
        for k, v in dict(h_fit=h_fit, h=h, geo_fit=geo_fit, geo=geo, center=center).items():
            out[f"{tag}_in_{k}"] = v.numpy()
        for name, p in ref.named_parameters():
            out[f"{tag}_param_{name}"] = p.detach().numpy().copy()
        out[f"{tag}_calibration_step"] = cal["step"]
    np.savez_compressed(HERE / "gcrk_reference_synthetic.npz", **out)
    digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.reference_dir / "model.py", a.reference_dir / "kernel.py")}
    meta = dict(reference_source_sha256=digest, reference_solver="PyTorch recurrence (fast=False)",
                reference_private_dropout_seed_offset=1, shapes=dict(B_fit=B_FIT, B=B, T=T, D=D, G=G),
                config_constants={k: cfg[k] for k in ("memory_min", "memory_max", "memory_init", "a_max",
                                                      "gain_log_bound", "gate_quantile", "gate_slope",
                                                      "scale_floor", "drop_path", "warmup_steps")},
                data="synthetic random tensors only (seeded); no observations of any kind")
    (HERE / "gcrk_reference_synthetic.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote fixture", {k: v.shape if hasattr(v, "shape") else v for k, v in list(out.items())[:6]})


if __name__ == "__main__":
    main()
