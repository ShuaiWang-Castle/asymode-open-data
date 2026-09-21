"""TBc: the TB world of make_planted_worlds.py without unobserved heterogeneity (sigma_unit = sigma_block = 0).
Same truth network, same seeded random geography-to-code and code-to-damping/gain maps, same kappa (read from
results/planted_worlds.json); the generated path is the truth's deterministic rollout, quantised to whole
customers. Separates 'the protocol cannot learn a geography-conditioned memory' from 'it cannot learn it at
this noise level and sample size'. Writes features_synTBc.npz and splits_synTBc.json."""
import json, os, sys
import numpy as np, torch
import make_planted_worlds as M
C = M.C

F = C.load_features(); n = len(F["y"]); m = F["m"].astype(bool); y0 = F["y0"].astype(np.float64); cust = F["cust"].astype(np.float64)
rep = json.loads((C.HERE / "results" / "planted_worlds.json").read_text()); kappa = rep["kappa"]
model, stats = M.truth_model(F); b = M.make_batch(F, np.arange(n), stats); k = model.kernel
neutral = (3.0 * torch.atanh(k.geo_center.clamp(-0.999, 0.999)))[None].repeat(n, 1)
u0, r0 = M.rates(model, b, geo=neutral)
gen = torch.Generator().manual_seed(777)
d, r_, G = k.Vl.shape[0], k.Vl.shape[1], k.U.shape[1]
U_t = torch.randn(r_, G, generator=gen) / np.sqrt(G); Vl_t = torch.randn(d, r_, generator=gen); Vg_t = torch.randn(d, r_, generator=gen)
with torch.no_grad():
    k.U.copy_(U_t); k.Vl.copy_(kappa * Vl_t); k.Vg.copy_(0.5 * kappa * Vg_t)
u1, r1 = M.rates(model, b)
one = np.ones(n)
y = M.quantise(M.rollout(u1, r1, y0, one), cust); mu = M.rollout(u1, r1, y0, one); mu_n = M.rollout(u0, r0, y0, one)
arr = {k_: F[k_] for k_ in F}; arr["y"] = y.astype(np.float32); arr["mu"] = mu.astype(np.float32); arr["mu_neutral"] = mu_n.astype(np.float32)
yf = np.array(F["y_full"], copy=True); yf[:, 72:] = y; arr["y_full"] = yf.astype(np.float32)
np.savez_compressed(M.OUT / "features_synTBc.npz", **arr)
sp = json.loads((C.HERE / "splits_r2.json").read_text())
(C.HERE / "splits_synTBc.json").write_text(json.dumps(dict(n_units=sp["n_units"], main=sp["main"])) + "\n")
info = dict(kappa=kappa, planted_gap=M.rmse(mu_n, y, m) / max(M.rmse(mu, y, m), 1e-12) - 1, rmse_zero=M.rmse(np.zeros_like(y), y, m),
            rmse_truth=M.rmse(mu, y, m), rmse_neutral_truth=M.rmse(mu_n, y, m), neutral_vs_truth_mean_path=M.rmse(mu_n, mu, m))
rep["worlds"]["TBc"] = info
(C.HERE / "results" / "planted_worlds.json").write_text(json.dumps(rep, indent=1) + "\n")
print(json.dumps(info))
