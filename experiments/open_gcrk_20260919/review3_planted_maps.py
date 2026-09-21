"""Learned against planted county memory in world TB: per held-out unit, the mean memory length and gain of the
trained GCRK against the truth's (Spearman), the spread across units, and the size of the conditioning maps.
Writes results/planted_maps_TB.csv."""
import os, json
os.environ["OPEN_GCRK_ROUND"] = "r2"
import numpy as np, pandas as pd, torch
import make_planted_worlds as M
from scipy.stats import spearmanr
from asymode.asym_host import AsymODE
C = M.C
F = C.load_features(); n = len(F["y"])
kappa = json.loads((C.HERE / "results" / "planted_worlds.json").read_text())["kappa"]
truth, ts = M.truth_model(F); bt = M.make_batch(F, np.arange(n), ts); k = truth.kernel
gen = torch.Generator().manual_seed(777); d, r_, G = k.Vl.shape[0], k.Vl.shape[1], k.U.shape[1]
U_t = torch.randn(r_, G, generator=gen) / np.sqrt(G); Vl_t = torch.randn(d, r_, generator=gen); Vg_t = torch.randn(d, r_, generator=gen)
with torch.no_grad():
    k.U.copy_(U_t); k.Vl.copy_(kappa * Vl_t); k.Vg.copy_(0.5 * kappa * Vg_t); lam_t, _, gain_t = k.condition(bt["geo"])
tau_t, g_t = 1 / np.log1p(lam_t.numpy()), gain_t.numpy()
sp = json.loads((C.HERE / "splits_synTB.json").read_text())["main"]; rows = []
for f in ("1", "2", "3"):
    snap = torch.load(C.ROOT / "runs/open_gcrk_20260919/synTB/main/seed0" / f"fold0{f}" / "GCRK" / "final.pt", weights_only=False)
    m = AsymODE(F["xu"].shape[-1], F["xr"].shape[-1], F["xo"].shape[-1]); m.attach_gcrk(torch.zeros(G)); m.load_state_dict(snap["model_state"]); m.eval()
    idx = np.array(sp[f]["outer"]); b = M.make_batch(F, idx, snap["stats"])
    with torch.no_grad():
        lam, _, gain = m.kernel.condition(b["geo"])
    tau, g = 1 / np.log1p(lam.numpy()), gain.numpy()
    rows.append(dict(fold=f, t_star=int(snap["step"]), beta=float(torch.tanh(m.kernel.alpha.detach())), memory_sd_learned=float(tau.mean(1).std()),
                     memory_sd_truth=float(tau_t[idx].mean(1).std()), spearman_memory=float(spearmanr(tau.mean(1), tau_t[idx].mean(1)).statistic),
                     spearman_gain=float(spearmanr(g.mean(1), g_t[idx].mean(1)).statistic), Vl_norm=float(m.kernel.Vl.detach().norm()),
                     Vl_norm_truth=float((kappa * Vl_t).norm())))
df = pd.DataFrame(rows); df.to_csv(C.HERE / "results" / "planted_maps_TB.csv", index=False); print(df.round(3).to_string(index=False))
