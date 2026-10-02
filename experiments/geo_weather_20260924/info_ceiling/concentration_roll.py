"""Concentration of the restoration-kernel gain over weather systems (held-out folds, origin shifted by `shift` hours):
each system's share of the design-weighted squared-error reduction of the kernel arm against the retrained host, and
the pooled improvement without the systems that contribute most.
usage: python concentration_roll.py [shift] -> results/v1/info_ceiling/roll_concentration_<shift>.json"""
from __future__ import annotations

import json
import sys

import numpy as np

from probe import EXP, RES, load_static
from score_dose import collect

sys.path.insert(0, str(EXP))


def main(shift: int) -> None:
    import screen
    F = screen.shift_origin(screen.load("v1D"), shift); s = load_static()
    y, m, w = F["y"].astype(float), F["m"].astype(float), s["w"].astype(float)
    sysv, reg = s["system"].astype(str), s["regime"].astype(str); n = len(y)
    Ph, Pk = collect(f"v1r{shift}_host_s0", n)[0], collect(f"v1r{shift}_rk_s0", n)[0]
    av = ~np.isnan(Ph).any(1) & ~np.isnan(Pk).any(1) & (m.sum(1) > 0)
    sh = w * (m * (np.nan_to_num(Ph) - y) ** 2).sum(1); sk = w * (m * (np.nan_to_num(Pk) - y) ** 2).sum(1)
    systems = np.unique(sysv[av])
    gain = np.array([(sh - sk)[av & (sysv == q)].sum() for q in systems]); base = np.array([sh[av & (sysv == q)].sum() for q in systems])
    order = np.argsort(-gain); tot = gain.sum()
    rel = lambda keep: float(1 - np.sqrt((base[keep] - gain[keep]).sum() / base[keep].sum()))  # noqa: E731
    out = dict(systems=int(len(systems)), better=int((gain > 0).sum()), worse=int((gain < 0).sum()), pooled=rel(np.ones(len(systems), bool)),
               top=[dict(system=str(systems[j]), regime=str(reg[sysv == systems[j]][0]), share_of_gain=float(gain[j] / tot), share_of_host_error=float(base[j] / base.sum())) for j in order[:6]],
               without_top={str(k): rel(np.isin(np.arange(len(systems)), order[k:])) for k in (1, 2, 3, 5)},
               worst=[dict(system=str(systems[j]), regime=str(reg[sysv == systems[j]][0]), share_of_gain=float(gain[j] / tot)) for j in order[::-1][:3]])
    (RES / f"roll_concentration_{shift}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"systems {out['systems']}: better {out['better']}, worse {out['worse']}; pooled {100 * out['pooled']:+.2f}%")
    print("top:", [(q["system"], q["regime"], f"{100 * q['share_of_gain']:.0f}% of the gain", f"{100 * q['share_of_host_error']:.0f}% of the host error") for q in out["top"]])
    print("without the top systems:", {k: f"{100 * v:+.2f}%" for k, v in out["without_top"].items()}, "| worst:", [(q["system"], q["regime"], f"{100 * q['share_of_gain']:+.1f}%") for q in out["worst"]])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 48)
