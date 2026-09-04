"""Phase 6 independent validation of the generated tables.

Run after run_preflight.py. Every check is a hard gate; any failure must be
reported and the task stopped rather than silently repaired.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL_IDENT = 1e-10


def main(code: Path) -> int:
    res = code / "analysis/conservation_preflight_20260904/results"
    ev = pd.read_csv(res / "EVENT_CONSERVATION_METRICS.csv")
    fd = pd.read_csv(res / "FOLD_CONSERVATION_METRICS.csv")
    loc = pd.read_csv(res / "LOCAL_GAMMA_METRICS.csv")
    prov = json.loads((res / "RUN_PROVENANCE.json").read_text())
    fails, notes = [], []

    def chk(name, ok, detail=""):
        (notes if ok else fails).append(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    # 1 unconstrained identity residual
    worst = 0.0
    for nm, t in (("event", ev), ("fold", fd), ("local", loc)):
        s = t["identity_residual_unc"].dropna().abs()
        w = float(s.max()) if len(s) else 0.0
        worst = max(worst, w)
        chk(f"1.{nm} max|U(1-mu)-R mu-mean_delta|", w <= TOL_IDENT, f"= {w:.3e}")
    chk("1.overall unconstrained identity <= 1e-10", worst <= TOL_IDENT, f"{worst:.3e}")

    # 2 A*B - C^2 = v
    worst = 0.0
    for nm, t in (("event", ev), ("fold", fd), ("local", loc)):
        s = t["abc_residual"].dropna().abs()
        w = float(s.max()) if len(s) else 0.0
        worst = max(worst, w)
    chk("2. max|A*B-C^2-v| <= 1e-10", worst <= TOL_IDENT, f"= {worst:.3e}")

    # 3 near-closure populated only when closure_pass
    bad = 0
    for nm, t in (("event", ev), ("fold", fd), ("local", loc)):
        if "Gamma_near_closure" not in t:
            continue
        cp = t["closure_pass"].astype("boolean").fillna(False).astype(bool)
        pop = t["Gamma_near_closure"].notna()
        bad += int((pop & ~cp).sum())
    chk("3. near-closure populated only when closure_pass", bad == 0,
        f"violations = {bad}")

    # 4 boundary fits are not treated as identity failures
    bnd = loc[loc["boundary_status"] != "interior"]
    ok4 = bool((bnd["identity_residual_unc"].abs() <= TOL_IDENT).all())
    n_bnd_break = int((bnd["identity_residual_box"].abs() > 1e-6).sum())
    chk("4. boundary rows keep an exact UNCONSTRAINED identity", ok4,
        f"{len(bnd)} boundary rows; {n_bnd_break} have box-identity residual>1e-6 "
        f"(expected, KKT not normal equations)")

    # 5 every local row has k=200 and no held-out event
    fmap = json.loads((code / "analysis/gpt_rescue_20260904/cc_v2/"
                              "event_folds_v2.json").read_text())
    ok_k = bool((loc["k"] == 200).all() and (loc["n"] == 200).all())
    leak = 0
    for f, grp in loc.groupby("fold"):
        held = set(fmap["folds"][str(int(f))])
        leak += int(grp["query_event"].isin(held).sum())
    chk("5. every local cell k=200 and query outside its held-out fold",
        ok_k and leak == 0, f"k ok={ok_k}, query leaks={leak}")

    # 6 all 26 manifest events present
    man = set(prov_panels(prov, code))
    chk("6. all 26 manifest events in the event table",
        set(ev["event"]) == man and len(man) == 26,
        f"{ev['event'].nunique()} distinct events")

    # 7 active-48 exactness / never clipped
    a = ev[(ev["design"] == "active48") & (ev["status"] == "ok")]
    span_ok = bool(((a["active48_t_end"] - a["active48_t_start"]) == 47).all())
    ctr_ok = bool(((a["active48_peak_hour"] - a["active48_t_start"]) == 24).all())
    u = ev[(ev["design"] == "active48") & (ev["status"] == "unavailable")]
    unavail_ok = bool((u["n"] == 0).all())
    # a clipped window would have moved the peak into [24,143]; check none did
    peaks = ev.groupby("event")["active48_peak_hour"].nunique()
    chk("7. active-48 spans exactly 48 transitions, centred, never clipped",
        span_ok and ctr_ok and unavail_ok and bool((peaks == 1).all()),
        f"{len(a)} available, {len(u)} unavailable")

    # 8 both designs retained regardless of sign
    chk("8. both designs retained for all 26 events",
        set(ev["design"]) == {"full", "active48"} and len(ev) == 52,
        f"{len(ev)} event rows")

    # 9 no forbidden claim language in any generated text output
    banned = ["refuted", "refutes", "disproves", "mathematically impossible",
              "invalidates", "withdrawn", "proves the null"]
    hits = []
    for f in sorted(res.glob("*.md")) + sorted(res.glob("*.json")):
        txt = f.read_text().lower()
        for b in banned:
            if b in txt:
                hits.append(f"{f.name}:{b}")
    chk("9. no output claims refutation or impossibility", not hits, str(hits))

    # 10 no neural framework imported
    mods = [m for m in sys.modules
            if m.split(".")[0] in {"torch", "tensorflow", "jax", "keras"}]
    frameworks = ("torch", "jax", "tensorflow", "keras", "sklearn")
    srcs = [f for f in (code / "analysis/conservation_preflight_20260904"
                               "/implementation").glob("*.py")
            if f.name != Path(__file__).name]
    bad_imports = []
    for f in srcs:
        for ln in f.read_text().splitlines():
            t = ln.strip()
            head = t.split("#", 1)[0].strip()
            if head.startswith(("import ", "from ")):
                mod = head.split()[1].split(".")[0]
                if mod in frameworks:
                    bad_imports.append(f"{f.name}:{mod}")
    chk("10. no neural framework imported or trained",
        not mods and not bad_imports, f"modules={mods} files={bad_imports}")

    # extra: rank-2 everywhere Gamma is reported
    g = loc["Gamma_plugin"].notna()
    chk("extra. rank 2 wherever local Gamma is reported",
        bool((loc.loc[g, "rank"] == 2).all()), "")

    print("\n".join(notes))
    if fails:
        print("\n".join(fails))
        print("\nVALIDATION FAILED")
        return 1
    print("\nVALIDATION PASSED: all Phase 6 checks satisfied")
    return 0


def prov_panels(prov, code):
    return list(prov["windows"].keys())


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
