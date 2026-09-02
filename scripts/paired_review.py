"""Paired review of one result file's arms against a reference arm.

The review routine for every experiment in this project, written once so it is
not re-typed per review. It refuses to compare files that `check_comparable`
would reject, and it compares only on the (seed, fold) units both arms share,
so fold difficulty cancels and a missing unit is reported rather than averaged
over.

    paired_review.py RESULT.json --ref control
    paired_review.py RETEST.json --against G2.json --ref control   # arms from one file, reference from another

Positive delta = arm worse than the reference. Sign gate: an arm "wins" a
horizon when it is better on every unit; the fold count is what the ledger's
[B] grade reads, not the mean.
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    d = json.loads(Path(path).read_text())
    return d["rows"], d.get("config", {})


def by_unit(rows, arm):
    """Rows of one arm keyed by (seed, fold).

    A result file may spread one unit over several rows -- one per horizon,
    each carrying only its own `rmse_h{h}` key (the per-horizon control does
    this). Rows sharing (seed, fold) are merged; a key present in two rows with
    different values is a malformed file and is refused.
    """
    out = {}
    for r in rows:
        if r["arm"] != arm:
            continue
        k = (r["seed"], r["fold"]); u = out.setdefault(k, {})
        for kk, v in r.items():
            if kk in u and kk.startswith("rmse_h") and u[kk] != v:
                sys.exit(f"unit {k} of arm '{arm}': conflicting {kk} across rows")
            u.setdefault(kk, v)
    return out


def paired(A, B, h):
    ks = sorted(set(A) & set(B))
    if not ks:
        return None
    ks = [k for k in ks if f"rmse_h{h}" in A[k] and f"rmse_h{h}" in B[k]]
    if not ks:
        return None
    d = np.array([A[k][f"rmse_h{h}"] - B[k][f"rmse_h{h}"] for k in ks], float)
    ref = np.array([B[k][f"rmse_h{h}"] for k in ks], float)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return {"delta_pct": float(100 * d.mean() / ref.mean()),
            "worse": int((d > 0).sum()), "better": int((d < 0).sum()), "n": len(ks),
            "t": float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("nan"),
            "missing": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result"); ap.add_argument("--against", default=None)
    ap.add_argument("--ref", default="control"); ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24, 48])
    ap.add_argument("--json", default=None, help="also write the table here")
    a = ap.parse_args()

    ref_file = a.against or a.result
    if a.against:
        rc = subprocess.run([sys.executable, str(ROOT / "scripts/check_comparable.py"),
                             a.result, a.against], capture_output=True, text=True)
        if rc.returncode != 0:
            print(rc.stdout.strip().splitlines()[-1] if rc.stdout.strip() else "NOT COMPARABLE")
            sys.exit(2)
    rows, cfg = load(a.result); ref_rows, _ = load(ref_file)
    B = by_unit(ref_rows, a.ref)
    if not B:
        sys.exit(f"reference arm '{a.ref}' not found in {ref_file}")
    arms = a.arms or sorted({r["arm"] for r in rows if r["arm"] != a.ref})
    n_units = len(B)
    print(f"reference '{a.ref}' from {Path(ref_file).name}: {n_units} units · "
          f"panels {cfg.get('digest') or cfg.get('panel_digest')} · channels {cfg.get('channel_digest')}")
    print(f"{'arm':<22}" + "".join(f"{'h+'+str(h):>24}" for h in a.horizons))
    table = {}
    for arm in arms:
        A = by_unit(rows, arm); line = f"{arm:<22}"; table[arm] = {}
        for h in a.horizons:
            r = paired(A, B, h)
            if r is None:
                line += f"{'(no shared units)':>24}"; continue
            if r["n"] != n_units:
                r["missing"] = n_units - r["n"]
            table[arm][h] = r
            flag = "*" if r["missing"] else " "
            line += f"{r['delta_pct']:>+8.2f}% {r['worse']:>2}/{r['n']:<2}t={r['t']:>+5.2f}{flag}"
        print(line)
    if any(v.get("missing") for t in table.values() for v in t.values()):
        print("* = fewer shared units than the reference has; do not read that cell as a full-protocol result")
    if a.json:
        Path(a.json).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
