"""Are these result files talking about the same samples?

Compares the panel digest and the protocol constants that decide which samples a
run scored. Prints a verdict rather than a diff: the question is binary and the
answer should not require reading two JSON files side by side.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

# Constants that change the sample set or the split. Two runs that disagree on
# any of these are not comparable no matter what their numbers look like.
KEYS = ["panel_digest", "horizon", "stride", "k", "seeds", "horizons"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()

    cfgs = {}
    for f in a.files:
        if not f.exists():
            raise SystemExit(f"no such result file: {f}")
        c = json.loads(f.read_text()).get("config", {})
        cfgs[f.name] = {k: c.get(k) for k in KEYS}
        if c.get("panel_digest") is None:
            print(f"  {f.name}: no panel digest -- written before the panel set "
                  f"was recorded, so what it scored cannot be established")

    bad = [k for k in KEYS if len({json.dumps(c[k], sort_keys=True) for c in cfgs.values()}) > 1]
    w = max(len(n) for n in cfgs)
    print(f"\n{'file':<{w}}  " + "  ".join(f"{k}" for k in KEYS))
    for n, c in cfgs.items():
        print(f"{n:<{w}}  " + "  ".join(str(c[k]) for k in KEYS))
    if bad:
        print(f"\nNOT COMPARABLE -- differ on: {bad}")
        sys.exit(1)
    print("\ncomparable: same panels, same folds, same horizons")


if __name__ == "__main__":
    main()
