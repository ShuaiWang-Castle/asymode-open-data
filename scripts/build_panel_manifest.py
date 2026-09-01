"""Write the panel manifest from what is currently built.

Run this once when a batch of covariate downloads finishes, then run the
experiments. Running it *between* experiments is what makes two result files
incomparable, so it prints the digest it wrote and the digest it replaced.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asymode.panels import MANIFEST_NAME, available, digest, read_manifest  # noqa: E402

INTERIM = ROOT / "data" / "interim"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", required=True,
                    help="a name for this panel set, e.g. 'g1-convective'")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    have = available(INTERIM)
    if not have:
        raise SystemExit(f"no built panels in {INTERIM}")
    old = read_manifest(INTERIM)
    new = {"generation": a.generation, "panels": have,
           "digest": digest(have), "note": a.note}
    (INTERIM / MANIFEST_NAME).write_text(json.dumps(new, indent=2))

    if old:
        gone = sorted(set(old["panels"]) - set(have))
        added = sorted(set(have) - set(old["panels"]))
        print(f"replaced generation {old.get('generation')!r} [{old.get('digest')}] "
              f"with {a.generation!r} [{new['digest']}]")
        if added:
            print(f"  added:   {added}")
        if gone:
            print(f"  removed: {gone}")
        print("  every result file written under the old digest is now "
              "incomparable to anything written from here on.")
    else:
        print(f"wrote generation {a.generation!r} [{new['digest']}]")
    print(f"  {len(have)} panels")


if __name__ == "__main__":
    main()
