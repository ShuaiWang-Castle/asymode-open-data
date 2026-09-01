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
from asymode.panels import (MANIFEST_NAME, available, channel_digest,   # noqa: E402
                            channel_names, digest, read_manifest)

INTERIM = ROOT / "data" / "interim"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", required=True,
                    help="a name for this panel set, e.g. 'g1-convective'")
    ap.add_argument("--note", default="")
    ap.add_argument("--panels", nargs="*", default=None,
                    help="name the panel days explicitly; default is every panel "
                         "that has drivers")
    a = ap.parse_args()

    built = available(INTERIM)
    if not built:
        raise SystemExit(f"no built panels in {INTERIM}")
    have = sorted(a.panels) if a.panels else built
    missing = sorted(set(have) - set(built))
    if missing:
        raise SystemExit(f"named but not built: {missing}")
    chans = channel_names(INTERIM)
    old = read_manifest(INTERIM)
    new = {"generation": a.generation, "panels": have,
           "digest": digest(have), "channels": chans,
           "channel_digest": channel_digest(chans), "note": a.note}
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
    if old and old.get("channel_digest") not in (None, new["channel_digest"]):
        print(f"  channel set also changed: {old.get('channel_digest')} -> "
              f"{new['channel_digest']} ({len(chans)} channels). Results across "
              f"that change are not comparable even on identical panels.")
    
    # A manifest under data/ is not versioned -- data/ is ignored -- and the file
    # that says which panels a generation contained is exactly the thing that must
    # survive. Write a second, generation-named copy where git can see it.
    import shutil
    cfg = ROOT / "configs" / f"panel_manifest_{args.generation}.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(dst, cfg)
    print(f"versioned copy: {cfg.relative_to(ROOT)}")

print(f"  {len(have)} panels, {len(chans)} channels [{new['channel_digest']}]")


if __name__ == "__main__":
    main()
