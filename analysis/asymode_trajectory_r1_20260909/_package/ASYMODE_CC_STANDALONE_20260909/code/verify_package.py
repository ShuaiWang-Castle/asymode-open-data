"""Verify package checksums using Python's standard library only."""
from pathlib import Path
import argparse
import hashlib
import json


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = root/'SHA256SUMS.txt'
    failures, count = [], 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split('  ',1)
        p = (root/rel).resolve()
        if not p.is_relative_to(root):
            raise ValueError(f'Unsafe manifest path: {rel}')
        if not p.is_file():
            failures.append({'path':rel,'reason':'missing'})
        else:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h != digest:
                failures.append({'path':rel,'reason':'hash_mismatch'})
        count += 1
    result = {'status':'PASS' if not failures else 'FAIL','files_checked':count,'failures':failures}
    print(json.dumps(result,indent=2))
    if failures:
        raise SystemExit(1)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args();verify(a.root)
