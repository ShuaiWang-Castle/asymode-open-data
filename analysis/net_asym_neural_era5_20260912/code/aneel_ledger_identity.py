#!/usr/bin/env python3
"""Verify the ANEEL ledger by array identity, not by file name or container hash."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--ledger', required=True); ap.add_argument('--reference', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ref = json.loads(Path(a.reference).read_text())
z = np.load(a.ledger, allow_pickle=False)
rows, ok = [], 0
for k, spec in sorted(ref['arrays'].items()):
    if k not in z.files:
        rows.append({'array': k, 'status': 'MISSING'}); continue
    arr = np.ascontiguousarray(z[k], dtype='<f8')
    h = hashlib.sha256(arr.tobytes()).hexdigest()
    good = (list(arr.shape) == list(spec['shape'])) and h == spec['sha256_f64le']
    ok += good
    rows.append({'array': k, 'shape': list(arr.shape), 'sha256_f64le': h,
                 'status': 'PASS' if good else 'FAIL'})
extra = sorted(set(z.files) - set(ref['arrays']))
res = {'ledger_bytes': Path(a.ledger).stat().st_size,
       'ledger_file_sha256': hashlib.sha256(Path(a.ledger).read_bytes()).hexdigest(),
       'reference_canonical_file_sha256': ref['canonical_file_sha256'],
       'rule': ref['array_hash_rule'], 'arrays_expected': len(ref['arrays']),
       'arrays_pass': ok, 'extra_arrays_in_file': extra,
       'status': 'PASS' if ok == len(ref['arrays']) and not extra else 'FAIL', 'arrays': rows}
Path(a.out).write_text(json.dumps(res, indent=1))
print(f"ANEEL ledger identity: {ok}/{len(ref['arrays'])} arrays PASS, extra={extra} -> {res['status']}")
