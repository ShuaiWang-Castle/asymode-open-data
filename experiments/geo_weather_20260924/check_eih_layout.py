"""Layout check for an eih_*.npz file: physical invariants that hold only if the columns match their names
(nested gust ramps x10 >= x15 >= x20 under every modulator and memory; modulator 'one' >= canopy >= canopy x leaf-on
for every psi and memory, since canopy fraction <= 1). usage: python check_eih_layout.py <file.npz>"""
import sys
import numpy as np

z = np.load(sys.argv[1]); names = list(z["names"].astype(str)); phi = z["phi"][:800].astype(np.float32)
col = lambda n: phi[..., names.index(n)]
bad = []
for tau in (0, 3, 12, 48):
    for mod in ("one", "canopy", "canopy_leafon", "wet"):
        for a, b in (("gust_x10", "gust_x15"), ("gust_x15", "gust_x20")):
            f = np.mean(col(f"{a}*{mod}@{tau}") >= col(f"{b}*{mod}@{tau}") - 1e-3)
            if f < 0.999: bad.append((f"{a}>={b} {mod}@{tau}", round(float(f), 4)))
    if "quadn" in sys.argv[1]:          # normalised modulators (mean one in the county) may exceed 'one' by design
        continue
    for p in ("gust_x10", "rain", "p_tw+0.0", "convective"):
        for a, b in (("one", "canopy"), ("canopy", "canopy_leafon")):
            f = np.mean(col(f"{p}*{a}@{tau}") >= col(f"{p}*{b}@{tau}") - 1e-3)
            if f < 0.999: bad.append((f"{p} {a}>={b} @{tau}", round(float(f), 4)))
print(sys.argv[1], "OK" if not bad else f"FAILED {len(bad)}: {bad[:6]}")
sys.exit(1 if bad else 0)
