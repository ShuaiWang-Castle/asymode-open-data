"""Export already audited scalar input-pair contrasts; never export trajectories."""
from pathlib import Path
import csv
import hashlib
import gzip
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = ROOT / 'runs/geo_weather_20260924/d08_panel_20260929_iofix1'
OUT = HERE / 'results/v1/d08_pair_contrasts.csv'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve previous pair export')
    marker = json.loads((RUNTIME / 'FROZEN.json').read_text())
    assert marker['status'] == 'input_roster_frozen'
    result = json.loads(gzip.decompress((HERE / 'results/v1/d08_panel_symptoms.json.gz').read_bytes()))
    assert marker['roster_sha256'] == result['provenance']['roster_sha256']
    path = RUNTIME / 'symptoms_arrays.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result['provenance']['local_arrays_sha256']
    with np.load(path, allow_pickle=False) as z:
        data = {k.removeprefix('pair_'): z[k] for k in z.files if k.startswith('pair_')}
    assert len(data['kind']) == result['frozen_input_pair_outcome_descriptives']['accepted_pairs'] == 284
    assert all(v.shape == (284,) for v in data.values())
    assert all(np.isfinite(v).all() for k, v in data.items() if k != 'kind')
    fields = list(data)
    with OUT.open('x', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(fields)
        for i in range(284):
            writer.writerow([data[k][i].item() for k in fields])


if __name__ == '__main__':
    main()
