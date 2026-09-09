"""Locate canonical ANEEL data or rebuild it from local records. No old handoff needed.
Only writes inside --output. Never selects a new cohort or downloads datasets.
Full legacy CSV rebuild is supported but may be expensive; it is not a model run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
REF = json.loads((PACKAGE/'metadata/LEDGER_REFERENCE.json').read_text())
EXPECTED_COLUMNS = json.loads((PACKAGE/'metadata/ACTUAL_COLUMNS.json').read_text())
SKIP_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.cache', '.mypy_cache'}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def cohort() -> pd.DataFrame:
    d = pd.read_csv(PACKAGE/'metadata/unit_selection.csv')
    d = d[d.evaluation_status.eq('included')].sort_values('group').reset_index(drop=True)
    if len(d) != 24 or d.company.nunique() != 16 or d.group.duplicated().any():
        raise ValueError('Packaged cohort metadata is invalid')
    return d


def walk_files(root: Path, exclude: Path):
    """Do not traverse symlink directories or scan outside explicitly supplied roots."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                         and (Path(directory)/d).resolve() not in {exclude, PACKAGE})
        for name in sorted(files):
            yield Path(directory)/name


def validate_ledger(path: Path) -> dict:
    """Container hashes may change after repacking; every used array must match exactly."""
    with np.load(path, allow_pickle=False) as a:
        stock_keys = sorted(k for k in REF['arrays'] if k.endswith('_y'))
        missing = [k for k in stock_keys if k not in a.files]
        if missing:
            raise ValueError(f'Missing {len(missing)} canonical stock arrays: {missing[:3]}')
        checked = []
        for key, expected in REF['arrays'].items():
            if key not in a.files:
                continue
            x = np.asarray(a[key])
            if list(x.shape) != expected['shape'] or x.dtype.kind not in 'fiu':
                raise ValueError(f'Bad shape/dtype: {key}')
            if not np.isfinite(x).all():
                raise ValueError(f'Nonfinite: {key}')
            digest = hashlib.sha256(np.ascontiguousarray(x, dtype='<f8').tobytes()).hexdigest()
            if digest != expected['sha256_f64le']:
                raise ValueError(f'Canonical array mismatch: {key}; do not loosen cohort/normalization')
            checked.append(key)
        maximum_balance_error = 0.0
        balance_count = 0
        for key in stock_keys:
            y = a[key]
            if y.min() < -1e-12 or y.max() > 1+1e-12:
                raise ValueError(f'Invalid bounded stock: {key}')
            prefix = key[:-1]
            if prefix+'fp' in a.files and prefix+'fm' in a.files:
                err = float(np.max(np.abs(np.diff(y)-(a[prefix+'fp']-a[prefix+'fm']))))
                if err > 2e-12:
                    raise ValueError(f'Balance failed: {key}')
                maximum_balance_error = max(maximum_balance_error, err)
                balance_count += 1
    digest = file_sha(path)
    return {'ledger_path':str(path.resolve()), 'ledger_sha256':digest,
            'canonical_container_identical':digest == REF['canonical_file_sha256'],
            'canonical_stock_arrays':len(stock_keys), 'checked_array_count':len(checked),
            'gross_audit_status':'PASS' if balance_count == 48 else 'PARTIAL_OR_NOT_AVAILABLE',
            'balance_pairs_checked':balance_count, 'max_balance_error':maximum_balance_error,
            'identity_basis':'exact per-array canonical float64 little-endian hashes'}


def hourly_ledger(rows: np.ndarray, year: int, denominator: float) -> dict:
    """Legacy convention: right-closed flux bins, s <= boundary < e for stock.
    Integer-second, timezone-naive timestamps are kept unchanged (no UTC/DST conversion).
    Ported algebraically unchanged from the previously audited ANEEL ledger function.
    """
    base = int(np.datetime64(f'{year}-01-01T00:00:00','s').astype('int64'))
    stop = int(np.datetime64(f'{year+1}-01-01T00:00:00','s').astype('int64'))
    step = 3600
    T = (stop-base)//step
    s, e, n = rows['start'], rows['end'], rows['n'].astype('float64')
    initial = n[(s <= base)&(e > base)].sum()
    fp, fm, corr = np.zeros(T), np.zeros(T), np.zeros(T)
    for tt, sign, out in [(s,1,fp),(e,-1,fm)]:
        ok = (tt > base)&(tt <= stop)
        v = tt[ok]-base
        j = (v-1)//step
        out[:] = np.bincount(j, weights=n[ok], minlength=T)
        corr += sign*np.bincount(j, weights=n[ok]*(((j+1)*step-v)/3600.), minlength=T)
    stock = np.r_[initial, initial+np.cumsum(fp-fm)]
    area = stock[:-1]+corr
    if np.min(stock) < -1e-5 or np.min(area) < -1e-5:
        raise ValueError('Negative ledger: audit records rather than clipping')
    return {'y':stock/denominator,'fp':fp/denominator,'fm':fm/denominator,'area':area/denominator}


def rebuild_from_records(record_map: dict[int, np.ndarray], output: Path) -> Path:
    arrays = {}
    for row in cohort().itertuples():
        records = record_map[int(row.group)]
        names = set(records.dtype.names or ())
        if not {'start','end','n','denom'}.issubset(names):
            raise ValueError('Record schema requires start/end/n/denom')
        valid = ((records['start'] > 0)&(records['end'] > records['start']) &
                 (records['n'] > 0)&(records['denom'] > 0)&(records['n'] <= records['denom']))
        records = records[valid]
        for year in (2018,2019):
            a = hourly_ledger(records, year, float(row.denominator))
            arrays.update({f'g{int(row.group)}_y{year}_{k}':v for k,v in a.items()})
    path = output/'rebuilt_selected_ledgers.npz'
    np.savez_compressed(path, **arrays)
    validate_ledger(path)  # identity gate, not just bounds or conservation
    return path


def read_selected_records(paths: list[Path]) -> dict[int,np.ndarray]:
    buckets = {}
    required = set(int(g) for g in cohort().group)
    for path in paths:
        m = re.search(r'pilot_records_g(\d+)\.npz$',path.name)
        if not m or int(m.group(1)) not in required:
            continue
        g = int(m.group(1))
        with np.load(path, allow_pickle=False) as a:
            if 'records' not in a.files:
                raise ValueError(f'No records key: {path}')
            x = a['records']
        if 'group' in (x.dtype.names or ()) and not np.all(x['group'] == g):
            raise ValueError(f'Record group mismatch: {path}')
        if g in buckets:
            # Duplicated archive copies must not double the event counts.
            if x.dtype != buckets[g].dtype or not np.array_equal(x,buckets[g]):
                raise ValueError(f'Conflicting raw copies for group {g}')
        else:
            buckets[g] = x
    if set(buckets) != required:
        raise ValueError(f'Incomplete selected records; missing groups {sorted(required-set(buckets))}')
    return buckets


def legacy_csv_records(paths_by_year: dict[int,Path]) -> dict[int,np.ndarray]:
    """Fixed cohort only. Preserve distinct complete source rows; dedup full-row hash pairs.
    This pathway requires the historical 18-column UTF8 semicolon schema, not modern fields.
    """
    selected = cohort()
    group_by_key = dict(zip(selected['key'].astype(str), selected.group.astype(int)))
    dtype = [('group','u4'),('start','i8'),('end','i8'),('n','i8'),('denom','i8'),('h1','u8'),('h2','u8')]
    years = []
    for year in (2018,2019):
        path = paths_by_year[year]
        archive = None
        try:
            if path.suffix.lower() == '.zip':
                archive = zipfile.ZipFile(path)
                names = [n for n in archive.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/')]
                if len(names) != 1:
                    raise ValueError(f'Expected exactly one CSV in {path}')
                stream = archive.open(names[0])
            else:
                stream = path.open('rb')
            chunks = []
            with stream:
                for d in pd.read_csv(stream, sep=';', dtype=str, encoding='utf-8-sig',
                                     keep_default_na=False, chunksize=200000):
                    if d.columns.tolist() != EXPECTED_COLUMNS:
                        raise ValueError(f'Unsupported legacy CSV schema: {path}')
                    keys = d.NumCPFCNPJ.str.strip()+'|'+d.IdeConjuntoUnidadeConsumidora.str.strip()
                    mask = keys.isin(group_by_key)
                    if not mask.any():
                        continue
                    keep = d.loc[mask]
                    a = np.empty(len(keep),dtype=dtype)
                    a['group'] = keys[mask].map(group_by_key).to_numpy()
                    for field, key in [('h1','0123456789123456'),('h2','fedcba9876543210')]:
                        a[field] = pd.util.hash_pandas_object(keep,index=False,hash_key=key).to_numpy()
                    for field, col in [('start','DatInicioInterrupcao'),('end','DatFimInterrupcao')]:
                        t = pd.to_datetime(keep[col],format='%Y-%m-%d %H:%M:%S',errors='coerce')
                        a[field] = t.to_numpy(dtype='datetime64[s]').astype('int64')
                    for field, col in [('n','NumUnidadeConsumidora'),('denom','NumConsumidorConjunto')]:
                        a[field] = pd.to_numeric(keep[col],errors='coerce').fillna(-1).astype('int64').to_numpy()
                    chunks.append(a)
            if not chunks:
                raise ValueError(f'No frozen cohort records in {path}')
            a = np.concatenate(chunks)
            hh = np.empty(len(a),dtype=[('h1','u8'),('h2','u8')]); hh['h1']=a['h1']; hh['h2']=a['h2']
            _, idx = np.unique(hh,return_index=True)
            years.append(a[np.sort(idx)])
        finally:
            if archive is not None:
                archive.close()
    both = np.concatenate(years)
    return {int(g):both[both['group']==g] for g in selected.group}


def prepare(roots: list[Path], output: Path, explicit_ledger: Path|None = None) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    attempts = []
    files = sorted(set(p.resolve() for root in roots for p in walk_files(root,output)
                       if p.suffix.lower() in {'.npz','.zip','.csv'}))
    candidates = [explicit_ledger.resolve()] if explicit_ledger else [p for p in files if p.suffix.lower()=='.npz' and not p.name.startswith('pilot_records_g')]
    # Filename is only a search priority, never an identity criterion.
    candidates = sorted(candidates,key=lambda p:('ledger' not in p.name.lower(),str(p)))
    result, mode = None, None
    for path in candidates:
        try:
            with np.load(path,allow_pickle=False) as a:
                if not any(k in a.files for k in REF['arrays'] if k.endswith('_y')):
                    continue
            result = validate_ledger(path)
            mode = 'EXISTING_LEDGER'
            break
        except Exception as e:
            attempts.append({'path':str(path),'error':str(e)})
    if result is None and explicit_ledger:
        write_json(output/'DATA_DISCOVERY_FAILURE.json',{'status':'BLOCKED_DATA','attempts':attempts})
        raise ValueError('Explicit ledger did not pass; see DATA_DISCOVERY_FAILURE.json')
    if result is None:
        record_paths = [p for p in files if re.search(r'pilot_records_g\d+\.npz$',p.name)]
        if record_paths:
            try:
                path = rebuild_from_records(read_selected_records(record_paths),output)
                result, mode = validate_ledger(path), 'REBUILT_SELECTED_RECORDS'
                result['raw_sources'] = [{'path':str(p),'sha256':file_sha(p)} for p in record_paths]
            except Exception as e:
                attempts.append({'stage':'selected_records','error':str(e)})
    if result is None:
        by_year = {}
        for year in (2018,2019):
            options = [p for p in files if str(year) in p.name and p.suffix.lower() in {'.csv','.zip'}
                       and ('interrup' in p.name.lower() or 'aneel' in p.name.lower())]
            options = sorted(options,key=lambda p:(p.suffix.lower()!='.zip',str(p)))
            if options:
                by_year[year] = options[0]
        if len(by_year)==2:
            try:
                path = rebuild_from_records(legacy_csv_records(by_year),output)
                result, mode = validate_ledger(path), 'REBUILT_LEGACY_CSV'
                result['raw_sources'] = [{'year':y,'path':str(p),'sha256':file_sha(p)} for y,p in by_year.items()]
            except Exception as e:
                attempts.append({'stage':'legacy_csv','error':str(e)})
    if result is None:
        write_json(output/'DATA_DISCOVERY_FAILURE.json',{'status':'BLOCKED_DATA','searched_roots':[str(r) for r in roots],
                    'attempts':attempts,'next_action':'Inspect local directories or write only a deterministic data adapter; do not request the deleted ZIP or change the cohort.',
                    'supported':'canonical stock NPZ; complete pilot_records_g*.npz; legacy 18-column annual CSV/ZIP'})
        raise ValueError('No canonical dataset resolved; see DATA_DISCOVERY_FAILURE.json')
    manifest = {'status':'PASS','schema_version':1,'mode':mode,'original_handoff_required':False,
                'data_roots':[str(r) for r in roots],**result,
                'selection_path':str((PACKAGE/'metadata/unit_selection.csv').resolve()),
                'selection_sha256':file_sha(PACKAGE/'metadata/unit_selection.csv'),
                'reference_sha256':file_sha(PACKAGE/'metadata/LEDGER_REFERENCE.json'),
                'stock_feature_only':True,'target_year_status':'REUSED_2019_RETROSPECTIVE_EVALUATION',
                'earlier_attempts':attempts}
    write_json(output/'DATA_MANIFEST.json',manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,action='append',help='Repeat for known local roots. Default: current directory.')
    p.add_argument('--ledger',type=Path,help='Optional exact local NPZ path. Identity checks still mandatory.')
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    roots = [r.resolve() for r in (a.data_root or [Path.cwd()])]
    for r in roots:
        if not r.is_dir():
            p.error(f'Not a directory: {r}')
    result = prepare(roots,a.output.resolve(),a.ledger)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
