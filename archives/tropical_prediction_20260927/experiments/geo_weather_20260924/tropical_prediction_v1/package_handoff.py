"""Package verified public-development results and exact reproduction artifacts."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    audit = json.loads((here / 'results/evaluation_audit.json').read_text())
    assert audit['neural_fits'] == 70 and audit['tree_fits'] == 75
    assert audit['every_outer_row_once'] and not audit['sealed_confirmation_read']
    handoff = root / 'handoff'
    handoff.mkdir(exist_ok=True)
    base = '493aa54fb9233f27c4b6a10b5bacf3c1592f33ba'
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    patch = subprocess.check_output(['git', 'format-patch', '--stdout', '--binary', f'{base}..{head}'], cwd=root)
    files = []
    for directory in [here, root / 'src/asymode', root / 'runs/tropical_prediction_v1']:
        files += [p for p in directory.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts
                  and p.suffix not in {'.pyc', '.aux', '.out', '.synctex.gz'}
                  and not (p.parent == here / 'results' and p.suffix == '.log')]
    files += [root / 'data/interim/tropical_prediction_v1/development.npz',
              here.parent / 'data_v1/outcomes/development_systems.csv',
              root / 'FIREWALL.md']
    for p in root.glob('LICENSE*'):
        if p.is_file(): files.append(p)
    files = sorted(set(files))
    manifest = dict(base_commit=base, head_commit=head,
                    branch='research/tropical-evidence-20260926', evaluation=audit,
                    files={str(p.relative_to(root)): dict(bytes=p.stat().st_size,
                           sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files})
    manifest['patch_sha256'] = hashlib.sha256(patch).hexdigest()
    dest = handoff / 'tropical_prediction_results_20260927.zip'
    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files: z.write(p, str(p.relative_to(root)))
        z.writestr('branch-integration.patch', patch)
        z.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        z.writestr('START_HERE.md', (here / 'REPRODUCE.md').read_bytes())
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
    print(json.dumps(dict(path=str(dest),bytes=dest.stat().st_size,files=len(files),
                         sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),head=head)))


if __name__ == '__main__':
    main()
