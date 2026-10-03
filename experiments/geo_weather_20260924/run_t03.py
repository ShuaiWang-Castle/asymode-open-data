"""Run the fixed T03 budget; resume only completed jobs, fail on any subprocess error."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess,sys,json,os,hashlib,zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIVATE=Path(os.environ.get('T03_ARTIFACTS',str(ROOT.parent/'t03_artifacts')))

def run(args):
    name='_'.join(args).replace('--','')
    with (PRIVATE/(name+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(HERE/'t03_poster_trajectory.py'),*args],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name}: return {result.returncode}; see job log')
    return name

def main():
    cache=PRIVATE/'data.npz'
    audit=json.loads((HERE/'results/t03/build_audit.json').read_text())
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==audit['cache_sha256']
    with zipfile.ZipFile(cache) as archive:assert archive.testzip() is None
    jobs=[]
    for fold in range(5):
        jobs.append(['tree','--fold',str(fold)])
        for seed in range(3):
            for arm in ('GCRK','HOST','DOSE','GEO_MLP'):
                jobs.append(['neural','--fold',str(fold),'--seed',str(seed),'--arm',arm])
    outcomes=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(run,j):j for j in jobs}
        for fut in as_completed(pending):
            try:
                name=fut.result();outcomes.append(dict(job=name,status='complete'));print(len(outcomes),'/',len(jobs),'DONE',name,flush=True)
            except Exception as e:
                outcomes.append(dict(job=pending[fut],status='failed',error=str(e)));print(str(e),flush=True)
            (PRIVATE/'run_manifest.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    if any(r['status']!='complete' for r in outcomes):raise SystemExit('T03 has failed jobs')

if __name__=='__main__':main()
