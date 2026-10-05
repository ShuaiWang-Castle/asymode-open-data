"""75 registered jobs; four processes, one numerical thread each; inspected resume."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess,sys,json
HERE=Path(__file__).resolve().parent
PRIVATE=HERE.parents[1].parent/'t06_artifacts'
ARMS=('DOSE','GCRK','CRK','CRK_SHUFFLE','GEO_MLP')
def job(arm,fold,seed):
    name=f'{arm}_f{fold}_s{seed}'
    with (PRIVATE/(name+'.log')).open('a') as log:
        subprocess.run([sys.executable,str(HERE/'t06_featured_subset.py'),'train','--arm',arm,'--fold',str(fold),'--seed',str(seed)],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    return name
if __name__=='__main__':
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(job,a,f,s):(a,f,s) for f in range(5) for s in range(3) for a in ARMS}
        for future in as_completed(jobs):
            try:records.append(dict(job=future.result(),status='complete'))
            except Exception as e:records.append(dict(job=jobs[future],status='failed',error=str(e)))
            (PRIVATE/'run_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
            print(len(records),'/75',records[-1],flush=True)
    if any(r['status']!='complete' for r in records):raise SystemExit('T06 failed jobs; inspect preserved logs')
