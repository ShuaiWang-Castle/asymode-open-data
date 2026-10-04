"""Run the fixed 45-fit T04 pilot; no model/seed selection from outer scores."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import sys,subprocess,json
HERE=Path(__file__).resolve().parent
PRIVATE=HERE.parents[1].parent/'t04_artifacts'

def job(arm,fold,seed):
    name=f'{arm}_f{fold}_s{seed}'
    with (PRIVATE/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(HERE/'t04_simple_geography.py'),'train','--arm',arm,'--fold',str(fold),'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True)
    return name

if __name__=='__main__':
    PRIVATE.mkdir(parents=True,exist_ok=True);records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(job,a,f,s):(a,f,s) for f in range(5) for s in range(3) for a in ('SHARED','GEO_GAIN','GEO_MIX')}
        for fut in as_completed(fs):
            try:records.append(dict(job=fut.result(),status='complete'))
            except Exception as e:records.append(dict(job=fs[fut],status='failed',error=str(e)))
            print(len(records),'/45',records[-1],flush=True)
            (PRIVATE/'run_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    if any(r['status']!='complete' for r in records):raise SystemExit('T04 failures; inspect private logs')
