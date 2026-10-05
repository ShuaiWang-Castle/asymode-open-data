"""Fixed 7 arms x 5 folds x 3 seeds; resume completed fits only."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess,sys,json
HERE=Path(__file__).resolve().parent
PRIVATE=HERE.parents[1].parent/'t05_artifacts'
ARMS=('HOST','BASE_GCRK','DOSE','DOSE_GCRK','SHARED','GEO_GAIN','GEO_MIX')
def job(a,f,s):
    name=f'{a}_f{f}_s{s}'
    with (PRIVATE/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(HERE/'t05_repeated_weather.py'),'train','--arm',a,'--fold',str(f),'--seed',str(s)],stdout=log,stderr=subprocess.STDOUT,check=True)
    return name
if __name__=='__main__':
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(job,a,f,s):(a,f,s) for f in range(5) for s in range(3) for a in ARMS}
        for fut in as_completed(fs):
            try:records.append(dict(job=fut.result(),status='complete'))
            except Exception as e:records.append(dict(job=fs[fut],status='failed',error=str(e)))
            print(len(records),'/105',records[-1],flush=True)
            (PRIVATE/'run_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    if any(r['status']!='complete' for r in records):raise SystemExit('T05 failed jobs')
