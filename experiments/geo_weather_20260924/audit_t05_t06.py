"""Independent arithmetic checks from individual OOF exports, without scorer helpers."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def audit(label):
    out=HERE/'results'/label;private=ROOT.parent/(label+'_artifacts')
    d=dict(np.load(private/'data.npz',allow_pickle=False));n=len(d['y'])
    metrics=pd.read_csv(out/'metrics.csv');arms=metrics.model.unique();checks=0;error=0.
    per=pd.read_csv(out/'per_seed.csv')
    ensemble={}
    for arm in arms:
        allpred=[]
        for s in range(3):
            p=np.empty((n,144));covered=np.zeros(n,int)
            for f in range(5):
                z=np.load(private/f'{arm}_f{f}_s{s}.npz');te=z['te'];covered[te]+=1;p[te]=z['tp']
                outer=set(d['fips'][te]);inner=set(d['fips'][z['va']]);assert not outer&inner
                assert np.all(d['outer'][te]==f);checks+=3
            assert (covered==1).all() and np.isfinite(p).all();checks+=2;allpred.append(p)
            row=per[(per.model==arm)&(per.group=='all')&(per.seed==s)].iloc[0]
            wm=d['m']*d['w'].astype(np.float64)[:,None];diff=p-d['y']
            assert abs(np.sqrt(np.sum(wm*diff**2)/wm.sum())-row.rmse)<1e-12
            assert abs(np.sum(wm*abs(diff))/wm.sum()-row.mae)<1e-12;checks+=2
        ensemble[arm]=np.stack(allpred).mean(0)
    for _,row in metrics[metrics.group=='all'].iterrows():
        cols=np.arange(144) if row.scope=='full144' else np.array([int(row.scope.split('+')[1])-1])
        p=ensemble[row.model][:,cols];truth=d['y'][:,cols];mask=d['m'][:,cols]
        w=d['w'].astype(np.float64) if row.weighting=='design' else np.ones(n)
        # Flatten supported cells; compute via dot products rather than summarize().
        weights=np.broadcast_to(w[:,None],mask.shape)[mask];diff=(p-truth)[mask]
        rmse=float(np.sqrt(np.dot(weights,diff*diff)/weights.sum()));mae=float(np.dot(weights,abs(diff))/weights.sum())
        e=max(abs(rmse-row.rmse),abs(mae-row.mae));assert e<1e-12;error=max(error,e);checks+=2
    comp=pd.read_csv(out/'comparisons.csv')
    headline=metrics.query("group=='all' and weighting=='design' and scope=='full144'").set_index('model')
    for _,r in comp[comp.group=='all'].iterrows():
        expected=100*(1-headline.loc[r.model,'rmse']/headline.loc[r.reference,'rmse'])
        assert abs(expected-r.rmse_gain_pct)<1e-10;checks+=1
    # Independently identify repeated (FIPS, absolute UTC-hour) cells using records.
    if 'starts' in d:
        cells={}
        for i in range(n):
            for t in np.flatnonzero(d['m'][i]):
                key=(str(d['fips'][i]),int(d['starts'][i])+72+int(t));cells.setdefault(key,[]).append((i,int(t)))
        total=sum(map(len,cells.values()));duplicate=total-len(cells)
        recorded=json.loads((out/'verification.json').read_text())
        assert duplicate==recorded['duplicate_extra_county_hours'] and len(cells)==recorded['unique_county_hours'];checks+=2
        sens=pd.read_csv(out/'unique_county_hour_sensitivity.csv')
        # Each unique cell gives equal mass, while its repeated forecast errors share that mass.
        for arm in arms:
            se=ae=0.
            for rows in cells.values():
                err=np.array([ensemble[arm][i,t]-d['y'][i,t] for i,t in rows]);se+=np.mean(err**2);ae+=np.mean(abs(err))
            row=sens[sens.model==arm]
            if 'weighting' in sens:row=row[row.weighting=='unique_equal']
            row=row.iloc[0]
            assert abs(np.sqrt(se/len(cells))-row.rmse)<1e-10 and abs(ae/len(cells)-row.mae)<1e-10;checks+=2
    receipt=dict(checks=checks,headline_arithmetic_max_error=error,all_seeds_retained=True,
                 duplicate_check='independent tuple-key grouping',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'independent_audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(label,receipt,flush=True)

if __name__=='__main__':
    import sys
    for label in sys.argv[1:] or ['t05','t06']:audit(label)
