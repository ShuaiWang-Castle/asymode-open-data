"""Prepare a development-only tropical view without changing the parent design.

Reads committed data_v1 products, not raw archives or confirmation outcomes.
The branch's existing event folds are inherited exactly; all inner folds keep
whole event groups together. No model is fitted and no outcome score is computed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
ROOT=EXP.parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source=EXP/'data_v1/outcomes/panel_v1D_outcomes.npz'
    z=np.load(source,allow_pickle=False)
    take=np.flatnonzero(z['regime']=='tropical')
    a={k:z[k][take] for k in z.files}
    n=len(take);assert n==1633
    assert len(set(zip(a['system'],a['fips'])))==n
    assert a['y'].shape==(n,216)
    assert np.isfinite(a['y'][a['observed']]).all()
    tranches=pd.read_parquet(EXP/'data_provenance/frame_v1/tranches.parquet')
    development=set(tranches.loc[tranches.tranche=='D','system'])
    assert set(a['system'])<=development
    assert len(set(a['system']))==15
    a['source_row']=take
    a['history']=a['y'][:,:72].astype('float32')
    a['history_observed']=a['observed'][:,:72]
    a['target']=a['y'][:,72:].astype('float32')
    a['target_observed']=a['observed'][:,72:]
    spfile=EXP/'splits_v1D.json'
    sp=json.loads(spfile.read_text())
    assert sp['n_units']==len(z['system'])
    fold=np.zeros(n,dtype=int)
    for k in range(1,6):
        held=np.isin(take,sp['event'][str(k)]['outer'])
        assert (fold[held]==0).all();fold[held]=k
    assert (fold>0).all();a['outer_fold']=fold
    folds={};seen=np.zeros(n,int)
    for k in range(1,6):
        dev=np.flatnonzero(fold!=k);held=np.flatnonzero(fold==k)
        assert set(a['family'][dev]).isdisjoint(a['family'][held])
        assert np.array_equal(take[dev],np.intersect1d(take,sp['event'][str(k)]['dev']))
        inner=[]
        for j in range(1,6):
            if j==k:continue
            fit=np.flatnonzero((fold!=k)&(fold!=j));val=np.flatnonzero(fold==j)
            assert set(a['family'][fit]).isdisjoint(a['family'][val])
            inner.append(dict(validation_source_fold=j,train=fit.tolist(),validation=val.tolist()))
        folds[str(k)]=dict(dev=dev.tolist(),outer=held.tolist(),inner=inner)
        seen[held]+=1
    assert (seen==1).all()
    weather={};checksums={str(source.relative_to(ROOT)):sha(source),str(spfile.relative_to(ROOT)):sha(spfile)}
    for src in ['era5','hrrr']:
        blocks=[];names=None;missing={};weather_observed=np.ones((n,216),bool)
        for system in sorted(set(a['system'])):
            path=EXP/f'data_v1/weather/hazard_v2/{src}/{src}_{system}.npz'
            w=np.load(path,allow_pickle=False)
            idx=np.flatnonzero(a['system']==system)
            pos={str(f):i for i,f in enumerate(w['fips'])}
            order=[pos[str(f)] for f in a['fips'][idx]]
            x=w['X'][order].astype('float32')
            assert x.shape[:2]==(len(idx),216) and np.isfinite(x).all()
            if names is None:names=w['names']
            else:assert np.array_equal(names,w['names'])
            blocks.append((idx,x));missing[system]=int(len(w['missing_hours']))
            if len(w['missing_hours']):
                starts=np.unique(a['window_start_utc'][idx]);assert len(starts)==1
                offsets=(pd.to_datetime(w['missing_hours'])-pd.Timestamp(str(starts[0])))/pd.Timedelta(hours=1)
                assert np.equal(offsets,np.round(offsets)).all()
                hours=np.asarray(offsets,dtype=int)
                assert ((hours>=0)&(hours<216)).all()
                weather_observed[np.ix_(idx,hours)]=False
            checksums[str(path.relative_to(ROOT))]=sha(path)
        wx=np.empty((n,216,len(names)),dtype='float32')
        for idx,x in blocks:wx[idx]=x
        a[src+'_weather']=wx;a[src+'_names']=names
        a[src+'_weather_observed']=weather_observed
        weather[src]=dict(features=len(names),missing_hours_by_system=missing)
    geo_parts=[];geo_names=[]
    for filename,prefix,allowed in [
        ('county_geography_v1','geo',None),
        ('county_geography_ext_v1','geo_ext',None),
        ('county_axes','axis',None),
        ('county_statics_v1','context',['log_cust','rucc','log_pop_density','coop_share','n_utilities','saidi']),
    ]:
        path=EXP/f'data_v1/geography/{filename}.parquet'
        df=pd.read_parquet(path)
        if 'fips' in df:df=df.set_index('fips')
        df.index=df.index.astype(str).str.zfill(5)
        cols=list(df.select_dtypes(include='number').columns) if allowed is None else allowed
        geo_parts.append(df.reindex(a['fips'])[cols].to_numpy(dtype='float32'))
        geo_names += [prefix+'__'+c for c in cols]
        checksums[str(path.relative_to(ROOT))]=sha(path)
    a['geography']=np.concatenate(geo_parts,axis=1);a['geography_names']=np.array(geo_names)
    # Keep missing geography as NaN; downstream imputers must fit on training only.
    assert not any('2023' in x for x in geo_names)
    out=ROOT/'data/interim/tropical_prediction_v1';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'development.npz',**a)
    (HERE/'splits.json').write_text(json.dumps(folds,indent=2)+'\n')
    audit=dict(source_branch_commit='493aa54fb9233f27c4b6a10b5bacf3c1592f33ba',
        tranche='D',systems=15,county_events=n,history_hours=72,forecast_hours=144,
        sealed_outcomes_read=False,source_rows_preserved=True,all_outer_rows_scored_once=True,
        weather=weather,geography_columns=geo_names,
        geography_missing_values=int(np.isnan(a['geography']).sum()),
        origin_history_last_hour_observed=int(a['history_observed'][:,-1].sum()),
        target_observed_cells=int(a['target_observed'].sum()),
        folds={str(k):dict(dev=int((fold!=k).sum()),outer=int((fold==k).sum()),
                          systems=sorted(set(a['system'][fold==k]))) for k in range(1,6)},
        source_sha256=checksums,prepared_data_sha256=sha(out/'development.npz'))
    (HERE/'preparation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['systems','county_events','history_hours','forecast_hours','weather','geography_missing_values','origin_history_last_hour_observed','target_observed_cells','folds']},indent=2))

if __name__=='__main__':main()
