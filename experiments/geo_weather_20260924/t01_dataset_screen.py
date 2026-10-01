"""Finite T01 development probe; see notes/T01_WITHIN_EVENT_DATASET_SCREEN_20261001.md."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import hashlib, json, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data_v1'
OUT = HERE / 'results/t01'
SEED = 20261001
ALPHAS = (0.1, 10., 1000.)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def fit_predict(x,y,w,tr,te,alpha):
    model = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha,solver='svd'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        model.fit(x[tr],y[tr],ridge__sample_weight=w[tr]/w[tr].mean())
    return model.predict(x[te]).clip(0,1)

def score(y,p,w):
    return float(np.sqrt(np.average((y-p)**2,weights=w))),float(np.average(abs(y-p),weights=w))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'audit.json').exists():
        raise SystemExit('Refusing to overwrite completed T01')
    paths=[DATA/'outcomes/panel_v1D_outcomes.npz',DATA/'outcomes/development_systems.csv',DATA/'geography/county_geography_v1.parquet',DATA/'geography/county_geography_ext_v1.parquet']
    z=np.load(paths[0],allow_pickle=False)
    systems=pd.read_csv(paths[1]);systems=systems[systems.regime.eq('tropical')].sort_values('system')
    geo=pd.read_parquet(paths[2]).drop(columns=['n_land_pixels'])
    ext=pd.read_parquet(paths[3])
    ext=ext[['forest_near_developed','soil_wet_share','soil_windthrow_hazard','forest_wet_coloc','wet_in_forest','hazard_in_forest','elev_mean5','relief5','fia_forest_land_share']]
    geo=geo.join(ext);geo.index=geo.index.astype(str).str.zfill(5)
    assert geo.shape[1]==40 and geo.index.is_unique
    roster=[];metrics=[];splits={};choices=[];fits=0;coverage_checks=0
    for s in systems.itertuples():
        ids=np.flatnonzero(z['system']==s.system)
        fips=z['fips'][ids];obs=z['observed'][ids,72:];y=z['y'][ids,72:].astype(float)
        assert len(set(fips))==len(fips)
        valid=obs.any(axis=1)
        peak=np.max(np.where(obs,y,-np.inf),axis=1)
        gp=geo.reindex(fips).to_numpy(float)
        ep=DATA/f'weather/hazard_v2/era5/era5_{s.system}.npz'
        hp=DATA/f'weather/hazard_v2/hrrr/hrrr_{s.system}.npz'
        row=dict(system=s.system,storm=s.storm,used=bool(s.used),window_start=s.window_start,n_counties=len(ids),observed_fraction=float(obs.mean()),partial_counties=int((obs.sum(1)<144).sum()),valid_counties=int(valid.sum()),severe_counties=int(((peak>=.1)&valid).sum()),geo_finite_fraction=float(np.isfinite(gp).mean()),era5_present=ep.exists(),hrrr_present=hp.exists(),status='not_run')
        if not ep.exists() or valid.sum()<25:
            row['status']='insufficient_counties_or_weather';roster.append(row);continue
        paths.append(ep);e=np.load(ep,allow_pickle=False)
        lookup={str(f).zfill(5):i for i,f in enumerate(e['fips'])}
        if not set(fips)<=set(lookup):
            row['status']='weather_county_mismatch';roster.append(row);continue
        wx=e['X'][[lookup[f] for f in fips]].astype(float)
        take=np.array(['*' not in str(n) for n in e['names']])
        wx=wx[:,:,take];wx[~np.isfinite(wx)]=np.nan
        row['era5_missing_hours']=int(e['missing_hours'].size)
        row['weather_finite_fraction']=float(np.isfinite(wx).mean())
        hist=np.where(z['observed'][ids,:72],z['y'][ids,:72].astype(float),np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            b=np.column_stack([np.nanmean(wx[:,:72],axis=1),np.nanmax(wx[:,:72],axis=1),np.nanmean(wx[:,72:],axis=1),np.nanmax(wx[:,72:],axis=1),hist[:,-1],np.nanmean(hist,axis=1),np.nanmax(hist,axis=1)])
        b,gp,fips,peak,ids=[a[valid] for a in (b,gp,fips,peak,ids)]
        w=z['w'][ids].astype(float);assert np.all(np.isfinite(w)&(w>0))
        n=len(ids);outer=list(KFold(5,shuffle=True,random_state=SEED).split(b))
        folds=np.full(n,-1);counts=np.zeros(n,int)
        for fold,(tr,te) in enumerate(outer):
            assert not set(fips[tr])&set(fips[te]);folds[te]=fold;counts[te]+=1
        assert np.all(counts==1);coverage_checks+=n
        splits[s.system]=dict(zip(fips.tolist(),folds.tolist()))
        perm=np.random.default_rng(SEED).permutation(n)
        arms={'B':b,'G':np.column_stack([b,gp]),'P':np.column_stack([b,gp[perm]])}
        predictions={}
        for arm,x in arms.items():
            pred=np.full(n,np.nan)
            for fold,(tr,te) in enumerate(outer):
                inner=list(KFold(3,shuffle=True,random_state=SEED+fold).split(tr))
                losses=[]
                for alpha in ALPHAS:
                    ip=np.full(len(tr),np.nan)
                    for it,iv in inner:
                        ip[iv]=fit_predict(x,peak,w,tr[it],tr[iv],alpha);fits+=1
                    assert np.isfinite(ip).all()
                    losses.append(float(np.average((ip-peak[tr])**2,weights=w[tr])))
                alpha=ALPHAS[int(np.argmin(losses))]
                pred[te]=fit_predict(x,peak,w,tr,te,alpha);fits+=1
                choices.append(dict(system=s.system,arm=arm,fold=fold,alpha=alpha,inner_mse=losses))
            assert np.isfinite(pred).all();predictions[arm]=pred
            rmse,mae=score(peak,pred,w)
            metrics.append(dict(system=s.system,storm=s.storm,arm=arm,n=n,rmse=rmse,mae=mae))
        br,bm=score(peak,predictions['B'],w);gr,gm=score(peak,predictions['G'],w);pr,pm=score(peak,predictions['P'],w)
        row.update(status='complete',geo_gain_vs_B_pct=100*(1-gr/br) if br else None,geo_gain_vs_P_pct=100*(1-gr/pr) if pr else None,geo_fold_wins_vs_B=sum(score(peak[te],predictions['G'][te],w[te])[0]<score(peak[te],predictions['B'][te],w[te])[0] for _,te in outer))
        roster.append(row)
        print(s.system,s.storm,n,round(row['geo_gain_vs_B_pct'],3),flush=True)
    pd.DataFrame(roster).to_csv(OUT/'candidate_audit.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'peak_probe_metrics.csv',index=False)
    for name,obj in [('county_splits.json',splits),('inner_choices.json',choices)]:
        (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    audit=dict(seed=SEED,alphas=ALPHAS,source_sha256=digest(Path(__file__)),inputs={str(p.relative_to(HERE)):digest(p) for p in paths},geography_columns=geo.columns.tolist(),n_systems=len(roster),n_complete=sum(r['status']=='complete' for r in roster),fits=fits,county_oof_coverage_verified=coverage_checks,scope='Exploratory conditional hindcast, within-event county-held-out peak ridge, not neural trajectory evaluation',versions={'numpy':np.__version__,'pandas':pd.__version__})
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
