"""T02 finite split-stability screen; see notes/T02_SPLIT_STABILITY_PROTOCOL_20261002.md."""
from t01_dataset_screen import HERE, DATA, ALPHAS, SEED, digest, fit_predict, score
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold

OUT = HERE / 'results/t02'
SYSTEMS = ('S00023', 'S00034', 'S00037', 'S00040', 'S00046', 'S00047')
CONFIGS = [('random', s) for s in range(SEED, SEED+5)] + [('spatial', a) for a in (0,45,90,135)]


def spatial_labels(xy, k, angle):
    """Balanced recursive coordinate partition; no target/feature-dependent choices."""
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    q = xy @ rotation
    labels = np.full(len(q), -1)
    def split(ids, groups, offset):
        if groups == 1:
            labels[ids] = offset
            return
        axis = int(np.argmax(np.ptp(q[ids], axis=0)))
        order = ids[np.argsort(q[ids, axis], kind='stable')]
        left = groups // 2
        cut = int(round(len(ids)*left/groups))
        assert left <= cut <= len(ids)-(groups-left)
        split(order[:cut], left, offset)
        split(order[cut:], groups-left, offset+left)
    split(np.arange(len(q)), k, 0)
    assert set(labels) == set(range(k))
    return labels


def split_pairs(xy, method, value, k):
    if method == 'random':
        return list(KFold(k, shuffle=True, random_state=value).split(xy))
    labels = spatial_labels(xy, k, value)
    return [(np.flatnonzero(labels != f), np.flatnonzero(labels == f)) for f in range(k)]


def main():
    if (OUT/'audit.json').exists():
        raise SystemExit('Refusing to overwrite completed T02')
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((HERE/'results/t01/audit.json').read_text())
    # Refuse silent changes to any T01 input, including negative/non-selected candidates.
    for name, sha in old['inputs'].items():
        assert digest(HERE/name) == sha, name
    z = np.load(DATA/'outcomes/panel_v1D_outcomes.npz', allow_pickle=False)
    systems = pd.read_csv(DATA/'outcomes/development_systems.csv').set_index('system')
    geo = pd.read_parquet(DATA/'geography/county_geography_v1.parquet').drop(columns=['n_land_pixels'])
    ext = pd.read_parquet(DATA/'geography/county_geography_ext_v1.parquet')
    geo = geo.join(ext[[c for c in old['geography_columns'] if c not in geo]])
    geo = geo[old['geography_columns']]
    geo.index = geo.index.astype(str).str.zfill(5)
    assert geo.index.is_unique
    coord_path = DATA/'geography/county_statics_v1.parquet'
    coords = pd.read_parquet(coord_path)[['fips','lat','lon']]
    coords['fips'] = coords.fips.astype(str).str.zfill(5)
    coords = coords.set_index('fips'); assert coords.index.is_unique
    t01 = pd.read_csv(HERE/'results/t01/peak_probe_metrics.csv')
    metrics, comparisons, folds, choices, memberships, observation_audit = [], [], [], [], {}, []
    fit_count = 0; replay_count = 0; replay_max = 0.; oof_count = 0
    for system in SYSTEMS:
        meta = systems.loc[system]
        assert meta.regime == 'tropical'  # D includes previously used events; none is unseen confirmation.
        ids = np.flatnonzero(z['system'] == system)
        fips = z['fips'][ids]; assert len(set(fips)) == len(fips)
        observed = z['observed'][ids,72:]
        peak = np.max(np.where(observed, z['y'][ids,72:].astype(float), -np.inf), axis=1)
        valid = observed.any(1)
        complete = observed.all(1)
        e = np.load(DATA/f'weather/hazard_v2/era5/era5_{system}.npz', allow_pickle=False)
        lookup = {str(f).zfill(5):i for i,f in enumerate(e['fips'])}
        wx = e['X'][[lookup[f] for f in fips]].astype(float)
        wx = wx[:,:,np.array(['*' not in str(n) for n in e['names']])]
        assert wx.shape[1] == 216 and np.isfinite(wx).all()
        hist = np.where(z['observed'][ids,:72], z['y'][ids,:72].astype(float), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            b = np.column_stack([np.nanmean(wx[:,:72],axis=1), np.nanmax(wx[:,:72],axis=1),
                np.nanmean(wx[:,72:],axis=1), np.nanmax(wx[:,72:],axis=1), hist[:,-1], np.nanmean(hist,axis=1), np.nanmax(hist,axis=1)])
        gp = geo.reindex(fips).to_numpy(float)
        ll = coords.reindex(fips).to_numpy(float); assert np.isfinite(ll).all()
        xy = np.column_stack([(ll[:,1]-ll[:,1].mean())*111.32*np.cos(np.deg2rad(ll[:,0].mean())), (ll[:,0]-ll[:,0].mean())*111.32])
        observation_audit.append(dict(system=system,storm=meta.storm,previously_used=bool(meta['used']),n=len(ids),complete_n=int(complete.sum()),
            observed_fraction=float(observed.mean()),earliest_window=str(meta.window_start),
            partial_observed_peak_max=float(peak[valid&~complete].max()) if (valid&~complete).any() else None))
        samples = [('primary', valid)] + ([('complete_only', complete)] if not complete.all() else [])
        for sample, mask in samples:
            bb,gg,ff,yy,ww,xx = b[mask],gp[mask],fips[mask],peak[mask],z['w'][ids[mask]].astype(float),xy[mask]
            assert np.isfinite(yy).all() and np.all(np.isfinite(ww)&(ww>0))
            n = len(yy)
            arms = {'B':bb, 'G':np.column_stack([bb,gg])}
            for j in range(3):
                perm = np.random.default_rng(SEED+j*10000).permutation(n)
                arms[f'P{j}'] = np.column_stack([bb,gg[perm]])
            for method, value in CONFIGS:
                key = f'{system}/{sample}/{method}/{value}'
                outer = split_pairs(xx,method,value,5)
                counts = np.zeros(n,int); labels = np.full(n,-1)
                for fold,(tr,te) in enumerate(outer):
                    assert not set(ff[tr]) & set(ff[te])
                    counts[te] += 1; labels[te] = fold
                assert np.all(counts==1); oof_count += n
                memberships[key] = dict(zip(ff.tolist(), labels.tolist()))
                preds = {}; scores = {}
                base = dict(system=system,storm=meta.storm,sample=sample,method=method,split=value,n=n)
                for arm,x in arms.items():
                    pred = np.full(n,np.nan)
                    for fold,(tr,te) in enumerate(outer):
                        inner = split_pairs(xx[tr],method,value+fold if method=='random' else value,3)
                        losses = []
                        for alpha in ALPHAS:
                            ip = np.full(len(tr),np.nan)
                            for it,iv in inner:
                                assert not set(ff[tr[it]]) & set(ff[tr[iv]])
                                ip[iv] = fit_predict(x,yy,ww,tr[it],tr[iv],alpha); fit_count += 1
                            assert np.isfinite(ip).all()
                            losses.append(float(np.average((ip-yy[tr])**2,weights=ww[tr])))
                        alpha = ALPHAS[int(np.argmin(losses))]
                        pred[te] = fit_predict(x,yy,ww,tr,te,alpha); fit_count += 1
                        choices.append(dict(key=key,arm=arm,fold=fold,alpha=alpha,inner_mse=losses))
                    assert np.isfinite(pred).all()
                    preds[arm] = pred
                    rmse,mae = score(yy,pred,ww); scores[arm] = rmse
                    metrics.append(dict(**base,arm=arm,rmse=rmse,mae=mae))
                    if sample=='primary' and method=='random' and value==SEED and arm in ('B','G','P0'):
                        old_arm = 'P' if arm=='P0' else arm
                        target = t01[(t01.system==system)&(t01.arm==old_arm)].iloc[0]
                        error = max(abs(rmse-target.rmse), abs(mae-target.mae))
                        assert error < 1e-10, (system,arm,error)
                        replay_max = max(replay_max,error); replay_count += 1
                delta = ww*((yy-preds['B'])**2-(yy-preds['G'])**2)
                positive = np.maximum(delta,0)
                fold_deltas = []
                wins = 0
                for fold,(tr,te) in enumerate(outer):
                    br,bm = score(yy[te],preds['B'][te],ww[te]); gr,gm = score(yy[te],preds['G'][te],ww[te])
                    distances = np.sqrt(((xx[te,None,:]-xx[None,tr,:])**2).sum(2)).min(1)
                    fold_deltas.append(float(delta[te].sum())); wins += gr < br
                    folds.append(dict(**base,fold=fold,n_test=len(te),b_rmse=br,g_rmse=gr,b_mae=bm,g_mae=gm,
                        sse_improvement=float(delta[te].sum()),test_weight=float(ww[te].sum()),nearest_train_km_median=float(np.median(distances))))
                pmedian = float(np.median([scores[f'P{j}'] for j in range(3)]))
                total = float(delta.sum())
                comparisons.append(dict(**base,b_rmse=scores['B'],g_rmse=scores['G'],p_median_rmse=pmedian,
                    gain_vs_b_pct=100*(1-scores['G']/scores['B']),gain_vs_p_pct=100*(1-scores['G']/pmedian),fold_wins=int(wins),
                    top5_positive_sse_share=float(np.sort(positive)[-5:].sum()/positive.sum()) if positive.sum() else 0.,
                    best_fold_share_net_gain=float(max(fold_deltas)/total) if total>0 else None,
                    rmse_gain_without_best_fold_pct=100*(1-score(yy[labels!=np.argmax(fold_deltas)],preds['G'][labels!=np.argmax(fold_deltas)],ww[labels!=np.argmax(fold_deltas)])[0]/score(yy[labels!=np.argmax(fold_deltas)],preds['B'][labels!=np.argmax(fold_deltas)],ww[labels!=np.argmax(fold_deltas)])[0])))
                print(key, 'gain', round(comparisons[-1]['gain_vs_b_pct'],3),flush=True)
    assert replay_count == 18
    for name,rows in [('metrics',metrics),('comparisons',comparisons),('fold_diagnostics',folds),('observation_audit',observation_audit)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    for name,obj in [('county_splits',memberships),('inner_choices',choices)]:
        (OUT/f'{name}.json').write_text(json.dumps(obj,separators=(',',':'),allow_nan=False)+'\n')
    audit = dict(source_sha256=digest(Path(__file__)),t01_source_sha256=digest(HERE/'t01_dataset_screen.py'),
        t01_input_hashes_verified=old['inputs'],coordinates_sha256=digest(coord_path),
        protocol_sha256=digest(HERE/'notes/T02_SPLIT_STABILITY_PROTOCOL_20261002.md'),
        fits=fit_count,configurations=len(comparisons),oof_coverage_count=oof_count,replayed_t01_metrics=replay_count,
        replay_max_absolute_error=replay_max,systems=SYSTEMS,configs=CONFIGS,alphas=ALPHAS,
        versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        scope='Development-only conditional hindcast peak ridge; no neural kernel training; no prediction uploads')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
