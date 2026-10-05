"""Weather-only repeated-impact screening across the existing 81 D windows."""
from t03_poster_trajectory import *
from scipy.signal import find_peaks
OUT=HERE/'results/t05'
PRIVATE=ROOT.parent/'t05_artifacts'

def tailmean(x,k=3):
    out=np.empty_like(x)
    for t in range(x.shape[1]):out[:,t]=np.nanmean(x[:,max(0,t-k+1):t+1],axis=1)
    return out

def screen():
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    p=DATA/'outcomes/panel_v1D_outcomes.npz';z=np.load(p,allow_pickle=False)
    rows=[];details=[];inputs={str(p.relative_to(HERE)):digest(p)}
    for event in sorted(set(z['system'])):
        ix=np.flatnonzero(z['system']==event);start=pd.Timestamp(str(z['window_start_utc'][ix[0]]))
        weather={};availability={}
        for src in ('era5','hrrr'):
            path=DATA/f'weather/hazard_v2/{src}/{src}_{event}.npz';inputs[str(path.relative_to(HERE))]=digest(path)
            e=np.load(path,allow_pickle=False);lookup={str(f):j for j,f in enumerate(e['fips'])}
            x=e['X'][[lookup[str(f)] for f in z['fips'][ix]]].astype(float)
            for stamp in e['missing_hours']:
                h=int((pd.Timestamp(str(stamp))-start).total_seconds()/3600)
                if 0<=h<216:x[:,h]=np.nan
            chosen=[j for j,n in enumerate(e['names']) if '*' not in str(n)]
            availability[src]=float(np.isfinite(x[:,:,chosen]).all(2).mean())
            weather[src]=(x,list(e['names']))
        x,names=weather['era5']
        hs=[tailmean(x[:,:,names.index(c)])/v for c,v in [('g_mean',18),('rain_mean',2),('snow_mean',1),('frz_mean',.5)]]
        intensity=np.nanmax(np.stack(hs),axis=0)
        count=0;cross=0
        for j,row in enumerate(intensity):
            peaks,_=find_peaks(np.nan_to_num(row),height=1,prominence=.25,distance=3)
            peaks=peaks[(peaks>=3)&(peaks<=212)];pairs=[]
            for a in peaks:
                for b in peaks:
                    if b<72 or not 24<=b-a<=120:continue
                    low=min(row[a],row[b]);trough=float(np.nanmin(row[a:b+1]))
                    if trough<=.75*low:pairs.append((float(low),int(a),int(b),trough))
            chosen=max(pairs) if pairs else None
            if chosen:count+=1;cross+=chosen[1]<72
            details.append(dict(index=int(ix[j]),system=event,fips=str(z['fips'][ix[j]]),repeated=bool(chosen),
                cross_cutoff=bool(chosen and chosen[1]<72),peak1=chosen[1] if chosen else -1,peak2=chosen[2] if chosen else -1))
        reason=[]
        if len(ix)<50:reason.append('counties<50')
        if z['observed'][ix].mean()<.99:reason.append('outage_coverage<99%')
        if not z['observed'][ix,71].all():reason.append('missing_hour71')
        for src in availability:
            if availability[src]<.95:reason.append(src+'_coverage<95%')
        if count<20 or count/len(ix)<.2:reason.append('insufficient_repeated_exposure')
        states=','.join(sorted(set(f[:2] for f in z['fips'][ix])))
        rows.append(dict(system=event,regime=str(z['regime'][ix[0]]),start=str(start),state_fips=states,
            n=len(ix),outage_coverage=float(z['observed'][ix].mean()),era5_available=availability['era5'],hrrr_available=availability['hrrr'],
            repeated_count=count,repeated_share=count/len(ix),cross_cutoff_count=cross,cross_cutoff_share=cross/len(ix),
            eligible=not reason,reason=';'.join(reason)))
    frame=pd.DataFrame(rows).sort_values(['repeated_share','n','system'],ascending=[False,False,True])
    selected=frame[frame.eligible].head(8).system.tolist();frame['selected']=frame.system.isin(selected)
    frame.to_csv(OUT/'weather_window_screen.csv',index=False)
    dump(PRIVATE/'county_weather_groups.json',details)
    dump(OUT/'selection.json',dict(events=selected,selection_rule='fixed weather/coverage thresholds; top8 repeated share; no outage magnitudes or model scores',
        inspected_windows=len(frame),eligible_windows=int(frame.eligible.sum()),source_sha256=digest(__file__),inputs=inputs))
    print(frame.head(25).to_string(index=False));print('SELECTED',selected,flush=True)
    if not selected:raise RuntimeError('No eligible windows; retain empty screen without relaxing rule')

if __name__=='__main__':screen()
