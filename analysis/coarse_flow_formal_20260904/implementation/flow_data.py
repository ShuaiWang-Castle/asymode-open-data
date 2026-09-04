from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

CAP_U = 0.265
CAP_R = 0.25
TIE_CHANNELS = ("gust", "wind_speed", "precip", "snowfall", "cape")
FEATURE_STATICS = ("log_area", "n_neighbours", "lat", "lon")


def sha_u64(*parts: object, salt: str = "") -> int:
    s = salt + "|" + "|".join(str(x) for x in parts)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def county_split(fips: np.ndarray, seed: int = 20260904, train_frac: float = 0.8) -> np.ndarray:
    # True means source/train. Stable approximately 80/20 threshold on uint64.
    threshold = int(train_frac * (2**64 - 1))
    return np.array([sha_u64(seed, str(f).zfill(5), salt="county-split") <= threshold for f in fips], dtype=bool)


def to_hourly(y15: np.ndarray, obs15: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    C, T = y15.shape
    n = T // 4
    y = np.asarray(y15[:, : n * 4], float).reshape(C, n, 4)
    o = np.asarray(obs15[:, : n * 4], bool).reshape(C, n, 4)
    cnt = o.sum(axis=2)
    vals = np.where(o, np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), 0.0).sum(axis=2)
    yh = np.where(cnt > 0, vals / np.maximum(cnt, 1), np.nan)
    return yh, cnt > 0


@dataclass
class EventData:
    event: str
    family: str
    fips: np.ndarray
    y: np.ndarray
    obs: np.ndarray
    X: np.ndarray
    channels: List[str]
    ts: pd.DatetimeIndex
    denominator: np.ndarray
    peak: int
    peak_footprint: float
    active_available: bool
    active_start: int | None
    active_end: int | None
    features: np.ndarray  # C,H,24
    feature_names: List[str]


def _trailing_max(X: np.ndarray, j: int, window: int) -> np.ndarray:
    C, H, _ = X.shape
    out = np.empty((C, H), dtype=np.float32)
    for t in range(H):
        out[:, t] = np.nanmax(X[:, max(0, t-window+1):t+1, j], axis=1)
    return out


def _trailing_sum(X: np.ndarray, j: int, window: int) -> np.ndarray:
    C, H, _ = X.shape
    out = np.empty((C, H), dtype=np.float32)
    for t in range(H):
        out[:, t] = np.nansum(X[:, max(0, t-window+1):t+1, j], axis=1)
    return out


def noaa_peak(events_df: pd.DataFrame, fips: np.ndarray, ts: pd.DatetimeIndex, X: np.ndarray, channels: List[str]) -> Tuple[int, float]:
    n_hours = min(len(ts), X.shape[1])
    fips_set = set(str(x).zfill(5) for x in fips)
    sub = events_df[events_df["fips"].isin(fips_set)]
    active = np.zeros((len(fips), n_hours), dtype=bool)
    code = {str(f).zfill(5): i for i, f in enumerate(fips)}
    hv = ts[:n_hours].values.astype("datetime64[ns]")
    for row in sub.itertuples(index=False):
        ci = code.get(str(row.fips).zfill(5))
        if ci is None:
            continue
        b = np.datetime64(row.t_begin_utc)
        e = np.datetime64(row.t_end_utc)
        if e < hv[0] or b > hv[-1]:
            continue
        lo = int(np.searchsorted(hv, b, side="left"))
        hi = int(np.searchsorted(hv, e, side="right"))
        if hi > lo:
            active[ci, lo:hi] = True
    fp = active.mean(axis=0)
    best = fp.max()
    tied = np.flatnonzero(fp == best)
    if len(tied) > 1:
        idx = [channels.index(c) for c in TIE_CHANNELS if c in channels]
        if idx:
            Z = X[:, :n_hours, idx].astype(float)
            mu = Z.mean(axis=(0,1), keepdims=True)
            sd = Z.std(axis=(0,1), keepdims=True)
            sd = np.where(sd > 0, sd, 1.0)
            comp = np.maximum((Z-mu)/sd, 0.0).mean(axis=(0,2))
            mx = comp[tied].max()
            tied = tied[comp[tied] == mx]
    return int(tied[0]), float(best)


def build_features(X: np.ndarray, channels: List[str], ts: pd.DatetimeIndex, peak: int,
                   fips: np.ndarray, statics: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    C, H, D = X.shape
    feats = [X.astype(np.float32)]
    names = list(channels)
    for c in ("gust", "wind_speed", "cape"):
        j = channels.index(c)
        feats.append(_trailing_max(X, j, 6)[:, :, None])
        names.append(f"{c}_max6")
    for c in ("precip", "snowfall"):
        j = channels.index(c)
        feats.append(_trailing_sum(X, j, 12)[:, :, None])
        names.append(f"{c}_sum12")
    phase = ((np.arange(H, dtype=np.float32)-float(peak))/24.0)[None,:,None]
    feats.append(np.repeat(phase, C, axis=0)); names.append("phase_peak_24")
    hour = ts[:H].hour.to_numpy(float)
    clock = np.stack([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)], axis=-1).astype(np.float32)
    feats.append(np.repeat(clock[None,:,:], C, axis=0)); names.extend(["clock_sin","clock_cos"])
    smap = statics.copy()
    smap["fips"] = smap["fips"].astype(str).str.zfill(5)
    smap = smap.set_index("fips")
    S = np.zeros((C, len(FEATURE_STATICS)), dtype=np.float32)
    for i,f in enumerate(fips):
        key=str(f).zfill(5)
        if key in smap.index:
            vals=smap.loc[key,list(FEATURE_STATICS)]
            if isinstance(vals,pd.DataFrame): vals=vals.iloc[0]
            S[i]=np.asarray(vals,dtype=float)
        else:
            S[i]=np.nan
    # fill missing per event by median
    med=np.nanmedian(S,axis=0)
    inds=np.where(~np.isfinite(S)); S[inds]=med[inds[1]]
    feats.append(np.repeat(S[:,None,:],H,axis=1)); names.extend(FEATURE_STATICS)
    F=np.concatenate(feats,axis=2)
    assert F.shape[2]==24,(F.shape,names)
    return F,names


def load_all(root: Path) -> Tuple[Dict[str,EventData], dict]:
    manifest=json.loads((root/"configs/panel_manifest_g3-all-26.json").read_text())
    if manifest["digest"]!="db286b4960a4": raise ValueError("manifest mismatch")
    interim=root/"data/interim"
    def read_table(name: str, dtype=None):
        csv_path = interim / f"{name}.csv"
        parquet_path = interim / f"{name}.parquet"
        if csv_path.exists():
            return pd.read_csv(csv_path, dtype=dtype)
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        raise FileNotFoundError(f"missing both {csv_path} and {parquet_path}")

    events_df=read_table("storm_events_county",dtype={"fips":str})
    events_df["fips"]=events_df["fips"].astype(str).str.zfill(5)
    events_df["t_begin_utc"]=pd.to_datetime(events_df["t_begin_utc"])
    events_df["t_end_utc"]=pd.to_datetime(events_df["t_end_utc"])
    fam=read_table("event_days_stratified")
    fam["day"]=pd.to_datetime(fam["day"]).dt.strftime("%Y-%m-%d")
    family=dict(zip(fam["day"],fam["dominant"]))
    statics=read_table("county_statics",dtype={"fips":str})
    out={}
    for event in manifest["panels"]:
        p=np.load(interim/f"panel_{event}.npz",allow_pickle=True)
        d=np.load(interim/f"drivers_{event}.npz",allow_pickle=True)
        fips=np.asarray(p["fips"]).astype(str)
        if not np.array_equal(fips,np.asarray(d["fips"]).astype(str)):
            raise ValueError(f"county order {event}")
        y,obs=to_hourly(p["y"],p["observed"])
        X=np.asarray(d["X"],dtype=np.float32)
        channels=[str(c) for c in d["channels"]]
        ts=pd.to_datetime([str(x) for x in d["ts"]])
        H=min(y.shape[1],X.shape[1],len(ts))
        y=y[:,:H]; obs=obs[:,:H]; X=X[:,:H]; ts=ts[:H]
        peak,fp=noaa_peak(events_df,fips,ts,X,channels)
        start=peak-24; end=peak+23
        avail=(start>=0 and end+1 < H)
        F,names=build_features(X,channels,ts,peak,fips,statics)
        out[event]=EventData(event,family.get(event,"unknown"),fips,y,obs,X,channels,ts,np.asarray(p["denominator"],float),peak,fp,avail,start if avail else None,end if avail else None,F,names)
    return out,manifest


def transitions(ev: EventData, t_start: int=0, t_end: int|None=None, counties: np.ndarray|None=None) -> pd.DataFrame:
    C,H=ev.y.shape
    if counties is None: counties=np.arange(C)
    if t_end is None: t_end=H-2
    t_start=max(0,int(t_start)); t_end=min(H-2,int(t_end))
    cur=np.arange(t_start,t_end+1)
    ok=ev.obs[np.ix_(counties,cur)] & ev.obs[np.ix_(counties,cur+1)]
    yy=ev.y[np.ix_(counties,cur)]
    yn=ev.y[np.ix_(counties,cur+1)]
    ok &= np.isfinite(yy)&np.isfinite(yn)
    ci,ti=np.nonzero(ok)
    cidx=counties[ci]; t=cur[ti]
    data={"event":ev.event,"family":ev.family,"county":ev.fips[cidx],"ci":cidx,"t":t,
          "y":ev.y[cidx,t],"delta":ev.y[cidx,t+1]-ev.y[cidx,t]}
    df=pd.DataFrame(data)
    for j,n in enumerate(ev.feature_names): df[n]=ev.features[cidx,t+1,j]
    return df


def equal_event_weights(df: pd.DataFrame) -> np.ndarray:
    cnt=df.groupby("event")["event"].transform("size").to_numpy(float)
    E=df["event"].nunique()
    w=1.0/(E*cnt)
    return w/w.sum()


def deterministic_cap(df:pd.DataFrame,cap:int,salt:str)->pd.DataFrame:
    parts=[]
    for e,g in df.groupby("event",sort=True):
        if len(g)<=cap: parts.append(g); continue
        h=np.array([sha_u64(e,c,int(t),salt=salt) for c,t in zip(g.county,g.t)],dtype=np.uint64)
        ix=np.argsort(h,kind="mergesort")[:cap]
        parts.append(g.iloc[np.sort(ix)])
    return pd.concat(parts,ignore_index=True)


def fit_one(y,delta,w):
    p=1-y
    A=np.sum(w*p*p); B=np.sum(w*y*y)
    a=np.clip(np.sum(w*p*delta)/A if A>1e-18 else 0,0,CAP_U)
    b=np.clip(-np.sum(w*y*delta)/B if B>1e-18 else 0,0,CAP_R)
    ja=np.sum(w*(delta-a*p)**2); jb=np.sum(w*(delta+b*y)**2)
    return (float(a),0.0,"U",float(ja)) if ja<=jb else (0.0,float(b),"R",float(jb))


def fit_two(y,delta,w):
    X=np.c_[1-y,-y]; sw=np.sqrt(w)
    sol=lsq_linear(X*sw[:,None],delta*sw,bounds=([0,0],[CAP_U,CAP_R]),max_iter=300,lsmr_tol="auto")
    U,R=sol.x
    return float(U),float(R),"two",float(np.sum(w*(delta-(U*(1-y)-R*y))**2))


def fit_clusters(df:pd.DataFrame,K:int,seed:int=0,cap:int|None=12000):
    d=df.reset_index(drop=True) if cap is None else deterministic_cap(df,cap,"fit")
    feat=d[EVENT_FEATURES].to_numpy(float)
    scaler=StandardScaler().fit(feat)
    Z=scaler.transform(feat)
    w=equal_event_weights(d)
    km=MiniBatchKMeans(n_clusters=K,random_state=seed,n_init=5,batch_size=4096,max_iter=200,reassignment_ratio=0.0)
    lab=km.fit_predict(Z,sample_weight=w*len(w))
    y=d.y.to_numpy(float); delta=d.delta.to_numpy(float)
    one=[]; two=[]
    for k in range(K):
        z=lab==k
        if z.sum()<8:
            wk=w/w.sum(); one.append(fit_one(y,delta,wk)); two.append(fit_two(y,delta,wk)); continue
        wk=w[z]/w[z].sum(); one.append(fit_one(y[z],delta[z],wk)); two.append(fit_two(y[z],delta[z],wk))
    return {"scaler":scaler,"km":km,"one":one,"two":two,"K":K}


def predict_rate(model,features,y,arm="two"):
    Z=model["scaler"].transform(np.asarray(features, dtype=np.float32)).astype(model["km"].cluster_centers_.dtype, copy=False)
    labels=model["km"].predict(Z)
    pars=model[arm]
    U=np.array([pars[k][0] for k in labels]); R=np.array([pars[k][1] for k in labels])
    return U*(1-y)-R*y,U,R,labels


def one_step_event(model,ev:EventData,t_start:int,t_end:int,counties=None):
    df=transitions(ev,t_start,t_end,counties)
    X=df[EVENT_FEATURES].to_numpy(float); y=df.y.to_numpy(float); d=df.delta.to_numpy(float)
    out={}
    for arm in ("one","two"):
        p,U,R,L=predict_rate(model,X,y,arm)
        out[arm]={"mse":float(np.mean((d-p)**2)),"n":len(d),"pred":p,"U":U,"R":R,"labels":L}
    return df,out


def path_event(model,ev:EventData,origins:List[int],counties=None,horizon=24):
    C,H=ev.y.shape
    if counties is None: counties=np.arange(C)
    records=[]
    for o in origins:
        if o<0 or o+horizon>=H: continue
        valid0=ev.obs[counties,o]&np.isfinite(ev.y[counties,o])
        idx=counties[valid0]
        if len(idx)==0: continue
        yhat_one=ev.y[idx,o].copy(); yhat_two=yhat_one.copy()
        sse={"one":0.0,"two":0.0}; n={"one":0,"two":0}; h24={}
        for step in range(1,horizon+1):
            t=o+step
            F=ev.features[idx,t,:]
            for arm,yh in (("one",yhat_one),("two",yhat_two)):
                delta,U,R,L=predict_rate(model,F,yh,arm)
                yh[:] = np.clip(yh+delta,0,1)
                mask=ev.obs[idx,t]&np.isfinite(ev.y[idx,t])
                sse[arm]+=float(np.sum((yh[mask]-ev.y[idx[mask],t])**2)); n[arm]+=int(mask.sum())
                if step==horizon: h24[arm]=(float(np.mean((yh[mask]-ev.y[idx[mask],t])**2)) if mask.any() else np.nan)
        records.append({"origin":o,"one_path_mse":sse["one"]/max(n["one"],1),"two_path_mse":sse["two"]/max(n["two"],1),
                        "one_h24_mse":h24.get("one",np.nan),"two_h24_mse":h24.get("two",np.nan),"n":n["two"]})
    return pd.DataFrame(records)


# assigned after first load
# Set by runner after loading the pinned corpus.
EVENT_FEATURES: List[str]=[]
