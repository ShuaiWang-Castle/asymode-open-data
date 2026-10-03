"""Replay representative saved checkpoints independently of stored prediction files."""
from t03_poster_trajectory import PRIVATE,OUT,ARMS,AsymODE,standardize,indices,predict,dump
import numpy as np
import torch,json
import lightgbm as lgb

def main():
    d=dict(np.load(PRIVATE/'data.npz',allow_pickle=False));tr,va,te=indices(d,0)
    base,bs=standardize(d['base'],tr);full,fs=standardize(d['full'],tr);geo,gs=standardize(d['geo'],tr)
    b={'xr':torch.from_numpy(base),'xo':torch.from_numpy(base[:,:,d['occ']]),'geo':torch.from_numpy(geo),'ctx':torch.from_numpy(geo),'y0':torch.from_numpy(d['y0'])}
    errors={}
    for arm in ARMS:
        ck=torch.load(PRIVATE/f'{arm}_f0_s0.pt',weights_only=False,map_location='cpu')
        assert ck['stats']==dict(base=bs,full=fs,geo=gs)
        model=AsymODE(base.shape[-1],base.shape[-1],len(d['occ']))
        if arm!='HOST':model.expand_damage_inputs(full.shape[-1]-base.shape[-1])
        if arm=='GCRK':model.attach_gcrk(torch.tanh(b['geo'][tr]/3).mean(0))
        if arm=='GEO_MLP':model.attach_context_input(geo.shape[-1])
        model.load_state_dict(ck['state']);b['xu']=torch.from_numpy(base if arm=='HOST' else full)
        p=predict(model,b,te);saved=np.load(PRIVATE/f'{arm}_f0_s0.npz')['tp']
        error=float(np.max(abs(p-saved)));assert error<1e-6;errors[arm]=error
    z=np.load(PRIVATE/'tree_f0.npz');rate=float(z['rate']);dp=d['y0'][:,None]*np.exp(-rate*np.arange(1,145)[None,:])
    lead=np.broadcast_to(np.arange(1,145)[None,:,None]/144,(len(d['y']),144,1))
    x=np.concatenate([d['full'][:,72:],lead],axis=2).astype(np.float32)
    for arm in ('TREE_W','TREE_G'):
        xx=x if arm=='TREE_W' else np.concatenate([x,np.broadcast_to(d['geo'][:,None,:],(len(x),144,d['geo'].shape[-1]))],axis=2)
        model=lgb.Booster(model_file=str(PRIVATE/f'{arm}_f0.txt'))
        p=(model.predict(xx[te].reshape(-1,xx.shape[-1]),num_threads=1).reshape(len(te),144)+dp[te]).clip(0,1)
        error=float(np.max(abs(p-z[arm+'_tp'])));assert error<1e-6;errors[arm]=error
    audit=json.loads((OUT/'verification.json').read_text());audit['checkpoint_replay_fold0_seed0_max_errors']=errors
    dump(OUT/'verification.json',audit);print(errors)

if __name__=='__main__':main()
