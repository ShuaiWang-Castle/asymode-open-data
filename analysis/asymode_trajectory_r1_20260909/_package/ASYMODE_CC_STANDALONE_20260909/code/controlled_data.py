"""Locked E2 data generator. This is simulated data, never field-benefit evidence."""
from __future__ import annotations
import numpy as np


def make_controlled_splits(seed: int) -> dict:
    specs=[('fit',2048,.25,.45),('validation',1024,.25,.45),
           ('source_eval',4096,.25,.45),('high_eval',4096,.55,.75)]
    children=np.random.SeedSequence(seed).spawn(len(specs))
    output={}
    for (name,n,lo,hi),child in zip(specs,children):
        rng=np.random.default_rng(child)
        x=rng.uniform(-1,1,n);y0=rng.uniform(lo,hi,n)
        u=.005+.035*(x+1)/2;r=np.full(n,.20)
        a=1-u-r;pi=u/(u+r)
        clean=pi[:,None]+a[:,None]**np.arange(1,25)*(y0[:,None]-pi[:,None])
        noise=rng.normal(0,.005,size=clean.shape)
        output[name]={'c':x[:,None], 'y0':y0,
                      'known_clock':np.broadcast_to(x[:,None,None],(n,24,1)).copy(),
                      'target':clean+noise,'clean_truth':clean,'u_truth':u,'r_truth':r}
    # Available inputs only, no future targets or true rates in initialization.
    fit=output['fit']['y0']
    output['fit_statistics']={'state_scale':max(float(np.sqrt(np.mean(fit*fit))),1e-6),
                              'source_mean':float(fit.mean())}
    return output
