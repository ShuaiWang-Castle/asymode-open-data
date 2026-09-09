"""Deterministic paired sampling and independent raw-state scoring (not a trainer)."""
from __future__ import annotations
import numpy as np
import pandas as pd


class CompanyEqualSampler:
    """RNG isolated from model initialization; persist state in checkpoints for resume.
    Uniform company, then uniform group within company, then uniform legal origin.
    Instances built with identical seed/origins produce the same training sample sequence.
    """
    def __init__(self, selected: pd.DataFrame, origins: np.ndarray, seed: int):
        self.rng = np.random.default_rng(int(seed)+100000)
        self.origins = np.asarray(origins, dtype=np.int64)
        if self.origins.ndim != 1 or len(self.origins)==0:
            raise ValueError('Nonempty origins needed')
        companies = selected.company.to_numpy()
        self.groups = [np.flatnonzero(companies==c) for c in sorted(np.unique(companies))]

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        co = self.rng.integers(0,len(self.groups),size=batch_size)
        gi = np.empty(batch_size,dtype=np.int64)
        for c, allowed in enumerate(self.groups):
            ix = np.flatnonzero(co==c)
            gi[ix] = self.rng.choice(allowed,size=len(ix))
        hours = self.rng.choice(self.origins,size=batch_size)
        return gi,hours

    def state_dict(self) -> dict:
        return self.rng.bit_generator.state

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state


def score_full_paths(prediction: np.ndarray, truth: np.ndarray, selected: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    """Inputs [group, legal_origin, horizon] in selected.sort_values('group') order.
    Float64 accumulation, no clipping, no direction labels, no fitted quantities.
    Mean within group, then within company, then across company, not pooled windows.
    """
    p,t = np.asarray(prediction,dtype=np.float64),np.asarray(truth,dtype=np.float64)
    if p.shape != t.shape or p.ndim != 3 or p.shape[0] != len(selected) or p.shape[2] != 24:
        raise ValueError('Expected matching [len(selected),origins,24] arrays')
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Nonfinite predictions: preserve as failed trial, do not ignore NaNs')
    if p.shape[1]==0:
        raise ValueError('Empty scoring population')
    e = (p-t)**2
    rows = []
    for i,r in enumerate(selected.itertuples()):
        rows.append({'group':int(r.group),'company':int(r.company),'n_origins':p.shape[1],
                     'mse_path':float(e[i].mean()),'mse_1h':float(e[i,:,0].mean()),
                     'mse_6h':float(e[i,:,5].mean()),'mse_24h':float(e[i,:,23].mean()),
                     'out_of_bounds_fraction':float(((p[i]<0)|(p[i]>1)).mean())})
    group = pd.DataFrame(rows)
    cols = ['mse_path','mse_1h','mse_6h','mse_24h','out_of_bounds_fraction']
    company = group.groupby('company',sort=True)[cols].mean().reset_index()
    result = {c:float(company[c].mean()) for c in cols}
    return group,company,result


def mean_company_relative_gain(direct_company: pd.DataFrame, model_company: pd.DataFrame) -> dict:
    a=direct_company.set_index('company').mse_path.sort_index()
    b=model_company.set_index('company').mse_path.sort_index()
    if not a.index.equals(b.index):
        raise ValueError('Companies differ between arms')
    if (a<0).any() or (b<0).any() or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid MSE')
    zero = int((a==0).sum())
    # Strictly N/A for the full-cohort statistic if its formula is undefined.
    return {'mean_company_relative_gain':None if zero else float(((a-b)/a).mean()),
            'zero_direct_denominator_companies':zero,
            'ratio_of_means_gain':None if a.mean()==0 else float((a.mean()-b.mean())/a.mean())}
