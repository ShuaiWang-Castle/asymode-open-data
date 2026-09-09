"""Small reference tests. Not formal training or a real-data performance comparison."""
from pathlib import Path
import argparse
import json
import tempfile
import numpy as np
import pandas as pd
import torch
from core_models import TrajectoryModel
from training_primitives import CompanyEqualSampler, score_full_paths, mean_company_relative_gain
from controlled_data import make_controlled_splits
from prepare_local_data import EXPECTED_COLUMNS, cohort, legacy_csv_records


def run_tests() -> dict:
    torch.set_num_threads(1)
    d=pd.DataFrame({'group':[2,7,9],'company':[0,0,1]})
    origins=np.arange(60,80)
    a=CompanyEqualSampler(d,origins,10);b=CompanyEqualSampler(d,origins,10)
    assert all(np.array_equal(x,y) for x,y in zip(a.sample(512),b.sample(512)))
    state=a.state_dict();x=a.sample(512);a.load_state_dict(state)
    assert all(np.array_equal(i,j) for i,j in zip(x,a.sample(512)))
    p=np.array([1.,3.,2.])[:,None,None]*np.ones((3,4,24));t=np.zeros_like(p)
    g,c,score=score_full_paths(p,t,d)
    assert score['mse_path']==4.5  # company means are 5 and 4, not pooled 14/3
    assert mean_company_relative_gain(c,c)['mean_company_relative_gain']==0
    c0=c.copy();c0.loc[0,'mse_path']=0
    assert mean_company_relative_gain(c0,c)['mean_company_relative_gain'] is None
    synth=make_controlled_splits(123);again=make_controlled_splits(123)
    for key in ('fit','validation','source_eval','high_eval'):
        assert np.array_equal(synth[key]['target'],again[key]['target'])
        x=synth[key];y=x['y0'].copy();pred=[]
        for k in range(24):
            y=y+x['u_truth']*(1-y)-x['r_truth']*y;pred.append(y.copy())
        assert np.max(np.abs(np.stack(pred,axis=1)-x['clean_truth']))<1e-14
    calls={};b0=synth['fit'];stats=synth['fit_statistics']
    c0=torch.tensor(b0['c'][:8],dtype=torch.float64)
    y0=torch.tensor(b0['y0'][:8],dtype=torch.float64)
    clock=torch.tensor(b0['known_clock'][:8],dtype=torch.float64)
    for kind in ['DIRECT','NET','ASYM','SR']:
        torch.manual_seed(10)
        model=TrajectoryModel(kind,1,clock_dim=1,parameter_target=8192,**stats).double()
        hits=[]
        hook=model.net.register_forward_hook(lambda *args: hits.append(1))
        pred=model(c0,y0,clock)
        hook.remove();calls[kind]=len(hits)
        assert pred.shape==(8,24) and torch.isfinite(pred).all()
        # A last-horizon-only loss must backpropagate through the entire recurrence.
        pred[:,-1].square().mean().backward()
        assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
        if kind in ['ASYM','SR']:
            assert pred.min()>=0 and pred.max()<=1
    assert calls=={'DIRECT':1,'NET':24,'ASYM':1,'SR':1}
    # Full-row de-dup fixture: same physical interval but a distinct source row remains.
    co=cohort().iloc[0];cnpj,unit=co['key'].split('|')
    with tempfile.TemporaryDirectory() as temp:
        files={}
        for year in (2018,2019):
            row={name:'0' for name in EXPECTED_COLUMNS}
            row.update(NumCPFCNPJ=cnpj,IdeConjuntoUnidadeConsumidora=unit,
                       DatInicioInterrupcao=f'{year}-01-03 02:00:00',
                       DatFimInterrupcao=f'{year}-01-03 03:00:00',
                       NumUnidadeConsumidora='2',NumConsumidorConjunto=str(int(co.denominator)),
                       NumAno=str(year),NumOrdemInterrupcao='first')
            distinct=dict(row,NumOrdemInterrupcao='second')
            frame=pd.DataFrame([row,row,distinct],columns=EXPECTED_COLUMNS)
            path=Path(temp)/f'interrupcoes-{year}.csv';frame.to_csv(path,index=False,sep=';')
            files[year]=path
        rec=legacy_csv_records(files)
        assert len(rec[int(co.group)])==4  # two distinct rows per annual CSV, not one or three
        assert (rec[int(co.group)]['n']==2).all()
    return {'status':'PASS','scope':'CPU primitives and CSV fixture only',
            'paired_sampling_and_resume':True,'company_equal_scoring':True,
            'controlled_formula_and_reproducibility':True,'model_network_calls':calls,
            'last_horizon_backprop_finite':True,'legacy_csv_full_row_dedup_fixture':True,
            'full_national_csv_reconstruction_tested':False,'formal_training_run':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    a=p.parse_args();result=run_tests();text=json.dumps(result,indent=2)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text)
