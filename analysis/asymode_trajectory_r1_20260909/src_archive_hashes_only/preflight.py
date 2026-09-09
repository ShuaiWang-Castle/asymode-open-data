"""CPU model/data checks, NOT real tuning or a scientific performance comparison."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
import torch
from core_models import TrajectoryModel, path_loss, one_step_ablation_loss
from data_contract import (PARTITIONS, load_source, legal_origins, fit_state_stats,
                           company_group_probabilities, assemble_batch)


def main(data_manifest: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20260909)
    rng = np.random.default_rng(20260909)
    ledger, selected = load_source(data_manifest)
    origins, rows = {}, []
    for name, year, start, end, cap in PARTITIONS:
        o = legal_origins(year, start, end, cap)
        origins[name] = o
        rows.append({'partition':name,'year':year,'history':24,'horizon':24,'n_per_group':len(o),
                     'n_total':len(o)*24,'first_hour':int(o[0]),'last_hour':int(o[-1]),
                     'sha256':hashlib.sha256(o.tobytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(output/'WINDOW_COUNTS.csv', index=False)
    np.savez_compressed(output/'origin_hours_L24_H24.npz', **origins)
    stats = fit_state_stats(ledger, selected, origins['fold_A_fit'])
    gi = rng.choice(24,64,p=company_group_probabilities(selected))
    hours = rng.choice(origins['fold_A_fit'],64)
    batch = assemble_batch(ledger,selected,2018,gi,hours,stats['state_scale'])
    b = {k:torch.tensor(v,dtype=torch.float64) for k,v in batch.items()}
    results, capacities = [], []
    for kind in ['DIRECT','NET','ASYM','SR']:
        for budget in [8192,32768]:
            model=TrajectoryModel(kind,b['c'].shape[1],parameter_target=budget,**stats).double()
            num=sum(p.numel() for p in model.parameters())
            assert abs(num-budget)/budget < .05
            capacities.append({'model':kind,'budget':budget,'parameters':num,'width':model.width})
        model=TrajectoryModel(kind,b['c'].shape[1],parameter_target=8192,**stats).double()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        losses=[]
        for step in range(12):
            pred=model(b['c'],b['y0'],b['known_clock'])
            assert pred.shape==b['target'].shape and torch.isfinite(pred).all()
            if kind in ['ASYM','SR']: assert float(pred.detach().min())>=-1e-12 and float(pred.detach().max())<=1+1e-12
            loss=path_loss(pred,b['target'],model.state_scale)
            opt.zero_grad();loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            p1=model(b['c'],b['y0'],b['known_clock'])
            changed_target=b['target'].flip(1)+.123
            p2=model(b['c'],b['y0'],b['known_clock'])
            assert torch.equal(p1,p2) # forward API cannot consume changed_target
        if kind=='ASYM':
            step_loss=one_step_ablation_loss(model,b['c'],b['true_states_for_step_ablation_only'],b['known_clock'])
            assert torch.isfinite(step_loss)
        results.append({'model':kind,'finite_forward_backward_steps':12,
                        'initial_batch_loss':losses[0],'final_batch_loss':losses[-1],
                        'prediction_shape':list(p1.shape)})
    # Exact recurrence and perturbation-identity check, no learned-model claim.
    H=24; u,r=.02,.2; a=1-u-r; y0=.35
    truth=[y0]
    for k in range(H):truth.append(u+a*truth[-1])
    closed=np.array([u/(u+r)+a**h*(y0-u/(u+r)) for h in range(H+1)])
    assert np.max(np.abs(closed-truth))<1e-14
    uh=np.full(H,.021); rh=np.full(H,.198); ah=1-uh-rh
    yp=y0; err=0.
    for k in range(H):
        yp=uh[k]+ah[k]*yp
        eta=(uh[k]-u)*(1-truth[k])-(rh[k]-r)*truth[k]
        err=ah[k]*err+eta
        assert abs((yp-truth[k+1])-err)<1e-14
    report={'status':'PASS','scope':'CPU primitives and 12-step gradient smoke tests, not formal experiments',
            'real_tuning_run':False,'cuda_available':torch.cuda.is_available(),'cuda_checks_run':False,
            'target_year_model_evaluation_run':False,'torch_version':torch.__version__,'history':24,'horizon':24,
            'context_dim':b['c'].shape[1],'fit_A_statistics':stats,'model_checks':results,'capacity_checks':capacities,
            'closed_form_max_abs_error':float(np.max(np.abs(closed-truth))),
            'time_varying_error_identity_checked':True,
            'limitation':'Target-independence smoke check complements interface/code review; not a proof against all loader leakage.'}
    (output/'PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.data_manifest.resolve(),a.output.resolve())
