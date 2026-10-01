"""Replay a saved neural checkpoint on its recorded held-out rows.

Uses only weather, geography and pre-origin outage inputs. Training indices
reconstruct and check the saved transforms; future target arrays are not read
by either batches() or the model. Used as a reproducibility check, not scoring.
"""
import argparse,json
import numpy as np
import torch
from run_campaign import load,batches,model_for,OUT,CODE_HASH

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    root=OUT/'cache'/args.tag
    record=json.loads(root.with_suffix('.json').read_text());ident=record['identity']
    assert ident['code']==CODE_HASH
    a=load();fit=np.array(ident['fit']);idx=np.array(ident['predict'])
    b,stats=batches(a,fit)
    # Only locally generated checkpoints from this experiment are accepted.
    checkpoint=torch.load(root.with_suffix('.pt'),map_location='cpu',weights_only=False)
    assert checkpoint['identity']==ident
    for k in stats:
        for v in ['mean','sd']:assert np.array_equal(stats[k][v],checkpoint['stats'][k][v])
    model=model_for({k:v[fit] for k,v in b.items()},ident['kind'],ident['seed'])
    model.load_state_dict(checkpoint['state']);model.eval();out=[]
    with torch.no_grad():
        for ii in np.array_split(idx,max(1,int(np.ceil(len(idx)/128)))):
            out.append(model({k:v[ii] for k,v in b.items()})['P'].numpy())
    p=np.concatenate(out);ref=np.load(root.with_suffix('.npz'))['P']
    error=float(np.max(np.abs(p-ref)));assert error<1e-6
    print(json.dumps(dict(tag=args.tag,rows=len(idx),max_absolute_difference=error,passed=True)))

if __name__=='__main__':main()
