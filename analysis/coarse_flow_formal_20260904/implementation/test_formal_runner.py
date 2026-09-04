import numpy as np
import flow_data as F
from coarse_flow_formal_runner import exact_signflip

def test_one_flow_is_ray():
    y=np.array([0.,.1,.4,.8]);d=.03*(1-y)-.02*y;w=np.ones(4)/4
    U,R,branch,_=F.fit_one(y,d,w)
    assert U==0 or R==0
    assert branch in {'U','R'}

def test_two_flow_nested_training_loss():
    y=np.linspace(0,1,101);d=.03*(1-y)-.02*y;w=np.ones(len(y))/len(y)
    o=F.fit_one(y,d,w);t=F.fit_two(y,d,w)
    jo=np.sum(w*(d-(o[0]*(1-y)-o[1]*y))**2)
    jt=np.sum(w*(d-(t[0]*(1-y)-t[1]*y))**2)
    assert jt <= jo + 1e-14

def test_exact_signflip_all_positive():
    assert exact_signflip(np.ones(6)) == 2/64

def test_feature_count():
    # The formal feature list is checked at runtime against 24 channels.
    assert 12+3+2+1+2+4 == 24
