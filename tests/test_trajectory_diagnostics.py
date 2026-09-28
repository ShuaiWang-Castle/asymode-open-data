"""Accounting and censoring guards for the D-only frozen diagnostic, no real data."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/geo_weather_20260924'))
from diagnose_trajectory_v1 import first_cross, half_time, oracle_errors, stock


def test_oracle_separates_pure_scale_and_shift_without_reading_masked_values():
    p=np.zeros((2,60)); p[:,25:30]=.1
    y=np.zeros_like(p); y[0,25:30]=.5; y[1,30:35]=.1
    m=np.ones_like(p,bool); m[:,0]=False; y[:,0]=100
    err,scale,_=oracle_errors(p,y,m)
    assert err['amplitude'][0] < 1e-14
    assert err['timing'][1] < 1e-14
    assert err['joint'].max() < 1e-14
    assert err['timing'][0] > 0
    np.testing.assert_allclose(scale,[5,0])


def test_half_time_requires_sustained_observed_followup_and_reports_censoring():
    x=np.array([[.8,.3,.6,.3,.3,.3], [.1,.2,.3,.4,.5,.8], [.8,.3,.3,.3,.3,.3]])
    m=np.ones_like(x,bool); m[2,2]=False
    got=half_time(x,m,sustain=3)
    assert got[0]==3
    assert np.isnan(got[1])
    assert got[2]==3
    m[2,4]=False
    assert np.isnan(half_time(x,m,sustain=3)[2])
    assert first_cross(x,m,threshold=.7).tolist()==[0,5,0]


def test_zero_recovery_is_a_fixed_damage_upper_envelope():
    rng=np.random.default_rng(7)
    u=rng.uniform(0,.515,(12,144)); r=rng.uniform(0,.5,(12,144))
    y0=rng.uniform(0,1,12)
    p=stock(u,r,y0); ceil=stock(u,np.zeros_like(r),y0)
    assert ((p>=0)&(p<=1)).all()
    assert (ceil>=p-1e-12).all()
    exact=1-(1-y0[:,None])*np.cumprod(1-u,axis=1)
    np.testing.assert_allclose(ceil,exact)


def test_zero_prediction_has_no_defined_relative_half_time():
    x=np.zeros((2,12)); m=np.ones_like(x,bool)
    assert np.isnan(half_time(x,m,min_peak=0)).all()
