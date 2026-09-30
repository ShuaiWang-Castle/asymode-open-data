"""Artificial fixed-downstream convex oracle regressions; no real arrays."""
import unittest
from unittest.mock import patch
import json

import numpy as np

from d10_downstream_oracle import project, _solve_band, GAP_TOLERANCE


class DownstreamOracle(unittest.TestCase):
    def assert_certificate(self, path, certificate):
        self.assertTrue(np.isfinite(path).all())
        self.assertGreaterEqual(certificate['lower'],0.)
        self.assertLessEqual(certificate['lower'],certificate['upper']+1e-10)
        self.assertLessEqual(certificate['maximum_band_violation'],1e-10)
        self.assertLessEqual(certificate['maximum_box_violation'],1e-10)
        self.assertLessEqual(certificate['rate_reconstruct_maximum_residual'],1e-10)
        self.assertEqual(certificate['gap_certified'],certificate['gap']<=GAP_TOLERANCE)
        json.dumps(certificate,allow_nan=False)

    def test_one_hour_known_projection(self):
        p,c=project([.6],[True],.2,[.1],[.4],[.01])
        upper=(1-.1-.21)*.2+.21
        self.assertAlmostEqual(p[0],upper,places=12)
        self.assertAlmostEqual(c['upper'],(.6-upper)**2,places=12)
        self.assertTrue(c['gap_certified']);self.assert_certificate(p,c)

    def test_zero_gate_has_unique_forced_trajectory(self):
        p,c=project([.9,.1,.8],[1,1,1],.4,[.1,.2,.3],[0,0,0],[.01,.005,0])
        previous=.4;expected=[]
        for r,b in zip([.1,.2,.3],[.01,.005,0]):
            previous=previous+b*(1-previous)-r*previous;expected.append(previous)
        np.testing.assert_allclose(p,expected,rtol=0,atol=1e-12)
        self.assertTrue(c['gap_certified']);self.assert_certificate(p,c)

    def test_missing_hour_bridges_fixed_recovery(self):
        p,c=project([np.nan,0],[False,True],.8,[.5,.5],[0,0],[0,0])
        np.testing.assert_allclose(p,[.4,.2],atol=1e-12)
        self.assertAlmostEqual(c['lower'],.04,places=9)
        self.assertTrue(c['gap_certified']);self.assert_certificate(p,c)

    def test_legal_host_path_is_reachable(self):
        n=144;t=np.arange(n);r=.1+.05*np.sin(t/7);g=.2+.1*np.cos(t/9);b=.006+.003*np.sin(t/11)
        conditional=.2+.15*np.sin(t/13);previous=.05;y=[]
        for rr,gg,bb,cc in zip(r,g,b,conditional):
            previous=previous+(bb+gg*cc)*(1-previous)-rr*previous;y.append(previous)
        p,c=project(y,np.ones(n,bool),.05,r,g,b)
        self.assertLess(c['upper'],1e-15)
        self.assertTrue(c['gap_certified']);self.assert_certificate(p,c)

    def test_narrow_gate_cannot_reach_large_peak(self):
        p,c=project([.3],[1],0,[.1],[.001],[0])
        self.assertAlmostEqual(p[0],.0005,places=12)
        self.assertGreater(c['lower'],.089)
        self.assertTrue(c['gap_certified']);self.assert_certificate(p,c)

    def test_time_varying_gate_recovery_and_negative_transition_coefficient(self):
        y=np.array([.9,.1,.7,.3]);r=np.array([.5,.1,.4,.2]);g=np.array([1,0,.2,.8]);b=np.array([.015,0,.003,.01])
        p,c=project(y,np.ones(4,bool),.99,r,g,b)
        self.assertTrue(c['gap_certified']);self.assert_certificate(p,c)
        previous=.99
        for i in range(4):
            l,h=b[i],b[i]+.5*g[i]
            self.assertGreaterEqual(p[i]+1e-10,(1-r[i]-l)*previous+l)
            self.assertLessEqual(p[i]-1e-10,(1-r[i]-h)*previous+h)
            previous=p[i]

    def test_d06_wider_envelope_has_no_higher_optimum(self):
        y=np.array([.3,0,.5]);m=np.ones(3,bool);r=np.full(3,.1);g=np.full(3,.001);b=np.zeros(3)
        p,narrow=project(y,m,0,r,g,b)
        broad,wide=_solve_band(y,m,0,np.zeros(3),np.full(3,.515),np.full(3,.5),np.full(3,.485))
        self.assertTrue(narrow['gap_certified']);self.assertTrue(wide['gap_certified'])
        self.assertLessEqual(wide['upper'],narrow['lower']+1e-8)
        self.assertLess(wide['upper'],narrow['lower']-.01)
        self.assert_certificate(p,narrow)

    def test_unit_stock_equal_one_avoids_rate_division(self):
        p,c=project([.8,.4],[1,1],1,[.2,.5],[1,0],[.015,0])
        np.testing.assert_allclose(p,[.8,.4],atol=1e-12)
        self.assertTrue(c['gap_certified']);self.assert_certificate(p,c)

    def test_missing_objective_still_keeps_all_state_constraints(self):
        p,c=project([np.nan,np.nan],[False,False],.3,[.1,.2],[.2,.3],[.002,.004])
        self.assertEqual(c['lower'],0.);self.assertEqual(c['upper'],0.)
        self.assertEqual(c['observed_hours'],0);self.assertTrue(c['gap_certified']);self.assert_certificate(p,c)

    def test_invalid_shapes_masks_and_rates_are_rejected(self):
        for bad in ([.6],[-.1],[np.nan]):
            with self.assertRaises(ValueError):project([.2],[True],0,bad,[.3],[0])
        with self.assertRaises(ValueError):project([.2],[2],0,[.1],[.3],[0])
        with self.assertRaises(ValueError):project([.2],[True],0,[.1],[.3],[.016])
        with self.assertRaises(ValueError):project([.2],[True],0,[.1],[1.1],[0])

    def test_unconverged_solver_retains_valid_bounds_and_gap(self):
        with patch('d10_downstream_oracle.PRIMAL_MAXITER',1),patch('d10_downstream_oracle.DUAL_MAXITER',1):
            p,c=project([.2,0],[True,True],0,[.1,.1],[1,1],[0,0])
        exact=.04*.81/1.81
        self.assertFalse(c['primal_success'])
        self.assertLessEqual(c['lower'],exact+1e-10)
        self.assertGreaterEqual(c['upper'],exact-1e-10)
        self.assertEqual(c['primal_iteration_budget'],1)
        self.assertEqual(c['dual_iteration_budget'],1)
        self.assert_certificate(p,c)


if __name__ == '__main__':
    unittest.main()
