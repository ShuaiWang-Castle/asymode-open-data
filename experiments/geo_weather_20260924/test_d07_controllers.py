"""Synthetic-only checks for D07 controller diagnostics and intervention."""
import math
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch import nn

from d07_controllers import (preactivation, scaled_deposit, deposit_jacobian,
                             write_scale, cosine, timing_masks, EPS)
from asymode.controlled_relaxation import ControlledRelaxationLayer


class ControllerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(729)

    def test_analytic_jacobian_autograd_finite_difference(self):
        fw = torch.randn(32, 64, dtype=torch.float64) / 8
        hw = torch.randn(56, 64, dtype=torch.float64) / 20
        hf = torch.randn(56, 32, dtype=torch.float64) / 6
        fg = torch.randn(1, 32, dtype=torch.float64)
        a = torch.randn(1, 56, dtype=torch.float64)
        for magnitude in (.2, 4.):
            u = magnitude * torch.randn(1, 1, 64, dtype=torch.float64)
            direction = torch.randn_like(u)
            direction /= direction.norm()
            for scale in (1., .95, 1.05):
                psi, raw = preactivation(u, fg, a, fw, hw, hf)
                analytic = deposit_jacobian(raw, psi, a, fw, hw, hf, scale)[0, 0]
                def fun(x):
                    _, r = preactivation(x.reshape(1, 1, 64), fg, a, fw, hw, hf)
                    return scaled_deposit(r, a, scale).flatten()
                automatic = torch.autograd.functional.jacobian(fun, u.flatten())
                self.assertTrue(torch.allclose(analytic, automatic, atol=2e-12, rtol=2e-10))
                step = 1e-5
                fd = (fun((u + step * direction).flatten()) - fun((u - step * direction).flatten())) / (2 * step)
                self.assertTrue(torch.allclose(fd, analytic @ direction.flatten(), atol=2e-10, rtol=2e-7))

    def test_scale_bound_anchor_and_other_controls(self):
        layer = ControlledRelaxationLayer(nn.Linear(32, 32), torch.randn(5, 40),
                                         checkpoint_steps=0).double().eval()
        u = 20 * torch.randn(5, 4, 64, dtype=torch.float64)
        fg = torch.randn(5, 32, dtype=torch.float64)
        hg = torch.randn(5, 56, dtype=torch.float64)
        anchor = hg + torch.nn.functional.linear(torch.tanh(fg), layer.head_fusion)
        before = {k: v.clone() for k, v in layer.state_dict().items() if isinstance(v, torch.Tensor)}
        controls = layer._controls(u, fg, hg, anchor)
        for scale in (.95, 1.05):
            with write_scale(layer, scale):
                changed = layer._controls(u, fg, hg, anchor)
                self.assertLessEqual(float(changed['deposit'].norm(dim=-1).max()), 1.)
                for key in ('tau', 'rho', 'eta', 'gain'):
                    self.assertTrue(torch.equal(changed[key], controls[key]))
                zero = layer._controls(torch.zeros_like(u), fg, hg, anchor)['deposit']
                self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
            self.assertTrue(torch.equal(layer._controls(u, fg, hg, anchor)['deposit'], controls['deposit']))
            self.assertNotIn('_controls', layer.__dict__)
        for key, val in before.items():
            self.assertTrue(torch.equal(val, layer.state_dict()[key]))

    def test_cosine_missing_and_peak_support(self):
        a = torch.tensor([[0., 0.], [EPS / 2, 0.], [1., 0.]])
        b = torch.tensor([[1., 0.], [1., 0.], [-1., 0.]])
        c = cosine(a, b)
        self.assertTrue(torch.isnan(c[:2]).all())
        self.assertEqual(float(c[2]), -1.)
        m = np.zeros((2, 144), bool)
        m[:, 0:3] = [[1, 0, 1], [0, 1, 1]]
        y = np.zeros_like(m, dtype=float)
        y[0, :3] = [1, 100, 2]
        y[1, :3] = [100, 3, 2]
        obs = np.zeros((2, 216), bool)
        obs[:, 72:] = m
        data = dict(m=m, y=y, obs_full=obs)
        timings = timing_masks(data, y, y)
        self.assertTrue(np.array_equal(np.nonzero(timings['true_peak'])[1], [2, 1]))
        self.assertFalse(timings['observed_adjacent_hours'][0].any())
        self.assertTrue(timings['observed_adjacent_hours'][1, 2])


if __name__ == '__main__':
    unittest.main()
