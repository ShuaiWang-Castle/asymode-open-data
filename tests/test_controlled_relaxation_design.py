"""Synthetic algebra checks for the untrained controlled-relaxation proposal.

No panel, model checkpoint, target, or production training implementation is read.
These checks verify mathematical construction, not predictive performance.
"""
import itertools
import math
import unittest

import numpy as np


def subset_means(k):
    """Normalized elementary symmetric polynomials, including order zero."""
    result = np.zeros(len(k) + 1, dtype=np.float64)
    result[0] = 1.0
    for count, value in enumerate(k, 1):
        for order in range(count, 0, -1):
            result[order] = (
                (count - order) / count * result[order]
                + order / count * value * result[order - 1]
            )
    return result


def cayley_apply(a, b, eta, x):
    u = np.column_stack((eta * a, -eta * b))
    v = np.column_stack((b, a))
    return x + 2 * u @ np.linalg.solve(np.eye(2) - v.T @ u, v.T @ x)


class ControlledRelaxationDesign(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(928)

    def test_all_orders_against_enumeration(self):
        values = self.rng.random(7)
        actual = subset_means(values)
        for order in range(1, 8):
            expected = np.mean([
                np.prod(values[list(indices)])
                for indices in itertools.combinations(range(7), order)
            ])
            self.assertAlmostEqual(actual[order], expected, places=14)
        np.testing.assert_allclose(subset_means(np.ones(40)), 1.0, atol=2e-15)
        np.testing.assert_allclose(subset_means(np.zeros(40))[1:], 0.0)

    def test_anchored_forcing_value_bound_and_reference_derivative(self):
        geo = self.rng.normal(size=40)
        weights = self.rng.normal(size=(8, 40))
        input_weights = self.rng.normal(size=(8, 12))

        def forcing(g, u, w):
            reference = w @ g
            return (np.tanh(reference + input_weights @ u)
                    - np.tanh(reference)) / (2 * math.sqrt(8))

        zero = np.zeros(12)
        np.testing.assert_array_equal(forcing(geo, zero, weights), np.zeros(8))
        for scale in (0.01, 1, 100):
            actual = forcing(geo, scale * self.rng.normal(size=12), weights)
            self.assertLessEqual(np.linalg.norm(actual), 1)
        epsilon = 1e-5
        dg = self.rng.normal(size=40)
        dw = self.rng.normal(size=weights.shape)
        derivative = (forcing(geo + epsilon * dg, zero, weights + epsilon * dw)
                      - forcing(geo - epsilon * dg, zero, weights - epsilon * dw))
        np.testing.assert_array_equal(derivative, np.zeros(8))
        du = self.rng.normal(size=12)
        input_derivative = (forcing(geo, epsilon * du, weights)
                            - forcing(geo, -epsilon * du, weights)) / (2 * epsilon)
        self.assertGreater(np.linalg.norm(input_derivative), 1e-9)

    def test_geography_kernel_psd_and_high_order_range(self):
        points = self.rng.normal(size=(18, 40))
        weights = self.rng.random(40)
        weights /= weights.sum()
        gram = np.empty((18, 18))
        for i in range(18):
            for j in range(18):
                bases = np.exp(-0.5 * (points[i] - points[j]) ** 2)
                orders = subset_means(bases)[1:]
                self.assertTrue(np.all((orders >= 0) & (orders <= 1)))
                gram[i, j] = orders @ weights
        np.testing.assert_allclose(np.diag(gram), 1.0, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(gram).min(), -1e-12)

    def test_cayley_low_rank_dense_and_orthogonality(self):
        for _ in range(30):
            a, b = self.rng.normal(size=(2, 8))
            a /= math.sqrt(1 + a @ a)
            b /= math.sqrt(1 + b @ b)
            eta = self.rng.uniform(-0.25, 0.25)
            skew = eta * (np.outer(a, b) - np.outer(b, a))
            q = np.linalg.solve(np.eye(8) - skew, np.eye(8) + skew)
            low_rank = cayley_apply(a, b, eta, np.eye(8))
            np.testing.assert_allclose(low_rank, q, atol=1e-14)
            np.testing.assert_allclose(q.T @ q, np.eye(8), atol=1e-14)

    def test_bounds_contraction_and_readout(self):
        x = np.zeros((4, 8))
        y = np.tanh(self.rng.normal(size=(4, 8))) / math.sqrt(8)
        planes = self.rng.normal(size=(4, 4, 2, 8))
        planes /= np.sqrt(1 + np.sum(planes**2, axis=-1, keepdims=True))
        for _ in range(240):
            rho = np.exp(-1 / self.rng.uniform(2, 96, size=4))
            old_difference = np.linalg.norm(x - y, axis=1)
            for mode in range(4):
                for a, b in planes[mode]:
                    eta = self.rng.uniform(-0.25, 0.25)
                    x[mode] = cayley_apply(a, b, eta, x[mode])
                    y[mode] = cayley_apply(a, b, eta, y[mode])
            d = np.tanh(self.rng.normal(size=(4, 8))) / math.sqrt(8)
            x = rho[:, None] * x + (1 - rho[:, None]) * d
            y = rho[:, None] * y + (1 - rho[:, None]) * d
            self.assertLessEqual(np.linalg.norm(x, axis=1).max(), 1 + 1e-12)
            self.assertLessEqual(np.linalg.norm(y, axis=1).max(), 1 + 1e-12)
            np.testing.assert_allclose(
                np.linalg.norm(x - y, axis=1), rho * old_difference,
                atol=1e-13, rtol=1e-9,
            )
            gain = self.rng.uniform(0.5, 2, size=(4, 1))
            self.assertLessEqual(np.linalg.norm(gain * x / 2), 2 + 1e-12)

    def test_relaxation_equilibrium_and_pulse_tradeoff(self):
        d = np.tanh(self.rng.normal(size=8)) / math.sqrt(8)
        pulse_norms = []
        for tau in (3, 8, 24, 72):
            rho = math.exp(-1 / tau)
            np.testing.assert_allclose(rho * d + (1 - rho) * d, d, atol=1e-15)
            final = (1 - rho**2400) * d
            np.testing.assert_allclose(final, d, atol=1e-13)
            pulse_norms.append(np.linalg.norm((1 - rho) * d))
        self.assertTrue(np.all(np.diff(pulse_norms) < 0))

    def test_response_expansion_and_affine_order_identity(self):
        dim = 8
        a = []
        b = []
        x = np.zeros(dim)
        for _ in range(12):
            v, w = self.rng.normal(size=(2, dim))
            q = cayley_apply(v / (1 + np.linalg.norm(v)),
                             w / (1 + np.linalg.norm(w)), 0.2, np.eye(dim))
            a.append(0.9 * q)
            b.append(0.1 * np.tanh(self.rng.normal(size=dim)) / math.sqrt(dim))
            x = a[-1] @ x + b[-1]
        expanded = np.zeros(dim)
        product = np.eye(dim)
        for j in reversed(range(len(a))):
            expanded += product @ b[j]
            product = product @ a[j]
        np.testing.assert_allclose(x, expanded, atol=1e-14)
        initial = self.rng.normal(size=dim)
        ab = a[1] @ (a[0] @ initial + b[0]) + b[1]
        ba = a[0] @ (a[1] @ initial + b[1]) + b[0]
        expected = (
            (a[1] @ a[0] - a[0] @ a[1]) @ initial
            + (a[1] - np.eye(dim)) @ b[0]
            - (a[0] - np.eye(dim)) @ b[1]
        )
        np.testing.assert_allclose(ab - ba, expected, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
