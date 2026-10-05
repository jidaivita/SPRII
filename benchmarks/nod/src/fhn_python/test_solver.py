from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from solver import FHNConfig, apply_laplacian, gaussian_random_fields, simulate


class SolverTests(unittest.TestCase):
    def test_constant_laplacian_is_zero(self):
        field = np.full((2, 8, 8), 3.25)
        np.testing.assert_allclose(apply_laplacian(field, 1 / 8), 0.0, atol=2e-13)

    def test_periodic_fourier_mode_matches_stencil_symbol(self):
        size = 16
        mode = 3
        x = np.arange(size) / size
        field = np.sin(2 * np.pi * mode * x)[None, :] * np.ones((size, 1))
        result = apply_laplacian(field, 1 / size)
        theta = 2 * np.pi * mode / size
        eigenvalue = size**2 * (-2.5 + (8 / 3) * np.cos(theta) - (1 / 6) * np.cos(2 * theta))
        np.testing.assert_allclose(result, eigenvalue * field, rtol=1e-12, atol=1e-11)

    def test_grf_is_deterministic_and_zero_mean_mode_removed(self):
        first = gaussian_random_fields(2, 8, seed=7)
        second = gaussian_random_fields(2, 8, seed=7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first.mean(axis=(-2, -1)), 0.0, atol=1e-14)

    def test_short_simulation_shape_and_finiteness(self):
        config = FHNConfig(grid_size=8, dt=1e-3, steps=10, sample_every=5)
        initial = gaussian_random_fields(2, 8, seed=11)
        result = simulate(initial, k=0.03, beta=0.2, config=config)
        self.assertEqual(result.shape, (2, 3, 2, 8, 8))
        self.assertTrue(np.isfinite(result).all())

    def test_translation_equivariance(self):
        config = FHNConfig(grid_size=8, dt=1e-3, steps=4, sample_every=2)
        initial = gaussian_random_fields(1, 8, seed=19)
        base = simulate(initial, k=0.02, beta=0.15, config=config, output_dtype=np.float64)
        shifted = simulate(
            np.roll(initial, shift=(2, -1), axis=(-2, -1)),
            k=0.02,
            beta=0.15,
            config=config,
            output_dtype=np.float64,
        )
        np.testing.assert_allclose(
            shifted,
            np.roll(base, shift=(2, -1), axis=(-2, -1)),
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
