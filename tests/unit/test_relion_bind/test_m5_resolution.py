"""M5: Resolution from data_vs_prior — RELION vs recovar.

Tests that find_current_resolution produces exactly the same shell index
as recovar's resolution_from_data_vs_prior. Both default to RELION's
--minres_map of 5 (relion/src/ml_optimiser.cpp:1298, floor at :6819).

Exact parity required: integer shell index must match exactly.
"""

import numpy as np
import pytest
from relax.relion_bind._relion_bind_core import find_current_resolution

from relax.reconstruction.regularization_relion import resolution_from_data_vs_prior


class TestM5Parity:
    """Exact parity between RELION binding and recovar."""

    def test_monotone_decay(self):
        """data_vs_prior decays monotonically — both should find same cutoff."""
        dvp = np.array([999.0, 50.0, 20.0, 10.0, 5.0, 2.0, 0.8, 0.3, 0.1])
        relion = find_current_resolution(dvp)
        recovar = resolution_from_data_vs_prior(dvp)
        assert relion == recovar, f"RELION={relion}, recovar={recovar}"

    def test_all_above_one(self):
        """All shells above 1.0 — should return last shell."""
        dvp = np.array([999.0, 10.0, 5.0, 3.0, 2.0, 1.5, 1.1])
        relion = find_current_resolution(dvp)
        recovar = resolution_from_data_vs_prior(dvp)
        assert relion == recovar, f"RELION={relion}, recovar={recovar}"

    def test_immediate_drop(self):
        """Drops below 1.0 at shell 1."""
        dvp = np.array([999.0, 0.5, 0.3, 0.1])
        relion = find_current_resolution(dvp)
        recovar = resolution_from_data_vs_prior(dvp)
        assert relion == recovar, f"RELION={relion}, recovar={recovar}"

    def test_realistic_curve(self):
        """Realistic data_vs_prior curve from refinement."""
        n_shells = 65
        dvp = np.zeros(n_shells)
        dvp[0] = 999.0
        for i in range(1, n_shells):
            dvp[i] = 50.0 * np.exp(-0.15 * i)
        relion = find_current_resolution(dvp)
        recovar = resolution_from_data_vs_prior(dvp)
        assert relion == recovar, f"RELION={relion}, recovar={recovar}"

    @pytest.mark.parametrize("seed", range(10))
    def test_random_curves(self, seed):
        """Random data_vs_prior curves — parity must hold for all."""
        rng = np.random.default_rng(seed)
        n_shells = rng.integers(10, 65)
        dvp = np.zeros(n_shells)
        dvp[0] = 999.0
        scale = rng.uniform(10, 100)
        rate = rng.uniform(0.05, 0.5)
        for i in range(1, n_shells):
            dvp[i] = scale * np.exp(-rate * i) + rng.normal(0, 0.1)
        dvp = np.maximum(dvp, 0.0)

        relion = find_current_resolution(dvp)
        recovar = resolution_from_data_vs_prior(dvp)
        assert relion == recovar, f"seed={seed}: RELION={relion}, recovar={recovar}"

    def test_explicit_minres_map(self):
        """A non-default minres_map is honoured identically by both sides."""
        dvp = np.array([999.0, 50.0, 0.5, 0.3, 0.1, 0.05, 0.01, 0.0, 0.0])
        for minres_map in (0, 3, 7):
            relion = find_current_resolution(dvp, minres_map=minres_map)
            recovar = resolution_from_data_vs_prior(dvp, minres_map=minres_map)
            assert relion == recovar == max(1, minres_map), f"minres_map={minres_map}: RELION={relion}, recovar={recovar}"
