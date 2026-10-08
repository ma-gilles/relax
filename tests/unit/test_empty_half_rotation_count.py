"""An empty half's rotation sums must be on the grid its occupied twin sums over (Class3D and single-half runs)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from relax.helpers.convergence import RefinementState, _direction_prior_healpix_order_for_scoring
from relax.refinement.expectation import empty_half_rotation_count
from relax.refinement.local_sampling import LocalSearchSettings
from relax.sampling import rotation_grid_size

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("oversampling", [0, 1])
@pytest.mark.parametrize("symmetry", ["C1", "C4"])
def test_local_empty_half_uses_the_direction_prior_grid(oversampling, symmetry):
    # A coarse order-4 local search: the fine grid is order 4 + oversampling, the parent grid order 4.
    search = LocalSearchSettings(healpix_order=4 + oversampling, oversampling_order=oversampling, sigma_rot=0.03, sigma_psi=0.03, symmetry="C1")
    sampling = SimpleNamespace(search=search)
    state = RefinementState(healpix_order=4, adaptive_oversampling=oversampling, do_local_search=True)
    prior_order = _direction_prior_healpix_order_for_scoring(
        state, use_local=True, grid_healpix_order=3, local_search_order=search.healpix_order
    )
    count = empty_half_rotation_count(sampling, 123, use_local=True, symmetry=symmetry)
    assert count == rotation_grid_size(prior_order, symmetry=symmetry)
    if oversampling:
        # The fine grid it was sized to before (Class3D crash, job 15086265).
        assert count != rotation_grid_size(search.healpix_order, symmetry=symmetry)


def test_global_empty_half_uses_the_trial_grid():
    assert empty_half_rotation_count(SimpleNamespace(), 4608, use_local=False, symmetry="C1") == 4608
