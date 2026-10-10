"""RELION's local-search orientational prior widths have one owner shared by both passes."""

from __future__ import annotations

import numpy as np
import pytest

from relax.refinement.refinement_state import RefinementState, healpix_angular_step
from relax.sampling.orientation_priors import relion_local_search_sigmas

pytestmark = pytest.mark.unit


def _state(sigma_rot, sigma_psi, healpix_order, adaptive_oversampling):
    return RefinementState(
        sigma_rot=sigma_rot, sigma_psi=sigma_psi, healpix_order=healpix_order,
        adaptive_oversampling=adaptive_oversampling,
    )


def test_configured_widths_are_kept_and_psi_falls_back_to_rot():
    assert relion_local_search_sigmas(_state(0.05, 0.0, 3, 1), use_local=True) == (0.05, 0.05)
    assert relion_local_search_sigmas(_state(0.05, 0.03, 3, 1), use_local=True) == (0.05, 0.03)


@pytest.mark.parametrize("order,oversampling", [(3, 0), (4, 1), (5, 2)])
def test_unset_width_under_local_search_is_twice_the_oversampled_step(order, oversampling):
    rot, psi = relion_local_search_sigmas(_state(0.0, 0.0, order, oversampling), use_local=True)
    expected = np.sqrt(2.0 * 2.0) * np.deg2rad(healpix_angular_step(order) / (2**oversampling))
    assert rot == expected and psi == expected


def test_global_search_keeps_unset_widths():
    assert relion_local_search_sigmas(_state(0.0, 0.0, 3, 1), use_local=False) == (0.0, 0.0)
