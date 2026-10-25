"""A subtomogram half's orientational search (relax.steps.tomo_half): which search a half gets
(``tomo_half_search``: local, ``--sigma_tilt``, or none) and the orientations it builds once per half
(``tomo_search_rotations``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from relax.runtime.engine_tuning import EngineTuning
from relax.steps import tomo_half
from relax.steps.half_inputs import TiltPrior
from relax.steps.tomo_half import TomoSampling, TomoSearch, tomo_half_search, tomo_search_rotations

pytestmark = pytest.mark.unit

SAMPLING = TomoSampling(
    healpix_order=1,
    oversampling_order=1,
    offset_range_angst=4.0,
    offset_step_angst=2.0,
    random_perturbation=0.25,
    coarse_size=8,
    fine_size=16,
)


def _half(eulers, n_units=3):
    return SimpleNamespace(rotation_eulers=eulers, dataset=SimpleNamespace(n_units=n_units))


def test_a_local_search_takes_the_previous_angles_and_the_widths():
    eulers = np.arange(9.0).reshape(3, 3)
    prior = TiltPrior(sigma_deg=10.0, centre_deg=np.full(3, 90.0))
    search = tomo_half_search(_half(eulers), local_widths=(3.0, 4.0), tilt_prior=prior)
    # The local search wins over a tilt prior: RELION applies --sigma_tilt only below the local-search order.
    assert search.previous_eulers_deg is eulers
    assert (search.sigma_rot, search.sigma_psi, search.sigma_tilt_deg, search.prior_tilt_deg) == (3.0, 4.0, 0.0, None)


@pytest.mark.parametrize("have_angles", [True, False])
def test_a_tilt_prior_search_centres_on_the_prior_with_absent_angles_read_as_zero(have_angles):
    eulers = np.arange(9.0).reshape(3, 3) if have_angles else None
    prior = TiltPrior(sigma_deg=10.0, centre_deg=np.array([80.0, 90.0, 100.0]))
    search = tomo_half_search(_half(eulers), local_widths=None, tilt_prior=prior)
    np.testing.assert_array_equal(search.previous_eulers_deg, eulers if have_angles else np.zeros((3, 3)))
    assert (search.sigma_rot, search.sigma_psi, search.sigma_tilt_deg) == (0.0, 0.0, 10.0)
    assert search.prior_tilt_deg is prior.centre_deg


def test_no_local_search_and_no_tilt_prior_is_a_global_search():
    assert tomo_half_search(_half(np.zeros((3, 3))), local_widths=None, tilt_prior=None) is None
    assert tomo_search_rotations(None, SAMPLING, voxel_size=2.0, symmetry="C1", engine_tuning=EngineTuning()) is None


def test_the_orientations_are_built_at_the_sampling_order_and_perturbation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tomo_half, "tomo_local_rotations", lambda eulers, **kw: calls.append((eulers, kw)) or "rotations"
    )
    tuning = EngineTuning()
    search = TomoSearch(np.ones((3, 3)), sigma_tilt_deg=10.0, prior_tilt_deg=np.full(3, 90.0))
    assert tomo_search_rotations(search, SAMPLING, voxel_size=2.5, symmetry="C2", engine_tuning=tuning) == "rotations"
    ((eulers, kw),) = calls
    assert eulers is search.previous_eulers_deg
    assert kw == dict(
        sigma_rot=0.0,
        sigma_psi=0.0,
        sigma_tilt_deg=10.0,
        prior_tilt_deg=search.prior_tilt_deg,
        healpix_order=1,
        random_perturbation=0.25,
        voxel_size=2.5,
        symmetry="C2",
        engine_tuning=tuning,
    )


def test_a_local_search_builds_each_particles_orientations_in_one_union():
    eulers = np.array([[10.0, 40.0, 30.0], [200.0, 120.0, -60.0]])
    search = tomo_half_search(_half(eulers, n_units=2), local_widths=(10.0, 10.0), tilt_prior=None)
    rotations = tomo_search_rotations(search, SAMPLING, voxel_size=2.0, symmetry="C1", engine_tuning=EngineTuning())
    union = rotations.coarse_rotation_ids
    assert union.size and np.all(np.diff(union) > 0)
    assert len(rotations.unit_rotation_ids) == 2
    for ids, priors in zip(rotations.unit_rotation_ids, rotations.unit_rotation_log_priors, strict=True):
        assert ids.size == priors.size > 0 and np.all(np.diff(ids) > 0) and ids.max() < union.size
    covered = np.unique(np.concatenate(rotations.unit_rotation_ids))
    np.testing.assert_array_equal(covered, np.arange(union.size))
