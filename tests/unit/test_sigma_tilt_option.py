"""RELION's ``--sigma_tilt`` as a relax option: parsing, the routes that take it, the run state that carries it
and each half's prior centres (relax#84). The selection itself is tests/unit/test_tilt_prior_selection.py."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from relax.commands import refinement_arguments
from relax.refinement.refinement_options import LocalSearchOptions
from relax.refinement.refinement_state import RefinementState, refine_angular_sampling
from relax.steps import half_inputs


def _args(*extra):
    return refinement_arguments.parse_refinement_args(["--data_dir", "data", "--output", "out", *extra])


@pytest.mark.unit
def test_sigma_tilt_is_parsed_in_degrees_and_off_by_default():
    assert _args().sigma_tilt is None
    args = _args("--sigma_tilt", "10")
    refinement_arguments.validate_sigma_ang(args)
    refinement_arguments.validate_sigma_tilt(args)
    assert args.sigma_tilt == 10.0
    assert refinement_arguments.resolve_local_search(args).sigma_tilt_deg == 10.0


@pytest.mark.unit
@pytest.mark.parametrize("extra, message", [
    (("--sigma_tilt", "0"), "must be positive"),
    (("--sigma_tilt", "-3"), "must be positive"),
    (("--sigma_tilt", "10", "--skip_align"), "no effect with --skip_align"),
])  # fmt: skip
def test_sigma_tilt_refusals(extra, message):
    with pytest.raises(SystemExit, match=message):
        refinement_arguments.validate_sigma_tilt(_args(*extra))


@pytest.mark.unit
def test_sigma_ang_wins_over_sigma_tilt_as_in_relion(caplog):
    """parseInitial reads --sigma_tilt only when --sigma_ang is not positive (ml_optimiser.cpp:739-752)."""

    args = _args("--sigma_tilt", "10", "--sigma_ang", "5")
    with caplog.at_level(logging.WARNING):
        refinement_arguments.validate_sigma_ang(args)
        refinement_arguments.validate_sigma_tilt(args)
    assert args.sigma_tilt is None and args.sigma_ang == 5.0
    assert "--sigma_tilt 10 is ignored" in caplog.text


@pytest.mark.unit
def test_sigma_tilt_routes():
    args = _args("--sigma_tilt", "10")
    classes = _args("--sigma_tilt", "10", "--n_classes", "2")
    refinement_arguments.validate_sigma_tilt_route(args, subtomograms=True)  # subtomogram auto-refine
    refinement_arguments.validate_sigma_tilt_route(classes, subtomograms=True)  # subtomogram Class3D (relax#85)
    for single_particle in (args, classes):
        with pytest.raises(SystemExit, match="subtomogram particles .* only"):
            refinement_arguments.validate_sigma_tilt_route(single_particle, subtomograms=False)
    refinement_arguments.validate_sigma_tilt_route(_args(), subtomograms=False)  # no option: nothing to refuse


def _state(**overrides):
    values = dict(
        iteration=1, healpix_order=2, adaptive_oversampling=1, translation_range=5.0, translation_step=1.0,
        auto_local_healpix_order=4, auto_sampling=True, voxel_size_angstrom=4.0,
    )  # fmt: skip
    return RefinementState(**{**values, **overrides})


TILT = LocalSearchOptions(sigma_tilt_deg=10.0)


@pytest.mark.unit
def test_tilt_prior_holds_until_the_local_search_order():
    """updateAngularSampling keeps the widths below --auto_local_healpix_order and sets all three to twice
    the step at it (ml_optimiser.cpp:9923-9931): the tilt prior then gives way to the local search."""

    state = _state()
    assert not state.do_local_search and TILT.tilt_prior_deg(state.do_local_search) == 10.0
    order3 = refine_angular_sampling(state)
    assert order3.healpix_order == 3 and TILT.tilt_prior_deg(order3.do_local_search) == 10.0
    assert order3.sigma_rot == 0.0 and order3.sigma_psi == 0.0
    order4 = refine_angular_sampling(order3)
    assert order4.healpix_order == 4 and order4.do_local_search and TILT.tilt_prior_deg(order4.do_local_search) == 0.0
    assert order4.sigma_rot > 0.0 and order4.sigma_rot == order4.sigma_psi


@pytest.mark.unit
def test_no_tilt_prior_without_the_option_or_under_a_local_search():
    assert LocalSearchOptions().tilt_prior_deg(False) == 0.0
    assert TILT.tilt_prior_deg(_state(healpix_order=4).do_local_search) == 0.0  # a start at the local order
    # --sigma_ang sets all three widths; the tilt width is not read (parseInitial).
    assert LocalSearchOptions(sigma_tilt_deg=10.0, sigma_ang_deg=5.0).tilt_prior_deg(False) == 0.0
    # Class3D never switches from the order: the tilt prior stays for the whole run.
    class3d = refine_angular_sampling(_state(auto_sampling=False, healpix_order=3))
    assert TILT.tilt_prior_deg(class3d.do_local_search) == 10.0


@pytest.mark.unit
def test_the_option_adds_nothing_to_the_run_state():
    """The width is an option: runs without it write the state, checkpoints and run files they always did."""

    import dataclasses

    assert not any("tilt" in field.name for field in dataclasses.fields(RefinementState))


@pytest.mark.unit
def test_tilt_prior_centres_are_the_given_priors_else_the_current_tilt():
    """rlnAngleTiltPrior where the STAR has it, the particle's current tilt where it is absent (RELION's 999);
    rot and psi priors are not read (their widths are 0)."""

    eulers = np.array([[10.0, 33.25, 5.0], [20.0, 120.5, 6.0], [30.0, 71.0, 7.0]], dtype=np.float32)
    half = SimpleNamespace(dataset=SimpleNamespace(n_units=3), rotation_eulers=eulers)
    priors = np.array([[1.0, 90.0, 2.0], [np.nan, np.nan, np.nan], [3.0, 37.3, np.nan]])
    prior = half_inputs.tilt_prior_for_half(half, priors, 10.0)
    assert prior.sigma_deg == 10.0 and prior.centre_deg.dtype == np.float64
    np.testing.assert_array_equal(prior.centre_deg, [90.0, 120.5, 37.3])
    np.testing.assert_array_equal(half_inputs.tilt_prior_for_half(half, None, 10.0).centre_deg, [33.25, 120.5, 71.0])
    no_poses = SimpleNamespace(dataset=SimpleNamespace(n_units=2), rotation_eulers=None)
    np.testing.assert_array_equal(half_inputs.tilt_prior_for_half(no_poses, None, 10.0).centre_deg, [0.0, 0.0])
    assert half_inputs.tilt_prior_for_half(half, priors, 0.0) is None


@pytest.mark.unit
def test_class3d_start_reads_the_tilt_priors_and_the_input_tilts_they_fall_back_on():
    """A Class3D start under --sigma_tilt centres each particle's prior on its rlnAngleTiltPrior and, where the
    STAR has none (RELION's 999), on its input tilt (ml_optimiser.cpp:5970-5992): the loader returns both, in
    the order of the run's one all-data accumulator."""

    import pandas as pd

    from relax.star import input_poses

    rng = np.random.default_rng(4)
    n = 5
    tilt = rng.uniform(0.0, 180.0, n)
    particles = pd.DataFrame(
        {
            "rlnAngleRot": rng.uniform(-180.0, 180.0, n),
            "rlnAngleTilt": tilt,
            "rlnAnglePsi": rng.uniform(-180.0, 180.0, n),
            "rlnAngleTiltPrior": [90.0, 999.0, 85.5, 999.0, 90.0],
            "rlnOriginXAngst": np.zeros(n),
            "rlnOriginYAngst": np.zeros(n),
        }
    )
    rows = rng.permutation(n)
    start = input_poses._load_input_star_class3d_translations(particles, rows, voxel_size=2.0, with_orientations=True)
    priors, empty = start["angle_priors"]
    assert empty.shape == (0, 3)
    np.testing.assert_array_equal(np.isnan(priors[:, 1]), np.array([False, True, False, True, False])[rows])
    assert np.all(np.isnan(priors[:, [0, 2]]))  # no rot or psi prior column
    half = SimpleNamespace(dataset=SimpleNamespace(n_units=n), rotation_eulers=start["previous_best_rotation_eulers"][0])
    centre = half_inputs.tilt_prior_for_half(half, priors, 10.0).centre_deg
    np.testing.assert_array_equal(
        centre, np.where(np.isin(np.arange(n), [1, 3]), tilt.astype(np.float32), particles["rlnAngleTiltPrior"])[rows]
    )
    # A global Class3D start without the option keeps reading no orientations and no priors.
    searched = input_poses._load_input_star_class3d_translations(particles, rows, voxel_size=2.0)
    assert searched["previous_best_rotation_eulers"] == [None, None] and searched["angle_priors"] is None
    assert input_poses._input_star_angle_priors(particles.drop(columns="rlnAngleTiltPrior")) is None
