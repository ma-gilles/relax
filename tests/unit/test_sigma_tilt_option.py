"""RELION's ``--sigma_tilt`` as a relax option: parsing, the routes that take it, the run state that carries it
and each half's prior centres (relax#84). The selection itself is tests/unit/test_tilt_prior_selection.py."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from relax.refinement import command_options
from relax.refinement.refinement_options import LocalSearchOptions
from relax.refinement.refinement_state import RefinementState, refine_angular_sampling
from relax.steps import half_inputs


def _args(*extra):
    return command_options.parse_refinement_args(["--data_dir", "data", "--output", "out", *extra])


@pytest.mark.unit
def test_sigma_tilt_is_parsed_in_degrees_and_off_by_default():
    assert _args().sigma_tilt is None
    args = _args("--sigma_tilt", "10")
    command_options.validate_sigma_ang(args)
    command_options.validate_sigma_tilt(args)
    assert args.sigma_tilt == 10.0
    assert command_options.resolve_local_search(args).sigma_tilt_deg == 10.0


@pytest.mark.unit
@pytest.mark.parametrize("extra, message", [
    (("--sigma_tilt", "0"), "must be positive"),
    (("--sigma_tilt", "-3"), "must be positive"),
    (("--sigma_tilt", "10", "--skip_align"), "no effect with --skip_align"),
])  # fmt: skip
def test_sigma_tilt_refusals(extra, message):
    with pytest.raises(SystemExit, match=message):
        command_options.validate_sigma_tilt(_args(*extra))


@pytest.mark.unit
def test_sigma_ang_wins_over_sigma_tilt_as_in_relion(caplog):
    """parseInitial reads --sigma_tilt only when --sigma_ang is not positive (ml_optimiser.cpp:739-752)."""

    args = _args("--sigma_tilt", "10", "--sigma_ang", "5")
    with caplog.at_level(logging.WARNING):
        command_options.validate_sigma_ang(args)
        command_options.validate_sigma_tilt(args)
    assert args.sigma_tilt is None and args.sigma_ang == 5.0
    assert "--sigma_tilt 10 is ignored" in caplog.text


@pytest.mark.unit
def test_sigma_tilt_routes():
    args = _args("--sigma_tilt", "10")
    command_options.validate_sigma_tilt_route(args, subtomograms=True)  # subtomogram auto-refine
    with pytest.raises(SystemExit, match="subtomogram particles .* only"):
        command_options.validate_sigma_tilt_route(args, subtomograms=False)
    with pytest.raises(SystemExit, match="subtomogram auto-refine only"):
        command_options.validate_sigma_tilt_route(_args("--sigma_tilt", "10", "--n_classes", "2"), subtomograms=True)
    command_options.validate_sigma_tilt_route(_args(), subtomograms=False)  # no option: nothing to refuse


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
