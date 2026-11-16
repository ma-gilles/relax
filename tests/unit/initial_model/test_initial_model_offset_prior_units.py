"""relax initial_model --offset_prior_units: VDAM's coarse pdf_offset in RELION's mixed units or in one unit."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.commands import initial_model
from relax.steps.optics_shapes import MultiShapeDataset
from relax.vdam import estep_setup, native_options, native_sampling

pytestmark = pytest.mark.unit


def _frozen_plan():
    """The sampling plan of the frozen RELION capture in ``test_offset_prior_recovery``: one coarse parent at
    (0, -6) px with two fine children."""
    plan = native_sampling.NativeSamplingPlan(
        rotations=np.zeros((1, 3, 3), np.float32),
        translations=np.asarray([[0, -6.5], [0, -5.5]], np.float32),
        random_perturbation=0,
        coarse_translations=np.asarray([[99, 0]], np.float32),
        coarse_prior_translations=np.asarray([[0, -6]], np.float32),
        healpix_order=1,
        oversampling=1,
        offset_range_px=6.0,
        offset_step_px=2.0,
        offset_range_angstrom=6.0,
        offset_step_angstrom=2.0,
    )
    fields = vars(plan).copy()
    fields["translation_parent"] = np.asarray([0, 0], np.int64)
    fields["coarse_base_translations"] = np.asarray([[99, 0]], np.float64)
    return SimpleNamespace(**fields)


def _estep_config(dataset, units):
    """The E-step config of the frozen plan for a particle at (4.2, -14.1) px, sigma 10 A."""
    return estep_setup.initial_model_estep_config(
        dataset,
        native_options.NativeInitialModelOptions(fn_img="particles.star", oversampling=1, offset_prior_units=units),
        np.ones(5, np.float32),
        _frozen_plan(),
        np.asarray([[4.2, -14.1]], np.float32),
        sigma_offset_angstrom=10,
        pass1_healpix_order=1,
        max_significants=-1,
    )


def test_consistent_prior_is_a_gaussian_of_sigma_on_the_whole_offset():
    rng = np.random.default_rng(3)
    translations = rng.uniform(-6, 6, size=(21, 2)).astype(np.float32)
    old_offsets = np.round(rng.uniform(-9, 9, size=(5, 2))).astype(np.float32)
    voxel, sigma = 6.0, 10.0

    prior = native_sampling.sampling_translation_log_prior(
        translations, voxel_size=voxel, sigma_angstrom=sigma, old_offsets=old_offsets, consistent_units=True
    )

    whole_offset_angstrom = voxel * (old_offsets[:, None, :].astype(np.float64) + translations[None])
    assert_matches(prior, -np.sum(whole_offset_angstrom**2, axis=-1) / (2 * sigma**2))


def test_relion_and_consistent_priors_agree_at_one_angstrom_per_pixel():
    translations = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, -4.0]], np.float32)
    old_offsets = np.asarray([[3.0, -1.0]], np.float32)

    priors = [
        native_sampling.sampling_translation_log_prior(
            translations, voxel_size=1.0, sigma_angstrom=5.0, old_offsets=old_offsets, consistent_units=units
        )
        for units in (False, True)
    ]

    assert_matches(priors[0], priors[1])


def test_estep_config_scores_the_selected_prior_and_keeps_the_statistic_centres():
    dataset = SimpleNamespace(voxel_size=1.6375, n_images=1, image_shape=(8, 8))

    relion = _estep_config(dataset, "relion")
    consistent = _estep_config(dataset, "consistent")

    # RELION row114 capture (test_offset_prior_recovery): |(4, -14) + 1.6375 (0, -6)|^2 1.6375^2 / 200.
    assert_matches(relion.sampling.coarse_translation_log_prior, [[-7.8247542]])
    # The whole offset (4, -20) px: 416 * 1.6375^2 / 200.
    expected = np.asarray([[-416 * 1.6375**2 / 200]], np.float32)
    assert_matches(consistent.sampling.coarse_translation_log_prior, expected)
    assert_matches(consistent.engine_kwargs["translation_log_prior"], expected[:, [0, 0]])
    # Pre-shifts and the sigma-offset statistic centres do not depend on the option.
    assert_matches(consistent.engine_kwargs["image_pre_shifts"], relion.engine_kwargs["image_pre_shifts"])
    assert_matches(
        consistent.engine_kwargs["translation_prior_centers"], relion.engine_kwargs["translation_prior_centers"]
    )


def test_several_image_shapes_rebuild_the_selected_prior():
    shapes = [
        SimpleNamespace(image_shape=(8, 8), volume_shape=(8,) * 3, voxel_size=pixel, n_units=1)
        for pixel in (1.6375, 2.0)
    ]
    dataset = MultiShapeDataset(shapes, [np.array([0]), np.array([1])])

    assert _estep_config(dataset, "consistent").sampling.multi_shape_translations.consistent_units is True
    assert _estep_config(dataset, "relion").sampling.multi_shape_translations.consistent_units is False


def test_command_resolves_offset_prior_units():
    default = initial_model._native_options_dict(initial_model.make_parser().parse_args(["--i", "particles.star"]))
    chosen = initial_model._native_options_dict(
        initial_model.make_parser().parse_args(["--i", "particles.star", "--offset_prior_units", "consistent"])
    )

    assert default["offset_prior_units"] == "relion"
    assert native_options.NativeInitialModelOptions(**chosen).offset_prior_units == "consistent"
    with pytest.raises(SystemExit):
        initial_model.make_parser().parse_args(["--i", "particles.star", "--offset_prior_units", "angstrom"])


def test_validate_run_refuses_consistent_units_where_they_do_not_apply():
    consistent = native_options.NativeInitialModelOptions(fn_img="particles.star", offset_prior_units="consistent")
    consistent.validate_run()

    with pytest.raises(NotImplementedError, match="subtomogram 3-D offset prior"):
        replace(consistent, fn_tomograms="tomograms.star").validate_run()
    with pytest.raises(ValueError, match="cannot replay RELION's next iteration"):
        replace(
            consistent, diagnostic_continue_optimiser="run_it010_optimiser.star", diagnostic_stop_after_iteration=11
        ).validate_run()
    with pytest.raises(ValueError, match="relion or consistent"):
        replace(consistent, offset_prior_units="angstrom").validate_run()
