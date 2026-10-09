"""Final sampling preserves replay admission, RNG draws and rounding boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.run_options import stand_in

from relax.parity import relion_replay_source
from relax.refinement import final_sampling
from relax.refinement.ports import InputSource

pytestmark = pytest.mark.unit


def test_image_geometry_preserves_host_precision_for_mask_radius():
    pixel_size = np.float32(2.125)
    geometry = final_sampling.ImageGeometry(image_shape=(128, 128), pixel_size_angstrom=pixel_size)
    radius = 200.0 / (2.0 * geometry.pixel_size_angstrom)
    assert isinstance(geometry.pixel_size_angstrom, float)
    assert_matches(radius, 200.0 / (2.0 * float(pixel_size)), rtol=1e-12)
    float32_radius = float(np.float32(200.0) / (np.float32(2.0) * pixel_size))
    assert not np.isclose(radius, float32_radius, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("pixel_size", [0.0, -1.0, np.inf, np.nan])
def test_image_geometry_rejects_invalid_physical_spacing(pixel_size):
    with pytest.raises(ValueError, match="finite and positive"):
        final_sampling.ImageGeometry(image_shape=(128, 128), pixel_size_angstrom=pixel_size)


@pytest.fixture
def preparation(monkeypatch):
    rotations = np.eye(3, dtype=np.float32)[None]
    eulers = np.zeros((1, 3), dtype=np.float32)
    # A value which changes on float32 conversion catches moving rounding
    # from before perturbation to after it.
    host_grid = np.array([[1.0 / 7.0, -1.0 / 3.0]], dtype=np.float64)
    events = []
    monkeypatch.setattr(final_sampling, "_exhaustive_grid_order_for_state", lambda state: 3)
    monkeypatch.setattr(final_sampling, "_native_final_perturbation_healpix_order", lambda state, order: 5)
    monkeypatch.setattr(final_sampling.sampling, "relion_base_translation_grid", lambda *a, **kw: host_grid)
    monkeypatch.setattr(final_sampling.sampling, "relion_mstep_source_eulers", lambda eulers, *a, **kw: eulers)
    monkeypatch.setattr(final_sampling.sampling, "relion_angular_sampling_deg", lambda *a, **kw: 7.5)

    def perturb(**kwargs):
        events.append(("perturb", kwargs))
        return final_sampling.sampling.TrialGrid(
            rotations, kwargs["rotation_eulers"], rotations, kwargs["base_translations"],
        )

    def advance(previous, **kwargs):
        events.append(("advance", (previous, kwargs)))
        return 0.125, kwargs["perturb_seed"]

    monkeypatch.setattr(final_sampling.sampling, "perturbed_trial_grid", perturb)
    monkeypatch.setattr(final_sampling.sampling, "advance_relion_perturbation_for_iteration", advance)
    inputs = dict(
        state=SimpleNamespace(translation_range=4.25, translation_step=1.416667),
        image_geometry=final_sampling.ImageGeometry(image_shape=(128, 128), pixel_size_angstrom=1.5),
        previous_rotation_grid=final_sampling.sampling.RotationGrid(
            rotations=rotations, rotation_eulers=eulers, healpix_order=3, symmetry="C1",
        ),
        options=stand_in.options(schedule=stand_in.schedule(init_relion_iteration=10)),
        numbered_iteration_count=2,
        source=InputSource(),
        previous_perturbation=0.25,
        rng=object(),
        dtype=np.float32,
    )
    return inputs, events, host_grid


def _replay_source(inputs, *, star_directory, **replay):
    """The final pass's input source when the numbered STAR replay's directory, still live at the end of the
    numbered iterations, is ``star_directory`` (None: the replay ended or never ran); ``replay`` are further
    ``RelionReplay`` fields."""
    inputs["source"] = relion_replay_source.RelionReplaySource(
        relion_replay_source.RelionReplay(perturb_replay_relion_dir=star_directory, **replay), inputs["options"],
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("perturb_factor", [0.0, 0.5])
def test_final_grids_are_ready_for_scoring_at_the_existing_rounding_boundary(preparation, dtype, perturb_factor):
    inputs, events, host_grid = preparation
    inputs.update(
        dtype=dtype,
        options=replace(inputs["options"], parity=stand_in.parity(perturb_factor=perturb_factor, perturb_seed=7)),
    )
    result = final_sampling.prepare_final_sampling(**inputs)
    assert result.base_rotations is inputs["previous_rotation_grid"].rotations
    assert isinstance(result.base_translations, jnp.ndarray)
    assert result.base_translations.dtype == dtype
    assert_matches(result.base_translations, host_grid.astype(dtype))
    assert result.settings.relion_iteration == 13
    assert result.settings.grid_order == 3
    assert result.settings.perturbation_order == 5
    if perturb_factor:
        assert [name for name, _ in events] == ["advance", "perturb"]
        previous, advance = events[0][1]
        assert_matches(previous, inputs["previous_perturbation"])
        assert advance["rng"] is inputs["rng"]
        assert advance["relion_iteration"] == 13
        operands = events[1][1]
        assert operands["base_translations"] is result.base_translations
        assert operands["rotation_eulers"].dtype == dtype
        assert_matches(result.settings.perturbation, 0.125)
    else:
        assert not events
        assert result.grid.translations is result.base_translations
        assert result.settings.perturbation is None


def test_default_scoring_precision_is_float32(preparation):
    inputs, _, _ = preparation
    result = final_sampling.prepare_final_sampling(**inputs)
    assert result.base_translations.dtype == np.float32


def test_changed_coarse_order_rebuilds_with_the_requested_precision_and_symmetry(preparation, monkeypatch):
    inputs, _, _ = preparation
    inputs.update(
        previous_rotation_grid=replace(inputs["previous_rotation_grid"], healpix_order=2, symmetry="D2"),
        dtype=np.float64,
    )
    rebuilt = np.eye(3, dtype=np.float64)[None]
    calls = []

    def build(order, **kwargs):
        calls.append((order, kwargs))
        return final_sampling.sampling.RotationGrid(rotations=rebuilt, rotation_eulers=np.zeros((1, 3), dtype=np.float64), healpix_order=order, symmetry=kwargs.get("symmetry", "C1"))

    monkeypatch.setattr(final_sampling.sampling, "relion_scoring_rotation_grid", build)
    result = final_sampling.prepare_final_sampling(**inputs)
    assert result.base_rotations is rebuilt
    assert calls == [(3, dict(dtype=np.float64, symmetry="D2"))]


@pytest.mark.parametrize("source, replay_iteration", [("final-numbered", 13), ("final", 13), ("last-numbered", 12)])
def test_replay_uses_its_sampling_iteration_without_advancing_native_rng(preparation, monkeypatch, tmp_path, source, replay_iteration):
    inputs, events, _ = preparation
    _replay_source(inputs, star_directory=str(tmp_path))
    star = str(tmp_path / "selected_sampling.star")
    monkeypatch.setattr(relion_replay_source, "select_final_sampling_star", lambda *a, **kw: (star, source, []))
    monkeypatch.setattr(relion_replay_source, "read_relion_sampling_metadata", lambda path: dict(
        random_perturbation=0.0, perturbation_factor=0.5,
        healpix_order=7, offset_range=6.0, offset_step=3.0,
    ))
    resolutions = []

    def resolve(**kwargs):
        resolutions.append(kwargs)
        return kwargs["star_value"], "star"

    monkeypatch.setattr(relion_replay_source, "_resolve_replay_random_perturbation", resolve)
    result = final_sampling.prepare_final_sampling(**inputs)
    assert [name for name, _ in events] == ["perturb"]
    assert resolutions[0]["relion_iteration"] == replay_iteration
    assert result.settings.relion_iteration == 13
    assert result.settings.perturbation is not None
    assert_matches(result.settings.perturbation, 0.0)
    assert result.settings.perturbation_order == 7
    assert_matches(result.settings.translation_range, 4.0)
    assert_matches(result.settings.translation_step, 2.0)
    assert result.settings.sampling_star == star
    assert result.settings.sampling_star_source == source


@pytest.mark.parametrize("active_replay, factor, applied", [(True, 0.5, False), (False, 0.5, True), (False, 0.0, False)])
def test_missing_final_star_preserves_zero_application_and_rng_semantics(preparation, monkeypatch, tmp_path, active_replay, factor, applied):
    inputs, events, _ = preparation
    inputs["options"] = replace(inputs["options"], parity=stand_in.parity(perturb_factor=factor))
    _replay_source(
        inputs, star_directory=str(tmp_path) if active_replay else None,
        final_sampling_replay_relion_dir=str(tmp_path),
    )
    monkeypatch.setattr(relion_replay_source, "select_final_sampling_star", lambda *a, **kw: (None, None, []))
    result = final_sampling.prepare_final_sampling(**inputs)
    assert [name for name, _ in events] == (["perturb"] if applied else [])
    assert (result.settings.perturbation is not None) == applied
    assert_matches(result.settings.random_perturbation, 0.0)


def test_strict_final_replay_requires_its_files(preparation, tmp_path):
    inputs, events, _ = preparation
    _replay_source(inputs, star_directory=str(tmp_path), replay_iteration_overrides=[{}])
    with pytest.raises(RuntimeError, match="Strict RELION final all-data replay requires"):
        final_sampling.prepare_final_sampling(**inputs)
    assert not events
