"""Initial coarse-grid planning exposes sampling, replay, and logging inputs.

``build_initial_coarse_grids`` builds the first exhaustive grid (sealed capture,
caller translation table, or RELION translation grid) and
final-pass grid preparation is owned by ``prepare_final_sampling``.
"""

from __future__ import annotations

import inspect
import logging

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import matches

import relax.refinement.finalization as finalization
import relax.refinement.iteration_loop as iteration_loop
import relax.refinement.iteration_planning as iteration_planning
import relax.sampling as sampling_module
from relax.dense.scoring_policy import _dense_global_scoring_dtype
from relax.parity import relion_replay_source
from relax.parity.relion_replay import _sealed_sampling_base_grids
from relax.sampling import _translation_grid_for_class_count

pytestmark = pytest.mark.unit

SEALED = {
    "directions_ipix": np.asarray([7, 19, 503], dtype=np.int64),
    "rot_angles_deg": np.asarray([10.0, 20.0, 30.0]),
    "tilt_angles_deg": np.asarray([40.0, 50.0, 60.0]),
    "psi_angles_deg": np.asarray([0.0, 90.0]),
    "translations_x_angstrom": np.asarray([-2.0, 0.0, 2.0]),
    "translations_y_angstrom": np.asarray([0.0, 1.0, 0.0]),
    "healpix_order_original": 3,
}


def _fake_rotation_grid(order, dtype=np.float32, *, symmetry='C1'):
    n = 3 * (int(order) + 1)
    rotations = np.repeat(np.eye(3, dtype=dtype)[None], n, axis=0)
    return sampling_module.RotationGrid(rotations=rotations, rotation_eulers=np.arange(3 * n, dtype=dtype).reshape(n, 3), healpix_order=order, symmetry=symmetry)


def _same(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and matches(x, y)


def _initial_grids(**overrides):
    kwargs = dict(
        healpix_order=3,
        translations=None,
        init_healpix_order=3,
        init_translation_range=4.25,
        init_translation_step=1.416667,
        n_classes=1,
        voxel_size=2.0,
    )
    kwargs.update(overrides)
    sealed_sampling_state = kwargs.pop("sealed_sampling_state", None)
    initialized_healpix_order = kwargs.pop("init_healpix_order")
    translation_range = kwargs.pop("init_translation_range")
    translation_step = kwargs.pop("init_translation_step")
    if sealed_sampling_state is not None:
        return relion_replay_source.build_sealed_initial_coarse_grids(
            sealed_sampling_state,
            initialized_healpix_order=initialized_healpix_order,
            voxel_size=kwargs["voxel_size"],
            symmetry=kwargs.get("symmetry", "C1"),
            log=logging.getLogger("test_initial_coarse_grid_owner"),
        )
    return iteration_planning.build_initial_coarse_grids(
        kwargs.pop("healpix_order"),
        kwargs.pop("translations"),
        translation_range=translation_range,
        translation_step=translation_step,
        dtype=_dense_global_scoring_dtype(),
        **kwargs,
    )


@pytest.mark.parametrize("n_classes", [1, 4])
def test_base_translation_grid_is_host_float64_in_source_units(n_classes):
    grid = sampling_module.relion_base_translation_grid(4.25, 1.416667, n_classes=n_classes, voxel_size=2.0)
    expected = _translation_grid_for_class_count(4.25, 1.416667, n_classes=n_classes, source_units_per_pixel=2.0)
    assert isinstance(grid, np.ndarray) and grid.dtype == np.float64
    assert _same(grid, expected.astype(np.float64))


def test_base_translation_grid_falls_back_to_pixel_units_without_a_voxel_size():
    fallback = sampling_module.relion_base_translation_grid(5.0, 1.0, n_classes=1, voxel_size=0.0)
    pixel_units = sampling_module.relion_base_translation_grid(5.0, 1.0, n_classes=1, voxel_size=1.0)
    assert _same(fallback, pixel_units)


@pytest.mark.parametrize("symmetry", ["C1", "C4"])
def test_sealed_state_supplies_the_initial_grid(caplog, symmetry):
    caplog.set_level(logging.INFO, logger="test_initial_coarse_grid_owner")
    grids = _initial_grids(sealed_sampling_state=SEALED, symmetry=symmetry)
    rotations, eulers, translations = _sealed_sampling_base_grids(SEALED, voxel_size_angstrom=2.0, dtype=np.float32)
    assert _same(grids.rotation_grid.rotations, rotations) and _same(grids.rotation_grid.rotation_eulers, eulers)
    assert isinstance(grids.translations, jnp.ndarray) and _same(grids.translations, translations)
    assert grids.base_translations.dtype == np.float64
    assert _same(grids.base_translations, np.asarray(translations, dtype=np.float64))
    assert grids.rotation_grid.healpix_order == 3 and isinstance(grids.rotation_grid.healpix_order, int)
    assert grids.rotation_grid.symmetry == symmetry
    assert "Frozen-boundary v3 directly materialized 6 Euler rows and 3 translations" in caplog.text


def test_sealed_state_must_sit_at_the_initialized_order():
    with pytest.raises(ValueError, match="sealed=3 init=2"):
        _initial_grids(sealed_sampling_state=SEALED, init_healpix_order=2)


def test_relion_translation_grid_pairs_with_the_canonical_rotation_grid(monkeypatch):
    monkeypatch.setattr(sampling_module, "relion_scoring_rotation_grid", _fake_rotation_grid)
    grids = _initial_grids(n_classes=4)
    _rotation_grid_rotations = _fake_rotation_grid(3, dtype=_dense_global_scoring_dtype())
    rotations = _rotation_grid_rotations.rotations
    eulers = _rotation_grid_rotations.rotation_eulers
    assert _same(grids.rotation_grid.rotations, rotations) and _same(grids.rotation_grid.rotation_eulers, eulers)
    expected = sampling_module.relion_base_translation_grid(4.25, 1.416667, n_classes=4, voxel_size=2.0)
    assert _same(grids.base_translations, expected)
    assert isinstance(grids.translations, jnp.ndarray)
    assert _same(grids.translations, jnp.asarray(expected, dtype=_dense_global_scoring_dtype()))
    assert grids.rotation_grid.healpix_order == 3


def test_caller_translation_table_is_kept_as_the_base_grid(monkeypatch):
    monkeypatch.setattr(sampling_module, "relion_scoring_rotation_grid", _fake_rotation_grid)
    table = np.asarray([[0.5, -1.0], [0.0, 0.0]], dtype=np.float32)
    grids = _initial_grids(translations=table)
    assert grids.base_translations.dtype == np.float64 and _same(grids.base_translations, table.astype(np.float64))
    assert _same(grids.translations, jnp.asarray(table, dtype=_dense_global_scoring_dtype()))


def test_controller_materializes_explicit_coarse_grid_variants():
    assert tuple(inspect.signature(iteration_planning.build_initial_coarse_grids).parameters) == (
        "healpix_order",
        "translations",
        "translation_range",
        "translation_step",
        "n_classes",
        "voxel_size",
        "symmetry",
        "dtype",
    )
    assert tuple(
        inspect.signature(relion_replay_source.build_sealed_initial_coarse_grids).parameters
    ) == (
        "sealed_sampling_state",
        "initialized_healpix_order",
        "voxel_size",
        "symmetry",
        "log",
    )
    for name in ("InitialGridSampling", "_sealed_sampling_base_grids", "_translation_grid_for_class_count"):
        assert not hasattr(iteration_loop, name)


# A schema-v3 sealed sampling state at HEALPix order 2: three directions, two psi angles, three translations.
TINY_SEALED = {
    "consumer_relion_iteration": 1,
    "directions_ipix": np.asarray([7, 19, 103], dtype=np.int64),
    "rot_angles_deg": np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
    "tilt_angles_deg": np.asarray([40.0, 50.0, 60.0], dtype=np.float64),
    "psi_angles_deg": np.asarray([0.0, 90.0], dtype=np.float64),
    "translations_x_angstrom": np.asarray([-1.0, 0.0, 1.0], dtype=np.float64),
    "translations_y_angstrom": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    "healpix_order_original": 2, "psi_step_deg": 90.0,
    "offset_range_angstrom": 1.0, "offset_step_angstrom": 1.0,
    "perturbation_factor": 0.5, "random_perturbation": 0.125,
    "sigma_rot_deg": 0.0, "sigma_psi_deg": 0.0,
    "coarse_size": 4, "current_size": 6,
}


@pytest.mark.parametrize("sealed", [False, True])
def test_controller_builds_one_initial_grid_and_one_final_sampling(monkeypatch, sealed):
    """A run builds its initial coarse grid once, through the variant's owner, and its final pass prepares
    its sampling once."""
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.parity import relion_replay_source
    from relax.parity.relion_replay_source import RelionReplay

    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "build_initial_coarse_grids", "initial")
    trace.wrap(relion_replay_source, "build_sealed_initial_coarse_grids", "sealed")
    trace.wrap(finalization, "run_final_all_data", "final")
    trace.wrap(finalization, "prepare_final_sampling", "final_sampling")
    extra = {"relion_replay": RelionReplay(sealed_sampling_state=TINY_SEALED)} if sealed else {}
    run_tiny_refinement(monkeypatch, max_iter=1, **extra)

    assert trace.labels() == ["sealed" if sealed else "initial", "final", "final_sampling"]
    assert trace.calls("final_sampling")[0].inside == ("final",)
