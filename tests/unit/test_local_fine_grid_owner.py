"""The exact fine local-search grid and its M-step matrices are owned by the sampling module.

Both the regular iterations and the final all-data pass call
``_exact_local_fine_grid`` (RELION SamplingPerturbation of the materialized
fine grid plus the exact M-step rotations) and ``_local_search_mstep_rotations``
(reuse of a scoring grid's M-step matrices); the final pass sizes its parent
pass with ``relion_local_pass1_current_size`` only under adaptive oversampling.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import matches

import relax.refinement.iteration_loop as iteration_loop
import relax.sampling as sampling_module
from relax.helpers.resolution import ImageGeometry
from relax.refinement import finalization, iteration_planning, local_sampling
from relax.refinement.local_sampling import prepare_final_local_sampling, prepare_numbered_local_sampling
from relax.sampling import (
    _relion_mstep_rotations_from_eulers,
    apply_relion_rotation_perturbation_to_eulers,
    relion_angular_sampling_deg,
    rotation_grid_size,
)

pytestmark = pytest.mark.unit

ORDER = 1
N_ROT = rotation_grid_size(ORDER)
ANGULAR_SAMPLING = relion_angular_sampling_deg(ORDER, adaptive_oversampling=0)


def _canonical_eulers(order, *, symmetry='C1'):
    n = rotation_grid_size(order)
    return np.stack([np.linspace(0.0, 350.0, n), np.linspace(10.0, 170.0, n), np.linspace(5.0, 355.0, n)], axis=1) + 0.1234567


def _fake_grid(order, dtype=np.float32, *, symmetry='C1'):
    source = _canonical_eulers(order)
    return sampling_module.RotationGrid(rotations=_relion_mstep_rotations_from_eulers(source, dtype=dtype), rotation_eulers=source.astype(dtype), healpix_order=order, symmetry=symmetry)


def _same(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and matches(x, y)


@pytest.fixture(autouse=True)
def _fake_canonical_grid(monkeypatch):
    monkeypatch.setattr(sampling_module, "_get_relion_rotation_grid_eulers_float64", _canonical_eulers)
    monkeypatch.setattr(sampling_module, "relion_scoring_rotation_grid", _fake_grid)


@pytest.mark.parametrize("random_perturbation", [0.3, -0.125, 0.0])
def test_fine_grid_is_perturbed_with_exact_mstep_rotations(random_perturbation):
    rotations, eulers, mstep = sampling_module._exact_local_fine_grid(
        healpix_order=ORDER, angular_sampling_deg=ANGULAR_SAMPLING, random_perturbation=random_perturbation
    )
    _rotation_grid_ = _fake_grid(ORDER)
    _ = _rotation_grid_.rotations
    grid_eulers = _rotation_grid_.rotation_eulers
    exp_rot, exp_eulers = apply_relion_rotation_perturbation_to_eulers(grid_eulers, random_perturbation, ANGULAR_SAMPLING)
    exp_mstep, _ = apply_relion_rotation_perturbation_to_eulers(
        _canonical_eulers(ORDER), random_perturbation, ANGULAR_SAMPLING,
    )
    assert _same(rotations, exp_rot) and _same(eulers, exp_eulers) and _same(mstep, exp_mstep)
    assert rotations.dtype == np.float32 and rotations.shape == (N_ROT, 3, 3)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pass_without_perturbation_keeps_the_grid_matrices(dtype):
    rotations, eulers, mstep = sampling_module._exact_local_fine_grid(
        healpix_order=ORDER, angular_sampling_deg=ANGULAR_SAMPLING, random_perturbation=None, dtype=dtype
    )
    _rotation_grid_grid_rot = _fake_grid(ORDER, dtype=dtype)
    grid_rot = _rotation_grid_grid_rot.rotations
    grid_eulers = _rotation_grid_grid_rot.rotation_eulers
    assert _same(rotations, grid_rot) and _same(eulers, grid_eulers)
    exp_mstep, _ = apply_relion_rotation_perturbation_to_eulers(
        _canonical_eulers(ORDER), 0.0, ANGULAR_SAMPLING,
    )
    assert _same(mstep, exp_mstep)


def test_reused_grid_keeps_its_own_mstep_rotations():
    mstep = np.repeat(np.eye(3, dtype=np.float32)[None], N_ROT, axis=0)
    eulers = np.zeros((N_ROT, 3), dtype=np.float32)
    assert sampling_module._local_search_mstep_rotations(mstep, eulers, ORDER) is mstep


@pytest.mark.parametrize("n_rows", [N_ROT, N_ROT + 3])
def test_reused_grid_without_mstep_rotations_rebuilds_them_from_source_angles(n_rows):
    eulers = (np.arange(3 * n_rows, dtype=np.float64).reshape(n_rows, 3) / 3.0).astype(np.float32)
    got = sampling_module._local_search_mstep_rotations(None, eulers, ORDER)
    expected, _ = apply_relion_rotation_perturbation_to_eulers(
        sampling_module._relion_mstep_source_eulers(eulers, ORDER), 0.0, ANGULAR_SAMPLING,
    )
    assert _same(got, expected) and got.shape == (n_rows, 3, 3)


def _optics(image_sizes, pixel_sizes):
    return iteration_planning.RunOptics(
        image_geometry=ImageGeometry(image_shape=(64, 64), pixel_size_angstrom=1.5), model_pixel_size=1.5,
        optics_image_sizes=image_sizes, optics_pixel_sizes=pixel_sizes, multi_shape_halves=False,
    )


def _numbered_local_inputs(*, reuse=False, oversampling=0, symmetry="C1"):
    import logging

    _rotation_grid_rotations = _fake_grid(ORDER) if symmetry == "C1" else _fake_reduced_grid(ORDER, symmetry=symmetry)
    rotations = _rotation_grid_rotations.rotations
    eulers = _rotation_grid_rotations.rotation_eulers
    grid = sampling_module.TrialGrid(
        rotations=rotations if reuse else rotations[:1],
        rotation_eulers=eulers,
        mstep_rotations=rotations,
        translations=np.zeros((1, 2), dtype=np.float32),
    )
    return dict(
        search=local_sampling.LocalSearchSettings(
            healpix_order=ORDER, oversampling_order=oversampling, sigma_rot=0.1, sigma_psi=0.2, symmetry=symmetry,
        ),
        grid=grid, base_translations=grid.translations,
        image_window_size=32, model_support_size=24, base_healpix_order=0,
        coarse_size_healpix_order=0, perturbation=0.125,
        optics=_optics(None, None),
        particle_diameter_angstrom=100.0, log=logging.getLogger(__name__),
    )


def _final_local_inputs(*, oversampling=0, symmetry="C1"):
    return dict(
        search=local_sampling.LocalSearchSettings(
            healpix_order=ORDER, oversampling_order=oversampling, sigma_rot=0.1, sigma_psi=0.2, symmetry=symmetry,
        ),
        image_geometry=ImageGeometry(image_shape=(64, 64), pixel_size_angstrom=1.5),
        translations=np.zeros((1, 2), dtype=np.float32), base_translations=np.zeros((1, 2), dtype=np.float32),
        image_window_size=32, particle_diameter_angstrom=100.0, perturbation=0.125, rotation_dtype=np.float32,
    )


def _trace_local_grid_owners(monkeypatch):
    from helpers.tiny_refinement import CallTrace

    trace = CallTrace(monkeypatch)
    trace.wrap(sampling_module, "_exact_local_fine_grid")
    trace.wrap(sampling_module, "_local_search_mstep_rotations")
    trace.wrap(local_sampling, "relion_local_pass1_current_size")
    trace.wrap(local_sampling, "_precompute_exact_local_fine_grid_enabled")
    return trace


_LOCAL_ROUTES = {
    # route: (preparer, inputs, the owners it calls)
    "numbered_fine": (prepare_numbered_local_sampling, lambda **kw: _numbered_local_inputs(**kw),
                      ["_precompute_exact_local_fine_grid_enabled", "_exact_local_fine_grid"]),
    "numbered_parents": (prepare_numbered_local_sampling, lambda **kw: _numbered_local_inputs(oversampling=1, **kw),
                         ["relion_local_pass1_current_size"]),
    "numbered_reuse": (prepare_numbered_local_sampling, lambda **kw: _numbered_local_inputs(reuse=True, **kw),
                       ["_local_search_mstep_rotations"]),
    "final_fine": (prepare_final_local_sampling, lambda **kw: _final_local_inputs(**kw),
                   ["_precompute_exact_local_fine_grid_enabled", "_exact_local_fine_grid"]),
    "final_parents": (prepare_final_local_sampling, lambda **kw: _final_local_inputs(oversampling=1, **kw),
                      ["relion_local_pass1_current_size"]),
}


@pytest.mark.parametrize("route", sorted(_LOCAL_ROUTES))
def test_controller_builds_local_search_grids_through_the_owners(monkeypatch, route):
    """Each local-search route builds its fine grid, M-step rotations or parent window through one owner."""
    preparer, inputs, owners = _LOCAL_ROUTES[route]
    trace = _trace_local_grid_owners(monkeypatch)
    preparer(**inputs())
    assert trace.labels() == owners


def test_coarse_image_size_is_owned_by_the_adaptive_pass1_sizing(monkeypatch):
    """The remaining coarse-size arithmetic belongs to the adaptive (non-local) pass-1 sizing of each pass:
    the numbered iterations' plan and the Class3D final pass."""
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "plan_adaptive_image_size", "numbered")
    trace.wrap(finalization, "run_final_all_data", "final")
    for module in (iteration_planning, finalization):
        trace.wrap(module, "compute_coarse_image_size", "compute")
        trace.wrap(module, "clamp_relion_coarse_image_size", "clamp")
    run_tiny_refinement(monkeypatch, n_classes=2, converge_after=2)
    sizing = [(call.label, call.inside) for call in trace.calls_seen if call.label in ("compute", "clamp")]
    assert sizing == 2 * [("compute", ("numbered",)), ("clamp", ("numbered",))] + [
        ("compute", ("final",)), ("clamp", ("final",)),
    ]


@pytest.mark.parametrize("reuse", [False, True])
def test_numbered_fine_grid_keeps_pose_metadata_and_does_not_read_unused_optics(reuse):
    class UnusedOptics:
        def __getitem__(self, index):
            raise AssertionError("Non-expanded local sampling must not read optics sizing")

    inputs = _numbered_local_inputs(reuse=reuse)
    inputs.update(optics=_optics(UnusedOptics(), UnusedOptics()))
    result = prepare_numbered_local_sampling(**inputs)
    assert result.translations is inputs['grid'].translations
    assert result.base_translations is inputs['base_translations']
    assert result.model_support_size == 24
    if reuse:
        assert result.rotations is inputs['grid'].rotations
        assert result.mstep_rotations is inputs['grid'].mstep_rotations
        assert result.rotation_eulers is None
    else:
        expected, eulers, mstep = sampling_module._exact_local_fine_grid(
            healpix_order=ORDER, angular_sampling_deg=ANGULAR_SAMPLING,
            random_perturbation=0.125,
        )
        assert _same(result.rotations, expected)
        assert _same(result.rotation_eulers, eulers)
        assert _same(result.mstep_rotations, mstep)


@pytest.mark.parametrize("sizing_order", [0, 1])
def test_numbered_parent_window_uses_preceding_order_and_optics_geometry(monkeypatch, sizing_order):
    inputs = _numbered_local_inputs(oversampling=1)
    inputs.update(coarse_size_healpix_order=sizing_order, optics=_optics(np.array([48]), np.array([2.0])))
    calls = []

    def parent_window(**kwargs):
        calls.append(kwargs)
        return 12

    def unexpected_fine_grid(*args, **kwargs):
        raise AssertionError("Expanded parents must not materialize the exhaustive fine grid")

    monkeypatch.setattr(local_sampling, 'relion_local_pass1_current_size', parent_window)
    monkeypatch.setattr(local_sampling, '_precompute_exact_local_fine_grid_enabled', unexpected_fine_grid)
    result = prepare_numbered_local_sampling(**inputs)
    assert result.rotations is None and result.rotation_eulers is None
    assert result.mstep_rotations is None
    assert result.coarse_image_window_size == 12
    assert result.search.parent_order == 0
    assert calls[0]['pre_update_healpix_order'] == sizing_order
    assert calls[0]['box_size'] == 48
    assert matches(calls[0]['pixel_size'], 2.0)
    assert calls[0]['current_size'] == 32


def _fake_reduced_eulers(order, *, symmetry="C1"):
    """Point-group stand-in: a quarter of the C1 rows, off the float32 lattice."""
    if symmetry == "C1":
        return _canonical_eulers(order)
    return _canonical_eulers(order)[: rotation_grid_size(order) // 4] + 0.0765432


def _fake_reduced_grid(order, dtype=np.float32, *, symmetry="C1"):
    source = _fake_reduced_eulers(order, symmetry=symmetry)
    return sampling_module.RotationGrid(rotations=_relion_mstep_rotations_from_eulers(source, dtype=dtype), rotation_eulers=source.astype(dtype), healpix_order=order, symmetry=symmetry)


@pytest.mark.parametrize("random_perturbation", [0.3, None])
def test_point_group_fine_grid_uses_the_reduced_grid_and_its_source_angles(monkeypatch, random_perturbation):
    """Final Q a087087cc: a non-C1 lazy local fine grid is the point-group grid, not the C1 grid."""

    monkeypatch.setattr(sampling_module, "_get_relion_rotation_grid_eulers_float64", _fake_reduced_eulers)
    monkeypatch.setattr(sampling_module, "relion_scoring_rotation_grid", _fake_reduced_grid)
    rotations, eulers, mstep = sampling_module._exact_local_fine_grid(
        healpix_order=ORDER,
        angular_sampling_deg=ANGULAR_SAMPLING,
        random_perturbation=random_perturbation,
        symmetry="C4",
    )
    _rotation_grid_grid_rot = _fake_reduced_grid(ORDER, symmetry="C4")
    grid_rot = _rotation_grid_grid_rot.rotations
    grid_eulers = _rotation_grid_grid_rot.rotation_eulers
    if random_perturbation is None:
        exp_rot, exp_eulers = grid_rot, grid_eulers
    else:
        exp_rot, exp_eulers = apply_relion_rotation_perturbation_to_eulers(
            grid_eulers, random_perturbation, ANGULAR_SAMPLING
        )
    exp_mstep, _ = apply_relion_rotation_perturbation_to_eulers(
        _fake_reduced_eulers(ORDER, symmetry="C4"),
        0.0 if random_perturbation is None else random_perturbation,
        ANGULAR_SAMPLING,
    )
    assert rotations.shape == (N_ROT // 4, 3, 3)
    assert _same(rotations, exp_rot) and _same(eulers, exp_eulers) and _same(mstep, exp_mstep)


def test_point_group_reused_grid_rebuilds_mstep_rotations_from_reduced_source_angles(monkeypatch):
    monkeypatch.setattr(sampling_module, "_get_relion_rotation_grid_eulers_float64", _fake_reduced_eulers)
    source = _fake_reduced_eulers(ORDER, symmetry="C4")
    got = sampling_module._local_search_mstep_rotations(None, source.astype(np.float32), ORDER, symmetry="C4")
    expected, _ = apply_relion_rotation_perturbation_to_eulers(source, 0.0, ANGULAR_SAMPLING)
    assert _same(got, expected) and got.shape == (N_ROT // 4, 3, 3)


@pytest.mark.parametrize("route", sorted(_LOCAL_ROUTES))
def test_local_grid_owners_receive_the_point_group(monkeypatch, route):
    """Every local-search grid owner of each route is called with the search's point group (final Q a087087cc)."""
    monkeypatch.setattr(sampling_module, "_get_relion_rotation_grid_eulers_float64", _fake_reduced_eulers)
    monkeypatch.setattr(sampling_module, "relion_scoring_rotation_grid", _fake_reduced_grid)
    preparer, inputs, owners = _LOCAL_ROUTES[route]
    arguments = inputs(symmetry="C4")
    trace = _trace_local_grid_owners(monkeypatch)
    preparer(**arguments)
    assert trace.labels() == owners
    for call in trace.calls_seen:
        if call.label != "relion_local_pass1_current_size":
            assert call.kwargs.get("symmetry") == "C4", call.label


def test_controller_passes_the_point_group_to_every_grid_owner(monkeypatch):
    """The numbered trial grid and the final pass build their M-step source angles in the run's point group."""
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.refinement.refinement_options import SymmetryOptions

    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "iteration_trial_grid", "numbered")
    trace.wrap(finalization, "prepare_final_sampling", "final")
    trace.wrap(sampling_module, "_relion_mstep_source_eulers", "source_eulers")
    # A point group other than C1 requires the x-half M-step, which defaults off without a GPU.
    monkeypatch.setenv("RELAX_K1_RELION_X_HALF_MSTEP", "1")
    run_tiny_refinement(monkeypatch, parity=dict(perturb_factor=0.5), symmetry=SymmetryOptions(point_group="C4"))
    calls = trace.calls("source_eulers")
    assert [call.inside[-1] for call in calls] == ["numbered", "numbered", "final"]
    assert all(call.kwargs["symmetry"] == "C4" for call in calls)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fine_perturbation_releases_the_unperturbed_grid_before_mstep(monkeypatch, dtype):
    import weakref

    references = {}
    perturbations = []
    perturb = sampling_module.apply_relion_rotation_perturbation_to_eulers

    def grid(order, **kwargs):
        result = _fake_grid(order, **kwargs)
        references["grid"] = weakref.ref(result)
        references["rotations"] = weakref.ref(result.rotations)
        references["eulers"] = weakref.ref(result.rotation_eulers)
        return result

    def record_perturbation(*args, **kwargs):
        assert references["grid"]() is None
        if perturbations:
            assert references["rotations"]() is None
            assert references["eulers"]() is None
        perturbations.append(True)
        return perturb(*args, **kwargs)

    monkeypatch.setattr(sampling_module, "relion_scoring_rotation_grid", grid)
    monkeypatch.setattr(sampling_module, "apply_relion_rotation_perturbation_to_eulers", record_perturbation)
    rotations, eulers, mstep = sampling_module._exact_local_fine_grid(
        healpix_order=ORDER, angular_sampling_deg=ANGULAR_SAMPLING,
        random_perturbation=0.125, dtype=dtype,
    )
    assert len(perturbations) == 2
    assert rotations.shape[0] == eulers.shape[0] == mstep.shape[0]
