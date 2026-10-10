"""Optional local-search outputs retain their meanings."""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.refinement_specs import local_iteration_owners

from relax.local_search import half
from relax.types import LocalEMResult

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("return_profile", [False, True])
def test_local_sample_capture_preserves_profile_visibility(monkeypatch, return_profile):
    """Sample capture enables an internal profile even if the caller hides it."""
    profile = {"reconstruction_sample_indices_by_image": (np.array([1]), np.array([2]))}
    stats = object()

    def run_local(data, layout, kernel, support, **kwargs):
        assert support.return_reconstruction_sample_indices is True
        assert support.return_profile == return_profile
        return LocalEMResult(
            Ft_y=np.zeros(8, dtype=np.complex64),
            Ft_ctf=np.ones(8, dtype=np.float32),
            hard_assignments=np.array([0, 1], dtype=np.int32),
            stats=stats,
            profile=profile,
        )

    monkeypatch.setattr(half, "compute_local_search_resident", run_local)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    translations = np.zeros((2, 2), dtype=np.float32)
    result = half._run_local_search_iteration(*local_iteration_owners(
        SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 2), particles_file=None),
        None, None, rotations, rotations,
        healpix_order=0, sigma_rot=1.0, sigma_psi=1.0,
        translations=translations[:1], prior_translations=translations,
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp",
        current_size=2,
        pass2_layout=SimpleNamespace(
            rotation_counts=np.ones(2, dtype=np.int32), translation_grid=translations[:1], n_classes=1
        ),
        return_reconstruction_sample_indices=True,
        return_profile=return_profile,
    ))
    assert result.relion_stats is stats
    if return_profile:
        assert result.profile_summary is not profile
        assert result.profile_summary["reconstruction_sample_indices_by_image"] is profile["reconstruction_sample_indices_by_image"]
    else:
        assert result.profile_summary is None
    assert set(profile) == {"reconstruction_sample_indices_by_image"}


@pytest.mark.parametrize("route", ["local grid", "oversampled children", "pass-1 probe"])
def test_rescaled_local_rows_pass_the_provenance_check_and_are_projected(monkeypatch, route):
    """Images on another grid (s != 1) reach local search: the rows handed to the engine are the layout's own
    float64 rows divided by s and cast once, rebuilt by the rule and arguments that built the layout.

    "oversampled children": parent oversampling makes the rows children on the fine grid, rebuilt from their
    source Euler rows. "pass-1 probe": as dense_half's adaptive local pass 1 does, the parent layout is built
    with the iteration's perturbation and handed in (``pass2_layout``) under a grid that states no perturbation.
    411ed2c1 rebuilt both from the caller's grid and raised RotationProvenanceError (by 0.04-0.07)."""
    from scipy.spatial.transform import Rotation

    from relax.local_search.layout import (
        build_local_hypothesis_layout,
        build_local_search_grid_metadata,
        local_layout_host_rotations,
    )
    from relax.relion.optics_aberrations import projection_rotations

    scale, rp, step = 1.12, 0.29753, 15.0
    seen = {}

    def run_local(data, layout, *args, **kwargs):
        seen["layout"] = layout
        return LocalEMResult(
            Ft_y=np.zeros(8, dtype=np.complex64),
            Ft_ctf=np.ones(8, dtype=np.float32),
            hard_assignments=np.zeros(layout.rotation_counts.size, dtype=np.int32),
            stats=object(),
        )

    monkeypatch.setattr(half, "compute_local_search_resident", run_local)
    prior = Rotation.random(3, random_state=7).as_matrix()
    translations = np.zeros((3, 2), dtype=np.float32)
    grid = dict(rotation_grid_random_perturbation=rp, rotation_grid_angular_sampling_deg=step)
    if route == "oversampled children":
        grid.update(local_parent_oversampling_order=1, generate_relion_mstep_rotations=True)
    elif route == "local grid":
        grid.update(generate_relion_mstep_rotations=True)
    else:
        grid = dict(
            rotation_grid_random_perturbation=0.0,
            rotation_grid_angular_sampling_deg=None,
            pass2_layout=build_local_hypothesis_layout(
                prior, None, 0.2, 0.2, 2, translations[:1], translations, 1.0, None, 1.0,
                grid_metadata=build_local_search_grid_metadata(2, symmetry="C1"),
                rotation_grid_random_perturbation=rp, rotation_grid_angular_sampling_deg=step,
            ),
        )
    half._run_local_search_iteration(*local_iteration_owners(
        SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 2), voxel_size=1.0, particles_file=None),
        None, None, prior, None,
        healpix_order=2, sigma_rot=0.2, sigma_psi=0.2,
        translations=translations[:1], prior_translations=translations,
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp",
        current_size=2,
        projection_scale=scale,
        **grid,
    ))
    layout = seen["layout"]
    assert layout.rotations_flat.shape[0] > 0 and layout.oversampled_rows == (route == "oversampled children")
    pairs = [(False, layout.rotations_flat)]
    if layout.mstep_rotations_flat is not None:
        pairs.append((True, layout.mstep_rotations_flat))
    for mstep, rows in pairs:
        expected = projection_rotations(local_layout_host_rotations(layout, mstep=mstep), scale)
        np.testing.assert_array_equal(rows, expected)
    # The rebuilt rows are the perturbed grid's: an unperturbed rebuild is off by far more than float32 rounding.
    unperturbed = build_local_hypothesis_layout(
        prior, None, 0.2, 0.2, 2, translations[:1], translations, 1.0, None, 1.0,
        grid_metadata=build_local_search_grid_metadata(2, symmetry="C1"),
    )
    if route != "oversampled children":
        assert np.abs(unperturbed.rotations_flat / scale - layout.rotations_flat).max() > 1e-3


def test_adaptive_pass2_children_rebuild_from_their_source_eulers():
    """Local adaptive pass 2 scores the oversampled children of the probe's significant parents; their host rebuild
    (and so the s != 1 projection of their rows) reproduces both the scoring and the M-step rows."""
    from scipy.spatial.transform import Rotation

    from relax.local_search.layout import (
        build_local_adaptive_pass2_hypothesis_layout,
        build_local_hypothesis_layout,
        build_local_search_grid_metadata,
        local_layout_host_rotations,
    )

    rp = 0.29753
    parent = build_local_hypothesis_layout(
        Rotation.random(3, random_state=7).as_matrix(), None, 0.2, 0.2, 2, np.zeros((1, 2)), np.zeros((3, 2)), 1.0,
        None, 1.0, grid_metadata=build_local_search_grid_metadata(2, symmetry="C1"),
        rotation_grid_random_perturbation=rp, rotation_grid_angular_sampling_deg=15.0,
    )
    children = build_local_adaptive_pass2_hypothesis_layout(
        parent, [None] * parent.rotation_counts.size, 2, oversampling_order=1, random_perturbation=rp, symmetry="C1"
    )
    assert children.oversampled_rows and children.rotations_flat.shape[0] == 8 * parent.rotations_flat.shape[0]
    for mstep, rows in ((False, children.rotations_flat), (True, children.mstep_rotations_flat)):
        np.testing.assert_array_equal(local_layout_host_rotations(children, mstep=mstep).astype(np.float32), rows)
