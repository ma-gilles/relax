"""Stored startup/source identity and its NPZ representation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement.command_options import resolve_initial_sampling
from relax.refinement.refinement_options import RestartProvenance
from relax.refinement.result_files import _savez_deflate_fast, build_archive_metadata
from relax.relion.input_poses import PoseProvenance

pytestmark = pytest.mark.unit


def metadata_inputs(*, n_classes, diagnostics, max_order):
    """Uneven, permuted half rows and both absent/present diagnostic sources."""
    return dict(
        args=SimpleNamespace(
            initial_pose_source="auto", seed=17, max_iter=3, healpix_order=2,
            adaptive_oversampling=1, max_significants=500, offset_sigma_angstrom=10.0,
            firstiter_cc=True,
        ),
        dataset=SimpleNamespace(image_shape=(32, 32), volume_shape=(32, 32, 32), voxel_size=1.5),
        particle_layout=SimpleNamespace(
            half1_rows=np.array([4, 0], dtype=np.int64),
            half2_rows=np.array([2, 3, 1], dtype=np.int64),
        ),
        initial_sampling=resolve_initial_sampling(2, 1, n_classes=n_classes, max_healpix_order=max_order),
        symmetry_provenance=dict(
            label="D2", family="D", operator_count=4, operator_sha256="symmetry-hash",
            relion_point_group=201, relion_point_group_order=2,
        ),
        effective_tau2_fudge=1.0 if n_classes == 1 else 4.0,
        tau2_fudge_source="RELION auto-refine default" if n_classes == 1 else "RELION Class3D default",
        follower_replay=SimpleNamespace(
            relion_iterations=(1, 3), source="follower", oracle_id="follower-oracle",
            boundary="pre-score", source_artifact_relative_paths=("half1.star", "half2.star"),
        ) if diagnostics else None,
        frozen_boundary=SimpleNamespace(
            source_dir=Path("frozen"), source_manifest_sha256="manifest-hash",
            boundary_sha256="boundary-hash", completed_relion_iteration=2,
        ) if diagnostics else None,
        initial_pose_source=PoseProvenance(
            requested_source="auto", resolved_source="input-star" if diagnostics else "fresh",
            path=Path("poses.star") if diagnostics else None,
            sha256="poses-hash" if diagnostics else None,
        ),
        max_significants_resolution=dict(
            maximum_significants_argument=211 if diagnostics else None,
            source="explicit CLI" if diagnostics else "RELION default", do_grad=diagnostics,
        ),
        n_images=5,
        n_rotations=768,
        n_translations=9,
        optimizer_seed_source="explicit CLI",
        particle_diameter_ang=24.0 if diagnostics else None,
        restart=(
            RestartProvenance((2,), Path("restart.json"), "restart-hash") if diagnostics
            else RestartProvenance((), None, None)
        ),
        relion_dispatch_schedule=SimpleNamespace(
            oracle_id="dispatch-oracle", oracle_manifest_sha256="dispatch-manifest",
            particle_order_sha256="particle-order",
        ) if diagnostics else None,
        captured_projector=SimpleNamespace(
            source_dir=Path("projectors"), source_manifest=Path("projectors/manifest.json"),
            replay_slot=1, source_manifest_sha256="projector-hash",
        ) if diagnostics else None,
        state_swap_probe=dict(
            target_relion_iteration=2, iteration=1, variant="recovar",
            replay_relion_references=False, replay_override_keys=["noise_variance"],
            required_replay_override_keys=["noise_variance"],
        ) if diagnostics else None,
        total_time=7.5,
        use_fresh_auto_refine_order=not diagnostics,
    )


@pytest.mark.parametrize("n_classes", [1, 4])
@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("max_order", [None, 7])
def test_archive_metadata_roundtrip_keeps_source_units_and_half_identity(
    tmp_path, n_classes, diagnostics, max_order,
):
    inputs = metadata_inputs(n_classes=n_classes, diagnostics=diagnostics, max_order=max_order)
    result = dict(
        current_sizes=[16, 24], pixel_resolutions=[12.0, 8.0], wall_times=[2.5, 3.5],
        state_swap_probe_applied_relion_iterations=[2] if diagnostics else [],
    )
    metadata = build_archive_metadata(result, **inputs)
    assert metadata["half1_indices"] is inputs["particle_layout"].half1_rows
    assert metadata["half2_indices"] is inputs["particle_layout"].half2_rows
    path = tmp_path / "startup.npz"
    _savez_deflate_fast(path, metadata)

    with np.load(path, allow_pickle=False) as saved:
        assert saved["symmetry_label"].item() == "D2"
        assert saved["symmetry_family"].item() == "D"
        assert saved["symmetry_operator_count"].item() == 4
        assert saved["symmetry_operator_count"].dtype == np.int64
        assert saved["symmetry_operator_sha256"].item() == "symmetry-hash"
        assert saved["symmetry_relion_point_group"].item() == 201
        assert saved["symmetry_relion_point_group_order"].item() == 2
        assert saved["random_seed"].item() == 17
        assert saved["random_seed_source"].item() == "explicit CLI"
        assert saved["n_iterations"].item() == 3
        assert saved["n_images"].item() == 5
        np.testing.assert_array_equal(saved["half1_indices"], [4, 0])
        np.testing.assert_array_equal(saved["half2_indices"], [2, 3, 1])
        np.testing.assert_array_equal(saved["current_sizes"], [16, 24])
        np.testing.assert_array_equal(saved["image_shape"], [32, 32])
        np.testing.assert_array_equal(saved["volume_shape"], [32, 32, 32])
        assert_matches(saved["voxel_size"], 1.5, rtol=1e-13)
        assert_matches(saved["offset_sigma_angstrom"], 10.0, rtol=1e-13)
        assert_matches(saved["tau2_fudge"], 1.0 if n_classes == 1 else 4.0, rtol=1e-13)
        assert saved["tau2_fudge"].dtype == np.float64
        assert saved["coarse_healpix_order"].item() == 2
        assert saved["finest_healpix_order"].item() == 3
        expected_cap = 7 if max_order is not None else (2 if n_classes == 4 else -1)
        assert saved["max_healpix_order"].item() == expected_cap
        assert saved["max_healpix_order_source"].item() == inputs["initial_sampling"].max_order_source
        assert saved["relion_fresh_particle_order_applied"].dtype == np.bool_
        assert saved["relion_fresh_particle_order_applied"].item() == (not diagnostics)
        if diagnostics:
            assert saved["frozen_boundary_dir"].item() == "frozen"
            assert saved["frozen_boundary_completed_relion_iteration"].item() == 2
            assert saved["initial_pose_source_path"].item() == "poses.star"
            assert saved["initial_pose_source_sha256"].item() == "poses-hash"
            assert saved["max_significants_argument"].item() == 211
            assert_matches(saved["particle_diameter_ang"], 24.0, rtol=1e-13)
            np.testing.assert_array_equal(saved["relion_follower_scale_replay_iterations"], [1, 3])
            assert saved["relion_dispatch_oracle_id"].item() == "dispatch-oracle"
            np.testing.assert_array_equal(saved["state_swap_probe_replay_override_keys"], ["noise_variance"])
        else:
            assert saved["frozen_boundary_dir"].item() == ""
            assert saved["frozen_boundary_completed_relion_iteration"].item() == -1
            assert saved["initial_pose_source_path"].item() == ""
            assert saved["initial_pose_source_sha256"].item() == ""
            assert np.isnan(saved["max_significants_argument"])
            assert np.isnan(saved["particle_diameter_ang"])
            assert "relion_follower_scale_replay_iterations" not in saved.files
            assert "relion_dispatch_oracle_id" not in saved.files
            assert saved["state_swap_probe_replay_override_keys"].size == 0
