"""The profile-only summary and the benchmark ledger of a run (relax.refinement.result_files)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers.refinement_results import numbered_metadata, refinement_result

from relax.helpers.iteration_history import RefinementHistory
from relax.refinement import result_files
from relax.refinement.refinement_options import RestartProvenance
from relax.refinement.refinement_result import ProfileStop

pytestmark = pytest.mark.unit

SHARED = {
    "git_commit", "python_version", "platform", "numpy_version", "jax_version", "jaxlib_version", "jax_devices",
    "data_dir", "output_dir", "timing_dir", "total_time_s", "wall_times_trajectory", "current_sizes", "n_images",
    "image_shape", "volume_shape", "voxel_size", "healpix_order", "auto_local_healpix_order", "sigma_ang",
    "adaptive_oversampling", "max_significants", "max_significants_resolution", "timing_rows", "timing_summary",
    "perturb_replay_restart_state_iterations", "perturb_replay_restart_provenance_path",
    "perturb_replay_restart_provenance_sha256", "relion_projector_replay_slot",
    "relion_projector_source_manifest_sha256", "relion_projector_capture_dir", "relion_projector_capture_manifest",
}


def _report(tmp_path, captured=None):
    return result_files.RunReport(
        data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "out"), timing_dir=None, total_time_s=12.5,
        n_images=6, image_shape=(8, 8), volume_shape=(8, 8, 8), voxel_size=1.5, healpix_order=2,
        auto_local_healpix_order=4, sigma_ang=None, adaptive_oversampling=1, max_significants=-1,
        max_significants_resolution={"active_max_significants": -1}, restart=RestartProvenance((3,), tmp_path / "provenance.json", "ab" * 32),
        captured_projector=captured, max_iter=3, random_seed=42,
        random_seed_source="explicit CLI", n_rotations=96, n_translations=21,
        initial_sampling=SimpleNamespace(coarse_order=1, fine_order=2, max_order=None, max_order_source="none"),
        frozen_boundary=None, symmetry_provenance={"label": "C1"},
        initial_pose_source=SimpleNamespace(requested_source="auto", resolved_source="input-star", sha256="ef" * 32),
        diagnostic_single_half=True, state_swap_probe={"iteration": 4},
    )


def test_profile_only_summary_writes_the_summary_and_the_ledger_copy(tmp_path, capsys):
    captured = SimpleNamespace(replay_slot=2, source_manifest_sha256="cd" * 32, source_dir=tmp_path / "capture",
                               source_manifest=tmp_path / "capture" / "SUMS")
    result = refinement_result(
        history=RefinementHistory(
            current_sizes=[12, 16], wall_times=[1.0], local_profile_history=[{"iteration": 1, "rows": 3}],
            global_profile_history=[], state_swap_probe_applied_relion_iterations=[5],
        ),
        numbered=numbered_metadata(setup_phase_seconds={"state_init": 0.5}),
        follower_scale=None,
        profile_stop=ProfileStop(score_only=True, wall_seconds=2.0, significant_count=None),
    )
    ledger = tmp_path / "ledger" / "ledger.json"
    path = result_files.write_profile_only_summary(result, _report(tmp_path, captured), benchmark_ledger_json=ledger)
    summary = json.loads(Path(path).read_text())
    assert path == tmp_path / "out" / "local_search_profile_only.json"
    assert json.loads(ledger.read_text()) == summary
    assert set(summary) == SHARED | {
        "symmetry", "initial_pose_source_requested", "initial_pose_source_resolved", "initial_pose_source_sha256",
        "profile_only", "stop_after_local_search_score_only", "diagnostic_single_half", "setup_phase_seconds",
        "local_profile_rows", "global_profile_rows", "state_swap_probe", "state_swap_probe_applied_relion_iterations",
    }
    assert summary["profile_only"] is True and summary["diagnostic_single_half"] is True
    assert summary["relion_projector_replay_slot"] == 2 and summary["perturb_replay_restart_state_iterations"] == [3]
    assert summary["state_swap_probe_applied_relion_iterations"] == [5]
    assert summary["stop_after_local_search_score_only"] is True
    assert summary["setup_phase_seconds"] == {"state_init": 0.5}
    assert summary["wall_times_trajectory"] == [2.0]  # the stopped iteration's own wall time
    assert summary["local_profile_rows"] == [{"iteration": 1, "rows": 3}]
    printed = capsys.readouterr().out
    assert "LOCAL SEARCH PROFILE ONLY" in printed and "Profiles: 1" in printed and "Current size: 16" in printed


def test_benchmark_ledger_holds_the_shared_fields_and_the_run_trajectories(tmp_path):
    archive = result_files.ArchiveReport(git_provenance={"head": "x"}, local_profile_rows=[], global_profile_rows=[],
                                         setup_phase_seconds={"state_init": 0.5})
    result = refinement_result(history=RefinementHistory(
        current_sizes=[12], wall_times=[1.0], pixel_resolutions=[3.0], ave_Pmax_trajectory=[0.5],
    ))
    path = tmp_path / "ledger.json"
    result_files.write_benchmark_ledger(path, result, _report(tmp_path), archive)
    ledger = json.loads(path.read_text())
    assert set(ledger) == SHARED | {
        "git_provenance", "max_iter", "random_seed", "random_seed_source", "n_iterations_emitted", "n_wall_times",
        "pixel_resolutions", "ave_Pmax_trajectory", "n_rotations", "n_translations", "coarse_healpix_order",
        "finest_healpix_order", "max_healpix_order", "max_healpix_order_source", "setup_phase_seconds",
        "local_profile_rows", "global_profile_rows", "frozen_boundary_dir", "frozen_boundary_manifest_sha256",
        "frozen_boundary_sha256", "frozen_boundary_completed_relion_iteration",
    }
    assert (ledger["max_healpix_order"], ledger["frozen_boundary_dir"], ledger["n_iterations_emitted"]) == (None, None, 1)
