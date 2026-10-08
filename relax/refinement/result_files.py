"""Final refinement archives, profile payloads and RELION-convention maps.

The command supplies resolved startup/source facts and selects publication order;
this module owns stored metadata, array layouts, compression and map export. See
docs/math/relion_refinement_algorithm.md, section 9.
"""

import json
import logging
import os
import platform
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from recovar.core import fourier_transform_utils as ftu

from relax.diagnostics import parity_dump
from relax.diagnostics.parity_provenance import git_head_or_none, git_worktree_provenance
from relax.helpers import iteration_history
from relax.helpers.resolution import shell_index_to_resolution_angstrom
from relax.refinement.refinement_result import FinalPassResult, ModelMaps, RefinementResult
from relax.sparse_pass2.engine_record import take_coarse_engine_calls, take_pass_engines

if TYPE_CHECKING:
    from relax.refinement.refinement_options import InitialSampling, RestartProvenance
    from relax.relion.input_poses import PoseProvenance


logger = logging.getLogger("relax.refinement.full_refinement")


def final_pass_result(
    scores,
    sampling,
    *,
    accuracy,
    accuracy_status,
    prior_details,
    fsc,
    prior_weight_combination,
    class_assignments,
    gridding_kernel,
) -> FinalPassResult:
    """The final pass's outputs as a record, with the archive's None sentinels and host casts.

    The controller retains scores, sampling and reconstruction scratch while this
    immediate-return builder runs. Engine recordings are consumed in the same
    order as the original finalization boundary.
    ``prior_weight_combination`` names the mode's prior rule ("sum" for K=1, "class_iref" for
    Class3D); ``class_assignments`` is the per-half class assignments, None for K=1.
    """
    return FinalPassResult(
        pass2_engines=take_pass_engines(),
        coarse_engines=take_coarse_engine_calls(),
        expected_accuracy_status=accuracy_status,
        acc_rot=None if accuracy is None else accuracy.acc_rot,
        acc_trans=None if accuracy is None else accuracy.acc_trans_angstrom,
        acc_rot_per_class=None if accuracy is None else accuracy.acc_rot_per_class,
        acc_trans_per_class=None if accuracy is None else accuracy.acc_trans_per_class_angstrom,
        expected_accuracy_class_counts=None if accuracy is None else accuracy.class_counts,
        fsc=fsc,
        tau2_radial=(
            None
            if prior_details is None
            else np.asarray(prior_details["prior_shells"], dtype=np.float64)
        ),
        tau2_fsc_used=(
            None
            if prior_details is None or prior_details.get("fsc_shells") is None
            else np.asarray(prior_details["fsc_shells"], dtype=np.float64)
        ),
        tau2_ssnr=(
            None
            if prior_details is None
            else np.asarray(prior_details["ssnr_shells"], dtype=np.float64)
        ),
        tau2_weight_combination=prior_weight_combination,
        best_rotation_eulers=scores.best_pose_rotation_eulers,
        best_translations=scores.best_pose_translations,
        max_posterior=scores.max_posterior,
        class_assignments=class_assignments,
        sampling_perturbation=sampling.random_perturbation,
        sampling_perturbation_applied=(sampling.perturbation is not None),
        sampling_relion_iteration=sampling.relion_iteration,
        sampling_star=sampling.sampling_star,
        sampling_star_source=sampling.sampling_star_source,
        sampling_offset_range=sampling.translation_range,
        sampling_offset_step=sampling.translation_step,
        gridding_correct=gridding_kernel,
    )


class ArchiveReport(NamedTuple):
    """Reported run provenance and profiles reused by the benchmark ledger."""

    git_provenance: dict
    local_profile_rows: list
    global_profile_rows: list
    setup_phase_seconds: dict


## same this should be moved elsewhere
def _jsonable_profile_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable_profile_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_profile_value(item) for item in value]
    try:
        arr = np.asarray(value)
    except Exception:
        return str(value)
    if arr.shape == ():
        return _jsonable_profile_value(arr.item())
    return _jsonable_profile_value(arr.tolist())


def profile_rows_for_json(rows):
    return [
        {str(key): _jsonable_profile_value(value) for key, value in row.items()}
        for row in rows
        if isinstance(row, dict)
    ]


def _rotation_posterior_arrays(key, posterior_per_half):
    """One iteration's per-half orientation posterior as npz arrays.

    The halves share one ``key`` array when their posteriors have the same shape.
    A local-search pass scores each half on its own rotation subset, so the
    halves can differ in length; each half is then saved as ``key_half1`` and
    ``key_half2``, as run_multi_iter_parity.py saves uneven half sequences.
    """

    halves = [None if value is None else np.asarray(value, dtype=np.float64) for value in posterior_per_half]
    shapes = {value.shape for value in halves if value is not None}
    if all(value is not None for value in halves) and len(shapes) == 1:
        return {key: np.stack(halves)}
    return {f"{key}_half{half + 1}": value for half, value in enumerate(halves) if value is not None}


# Complex arrays at least this large are Fourier maps, which deflate does not
# shrink: the 10097 run's three 134 MB maps compressed to 94% and took most of
# the 12.5 s the archive cost; stored, the archive writes in 0.35 s at 437 MB
# instead of 391 MB (job 14514567's archive, login node).
_NPZ_STORED_COMPLEX_MIN_BYTES = 16 * 1024**2


def _savez_deflate_fast(path, arrays):
    """``np.savez_compressed`` at zlib level 1 instead of its default 6, with Fourier maps stored.

    The archive is an ordinary ``.npz`` that ``np.load`` reads unchanged; zip
    members may mix stored and deflated entries. The dense rotation posteriors
    are almost all zeros: level 1 compresses a 301 MB posterior in 0.5 s
    instead of 1.4 s. Complex arrays of at least
    ``_NPZ_STORED_COMPLEX_MIN_BYTES`` are the reference maps, which do not
    compress at either level, so they are stored.
    """

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for name, value in arrays.items():
            value = np.asanyarray(value)
            entry = f"{name}.npy"  # the archive's deflate at level 1
            if value.dtype.kind == "c" and value.nbytes >= _NPZ_STORED_COMPLEX_MIN_BYTES:
                entry = zipfile.ZipInfo(entry, date_time=time.localtime(time.time())[:6])
                entry.compress_type = zipfile.ZIP_STORED
            with archive.open(entry, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, value, allow_pickle=True)


def build_archive_metadata(
    result: RefinementResult,
    *,
    args,
    dataset,
    effective_tau2_fudge,
    follower_replay,
    frozen_boundary,
    initial_sampling: "InitialSampling",
    initial_pose_source: "PoseProvenance",
    max_significants_resolution,
    n_images,
    n_rotations,
    n_translations,
    optimizer_seed_source,
    particle_diameter_ang,
    particle_layout,
    restart: "RestartProvenance",
    relion_dispatch_schedule,
    state_swap_probe,
    symmetry_provenance,
    tau2_fudge_source,
    total_time,
    use_fresh_auto_refine_order,
) -> dict:
    """Format startup/source metadata for the refinement NPZ archive.

    Diagnostic provenance, defaults and half-row identity retain their stored
    schema. Numerical result arrays are appended by ``write_refinement_archive``.
    """
    save_dict = {
        "symmetry_label": np.asarray(symmetry_provenance["label"]),
        "symmetry_family": np.asarray(symmetry_provenance["family"]),
        "symmetry_operator_count": np.int64(symmetry_provenance["operator_count"]),
        "symmetry_operator_sha256": np.asarray(symmetry_provenance["operator_sha256"]),
        "symmetry_relion_point_group": np.int64(symmetry_provenance["relion_point_group"]),
        "symmetry_relion_point_group_order": np.int64(symmetry_provenance["relion_point_group_order"]),
        "initial_pose_source_requested": np.asarray(initial_pose_source.requested_source),
        "initial_pose_source_resolved": np.asarray(initial_pose_source.resolved_source),
        "initial_pose_source_path": np.asarray(str(initial_pose_source.path or "")),
        "initial_pose_source_sha256": np.asarray(initial_pose_source.sha256 or ""),
        "relion_fresh_particle_order_applied": np.bool_(use_fresh_auto_refine_order),
        # The seed actually used (RELION's default -1 takes the time) and where it came from.
        "random_seed": np.int64(args.seed),
        "random_seed_source": np.asarray(optimizer_seed_source),
        "current_sizes": np.array(result.history.current_sizes),
        "pixel_resolutions": np.array(result.history.pixel_resolutions),
        "wall_times": np.array(result.history.wall_times),
        "total_time": total_time,
        "n_iterations": args.max_iter,
        "healpix_order": args.healpix_order,
        "coarse_healpix_order": initial_sampling.coarse_order,
        "finest_healpix_order": initial_sampling.fine_order,
        "max_healpix_order": -1 if initial_sampling.max_order is None else initial_sampling.max_order,
        "max_healpix_order_source": np.asarray(initial_sampling.max_order_source),
        "n_rotations": n_rotations,
        "n_translations": n_translations,
        "n_images": n_images,
        "image_shape": np.array(dataset.image_shape),
        "volume_shape": np.array(dataset.volume_shape),
        "voxel_size": dataset.voxel_size,
        "adaptive_oversampling": args.adaptive_oversampling,
        "max_significants": args.max_significants,
        "max_significants_argument": (
            np.nan
            if max_significants_resolution["maximum_significants_argument"] is None
            else int(max_significants_resolution["maximum_significants_argument"])
        ),
        "max_significants_source": np.asarray(
            str(max_significants_resolution["source"])
        ),
        "max_significants_do_grad": np.bool_(
            bool(max_significants_resolution["do_grad"])
        ),
        "offset_sigma_angstrom": args.offset_sigma_angstrom,
        "tau2_fudge": np.float64(effective_tau2_fudge),
        "tau2_fudge_source": np.asarray(tau2_fudge_source),
        "particle_diameter_ang": (np.float64(particle_diameter_ang) if particle_diameter_ang is not None else np.nan),
        "firstiter_cc_effective": np.bool_(bool(args.firstiter_cc)),
        "half1_indices": particle_layout.half1_rows,
        "half2_indices": particle_layout.half2_rows,
        "perturb_replay_restart_state_iterations": np.asarray(
            restart.iterations,
            dtype=np.int64,
        ),
        "perturb_replay_restart_provenance_path": np.asarray(
            ""
            if restart.path is None
            else str(restart.path)
        ),
        "perturb_replay_restart_provenance_sha256": np.asarray(
            restart.sha256 or ""
        ),
        # The captured RELION projector is retired (git tag retired/captured-projector-20261006); its keys
        # keep the values a run without one wrote.
        "relion_projector_replay_slot": np.int64(-1),
        "relion_projector_source_manifest_sha256": np.asarray(""),
        "relion_projector_capture_dir": np.asarray(""),
        "relion_projector_capture_manifest": np.asarray(""),
        "frozen_boundary_dir": np.asarray(
            "" if frozen_boundary is None else str(frozen_boundary.source_dir)
        ),
        "frozen_boundary_manifest_sha256": np.asarray(
            "" if frozen_boundary is None else frozen_boundary.source_manifest_sha256
        ),
        "frozen_boundary_sha256": np.asarray(
            "" if frozen_boundary is None else frozen_boundary.boundary_sha256
        ),
        "frozen_boundary_completed_relion_iteration": np.int64(
            -1 if frozen_boundary is None else frozen_boundary.completed_relion_iteration
        ),
        "state_swap_probe_target_relion_iteration": np.int64(
            -1
            if state_swap_probe is None
            else int(state_swap_probe["target_relion_iteration"])
        ),
        "state_swap_probe_loop_index": np.int64(
            -1 if state_swap_probe is None else int(state_swap_probe["iteration"])
        ),
        "state_swap_probe_variant": np.asarray(
            "" if state_swap_probe is None else str(state_swap_probe["variant"])
        ),
        "state_swap_probe_replay_relion_references": np.bool_(
            False
            if state_swap_probe is None
            else bool(state_swap_probe["replay_relion_references"])
        ),
        "state_swap_probe_applied_relion_iterations": np.asarray(
            result.history.state_swap_probe_applied_relion_iterations,
            dtype=np.int64,
        ),
        "state_swap_probe_replay_override_keys": np.asarray(
            [] if state_swap_probe is None else state_swap_probe["replay_override_keys"],
            dtype=np.str_,
        ),
        "state_swap_probe_required_replay_override_keys": np.asarray(
            []
            if state_swap_probe is None
            else state_swap_probe["required_replay_override_keys"],
            dtype=np.str_,
        ),
    }
    if follower_replay is not None:
        save_dict["relion_follower_scale_replay_iterations"] = np.asarray(
            follower_replay.relion_iterations,
            dtype=np.int64,
        )
        save_dict["relion_follower_scale_replay_source"] = np.asarray(
            follower_replay.source
        )
        save_dict["relion_follower_scale_replay_oracle_id"] = np.asarray(
            follower_replay.oracle_id
        )
        save_dict["relion_follower_scale_replay_boundary"] = np.asarray(
            follower_replay.boundary
        )
        save_dict["relion_follower_scale_replay_source_artifacts"] = np.asarray(
            follower_replay.source_artifact_relative_paths
        )
    if relion_dispatch_schedule is not None:
        save_dict["relion_dispatch_oracle_id"] = np.asarray(
            relion_dispatch_schedule.oracle_id
        )
        save_dict["relion_dispatch_oracle_manifest_sha256"] = np.asarray(
            relion_dispatch_schedule.oracle_manifest_sha256
        )
        save_dict["relion_dispatch_particle_order_sha256"] = np.asarray(
            relion_dispatch_schedule.particle_order_sha256
        )
    return save_dict


def write_refinement_archive(
    result: RefinementResult,
    *,
    out_path,
    metadata: dict,
    half_indices,
    n_images: int,
    skip_large_outputs: bool,
) -> ArchiveReport:
    """Append result arrays to caller-owned metadata and write its NPZ archive.

    The caller retains the payload through subsequent reporting. Half indices
    map half-order poses and class histories back to original image rows.
    The archive is written from ``result.archive_fields()``, the saved-format mapping.
    """
    fields = result.archive_fields()
    save_dict = metadata
    half1_idx, half2_idx = half_indices
    if "healpix_order_trajectory" in fields:
        save_dict["healpix_order_trajectory"] = np.asarray(
            fields["healpix_order_trajectory"],
            dtype=np.int32,
        )
    # Which E-step engine each pass ran on, per iteration and for the final all-data
    # pass (relax.sparse_pass2.engine_record), as JSON: resident vs fallback per run.
    for key in (
        "pass2_engine_trajectory", "final_all_data_pass2_engines",
        "coarse_engine_trajectory", "final_all_data_coarse_engines",
    ):
        if fields.get(key) is not None:
            save_dict[key] = np.asarray(json.dumps(fields[key]))
    for key, dtype in (
        ("relion_follower_scale_replay_requested_iterations", np.int64),
        ("relion_follower_scale_replay_applied_iterations", np.int64),
        ("relion_scale_follower_scales", np.float64),
        ("relion_scale_rank1_serialized", np.float64),
        ("relion_scale_follower_owners_half1", np.int64),
        ("relion_scale_follower_owners_half1_trajectory", np.int64),
        ("relion_scale_follower_scales_numbered_pre_score_trajectory", np.float64),
        ("relion_scale_follower_scales_numbered_post_mstep_trajectory", np.float64),
    ):
        if fields.get(key) is not None:
            save_dict[key] = np.asarray(fields[key], dtype=dtype)
    for key, dtype in (
        ("ave_Pmax_trajectory", np.float64),
        ("frac_changed_trajectory", np.float64),
        ("acc_rot_trajectory", np.float64),
        ("acc_trans_trajectory", np.float64),
        ("smallest_change_angles_trajectory", np.float64),
        ("smallest_change_offsets_trajectory", np.float64),
        ("acc_rot_per_class_trajectory", np.float64),
        ("acc_trans_per_class_trajectory", np.float64),
        ("expected_accuracy_class_counts_trajectory", np.int64),
        ("expected_accuracy_status_trajectory", np.str_),
    ):
        if key in fields:
            save_dict[key] = np.asarray(fields[key], dtype=dtype)
    for indices_key in (
        "expected_accuracy_trial_local_indices",
        "expected_accuracy_trial_particle_ids",
    ):
        if fields.get(indices_key) is not None:
            save_dict[indices_key] = np.asarray(fields[indices_key], dtype=np.int64)
    for final_accuracy_key, final_accuracy_dtype in (
        ("final_all_data_acc_rot", np.float64),
        ("final_all_data_acc_trans", np.float64),
        ("final_all_data_acc_rot_per_class", np.float64),
        ("final_all_data_acc_trans_per_class", np.float64),
        ("final_all_data_expected_accuracy_class_counts", np.int64),
    ):
        if fields.get(final_accuracy_key) is not None:
            save_dict[final_accuracy_key] = np.asarray(
                fields[final_accuracy_key],
                dtype=final_accuracy_dtype,
            )
    if fields.get("final_all_data_expected_accuracy_status") is not None:
        save_dict["final_all_data_expected_accuracy_status"] = np.asarray(
            fields["final_all_data_expected_accuracy_status"],
            dtype=np.str_,
        )
    for key, dtype in (
        ("sigma_offset_trajectory", np.float64),
        ("sigma_offset_per_half_trajectory", object),
        ("sigma_offset_used_trajectory", np.float64),
        ("sigma_offset_used_per_half_trajectory", object),
    ):
        if key in fields:
            save_dict[key] = np.asarray(fields[key], dtype=dtype)
    if fields.get("direction_prior_trajectory_per_half") is not None:
        save_dict["direction_prior_trajectory_per_half"] = np.asarray(
            fields["direction_prior_trajectory_per_half"], dtype=object
        )
    if fields.get("rotation_posterior_trajectory_per_half") is not None:
        for i, posterior_per_half in enumerate(
            fields["rotation_posterior_trajectory_per_half"]
        ):
            if posterior_per_half is not None:
                save_dict.update(
                    _rotation_posterior_arrays(f"rotation_posterior_per_half_iter_{i:03d}", posterior_per_half)
                )
    if "convergence_state" in fields:
        state = fields["convergence_state"]
        save_dict["convergence_iteration"] = np.int32(state.iteration)
        save_dict["convergence_current_resolution"] = np.float64(state.current_resolution)
        save_dict["convergence_ave_Pmax"] = np.float64(state.ave_Pmax)
        save_dict["convergence_healpix_order"] = np.int32(state.healpix_order)
        save_dict["convergence_has_converged"] = np.bool_(state.has_converged)
    frozen_scoring_hashes = fields.get("frozen_initial_scoring_state_sha256")
    if frozen_scoring_hashes is not None:
        frozen_scoring_fields = sorted(frozen_scoring_hashes)
        save_dict["frozen_scoring_state_field_names"] = np.asarray(
            frozen_scoring_fields,
            dtype=str,
        )
        save_dict["frozen_scoring_state_array_sha256"] = np.asarray(
            [frozen_scoring_hashes[field]["sha256"] for field in frozen_scoring_fields],
            dtype=str,
        )
        save_dict["frozen_scoring_state_array_dtypes"] = np.asarray(
            [frozen_scoring_hashes[field]["dtype"] for field in frozen_scoring_fields],
            dtype=str,
        )
        save_dict["frozen_scoring_state_array_shapes_json"] = np.asarray(
            [
                json.dumps(frozen_scoring_hashes[field]["shape"], separators=(",", ":"))
                for field in frozen_scoring_fields
            ],
            dtype=str,
        )

    iteration_history.add_class_history_artifacts(save_dict, fields, half1_idx, half2_idx, n_images)

    local_profile_rows = profile_rows_for_json(fields.get("local_profile_history", []))
    global_profile_rows = profile_rows_for_json(fields.get("global_profile_history", []))
    setup_phase_seconds = {str(key): float(value) for key, value in fields.get("setup_phase_seconds", {}).items()}

    iteration_history.add_refinement_history_artifacts(save_dict, fields, half1_idx, half2_idx, n_images)

    git_provenance = git_worktree_provenance()
    save_dict["git_commit"] = np.asarray(git_provenance["head"])
    save_dict["git_branch"] = np.asarray(git_provenance["branch"])
    save_dict["git_dirty_count"] = np.asarray(git_provenance["dirty_count"], dtype=np.int64)
    save_dict["git_diff_sha256"] = np.asarray(git_provenance["diff_sha256"])
    save_dict["git_worktree_fingerprint_sha256"] = np.asarray(git_provenance["worktree_fingerprint_sha256"])
    save_dict["git_status_porcelain"] = np.asarray(git_provenance["status_porcelain"])
    save_dict["git_untracked_file_hashes"] = np.asarray(git_provenance["untracked_file_hashes"])

    # Save final merged volume (Fourier space)
    save_dict["final_mean_ft"] = np.asarray(fields["mean"])
    if setup_phase_seconds:
        save_dict["setup_phase_names"] = np.asarray(list(setup_phase_seconds.keys()))
        save_dict["setup_phase_cumulative_s"] = np.asarray(list(setup_phase_seconds.values()), dtype=np.float64)

    # Save per-half-set means
    for k in range(2):
        save_dict[f"half{k}_mean_ft"] = np.asarray(fields["means"][k])

    # Save hard assignments
    for k in range(2):
        if fields["hard_assignments"][k] is not None:
            save_dict[f"hard_assignments_half{k}"] = np.asarray(fields["hard_assignments"][k])

    if skip_large_outputs:
        logger.info("Skipping large refinement result archive (--skip-large-outputs): %s", out_path)
    else:
        _savez_deflate_fast(out_path, save_dict)
        logger.info("Results saved to %s", out_path)

    return ArchiveReport(
        git_provenance=git_provenance,
        local_profile_rows=local_profile_rows,
        global_profile_rows=global_profile_rows,
        setup_phase_seconds=setup_phase_seconds,
    )


def write_final_maps(
    maps: ModelMaps,
    *,
    output_dir,
    volume_shape,
    pixel_size_angstrom,
    n_classes: int,
    skip_large_outputs: bool,
) -> None:
    """Export the final merged, class and half maps (``result.maps``) in RELION's real-space convention."""
    if skip_large_outputs:
        logger.info("Skipping final MRC volume writes (--skip-large-outputs)")
    else:
        # Final maps are written in RELION's map convention (relax.helpers.map_io).
        from relax.helpers.map_io import write_map

        def _ft_to_real_volume(ft_array):
            ft_reshape = np.asarray(ft_array).reshape(volume_shape)
            return np.real(np.array(ftu.get_idft3(jnp.asarray(ft_reshape)))).astype(np.float32)

        final_mean_real = _ft_to_real_volume(maps.mean)
        write_map(os.path.join(output_dir, "final_merged.mrc"), final_mean_real, voxel_size=pixel_size_angstrom)
        if n_classes == 1:
            logger.info("Final merged volume saved to final_merged.mrc")
            for k in range(2):
                half_real = _ft_to_real_volume(maps.means[k])
                write_map(
                    os.path.join(output_dir, f"final_half{k + 1}.mrc"),
                    half_real,
                    voxel_size=pixel_size_angstrom,
                )
                logger.info("Half-%d volume saved", k + 1)
            unfiltered_means = maps.unfiltered_means
            if unfiltered_means is not None:
                for k in range(2):
                    unfiltered_real = _ft_to_real_volume(unfiltered_means[k])
                    write_map(
                        os.path.join(output_dir, f"final_half{k + 1}_unfil.mrc"),
                        unfiltered_real,
                        voxel_size=pixel_size_angstrom,
                    )
                    logger.info("Unfiltered half-%d volume saved", k + 1)
        else:
            # K-class: maps.means[k] has shape (K, V); maps.class_means
            # has shape (K, V) for the merged final iter; maps.mean is the
            # class-weighted merged volume.
            class_means_arr = np.asarray(maps.class_means)
            for c in range(n_classes):
                vol_real = _ft_to_real_volume(class_means_arr[c])
                write_map(
                    os.path.join(output_dir, f"final_class{c + 1:03d}.mrc"),
                    vol_real,
                    voxel_size=pixel_size_angstrom,
                )
            logger.info("Saved %d per-class merged final volumes", n_classes)


@dataclass(frozen=True)
class RunReport:
    """What a run's benchmark ledger and profile-only summary state beside its result.

    The command builds it once the controller has returned; the reports only read it.
    """

    data_dir: str
    output_dir: str
    timing_dir: Path | None
    total_time_s: float
    n_images: int
    image_shape: tuple
    volume_shape: tuple
    voxel_size: float
    healpix_order: int
    auto_local_healpix_order: int
    sigma_ang: float | None
    adaptive_oversampling: int
    max_significants: int
    max_significants_resolution: dict
    restart: "RestartProvenance"
    # Sampling and seed, which the benchmark ledger records.
    max_iter: int
    random_seed: int
    random_seed_source: str
    n_rotations: int
    n_translations: int
    initial_sampling: "InitialSampling"
    frozen_boundary: object | None
    # The profile-only summary's provenance.
    symmetry_provenance: dict
    initial_pose_source: "PoseProvenance"
    diagnostic_single_half: bool
    state_swap_probe: dict | None


def _report_fields(fields: dict, report: RunReport) -> dict:
    """The fields the ledger and the profile-only summary share: environment, inputs, sampling and provenance.

    ``fields`` is the result's ``archive_fields()``."""
    timing_rows = parity_dump._collect_timing_rows(report.timing_dir)
    return {
        "git_commit": git_head_or_none(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "jax_version": getattr(jax, "__version__", None),
        "jaxlib_version": getattr(jaxlib, "__version__", None),
        "jax_devices": [str(device) for device in jax.devices()],
        "data_dir": report.data_dir,
        "output_dir": report.output_dir,
        "timing_dir": str(report.timing_dir.resolve()) if report.timing_dir is not None else None,
        "total_time_s": float(report.total_time_s),
        "wall_times_trajectory": [float(x) for x in fields.get("wall_times", [])],
        "current_sizes": [int(x) for x in fields.get("current_sizes", [])],
        "n_images": int(report.n_images),
        "image_shape": [int(x) for x in report.image_shape],
        "volume_shape": [int(x) for x in report.volume_shape],
        "voxel_size": float(report.voxel_size),
        "healpix_order": int(report.healpix_order),
        "auto_local_healpix_order": int(report.auto_local_healpix_order),
        "sigma_ang": None if report.sigma_ang is None else float(report.sigma_ang),
        "adaptive_oversampling": int(report.adaptive_oversampling),
        "max_significants": int(report.max_significants),
        "max_significants_resolution": report.max_significants_resolution,
        "timing_rows": timing_rows,
        "timing_summary": parity_dump._summarize_timing_rows(timing_rows),
        "perturb_replay_restart_state_iterations": list(report.restart.iterations),
        "perturb_replay_restart_provenance_path": (
            str(report.restart.path)
            if report.restart.path is not None
            else None
        ),
        "perturb_replay_restart_provenance_sha256": report.restart.sha256,
        # Retired with the captured RELION projector (git tag retired/captured-projector-20261006).
        "relion_projector_replay_slot": None,
        "relion_projector_source_manifest_sha256": None,
        "relion_projector_capture_dir": None,
        "relion_projector_capture_manifest": None,
    }


def _write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_profile_only_summary(result: RefinementResult, report: RunReport, *, benchmark_ledger_json) -> Path:
    """Write ``local_search_profile_only.json`` (and the ledger copy) for a run that stopped after its local
    search profile, and print its summary; returns the summary path."""
    fields = result.archive_fields()  # the report reads the saved-format mapping
    local_profile_rows = profile_rows_for_json(fields.get("local_profile_history", []))
    profile_summary = {
        **_report_fields(fields, report),
        "symmetry": report.symmetry_provenance,
        "initial_pose_source_requested": report.initial_pose_source.requested_source,
        "initial_pose_source_resolved": report.initial_pose_source.resolved_source,
        "initial_pose_source_sha256": report.initial_pose_source.sha256,
        "profile_only": True,
        "stop_after_local_search_score_only": bool(fields.get("stop_after_local_search_score_only", False)),
        "diagnostic_single_half": bool(report.diagnostic_single_half),
        "setup_phase_seconds": {
            str(key): float(value) for key, value in fields.get("setup_phase_seconds", {}).items()
        },
        "local_profile_rows": local_profile_rows,
        "global_profile_rows": profile_rows_for_json(fields.get("global_profile_history", [])),
        "state_swap_probe": report.state_swap_probe,
        "state_swap_probe_applied_relion_iterations": [
            int(iteration) for iteration in fields.get("state_swap_probe_applied_relion_iterations", [])
        ],
    }
    profile_path = Path(report.output_dir) / "local_search_profile_only.json"
    _write_json(profile_path, profile_summary)
    logger.info("Profile-only summary saved to %s", profile_path)
    if benchmark_ledger_json:
        _write_json(benchmark_ledger_json, profile_summary)
        logger.info("Benchmark ledger saved to %s", benchmark_ledger_json)
    print("\n" + "=" * 70)
    print("LOCAL SEARCH PROFILE ONLY")
    print("=" * 70)
    print(f"Profiles: {len(local_profile_rows)}")
    print(f"Total wall time: {report.total_time_s:.1f}s")
    if fields.get("current_sizes"):
        print(f"Current size: {fields['current_sizes'][-1]}")
    print(f"Summary JSON: {profile_path}")
    print("=" * 70)
    return profile_path


def write_benchmark_ledger(path, result: RefinementResult, report: RunReport, archive_report: ArchiveReport) -> None:
    """Write a completed run's benchmark ledger: the shared report fields, the run's trajectories, its sampling
    and the profiles and provenance its archive reported."""
    fields = result.archive_fields()  # the report reads the saved-format mapping
    initial_sampling, frozen_boundary = report.initial_sampling, report.frozen_boundary
    ledger = {
        **_report_fields(fields, report),
        "git_provenance": archive_report.git_provenance,
        "max_iter": int(report.max_iter),
        "random_seed": int(report.random_seed),
        "random_seed_source": str(report.random_seed_source),
        "n_iterations_emitted": int(len(fields.get("current_sizes", []))),
        "n_wall_times": int(len(fields.get("wall_times", []))),
        "pixel_resolutions": [float(x) for x in fields.get("pixel_resolutions", [])],
        "ave_Pmax_trajectory": [float(x) for x in fields.get("ave_Pmax_trajectory", [])],
        "n_rotations": int(report.n_rotations),
        "n_translations": int(report.n_translations),
        "coarse_healpix_order": int(initial_sampling.coarse_order),
        "finest_healpix_order": int(initial_sampling.fine_order),
        "max_healpix_order": None if initial_sampling.max_order is None else int(initial_sampling.max_order),
        "max_healpix_order_source": str(initial_sampling.max_order_source),
        "setup_phase_seconds": archive_report.setup_phase_seconds,
        "local_profile_rows": archive_report.local_profile_rows,
        "global_profile_rows": archive_report.global_profile_rows,
        "frozen_boundary_dir": None if frozen_boundary is None else str(frozen_boundary.source_dir),
        "frozen_boundary_manifest_sha256": (
            None if frozen_boundary is None else frozen_boundary.source_manifest_sha256
        ),
        "frozen_boundary_sha256": None if frozen_boundary is None else frozen_boundary.boundary_sha256,
        "frozen_boundary_completed_relion_iteration": (
            None if frozen_boundary is None else frozen_boundary.completed_relion_iteration
        ),
    }
    _write_json(path, ledger)
    logger.info("Benchmark ledger saved to %s", path)


def print_refinement_summary(result: RefinementResult, *, total_time_s: float, box_size: int, pixel_size: float) -> None:
    """Print the per-iteration table and the final size and resolution of a completed run."""
    history = result.history
    print("\n" + "=" * 70)
    print("REFINEMENT SUMMARY")
    print("=" * 70)
    print(f"{'Iter':>4s}  {'CurSize':>8s}  {'PixRes':>8s}  {'ResA':>8s}  {'Time(s)':>8s}", end="")
    if any(c is not None for c in history.significant_counts):
        print(f"  {'MedSig':>8s}", end="")
    print()
    print("-" * 70)

    for i in range(len(history.current_sizes)):
        cs = history.current_sizes[i]
        pr = history.pixel_resolutions[i]
        res_a = shell_index_to_resolution_angstrom(pr, box_size, pixel_size)
        wt = history.wall_times[i]
        line = f"{i + 1:4d}  {cs:8d}  {pr:8.1f}  {res_a:8.2f}  {wt:8.1f}"
        if history.significant_counts[i] is not None:
            med_sig = int(np.median(np.asarray(history.significant_counts[i])))
            line += f"  {med_sig:8d}"
        print(line)

    print("-" * 70)
    print(f"Total wall time: {total_time_s:.1f}s")
    # A continuation from a converged state runs only the final all-data pass,
    # which records no per-iteration row.
    if history.current_sizes:
        print(f"Final current_size: {history.current_sizes[-1]}")
        print(f"Final pixel resolution: {history.pixel_resolutions[-1]:.1f}")
    # RELION reports the final all-data iteration's current resolution (updateCurrentResolution after it,
    # ml_optimiser_mpi.cpp:4329); without that pass, the last numbered iteration's.
    final_state = result.convergence_state
    if result.final_all_data_ran and final_state is not None:
        print(f"Final resolution: {float(final_state.current_resolution):.2f} A (final all-data iteration)")
    elif history.current_sizes:
        print(
            "Final resolution: "
            f"{shell_index_to_resolution_angstrom(history.pixel_resolutions[-1], box_size, pixel_size):.2f} A"
        )
    print("=" * 70)
