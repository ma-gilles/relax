"""Reconstruction diagnostic serialization; scheduling stays in the controller.

The finite guard of the half accumulators runs before cross-half mixing (the captures of the accumulators
are ``relax.diagnostics.observers.BpRefAccumulatorObserver``). Writers preserve historical NPZ field names,
casts and optional-field behavior; model and history updates remain with the controller.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from recovar.core import fourier_transform_utils

from relax.diagnostics import finite_check
from relax.reconstruction import regularization_relion

if TYPE_CHECKING:
    from relax.refinement.image_size_plans import ClassImageSize
    from relax.refinement.numbered_reconstruction import ReconstructionSettings
    from relax.refinement.priors import ClassPriorEstimate


def check_half_accumulators_before_join(numerators, denominators, *, iteration, init_relion_iteration, log) -> None:
    """The finite guard of the native half accumulators, before any cross-half join.

    ``RELAX_EM_BPREF_FINITE_GUARD`` (``finite_check.half_accumulator_guard_mode``) selects off, warn or raise;
    the mode is logged at the first iteration. Run before the join, so a report names the half that is
    actually damaged rather than the one the join copied it into.
    """
    guard_mode = finite_check.half_accumulator_guard_mode()
    if iteration == 0:
        log.info(
            "BPref accumulator guard (%s): mode=%s",
            finite_check.HALF_ACCUMULATOR_GUARD_ENV,
            guard_mode,
        )
    if guard_mode != "off":
        finite_check.check_half_accumulators(
            {
                "Ft_y_0": numerators[0],
                "Ft_y_1": numerators[1],
                "Ft_ctf_0": denominators[0],
                "Ft_ctf_1": denominators[1],
            },
            context=finite_check.describe_context(
                iteration=iteration,
                relion_iteration=int(init_relion_iteration) + int(iteration) + 1,
            ),
        )


def write_bpref_accumulators(
    dump_dir: str,
    *,
    stage: str,
    iteration: int,
    current_size: int,
    padding_factor: int,
    box_size: int,
    voxel_size: float,
    volume_shape,
    accumulator_shape,
    Ft_y_0,
    Ft_y_1,
    Ft_ctf_0,
    Ft_ctf_1,
) -> None:
    """Save K1 accumulators at the controller's ``prejoin`` or ``accum`` stage.

    ``iteration`` is zero-based; filenames and payload metadata are one-based.
    Preserve the numerator dtype and save the real part of each weight array.
    These captures locate a state boundary, without establishing that every
    upstream scoring input matched between runs.
    """
    import pathlib

    pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        pathlib.Path(dump_dir) / f"recovar_bpref_{stage}_it{iteration + 1:03d}.npz",
        schema=np.asarray(f"recovar-bpref-{stage}-v2"),
        run_id=np.asarray(os.environ.get("RELAX_BPREF_BOUNDARY_DUMP_RUN_ID", "unset")),
        iteration=np.int32(iteration + 1),
        current_size=np.int32(current_size),
        padding_factor=np.int32(padding_factor),
        grid_size=np.int32(box_size),  # the box size, under the stored key the dump readers use
        voxel_size=np.float32(voxel_size),
        volume_shape=np.asarray(volume_shape, dtype=np.int32),
        mstep_accumulator_shape=np.asarray(accumulator_shape, dtype=np.int32),
        Ft_y_0=np.asarray(Ft_y_0),
        Ft_y_1=np.asarray(Ft_y_1),
        Ft_ctf_0=np.asarray(Ft_ctf_0).real,
        Ft_ctf_1=np.asarray(Ft_ctf_1).real,
    )


def write_class_image_size(
    plan: ClassImageSize,
    *,
    output_dir,
    previous_size,
    box_size,
    iteration,
    has_high_fsc_at_limit,
    incr_size,
    state,
):
    """Write the existing kclass current size NPZ schema."""
    import pathlib

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        pathlib.Path(output_dir) / f"recovar_kclass_current_size_it{iteration + 1:03d}.npz",
        iteration=np.int32(iteration + 1),
        previous_current_size=np.int32(previous_size),
        grid_size=np.int32(box_size),  # the box size, under the stored key the dump readers use
        resolution_shell=np.int32(plan.resolution_shell),
        per_class_resolution_shells=np.asarray(plan.resolution_shells_per_class, dtype=np.int32),
        ave_Pmax=np.float64(float(state.ave_Pmax)),
        state_current_resolution=np.float64(float(state.current_resolution)),
        state_previous_resolution=np.float64(float(state.previous_resolution)),
        relion_incr_size=np.int32(incr_size),
        relion_has_high_fsc_at_limit=np.int32(int(has_high_fsc_at_limit)),
        data_vs_prior_prev_raw=np.asarray(plan.raw_data_vs_prior, dtype=np.float32),
        data_vs_prior_prev=np.asarray(plan.data_vs_prior, dtype=np.float32),
        raw_current_size=np.int32(plan.raw_size),
        quantized_current_size=np.int32(plan.size),
    )


def write_class_mstep(
    prior: ClassPriorEstimate,
    *,
    numerators,
    denominators,
    half_denominators,
    references,
    settings: ReconstructionSettings,
    output_dir,
    class_index,
    current_size,
    iteration,
    source,
    accumulator_shape,
    full_half_axis,
    frame_scale,
):
    """Capture a class prior and its reconstruction operands in the M-step NPZ."""
    reconstruct_floor_stats_k = regularization_relion.compute_relion_weight_shell_stats(
        denominators[class_index],
        settings.volume_shape,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        shell_rounding="floor",
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_shape,
    )
    import pathlib

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    _preserve_kclass_dump_dtype = os.environ.get("RELAX_KCLASS_DUMP_PRESERVE_DTYPE", "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }
    dump_dtype = None if _preserve_kclass_dump_dtype else np.complex64
    np.savez(
        pathlib.Path(output_dir) / f"recovar_kclass_mstep_it{iteration + 1:03d}_c{class_index + 1:02d}.npz",
        iteration=np.int32(iteration + 1),
        class_index=np.int32(class_index + 1),
        current_size=np.int32(current_size),
        padding_factor=np.int32(settings.padding_factor),
        grid_size=np.int32(settings.box_size),  # the box size, under the stored key the dump readers use
        mstep_accumulator_shape=np.asarray(accumulator_shape, dtype=np.int32),
        mstep_full_half_axis=np.int32(full_half_axis),
        tau2_fudge=np.float64(settings.tau2_fudge),
        tau2_frame_scale=np.float64(frame_scale),
        previous_mean=np.asarray(references[0][class_index], dtype=np.complex64),
        previous_mean_half0=np.asarray(references[0][class_index], dtype=np.complex64),
        previous_mean_half1=np.asarray(references[1][class_index], dtype=np.complex64),
        Ft_y_combined=np.asarray(numerators[class_index], dtype=dump_dtype),
        Ft_ctf_0=(
            np.asarray(half_denominators[0][class_index], dtype=dump_dtype)
            if half_denominators[0] is not None
            else np.empty(0, dtype=np.complex64)
        ),
        Ft_ctf_1=(
            np.asarray(half_denominators[1][class_index], dtype=dump_dtype)
            if half_denominators[1] is not None
            else np.empty(0, dtype=np.complex64)
        ),
        Ft_ctf_combined=np.asarray(denominators[class_index], dtype=dump_dtype),
        dump_preserve_dtype=np.int32(int(_preserve_kclass_dump_dtype)),
        tau2_shells=np.asarray(prior.shells, dtype=np.float64),
        tau2_shells_relion=np.asarray(prior.relion_shells, dtype=np.float64),
        tau2_source=np.asarray(source),
        sigma2_shells=prior.details["sigma2_shells"],
        avg_weight_shells=prior.details["avg_weight_shells"],
        shell_sum=prior.details["shell_sum"],
        shell_count=prior.details["shell_count"],
        reconstruct_floor_avg_weight_shells=np.asarray(
            reconstruct_floor_stats_k["avg_weight_shells"],
            dtype=np.float64,
        ),
        reconstruct_floor_shell_count=np.asarray(
            reconstruct_floor_stats_k["shell_count"],
            dtype=np.float64,
        ),
        data_vs_prior=np.asarray(prior.data_vs_prior, dtype=np.float64),
    )


def write_premask_mean(
    mean, *, output_dir, half_index, iteration, current_size, box_size,
    voxel_size, volume_shape, n_classes,
):
    """Write pre-mask Fourier/real maps, preserving the diagnostic NPZ schema."""
    import pathlib

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    preserve_dtype = os.environ.get("RELAX_PREMASK_DUMP_PRESERVE_DTYPE", "").strip().lower() not in {
        "", "0", "false", "no", "off",
    }
    fourier = np.asarray(mean)
    if n_classes > 1:
        real = np.stack(
            [
                np.asarray(fourier_transform_utils.get_idft3(mean[class_idx].reshape(volume_shape))).real
                for class_idx in range(n_classes)
            ],
            axis=0,
        )
    else:
        real = np.asarray(fourier_transform_utils.get_idft3(mean.reshape(volume_shape))).real
    np.savez(
        pathlib.Path(output_dir) / f"recovar_premask_it{iteration + 1:03d}_half{half_index + 1}.npz",
        iteration=np.int32(iteration + 1),
        half=np.int32(half_index + 1),
        current_size=np.int32(current_size),
        grid_size=np.int32(box_size),  # the box size, under the stored key the dump readers use
        voxel_size=np.float32(voxel_size),
        volume_shape=np.asarray(volume_shape, dtype=np.int32),
        means_premask=fourier if preserve_dtype else np.asarray(fourier, dtype=np.complex64),
        means_premask_real=real if preserve_dtype else np.asarray(real, dtype=np.float32),
        dump_preserve_dtype=np.int32(int(preserve_dtype)),
    )
