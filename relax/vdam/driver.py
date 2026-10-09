"""Native InitialModel / ab-initio K-class driver.

This module owns the executable path behind ``relax.commands.initial_model``; all
data loading, denovo seeding, K-class E-step wiring, VDAM iteration and
artifact writing are coordinated here through their implementation owners.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, replace

import numpy as np
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.data_io.starfile import read_star

from relax.helpers.fourier_window import VDAM_STABLE_FOURIER_WINDOW_QUANTUM
from relax.helpers.particle_io import ParticleReadPolicy, assert_reads_from_scratch, image_star, prepare_particle_reads
from relax.refinement.optics_shapes import (
    MultiShapeDataset,
    average_ctf2_parts,
    optics_shape_class_rows,
    shape_datasets,
)
from relax.refinement.tomo_half import (
    TiltImageAccuracyInputs,
    TomoDataset,
    load_tomo_dataset,
    tilt_image_accuracy_inputs,
)
from relax.relion import initial_model_io, relion_ctf, vdam_checkpoint
from relax.relion.initial_model_io import (
    _experiment_read_order,
    _particle_state_from_star,
    _tomo_particle_state_from_star,
    tomo_checkpoint_particle_state,
)
from relax.relion.relion_metadata import (
    INITIAL_MODEL_OPTICS_FEATURES,
    TOMO_INITIAL_MODEL_OPTICS_FEATURES,
    refuse_unsupported_optics,
)
from relax.sparse_pass2.resident_pass2 import stable_window_class_history
from relax.vdam import estep_meta_updates, estep_setup, native_sampling, output, schedules
from relax.vdam.bootstrap_iref import (
    initial_state_from_particles,
    initial_state_from_tomo_particles,
    load_raw_images,
)
from relax.vdam.estep_setup import run_initial_model_estep
from relax.vdam.iteration_loop import MomentumSgdUpdate, VdamUpdate, run_vdam_iterations
from relax.vdam.m_step import prepare_mstep_state_precision, relion_solvent_flatten_state, relion_solvent_mask
from relax.vdam.native_options import NativeInitialModelOptions
from relax.vdam.native_sampling import (
    NativeSamplingState,
    build_sampling_plan,
    estimate_native_sampling_accuracy,
    initial_sampling_state,
    prepare_native_sampling_for_iteration,
    record_native_sampling_assignment_changes,
    record_native_sampling_post_iteration,
)
from relax.vdam.output import write_final_outputs, write_iteration_artifacts
from relax.vdam.ports import VdamInputSource
from relax.vdam.schedules import (
    DEFAULT_SIGMA2_FUDGE,
    default_subset_sizes_for_3d_initial_model,
    phase_lengths_from_effective_fractions,
)
from relax.vdam.state import InitialModelState, NativeOpticsState, NativeParticleState
from relax.vdam.subset_schedule import restore_subset_order_for_continuation
from relax.vdam.tomo_estep import run_tomo_initial_model_estep

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class NativeInitialModelResult:
    """Summary returned by ``run_native_initial_model``."""

    state: InitialModelState
    output_prefix: str
    final_model_star: str
    final_mrc: str
    class_mrcs: tuple[str, ...]


def _native_expectation_step(
    dataset,
    opts: NativeInitialModelOptions,
    particle_state: NativeParticleState,
    sampling_state: NativeSamplingState,
    optics_state: NativeOpticsState | None = None,
    *,
    projector_context: estep_setup.IterationProjectorContext,
    tilt_images: TiltImageAccuracyInputs | None = None,
    optics_group_ids: np.ndarray | None = None,
    premultiplied_ctf: bool = False,
):
    """VDAM's E-step closure; ``dataset`` is a ``TomoDataset`` for subtomogram particles, with ``tilt_images``.

    ``projector_context`` hands each E-step the projector its iteration's refresh prepared.

    ``optics_group_ids`` (several optics groups only) gives each particle-STAR row's zero-based
    group: its image is scored with its group's noise row and adds to its group's noise sums.
    ``premultiplied_ctf``: some images (tilt images for subtomograms) are CTF-premultiplied, so each
    E-step records the subset's average CTF^2 for the M-step's SSNR.
    """

    tomo = isinstance(dataset, TomoDataset)
    # The expected accuracy's optics-table CTFs: a subtomogram's ``tilt_images`` index the tilt images of
    # every particle in particle-STAR row order, i.e. this half's images.
    accuracy_dataset = dataset.subset(np.arange(dataset.n_units)).images if tomo else dataset

    def _expectation_step(state: InitialModelState, particle_ids: np.ndarray, halfset_ids: np.ndarray):
        sampling_kwargs = {"defer_fine_rotations": True}  # the adaptive route builds its own grids
        iteration = max(1, int(state.iter))
        do_grad = bool(opts.stochastic_all_iterations) or schedules.native_initialmodel_do_grad(
            state, iteration, grad_em_iters=int(opts.grad_em_iters)
        )
        accuracy_estimate = None
        prepared_projector_inputs = projector_context.take(state, padding_factor=int(opts.padding_factor))
        pass1_healpix_order = int(sampling_state.healpix_order)
        skip_expected_accuracy = opts.environment.skip_expected_accuracy
        if (
            (optics_state is not None or tilt_images is not None)
            and not skip_expected_accuracy
            and schedules.should_estimate_native_sampling_accuracy(
                iteration=iteration, nr_iter=int(state.nr_iter), do_grad=do_grad
            )
        ):
            # RELION expectationSetup constructs the production PPref
            # before calculateExpectedAngularErrors and reuses that PPref
            # for scoring: the projector refresh built it before this estimate.
            accuracy_estimate = estimate_native_sampling_accuracy(
                sampling_state,
                state,
                particle_state,
                optics_state,
                particle_order=np.asarray(particle_ids, dtype=np.int64),
                random_seed=int(opts.random_seed),
                padding_factor=int(opts.padding_factor),
                sigma2_fudge=DEFAULT_SIGMA2_FUDGE,
                tilt_images=tilt_images,
                optics_group_ids=optics_group_ids,
                experiment_dataset=accuracy_dataset,
                isolate_in_subprocess=opts.environment.isolate_expected_accuracy,
            )
            if accuracy_estimate is not None:
                sampling_state.acc_rot = accuracy_estimate.accuracy.acc_rot
                sampling_state.acc_trans_angstrom = accuracy_estimate.accuracy.acc_trans_angstrom
        sampling_updated = (
            prepare_native_sampling_for_iteration(sampling_state, state, iteration=iteration, do_grad=do_grad)
            if opts.fixed_healpix_order is None
            else False
        )
        sampling_plan = build_sampling_plan(
            opts,
            iteration=iteration,
            sampling_state=sampling_state,
            **sampling_kwargs,
        )
        sigma_offset_angstrom = float(np.sqrt(max(float(state.sigma2_offset), 0.0)))
        current_noise_variance = estep_setup.noise_variance_from_sigma2(state.sigma2_noise, int(state.box_size))
        previous_translations = np.asarray(particle_state.translation_offsets, dtype=np.float64).copy()
        previous_rotations = (
            None
            if particle_state.best_pose_rotations is None
            else np.asarray(particle_state.best_pose_rotations, dtype=np.float64).copy()
        )
        previous_classes = np.asarray(particle_state.class_assignments, dtype=np.int32).copy()
        if particle_state.visited is not None:
            # RELION's old class number is zero until the first visit. Our
            # scorer uses zero-based classes, so preserve that distinct old
            # state only in this change-monitor snapshot, before visits update.
            previous_classes[~np.asarray(particle_state.visited, dtype=bool)] = -1
        if tomo:
            max_significants = schedules.active_relion_initialmodel_max_significants(state, do_grad=do_grad)
            ids = np.asarray(particle_ids, dtype=np.int64)
            result = run_tomo_initial_model_estep(
                dataset,
                state,
                sampling_plan=sampling_plan,
                pass1_healpix_order=pass1_healpix_order,
                particle_ids=ids,
                halfset_ids=halfset_ids,
                previous_offsets_px=previous_translations[ids],
                noise_variance=current_noise_variance,
                relion_projector_half_by_class=prepared_projector_inputs[0],
                relion_projector_r_max=int(prepared_projector_inputs[1]),
                class_rotation_log_prior=native_sampling.class_rotation_log_prior_for_sampling(
                    state, sampling_state, int(sampling_plan.healpix_order)
                ),
                max_significants=int(max_significants),
                sigma_offset_angstrom=sigma_offset_angstrom,
                particle_diameter_ang=float(opts.particle_diameter),
                padding_factor=int(opts.padding_factor),
                optics_group_ids=optics_group_ids,
            )
            effective_image_batch_size = int(opts.image_batch_size)
            if premultiplied_ctf:
                # setAverageCTF2 over this iteration's images: every tilt image counts once.
                result.meta["premultiplied_average_ctf2"] = relion_ctf.premultiplied_average_ctf2(
                    average_ctf2_parts(
                        dataset.subset(ids).images,
                        None,
                        current_size=int(state.effective_current_size),
                        image_current_size=int(state.effective_current_size),
                    ),
                    int(state.box_size),
                )
        else:
            result = _spa_estep(
                state,
                particle_ids,
                halfset_ids,
                sampling_plan=sampling_plan,
                prepared_projector_inputs=prepared_projector_inputs,
                noise_variance=current_noise_variance,
                sigma_offset_angstrom=sigma_offset_angstrom,
                pass1_healpix_order=pass1_healpix_order,
                do_grad=do_grad,
                iteration=iteration,
            )
            result, effective_image_batch_size, max_significants = result
            if premultiplied_ctf:
                # The subset's average CTF^2 corrects the M-step's SSNR (setAverageCTF2 over
                # this iteration's images, ml_optimiser.cpp:5697-5740).
                result.meta["premultiplied_average_ctf2"] = relion_ctf.premultiplied_average_ctf2(
                    average_ctf2_parts(
                        dataset.subset(np.asarray(particle_ids, dtype=np.int64)),
                        None,
                        current_size=int(state.effective_current_size),
                        image_current_size=int(state.effective_current_size),
                    ),
                    int(state.box_size),
                )
        result.meta.update(
            random_perturbation=float(sampling_plan.random_perturbation),
            n_rotations=sampling_plan.n_rotations,
            n_translations=int(sampling_plan.translations.shape[0]),
            requested_image_batch_size=int(opts.image_batch_size),
            effective_image_batch_size=effective_image_batch_size,
            healpix_order=int(sampling_plan.healpix_order),
            oversampling=int(sampling_plan.oversampling),
            offset_range_px=float(sampling_plan.offset_range_px),
            offset_step_px=float(sampling_plan.offset_step_px),
            offset_range_angstrom=float(sampling_plan.offset_range_angstrom),
            offset_step_angstrom=float(sampling_plan.offset_step_angstrom),
            max_significants=int(max_significants),
            sigma_offset_angstrom=sigma_offset_angstrom,
            sigma2_offset_before=float(state.sigma2_offset),
        )
        return _after_estep(
            state,
            result,
            sampling_plan=sampling_plan,
            accuracy_estimate=accuracy_estimate,
            sampling_updated=sampling_updated,
            iteration=iteration,
            previous_translations=previous_translations,
            previous_rotations=previous_rotations,
            previous_classes=previous_classes,
        )

    def _spa_estep(
        state,
        particle_ids,
        halfset_ids,
        *,
        sampling_plan,
        prepared_projector_inputs,
        noise_variance,
        sigma_offset_angstrom,
        pass1_healpix_order,
        do_grad,
        iteration,
    ):
        config = estep_setup.initial_model_estep_config(
            dataset,
            opts,
            noise_variance,
            sampling_plan,
            particle_state.translation_offsets,
            sigma_offset_angstrom=sigma_offset_angstrom,
            pass1_healpix_order=pass1_healpix_order,
            max_significants=schedules.active_relion_initialmodel_max_significants(state, do_grad=do_grad),
        )
        if optics_group_ids is not None:
            config = replace(
                config, engine_kwargs={**config.engine_kwargs, "optics_group_ids": np.asarray(optics_group_ids, dtype=np.int32)}
            )
        prepared_half, prepared_r_max = prepared_projector_inputs
        config = replace(
            config,
            relion_projector_half_by_class=prepared_half,
            relion_projector_r_max=prepared_r_max,
        )
        class_rotation_log_prior = native_sampling.class_rotation_log_prior_for_sampling(
            state,
            sampling_state,
            int(sampling_plan.healpix_order),
        )
        config.engine_kwargs["class_rotation_log_prior"] = class_rotation_log_prior
        config.engine_kwargs["debug_iteration"] = iteration
        result = run_initial_model_estep(
            dataset, state, config, particle_ids=particle_ids, halfset_ids=halfset_ids
        )
        return result, int(config.image_batch_size), config.sampling.max_significants

    def _after_estep(
        state,
        result,
        *,
        sampling_plan,
        accuracy_estimate,
        sampling_updated,
        iteration,
        previous_translations,
        previous_rotations,
        previous_classes,
    ):
        result.meta["sampling_accuracy_estimated"] = accuracy_estimate is not None
        result.meta["sampling_accuracy_skipped_by_diagnostic"] = bool(opts.environment.skip_expected_accuracy)
        result.meta["sampling_accuracy_isolated_by_diagnostic"] = bool(opts.environment.isolate_expected_accuracy)
        if accuracy_estimate is not None:
            result.meta.update(accuracy_estimate.meta())
        result.meta.update(
            sampling_updated=bool(sampling_updated),
            effective_offset_step_angstrom=float(sampling_state.effective_offset_step_angstrom),
            sampling_acc_rot=float(sampling_state.acc_rot),
            sampling_acc_trans_angstrom=float(sampling_state.acc_trans_angstrom),
            sampling_nr_iter_wo_resol_gain=int(sampling_state.nr_iter_wo_resol_gain),
            sampling_has_fine_enough_angular_sampling=bool(sampling_state.has_fine_enough_angular_sampling),
            orientational_prior_mode=int(sampling_state.orientational_prior_mode),
            uniform_local_orientation_prior=bool(sampling_state.uniform_local_orientation_prior),
        )
        if (evidence := result.meta.get("log_evidence_per_image")) is not None:
            ids = np.asarray(result.meta["selected_particle_ids"], dtype=np.int64)
            result.meta["log_likelihood_contribution"] = estep_meta_updates.relion_log_likelihood_contributions(
                evidence,
                sigma2_noise=state.sigma2_noise,
                groups=np.zeros(ids.size, np.int64) if optics_group_ids is None else np.asarray(optics_group_ids)[ids],
                n_images=np.ones(ids.size) if tilt_images is None else np.diff(tilt_images.image_offsets)[ids],
                box_size=int(state.box_size),
                current_size=int(state.effective_current_size),
            )
        estep_meta_updates.update_particle_state_from_estep_meta(
            particle_state,
            result.meta,
            (
                sampling_plan.translations
                if sampling_plan.metadata_translations is None
                else sampling_plan.metadata_translations
            ),
        )
        # Monitor hidden-variable changes every iteration, independently of autosampling.
        if int(iteration) <= int(state.nr_iter):
            record_native_sampling_assignment_changes(
                sampling_state,
                particle_ids=result.meta.get("selected_particle_ids"),
                previous_translations=previous_translations,
                current_translations=particle_state.translation_offsets,
                previous_rotations=previous_rotations,
                current_rotations=particle_state.best_pose_rotations,
                previous_classes=previous_classes,
                current_classes=particle_state.class_assignments,
            )
        result.meta["current_changes_optimal_offsets_angstrom"] = float(
            sampling_state.current_changes_optimal_offsets_angstrom
        )
        result.meta["current_changes_optimal_orientations"] = float(
            sampling_state.current_changes_optimal_orientations
        )
        result.meta["current_changes_optimal_classes"] = float(sampling_state.current_changes_optimal_classes)
        result.meta["sampling_nr_iter_wo_large_hidden_variable_changes"] = int(
            sampling_state.nr_iter_wo_large_hidden_variable_changes
        )
        result.meta["sampling_smallest_changes_optimal_offsets_angstrom"] = float(
            sampling_state.smallest_changes_optimal_offsets_angstrom
        )
        result.meta["sampling_smallest_changes_optimal_orientations"] = float(
            sampling_state.smallest_changes_optimal_orientations
        )
        result.meta["sampling_smallest_changes_optimal_classes"] = float(
            sampling_state.smallest_changes_optimal_classes
        )
        return result.accumulators, result.meta

    return _expectation_step


def _should_write_iteration_artifacts(iteration: int, nr_iter: int, grad_write_iter: int) -> bool:
    """Match RELION's gradient-output cadence, including the final iteration (``validate_run`` refuses
    ``grad_write_iter < 1``)."""

    return (iteration % grad_write_iter) == 0 or iteration == nr_iter


def _refuse_unsupported_multi_shape(opts: NativeInitialModelOptions, datasets) -> None:
    """Optics groups on several image shapes run the VDAM optimizer without optics-table CTF terms."""

    from relax.relion.optics_aberrations import dataset_needs_exact_ctf

    if opts.optimizer != "vdam" or opts.diagnostic_continue_optimiser is not None:
        raise NotImplementedError("optics groups on several image shapes run only fresh VDAM InitialModel")
    if any(dataset_needs_exact_ctf(d) for d in datasets):
        raise NotImplementedError(
            "optics groups on several image shapes do not yet take premultiplied or aberrated optics groups"
        )


def run_native_initial_model(
    opts: NativeInitialModelOptions, *, source: VdamInputSource | None = None
) -> NativeInitialModelResult:
    """Run native recovar InitialModel refinement.

    ``source`` (None: the native :class:`VdamInputSource`) is the run's input source, which the command chose.
    """

    source = VdamInputSource() if source is None else source
    profile = output.StageProfile(opts.environment.profile)

    if int(opts.random_seed) == -1:
        # relion_refine's default --random_seed -1 takes the time (ml_optimiser.cpp:2827).
        opts = replace(opts, random_seed=int(time.time()))
        logger.info("InitialModel random seed %d (RELION default -1: the time)", opts.random_seed)
    opts.validate_run()
    symmetry_warning = opts.symmetry_mode_warning()
    if symmetry_warning is not None:
        logger.warning(symmetry_warning)
    profile.record("validation")

    main_star, optics_star = read_star(opts.fn_img)
    # Single particles take CTF-premultiplied images and odd and even aberrations, as Class3D does, and
    # refuse magnification; subtomograms take the features whose per-tilt path is qualified.
    refuse_unsupported_optics(
        optics_star,
        source=str(opts.fn_img),
        supported=TOMO_INITIAL_MODEL_OPTICS_FEATURES if opts.fn_tomograms is not None else INITIAL_MODEL_OPTICS_FEATURES,
    )
    particle_order = _experiment_read_order(main_star)
    profile.record("input_star")
    particle_read_policy = ParticleReadPolicy(
        preread_images=bool(opts.preread_images),
        scratch_dir=str(opts.scratch_dir or ""),
        keep_free_scratch_gb=float(opts.keep_free_scratch_gb),
    )
    tomo = opts.fn_tomograms is not None
    # A subtomogram particle STAR names no image stacks; load_tomo_dataset stages from its per-tilt STAR.
    particle_scratch = None if tomo else prepare_particle_reads(
        opts.fn_img,
        particle_read_policy,
        datadir=opts.datadir,
        strip_prefix=opts.strip_prefix,
    )
    profile.record("particle_scratch")
    tilt_images: TiltImageAccuracyInputs | None = None
    if tomo:
        # RELION 5 subtomogram 2D stacks (--ios): the particles are the units, each over its tilt images.
        dataset = load_tomo_dataset(
            opts.fn_img,
            opts.fn_tomograms,
            os.path.join(os.path.dirname(os.path.abspath(opts.outputname)), "particles_2d.star"),
            datadir=opts.datadir or os.path.dirname(os.path.abspath(opts.fn_img)),
            lazy=not particle_read_policy.preread_images,
            read_policy=particle_read_policy,
        )
        image_dataset = dataset.images
        n_particles = int(dataset.n_units)
        tilt_images = tilt_image_accuracy_inputs(dataset.subset(np.arange(n_particles)))
    else:
        shape_class_rows = optics_shape_class_rows(opts.fn_img)
        # Images come through the scratch STAR copy when --scratch_dir staged compact stacks; image
        # names, CTF and poses for every output keep coming from opts.fn_img.
        images_star = image_star(opts.fn_img, particle_scratch)
        compact = images_star != opts.fn_img
        datasets = [
            load_dataset(
                images_star,
                lazy=not particle_read_policy.preread_images,
                datadir=None if compact else opts.datadir,
                strip_prefix=None if compact else opts.strip_prefix,
                ind=rows,
                # Each missing rlnAngleRot/Tilt/Psi label reads as 0 (exp_model.cpp:1104-1106, 1131-1136).
                absent_angles_zero=True,
            )
            for rows in (shape_class_rows or [None])
        ]
        if any(getattr(d, "tilt_series_flag", False) for d in datasets):
            raise NotImplementedError("native InitialModel reads RELION 5 2D stacks (--ios), not tilt-series STAR files")
        # Optics groups on several image shapes: one dataset per shape (RELION S3b).
        dataset = image_dataset = (
            datasets[0] if shape_class_rows is None else MultiShapeDataset(datasets, shape_class_rows)
        )
        if shape_class_rows is not None:
            _refuse_unsupported_multi_shape(opts, datasets)
        n_particles = int(dataset.n_images)
    for class_dataset in getattr(image_dataset, "datasets", (image_dataset,)):
        assert_reads_from_scratch(class_dataset, particle_scratch)
        estep_setup.configure_relion_image_mask(class_dataset, opts)
    profile.record("dataset_load")

    optics_state = None if tomo else initial_model_io._native_optics_state(main_star, optics_star, dataset)
    continuation = None
    if opts.diagnostic_continue_optimiser is not None:
        continuation = vdam_checkpoint._load_native_vdam_continuation(
            opts.diagnostic_continue_optimiser,
            expected_data_star=opts.fn_img,
            opts=opts,
            dataset=dataset,
        )
        if int(opts.diagnostic_stop_after_iteration) != int(continuation.iteration) + 1:
            raise ValueError(
                "diagnostic native VDAM continuation must execute exactly one next iteration: "
                f"checkpoint={continuation.iteration}, "
                f"stop={opts.diagnostic_stop_after_iteration}"
            )
    particle_state = (
        (tomo_checkpoint_particle_state if continuation is not None else _tomo_particle_state_from_star)(
            main_star, pixel_size=float(dataset.voxel_size)
        )
        if tomo
        else _particle_state_from_star(
            main_star,
            dataset,
            allow_unvisited_class_zero=continuation is not None,
            nr_classes=int(opts.nr_classes),
        )
    )
    if continuation is None:
        grad_ini_subset_size, grad_fin_subset_size = (
            default_subset_sizes_for_3d_initial_model(n_particles)
            if opts.pilot_controls is None
            else opts.pilot_controls.subset_sizes(n_particles)
        )
        grad_ini_frac = opts.grad_ini_frac
        grad_fin_frac = opts.grad_fin_frac
        continuation_phase_lengths = None
        sampling_state = initial_sampling_state(opts, pixel_size=float(dataset.voxel_size), subtomogram=tomo)
        state, optics_group_by_particle = (
            initial_state_from_tomo_particles(dataset, main_star, opts)
            if tomo
            else initial_state_from_particles(dataset, main_star, optics_star, opts, source=source)
        )
        sampling_state.last_current_resolution = float(state.current_resolution)
    else:
        vdam_checkpoint._validate_continuation_order_replay(continuation)
        grad_ini_subset_size = continuation.grad_ini_subset_size
        grad_fin_subset_size = continuation.grad_fin_subset_size
        grad_ini_frac = continuation.grad_ini_frac
        grad_fin_frac = continuation.grad_fin_frac
        continuation_phase_lengths = phase_lengths_from_effective_fractions(
            int(continuation.state.nr_iter),
            grad_ini_frac,
            grad_fin_frac,
        )
        optics_group_by_particle = initial_model_io._optics_group_indices(main_star)
        if int(np.unique(optics_group_by_particle).size) != 1:
            raise NotImplementedError(
                "diagnostic native VDAM continuation currently supports one optics group"
            )
        # Diagnostic: stock RELION --continue reshuffles from the input order (sorted_idx is not checkpointed).
        state = continuation.state if opts.diagnostic_continue_input_order else restore_subset_order_for_continuation(
            continuation.state,
            through_iteration=int(continuation.iteration),
            nr_particles=int(n_particles),
            optics_group_by_particle=optics_group_by_particle,
            grad_ini_subset_size=grad_ini_subset_size,
            grad_fin_subset_size=grad_fin_subset_size,
            random_seed=int(opts.random_seed),
            particle_order=particle_order,
            grad_ini_frac=grad_ini_frac,
            grad_fin_frac=grad_fin_frac,
            grad_em_iters=int(opts.grad_em_iters),
            phase_lengths=continuation_phase_lengths,
        )
        sampling_state = continuation.sampling_state
    state = prepare_mstep_state_precision(state, opts.mstep_compute_dtype)
    if opts.optimizer == "momentum_sgd" and int(np.unique(optics_group_by_particle).size) > 1:
        raise NotImplementedError("the momentum-SGD InitialModel takes one optics group")
    if opts.optimizer == "momentum_sgd":
        from relax.vdam.sgd import corner_white_sigma2, initialize_sgd_noise

        corner_count = min(int(opts.sigma2_min_particles), int(dataset.n_images))
        corner_images = load_raw_images(
            dataset, particle_order[:corner_count], batch_size=max(1, int(opts.image_batch_size))
        )
        state = initialize_sgd_noise(
            state,
            corner_white_sigma2(
                corner_images,
                dataset.image_source.backend.image_mask,
                image_multiplier=dataset.image_source.backend.mult,
            ),
        )
    profile.record("state_setup")
    projector_context = estep_setup.IterationProjectorContext()
    expectation_step = _native_expectation_step(
        dataset,
        opts,
        particle_state,
        sampling_state,
        optics_state,
        projector_context=projector_context,
        tilt_images=tilt_images,
        optics_group_ids=optics_group_by_particle if int(np.unique(optics_group_by_particle).size) > 1 else None,
        premultiplied_ctf=any(
            relion_ctf.dataset_has_premultiplied_ctf(d, tuple(int(v) for v in d.image_shape))
            for d in shape_datasets(image_dataset)
        ),
    )
    profile.record("expectation_setup")

    if opts.write_iter_artifacts:
        output.write_initial_run_metadata(opts, continuation)
        if continuation is None:
            initial_meta = {"checkpoint_iteration": 0, "phase": "bootstrap"}
            if opts.fourier_radius_schedule is not None or opts.stochastic_all_iterations:
                initial_meta["initial_iref_sha256"] = hashlib.sha256(
                    np.ascontiguousarray(np.asarray(state.Iref)).tobytes()
                ).hexdigest()
            write_iteration_artifacts(
                opts.outputname,
                state,
                0,
                initial_meta,
                main_star=main_star,
                optics_star=optics_star,
                dataset=dataset,
                particle_state=particle_state,
                profile_stages=opts.environment.profile,
            )
    profile.record("initial_artifacts")

    def record_iteration(current, _iteration, meta):
        record_native_sampling_post_iteration(sampling_state, current, meta=meta)

    written_iterations: set[int] = set()  # the iterations whose class maps and model.star this run wrote

    def artifact_sink(current, iteration, meta):
        if not opts.write_iter_artifacts or not _should_write_iteration_artifacts(
            iteration, int(opts.nr_iter), int(opts.grad_write_iter)
        ):
            return
        written_iterations.add(int(iteration))
        write_iteration_artifacts(
            opts.outputname,
            current,
            iteration,
            meta,
            main_star=main_star,
            optics_star=optics_star,
            dataset=dataset,
            particle_state=particle_state,
            profile_stages=opts.environment.profile,
        )

    solvent_mask = None
    if opts.do_solvent:
        solvent_mask = relion_solvent_mask(
            box_size=int(state.box_size),
            pixel_size=float(state.pixel_size),
            particle_diameter_ang=float(opts.particle_diameter),
            width_mask_edge_px=float(opts.width_mask_edge_px),
        )

    def post_mstep_update(current, iteration, meta):
        if solvent_mask is not None:
            current = relion_solvent_flatten_state(
                current,
                mask=solvent_mask,
                compute_dtype=opts.mstep_compute_dtype,
            )
        return source.iteration_references(current, iteration=int(iteration), meta=meta)
    profile.record("iteration_setup")

    # One stable-window class history for the run, as refinements have, on VDAM's ladder.
    with stable_window_class_history(quantum=VDAM_STABLE_FOURIER_WINDOW_QUANTUM):
        final_state = run_vdam_iterations(
            state,
            nr_particles=n_particles,
            optics_group_by_particle=optics_group_by_particle,
            grad_ini_subset_size=grad_ini_subset_size,
            grad_fin_subset_size=grad_fin_subset_size,
            pilot_controls=opts.pilot_controls,
            tau2_fudge_arg=float(opts.tau2_fudge),
            grad_em_iters=int(opts.grad_em_iters),
            random_seed=int(opts.random_seed),
            expectation_step=expectation_step,
            iter_artifact_sink=artifact_sink,
            record_iteration=record_iteration,
            post_mstep_update=post_mstep_update,
            particle_order=particle_order,
            grad_ini_frac=grad_ini_frac,
            grad_fin_frac=grad_fin_frac,
            phase_lengths=continuation_phase_lengths,
            grad_stepsize=float(opts.stepsize),
            mu=float(opts.mu),
            projector_padding_factor=int(opts.padding_factor),
            update=(
                VdamUpdate(
                    padding_factor=int(opts.padding_factor),
                    mstep_compute_dtype=opts.mstep_compute_dtype,
                    single_class_m_step=source.single_class_m_step,
                )
                if opts.optimizer == "vdam"
                else MomentumSgdUpdate(learning_rate=float(opts.sgd_learning_rate), padding_factor=int(opts.padding_factor))
            ),
            projector_refresh_fn=projector_context.refresh,
            start_iteration=int(state.iter),
            diagnostic_stop_after_iteration=opts.diagnostic_stop_after_iteration,
            fourier_radius_schedule=opts.fourier_radius_schedule,
            stochastic_all_iterations=bool(opts.stochastic_all_iterations),
            uniform_class_direction_prior=bool(opts.uniform_class_direction_prior),
            environment=opts.environment,
        )
    profile.record("iterations")
    if opts.pilot_controls is not None:
        opts.pilot_controls.check_completed(final_state.iter, opts.nr_iter)
    final_mrc, class_mrcs, align_report = write_final_outputs(
        opts.outputname,
        final_state,
        sym_name=opts.sym_name,
        seed=int(opts.random_seed),
        written_iterations=frozenset(written_iterations),
    )
    if "refined_rot_tilt_psi" in align_report:
        print("InitialModel symmetry alignment: " + json.dumps(align_report, sort_keys=True), flush=True)
    final_model_star = f"{opts.outputname}_it{final_state.iter:03d}_model.star"
    profile.record("final_artifacts")
    profile.report("driver")
    return NativeInitialModelResult(
        state=final_state,
        output_prefix=opts.outputname,
        final_model_star=final_model_star,
        final_mrc=final_mrc,
        class_mrcs=class_mrcs,
    )


__all__ = [
    "NativeInitialModelResult",
    "run_native_initial_model",
]
