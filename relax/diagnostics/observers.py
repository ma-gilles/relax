"""The run observers the command chooses (code rule 15; the port is ``relax.refinement.ports.RunObserver``).

``IntermediatesObserver`` writes ``--save_intermediates_dir``: every numbered iteration's maps, accumulators,
FSC, noise, prior, assignments and grids, each half's particle state and scoring manifest, the local-search
profiles and the final pass's manifests. ``ParityDumpObserver`` writes the per-iteration parity capture
(``RELAX_PARITY_DUMP_DIR``), with the per-half E-step capture it collects, or its stage timings alone
(``RELAX_PARITY_TIMING_DIR``, ``--timing_dir``). ``BpRefDiagnosticsObserver`` flushes the BPref device-panel
capture of each numbered half.
``ObserverGroup`` hands every hook to several observers. ``command_observer`` builds the observer of a
``relax refine`` / ``relax class3d`` command line. The algorithm's modules never import this module.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from relax.diagnostics import bpref_diagnostics, parity_dump
from relax.diagnostics.iteration import (
    dump_numbered_iteration,
    maybe_dump_noise_update_debug,
    save_iteration_intermediates,
    save_iteration_particle_states,
    write_final_half_manifest,
    write_numbered_half_manifest,
)
from relax.diagnostics.k_class_diagnostics import KClassDiagnostics
from relax.diagnostics.reconstruction import (
    write_bpref_accumulators,
    write_class_image_size,
    write_class_mstep,
    write_premask_mean,
)
from relax.refinement.ports import RunObserver
from relax.relion.geometry import RECONSTRUCTION_PADDING_FACTOR

logger = logging.getLogger(__name__)


_SIGNIFICANCE_DUMP_TARGET_HALF_ENV = "RELAX_SIGNIFICANCE_DUMP_TARGET_HALF"


_PASS2_NORM_DUMP_TARGET_HALF_ENV = "RELAX_PASS2_DUMP_TARGET_HALF"


def _source_image_indices(dataset) -> np.ndarray:
    """Original-stack image indices of a half dataset, in its local order."""

    index_layout = getattr(dataset, "_index_layout", None)
    if index_layout is None:
        return np.arange(dataset.n_images, dtype=np.int64)
    return np.asarray(
        index_layout.original_image_indices_for_local(np.arange(dataset.n_images, dtype=np.int32)),
        dtype=np.int64,
    )


def _significance_dump_half_indices(
    *,
    numbered_iteration: int,
    n_classes: int,
    experiment_datasets,
    environ=None,
) -> tuple[int, ...]:
    """Select one half only at an explicitly terminating diagnostic boundary."""

    env = os.environ if environ is None else environ
    raw_significance_half = str(env.get(_SIGNIFICANCE_DUMP_TARGET_HALF_ENV, "")).strip()
    raw_pass2_half = str(env.get(_PASS2_NORM_DUMP_TARGET_HALF_ENV, "")).strip()
    if raw_significance_half and raw_pass2_half:
        raise RuntimeError(
            f"{_SIGNIFICANCE_DUMP_TARGET_HALF_ENV} and "
            f"{_PASS2_NORM_DUMP_TARGET_HALF_ENV} are mutually exclusive"
        )
    if not raw_significance_half and not raw_pass2_half:
        return (0, 1)
    pass2_norm_mode = bool(raw_pass2_half)
    target_half_env = (
        _PASS2_NORM_DUMP_TARGET_HALF_ENV
        if pass2_norm_mode
        else _SIGNIFICANCE_DUMP_TARGET_HALF_ENV
    )
    raw_half = raw_pass2_half if pass2_norm_mode else raw_significance_half
    if pass2_norm_mode:
        if str(env.get("RELAX_PASS2_DUMP_NORM_RESIDUAL_INPUTS", "")).strip() != "1":
            raise RuntimeError(
                f"{_PASS2_NORM_DUMP_TARGET_HALF_ENV} requires "
                "RELAX_PASS2_DUMP_NORM_RESIDUAL_INPUTS=1"
            )
        if str(env.get("RELAX_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET", "")).strip() != "1":
            raise RuntimeError(
                f"{_PASS2_NORM_DUMP_TARGET_HALF_ENV} requires "
                "RELAX_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET=1"
            )
    elif str(env.get("RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "")).strip() != "1":
        raise RuntimeError(
            f"{_SIGNIFICANCE_DUMP_TARGET_HALF_ENV} requires "
            "RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET=1"
        )
    if int(n_classes) != 1:
        raise RuntimeError(f"{target_half_env} is K=1 diagnostic-only")
    try:
        target_half = int(raw_half)
    except ValueError as exc:
        raise ValueError(f"{target_half_env} must be 1 or 2") from exc
    if target_half not in {1, 2}:
        raise ValueError(f"{target_half_env} must be 1 or 2")

    prefix = "RELAX_PASS2_DUMP" if pass2_norm_mode else "RELAX_SIGNIFICANCE_DUMP"
    raw_iteration = str(env.get(f"{prefix}_ITERATION", "")).strip()
    raw_targets = str(env.get(f"{prefix}_ORIGINAL_INDICES", "")).strip()
    dump_dir = str(env.get(f"{prefix}_DIR", "")).strip()
    if not raw_iteration or not raw_targets or not dump_dir:
        raise RuntimeError(
            f"{target_half_env} requires an explicit dump directory, "
            "iteration, and original-index target set"
        )
    try:
        target_iteration = int(raw_iteration)
        target_indices = {
            int(token) for token in raw_targets.replace(",", " ").split()
        }
    except ValueError as exc:
        raise ValueError(f"{prefix}_ITERATION and {prefix}_ORIGINAL_INDICES must be integers") from exc
    if target_iteration <= 0 or not target_indices:
        raise ValueError(f"{prefix}_ITERATION must be positive and {prefix}_ORIGINAL_INDICES nonempty")
    if int(numbered_iteration) != target_iteration:
        return (0, 1)

    selected_half_indices = set(
        np.asarray(experiment_datasets[target_half - 1].dataset_indices, dtype=np.int64).tolist()
    )
    missing = sorted(target_indices - selected_half_indices)
    if missing:
        raise RuntimeError(
            f"{prefix}_ORIGINAL_INDICES targets are not all present in the selected half: "
            f"half={target_half} missing={missing}"
        )
    logger.info(
        "RECOVAR %s diagnostic: iteration=%d entering only half=%d "
        "for %d target particles; completion is mandatory before returning from the half",
        "pass-2 norm operand" if pass2_norm_mode else "coarse-significance",
        int(numbered_iteration),
        target_half,
        len(target_indices),
    )
    return (target_half - 1,)


class IntermediatesObserver(RunObserver):
    """Writes the run's intermediates under ``directory`` (created now); ``skip_unregularized`` saves the
    regularized maps only, so the run does not reconstruct the unregularized ones for it."""

    keeps_rotation_posteriors = True
    collects_local_search_profiles = True

    def __init__(self, directory, *, skip_unregularized: bool = False):
        self.directory = str(directory)
        self.skip_unregularized = bool(skip_unregularized)
        os.makedirs(self.directory, exist_ok=True)

    def wants_unfiltered_maps(self, numbered_relion_iteration: int) -> bool:
        return not self.skip_unregularized

    def maps_reconstructed(self, maps):
        save_iteration_intermediates(
            self.directory, maps.numerators, maps.denominators, maps.reference_model, maps.noise_model,
            maps.per_half, maps.trial_grid, maps.sampling_plan, maps.options, iteration=maps.iteration,
            unreg_means=maps.unfiltered_maps, fsc=maps.fsc, cs=maps.current_size, state=maps.state,
            volume_shape=maps.volume_shape, voxel_size=maps.voxel_size,
        )

    def poses_updated(self, iteration, *, poses, per_half, significance, datasets):
        save_iteration_particle_states(
            self.directory, poses, per_half, significance, iteration=iteration,
            original_image_indices_per_half=[_source_image_indices(dataset) for dataset in datasets],
        )

    def dense_half_scored(self, scored):
        write_numbered_half_manifest(self.directory, scored)

    def local_search_profile(self, iteration, half_index, profile):
        np.savez_compressed(
            os.path.join(self.directory, f"it{iteration:03d}_half{half_index + 1}_local_profile.npz"), **profile,
        )

    def final_half_scored(self, scored):
        if scored.final_inputs is None:
            raise NotImplementedError("--save_intermediates_dir has no final all-data manifest for subtomograms")
        write_final_half_manifest(
            self.directory, scored.half, scored.final_sampling, scored.final_inputs,
            translation_search_base=scored.translation_search_base, reference=scored.reference,
            reference_model=scored.reference_model, noise_variance=scored.noise_variance,
            current_size=scored.current_size, precision=scored.precision, use_local=scored.use_local, log=logger,
        )


class ParityDumpObserver(RunObserver):
    """The per-iteration parity capture and stage timings of ``relax.diagnostics.parity_dump``.

    The capture (``RELAX_PARITY_DUMP_DIR``) writes each numbered iteration's state, maps and poses and asks for
    the unregularized maps, read from the environment when each hook runs, as the module does; without it,
    ``timing_dir`` (``--timing_dir`` or ``RELAX_PARITY_TIMING_DIR``, chosen by :func:`observers_from_environment`;
    None: no timing rows) receives each iteration's stage times.
    """

    def __init__(self, timing_dir: Path | None):
        self.timing_dir = timing_dir

    def wants_unfiltered_maps(self, numbered_relion_iteration: int) -> bool:
        return parity_dump.is_active()

    @property
    def keeps_pose_rotation_eulers(self):
        return parity_dump.is_active()

    def iteration_started(self, iteration):
        parity_dump.start_iteration(iteration)

    def numbered_half_recorded(self, recorded):
        # The E-step capture of one half; an empty half records its own noise statistics and no image rows.
        k, per_half, dataset = recorded.half_index, recorded.per_half, recorded.particles.dataset
        if dataset.n_units == 0:
            noise_stats, original_image_indices = recorded.score_result.noise_stats, np.zeros(0, dtype=np.int64)
        else:
            noise_stats = per_half.noise_stats[k]
            # The parity dump records each image's row in the input stack; a dataset without an index layout has
            # none.
            index_layout = getattr(dataset, "_index_layout", None)
            original_image_indices = (
                np.asarray(
                    index_layout.original_image_indices_for_local(np.arange(dataset.n_images, dtype=np.int32)),
                    dtype=np.int64,
                )
                if index_layout is not None and parity_dump.is_active()
                else None
            )
        parity_dump.collect_e_step(
            half=k,
            em_stats=recorded.score_result.em_stats,
            hard_assignment=recorded.score_result.ha,
            coarse_hard_assignment=per_half.coarse_ha[k],
            noise_stats=noise_stats,
            Ft_y=recorded.score_result.Ft_y,
            Ft_ctf=recorded.score_result.Ft_ctf,
            pose_rotation_eulers=per_half.pose_rotation_eulers[k],
            best_pose_rotation_eulers=per_half.best_pose_rotation_eulers[k],
            best_pose_translations=per_half.best_pose_translations[k],
            translation_search_base=per_half.translation_search_bases[k],
            original_image_indices=original_image_indices,
        )

    def stage_finished(self, iteration, stage):
        parity_dump.mark_stage(iteration, stage)

    def iteration_finished(self, finished):
        if parity_dump.is_active():
            dump_numbered_iteration(
                finished.iteration, init_relion_iteration=finished.init_relion_iteration, state=finished.state,
                current_size=finished.current_size, sigma_offset_angstrom=finished.sigma_offset_angstrom,
                random_perturbation=finished.random_perturbation, settings=finished.settings,
                pixel_size_angstrom=finished.pixel_size_angstrom, ave_pmax=finished.ave_pmax, fsc=finished.fsc,
                noise_variance=finished.noise_variance, means=finished.means,
                unfiltered_means=finished.unfiltered_means, poses=finished.poses, half_inputs=finished.half_inputs,
                corrections=finished.corrections, scale_correction_data_vs_prior=finished.scale_correction_data_vs_prior,
                log=logger,
            )
        elif self.timing_dir is not None:
            try:
                parity_dump.dump_timing_iteration(
                    out=self.timing_dir,
                    iteration=finished.iteration,
                    init_relion_iteration=int(finished.init_relion_iteration),
                )
            except Exception as exc:
                logger.warning("parity_dump.dump_timing_iteration failed at iter %d: %s", finished.iteration, exc)


class BpRefAccumulatorObserver(RunObserver):
    """Writes a K=1 iteration's half accumulators before the join (``prejoin_dir``) and after it
    (``accum_dir``), each ``recovar_bpref_<stage>_itNNN.npz``; ``target_iteration`` (1-based) limits both to one
    iteration. From the environment: ``RELAX_BPREF_PREJOIN_DUMP_DIR``, ``RELAX_BPREF_ACCUM_DUMP_DIR``,
    ``RELAX_BPREF_BOUNDARY_DUMP_ITERATION``."""

    def __init__(self, *, prejoin_dir=None, accum_dir=None, target_iteration=None):
        self.prejoin_dir, self.accum_dir = prejoin_dir or None, accum_dir or None
        self.target_iteration = None if target_iteration is None else int(target_iteration)

    @classmethod
    def from_environment(cls):
        target = os.environ.get("RELAX_BPREF_BOUNDARY_DUMP_ITERATION")
        return cls(
            prejoin_dir=os.environ.get("RELAX_BPREF_PREJOIN_DUMP_DIR"),
            accum_dir=os.environ.get("RELAX_BPREF_ACCUM_DUMP_DIR"),
            target_iteration=int(target) if target else None,
        )

    def _write(self, directory, stage, iteration, numerators, denominators, settings, current_size,
               accumulator_shape, pixel_size_angstrom):
        if not directory or (self.target_iteration is not None and iteration + 1 != self.target_iteration):
            return
        write_bpref_accumulators(
            directory, stage=stage, iteration=iteration, current_size=current_size,
            padding_factor=settings.padding_factor if stage == "prejoin" else RECONSTRUCTION_PADDING_FACTOR,
            box_size=settings.box_size, voxel_size=pixel_size_angstrom, volume_shape=settings.volume_shape,
            accumulator_shape=accumulator_shape, Ft_y_0=numerators[0], Ft_y_1=numerators[1],
            Ft_ctf_0=denominators[0], Ft_ctf_1=denominators[1],
        )

    def run_started(self, context):
        # The prejoin capture is K=1's; the run's mode is read once from its context.
        self.k_class_enabled = context.k_class_enabled

    def half_accumulators_ready(self, iteration, *, numerators, denominators, settings, current_size,
                                accumulator_shape, pixel_size_angstrom):
        if not self.k_class_enabled:
            self._write(self.prejoin_dir, "prejoin", iteration, numerators, denominators, settings, current_size,
                        accumulator_shape, pixel_size_angstrom)

    def k1_accumulators_joined(self, iteration, *, numerators, denominators, settings, current_size,
                               accumulator_shape, pixel_size_angstrom):
        self._write(self.accum_dir, "accum", iteration, numerators, denominators, settings, current_size,
                    accumulator_shape, pixel_size_angstrom)


class BpRefDiagnosticsObserver(RunObserver):
    """Flushes the BPref device-panel capture of each scored numbered half (``RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR``,
    limited by ``RELAX_BPREF_CONTRIBUTION_DUMP_ITERATION`` and ``_HALF``), and says when a BPref capture or
    diagnostic override (``bpref_diagnostics.BPREF_DIAGNOSTIC_ENVS``) is armed for a numbered half, read from the
    environment at each call.

    An armed capture (among them RELAX_BPREF_ACCUMULATOR_DELTA_DUMP_DIR, RELAX_SPARSE_PASS2_NATIVE_DUMP_DIR and
    RELAX_BPREF_HIGH_PRECISION_OPERAND_BUNDLE) forces that half's first-iteration BPref onto direct planning, where
    the captures are taken; direct planning matches compact planning to float noise, not bit for bit."""

    def bpref_diagnostics_armed(self, iteration, half_index, *, bpref_device_signature_active):
        return bpref_diagnostics.relion_firstiter_bpref_diagnostics_active(
            bpref_device_signature_active=bpref_device_signature_active, iteration=iteration + 1, half=half_index + 1,
        )

    def numbered_half_recorded(self, recorded):
        if recorded.particles.dataset.n_units != 0:
            bpref_diagnostics.flush_selected_bpref_device_panel(
                iteration_index=recorded.iteration, half_index=recorded.half_index,
            )


class NoiseUpdateObserver(RunObserver):
    """Writes the raw terms of each noise update (``RELAX_NOISE_DEBUG_DUMP_DIR``, optionally limited to the
    iterations of ``RELAX_NOISE_DEBUG_DUMP_ITERATION``), read from the environment at each update."""

    def noise_updated(self, iteration, **terms):
        maybe_dump_noise_update_debug(iteration=iteration, **terms)


class ClassDumpObserver(RunObserver):
    """Writes each Class3D iteration's image-size plan and each class's prior under ``directory``
    (``RELAX_KCLASS_DUMP_DIR``; ``recovar_kclass_current_size_itNNN.npz``, ``recovar_kclass_mstep_itNNN_cKK.npz``)."""

    def __init__(self, directory):
        self.directory = str(directory)

    def class_image_size_planned(self, iteration, plan, *, previous_size, box_size, has_high_fsc_at_limit,
                                 incr_size, state):
        write_class_image_size(
            plan, output_dir=self.directory, previous_size=previous_size, box_size=box_size, iteration=iteration,
            has_high_fsc_at_limit=has_high_fsc_at_limit, incr_size=incr_size, state=state,
        )

    def class_prior_estimated(self, estimated):
        write_class_mstep(
            estimated.prior, numerators=estimated.numerators, denominators=estimated.denominators,
            half_denominators=estimated.half_denominators, references=estimated.references,
            settings=estimated.settings, output_dir=self.directory, class_index=estimated.class_index,
            current_size=estimated.current_size, iteration=estimated.iteration, source=estimated.source,
            accumulator_shape=estimated.accumulator_shape, full_half_axis=estimated.full_half_axis,
            frame_scale=estimated.frame_scale,
        )


class PremaskObserver(RunObserver):
    """Writes each half's solved numbered map before its low-pass and solvent mask under ``directory``
    (``RELAX_PREMASK_DUMP_DIR``)."""

    def __init__(self, directory):
        self.directory = str(directory)

    def map_solved(self, iteration, half_index, mean, *, settings, current_size, n_classes):
        write_premask_mean(
            mean, output_dir=self.directory, half_index=half_index, iteration=iteration, current_size=current_size,
            box_size=settings.box_size, voxel_size=settings.voxel_size, volume_shape=settings.volume_shape,
            n_classes=n_classes,
        )


class TargetHalfObserver(RunObserver):
    """Scores one half only at a terminating coarse-significance or pass-2 norm dump boundary
    (``RELAX_SIGNIFICANCE_DUMP_TARGET_HALF`` or ``RELAX_PASS2_DUMP_TARGET_HALF``, with the dump directory, iteration
    and original-index targets they require), read from the environment at each numbered iteration."""

    def scored_halves(self, numbered_relion_iteration, *, n_classes, experiment_datasets):
        return _significance_dump_half_indices(
            numbered_iteration=numbered_relion_iteration, n_classes=n_classes, experiment_datasets=experiment_datasets,
        )


class ObserverGroup(RunObserver):
    """Hands every hook to each of ``observers`` in order; a collection request holds if any asks for it."""

    def __init__(self, observers):
        self.observers = tuple(observers)
        self.keeps_rotation_posteriors = any(o.keeps_rotation_posteriors for o in self.observers)
        self.collects_local_search_profiles = any(o.collects_local_search_profiles for o in self.observers)
        self.keeps_pose_rotation_eulers = any(o.keeps_pose_rotation_eulers for o in self.observers)

    def wants_unfiltered_maps(self, numbered_relion_iteration):
        return any([o.wants_unfiltered_maps(numbered_relion_iteration) for o in self.observers])

    def scored_halves(self, numbered_relion_iteration, **values):
        # Every observer is asked (one may refuse its settings); the first that targets one half decides.
        selected = [o.scored_halves(numbered_relion_iteration, **values) for o in self.observers]
        return next((halves for halves in selected if halves != (0, 1)), (0, 1))

    def run_started(self, context):
        for o in self.observers:
            o.run_started(context)

    def iteration_started(self, iteration):
        for o in self.observers:
            o.iteration_started(iteration)

    def stage_finished(self, iteration, stage):
        for o in self.observers:
            o.stage_finished(iteration, stage)

    def iteration_finished(self, finished):
        for o in self.observers:
            o.iteration_finished(finished)

    def maps_reconstructed(self, maps):
        for o in self.observers:
            o.maps_reconstructed(maps)

    def half_accumulators_ready(self, iteration, **values):
        for o in self.observers:
            o.half_accumulators_ready(iteration, **values)

    def k1_accumulators_joined(self, iteration, **values):
        for o in self.observers:
            o.k1_accumulators_joined(iteration, **values)

    def noise_updated(self, iteration, **values):
        for o in self.observers:
            o.noise_updated(iteration, **values)

    def class_image_size_planned(self, iteration, plan, **values):
        for o in self.observers:
            o.class_image_size_planned(iteration, plan, **values)

    def class_prior_estimated(self, estimated):
        for o in self.observers:
            o.class_prior_estimated(estimated)

    def map_solved(self, iteration, half_index, mean, **values):
        for o in self.observers:
            o.map_solved(iteration, half_index, mean, **values)

    def poses_updated(self, iteration, **values):
        for o in self.observers:
            o.poses_updated(iteration, **values)

    def dense_half_scored(self, scored):
        for o in self.observers:
            o.dense_half_scored(scored)

    def numbered_half_recorded(self, recorded):
        for o in self.observers:
            o.numbered_half_recorded(recorded)

    def bpref_diagnostics_armed(self, iteration, half_index, **values):
        return any(o.bpref_diagnostics_armed(iteration, half_index, **values) for o in self.observers)

    def local_search_profile(self, iteration, half_index, profile):
        for o in self.observers:
            o.local_search_profile(iteration, half_index, profile)

    def final_half_scored(self, scored):
        for o in self.observers:
            o.final_half_scored(scored)


def observers_from_environment(*, timing_dir: Path | None) -> list[RunObserver]:
    """The observers the environment asks for: the BPref device-panel capture, the parity capture or its
    timings, the BPref accumulator captures, the noise-update terms, the Class3D M-step and image-size dumps, the pre-mask maps, a one-half dump
    target. ``timing_dir`` (the command's ``--timing_dir``; None: ``RELAX_PARITY_TIMING_DIR``, if set) receives
    the stage timings. A malformed ``RELAX_BPREF_BOUNDARY_DUMP_ITERATION`` refuses here."""

    found = []
    # Before the parity capture, as the device-panel flush came before the E-step capture of each half.
    if any(os.environ.get(name, "").strip() for name in bpref_diagnostics.BPREF_DIAGNOSTIC_ENVS):
        found.append(BpRefDiagnosticsObserver())
    timings = timing_dir if timing_dir is not None else parity_dump.timing_dir()
    if parity_dump.is_active() or timings is not None:
        found.append(ParityDumpObserver(timings))
    accumulators = BpRefAccumulatorObserver.from_environment()
    if accumulators.prejoin_dir or accumulators.accum_dir:
        found.append(accumulators)
    if os.environ.get("RELAX_NOISE_DEBUG_DUMP_DIR"):
        found.append(NoiseUpdateObserver())
    if os.environ.get("RELAX_KCLASS_DUMP_DIR"):
        found.append(ClassDumpObserver(os.environ["RELAX_KCLASS_DUMP_DIR"]))
    if os.environ.get("RELAX_PREMASK_DUMP_DIR"):
        found.append(PremaskObserver(os.environ["RELAX_PREMASK_DUMP_DIR"]))
    if any(os.environ.get(name, "").strip() for name in ("RELAX_SIGNIFICANCE_DUMP_TARGET_HALF", "RELAX_PASS2_DUMP_TARGET_HALF")):
        found.append(TargetHalfObserver())
    return found


def combine(observers: list[RunObserver]) -> RunObserver:
    """One observer for ``observers``: the do-nothing ``RunObserver()`` for none, the observer itself for one,
    else a group."""

    if not observers:
        return RunObserver()
    return observers[0] if len(observers) == 1 else ObserverGroup(observers)


def command_observer(args, *, timing_dir: Path | None) -> RunObserver:
    """The observer a refinement command line (and the environment it runs in) asks for (``RunObserver()``:
    none); ``timing_dir`` is the command's ``--timing_dir`` (None when not given)."""

    observers = []
    if args.save_intermediates_dir is not None:
        observers.append(IntermediatesObserver(
            args.save_intermediates_dir, skip_unregularized=bool(args.save_intermediates_skip_unregularized),
        ))
    return combine(observers + observers_from_environment(timing_dir=timing_dir))


def command_k_class_probe() -> KClassDiagnostics:
    """The K-class E-step's diagnostics a refinement command hands the run (``refine_single_volume``'s
    ``k_class_probe``): the environment's pass-1 dumps, support audit, score-dump labels and winner summary, each
    read when its hook is called."""

    return KClassDiagnostics()
