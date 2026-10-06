"""The run observers the command chooses (code rule 15; the port is ``relax.refinement.ports.RunObserver``).

``IntermediatesObserver`` writes ``--save_intermediates_dir``: every numbered iteration's maps, accumulators,
FSC, noise, prior, assignments and grids, each half's particle state and scoring manifest, the local-search
profiles and the final pass's manifests. ``ParityDumpObserver`` writes the per-iteration parity capture
(``RELAX_PARITY_DUMP_DIR``) or its stage timings alone (``RELAX_PARITY_TIMING_DIR``, ``--timing_dir``);
the per-half E-step capture it completes is still collected by ``relax.refinement.expectation``.
``ObserverGroup`` hands every hook to several observers. ``command_observer`` builds the observer of a
``relax refine`` / ``relax class3d`` command line. The algorithm's modules never import this module.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from relax.diagnostics import parity_dump
from relax.diagnostics.iteration import (
    _save_iteration_intermediates,
    _save_iteration_particle_states,
    _source_image_indices,
    dump_numbered_iteration,
    write_final_half_manifest,
    write_numbered_half_manifest,
)
from relax.refinement.ports import RunObserver

logger = logging.getLogger(__name__)


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
        _save_iteration_intermediates(
            self.directory, maps.numerators, maps.denominators, maps.reference_model, maps.noise_model,
            maps.per_half, maps.trial_grid, maps.sampling_plan, maps.options, iteration=maps.iteration,
            unreg_means=maps.unfiltered_maps, fsc=maps.fsc, cs=maps.current_size, state=maps.state,
            volume_shape=maps.volume_shape, voxel_size=maps.voxel_size,
        )

    def poses_updated(self, iteration, *, poses, per_half, significance, datasets):
        _save_iteration_particle_states(
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
    the unregularized maps; without it, the timing directory (``RELAX_PARITY_TIMING_DIR``) receives each
    iteration's stage times. Both are read from the environment when each hook runs, as the module does.
    """

    def wants_unfiltered_maps(self, numbered_relion_iteration: int) -> bool:
        return parity_dump.is_active()

    def iteration_started(self, iteration):
        parity_dump.start_iteration(iteration)

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
        elif parity_dump.timing_is_active():
            try:
                parity_dump.dump_timing_iteration(
                    iteration=finished.iteration,
                    init_relion_iteration=int(finished.init_relion_iteration),
                    iteration_start=finished.iteration_start,
                )
            except Exception as exc:
                logger.warning("parity_dump.dump_timing_iteration failed at iter %d: %s", finished.iteration, exc)


class ObserverGroup(RunObserver):
    """Hands every hook to each of ``observers`` in order; a collection request holds if any asks for it."""

    def __init__(self, observers):
        self.observers = tuple(observers)
        self.keeps_rotation_posteriors = any(o.keeps_rotation_posteriors for o in self.observers)
        self.collects_local_search_profiles = any(o.collects_local_search_profiles for o in self.observers)

    def wants_unfiltered_maps(self, numbered_relion_iteration):
        return any([o.wants_unfiltered_maps(numbered_relion_iteration) for o in self.observers])

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

    def poses_updated(self, iteration, **values):
        for o in self.observers:
            o.poses_updated(iteration, **values)

    def dense_half_scored(self, scored):
        for o in self.observers:
            o.dense_half_scored(scored)

    def local_search_profile(self, iteration, half_index, profile):
        for o in self.observers:
            o.local_search_profile(iteration, half_index, profile)

    def final_half_scored(self, scored):
        for o in self.observers:
            o.final_half_scored(scored)


def observers_from_environment() -> list[RunObserver]:
    """The observers the environment asks for: the parity capture or its timings."""

    return [ParityDumpObserver()] if parity_dump.timing_is_active() else []


def combine(observers) -> RunObserver | None:
    """One observer for ``observers``: None for none, the observer itself for one, else a group."""

    observers = list(observers)
    if not observers:
        return None
    return observers[0] if len(observers) == 1 else ObserverGroup(observers)


def command_observer(args) -> RunObserver | None:
    """The observer a refinement command line (and the environment it runs in) asks for, or None."""

    observers = []
    if args.save_intermediates_dir is not None:
        observers.append(IntermediatesObserver(
            args.save_intermediates_dir, skip_unregularized=bool(args.save_intermediates_skip_unregularized),
        ))
    return combine(observers + observers_from_environment())
