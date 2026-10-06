"""The run observers the command chooses (code rule 15; the port is ``relax.refinement.ports.RunObserver``).

``IntermediatesObserver`` writes ``--save_intermediates_dir``: every numbered iteration's maps, accumulators,
FSC, noise, prior, assignments and grids, each half's particle state and scoring manifest, the local-search
profiles and the final pass's manifests. ``command_observer`` builds the observer of a ``relax refine`` /
``relax class3d`` command line. The algorithm's modules never import this module.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from relax.diagnostics.iteration import (
    _save_iteration_intermediates,
    _save_iteration_particle_states,
    _source_image_indices,
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


def command_observer(args) -> RunObserver | None:
    """The observer a refinement command line asks for, or None (nothing observes the run)."""

    if args.save_intermediates_dir is None:
        return None
    return IntermediatesObserver(
        args.save_intermediates_dir, skip_unregularized=bool(args.save_intermediates_skip_unregularized),
    )
