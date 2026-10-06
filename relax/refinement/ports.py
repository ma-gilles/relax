"""The ports through which observation enters the refinement (code rule 15).

A :class:`RunObserver` watches a run: dumps, captures and profiles of what the run computed. It never changes
the run: no hook returns a value the algorithm uses, and the run's numbers are the same with any observer.
The command chooses the observer once (``relax.diagnostics.observers``) and hands it to
``refine_single_volume``; the algorithm's modules never import an implementation, only this interface.

This base class is the default and does nothing. Each hook is named for the moment of the run it is called
at and receives the values the run holds then; an observer must not mutate them, and must not keep a device
array past the hook unless it says so (it would extend that array's lifetime, code rule 3). The two
collection requests (``keeps_rotation_posteriors``, ``collects_local_search_profiles``) and
``wants_unfiltered_maps`` ask the run to keep or compute something beside its own values; they cost memory
or time, never a number of the run.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class ReconstructedIteration(NamedTuple):
    """A numbered iteration once its maps and FSC are made, before its poses are updated.

    ``numerators`` and ``denominators`` are the two halves' M-step accumulators as the reconstruction used
    them; ``unfiltered_maps`` holds None per half unless the observer (``wants_unfiltered_maps``) or the run
    files asked for them; ``fsc`` is None for Class3D.
    """

    iteration: int
    numerators: Any
    denominators: Any
    reference_model: Any
    noise_model: Any
    per_half: Any
    trial_grid: Any
    sampling_plan: Any
    options: Any
    unfiltered_maps: Any
    fsc: Any
    current_size: int
    state: Any
    volume_shape: Any
    voxel_size: float


class DenseHalfScored(NamedTuple):
    """One half of a numbered iteration scored on the single-pass dense route (no adaptive pass 2).

    ``half`` is the half's ``HalfScoringData`` (reference, prior, noise), ``particles`` its ``HalfSet``,
    ``grid`` the iteration's trial grid, ``sampling`` its ``DenseSamplingSpec`` and ``result`` the engine's
    result for the half.
    """

    iteration: int
    half_index: int
    grid: Any
    sampling: Any
    direction_priors: Any
    translation_log_prior: Any
    particles: Any
    translation_search_base: Any
    previous_translations: Any
    half: Any
    image_window_size: Any
    perturb_factor: float
    result: Any


class FinalHalfScored(NamedTuple):
    """One half of the final all-data pass scored. ``final_inputs`` (its ``PreparedFinalHalf``) is None for
    subtomograms."""

    half: Any
    final_sampling: Any
    final_inputs: Any
    translation_search_base: Any
    reference: Any
    reference_model: Any
    noise_variance: Any
    current_size: int
    precision: Any
    use_local: bool


class RunObserver:
    """Watches a refinement run and never steers it; every hook does nothing by default."""

    # The run keeps each numbered iteration's rotation posteriors in its history (the archive's
    # ``rotation_posterior_*`` keys).
    keeps_rotation_posteriors: bool = False
    # A local search collects its per-bucket profile where ``--local_search_profile`` is ``auto``.
    collects_local_search_profiles: bool = False

    def wants_unfiltered_maps(self, numbered_relion_iteration: int) -> bool:
        """Whether this numbered iteration should also reconstruct its unregularized maps for the observer."""
        return False

    def maps_reconstructed(self, maps: ReconstructedIteration) -> None:
        """A numbered iteration's maps and FSC are made."""

    def poses_updated(self, iteration: int, *, poses, per_half, significance, datasets) -> None:
        """A numbered iteration's best poses (each half's ``ParticlePoses``) are installed in the halves."""

    def dense_half_scored(self, scored: DenseHalfScored) -> None:
        """One half of a numbered iteration was scored on the single-pass dense route."""

    def local_search_profile(self, iteration: int, half_index: int, profile: dict) -> None:
        """A local search of one half returned its profile (only when the run collects profiles)."""

    def final_half_scored(self, scored: FinalHalfScored) -> None:
        """One half of the final all-data pass was scored."""
