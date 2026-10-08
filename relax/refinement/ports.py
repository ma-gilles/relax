"""The ports through which comparison and observation enter the refinement (code rule 15).

An :class:`InputSource` supplies inputs a comparison run takes from elsewhere (a RELION run) instead of the
values the run computed: at a call site of the controller it receives the run's values and returns the
values the iteration uses. This base class is the native source: it returns what it is given. The command
chooses the source once (``relax.parity``); the algorithm never imports an implementation.

A :class:`RunObserver` watches a run: dumps, captures and profiles of what the run computed. It never changes
the run: no hook returns a value the algorithm uses, and the run's numbers are the same with any observer.
The command chooses the observer once (``relax.diagnostics.observers``) and hands it to
``refine_single_volume``; the algorithm's modules never import an implementation, only this interface.

The base ``RunObserver`` is the default and does nothing. Each hook is named for the moment of the run it is called
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


class ClassPriorEstimated(NamedTuple):
    """One class's prior of a Class3D M-step (``prior``, its ``ClassPrior``), with the accumulators and references it
    came from, the source label of its shells (``ClassTau2.source``) and the frame scale it was computed in."""

    iteration: int
    class_index: int
    prior: Any
    numerators: Any
    denominators: Any
    half_denominators: Any
    references: Any
    settings: Any
    current_size: int
    source: str
    accumulator_shape: Any
    full_half_axis: Any
    frame_scale: float


class FinishedIteration(NamedTuple):
    """A numbered iteration at its end, after the convergence update and the run files: its state, maps,
    poses, corrections and statistics. An observer that times the iteration starts its clock at
    ``iteration_started``."""

    iteration: int
    init_relion_iteration: int
    state: Any
    current_size: int
    sigma_offset_angstrom: Any
    random_perturbation: Any
    settings: Any
    pixel_size_angstrom: Any
    ave_pmax: Any
    fsc: Any
    noise_variance: Any
    means: Any
    unfiltered_means: Any
    poses: Any
    half_inputs: Any
    corrections: Any
    scale_correction_data_vs_prior: Any


class NumberedState(NamedTuple):
    """The inputs of a numbered iteration a source may replace, as the run holds them before its expectation.

    ``sigma_offset`` is the run's ``SigmaOffset``; ``mean_variance`` the reference model's tau2;
    ``class_mixture`` its ``ClassMixture``; ``prior_translations`` the translation grid a replayed sampling
    centres the local translation prior on, and ``sampling_healpix_order`` the HEALPix order of the sampling a
    RELION STAR file or sealed sampling state set this iteration: its angular step scales the perturbation,
    which then always applies (None natively: the run's own grid and ``parity.perturb_factor`` decide).
    """

    current_size: int
    noise_model: Any
    sigma_offset: Any
    previous_best_rotations: list
    mean_variance: Any
    class_mixture: Any
    prior_translations: Any
    sampling_healpix_order: int | None


class ScoringState(NamedTuple):
    """A numbered iteration's scoring state as the run holds it at a source's call: the ``RefinementState``, the
    current size, the reference and noise models, the halves (poses and corrections), the previous best
    rotations, the ``SigmaOffset`` and the direction priors. A source that keeps any of it past the call copies
    it."""

    state: Any
    current_size: int
    reference_model: Any
    noise_model: Any
    halves: Any
    previous_best_rotations: Any
    sigma_offset: Any
    direction_priors: Any


class ScoringArrays(NamedTuple):
    """The arrays the run scores with, as it holds them at a source's call: the reference and noise models, the
    ``SigmaOffset``, the halves, the direction priors and the experiment datasets."""

    reference_model: Any
    noise_model: Any
    sigma_offset: Any
    halves: Any
    direction_priors: Any
    experiment_datasets: Any


class ClassTau2(NamedTuple):
    """A Class3D iteration's prior shells taken from elsewhere (``shells``, ``(K, n_shells)``; None: the M-step
    computes its own from the previous references) and the label its log and dumps give their source."""

    shells: Any
    source: str


class FinalState(NamedTuple):
    """The inputs of the final all-data pass a source may replace: the references each half is scored
    against (``join_means``), the ``SigmaOffset`` and the noise model."""

    join_means: list
    sigma_offset: Any
    noise_model: Any


class InputSource:
    """The native source: every input is the one the run computed."""

    # What a replaying source replays (``relax.parity.relion_replay_source.RelionReplay``), for the admission
    # checks that refuse combinations with it; None for the native source.
    relion_replay = None
    # A frozen RELION boundary's sealed sampling state (captured directions, psi angles, translations and
    # sizes) and the scoring context sealed with it; None natively. The sampling plan reads them.
    sealed_sampling_state = None
    sealed_scoring_context = None
    # Whether a state-swap probe swaps state in some iteration (it then admits a fresh BPref particle order).
    swaps_state = False
    # Whether the run follows a RELION run's numbered STAR files or override slots, and whether it starts from a
    # frozen boundary (a sealed sampling state or its RefinementState fields); a continuation refuses both.
    replays_relion_trajectory = False
    starts_from_frozen_boundary = False
    # A replayed MPI RELION run's follower topology and captured dispatch schedule (a
    # ``relax.relion.relion_worker_scale.PreparedFollowerTopology``); None natively: no followers.
    follower_topology = None

    def replays_relion_state(self) -> bool:
        """Whether the run's state comes from a RELION run (replayed slots, STAR files, final-pass state, a
        frozen boundary or a state-swap probe), which options computed with relax's own rules cannot be
        combined with."""
        return False

    def restore_boundary_state(self, state) -> None:
        """After the initial resolution is set: install a frozen boundary's ``RefinementState`` fields in
        ``state``."""

    def initial_coarse_grids(self, *, initialized_healpix_order: int, voxel_size, symmetry: str, native):
        """The first iteration's coarse grids (a ``CoarseGrids``); ``native()`` builds the run's own."""
        return native()

    def coarse_grids(self, iteration: int, grids, state, *, voxel_size, dtype):
        """Iteration ``iteration``'s coarse grids (a ``CoarseGrids``) before they are refreshed for ``state``'s
        sampling (the native source: ``grids``)."""
        return grids

    def adaptive_coarse_size(self, plan, *, model_size: int):
        """The adaptive pass-1 width (a ``CoarseImageSize``; the native source: the run's ``plan``), at most the
        model's ``model_size``."""
        return plan

    def numbered_state(
        self, iteration: int, inputs: NumberedState, *, state, halves, direction_priors, image_geometry,
    ) -> NumberedState:
        """The state numbered iteration ``iteration`` scores with.

        A replaying source may also update ``state`` (its sampling controls), the ``halves`` (poses and
        corrections) and the ``direction_priors`` list in place, and says so.
        """
        return inputs

    def state_swap_snapshot(self, iteration: int, scoring: ScoringState) -> None:
        """Before iteration ``iteration``'s numbered state is installed, with the run's own ``scoring`` state."""

    def scoring_references(self, iteration: int, reference_model, *, volume_shape):
        """The maps iteration ``iteration`` scores against (the native source: the model's own)."""
        return reference_model.maps

    def swapped_state(self, iteration: int, scoring: ScoringState, *, volume_shape):
        """The ``scoring`` state with a probe's components swapped back to the run's own snapshot (a tuple of
        size, reference model, noise model, previous best rotations, the two sigma offsets and direction
        priors), or None when iteration ``iteration`` swaps nothing."""
        return None

    def scoring_state_bound(self, arrays: ScoringArrays) -> None:
        """Before the first iteration, with the ``arrays`` the run will score with."""

    def scoring_state_checked(self, iteration: int, arrays: ScoringArrays):
        """Right before iteration ``iteration`` scores: a check of its ``arrays`` against those bound before the
        first iteration; its digest, or None when nothing is checked."""
        return None

    def scoring_rotation_ids(self, trial_grid, *, use_local: bool):
        """The rotation ids the scorer reads for ``trial_grid`` (None: the grid's own)."""
        return None

    def relion_run_directory(self, iteration: int):
        """The RELION run directory whose numbered STAR files supply iteration ``iteration``'s sampling (None:
        the run samples natively). ``iteration`` -1 asks before the first iteration."""
        return None

    def random_perturbation(self, iteration: int, native) -> float:
        """This iteration's sampling perturbation, after its ``numbered_state``; ``native()`` computes the run's own
        (and advances its RNG)."""
        return native()

    def final_state(
        self, inputs: FinalState, *, means, numbered_iteration_count: int, halves, direction_priors, healpix_order: int,
        image_geometry,
    ) -> FinalState:
        """The state the final all-data pass scores with. ``means`` are the run's half maps; a replaying source
        may also update the ``halves`` and the ``direction_priors`` list in place, and says so."""
        return inputs

    def final_sampling_settings(self, state, image_geometry, *, grid_order: int, relion_iteration: int, native):
        """The final all-data pass's sampling settings (a ``FinalSamplingSettings``); ``native()`` resolves the
        run's own (and advances its RNG)."""
        return native()

    def class_tau2(self, iteration: int, n_classes: int) -> ClassTau2:
        """The Class3D M-step's prior shells of this iteration (the native source: its own)."""
        return ClassTau2(None, "previous Iref power spectra")

    def restore_convergence_state(self, state) -> bool:
        """Restore ``state``'s convergence counters from a replayed run's restart iteration, in place; return
        whether it did (then the run's own initial resolution is not set)."""
        return False

    def convergence_accuracy(self, iteration: int, accuracy):
        """The accuracies the iteration's convergence update reads (an ``OptimiserAccuracyReplay``; the run's
        own carry no optimiser record)."""
        return accuracy

    def apply_optimiser_controls(self, iteration: int, state, accuracy) -> None:
        """After the iteration's convergence update: install a replayed optimiser's counters in ``state``."""


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

    def iteration_started(self, iteration: int) -> None:
        """A numbered iteration starts (after the convergence check that could have ended the loop)."""

    def stage_finished(self, iteration: int, stage: str) -> None:
        """A stage of a numbered iteration is done: ``e_step``, ``recon``, ``fsc``, ``noise_update`` (not in a
        first-iteration CC) or ``convergence``, in that order."""

    def iteration_finished(self, finished: FinishedIteration) -> None:
        """A numbered iteration is complete, before its timing log and end-of-iteration releases."""

    def half_accumulators_ready(
        self, iteration: int, *, numerators, denominators, settings, current_size: int, accumulator_shape,
        k_class_enabled: bool, pixel_size_angstrom,
    ) -> None:
        """A numbered iteration's two half accumulators are complete, before any check or cross-half join."""

    def k1_accumulators_joined(
        self, iteration: int, *, numerators, denominators, settings, current_size: int, accumulator_shape,
        pixel_size_angstrom,
    ) -> None:
        """A K=1 iteration's half accumulators after the low-resolution join, before its prior reads them."""

    def noise_updated(
        self, iteration: int, *, current_size: int, image_shape, noise_stats_per_half, previous_noise_radial_per_half,
        noise_from_res_per_half, noise_from_res,
    ) -> None:
        """A numbered iteration's noise spectra are estimated (the new ones beside the previous model's)."""

    def class_image_size_planned(
        self, iteration: int, plan, *, previous_size: int, box_size: int, has_high_fsc_at_limit: bool,
        incr_size: int, state,
    ) -> None:
        """A Class3D iteration's image size is planned (``plan``, its ``ClassImageSize``)."""

    def class_prior_estimated(self, estimated: ClassPriorEstimated) -> None:
        """One class's prior of a Class3D M-step is estimated."""

    def map_solved(self, iteration: int, half_index: int, mean, *, settings, current_size: int, n_classes: int) -> None:
        """A half's numbered map (or class stack) is solved, before its low-pass and solvent mask."""

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
