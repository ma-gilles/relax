"""Native InitialModel sampling state, plan and accuracy updates.

RELION's ``updateAngularSampling`` counterpart for the native driver: the
sampling state and plan records, the per-iteration random perturbation, the
expected-accuracy estimate and the assignment-change trackers that decide when
the angular sampling is refined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from recovar.utils.helpers import R_to_relion, recovar_volume_to_relion

from relax import sampling
from relax.refinement.optics_shapes import MultiShapeDataset
from relax.refinement.refinement_state import (
    compute_relion_offset_changes_angstrom,
    compute_relion_orientation_changes,
    relion_mpi_hidden_variable_change_is_small,
)
from relax.relion import optics_scale
from relax.relion.optics_aberrations import expected_accuracy_optics
from relax.sampling.expected_accuracy import (
    ExpectedAccuracy,
    _combine_group_expected_accuracies,
    estimate_relion_expected_accuracy_from_prepared_inputs,
    estimate_relion_expected_accuracy_in_spawned_process_from_prepared_inputs,
)
from relax.vdam.native_options import NativeInitialModelOptions
from relax.vdam.schedules import relion_sampling_cadence
from relax.vdam.state import InitialModelState, NativeOpticsState, NativeParticleState

if TYPE_CHECKING:
    from relax.vdam.ports import ExpectationProbe

if TYPE_CHECKING:
    from relax.refinement.tomo_half import TiltImageAccuracyInputs

RELION_INITIALMODEL_LOCAL_SEARCH_HEALPIX_ORDER = 4


RELION_ORIENTATIONAL_PRIOR_NOPRIOR = 0


RELION_ORIENTATIONAL_PRIOR_ROTTILT_PSI = 1


RELION_INITIALMODEL_MIN_TRANSLATION_STEP_ANGSTROM = 1.5


RELION_INITIALMODEL_MAX_NR_ITER_WO_RESOL_GAIN = 1


RELION_INITIALMODEL_SMALL_CHANGE_INIT_OFFSETS = 999.0


RELION_INITIALMODEL_SMALL_CHANGE_INIT_ORIENTATIONS = 999.0


RELION_INITIALMODEL_SMALL_CHANGE_INIT_CLASSES = 9999999.0




@dataclass(frozen=True)
class NativeSamplingPlan:
    """Trial geometry of one iteration; the adaptive E-step builds its own fine grid (``rotations`` None)."""

    rotations: np.ndarray | None
    translations: np.ndarray
    random_perturbation: float
    healpix_order: int
    oversampling: int
    offset_range_px: float
    offset_step_px: float
    offset_range_angstrom: float
    offset_step_angstrom: float
    coarse_translations: np.ndarray | None = None
    coarse_prior_translations: np.ndarray | None = None
    metadata_translations: np.ndarray | None = None
    translation_parent: np.ndarray | None = None
    # RELION's unperturbed host-double coarse grid, before the float32 cast.
    coarse_base_translations: np.ndarray | None = None

    @property
    def n_rotations(self) -> int:
        if self.rotations is not None:
            return int(self.rotations.shape[0])
        return int(sampling.rotation_grid_size(self.healpix_order)) * 8 ** int(self.oversampling)


@dataclass
class NativeSamplingState:
    """RELION InitialModel autosampling state (Angstroms internally, like RELION)."""

    healpix_order: int
    adaptive_oversampling: int
    offset_range_angstrom: float
    offset_step_angstrom: float
    offset_range_ori_angstrom: float
    offset_step_ori_angstrom: float
    pixel_size: float
    max_healpix_order: int | None = None
    auto_local_healpix_order: int = RELION_INITIALMODEL_LOCAL_SEARCH_HEALPIX_ORDER
    # 0 until estimate_native_sampling_accuracy (calculateExpectedAngularErrors) sets it; the HEALPix
    # order does not refine before then.
    acc_rot: float = 0.0
    acc_trans_angstrom: float = 999.0
    current_changes_optimal_offsets_angstrom: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_OFFSETS
    current_changes_optimal_orientations: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_ORIENTATIONS
    current_changes_optimal_classes: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_CLASSES
    smallest_changes_optimal_offsets_angstrom: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_OFFSETS
    smallest_changes_optimal_orientations: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_ORIENTATIONS
    smallest_changes_optimal_classes: float = RELION_INITIALMODEL_SMALL_CHANGE_INIT_CLASSES
    nr_iter_wo_resol_gain: int = 0
    nr_iter_wo_large_hidden_variable_changes: int = 0
    has_fine_enough_angular_sampling: bool = False
    last_current_resolution: float = 0.0
    orientational_prior_mode: int = RELION_ORIENTATIONAL_PRIOR_NOPRIOR
    uniform_local_orientation_prior: bool = False
    # Subtomogram particles (2D stacks): updateAngularSampling keeps at least half the step.
    subtomogram: bool = False

    @property
    def offset_range_px(self) -> float:
        return float(self.offset_range_angstrom) / float(self.pixel_size)

    @property
    def offset_step_px(self) -> float:
        return float(self.offset_step_angstrom) / float(self.pixel_size)

    @property
    def effective_offset_step_angstrom(self) -> float:
        return float(self.offset_step_angstrom) / (2 ** int(self.adaptive_oversampling))


def n_directions_for_healpix_order(healpix_order: int) -> int:
    return int(
        sampling.rotation_grid_size(int(healpix_order)) // sampling.rotation_grid_n_in_planes(int(healpix_order))
    )


def initial_sampling_state(
    opts: NativeInitialModelOptions, *, pixel_size: float, subtomogram: bool = False
) -> NativeSamplingState:
    if pixel_size <= 0.0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")
    return NativeSamplingState(
        healpix_order=int(opts.healpix_order),
        adaptive_oversampling=int(opts.oversampling),
        offset_range_angstrom=float(opts.offset_range_px) * pixel_size,
        offset_step_angstrom=float(opts.offset_step_px) * pixel_size,
        offset_range_ori_angstrom=float(opts.offset_range_px) * pixel_size,
        offset_step_ori_angstrom=float(opts.offset_step_px) * pixel_size,
        pixel_size=pixel_size,
        max_healpix_order=None if opts.pilot_controls is None else opts.pilot_controls.max_healpix_order,
        subtomogram=bool(subtomogram),
    )


def _should_update_native_sampling(*, iteration: int, do_grad: bool) -> bool:
    """The InitialModel ``updateAngularSampling`` call cadence: RELION's sampling cadence, never in iteration 1."""
    return int(iteration) > 1 and relion_sampling_cadence(iteration=iteration, do_grad=do_grad)


def record_native_sampling_post_iteration(
    sampling_state: NativeSamplingState,
    state: InitialModelState,
    *,
    meta: dict,
) -> None:
    """Record sampling-controller state after the completed M-step.

    RELION calls ``updateAngularSampling`` near the start of expectation, then
    calls ``updateCurrentResolution`` after maximization.  Keep that ordering:
    the sampling decision for iteration ``N`` must only see the stall counter
    written by iteration ``N - 1``.
    """
    if state.current_resolution <= float(sampling_state.last_current_resolution) + 0.0001:
        sampling_state.nr_iter_wo_resol_gain += 1
    else:
        sampling_state.nr_iter_wo_resol_gain = 0
    sampling_state.last_current_resolution = state.current_resolution
    meta["sampling_nr_iter_wo_resol_gain"] = int(sampling_state.nr_iter_wo_resol_gain)
    meta["sampling_nr_iter_wo_large_hidden_variable_changes"] = int(
        sampling_state.nr_iter_wo_large_hidden_variable_changes
    )
    meta["sampling_last_current_resolution"] = float(sampling_state.last_current_resolution)


def _reset_native_sampling_change_trackers(sampling_state: NativeSamplingState) -> None:
    sampling_state.nr_iter_wo_resol_gain = 0
    sampling_state.nr_iter_wo_large_hidden_variable_changes = 0
    sampling_state.smallest_changes_optimal_offsets_angstrom = RELION_INITIALMODEL_SMALL_CHANGE_INIT_OFFSETS
    sampling_state.smallest_changes_optimal_orientations = RELION_INITIALMODEL_SMALL_CHANGE_INIT_ORIENTATIONS
    sampling_state.smallest_changes_optimal_classes = RELION_INITIALMODEL_SMALL_CHANGE_INIT_CLASSES


def _relion_update_native_sampling_state(
    sampling_state: NativeSamplingState,
    *,
    do_grad: bool,
) -> bool:
    """RELION InitialModel autosampling update (``--auto_sampling --grad`` mode).

    HEALPix growth stops before the local-search order; translation range/step
    follow the RELION formulas.
    """
    old_angular_step = sampling.relion_angular_sampling_deg(
        sampling_state.healpix_order,
        sampling_state.adaptive_oversampling,
    )
    if old_angular_step < 0.75 * float(sampling_state.acc_rot):
        sampling_state.has_fine_enough_angular_sampling = True
        return False
    sampling_state.has_fine_enough_angular_sampling = False

    oversampling_factor = 2 ** int(sampling_state.adaptive_oversampling)
    new_step = (
        min(
            RELION_INITIALMODEL_MIN_TRANSLATION_STEP_ANGSTROM,
            0.75 * float(sampling_state.acc_trans_angstrom),
        )
        * oversampling_factor
    )
    if sampling_state.subtomogram:
        # "For subtomogram averaging: use at least half times previous step size" (ml_optimiser.cpp:9832-9834).
        new_step = max(float(sampling_state.offset_step_angstrom) / 2.0, new_step)
    new_range = 5.0 * float(sampling_state.current_changes_optimal_offsets_angstrom)
    new_range = min(1.3 * float(sampling_state.offset_range_angstrom), new_range)
    new_range = max(new_range, 1.5 * new_step)
    if new_range > 4.0 * new_step:
        new_range /= 2.0
    if new_range > 4.0 * new_step:
        new_step = new_range / 4.0

    new_healpix_order = int(sampling_state.healpix_order)
    requested_healpix_order = new_healpix_order + 1
    gradient_ceiling_reached = (
        bool(do_grad)
        and requested_healpix_order >= int(sampling_state.auto_local_healpix_order)
    )
    if not gradient_ceiling_reached and (
        sampling_state.max_healpix_order is None
        or requested_healpix_order <= sampling_state.max_healpix_order
    ):
        new_healpix_order = requested_healpix_order

    if new_step > float(sampling_state.offset_step_angstrom):
        new_step = float(sampling_state.offset_step_angstrom)
        new_range = float(sampling_state.offset_range_angstrom)

    old_orientational_prior_mode = int(sampling_state.orientational_prior_mode)
    old_uniform_local_orientation_prior = bool(sampling_state.uniform_local_orientation_prior)
    if requested_healpix_order >= int(sampling_state.auto_local_healpix_order):
        sampling_state.orientational_prior_mode = RELION_ORIENTATIONAL_PRIOR_ROTTILT_PSI
        # RELION's gradient InitialModel stops at the exhaustive HEALPix-3
        # grid, but its pinned updateAngularSampling path still switches to
        # PRIOR_ROTTILT_PSI. The stored zero angular-prior widths then produce
        # uniform direction and psi priors, as observed at the live GPU score
        # boundary. Keep this transition explicit instead of carrying the
        # learned pdf_direction into iteration 90 and later.
        sampling_state.uniform_local_orientation_prior = bool(gradient_ceiling_reached)

    changed = (
        new_healpix_order != int(sampling_state.healpix_order)
        or abs(new_step - float(sampling_state.offset_step_angstrom)) > 1e-12
        or abs(new_range - float(sampling_state.offset_range_angstrom)) > 1e-12
        or int(sampling_state.orientational_prior_mode) != old_orientational_prior_mode
        or bool(sampling_state.uniform_local_orientation_prior) != old_uniform_local_orientation_prior
    )
    sampling_state.healpix_order = int(new_healpix_order)
    sampling_state.offset_step_angstrom = float(new_step)
    sampling_state.offset_range_angstrom = float(new_range)
    _reset_native_sampling_change_trackers(sampling_state)
    return changed


def prepare_native_sampling_for_iteration(
    sampling_state: NativeSamplingState,
    state: InitialModelState,
    *,
    iteration: int,
    do_grad: bool,
) -> bool:
    # MlOptimiser::iterate resets both convergence counters before expectation
    # during the initial gradient burn-in.  The completed M-step may populate
    # them again for the checkpoint written by this iteration.
    if bool(do_grad) and int(iteration) < 10:
        sampling_state.nr_iter_wo_resol_gain = 0
        sampling_state.nr_iter_wo_large_hidden_variable_changes = 0
    if not _should_update_native_sampling(iteration=iteration, do_grad=do_grad):
        return False
    if sampling_state.nr_iter_wo_resol_gain < RELION_INITIALMODEL_MAX_NR_ITER_WO_RESOL_GAIN:
        return False
    # RELION initialiseGeneral sets auto_ignore_angle_changes for the entire
    # gradient_refine run, including its final EM phase. InitialModel therefore
    # updates sampling on resolution stalls even when assignments still change.
    return _relion_update_native_sampling_state(sampling_state, do_grad=do_grad)


def _best_eulers_from_particle_state(
    particle_state: NativeParticleState,
    particle_ids: np.ndarray,
    *,
    rotation_grid_order: int,
) -> np.ndarray | None:
    """Each particle's Euler angles: the exact source triple where valid, else from its best rotation
    matrix, else from its best rotation id on the RELION grid; None when a particle has none of them."""
    ids = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
    n_particles = len(particle_state.translation_offsets)
    source = particle_state.best_pose_eulers_deg
    valid = particle_state.best_pose_eulers_valid
    if valid is not None:
        valid = np.asarray(valid)
        if valid.dtype != bool or valid.shape != (n_particles,):
            raise ValueError("source Euler validity must match the particle table")
    if source is not None:
        source = np.asarray(source)
        if (
            source.dtype != np.float64
            or source.shape != (n_particles, 3)
            or valid is None
            or not np.all(np.isfinite(source[valid]))
        ):
            raise ValueError("source Euler metadata must be finite float64 triples on valid particle rows")
    elif valid is not None and np.any(valid):
        raise ValueError("valid source Euler metadata requires an Euler array")
    resolved = np.zeros(ids.size, dtype=bool) if valid is None else valid[ids].copy()
    result = np.empty((ids.size, 3), dtype=np.float64)
    if source is not None:
        result[resolved] = source[ids[resolved]]
    rotations = particle_state.best_pose_rotations
    if rotations is not None and not np.all(resolved):
        rotations = np.asarray(rotations)
        if rotations.shape != (n_particles, 3, 3):
            raise ValueError("pose matrices must match the particle table")
        selected = rotations[ids]
        matrix_rows = ~resolved & np.any(np.abs(selected.reshape(ids.size, 9)) > 0, axis=1)
        if np.any(matrix_rows):
            result[matrix_rows] = np.asarray(R_to_relion(selected[matrix_rows].astype(np.float64), degrees=True))
            resolved[matrix_rows] = True
    if not np.all(resolved) and particle_state.best_pose_rotation_ids is not None:
        best_ids = np.asarray(particle_state.best_pose_rotation_ids, dtype=np.int64)[ids]
        eulers = sampling.get_relion_rotation_grid_eulers(int(rotation_grid_order), rotation_index_order="relion")
        grid_rows = ~resolved & (best_ids >= 0) & (best_ids < len(eulers))
        result[grid_rows] = eulers[best_ids[grid_rows]]
        resolved[grid_rows] = True
    return result if np.all(resolved) else None


@dataclass(frozen=True)
class SamplingAccuracyEstimate:
    """One ``calculateExpectedAngularErrors`` estimate over the subset's first trials: the accuracy, the trial
    count, the sigma2 fudge and the trials' RELION part ids (their random seeds). The driver installs
    ``accuracy.acc_rot`` and ``accuracy.acc_trans_angstrom`` on the sampling state; :meth:`meta` is the report."""

    accuracy: ExpectedAccuracy
    n_trials: int
    sigma2_fudge: float
    seed_part_ids: np.ndarray

    def meta(self) -> dict[str, object]:
        """The iteration meta's ``estimated_acc_*`` keys."""
        return {
            "estimated_acc_rot": self.accuracy.acc_rot,
            "estimated_acc_trans_angstrom": self.accuracy.acc_trans_angstrom,
            "estimated_acc_rot_class": self.accuracy.acc_rot_per_class,
            "estimated_acc_trans_class": self.accuracy.acc_trans_per_class_angstrom,
            "estimated_acc_class_counts": self.accuracy.class_counts,
            "estimated_acc_n_trials": self.n_trials,
            "estimated_acc_sigma2_fudge": self.sigma2_fudge,
            "estimated_acc_seed_part_ids": self.seed_part_ids,
        }


@dataclass(frozen=True)
class AccuracyEstimateInputs:
    """What one expected-accuracy estimate read, for the observer (``ExpectationProbe.expected_accuracy_estimated``):
    the state, particle and optics states (``optics_state`` None for subtomograms), the references in RELION's
    layout, every trial's Euler angles, the trials' particle ids, classes and RELION part ids (their random
    seeds), and the estimator's settings."""

    state: InitialModelState
    particle_state: NativeParticleState
    optics_state: NativeOpticsState | None
    references_relion: np.ndarray
    eulers: np.ndarray
    trial_particle_ids: np.ndarray
    class_ids: np.ndarray
    current_image_size: int
    padding_factor: int
    sigma2_fudge: float
    random_seed: int
    random_seed_particle_ids: np.ndarray


def estimate_native_sampling_accuracy(
    sampling_state: NativeSamplingState,
    state: InitialModelState,
    particle_state: NativeParticleState,
    optics_state: NativeOpticsState,
    *,
    particle_order: np.ndarray,
    random_seed: int,
    padding_factor: int,
    sigma2_fudge: float,
    tilt_images: TiltImageAccuracyInputs | None = None,
    optics_group_ids: np.ndarray | None = None,
    experiment_dataset=None,
    isolate_in_subprocess: bool = False,
    probe: ExpectationProbe,
) -> SamplingAccuracyEstimate | None:
    """RELION's expected accuracy of the subset's first 100 particles (calculateExpectedAngularErrors).

    Subtomogram particles pass ``tilt_images`` (:func:`relax.refinement.tomo_half.tilt_image_accuracy_inputs`
    of every particle, in particle-STAR row order) and no ``optics_state``: each trial sums its tilt
    images, each with its own ``Aproj`` and dose-damped CTF. With several optics groups
    (``optics_group_ids``, each particle's zero-based group) every trial takes its group's noise
    spectrum and CTF constants, as RELION does per particle; the estimate runs once per group on
    that group's trials and the per-class means are recombined with the trial counts.
    Single particles pass ``experiment_dataset`` (subtomograms: the flat dataset of ``tilt_images``' tilt
    images) for the trials' optics-table CTFs
    (:func:`relax.relion.optics_aberrations.expected_accuracy_optics`: CTF^2 of premultiplied
    images, even Zernike terms); None without optics-table terms.
    """
    n_trials = min(100, int(particle_order.size))
    if n_trials <= 0:
        return None
    trial_particle_ids = np.asarray(particle_order[:n_trials], dtype=np.int64)
    eulers = _best_eulers_from_particle_state(
        particle_state,
        trial_particle_ids,
        rotation_grid_order=int(sampling_state.healpix_order) + int(sampling_state.adaptive_oversampling),
    )
    if eulers is None:
        return None
    class_ids = np.asarray(particle_state.class_assignments, dtype=np.int32)[trial_particle_ids]
    if np.any(class_ids < 0) or np.any(class_ids >= int(state.K)):
        return None

    random_seed_particle_ids = np.arange(n_trials, dtype=np.int64)
    if state.sorted_particle_part_ids is not None:
        sorted_particle_ids = np.asarray(state.sorted_particle_ids, dtype=np.int64)
        sorted_part_ids = np.asarray(state.sorted_particle_part_ids, dtype=np.int64)
        if sorted_particle_ids.shape != sorted_part_ids.shape:
            raise ValueError("stored RELION particle ids and part ids must have matching shapes")
        if sorted_particle_ids.size < particle_order.size or not np.array_equal(
            sorted_particle_ids[: particle_order.size],
            np.asarray(particle_order, dtype=np.int64),
        ):
            raise ValueError("sampling-accuracy particle order is not the stored RELION subset prefix")
        random_seed_particle_ids = sorted_part_ids[:n_trials].copy()

    refs_relion = np.stack(
        [np.asarray(recovar_volume_to_relion(ref), dtype=np.float64) for ref in np.asarray(state.Iref)],
        axis=0,
    )
    current_image_size = int(state.effective_current_size)
    accuracy_estimator = (
        estimate_relion_expected_accuracy_in_spawned_process_from_prepared_inputs
        if isolate_in_subprocess
        else estimate_relion_expected_accuracy_from_prepared_inputs
    )
    def group_constant(values, trials, name, n_rows=len(particle_state.translation_offsets)):
        """The one value of an optics constant over ``trials`` (rows of ``values``, a scalar or ``n_rows`` values)."""
        unique = np.unique(np.broadcast_to(np.asarray(values, dtype=np.float64), (n_rows,))[trials])
        if unique.size != 1:
            raise ValueError(f"the expected-accuracy trials of one optics group have several {name} values")
        return float(unique[0])

    def trial_optics(rows):
        """The trials' optics operands, read from the dataset that holds them (a shape class's for several shapes)."""
        if experiment_dataset is None:
            return None
        if isinstance(experiment_dataset, MultiShapeDataset):
            return expected_accuracy_optics(*experiment_dataset.class_rows(rows))
        return expected_accuracy_optics(experiment_dataset, rows)

    def estimate(trials, sigma2_noise_relion):
        """The estimate over the trials at positions ``trials`` (one optics group's)."""
        ids = trial_particle_ids[trials]
        if tilt_images is None:
            optics_kwargs = dict(
                defocus_u=np.asarray(optics_state.defU, dtype=np.float64),
                defocus_v=np.asarray(optics_state.defV, dtype=np.float64),
                defocus_angle=np.asarray(optics_state.defAngle, dtype=np.float64),
                phase_shift=np.asarray(optics_state.phase_shift, dtype=np.float64),
                voltage=group_constant(optics_state.voltage, ids, "voltage"),
                spherical_aberration=group_constant(optics_state.Cs, ids, "Cs"),
                amplitude_contrast=group_constant(optics_state.Q0, ids, "amplitude contrast"),
                pixel_size=float(optics_state.pixel_size),
                optics=trial_optics(ids),
            )
        else:
            # The per-particle defocus arrays are unused; the group's images give the constants.
            zeros = np.zeros(len(particle_state.translation_offsets), dtype=np.float64)
            offsets = np.asarray(tilt_images.image_offsets, dtype=np.int64)
            images = np.concatenate([np.arange(offsets[p], offsets[p + 1]) for p in ids])
            optics_kwargs = dict(
                defocus_u=zeros,
                defocus_v=zeros,
                defocus_angle=zeros,
                phase_shift=zeros,
                voltage=group_constant(
                    tilt_images.voltage, images, "voltage", n_rows=len(tilt_images.voltage)
                ),
                spherical_aberration=group_constant(
                    tilt_images.spherical_aberration,
                    images,
                    "spherical_aberration",
                    n_rows=len(tilt_images.spherical_aberration),
                ),
                amplitude_contrast=group_constant(
                    tilt_images.amplitude_contrast,
                    images,
                    "amplitude_contrast",
                    n_rows=len(tilt_images.amplitude_contrast),
                ),
                pixel_size=float(state.pixel_size),
                tilt_images=tilt_images,
                # The trials' tilt images' exact CTF rows and the magnification's factor, when optics need them.
                optics=trial_optics(images),
            )
        grid_kwargs = dict(current_image_size=current_image_size)
        if optics_state is not None and optics_state.image_pixel_size is not None:
            # A group on another grid: its own pixel size, box and remapped size against the
            # model-size projector (ml_optimiser.cpp:9336-9353).
            pixel = group_constant(optics_state.image_pixel_size, ids, "pixel size")
            box = int(group_constant(optics_state.image_box, ids, "box"))
            scale = optics_scale.scale_difference(box, pixel, int(state.box_size), float(optics_state.pixel_size))
            optics_kwargs["pixel_size"] = pixel
            grid_kwargs = dict(
                current_image_size=optics_scale.group_current_size(current_image_size, box, scale),
                group_grid=dict(
                    model_pixel_size=float(optics_state.pixel_size),
                    image_full_size=box,
                    projector_current_size=current_image_size,
                ),
            )
        return accuracy_estimator(
            references_relion=refs_relion,
            trial_eulers_deg=eulers[trials],
            trial_local_indices=ids,
            trial_class_ids=class_ids[trials],
            class_weights=np.asarray(state.pdf_class, dtype=np.float64),
            sigma2_noise_relion=np.asarray(sigma2_noise_relion, dtype=np.float64),
            **optics_kwargs,
            model_box_size=int(state.box_size),
            **grid_kwargs,
            padding_factor=int(padding_factor),
            sigma2_fudge=float(sigma2_fudge),
            random_seed=int(random_seed),
            do_ctf_correction=True,
            # RELION seeds these trials with Experiment's internal ``part_id``,
            # not the original input-table row ids carried by RECOVAR's dataset.
            random_seed_particle_ids=random_seed_particle_ids[trials],
        )

    sigma2_noise = np.asarray(state.sigma2_noise, dtype=np.float64)
    if sigma2_noise.shape[0] == 1:
        accuracy = estimate(np.arange(n_trials), sigma2_noise[0])
    else:
        if optics_group_ids is None:
            raise ValueError(f"{sigma2_noise.shape[0]} optics groups need each particle's optics group")
        trial_groups = np.asarray(optics_group_ids, dtype=np.int64)[trial_particle_ids]
        group_positions = [np.flatnonzero(trial_groups == g) for g in np.unique(trial_groups)]
        accuracy = _combine_group_expected_accuracies(
            [estimate(positions, sigma2_noise[g]) for g, positions in zip(np.unique(trial_groups), group_positions)],
            group_positions,
            trial_particle_ids,
            random_seed_particle_ids,
        )
    probe.expected_accuracy_estimated(
        AccuracyEstimateInputs(
            state=state,
            particle_state=particle_state,
            optics_state=optics_state,
            references_relion=refs_relion,
            eulers=eulers,
            trial_particle_ids=trial_particle_ids,
            class_ids=class_ids,
            current_image_size=current_image_size,
            padding_factor=padding_factor,
            sigma2_fudge=sigma2_fudge,
            random_seed=random_seed,
            random_seed_particle_ids=random_seed_particle_ids,
        ),
        accuracy,
    )
    return SamplingAccuracyEstimate(
        accuracy=accuracy,
        n_trials=int(n_trials),
        sigma2_fudge=float(sigma2_fudge),
        seed_part_ids=random_seed_particle_ids,
    )


def record_native_sampling_assignment_changes(
    sampling_state: NativeSamplingState,
    *,
    particle_ids: np.ndarray | None,
    previous_translations: np.ndarray,
    current_translations: np.ndarray,
    previous_rotations: np.ndarray | None,
    current_rotations: np.ndarray | None,
    previous_classes: np.ndarray,
    current_classes: np.ndarray,
) -> None:
    if particle_ids is None:
        return
    ids = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return

    prev_t = np.asarray(previous_translations, dtype=np.float64)
    curr_t = np.asarray(current_translations, dtype=np.float64)
    current_offsets = compute_relion_offset_changes_angstrom(
        curr_t[ids],
        prev_t[ids],
        float(sampling_state.pixel_size),
    )
    sampling_state.current_changes_optimal_offsets_angstrom = current_offsets

    current_orientations = compute_relion_orientation_changes(
        None if current_rotations is None else np.asarray(current_rotations)[ids],
        None if previous_rotations is None else np.asarray(previous_rotations)[ids],
    )
    sampling_state.current_changes_optimal_orientations = current_orientations

    prev_c = np.asarray(previous_classes, dtype=np.int32)
    curr_c = np.asarray(current_classes, dtype=np.int32)
    class_changes = float(np.count_nonzero(curr_c[ids] != prev_c[ids])) / float(ids.size)
    sampling_state.current_changes_optimal_classes = class_changes

    if np.isfinite(current_offsets) and np.isfinite(current_orientations):
        # The shared predicate is named for the MPI controller because that
        # path supplies leader-held sampling steps.  InitialModel is the same
        # RELION predicate with this process's current effective steps.
        changes_are_small = relion_mpi_hidden_variable_change_is_small(
            current_classes=class_changes,
            current_offsets_angstrom=current_offsets,
            current_orientations_deg=current_orientations,
            smallest_classes=sampling_state.smallest_changes_optimal_classes,
            smallest_offsets_angstrom=(
                sampling_state.smallest_changes_optimal_offsets_angstrom
            ),
            smallest_orientations_deg=(
                sampling_state.smallest_changes_optimal_orientations
            ),
            mpi_leader_angular_step_deg=sampling.relion_angular_sampling_deg(
                sampling_state.healpix_order,
                sampling_state.adaptive_oversampling,
            ),
            mpi_leader_translation_step_angstrom=(
                sampling_state.effective_offset_step_angstrom
            ),
        )
        if changes_are_small:
            sampling_state.nr_iter_wo_large_hidden_variable_changes += 1
        else:
            sampling_state.nr_iter_wo_large_hidden_variable_changes = 0

        # RELION updates the sticky minima after evaluating the counter.
        if current_offsets < sampling_state.smallest_changes_optimal_offsets_angstrom:
            sampling_state.smallest_changes_optimal_offsets_angstrom = current_offsets
        if current_orientations < sampling_state.smallest_changes_optimal_orientations:
            sampling_state.smallest_changes_optimal_orientations = current_orientations
    else:
        sampling_state.nr_iter_wo_large_hidden_variable_changes = 0

    if class_changes < sampling_state.smallest_changes_optimal_classes:
        # RELION's ROUND macro is floor(x + 0.5), not Python's bankers round.
        sampling_state.smallest_changes_optimal_classes = float(
            np.floor(class_changes + 0.5)
        )


def build_sampling_plan(
    opts: NativeInitialModelOptions,
    *,
    iteration: int = 1,
    sampling_state: NativeSamplingState,
    defer_fine_rotations: bool = False,
) -> NativeSamplingPlan:
    healpix_order = sampling_state.healpix_order
    oversampling = sampling_state.adaptive_oversampling
    offset_range_px = sampling_state.offset_range_px
    offset_step_px = sampling_state.offset_step_px
    offset_range_angstrom = sampling_state.offset_range_angstrom
    offset_step_angstrom = sampling_state.offset_step_angstrom
    if oversampling < 0:
        raise ValueError("oversampling must be >= 0")

    random_perturbation = _random_perturbation_for_iteration(opts, iteration)
    perturbed = abs(random_perturbation) > 1e-12

    metadata_coarse_translations = sampling.get_relion_translation_grid(
        max_pixel=offset_range_px,
        pixel_offset=offset_step_px,
        source_units_per_pixel=float(sampling_state.pixel_size),
    )
    coarse_translations = metadata_coarse_translations.astype(np.float32)
    coarse_pass1_translations = (
        sampling.apply_relion_translation_perturbation(coarse_translations, random_perturbation, offset_step_px).astype(
            np.float32
        )
        if perturbed
        else coarse_translations
    )

    translation_parent = None
    if oversampling == 0:
        # RELION's getOrientations perturbs the angles, and the scorer uses their host inverse matrices
        # (healpix_sampling.cpp:1909-1934); the unperturbed grid is those matrices too.
        rotations = np.asarray(
            sampling.apply_relion_rotation_perturbation_to_eulers(
                sampling._get_relion_rotation_grid_eulers_float64(healpix_order, rotation_index_order="relion"),
                random_perturbation if perturbed else 0.0,
                sampling.relion_angular_sampling_deg(healpix_order),
            )[0],
            dtype=np.float32,
        )
        translations = coarse_translations
        metadata_translations = metadata_coarse_translations
        if perturbed:
            translations = sampling.apply_relion_translation_perturbation(
                translations, random_perturbation, offset_step_px
            ).astype(np.float32)
            metadata_translations = sampling.apply_relion_translation_perturbation(
                metadata_translations, random_perturbation, offset_step_px
            )
    else:
        rotations = None
        if not defer_fine_rotations:
            rotations, _ = sampling.get_oversampled_relion_hidden_rotation_grid_from_samples(
                np.arange(sampling.rotation_grid_size(healpix_order), dtype=np.int64),
                parent_nside_level=healpix_order,
                oversampling_order=oversampling,
                random_perturbation=random_perturbation,
            )
        oversampled_trans, translation_parent = sampling.get_oversampled_translation_grid(
            coarse_translations, pixel_offset=offset_step_px, oversampling_order=oversampling
        )
        metadata_translations, _metadata_translation_parent = sampling.get_oversampled_translation_grid(
            metadata_coarse_translations,
            pixel_offset=offset_step_px,
            oversampling_order=oversampling,
        )
        if not np.array_equal(translation_parent, _metadata_translation_parent):
            raise RuntimeError("GPU and metadata translation parent maps differ")
        translations = sampling.apply_relion_translation_perturbation(
            oversampled_trans.astype(np.float32, copy=False), random_perturbation, offset_step_pixels=offset_step_px
        )
        metadata_translations = sampling.apply_relion_translation_perturbation(
            metadata_translations,
            random_perturbation,
            offset_step_pixels=offset_step_px,
        )

    return NativeSamplingPlan(
        rotations=None if rotations is None else np.asarray(rotations, dtype=np.float32),
        translations=np.asarray(translations, dtype=np.float32),
        random_perturbation=random_perturbation,
        healpix_order=healpix_order,
        oversampling=oversampling,
        offset_range_px=offset_range_px,
        offset_step_px=offset_step_px,
        offset_range_angstrom=offset_range_angstrom,
        offset_step_angstrom=offset_step_angstrom,
        coarse_translations=coarse_pass1_translations,
        coarse_prior_translations=coarse_translations,
        metadata_translations=np.asarray(metadata_translations, dtype=np.float64),
        translation_parent=None if translation_parent is None else np.asarray(translation_parent, dtype=np.int64),
        coarse_base_translations=np.asarray(metadata_coarse_translations, dtype=np.float64),
    )


def _random_perturbation_for_iteration(opts: NativeInitialModelOptions, iteration: int) -> float:
    """The iteration's SamplingPerturbation: the fixed option, or RELION's per-iteration sequence."""
    if opts.random_perturbation is not None:
        return float(opts.random_perturbation)
    if float(opts.perturbation_factor) <= 0.0:
        return 0.0
    # rnd_unif(low, high) performs its range scaling inside RELION's float
    # function. Scaling a separately rounded unit draw changes the result by
    # one float32 ulp for seed 0 / iteration 1, which is enough to flip the
    # integer-truncated fine-projector radius predicate on the rounded rim.
    return sampling.relion_sampling_perturbation_for_iteration(
        float(opts.perturbation_factor),
        int(opts.random_seed),
        max(1, int(iteration)),
    )


def sampling_translation_log_prior(
    translations: np.ndarray,
    *,
    voxel_size: float,
    sigma_angstrom: float,
    old_offsets: np.ndarray,
) -> np.ndarray:
    """Mirror InitialModel's accelerated coarse ``pdf_offset`` arithmetic.

    RELION stores the sampling translations in Angstroms, but its accelerated
    InitialModel path adds them directly to the rounded, pixel-valued previous
    offset before applying one more ``pixel_size**2`` factor.  This mixed-unit
    arithmetic is source behavior and is distinct from both the image
    pre-shift and the offset-variance sufficient statistic. See
    ``docs/math/relion_initial_model_em_parity_conventions.md#prior-preparation``.
    """

    if sigma_angstrom <= 0.0:
        raise ValueError("translation_sigma_angstrom must be positive when provided")
    translations_arr = np.asarray(translations, dtype=np.float64)
    if translations_arr.ndim != 2 or translations_arr.shape[1] != 2:
        raise ValueError(f"translations must have shape (N, 2), got {translations_arr.shape}")
    old_offsets_arr = np.asarray(old_offsets, dtype=np.float64)
    if old_offsets_arr.ndim != 2 or old_offsets_arr.shape[1] != 2:
        raise ValueError(f"old_offsets must have shape (N, 2), got {old_offsets_arr.shape}")

    sampled_translations_angstrom = translations_arr[None, :, :] * float(voxel_size)
    # The prior centre is zero (no particle-offset priors in InitialModel).
    source_differences = old_offsets_arr[:, None, :] + sampled_translations_angstrom
    log_prior = -0.5 * np.sum(source_differences**2, axis=-1) * float(voxel_size) ** 2 / sigma_angstrom**2
    return log_prior.astype(np.float32, copy=False)



def _class_direction_rotation_log_prior(state: InitialModelState, healpix_order: int) -> np.ndarray:
    """Return RELION's class-specific direction prior over coarse rotations.

    RELION copies ``pdf_direction`` into an ``RFLOAT`` buffer and its CUDA
    ``initOrientations`` kernel stores ``log(pdf)`` directly in ``XFLOAT``.
    Do not remove the class-common scale before taking the logarithm.  Although
    that scale cancels analytically, changing it changes float32 addition and
    adaptive-significance ties.
    """

    n_psi = int(sampling.rotation_grid_n_in_planes(int(healpix_order)))
    n_dir = n_directions_for_healpix_order(int(healpix_order))
    n_rot = int(n_dir * n_psi)
    pdf_direction = np.asarray(state.pdf_direction, dtype=np.float64)
    if pdf_direction.shape != (int(state.K), n_dir):
        pdf_direction = np.full((int(state.K), n_dir), 1.0 / float(int(state.K) * n_dir), dtype=np.float64)
    direction_ids = np.arange(n_rot, dtype=np.int64) // n_psi
    values = pdf_direction[:, direction_ids]
    out = np.full(values.shape, -1.0e30, dtype=np.float64)
    positive = values > 0.0
    out[positive] = np.log(values[positive])
    return out.astype(np.float32)



def class_rotation_log_prior_for_sampling(
    state: InitialModelState,
    sampling_state: NativeSamplingState,
    healpix_order: int,
) -> np.ndarray:
    """The live RELION orientation prior; zero prior widths give 1/(n_dir n_psi) (healpix_sampling.cpp:837-842)."""

    if bool(sampling_state.uniform_local_orientation_prior):
        n_rot = int(sampling.rotation_grid_size(int(healpix_order)))
        return np.full((int(state.K), n_rot), np.log(1.0 / float(n_rot)), dtype=np.float32)
    return _class_direction_rotation_log_prior(state, int(healpix_order))
