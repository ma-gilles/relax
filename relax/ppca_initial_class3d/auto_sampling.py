"""RELION's gradient-refinement auto-sampling for the mixture (relion_refine ``--auto_sampling``).

Every ``accuracy_interval`` updates relion_refine estimates how far the orientations and shifts of 100
particles can move before the projections of their class differ from those at their stored poses at
P = 0.01 (``calculateExpectedAngularErrors``, ml_optimiser.cpp:9291-9686: ``exp(-4.60517) = 0.01`` of the
noise-weighted squared difference, the angle or shift grown through RELION's step ladder along one Euler
angle or shift axis drawn per particle), and only when the model's resolution has not improved for two
updates (``updateCurrentResolution``: the first shell whose signal-to-noise ratio drops below one), raises
the angular order while the oversampled step stays above 90% of that accuracy, tightens the offset step
to 90% of the shift accuracy (never below three quarters of the previous step) and switches to local
searches once the order reaches ``--auto_local_healpix_order`` (ml_optimiser_mpi.cpp:3680-3830). The
offset range follows the owner's non-MPI rule: three times the last observed shift changes, at most 30%
wider than before, between 1.5 and 4 offset steps (ml_optimiser.cpp:9800-9870).

The mixture's children are the oversampling, so the "oversampled step" is the children's angular step of a
coarse order, local searches start at ``local_order`` (coarse 3 with children is RELION's base order 4,
3.75 degrees) and the order never exceeds ``max_order`` (the coarse support mask of a tile is dense). The
resolution comes from the VDAM gates (``rho = 2 fudge signal / disagreement``, gate below 0.5 is RELION's
``data_vs_prior < 1``). Two documented departures: the switch to local searches is immediate instead of
suspended for two sampling updates (20 updates on RELION's schedule, longer than a short run), and the
estimate uses the class means (the loadings do not enter RELION's reference comparison).
"""

import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from recovar import core
from recovar.reconstruction import noise as noise_utils
from recovar.reconstruction.noise import make_radial_noise

from relax.healpix_sampling import euler_angles_to_matrix
from relax.helpers.convergence import healpix_angular_step
from relax.helpers.preprocessing import half_translation_phase_table, relion_half_translation_lattice
from relax.ppca_initial_class3d.local_search import relion_local_sigma_deg
from relax.ppca_initial_model.tomo import TiltParticles, tilt_shifts

PVALUE = 4.60517  # exp(-4.60517) = 0.01: RELION's probability threshold on the projection difference
ANGLE_LADDER = ((0.2, 0.05), (1.0, 0.1), (2.0, 0.2), (5.0, 0.5), (10.0, 1.0), (20.0, 2.0), (np.inf, 5.0))
SHIFT_LADDER = ((1.0, 0.1), (2.0, 0.2), (5.0, 0.5), (10.0, 1.0), (np.inf, 2.0))
MAX_ANGLE_DEG, MAX_SHIFT_PX = 30.0, 10.0
ACCURACY_PARTICLES = 100  # RELION's n_trials_acc for 3D references
MIN_CLASS_PRIOR = 0.01  # classes below it are not estimated (``pdf_class < 0.01``)
SNR_THRESHOLD = 0.5  # VDAM gate rho / (1 + rho) at rho = 1, RELION's data_vs_prior = 1


def children_step_deg(coarse_order: int) -> float:
    """The children's angular step of a coarse order: RELION's oversampled step."""
    return healpix_angular_step(int(coarse_order) + 1)


def error_ladder(ladder, maximum):
    """RELION's growing perturbations up to ``maximum``, then the first value beyond it (the accuracy of a
    particle whose projections never differ enough)."""
    values, error = [], 0.0
    while True:
        step = next(step for bound, step in ladder if error < bound)
        error = round(error + step, 6)
        if error > maximum:
            return np.asarray(values, np.float64), error
        values.append(error)


@dataclass(frozen=True)
class SamplingState:
    """The sampling an auto-sampled run is at, checkpointed with the model."""

    healpix_order: int
    offset_step_px: float
    offset_range_px: float
    local_searches: bool = False
    sigma_deg: float = 0.0
    acc_rot_deg: float = 999.0  # RELION's initial 999
    acc_trans_px: float = 999.0
    resolution_shell: int = 0
    updates_without_resolution_gain: int = 0
    offset_changes_px: float = 999.0  # RELION's current_changes_optimal_offsets, 999 before any
    last_estimate_iteration: int = 0

    @classmethod
    def initial(cls, config):
        return cls(int(config.stage(1)[1]), float(config.shift_step), float(config.shift_range))

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, values) -> "SamplingState":
        return cls(**values)

    def grid_key(self):
        return (self.healpix_order, self.offset_step_px, self.offset_range_px)


def resolution_shell(class_diagnostics, *, radius: int) -> int:
    """RELION's ``updateCurrentResolution`` on the VDAM gates: per class the shell before the first one (from
    shell 1) whose mean-channel gate is below 0.5, the best class over all; at most the support radius."""
    best = 0
    for diagnostics in class_diagnostics:
        gates = np.asarray(diagnostics["gates"], np.float64)
        gates = gates[:, 0] if gates.ndim == 2 else gates
        shell = 1
        while shell < gates.shape[0] and gates[shell] >= SNR_THRESHOLD:
            shell += 1
        best = max(best, shell - 1)
    return int(min(best, radius))


def track_resolution(sampling: SamplingState, shell: int) -> SamplingState:
    """Count the updates without a resolution gain, as ``updateCurrentResolution`` does."""
    gained = shell > sampling.resolution_shell
    return dataclasses.replace(
        sampling, resolution_shell=max(int(shell), sampling.resolution_shell) if gained else sampling.resolution_shell,
        updates_without_resolution_gain=0 if gained else sampling.updates_without_resolution_gain + 1,
    )


def update_due(iteration: int, config) -> bool:
    """A sampling update happens every ``accuracy_interval`` updates after the first and before the last
    (``iter % 10 == 0``, ml_optimiser.cpp:3556); it estimates the accuracy only when the resolution stalled for
    two updates (``nr_iter_wo_resol_gain >= 2``, ml_optimiser_mpi.cpp:3690), else nothing changes."""
    return iteration > 1 and iteration % int(config.accuracy_interval) == 0 and iteration < int(config.iterations)


STALLED_UPDATES = 2


def update_sampling(sampling: SamplingState, config, *, acc_rot_deg=None, acc_trans_px=None, iteration=None):
    """The next sampling after an accuracy estimate (``None``: the resolution still gains, nothing changes)."""
    if acc_rot_deg is None:
        return sampling, {"changed": False, "reason": "resolution still improving"}
    order, step, offset_range = sampling.healpix_order, sampling.offset_step_px, sampling.offset_range_px
    # Twice as fine angular sampling while the oversampled step is not below 90% of the accuracy.
    candidate = order + 1
    if children_step_deg(candidate) < 0.9 * acc_rot_deg:
        candidate = order
    new_order = min(candidate, int(config.max_order))
    # Translational sampling: 90% of the accuracy (oversampled), at least three quarters of the previous step,
    # never coarser than 95% of the range.
    min_step = min(0.9 * acc_trans_px * 2.0, 0.95 * offset_range)
    new_step = max(0.75 * step, min_step)
    new_range = offset_range
    if new_step > step:  # don't go to coarser samplings: keep the previous ones
        new_step = step
    else:
        changes = sampling.offset_changes_px
        new_range = 3.0 * changes if np.isfinite(changes) and changes < 999.0 else offset_range
        new_range = min(1.3 * offset_range, new_range)
        new_range = max(new_range, 1.5 * new_step)
        if new_range > 4.0 * new_step:
            new_range /= 2.0
        if new_range > 4.0 * new_step:
            new_step = new_range / 4.0
    local = sampling.local_searches or new_order >= int(config.local_order)
    sigma = relion_local_sigma_deg(new_order) if local else 0.0
    changed = (new_order != order or new_step != step or new_range != offset_range
               or local != sampling.local_searches)
    updated = dataclasses.replace(
        sampling, healpix_order=int(new_order), offset_step_px=float(new_step), offset_range_px=float(new_range),
        local_searches=bool(local), sigma_deg=float(sigma), acc_rot_deg=float(acc_rot_deg),
        acc_trans_px=float(acc_trans_px),
        updates_without_resolution_gain=0 if changed else sampling.updates_without_resolution_gain,
        last_estimate_iteration=int(iteration) if iteration is not None else sampling.last_estimate_iteration,
    )
    return updated, {"changed": bool(changed), "reason": "accuracy", "order": int(new_order),
                     "offset_step_px": float(new_step), "offset_range_px": float(new_range),
                     "local_searches": bool(local), "sigma_deg": float(sigma)}


def _snr(f1, f2, ctf, noise_half, window):
    """RELION's ``my_snr``: the noise-weighted squared projection difference over the images and pixels."""
    diff = jnp.abs(f1[None] - f2) ** 2 * (ctf[None] ** 2) / (2.0 * noise_half)
    return jnp.sum(jnp.where(window, diff, 0.0), axis=(1, 2))


def expected_errors(dataset, state, config, membership, *, radius: int, rng, image_shape=None):
    """RELION's expected angular (degrees) and shift (pixels) accuracy of the best class on up to
    ``accuracy_particles`` particles with stored poses, and the per-class values."""
    if not isinstance(dataset, TiltParticles):
        raise NotImplementedError("Auto-sampling's accuracy estimate is implemented for tilt particles")
    image_shape = dataset.image_shape if image_shape is None else image_shape
    known = np.flatnonzero(np.isfinite(membership.euler_deg).all(axis=1))
    if known.size == 0:
        raise ValueError("The accuracy estimate needs particles with stored poses")
    original = dataset.original_image_indices_from_local(np.arange(dataset.n_images))
    local_of = {int(o): i for i, o in enumerate(original)}
    chosen = rng.choice(known, size=min(int(config.accuracy_particles), known.size), replace=False)
    lattice = np.asarray(relion_half_translation_lattice(image_shape), np.float64)
    shell = np.hypot(lattice[:, 0], lattice[:, 1])
    window = jnp.asarray((shell > 0) & (shell <= radius))
    angles, angle_beyond = error_ladder(ANGLE_LADDER, MAX_ANGLE_DEG)
    shifts, shift_beyond = error_ladder(SHIFT_LADDER, MAX_SHIFT_PX)
    noise_rows = np.asarray(state.noise)
    noise_half = {}
    per_class = {}
    priors = np.asarray(state.class_prior, np.float64)
    for k in range(int(state.theta.shape[0])):
        if priors[k] < MIN_CLASS_PRIOR:
            per_class[k] = {"acc_rot_deg": 999.0, "acc_trans_px": 999.0, "particles": 0}
            continue
        mean = jnp.asarray(state.theta[k][:, 0])[None]
        rot_sum = trans_sum = 0.0
        for row in chosen:
            particle = local_of[int(membership.particle_ids[row])]
            images, _ = dataset.particle_images([particle])
            frames = np.asarray(dataset.group_frames[int(dataset.particle_group[particle])], np.float64)
            frames = frames[np.asarray(dataset.image_frame[images])]
            _, _, ctf = dataset.read(images)
            group = int(dataset.particle_noise_group[particle])
            if group not in noise_half:
                nv = make_radial_noise(noise_rows[group], image_shape)
                noise_half[group] = jnp.asarray(noise_utils.to_batched_half_pixel_noise(nv, image_shape)).squeeze()
            eulers = np.asarray(membership.euler_deg[row], np.float64)
            draw = np.random.default_rng(int(config.seed) + int(membership.particle_ids[row])).uniform()
            axis = 0 if draw < 1 / 3 else (1 if draw < 2 / 3 else 2)
            base = euler_angles_to_matrix(eulers[None])[0]
            perturbed = np.repeat(eulers[None], angles.size, axis=0)
            perturbed[:, axis] += angles
            rotations = np.einsum("fab,rbc->rfac", frames, euler_angles_to_matrix(perturbed))  # (E, K, 3, 3)
            reference = np.einsum("fab,bc->fac", frames, base)  # (K, 3, 3)
            f1 = _project(mean, reference, image_shape, dataset.volume_shape)
            f2 = _project(mean, rotations.reshape(-1, 3, 3), image_shape, dataset.volume_shape)
            snr = np.asarray(_snr(f1, f2.reshape(angles.size, frames.shape[0], -1), jnp.asarray(ctf),
                                  noise_half[group], window))
            rot_sum += _first_beyond(snr, angles, angle_beyond)
            shift = np.zeros((shifts.size, 3))
            shift[:, axis] = shifts
            planar = tilt_shifts(frames, shift)  # (K, E, 2)
            phases = jnp.stack([
                half_translation_phase_table(planar[f], image_shape) for f in range(frames.shape[0])
            ], axis=1)  # (E, K, n_half)
            snr = np.asarray(_snr(f1, f1[None] * phases, jnp.asarray(ctf), noise_half[group], window))
            trans_sum += _first_beyond(snr, shifts, shift_beyond)
        per_class[k] = {"acc_rot_deg": rot_sum / chosen.size, "acc_trans_px": trans_sum / chosen.size,
                        "particles": int(chosen.size)}
    acc_rot = min(v["acc_rot_deg"] for v in per_class.values())
    acc_trans = min(v["acc_trans_px"] for v in per_class.values())
    return float(acc_rot), float(acc_trans), per_class


def _project(mean, rotations, image_shape, volume_shape):
    """Half-image projections ``(R, n_half)`` of the class mean at ``rotations``, the stream's projector."""
    return core.batch_slice_volume(
        mean, jnp.asarray(np.asarray(rotations, np.float32)), tuple(image_shape), tuple(volume_shape),
        "linear_interp", half_volume=True, half_image=True,
    )[0]


def _first_beyond(snr, errors, beyond):
    hits = np.flatnonzero(snr > PVALUE)
    return float(errors[hits[0]]) if hits.size else float(beyond)
