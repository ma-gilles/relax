"""Given particle poses for subtomogram PPCA (opt-in ``Config.init_poses`` with ``Config.pose_mode``).

Pose file (``.npz``):

- ``rotations`` ``(P, 3, 3)``: each particle's rotation ``R_p`` in the stream's convention: tilt ``f`` of the particle
  is the slice at ``Aproj_f R_p`` (``Aproj_f`` the tilt series' frame matrix, RELION's tilt x subtomogram matrix).
  RELION's particle angles give it as ``relax.sampling.healpix.euler_angles_to_matrix([rlnAngleRot, rlnAngleTilt,
  rlnAnglePsi])`` (``ParticleSet::getMatrix3x3`` = ``A_subtomogram A_particle``); ``--final-poses``'s
  ``best_rotation_matrix`` is the same quantity.
- ``translations_px`` ``(P, 3)``: each particle's 3D shift ``t_p`` in pixels of the particle box: tilt ``f`` is shifted
  by ``[Aproj_f t_p]_{1,2}`` (RELION's ``getTranslationInTiltSeries``; from a STAR file
  ``rlnOrigin{X,Y,Z}Angst / rlnImagePixelSize``; ``--final-poses``'s ``best_translation_px``).
- ``particle_ids`` ``(P,)`` int, optional: the training particle index of each row (the order of the particles STAR);
  without it the rows are the particles in that order. Every particle appears exactly once.

Each particle is then trained in its own frame (:func:`particles_at_poses`): its tilt frames become
``Aproj_f R_p`` and its tilt images are shifted by ``[Aproj_f t_p]_{1,2}`` (the phase the stream puts on a trial
shift), so the stream's pose ``(D, d)`` stands for the particle pose ``(R_p D, t_p + R_p d)``. ``pose_mode`` "fixed"
scores the one pose ``(I, 0)``; "local" the stage's HEALPix rotations within ``3 sigma`` of the identity with the
Gaussian log prior ``-angle^2 / (2 sigma^2)`` (:func:`local_rotations`) and the stage's shift grid with the learned
Gaussian offset prior. Both likelihoods read the composed frames and shifted images.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from relax.fourier.preprocessing import relion_half_translation_lattice, translation_phases
from relax.ppca_initial_model.tomo import TiltParticles, tilt_shifts


@dataclasses.dataclass(frozen=True)
class InputPoses:
    """The particles' given poses, in training-particle order."""

    rotations: np.ndarray  # (P, 3, 3) float64
    translations_px: np.ndarray  # (P, 3) float64


def read_pose_file(path, n_particles: int) -> InputPoses:
    """The poses of :mod:`this module's <relax.ppca_initial_model.pose_input>` file format, checked."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ("rotations", "translations_px") if k not in z.files]
        if missing:
            raise ValueError(f"{path}: pose file lacks {missing}")
        rotations = np.asarray(z["rotations"], np.float64)
        translations = np.asarray(z["translations_px"], np.float64)
        ids = np.asarray(z["particle_ids"], np.int64) if "particle_ids" in z.files else np.arange(len(rotations))
    if rotations.shape != (n_particles, 3, 3) or translations.shape != (n_particles, 3) or ids.shape != (n_particles,):
        raise ValueError(
            f"{path}: expected rotations ({n_particles}, 3, 3), translations_px ({n_particles}, 3) and particle_ids "
            f"({n_particles},), got {rotations.shape}, {translations.shape}, {ids.shape}"
        )
    if not np.array_equal(np.sort(ids), np.arange(n_particles)):
        raise ValueError(f"{path}: particle_ids must list every training particle exactly once")
    if not (np.all(np.isfinite(rotations)) and np.all(np.isfinite(translations))):
        raise ValueError(f"{path}: nonfinite poses")
    orthogonal = np.abs(np.einsum("pab,pcb->pac", rotations, rotations) - np.eye(3)).max()
    if orthogonal > 1e-4 or np.any(np.linalg.det(rotations) < 0):
        raise ValueError(f"{path}: rotations must be proper rotation matrices (orthogonality error {orthogonal:.2e})")
    order = np.argsort(ids)
    return InputPoses(rotations[order], translations[order])


def particles_at_poses(particles: TiltParticles, poses: InputPoses) -> TiltParticles:
    """``particles`` in their own frames: one tilt group per particle with frames ``Aproj_f R_p``, and every tilt
    image shifted by ``[Aproj_f t_p]_{1,2}`` (see the module docstring)."""
    if particles.core_subtraction is not None or particles.volume_observations is not None:
        raise ValueError("Given poses apply to the particles as read (before a fixed core or volume observations)")
    n = particles.n_images
    if poses.rotations.shape[0] != n:
        raise ValueError("One pose per particle")
    frames = [np.asarray(particles.group_frames[g], np.float64) for g in particles.particle_group]
    group_frames = tuple(f @ R for f, R in zip(frames, poses.rotations))
    # Each image's 2D shift in its original frame, (images, 2).
    shifts = np.zeros((int(particles.image_offsets[-1]), 2), np.float32)
    for p in range(n):
        start, stop = int(particles.image_offsets[p]), int(particles.image_offsets[p + 1])
        frame_ids = particles.image_frame[start:stop]
        shifts[start:stop] = tilt_shifts(frames[p][frame_ids], poses.translations_px[p][None].astype(np.float32))[:, 0]
    lattice = jnp.asarray(np.asarray(relion_half_translation_lattice(tuple(particles.image_shape)), np.float32))
    read = particles.read

    def read_shifted(image_ids):
        image_ids = np.asarray(image_ids, np.int64)
        real, half, ctf = read(image_ids)
        phases = translation_phases(jnp.asarray(shifts[image_ids]), lattice)
        return real, (jnp.asarray(half) * phases).astype(jnp.complex64), ctf

    return dataclasses.replace(
        particles,
        group_frames=group_frames,
        particle_group=np.arange(n, dtype=np.int64),
        read=read_shifted,
        input_poses=poses,
    )


def local_rotations(rotation_grid: np.ndarray, sigma_deg: float, cutoff: float = 3.0):
    """The rotations of ``rotation_grid`` within ``cutoff sigma`` (geodesic angle) of the identity and their normalised
    Gaussian log prior ``-angle^2 / (2 sigma^2)``: RELION's local search about a prior at the identity, isotropic in
    SO(3) (the particle frame makes every prior the identity). Raises when the cone holds no grid rotation."""
    grid = np.asarray(rotation_grid, np.float64)
    angle = np.degrees(np.arccos(np.clip((np.trace(grid, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    keep = np.flatnonzero(angle < cutoff * sigma_deg)
    if keep.size == 0:
        raise ValueError(f"No grid rotation within {cutoff} x {sigma_deg} deg of the given pose: widen the local sigma")
    log_prior = -0.5 * (angle[keep] / sigma_deg) ** 2
    log_prior = log_prior - np.log(np.sum(np.exp(log_prior - log_prior.max()))) - log_prior.max()
    return grid[keep].astype(np.float32), log_prior.astype(np.float32), keep


def absolute_poses(poses: InputPoses, rotations, translations):
    """Particle poses ``(R_p D, t_p + R_p d)`` of stream poses ``(D, d)`` ``(P, 3, 3)``, ``(P, 3)`` in the frames."""
    R = np.einsum("pab,pbc->pac", poses.rotations, np.asarray(rotations, np.float64))
    t = poses.translations_px + np.einsum("pab,pb->pa", poses.rotations, np.asarray(translations, np.float64))
    return R, t
