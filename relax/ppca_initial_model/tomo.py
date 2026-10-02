"""Subtomogram particles for the PPCA initial model (section 16 of ``docs/math/vdam_ppca_algorithm.md``).

A particle is its visible tilt images. They share one pose (rotation ``R``, 3D shift ``t``) and one
latent coordinate; tilt ``k`` is sliced at ``Aproj_k R``, shifted by ``[Aproj_k t]_{1,2}`` and
multiplied by its own (dose-damped) CTF. A *tilt group* is a set of particles whose images come
from one list of frame matrices, so a tile of one group projects every block row once per frame
for all its particles: the frames join the streamed engine's GEMM contraction
(:mod:`relax.ppca_refinement.full_row_stream`). A particle's invisible frames are zero operands.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from recovar import core
from scipy.spatial.transform import Rotation

from relax.helpers.half_spectrum import make_half_image_weights
from relax.helpers.preprocessing import relion_half_translation_lattice
from relax.ppca_initial_model.initialization import initial_noise, seed_model
from relax.ppca_refinement.full_row_stream import _SHIFT_ALIGN, _real_imag, _TileArrays, tile_support
from relax.ppca_refinement.residual_statistics import full_float32


@dataclasses.dataclass(frozen=True)
class TiltParticles:
    """Subtomogram particles as the PPCA's units, over a flat set of tilt images.

    Particle ``p`` owns images ``image_offsets[p]:image_offsets[p + 1]``; image ``i`` is frame
    ``image_frame[i]`` of its particle's group, whose frame matrices (RELION's ``Aproj``) are
    ``group_frames[particle_group[p]]``. ``read(image_ids)`` returns the images' real-space pixels
    ``(n, N, N)``, centered half spectra ``(n, n_half)`` complex64 and CTF rows ``(n, n_half)``
    float32 on the half grid of :meth:`process_images_half`.
    """

    image_shape: tuple
    volume_shape: tuple
    voxel_size: float
    image_offsets: np.ndarray
    image_frame: np.ndarray
    particle_group: np.ndarray
    group_frames: tuple
    read: Callable

    @property
    def n_images(self) -> int:
        """Number of particles: the units the controller selects, batches and embeds."""
        return int(self.image_offsets.size - 1)

    @property
    def grid_size(self) -> int:
        return int(self.image_shape[0])

    def original_image_indices_from_local(self, local):
        return np.asarray(local, dtype=np.int64)

    def particle_images(self, particles):
        """Image ids of these particles, in particle order, and each image's position in ``particles``."""
        particles = np.asarray(particles, dtype=np.int64)
        starts, stops = self.image_offsets[particles], self.image_offsets[particles + 1]
        images = (
            np.concatenate([np.arange(a, b) for a, b in zip(starts, stops)]) if particles.size else np.zeros(0, int)
        )
        return images.astype(np.int64), np.repeat(np.arange(particles.size), stops - starts)


def tilt_particles_from_tomo_dataset(tomo) -> TiltParticles:
    """:class:`TiltParticles` of a :class:`relax.refinement.tomo_half.TomoDataset`.

    Particles of one tomogram form one group when every image's ``Aproj`` equals its frame's
    matrix in the tomogram (one subtomogram matrix); a particle that disagrees forms its own group.
    Tomograms with identical frame lists share a group.
    """
    from relax.helpers.batch_fetch import fetch_indexed_batch
    from relax.relion.relion_ctf import _relion_exact_ctf_half_from_source_star

    offsets = np.asarray(tomo.unit_image_offsets, dtype=np.int64)
    projections = np.asarray(tomo.image_projections, dtype=np.float64)
    frames = np.asarray(tomo.image_frames, dtype=np.int64)
    tables = {}
    for unit, tomogram in enumerate(tomo.unit_tomogram):
        table = tables.setdefault(str(tomogram), {})
        for image in range(offsets[unit], offsets[unit + 1]):
            table.setdefault(int(frames[image]), projections[image])
    keys, group_frames, particle_group, image_frame = (
        {},
        [],
        np.empty(tomo.n_units, np.int64),
        np.empty(frames.size, np.int64),
    )

    def group_of(frame_ids, matrices):
        key = matrices.tobytes()
        if key not in keys:
            keys[key] = len(group_frames)
            group_frames.append(matrices)
        return keys[key], {f: k for k, f in enumerate(frame_ids)}

    for unit, tomogram in enumerate(tomo.unit_tomogram):
        images = np.arange(offsets[unit], offsets[unit + 1])
        table = tables[str(tomogram)]
        if all(np.array_equal(projections[i], table[int(frames[i])]) for i in images):
            frame_ids = sorted(table)
            matrices = np.stack([table[f] for f in frame_ids])
        else:
            frame_ids = [int(f) for f in frames[images]]
            matrices = projections[images]
        group, slot = group_of(frame_ids, matrices)
        particle_group[unit] = group
        image_frame[images] = [slot[int(f)] for f in frames[images]]

    dataset = tomo.images
    image_shape = tuple(int(s) for s in tomo.image_shape)

    def read(image_ids):
        rows = np.asarray(tomo.image_rows, dtype=np.int64)[np.asarray(image_ids, dtype=np.int64)]
        images, _ctf_params, fetched = fetch_indexed_batch(dataset, rows)
        if not np.array_equal(np.asarray(fetched), rows):
            raise RuntimeError("the dataset returned the tilt images in another order")
        half = dataset.process_images_half(jnp.asarray(images), apply_image_mask=False).reshape(rows.size, -1)
        ctf = _relion_exact_ctf_half_from_source_star(dataset, rows, image_shape)
        return np.asarray(images), half.astype(jnp.complex64), jnp.asarray(ctf, jnp.float32)

    return TiltParticles(
        image_shape=image_shape,
        volume_shape=tuple(int(s) for s in tomo.volume_shape),
        voxel_size=float(tomo.voxel_size),
        image_offsets=offsets,
        image_frame=image_frame,
        particle_group=particle_group,
        group_frames=tuple(np.asarray(m, np.float64) for m in group_frames),
        read=read,
    )


def tilt_tiles(particles: TiltParticles, ids, tile_size: int) -> list[np.ndarray]:
    """Cut ``ids`` into tiles of at most ``tile_size`` particles of one tilt group each.

    Groups appear in order of their first particle in ``ids``; particles keep their order within a group.
    """
    ids = np.asarray(ids, dtype=np.int64)
    groups = particles.particle_group[ids]
    _, first = np.unique(groups, return_index=True)
    tiles = []
    for group in groups[np.sort(first)]:
        members = ids[groups == group]
        tiles.extend(members[start : start + tile_size] for start in range(0, members.size, tile_size))
    return tiles


def tilt_shifts(frames, translations) -> np.ndarray:
    """2D shift ``[Aproj t]_{1,2}`` of each frame for each 3D shift, ``(K, T, 2)`` float32.

    RELION's ``Experiment::getTranslationInTiltSeries`` (:func:`relax.refinement.tomo_particles.tilt_image_shifts`).
    """
    frames = np.asarray(frames, dtype=np.float64)
    return np.einsum("kab,tb->kta", frames[:, :2, :], np.asarray(translations, dtype=np.float64)).astype(np.float32)


def load_tilt_tile(stream, image_indices, significant_rows, *, collect_observation: bool):
    """:class:`FullRowStream` tile reader for subtomogram particles (one tilt group per tile).

    Operands follow section 16.4: frame-major ``(frame, [Re | Im], pixel)`` GEMM axes, each tilt's
    image phase-shifted by its own projection of every 3D shift, and zeros at frames a particle
    does not see. ``n_observations`` counts the tile's tilt images (one noise observation each).
    """
    particles = stream.dataset
    ids = np.asarray(image_indices, dtype=np.int64)
    groups = np.unique(particles.particle_group[ids])
    if groups.size != 1:
        raise ValueError("A subtomogram tile must hold particles of one tilt group")
    frames = particles.group_frames[int(groups[0])]
    images, owner = particles.particle_images(ids)
    slots = particles.image_frame[images]
    _, half, ctf = particles.read(images)
    half, ctf = jax.device_put((half, ctf), stream.device)
    resolved = stream.resolved
    nv = stream.noise_variance_half
    window = np.arange(half.shape[1]) if resolved.score_indices is None else np.asarray(resolved.score_indices)
    lattice = np.asarray(relion_half_translation_lattice(stream.static.image_shape), np.float32)[window]
    shifts = jnp.asarray(tilt_shifts(frames, stream.translations)[slots])  # (n, T, 2)
    phases = jnp.exp(jnp.complex64(-2j * np.pi) * jnp.einsum("ntd,fd->ntf", shifts, lattice, precision="highest"))
    weighted = (half * ctf / nv)[:, window]
    ctf2 = (ctf * ctf / nv)[:, window]
    score_mask, recon_mask = resolved.score_mask[window], resolved.recon_mask[window]
    B, K, T, F = ids.size, frames.shape[0], int(stream.translations.shape[0]), window.size
    # GPU streams pad each frame's pixels to the GEMM window and the (image, shift) axis to its multiple.
    pad = (stream.arrays.gemm_window.shape[0] - F) if stream.static.cuda_kernels else 0
    shift_pad = -(B * T) % _SHIFT_ALIGN if stream.static.cuda_kernels else 0

    def per_frame(values):
        """``(n, ..., F)`` image values into zero-padded ``(B, K, ..., F + pad)`` frame slots."""
        values = jnp.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, pad)])
        full = jnp.zeros((B, K) + values.shape[1:], values.dtype)
        return full.at[owner, slots].set(values)

    Y_score = per_frame(weighted[:, None, :] * score_mask * phases)  # (B, K, T, F')
    Y_recon = per_frame(weighted[:, None, :] * recon_mask * phases)
    ctf2_score, ctf2_recon = per_frame(ctf2 * score_mask), per_frame(ctf2 * recon_mask)  # (B, K, F')
    norm = jnp.sum(jnp.abs(half) ** 2 / nv * make_half_image_weights(stream.static.image_shape), axis=1)
    coarse_mask, table, layout = tile_support(stream, significant_rows)
    tile = _TileArrays(
        coarse_mask=coarse_mask,
        rows=table,
        Y1=jnp.pad(jnp.transpose(_real_imag(Y_score), (1, 3, 0, 2)).reshape(-1, B * T), ((0, 0), (0, shift_pad))),
        ctf2=jnp.transpose(ctf2_score, (1, 2, 0)).reshape(-1, B),
        Y1_recon=jnp.pad(jnp.transpose(_real_imag(Y_recon), (0, 2, 1, 3)).reshape(B * T, -1), ((0, shift_pad), (0, 0))),
        ctf2_recon=ctf2_recon.reshape(B, -1),
        y_norm=jax.ops.segment_sum(norm, jnp.asarray(owner), num_segments=B),
        frames=jnp.asarray(frames, jnp.float32),
    )
    observation = jnp.sum(jnp.abs(half) ** 2, axis=0) if collect_observation else None
    layout.update(n_observations=int(images.size), original_ids=particles.original_image_indices_from_local(ids))
    return tile, observation, layout


def tilt_bootstrap(particles: TiltParticles, rng, channels: int, radius: int, *, image_target: int = 1000):
    """Random-angle round-robin reconstructions of the PPCA seed maps from tilt images.

    Random particles are read until about ``image_target`` tilt images; each particle gets one
    random rotation and every visible tilt backprojects at ``Aproj_k R`` with its own CTF, into
    seed map ``position % channels``. Returns the per-map RHS and CTF^2 half volumes and the
    tilt images for the initial noise estimate.
    """
    order = rng.permutation(particles.n_images)
    counts = np.diff(particles.image_offsets)[order]
    chosen = order[: int(np.searchsorted(np.cumsum(counts), image_target)) + 1]
    rotations = Rotation.random(chosen.size, random_state=rng).as_matrix()
    images, owner = particles.particle_images(chosen)
    frames = np.stack(
        [
            particles.group_frames[particles.particle_group[chosen[o]]][particles.image_frame[i]]
            for i, o in zip(images, owner)
        ]
    )
    tilt_rotations = np.einsum("nab,nbc->nac", frames, rotations[owner]).astype(np.float32)
    raw, half, ctf = particles.read(images)
    labels = owner % channels
    n_half_volume = int(np.prod(particles.volume_shape[:2])) * (particles.volume_shape[2] // 2 + 1)
    rhs = jnp.zeros((channels, n_half_volume), jnp.complex64)
    lhs = jnp.zeros(rhs.shape, jnp.float32)
    for k in range(channels):
        select = np.flatnonzero(labels == k)
        if not select.size:
            continue
        slice_args = (tilt_rotations[select], particles.image_shape, particles.volume_shape, "linear_interp")
        options = dict(half_image=True, half_volume=True, max_r=radius)
        rhs = rhs.at[k].add(core.adjoint_slice_volume(half[select] * ctf[select], *slice_args, **options))
        lhs = lhs.at[k].add(core.adjoint_slice_volume(ctf[select] ** 2, *slice_args, **options).real)
    return rhs, lhs, [(0, np.asarray(image, np.float32)) for image in raw], chosen


@full_float32
def initialize_tilts(particles: TiltParticles, *, seed, diameter_ang, q=2):
    """:func:`relax.ppca_initial_model.initialization.initialize` for subtomogram particles (section 16.5)."""
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q <= 0:
        raise ValueError("q must be a positive integer")
    rng = np.random.default_rng(seed)
    radius = max(1, int(np.floor(0.07 * particles.grid_size + 0.5)))
    rhs, lhs, images_for_noise, chosen = tilt_bootstrap(particles, rng, q + 1, radius)
    theta = seed_model(rhs, lhs, particles.volume_shape, particles.voxel_size, diameter_ang, radius, rng)
    noise = initial_noise(images_for_noise, particles.image_shape, particles.voxel_size, diameter_ang)
    info = {
        "seed": seed,
        "bootstrap_particles": chosen.tolist(),
        "bootstrap_tilt_images": len(images_for_noise),
        "bootstrap_radius": radius,
        "projection": "complex64",
        "accumulation": "complex64/float32",
        "noise_initializer": "deliberate host float64; coefficient spectrum float32",
    }
    return theta, noise, info
