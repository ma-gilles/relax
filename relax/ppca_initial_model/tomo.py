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
import functools
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from recovar import core
from scipy.spatial.transform import Rotation

from relax.fourier.half_spectrum import make_half_image_weights
from relax.fourier.preprocessing import relion_half_translation_lattice
from relax.ppca_initial_model.initialization import initial_noise, seed_model
from relax.ppca_refinement.full_row_stream import _SHIFT_ALIGN, _real_imag, _TileArrays, tile_size_bucket, tile_support
from relax.ppca_refinement.residual_statistics import full_float32
from relax.vdam.bootstrap_iref import compute_ini_high_shell


@dataclasses.dataclass(frozen=True)
class TiltParticles:
    """Subtomogram particles as the PPCA's units, over a flat set of tilt images.

    Particle ``p`` owns images ``image_offsets[p]:image_offsets[p + 1]``; image ``i`` is frame
    ``image_frame[i]`` of its particle's group, whose frame matrices (RELION's ``Aproj``) are
    ``group_frames[particle_group[p]]``. ``read(image_ids)`` returns the images' real-space pixels
    ``(n, N, N)``, centered half spectra ``(n, n_half)`` complex64 and CTF rows ``(n, n_half)``
    float32 on the half grid of :meth:`process_images_half`. ``noise_group`` (``0..G-1`` per
    particle, its optics group; ``None`` is one group) selects the particle's noise spectrum.
    """

    image_shape: tuple
    volume_shape: tuple
    voxel_size: float
    image_offsets: np.ndarray
    image_frame: np.ndarray
    particle_group: np.ndarray
    group_frames: tuple
    read: Callable
    noise_group: np.ndarray | None = None

    @property
    def particle_noise_group(self) -> np.ndarray:
        """Noise group of every particle (all zero with one group)."""
        if self.noise_group is None:
            return np.zeros(self.n_images, np.int64)
        return np.asarray(self.noise_group, np.int64)

    @property
    def n_noise_groups(self) -> int:
        return int(self.particle_noise_group.max()) + 1 if self.n_images else 1

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
    from relax.io.batch_fetch import fetch_indexed_batch
    from relax.relion.ctf import relion_exact_ctf_half_from_source_star

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
        ctf = relion_exact_ctf_half_from_source_star(dataset, rows, image_shape)
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
        # One noise spectrum per optics group (one per tomogram on RELION 5 imports).
        noise_group=np.unique(np.asarray(tomo.unit_optics_group), return_inverse=True)[1].reshape(-1),
    )


def tilt_tiles(particles: TiltParticles, ids, tile_size: int) -> list[np.ndarray]:
    """Cut ``ids`` into tiles of at most ``tile_size`` particles of one tilt group and one noise group each.

    Groups appear in order of their first particle in ``ids``; particles keep their order within a group.
    """
    ids = np.asarray(ids, dtype=np.int64)
    groups = particles.particle_group[ids] * particles.n_noise_groups + particles.particle_noise_group[ids]
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
    With a planned tile size on the stream, the tile is padded to its :func:`tile_size_bucket` with
    particles that see no frame (zero operands) after the real ones (``n_real``), so a stage compiles
    one operand program and one stream program per bucket.
    """
    particles = stream.dataset
    ids = np.asarray(image_indices, dtype=np.int64)
    groups = np.unique(particles.particle_group[ids])
    if groups.size != 1:
        raise ValueError("A subtomogram tile must hold particles of one tilt group")
    frames = particles.group_frames[int(groups[0])]
    images, owner = particles.particle_images(ids)
    B, K, T = ids.size, frames.shape[0], int(stream.translations.shape[0])
    padded = B if stream.tile_images is None else tile_size_bucket(B, stream.tile_images)
    # Read the tile's images in (particle, frame) slot order, so the operand program sees one shape per
    # tile size; a frame the particle does not see re-reads its first image and is masked to zero.
    slot_image = np.full(padded * K, -1, np.int64)
    slot_image[owner * K + particles.image_frame[images]] = images
    visible = slot_image >= 0
    _, half, ctf = particles.read(np.where(visible, slot_image, images[0]))
    half, ctf = jax.device_put((half, ctf), stream.device)
    window, constants, static = _operand_layout(stream, padded, K)
    Y1, ctf2, Y1_recon, ctf2_recon, y_norm, observation = _tilt_operands(
        half,
        ctf,
        jnp.asarray(visible),
        stream.noise_variance_half,
        jnp.asarray(window),
        constants["lattice"],
        jnp.asarray(tilt_shifts(frames, stream.translations)),
        constants["score_mask"],
        constants["recon_mask"],
        constants["weights"],
        **static,
    )
    coarse_mask, table, layout = tile_support(stream, significant_rows)
    tile = _TileArrays(
        # Padding particles take the first particle's support; the stream gives them no posterior mass.
        coarse_mask=jnp.concatenate([coarse_mask, jnp.repeat(coarse_mask[:1], padded - B, axis=0)]),
        rows=table,
        Y1=Y1,
        ctf2=ctf2,
        Y1_recon=Y1_recon,
        ctf2_recon=ctf2_recon,
        y_norm=y_norm,
        frames=jnp.asarray(frames, jnp.float32),
    )
    if stream.static.cuda_kernels and tile.Y1.shape[0] != K * 2 * stream.arrays.gemm_window.shape[0]:
        # The CUDA projector writes each frame's rows padded to the GEMM window, frame-major.
        raise RuntimeError("Tilt tile operands do not follow the frame-major GEMM window layout")
    observation = observation if collect_observation else None
    layout.update(
        n_real=B,
        n_observations=int(images.size),
        original_ids=particles.original_image_indices_from_local(ids),
        frames=frames,
    )
    return tile, observation, layout


def _operand_layout(stream, n_particles: int, n_frames: int):
    """The operand program's score window, per-pixel constants and static shape arguments for a tile size."""
    resolved = stream.resolved
    n_half = int(np.asarray(stream.noise_variance_half).size)
    window = np.arange(n_half) if resolved.score_indices is None else np.asarray(resolved.score_indices)
    T = int(stream.translations.shape[0])
    constants = {
        "lattice": jnp.asarray(
            np.asarray(relion_half_translation_lattice(stream.static.image_shape), np.float32)[window]
        ),
        "score_mask": resolved.score_mask[window],
        "recon_mask": resolved.recon_mask[window],
        "weights": make_half_image_weights(stream.static.image_shape),
    }
    static = {
        "n_particles": int(n_particles),
        "n_frames": int(n_frames),
        # GPU streams pad each frame's pixels to the GEMM window and the (image, shift) axis to its multiple.
        "pad": int(stream.arrays.gemm_window.shape[0] - window.size) if stream.static.cuda_kernels else 0,
        "shift_pad": -(int(n_particles) * T) % _SHIFT_ALIGN if stream.static.cuda_kernels else 0,
    }
    return window, constants, static


def tilt_operand_bytes(stream, n_particles: int) -> tuple[int, int]:
    """Device bytes of one tilt tile's operand program for ``n_particles``, from XLA's compiled memory analysis.

    The tile holds particles of one tilt group; the largest group's frame count sizes it. Returns
    ``(peak, resident)``: arguments plus outputs plus temporaries, and the outputs alone (they stay live with
    the tile). The tile planner reads it through ``load_tilt_tile.operand_bytes``; the program compiled at
    these shapes is the one a tile of this size reuses.
    """
    n_frames = max(len(frames) for frames in stream.dataset.group_frames)
    window, _, static = _operand_layout(stream, n_particles, n_frames)
    n_half = int(np.asarray(stream.noise_variance_half).size)
    return _operand_bytes(
        n_half,
        int(window.size),
        str(np.asarray(window).dtype),
        int(stream.translations.shape[0]),
        tuple(sorted(static.items())),
        stream.device,
    )


@functools.lru_cache(maxsize=64)
def _operand_bytes(n_half, window_size, window_dtype, n_translations, static, device):
    static = dict(static)
    images = static["n_particles"] * static["n_frames"]
    f32 = jnp.float32

    def spec(shape, dtype=f32):
        return jax.ShapeDtypeStruct(shape, dtype)

    arguments = (
        spec((images, n_half), jnp.complex64),  # half spectra
        spec((images, n_half)),  # CTF
        spec((images,), jnp.bool_),  # visible
        spec((n_half,)),  # noise
        spec((window_size,), window_dtype),
        spec((window_size, 2)),  # lattice
        spec((static["n_frames"], n_translations, 2)),  # frame shifts
        spec((window_size,)),  # score mask
        spec((window_size,)),  # reconstruction mask
        spec((n_half,)),  # half-spectrum weights
    )
    with jax.default_device(device):
        analysis = _tilt_operands.lower(*arguments, **static).compile().memory_analysis()
    if analysis is None:
        raise RuntimeError("XLA reports no memory analysis for the tilt operand program on this platform")
    peak = analysis.argument_size_in_bytes + analysis.output_size_in_bytes + analysis.temp_size_in_bytes
    return int(peak), int(analysis.output_size_in_bytes)


@partial(jax.jit, static_argnames=("n_particles", "n_frames", "pad", "shift_pad"))
def _tilt_operands(
    half,
    ctf,
    visible,
    nv,
    window,
    lattice,
    frame_shifts,
    score_mask,
    recon_mask,
    weights,
    *,
    n_particles,
    n_frames,
    pad,
    shift_pad,
):
    """A tilt tile's GEMM operands from its slot-ordered images, as one fused program (section 16.4).

    ``half`` and ``ctf`` are ``(B K, n_half)`` in (particle, frame) order with ``visible`` marking the
    frames each particle sees. Each frame's images are phase-shifted by the frame's projection of every
    3D shift (``frame_shifts`` ``(K, T, 2)``). Returns ``Y1 (K 2F', B T')``, ``ctf2 (K F', B)``,
    ``Y1_recon (B T', K 2F')``, ``ctf2_recon (B, K F')``, the particles' image energies and the summed
    observed power, with ``F' = F + pad`` and ``B T' = B T + shift_pad``.
    """
    B, K = n_particles, n_frames
    T = frame_shifts.shape[1]
    keep = visible.reshape(B, K, 1)
    half = jnp.where(keep, half.reshape(B, K, -1), 0)
    ctf = jnp.where(keep, ctf.reshape(B, K, -1), 0)
    phases = jnp.exp(jnp.complex64(-2j * np.pi) * jnp.einsum("ktd,fd->ktf", frame_shifts, lattice, precision="highest"))
    weighted = (half * ctf / nv)[..., window]  # (B, K, F)
    ctf2 = (ctf * ctf / nv)[..., window]

    def padded(values):
        return jnp.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, pad)])

    Y_score = padded(weighted[:, :, None, :] * score_mask * phases[None])  # (B, K, T, F')
    Y_recon = padded(weighted[:, :, None, :] * recon_mask * phases[None])
    Y1 = jnp.pad(jnp.transpose(_real_imag(Y_score), (1, 3, 0, 2)).reshape(-1, B * T), ((0, 0), (0, shift_pad)))
    Y1_recon = jnp.pad(jnp.transpose(_real_imag(Y_recon), (0, 2, 1, 3)).reshape(B * T, -1), ((0, shift_pad), (0, 0)))
    ctf2_score = jnp.transpose(padded(ctf2 * score_mask), (1, 2, 0)).reshape(-1, B)
    ctf2_recon = padded(ctf2 * recon_mask).reshape(B, -1)
    y_norm = jnp.sum(jnp.abs(half) ** 2 / nv * weights, axis=(1, 2))
    observation = jnp.sum(jnp.abs(half) ** 2, axis=(0, 1))
    return Y1, ctf2_score, Y1_recon, ctf2_recon, y_norm, observation


def tilt_bootstrap(particles: TiltParticles, rng, channels: int, radius: int, *, image_target: int = 1000):
    """Random-angle round-robin reconstructions of the PPCA seed maps from tilt images.

    Random particles are read until about ``image_target`` tilt images; each particle gets one
    random rotation and every visible tilt backprojects at ``Aproj_k R`` with its own CTF, into
    seed map ``position % channels``. Returns the per-map RHS and CTF^2 half volumes and the
    chosen particles.
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
    _, half, ctf = particles.read(images)
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
    return rhs, lhs, chosen


def tilt_noise_images(particles: TiltParticles, rng, *, images_per_group: int = 1000):
    """``(noise group, real-space image)`` pairs for the start-up noise of every noise group.

    Each group reads the tilt images of its particles in a random order until ``images_per_group``
    images (RELION's per-group start-up count of 1000 images), so every group gets a spectrum.
    """
    pairs = []
    for group in range(particles.n_noise_groups):
        members = rng.permutation(np.flatnonzero(particles.particle_noise_group == group))
        counts = np.diff(particles.image_offsets)[members]
        chosen = members[: int(np.searchsorted(np.cumsum(counts), images_per_group)) + 1]
        raw, _, _ = particles.read(particles.particle_images(chosen)[0])
        pairs.extend((group, np.asarray(image, np.float32)) for image in raw)
    return pairs


@full_float32
def initialize_tilts(particles: TiltParticles, *, seed, diameter_ang, q=2):
    """:func:`relax.ppca_initial_model.initialization.initialize` for subtomogram particles (section 16.5).

    The noise is one spectrum per noise group, ``(G, S)``.
    """
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q <= 0:
        raise ValueError("q must be a positive integer")
    rng = np.random.default_rng(seed)
    radius = max(1, compute_ini_high_shell(particles.grid_size))
    rhs, lhs, chosen = tilt_bootstrap(particles, rng, q + 1, radius)
    theta = seed_model(rhs, lhs, particles.volume_shape, particles.voxel_size, diameter_ang, radius, rng)
    images_for_noise = tilt_noise_images(particles, rng)
    noise = initial_noise(
        images_for_noise, particles.image_shape, particles.voxel_size, diameter_ang, n_groups=particles.n_noise_groups
    )
    info = {
        "seed": seed,
        "bootstrap_particles": chosen.tolist(),
        "noise_tilt_images": len(images_for_noise),
        "noise_groups": particles.n_noise_groups,
        "bootstrap_radius": radius,
        "projection": "complex64",
        "accumulation": "complex64/float32",
        "noise_initializer": "deliberate host float64; coefficient spectrum float32",
    }
    return theta, noise, info


# The tile planner reads the reader's operand memory from the reader (FullRowStream.tile_loader).
load_tilt_tile.operand_bytes = tilt_operand_bytes


def tilt_frames(stream) -> int:
    """Projection frames per tile row for the tile planner: the largest tilt group's frame count."""
    return max(len(frames) for frames in stream.dataset.group_frames)


load_tilt_tile.max_frames = tilt_frames


def tilt_shift_phases(stream, layout, translations):
    """Phase factors ``(T, K, F)`` of 3D shifts on a tile's frames (``layout["frames"]``) over the score window:
    the factors :func:`load_tilt_tile`'s operands carry for those shifts (:func:`_tilt_operands`)."""
    frames = layout["frames"]
    _, constants, _ = _operand_layout(stream, 1, len(frames))
    shifts = jnp.asarray(tilt_shifts(frames, translations))
    arguments = jnp.einsum("ktd,fd->ktf", shifts, constants["lattice"], precision="highest")
    return jnp.transpose(jnp.exp(jnp.complex64(-2j * np.pi) * arguments), (1, 0, 2))


# Adaptive oversampling forms each job's child-shifted operands from these factors.
load_tilt_tile.shift_phases = tilt_shift_phases
