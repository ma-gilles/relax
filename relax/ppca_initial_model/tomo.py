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
from relax.fourier.preprocessing import relion_half_translation_lattice, translation_phases
from relax.ppca_initial_model.initialization import initial_noise, seed_model
from relax.ppca_refinement.full_row_kernels import SHIFT_ALIGN, TileArrays, real_imag
from relax.ppca_refinement.full_row_stream import tile_size_bucket, tile_support
from relax.ppca_refinement.residual_statistics import full_float32
from relax.reconstruction.resolution import compute_ini_high_shell
from relax.sampling.tomo_particles import tilt_image_shifts

# Tilt groups (one tomogram, one noise group) with at most this many particles of a batch share mixed tiles
# (:func:`tilt_tiles`); larger groups keep tiles of their own, whose projections all their particles share.
MIXED_GROUP_MAX = 8


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
    """:class:`TiltParticles` of a :class:`relax.steps.tomo_half.TomoDataset`.

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


def _tile_keys(particles: TiltParticles, ids) -> np.ndarray:
    """Each particle's (tilt group, noise group) key: the particles of one key share frames and noise."""
    return particles.particle_group[ids] * particles.n_noise_groups + particles.particle_noise_group[ids]


def _slot_bucket(n: int) -> int:
    """Particle slots per tilt group of a mixed tile: ``n`` rounded up to a power of two."""
    return 1 << max(0, int(n) - 1).bit_length()


def tilt_tiles(particles: TiltParticles, ids, tile_size: int, *, mixed: bool = True) -> list[np.ndarray]:
    """Cut ``ids`` into tiles of at most ``tile_size`` particles.

    A key (tilt group and noise group) with more than :data:`MIXED_GROUP_MAX` particles (or ``tile_size``) of
    ``ids`` is cut into tiles of its own, in order of its first particle in ``ids``. With ``mixed`` the smaller
    keys share mixed tiles (:func:`load_tilt_tile`): largest first, each tile as many keys as fit
    ``tile_size`` at the slots its largest key takes (:func:`_slot_bucket`), so a batch spread over hundreds of
    tomograms makes a few tiles, not hundreds. Without ``mixed`` (adaptive oversampling) every key keeps tiles of
    its own. Particles keep their order within a key.
    """
    ids = np.asarray(ids, dtype=np.int64)
    groups = _tile_keys(particles, ids)
    _, first = np.unique(groups, return_index=True)
    members = [ids[groups == group] for group in groups[np.sort(first)]]
    limit = min(MIXED_GROUP_MAX, int(tile_size)) if mixed else 0
    tiles = []
    for group in members:
        if group.size > limit:
            tiles.extend(group[start : start + tile_size] for start in range(0, group.size, tile_size))
    small = sorted((group for group in members if group.size <= limit), key=lambda group: -group.size)
    while small:
        take = max(1, int(tile_size) // _slot_bucket(small[0].size))
        tiles.append(np.concatenate(small[:take]))
        small = small[take:]
    return tiles


def tilt_shifts(frames, translations) -> np.ndarray:
    """2D shift ``[Aproj t]_{1,2}`` of each frame for each 3D shift, ``(K, T, 2)`` float32.

    RELION's ``Experiment::getTranslationInTiltSeries`` in its summation order
    (:func:`relax.sampling.tomo_particles.tilt_image_shifts`, in float64), with no old offset: PPCA's
    translations are absolute.
    """
    n_frames = len(frames)
    return tilt_image_shifts(translations, np.zeros((1, 3)), frames, np.zeros(n_frames, np.int64)).astype(np.float32)


def _noise_table(stream):
    """The stream's noise spectra as rows ``(G, n_half)`` and whether it is a per-group table."""
    nv = stream.noise_variance_half
    return (nv, True) if nv.ndim == 2 else (nv[None], False)


def _per_noise_group(stream, values, groups):
    """``values`` (rows of ``groups``) summed per noise group of a stream noise table, else over all rows."""
    table, per_group = _noise_table(stream)
    if not per_group:
        return jnp.sum(values, axis=0)
    return jnp.zeros((table.shape[0], *values.shape[1:]), values.dtype).at[jnp.asarray(groups)].add(values)


def _observation_counts(stream, image_noise_groups):
    """Tilt images per noise group of a stream noise table, else their number."""
    table, per_group = _noise_table(stream)
    if not per_group:
        return int(image_noise_groups.size)
    return np.bincount(image_noise_groups, minlength=table.shape[0]).astype(np.int64)


def load_tilt_tile(stream, image_indices, significant_rows, *, collect_observation: bool):
    """:class:`FullRowStream` tile reader for subtomogram particles.

    A tile of one key (tilt group and noise group) is read here; a tile of several keys is a mixed tile
    (:func:`_load_mixed_tilt_tile`).

    Operands follow section 16.4: frame-major ``(frame, [Re | Im], pixel)`` GEMM axes, each tilt's
    image phase-shifted by its own projection of every 3D shift, and zeros at frames a particle
    does not see. ``n_observations`` counts the tile's tilt images (one noise observation each).
    With a planned tile size on the stream, the tile is padded to its :func:`tile_size_bucket` with
    particles that see no frame (zero operands) after the real ones (``n_real``), so a stage compiles
    one operand program and one stream program per bucket.
    """
    particles = stream.dataset
    ids = np.asarray(image_indices, dtype=np.int64)
    keys = _tile_keys(particles, ids)
    if np.unique(keys).size != 1:
        return _load_mixed_tilt_tile(stream, ids, keys, significant_rows, collect_observation=collect_observation)
    frames = particles.group_frames[int(particles.particle_group[ids[0]])]
    noise_group = int(particles.particle_noise_group[ids[0]])
    table, _ = _noise_table(stream)
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
        table[noise_group],
        jnp.asarray(window),
        constants["lattice"],
        jnp.asarray(tilt_shifts(frames, stream.translations)),
        constants["score_mask"],
        constants["recon_mask"],
        constants["weights"],
        **static,
    )
    coarse_mask, table, layout = tile_support(stream, significant_rows)
    tile = TileArrays(
        # Padding particles take the first particle's support; the stream gives them no posterior mass.
        coarse_mask=jnp.concatenate([coarse_mask, jnp.repeat(coarse_mask[:1], padded - B, axis=0)]),
        rows=table,
        Y1=Y1,
        ctf2=ctf2,
        Y1_recon=Y1_recon,
        ctf2_recon=ctf2_recon,
        y_norm=y_norm,
        frames=jnp.asarray(frames, jnp.float32),
        noise_groups=jnp.asarray([noise_group], jnp.int32),
    )
    if stream.static.cuda_kernels and tile.Y1.shape[0] != K * 2 * stream.arrays.gemm_window.shape[0]:
        # The CUDA projector writes each frame's rows padded to the GEMM window, frame-major.
        raise RuntimeError("Tilt tile operands do not follow the frame-major GEMM window layout")
    observation = _per_noise_group(stream, observation[None], [noise_group]) if collect_observation else None
    layout.update(
        n_real=B,
        n_observations=_observation_counts(stream, np.full(images.size, noise_group, np.int64)),
        original_ids=particles.original_image_indices_from_local(ids),
        frames=frames,
    )
    return tile, observation, layout


def _load_mixed_tilt_tile(stream, ids, keys, significant_rows, *, collect_observation: bool):
    """A tile of several keys (tilt groups with their noise groups): RELION's per-particle tilt matrices.

    Each key ``g`` holds ``b`` particle slots (:func:`_slot_bucket` of its largest key) and the frames of its tilt
    group, padded with identity frames no particle sees to the stream's largest group (:func:`tilt_frames`); the
    key count is padded to a :func:`tile_size_bucket` of the tile's slot capacity with keys of padding particles,
    so a stage compiles one program per (slots, keys) bucket. Operands are per key, group-major
    (:class:`TileArrays`); particles keep the tile's order, real ones first (``slots`` maps them). The pass-1 and
    pass-2 kernels run each key at its own frames (``full_row_kernels._score_block_mixed``,
    ``_moment_block_mixed``) and add each key's residual to its noise group.
    """
    particles = stream.dataset
    _, first = np.unique(keys, return_index=True)
    key_order = keys[np.sort(first)]
    lookup = {int(key): g for g, key in enumerate(key_order)}
    position_group = np.asarray([lookup[int(key)] for key in keys], np.int64)
    G = key_order.size
    counts = np.bincount(position_group, minlength=G)
    b = _slot_bucket(int(counts.max()))
    capacity = G if stream.tile_images is None else max(G, int(stream.tile_images) // b)
    G_pad = tile_size_bucket(G, capacity)
    B, n_real = G_pad * b, int(ids.size)
    rank = np.zeros(n_real, np.int64)
    seen = np.zeros(G, np.int64)
    for position, group in enumerate(position_group):
        rank[position] = seen[group]
        seen[group] += 1
    position_slot = np.empty(B, np.int64)
    position_slot[:n_real] = position_group * b + rank
    used = np.zeros(B, bool)
    used[position_slot[:n_real]] = True
    position_slot[n_real:] = np.flatnonzero(~used)  # padding particles take the free slots in order
    slot_position = np.argsort(position_slot)

    K = tilt_frames(stream)
    T = int(stream.translations.shape[0])
    frames = np.tile(np.eye(3), (G_pad, K, 1, 1))
    noise_groups = np.zeros(G_pad, np.int64)
    for g, key in enumerate(key_order):
        tilt_group, noise_group = divmod(int(key), particles.n_noise_groups)
        group_frames = particles.group_frames[tilt_group]
        frames[g, : len(group_frames)] = group_frames
        noise_groups[g] = noise_group
    images, owner = particles.particle_images(ids)
    slot_image = np.full(B * K, -1, np.int64)
    slot_image[position_slot[owner] * K + particles.image_frame[images]] = images
    visible = slot_image >= 0
    _, half, ctf = particles.read(np.where(visible, slot_image, images[0]))
    half, ctf = jax.device_put((half, ctf), stream.device)
    table, _ = _noise_table(stream)
    window, constants, static = _operand_layout(stream, b, K)
    # GPU streams pad each key's (image, shift) axis to SHIFT_ALIGN, so its GEMMs run cuBLAS's aligned kernels.
    shifts = np.stack([tilt_shifts(frames[g], stream.translations) for g in range(G_pad)])  # (G, K, T, 2)
    Y1, ctf2, Y1_recon, ctf2_recon, y_norm, observation = _mixed_tilt_operands(
        half,
        ctf,
        jnp.asarray(visible),
        table[jnp.asarray(noise_groups)],
        jnp.asarray(window),
        constants["lattice"],
        jnp.asarray(shifts),
        constants["score_mask"],
        constants["recon_mask"],
        constants["weights"],
        **static,
    )
    coarse_mask, rows, layout = tile_support(stream, significant_rows)
    tile = TileArrays(
        coarse_mask=jnp.concatenate([coarse_mask, jnp.repeat(coarse_mask[:1], B - n_real, axis=0)]),
        rows=rows,
        Y1=Y1,
        ctf2=ctf2,
        Y1_recon=Y1_recon,
        ctf2_recon=ctf2_recon,
        y_norm=y_norm.reshape(-1)[jnp.asarray(position_slot)],
        frames=jnp.asarray(frames, jnp.float32),
        noise_groups=jnp.asarray(noise_groups, jnp.int32),
        slots=jnp.asarray(np.stack([position_slot, slot_position]), jnp.int32),
    )
    if stream.static.cuda_kernels and tile.Y1.shape[1] != K * 2 * stream.arrays.gemm_window.shape[0]:
        raise RuntimeError("Tilt tile operands do not follow the frame-major GEMM window layout")
    image_noise = noise_groups[position_group[owner]]
    observation = _per_noise_group(stream, observation, noise_groups) if collect_observation else None
    layout.update(
        n_real=n_real,
        n_observations=_observation_counts(stream, image_noise),
        original_ids=particles.original_image_indices_from_local(ids),
        frames=None,  # adaptive oversampling reads one tilt group's frames: it runs on one-key tiles only
        n_keys=G,
    )
    return tile, observation, layout


# The operand constants of the stage's resolved band, built once instead of once per tile (a real-data batch spans
# hundreds of tomograms, so hundreds of tiles). Entries hold their ``resolved`` object, so an ``is`` match is exact.
_LAYOUT_CONSTANTS: list = []


def _layout_constants(stream, n_half: int):
    """``(window, constants)`` of :func:`_operand_layout` for the stream's resolved band, cached per band."""
    resolved, image_shape = stream.resolved, tuple(int(n) for n in stream.static.image_shape)
    for entry in _LAYOUT_CONSTANTS:
        if entry[0] is resolved and entry[1] == (image_shape, n_half):
            return entry[2]
    window = np.arange(n_half) if resolved.score_indices is None else np.asarray(resolved.score_indices)
    constants = {
        "lattice": jnp.asarray(np.asarray(relion_half_translation_lattice(image_shape), np.float32)[window]),
        "score_mask": resolved.score_mask[window],
        "recon_mask": resolved.recon_mask[window],
        "weights": make_half_image_weights(image_shape),
    }
    _LAYOUT_CONSTANTS.insert(0, (resolved, (image_shape, n_half), (window, constants)))
    del _LAYOUT_CONSTANTS[4:]
    return window, constants


def _operand_layout(stream, n_particles: int, n_frames: int):
    """The operand program's score window, per-pixel constants and static shape arguments for a tile size."""
    resolved = stream.resolved
    n_half = int(stream.noise_variance_half.shape[-1])
    T = int(stream.translations.shape[0])
    window, constants = _layout_constants(stream, n_half)
    static = {
        "n_particles": int(n_particles),
        "n_frames": int(n_frames),
        # GPU streams pad each frame's pixels to the GEMM window and the (image, shift) axis to its multiple.
        "pad": int(stream.arrays.gemm_window.shape[0] - window.size) if stream.static.cuda_kernels else 0,
        "shift_pad": -(int(n_particles) * T) % SHIFT_ALIGN if stream.static.cuda_kernels else 0,
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
    n_half = int(stream.noise_variance_half.shape[-1])
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


def _tilt_operands_body(
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
    phases = translation_phases(frame_shifts, lattice)  # (K, T, F)
    weighted = (half * ctf / nv)[..., window]  # (B, K, F)
    ctf2 = (ctf * ctf / nv)[..., window]

    def padded(values):
        return jnp.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, pad)])

    Y_score = padded(weighted[:, :, None, :] * score_mask * phases[None])  # (B, K, T, F')
    Y_recon = padded(weighted[:, :, None, :] * recon_mask * phases[None])
    Y1 = jnp.pad(jnp.transpose(real_imag(Y_score), (1, 3, 0, 2)).reshape(-1, B * T), ((0, 0), (0, shift_pad)))
    Y1_recon = jnp.pad(jnp.transpose(real_imag(Y_recon), (0, 2, 1, 3)).reshape(B * T, -1), ((0, shift_pad), (0, 0)))
    ctf2_score = jnp.transpose(padded(ctf2 * score_mask), (1, 2, 0)).reshape(-1, B)
    ctf2_recon = padded(ctf2 * recon_mask).reshape(B, -1)
    y_norm = jnp.sum(jnp.abs(half) ** 2 / nv * weights, axis=(1, 2))
    observation = jnp.sum(jnp.abs(half) ** 2, axis=(0, 1))
    return Y1, ctf2_score, Y1_recon, ctf2_recon, y_norm, observation


_tilt_operands = partial(jax.jit, static_argnames=("n_particles", "n_frames", "pad", "shift_pad"))(_tilt_operands_body)


@partial(jax.jit, static_argnames=("n_particles", "n_frames", "pad", "shift_pad"))
def _mixed_tilt_operands(
    half,
    ctf,
    visible,
    nv_rows,
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
    """:func:`_tilt_operands` per key of a mixed tile, one key at a time (its own noise row and frame shifts
    ``(G, K, T, 2)``): every output gains a leading key axis."""
    G = nv_rows.shape[0]
    per_key = n_particles * n_frames
    static = dict(n_particles=n_particles, n_frames=n_frames, pad=pad, shift_pad=shift_pad)

    def one(args):
        h, c, v, nv, shifts = args
        return _tilt_operands_body(h, c, v, nv, window, lattice, shifts, score_mask, recon_mask, weights, **static)

    return jax.lax.map(
        one,
        (half.reshape(G, per_key, -1), ctf.reshape(G, per_key, -1), visible.reshape(G, per_key), nv_rows, frame_shifts),
    )


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
    return jnp.transpose(translation_phases(shifts, constants["lattice"]), (1, 0, 2))


# Adaptive oversampling forms each job's child-shifted operands from these factors.
load_tilt_tile.shift_phases = tilt_shift_phases
