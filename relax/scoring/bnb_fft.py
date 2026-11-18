"""Level 0 of the branch-and-bound pose search as an FFT cross-correlation over the whole box (``--bnb_level0 fft``).

The search's Gaussian score of rotation ``r`` and shift ``t`` is ``X - A/2 - C/2`` (:mod:`relax.scoring.bnb_search`),
and only ``X(t) = Re sum_k G_r(k) exp(2 pi i k.t / N)``, ``G_r = w conj(y) p_r``, depends on the shift: RELION's
coarse operands shift an image by the phase ``exp(-2 pi i k.t / N)`` of each scored row (its whole-frequency label
``k``, RELION's coarse row relabelling included). One inverse real FFT of ``G_r`` on an ``M x M`` grid
(``M = N / step``) gives ``X`` at every circular shift of the box on the lattice of ``step`` pixels, so the level
covers the whole box at a cost that does not depend on the offset prior. Its offset prior over the lattice is flat or
the model's Gaussian at the absolute offsets.

Each image is first shifted to its origin, its rounded previous offset (zero fill), and masked there, as RELION's
old-offset pre-shift does: the lattice is the circular shifts around the origin, and a particle that the mask would
cut at the box centre comes into view as its offset is found.

Each image keeps rotations by its whole-box translation-marginal posterior, as level 0 does, and reports its
translation posterior on a disc of the level's translations around its frame: the lattice point whose window
(``centre_radius``) holds the most posterior mass. The finer levels and the window rule work relative to the frame
as they do relative to a level's pre-shift.
"""

from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# Elements of one block's cross-correlation ``[B, R, M, M]``: the rotations per device call.
_FFT_ELEMENTS = 1 << 27
# The frame prefers the posterior's mode among windows holding the same mass (a plateau around a sharp peak).
_MODE_PREFERENCE = 1e-3


def fft_step(box: int, size: int, max_step_px: float) -> int:
    """The FFT level's lattice step, whole pixels: RELION's coarse window ``size`` sampled twice per Nyquist step
    (``box / (2 size)``) rounded down, at most ``max_step_px``, at least 1, and dividing ``box``."""

    step = int(np.clip(np.floor(box / (2.0 * size)), 1, max(1.0, np.floor(max_step_px))))
    while box % step:
        step -= 1
    return step


def frequency_labels(operand_plan) -> np.ndarray:
    """Whole-frequency labels ``[S, 2]`` of every scored row, paired with the shift's ``(x, y)``, read off the operand
    route's own translate kernel (RELION's coarse row relabelling included): the phase of a unit shift per axis."""

    from relax.cuda import translate as cuda_translate
    from relax.fine_pass.bucket_io import relion_translation_angles_f32
    from relax.scoring.coarse_operands import relion_coarse_translate

    shape = operand_plan.image_shape
    n = int(shape[0])
    scale = float(operand_plan.relion_translation_angle_scale)
    if abs(scale - 1.0) > 1e-6:
        raise ValueError(f"--bnb_level0 fft needs the model and optics pixel sizes equal (angle scale {scale})")
    gaussian = operand_plan.gaussian_plan
    ones = jnp.ones((1, int(gaussian.score_indices.shape[0])), jnp.complex64)
    angles = jnp.asarray(relion_translation_angles_f32(np.array([[1.0, 0.0], [0.0, 1.0]]), shape))
    shifted = relion_coarse_translate(
        cuda_translate.relion_translate_score_f32, ones, angles, gaussian.score_indices, gaussian.score_indices_np,
        shape, window=operand_plan.coarse_kernel_window, r_max=operand_plan.coarse_kernel_r_max,
    )
    labels = -np.angle(np.asarray(shifted)[0]) * n / (2.0 * np.pi)  # [2, S]: the phase is -2 pi k / n
    rounded = np.rint(labels)
    if np.max(np.abs(labels - rounded), initial=0.0) > 1e-3:
        raise ValueError("the translate kernel's phases are not whole frequencies")
    return rounded.T.astype(np.int64)


def fft_tables(labels: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Gather tables ``[M (M/2 + 1)]`` (``-1``: empty) of an inverse real FFT whose output is ``Re sum_s G_s e``.

    The half grid's columns run over the labels' non-negative axis ``half`` (returned), its rows over the other.
    ``first`` places each row's ``G / 2`` at its cell; ``second`` places ``conj(G) / 2`` of the zero-column rows at
    their mirror cell, which makes that column Hermitian as the transform assumes.
    """

    half = 0 if np.all(labels[:, 0] >= 0) else 1
    k_half, k_full = labels[:, half], labels[:, 1 - half]
    if np.any(k_half < 0):
        raise ValueError("the scored rows have no non-negative frequency axis")
    if np.any(k_half >= m // 2) or np.any(np.abs(k_full) >= m // 2):
        raise ValueError(f"scored frequencies beyond the FFT grid of {m}")
    width = m // 2 + 1
    cells = (k_full % m) * width + k_half
    zero = np.nonzero(k_half == 0)[0]
    mirror = ((-k_full[zero]) % m) * width
    if np.unique(cells).size != cells.size or np.unique(mirror).size != mirror.size:
        raise ValueError("two scored rows share a frequency")
    first = np.full(m * width, -1, np.int64)
    second = np.full(m * width, -1, np.int64)
    first[cells] = np.arange(labels.shape[0])
    second[mirror] = zero
    return first, second, half


def fft_lattice(m: int, step: float, half: int) -> np.ndarray:
    """The translations ``[M * M, 2]`` (pixels) of the FFT output in its flat order (rows: the full axis, columns: the
    ``half`` axis), each circular shift once, signed in ``[-M/2, M/2)`` steps."""

    j = np.arange(m)
    j = np.where(j < m // 2, j, j - m).astype(np.float64)
    full, part = np.meshgrid(j, j, indexing="ij")
    out = np.empty((m * m, 2))
    out[:, half] = part.ravel()
    out[:, 1 - half] = full.ravel()
    return out * float(step)


def disc_kernel_hat(m: int, step: float, radius: float):
    """The rFFT ``[M, M/2 + 1]`` of the lattice's disc of ``radius`` pixels (a window), for circular sums."""

    j = np.arange(m)
    j = np.where(j < m // 2, j, j - m).astype(np.float64) * float(step)
    a, b = np.meshgrid(j, j, indexing="ij")
    disc = (a**2 + b**2 <= float(radius) ** 2 + 1e-9).astype(np.float32)
    return jnp.fft.rfft2(jnp.asarray(disc))


@partial(jax.jit, static_argnames=("m",))
def _fft_block(wy, image_term, weight, projections, rotation_prior, lattice_prior, first, second, *, m: int):
    """Scores of a batch against a block of projections ``[R, S]`` over the whole lattice, reduced: per-rotation log
    mass ``[B, R]``, per-translation log mass ``[B, M * M]``, the best score ``[B]`` and its flat ``(r, t)`` index.

    ``wy`` ``[B, S]`` is ``w y``, ``image_term`` ``[B]`` the pose-free score ``-C/2 - d0``, ``lattice_prior``
    ``[1 or B, M * M]`` the offset prior.
    """

    g = jnp.conj(wy)[:, None, :] * projections[None, :, :]
    padded = jnp.concatenate([g, jnp.zeros(g.shape[:2] + (1,), g.dtype)], axis=-1)
    empty = g.shape[-1]
    grid = 0.5 * jnp.take(padded, jnp.where(first >= 0, first, empty), axis=-1)
    grid = grid + 0.5 * jnp.conj(jnp.take(padded, jnp.where(second >= 0, second, empty), axis=-1))
    grid = grid.reshape(g.shape[:2] + (m, m // 2 + 1))
    cross = jnp.fft.irfft2(grid, s=(m, m)).reshape(g.shape[:2] + (m * m,)) * float(m * m)
    model_energy = jnp.einsum("bs,rs->br", weight, jnp.abs(projections) ** 2, precision=jax.lax.Precision.HIGHEST)
    scores = (
        cross
        - 0.5 * model_energy[:, :, None]
        + image_term[:, None, None]
        + lattice_prior[:, None, :]
        + rotation_prior[None, :, None]
    )
    flat = scores.reshape(scores.shape[0], -1)
    return (
        jax.nn.logsumexp(scores, axis=2),
        jax.nn.logsumexp(scores, axis=1),
        jnp.max(flat, axis=1),
        jnp.argmax(flat, axis=1).astype(jnp.int32),
    )


@partial(jax.jit, static_argnames=("m",))
def _frames(translation_mass, kernel_hat, *, m: int):
    """Each image's frame ``[B]`` (flat lattice index): the cell whose window (``kernel_hat``) holds the most of its
    translation posterior ``[B, M * M]``, the mode first among windows holding the same mass."""

    top = jnp.max(translation_mass, axis=1, keepdims=True)
    weights = jnp.exp(translation_mass - top)
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    held = jnp.fft.irfft2(jnp.fft.rfft2(weights.reshape(-1, m, m)) * kernel_hat[None], s=(m, m))
    return jnp.argmax(held.reshape(weights.shape) + _MODE_PREFERENCE * weights, axis=1).astype(jnp.int32)


@partial(jax.jit, static_argnames=("m",))
def _local_masses(translation_mass, frames, offset_rows, offset_columns, *, m: int):
    """The translation posterior ``[B, T]`` at each frame plus the lattice offsets ``(rows, columns)`` ``[T]``,
    wrapped around the box."""

    rows = (frames[:, None] // m + offset_rows[None, :]) % m
    columns = (frames[:, None] % m + offset_columns[None, :]) % m
    return jnp.take_along_axis(translation_mass, rows * m + columns, axis=1)


def lattice_log_prior(lattice, origins, kind: str, *, pixel_size: float, sigma_offset_angstrom: float):
    """The offset prior ``[1 or B, M * M]`` over the ``lattice`` ``[M * M, 2]`` around each image's origin ``[B, 2]``
    (pixels): ``"flat"`` (zero) or ``"model"`` (the model's Gaussian at the absolute offset, as
    :func:`relax.scoring.bnb_search.offset_log_prior`)."""

    if kind == "flat":
        return jnp.zeros((1, lattice.shape[0]), jnp.float32)
    if kind == "model":
        absolute = (jnp.asarray(origins, jnp.float32)[:, None, :] + jnp.asarray(lattice, jnp.float32)[None]) * float(
            pixel_size
        )
        return -0.5 * jnp.sum(absolute**2, axis=2) / float(sigma_offset_angstrom) ** 2
    raise ValueError(f"unknown FFT level prior {kind!r}")


def level_pass(
    prepared,
    dataset,
    *,
    local: np.ndarray,
    origins: np.ndarray,
    pixel_size: float,
    sigma_offset_angstrom: float,
    pruning,
    coarse: bool,
    centre_radius: float,
    image_batch_size: int,
    clock: dict,
):
    """Level 0 over every image by FFT: each image's kept rotations, kept count, window centre relative to its frame,
    offset spread and hold radius (on the level's disc around the frame), best rotation and translation (relative to
    the frame), the log posterior mass ``[N, T]`` of the disc's translations, and the frames ``[N, 2]`` (pixels).

    ``prepared`` is the level planned with one zero translation and the images pre-shifted to their ``origins``
    ``[N, 2]`` (whole pixels), with ``level.step`` the lattice step and ``level.fft_prior`` the lattice prior; frames,
    centres and best translations are relative to the origins. ``local`` ``[T, 2]`` (pixels, multiples of the step)
    is the disc around the frame. Coarse levels keep at most ``pruning.max_kept`` rotations per image.
    """

    from relax.io.batch_fetch import iter_indexed_batches
    from relax.scoring.bnb_search import _keep, _level_operands, centre_table, window_centres

    level = prepared.level
    n = int(prepared.operand_plan.image_shape[0])
    step = float(level.step)
    m = int(round(n / step))
    if abs(m * step - n) > 1e-6:
        raise ValueError(f"the FFT lattice step {step} does not divide the box {n}")
    tick = time.time()
    labels = frequency_labels(prepared.operand_plan)
    first, second, half = fft_tables(labels, m)
    first, second = jnp.asarray(first, jnp.int32), jnp.asarray(second, jnp.int32)
    lattice = fft_lattice(m, step, half)
    origins = np.asarray(origins, dtype=np.float64)
    kernel_hat = disc_kernel_hat(m, step, centre_radius)
    local = np.asarray(local, dtype=np.float64)
    offsets = np.rint(local / step).astype(np.int32)
    offset_rows, offset_columns = jnp.asarray(offsets[:, 1 - half]), jnp.asarray(offsets[:, half])
    table = centre_table(local, centre_radius)
    n_rot = int(prepared.projections.shape[0])
    block = max(8, (_FFT_ELEMENTS // max(int(image_batch_size) * m * m, 1)) // 8 * 8)
    block = min(block, -(-n_rot // 8) * 8)
    padded_rot = -(-n_rot // block) * block
    projections = jnp.pad(prepared.projections, ((0, padded_rot - n_rot), (0, 0)))
    rotation_prior = jnp.pad(prepared.rotation_prior, (0, padded_rot - n_rot), constant_values=-jnp.inf)
    clock["plan"] += time.time() - tick

    n_images = int(dataset.n_units)
    kept_rows: list[np.ndarray | None] = [None] * n_images
    kept_counts = np.zeros(n_images, dtype=np.int32)
    centres = np.zeros((n_images, 2), dtype=np.float64)
    spread = np.zeros(n_images, dtype=np.float64)
    hold = np.zeros(n_images, dtype=np.float64)
    best_rotation = np.zeros(n_images, dtype=np.int64)
    best_translation = np.zeros((n_images, 2), dtype=np.float64)
    masses = np.zeros((n_images, local.shape[0]), dtype=np.float32)
    frames = np.zeros((n_images, 2), dtype=np.float64)
    start = 0
    for batch_data, _, _, _, _, _, indices in iter_indexed_batches(dataset, np.arange(n_images), image_batch_size):
        end = start + len(indices)
        count = end - start
        tick = time.time()
        weighted, image_term, weight, batch_size = _level_operands(prepared, batch_data, indices, start, end)
        n_pix = int(weight.shape[1])
        wy = weighted[:, 0, :n_pix] + 1j * weighted[:, 0, n_pix:]
        term = image_term[:, 0]
        origin = np.zeros((batch_size, 2))
        origin[:count] = origins[start:end]
        prior = lattice_log_prior(
            lattice, origin, level.fft_prior, pixel_size=pixel_size, sigma_offset_angstrom=sigma_offset_angstrom
        )
        wy.block_until_ready()
        clock["operands"] += time.time() - tick
        tick = time.time()
        rotation_mass, translation_mass, best = [], None, None
        for r0 in range(0, padded_rot, block):
            mass, tmass, score, arg = _fft_block(
                wy, term, weight, projections[r0 : r0 + block], rotation_prior[r0 : r0 + block], prior, first,
                second, m=m,
            )
            rotation_mass.append(mass)
            translation_mass = tmass if translation_mass is None else jnp.logaddexp(translation_mass, tmass)
            rot, trans = r0 + arg // (m * m), arg % (m * m)
            if best is None:
                best = (score, rot, trans)
            else:
                better = score > best[0]
                best = tuple(jnp.where(better, new, old) for new, old in zip((score, rot, trans), best))
        rotation_mass = jnp.concatenate(rotation_mass, axis=1)[:, :n_rot]
        rotation_mass.block_until_ready()
        clock["scores"] += time.time() - tick
        tick = time.time()
        ids = jnp.broadcast_to(jnp.arange(n_rot, dtype=jnp.int32), (batch_size, n_rot))
        kept_device, n_kept = _keep(
            rotation_mass, ids, pruning.mass_fraction, pruning.margin,
            max_kept=int(pruning.max_kept) if coarse else n_rot,
            min_kept=int(pruning.min_kept) if coarse else 0,
        )
        kept = np.asarray(kept_device)[:count]
        for i, row in enumerate(kept):
            kept_rows[start + i] = np.sort(row[row >= 0]).astype(np.int32)
        kept_counts[start:end] = np.asarray(n_kept)[:count]
        clock["keep"] += time.time() - tick
        tick = time.time()
        frame = _frames(translation_mass, kernel_hat, m=m)
        local_mass = np.asarray(_local_masses(translation_mass, frame, offset_rows, offset_columns, m=m))[:count]
        frame_t = lattice[np.asarray(frame)[:count]]
        masses[start:end] = local_mass
        centres[start:end], spread[start:end], hold[start:end] = window_centres(
            local_mass, local, centre_radius, table=table
        )
        frames[start:end] = frame_t
        best_rotation[start:end] = np.asarray(best[1])[:count]
        best_translation[start:end] = lattice[np.asarray(best[2])[:count]] - frame_t
        clock["centres"] += time.time() - tick
        start = end
    return kept_rows, kept_counts, centres, spread, hold, best_rotation, best_translation, masses, frames, m
