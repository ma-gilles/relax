"""Branch-and-bound global pose search of a K=1 global-search half (``--pose_search bnb``).

RELION scores each image's translations in a window centred on its previous offset, so a particle whose offset lies
outside that window cannot reach it, and a wrong offset drags its pose with it. This search scores every translation
within a few widths of the offset prior, coarse to fine over rotations, in the approximate branch-and-bound of
cryoSPARC (Punjani et al. 2017, Supplementary Note 2):

1. Level 0 scores every rotation of a coarser HEALPix order over the whole translation grid, at the Fourier window
   that order samples adequately (RELION's coarse-size rule), where the score is smooth over a rotation cell.
2. Each image keeps the rotations that carry ``mass_fraction`` of its translation-marginal posterior and every
   rotation within ``margin`` nats of its best; the next level scores their children at the next order and window.
3. The last level is the iteration's own order. Its posterior, marginalized over rotations, gives each image's window
   centre (:func:`window_centres`); its kept rotations are the candidates of :func:`window_significance`, which replaces
   pass 1: each image's significant samples over its candidates and the translation window around the new centre.

Every score is pass 1's Gaussian score on RELION's exact coarse operands (``relax.scoring.pass1_operands``) and
projections (``relax.scoring.coarse_projector``), from a pass-1 plan of the level's window, translated over this
search's grid: ``X - A/2 - C/2`` with ``X = Re(conj(w y_t) . p)``, ``A = w . |p|^2`` and ``C = w . |y_t|^2``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# The product algorithm of the search's scores: float32, or TensorFloat-32 inputs with float32 accumulation.
_ALGORITHMS = {"float32": jax.lax.DotAlgorithmPreset.F32_F32_F32, "tf32": jax.lax.DotAlgorithmPreset.TF32_TF32_F32}
# Candidate rotations per device call: gathered per image, or a block of the batch's union scored densely.
_CANDIDATE_CHUNK = 512
_DENSE_BLOCK = 1024
# The dense union product is used unless it scores this many times more (image, rotation) pairs than the candidates.
_GATHER_RATIO = 16
# A device batch holds about this many complex coefficients of its images' translated rows (or gathered candidate
# projections), with at least the configured images and at most _MAX_IMAGE_BATCH.
_BATCH_ELEMENTS = 1 << 26
_MAX_IMAGE_BATCH = 128


@dataclass(frozen=True)
class BnBLevel:
    """One level of the search.

    ``rotations`` ``[R, 3, 3]`` are the projection matrices of the level's grid and ``rotation_log_prior`` ``[R]``
    their prior (``None``: flat); at a coarse level each entry bounds the prior of its descendants. ``children``
    ``[R_previous, max_children]`` lists, for each rotation of the previous level, this level's rotations under it
    (``-1`` pads; ``None`` at level 0). ``size`` is the level's Fourier window (a current size, even pixels).
    ``translations`` ``[T, 2]`` are the level's lattice of ``step`` pixels within the search radius, absolute offsets,
    and ``translation_log_prior`` ``[T]`` their prior; a finer level scores a disk of the same step around each
    image's centre instead (:func:`run_search`).
    """

    order: int
    rotations: np.ndarray
    rotation_log_prior: np.ndarray | None
    children: np.ndarray | None
    size: int
    step: float
    translations: np.ndarray
    translation_log_prior: np.ndarray


@dataclass(frozen=True)
class BnBPruning:
    """How a coarse level keeps rotations for the next: the smallest set carrying ``mass_fraction`` of the image's
    translation-marginal posterior, every rotation within ``margin`` nats of its best and its ``min_kept`` best, at most
    ``max_kept``. The last level keeps by the first two rules alone."""

    mass_fraction: float = 0.9999
    margin: float = 10.0
    max_kept: int = 1024
    min_kept: int = 16


@dataclass(frozen=True)
class BnBSearchResult:
    """What the search found for every image of the half.

    ``centres`` ``[N, 2]`` are the window centres in pixels (integers, RELION's rounded pre-shift).
    ``candidates`` holds each image's kept rotations of the last level (sorted int32) and ``kept_counts``
    ``[L + 1, N]`` the number kept at every level. ``best_rotation`` and ``best_translation`` are the last level's best pose,
    ``translation_spread`` ``[N]`` the posterior standard deviation (pixels) of the offset and ``hold_radius`` ``[N]``
    the radius (pixels) around the centre that holds 99% of its posterior mass. ``final_mass`` ``[N, T]`` is the last
    level's log posterior mass of its translations ``final_translations`` ``[T, 2]``, relative to each image's
    ``final_base`` ``[N, 2]``: what the window rule (``relax.scoring.bnb_half``) recentres from.
    """

    centres: np.ndarray
    candidates: list
    kept_counts: np.ndarray
    best_rotation: np.ndarray
    best_translation: np.ndarray
    translation_spread: np.ndarray
    hold_radius: np.ndarray
    final_mass: np.ndarray
    final_translations: np.ndarray
    final_base: np.ndarray
    seconds: float


def translation_grid(radius_px: float, step_px: float) -> np.ndarray:
    """The points of the square lattice of ``step_px`` within ``radius_px`` of the origin, ``[T, 2]`` float64."""

    n = int(np.floor(radius_px / step_px + 1e-9))
    axis = np.arange(-n, n + 1, dtype=np.float64) * float(step_px)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return points[np.sum(points**2, axis=1) <= float(radius_px) ** 2 + 1e-9]


def offset_log_prior(translations_px: np.ndarray, *, pixel_size: float, sigma_offset_angstrom: float) -> np.ndarray:
    """The Gaussian offset prior, centred on the origin, in one unit: ``-|t a|^2 / (2 sigma^2)`` (float32 ``[T]``)."""

    squared = np.sum(np.asarray(translations_px, dtype=np.float64) ** 2, axis=1) * float(pixel_size) ** 2
    return (-0.5 * squared / float(sigma_offset_angstrom) ** 2).astype(np.float32)


def rotation_angles_deg(parents: np.ndarray, rotations: np.ndarray, *, block: int = 4096) -> np.ndarray:
    """Angles (degrees) ``[N_rotations, N_parents]`` between every rotation and every parent."""

    parent_flat = np.asarray(parents, dtype=np.float64).reshape(-1, 9)
    rotation_flat = np.asarray(rotations, dtype=np.float64).reshape(-1, 9)
    out = np.empty((rotation_flat.shape[0], parent_flat.shape[0]), dtype=np.float32)
    for start in range(0, rotation_flat.shape[0], block):
        cosine = (rotation_flat[start : start + block] @ parent_flat.T - 1.0) / 2.0
        out[start : start + block] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return out


def children_table(parents: np.ndarray, rotations: np.ndarray, *, radius_deg: float) -> np.ndarray:
    """``[N_parents, max_children]`` rows of the rotations under each parent (``-1`` pads): every rotation within
    ``radius_deg`` of the parent, and every rotation whose nearest parent it is. Overlapping cells keep a pose near a
    cell boundary under each parent around it."""

    angles = rotation_angles_deg(parents, rotations)
    member = angles <= float(radius_deg)
    member[np.arange(angles.shape[0]), np.argmin(angles, axis=1)] = True
    counts = member.sum(axis=0)
    table = np.full((angles.shape[1], max(int(counts.max(initial=0)), 1)), -1, dtype=np.int32)
    child, parent = np.nonzero(member)
    order = np.lexsort((child, parent))
    child, parent = child[order], parent[order]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    table[parent, np.arange(parent.size) - starts[parent]] = child.astype(np.int32)
    return table


def coarse_level_prior(child_prior: np.ndarray | None, children: np.ndarray) -> np.ndarray | None:
    """A coarse level's rotation prior: each rotation's largest prior among its children (a bound for pruning)."""

    if child_prior is None:
        return None
    child_prior = np.asarray(child_prior, dtype=np.float32)
    values = np.where(children >= 0, child_prior[np.maximum(children, 0)], -np.inf)
    return np.max(values, axis=1).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------
# Device programs
# ----------------------------------------------------------------------------------------------------------------


def image_batch(minimum: int, n_translations: int, size: int) -> int:
    """Images per device batch for ``n_translations`` translations at Fourier window ``size``: the power of two that
    fills ``_BATCH_ELEMENTS`` (at most ``_MAX_IMAGE_BATCH``), at least ``minimum``. Small grids take larger batches."""

    per_image = max(int(n_translations), _CANDIDATE_CHUNK) * int(size) * (int(size) // 2 + 1)
    fit = max(_BATCH_ELEMENTS // per_image, 1)
    return max(int(minimum), min(_MAX_IMAGE_BATCH, 1 << (fit.bit_length() - 1)))


def _packed(values):
    """``[..., S]`` complex as ``[..., 2S]`` real ``[Re | Im]``."""

    return jnp.concatenate([values.real, values.imag], axis=-1)


@jax.jit
def _image_terms(shifted, weight, initial_diff2):
    """Per image: the weighted translated rows packed ``[B, T, 2S]`` and the pose-free score ``-C/2 - d0`` ``[B, T]``."""

    weighted = _packed(shifted * weight[:, None, :])
    image_energy = jnp.einsum("bts,bs->bt", jnp.abs(shifted) ** 2, weight, precision=jax.lax.Precision.HIGHEST)
    return weighted, -0.5 * image_energy - initial_diff2[:, None]


@partial(jax.jit, static_argnames=("precision",))
def _dense_block(weighted, image_term, weight, projections, rotation_prior, translation_prior, *, precision="float32"):
    """Scores ``[B, T, R]`` of every image against a block of shared projections ``[R, S]``; ``precision`` is the cross
    term's product algorithm (``_ALGORITHMS``), the energies stay float32."""

    cross = jax.lax.dot_general(
        weighted, _packed(projections), (((2,), (1,)), ((), ())), precision=_ALGORITHMS[precision],
        preferred_element_type=jnp.float32,
    )
    model_energy = jnp.einsum("bs,rs->br", weight, jnp.abs(projections) ** 2, precision=jax.lax.Precision.HIGHEST)
    return (
        cross
        - 0.5 * model_energy[:, None, :]
        + image_term[:, :, None]
        + translation_prior[None, :, None]
        + rotation_prior[None, None, :]
    )


@partial(jax.jit, static_argnames=("precision",))
def _masked_dense_block(
    weighted, image_term, weight, projections, member, rotation_prior, translation_prior, *, precision="float32"
):
    """:func:`_dense_block` scores ``[B, T, R]`` with ``-inf`` where ``member`` ``[B, R]`` is false."""

    scores = _dense_block(
        weighted, image_term, weight, projections, rotation_prior, translation_prior, precision=precision
    )
    return jnp.where(member[:, None, :], scores, -jnp.inf)


@partial(jax.jit, static_argnames=("precision",))
def _gathered_block(weighted, image_term, weight, table, rows, rotation_prior, translation_prior, *, precision="float32"):
    """Scores ``[B, T, C]`` of every image against its own rows ``[B, C]`` of the projection table (``-1``: none)."""

    valid = rows >= 0
    safe = jnp.maximum(rows, 0)
    projections = jnp.take(table, safe, axis=0)
    cross = jax.lax.dot_general(
        weighted, _packed(projections), (((2,), (2,)), ((0,), (0,))), precision=_ALGORITHMS[precision],
        preferred_element_type=jnp.float32,
    )
    model_energy = jnp.einsum("bs,bcs->bc", weight, jnp.abs(projections) ** 2, precision=jax.lax.Precision.HIGHEST)
    scores = (
        cross
        - 0.5 * model_energy[:, None, :]
        + image_term[:, :, None]
        + translation_prior[None, :, None]
        + jnp.take(rotation_prior, safe)[:, None, :]
    )
    return jnp.where(valid[:, None, :], scores, -jnp.inf)


@jax.jit
def _reduce_block(scores, labels, translation_mass, best):
    """Fold a block's scores ``[B, T, C]`` of the rotations ``labels`` ``[B, C]`` into the running reductions.

    Returns the block's per-rotation log mass ``[B, C]``, the running per-translation log mass ``[B, T]`` and the running
    best ``(score, rotation, translation index)``.
    """

    block_rotation = jax.nn.logsumexp(scores, axis=1)
    translation_mass = jnp.logaddexp(translation_mass, jax.nn.logsumexp(scores, axis=2))
    flat = scores.reshape(scores.shape[0], -1)
    block_best = jnp.max(flat, axis=1)
    block_arg = jnp.argmax(flat, axis=1).astype(jnp.int32)
    n_rot = scores.shape[2]
    label = jnp.take_along_axis(labels, (block_arg % n_rot)[:, None], axis=1)[:, 0]
    better = block_best > best[0]
    best = (
        jnp.where(better, block_best, best[0]),
        jnp.where(better, label, best[1]),
        jnp.where(better, block_arg // n_rot, best[2]),
    )
    return block_rotation, translation_mass, best


@partial(jax.jit, static_argnames=("max_kept", "min_kept"))
def _keep(rotation_mass, labels, mass_fraction, margin, *, max_kept: int, min_kept: int = 0):
    """Each image's kept labels ``[B, max_kept]`` (``-1`` pads) and their count, from its log masses ``[B, C]``: the
    smallest set carrying ``mass_fraction``, every label within ``margin`` of the best and the ``min_kept`` best."""

    k = min(int(max_kept), int(rotation_mass.shape[1]))
    top, where = jax.lax.top_k(rotation_mass, k)
    log_total = jax.nn.logsumexp(rotation_mass, axis=1, keepdims=True)
    mass = jnp.exp(top - log_total)
    before = jnp.cumsum(mass, axis=1) - mass
    rank = jnp.arange(k)[None, :]
    keep = ((before < mass_fraction) | (top >= top[:, :1] - margin) | (rank < min_kept)) & jnp.isfinite(top)
    kept = jnp.where(keep, jnp.take_along_axis(labels, where, axis=1), -1)
    if k < max_kept:
        kept = jnp.pad(kept, ((0, 0), (0, max_kept - k)), constant_values=-1)
    return kept, jnp.sum(keep, axis=1)


# ----------------------------------------------------------------------------------------------------------------
# Host orchestration
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class _PreparedLevel:
    """A level with its pass-1 plan, the plan's operand route over the level's translations and its projections."""

    level: BnBLevel
    plan: Any
    operand_plan: Any
    projections: Any
    rotation_prior: Any
    translation_prior: Any


def plan_level(level: BnBLevel, request_fields: dict, *, image_current_size: int | None) -> _PreparedLevel:
    """Plan pass 1 at the level's window (one zero translation; the search translates the operands itself) and
    project every rotation of the level once."""

    from relax.scoring.pass1_plan import plan_pass1
    from relax.scoring.pass1_request import Pass1Request

    fields = dict(request_fields)
    fields["engine_tuning"] = replace(fields["engine_tuning"], coarse_gemm_projection_cache=False)
    request = Pass1Request(
        rotations=np.asarray(level.rotations, dtype=np.float32),
        translations=np.zeros((1, 2), dtype=np.float32),
        current_size=int(level.size),
        image_current_size=image_current_size,
        coarse_healpix_order=int(level.order),
        **fields,
    )
    plan = plan_pass1(request)
    operand_plan = replace(
        plan.route.operand_plan, translations_source=np.asarray(level.translations, dtype=np.float64)
    )
    projector = plan.score_program_plan.projector
    rotations = jnp.asarray(level.rotations, dtype=jnp.float32)
    blocks = [
        projector.compact_rows(0, rotations[start : start + 4096], return_abs2=False)[0]
        for start in range(0, rotations.shape[0], 4096)
    ]
    projections = jnp.concatenate(blocks, axis=0).astype(jnp.complex64)
    rotation_prior = (
        jnp.zeros(level.rotations.shape[0], jnp.float32)
        if level.rotation_log_prior is None
        else jnp.asarray(level.rotation_log_prior, dtype=jnp.float32)
    )
    return _PreparedLevel(
        level=level,
        plan=plan,
        operand_plan=operand_plan,
        projections=projections,
        rotation_prior=rotation_prior,
        translation_prior=jnp.asarray(level.translation_log_prior, dtype=jnp.float32),
    )


def _level_operands(prepared: _PreparedLevel, batch_data, indices, start: int, end: int):
    """One image batch's operands at a level: ``(weighted [B, T, 2S], image_term [B, T], weight [B, S], rows)``."""

    from relax.scoring.pass1_batch import prepare_batch_inputs

    inputs = prepare_batch_inputs(prepared.plan.batch_input_plan, batch_data, indices, start_idx=start, end_idx=end)
    operands = prepared.operand_plan.prepare(inputs, indices)
    weight = jnp.asarray(operands.pixel_weight, dtype=jnp.float32)
    weighted, image_term = _image_terms(
        jnp.asarray(operands.shifted, dtype=jnp.complex64),
        weight,
        jnp.asarray(operands.initial_diff2, dtype=jnp.float32),
    )
    return weighted, image_term, weight, inputs.batch_size


def _pad_columns(rows: np.ndarray, multiple: int) -> np.ndarray:
    """``rows`` ``[B, C]`` (``-1`` pads) sorted ascending with the pads last, the columns padded to a multiple of
    ``multiple``."""

    rows = np.asarray(rows, dtype=np.int64)
    rows = np.sort(np.where(rows < 0, np.iinfo(np.int64).max, rows), axis=1)
    rows = np.where(rows == np.iinfo(np.int64).max, -1, rows)
    used = int(np.max(np.sum(rows >= 0, axis=1), initial=1))
    width = max(multiple, int(np.ceil(used / multiple)) * multiple)
    rows = rows[:, :width]
    if rows.shape[1] < width:
        rows = np.pad(rows, ((0, 0), (0, width - rows.shape[1])), constant_values=-1)
    return rows.astype(np.int32)


def _candidate_blocks(
    prepared: _PreparedLevel, weighted, image_term, weight, candidates: np.ndarray | None, *, precision: str = "float32"
):
    """Yield one batch's scores ``[B, T, C]`` and their rotations ``[B, C]`` (``-1``: not scored; on the device and on
    the host), block by block.

    ``candidates`` ``[B, K]`` are each image's rotations of the level (``-1`` pads), None for every rotation. One dense
    product over the batch's union of candidates, masked per image, is the fast route unless the union is much larger
    than the images' own candidates; then each image's projections are gathered and scored alone.
    """

    batch = int(weighted.shape[0])
    n_rot = int(prepared.projections.shape[0])
    if candidates is None:
        for start in range(0, n_rot, _DENSE_BLOCK):
            stop = min(start + _DENSE_BLOCK, n_rot)
            ids = np.broadcast_to(np.arange(start, stop, dtype=np.int32), (batch, stop - start))
            yield _dense_block(
                weighted, image_term, weight, prepared.projections[start:stop], prepared.rotation_prior[start:stop],
                prepared.translation_prior, precision=precision,
            ), jnp.asarray(ids), ids
        return
    valid = candidates >= 0
    union = np.unique(candidates[valid])
    if union.size * batch > _GATHER_RATIO * max(int(valid.sum()), 1):
        rows_host = _pad_columns(candidates, _CANDIDATE_CHUNK)
        rows = jnp.asarray(rows_host)
        for start in range(0, rows.shape[1], _CANDIDATE_CHUNK):
            chunk = rows[:, start : start + _CANDIDATE_CHUNK]
            yield _gathered_block(
                weighted, image_term, weight, prepared.projections, chunk, prepared.rotation_prior,
                prepared.translation_prior, precision=precision,
            ), chunk, rows_host[:, start : start + _CANDIDATE_CHUNK]
        return
    member = np.zeros((batch, union.size), dtype=bool)
    image = np.broadcast_to(np.arange(batch)[:, None], candidates.shape)
    member[image[valid], np.searchsorted(union, candidates[valid])] = True
    for start in range(0, union.size, _DENSE_BLOCK):
        ids = union[start : start + _DENSE_BLOCK]
        block_member = member[:, start : start + _DENSE_BLOCK]
        if ids.size < _DENSE_BLOCK:
            block_member = np.pad(block_member, ((0, 0), (0, _DENSE_BLOCK - ids.size)))
            ids = np.pad(ids, (0, _DENSE_BLOCK - ids.size))
        ids_device = jnp.asarray(ids, dtype=jnp.int32)
        member_device = jnp.asarray(block_member)
        scores = _masked_dense_block(
            weighted, image_term, weight, jnp.take(prepared.projections, ids_device, axis=0), member_device,
            jnp.take(prepared.rotation_prior, ids_device), prepared.translation_prior, precision=precision,
        )
        yield scores, jnp.where(member_device, ids_device[None, :], -1), np.where(block_member, ids[None, :], -1)


def _score_level(
    prepared: _PreparedLevel, weighted, image_term, weight, candidates: np.ndarray | None, *, precision: str
):
    """One batch at one level: per-rotation log mass ``[B, C]`` with its rotations ``[B, C]`` (``-1``: none), the
    per-translation log mass ``[B, T]`` and the best ``(score, rotation, translation index)``."""

    batch = int(weighted.shape[0])
    translation_mass = jnp.full((batch, prepared.translation_prior.shape[0]), -jnp.inf, jnp.float32)
    best = (jnp.full(batch, -jnp.inf, jnp.float32), jnp.zeros(batch, jnp.int32), jnp.zeros(batch, jnp.int32))
    masses, labels = [], []
    for scores, block_labels, _ in _candidate_blocks(
        prepared, weighted, image_term, weight, candidates, precision=precision
    ):
        block_mass, translation_mass, best = _reduce_block(scores, block_labels, translation_mass, best)
        masses.append(block_mass)
        labels.append(block_labels)
    return jnp.concatenate(masses, axis=1), jnp.concatenate(labels, axis=1), translation_mass, best


def _expand_children(kept: np.ndarray, children: np.ndarray) -> np.ndarray:
    """Each image's children ``[B, K * max_children]`` of its kept rotations ``[B, K]``, each child once (``-1`` pads)."""

    expanded = np.where(kept[:, :, None] >= 0, children[np.maximum(kept, 0)], -1).reshape(kept.shape[0], -1)
    expanded = np.sort(expanded, axis=1)
    repeated = np.zeros_like(expanded, dtype=bool)
    repeated[:, 1:] = expanded[:, 1:] == expanded[:, :-1]
    return np.where(repeated, -1, expanded)


def centre_table(translations: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """The whole-pixel window centres within the reach of ``translations`` ``[T, 2]`` and, for each, which
    translations its window of ``radius`` holds: ``(centres [C, 2], inside [T, C] float)``."""

    reach = int(np.ceil(np.max(np.abs(translations), initial=0.0)))
    axis = np.arange(-reach, reach + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    centres = np.stack([xx.ravel(), yy.ravel()], axis=1)
    centres = centres[np.sum(centres**2, axis=1) <= reach**2 + 1e-9]
    inside = np.sum((translations[:, None, :] - centres[None, :, :]) ** 2, axis=2) <= float(radius) ** 2 + 1e-9
    return centres, inside.astype(np.float64)


def window_centres(
    translation_mass: np.ndarray,
    translations: np.ndarray,
    radius: float,
    *,
    hold: float = 0.99,
    table=None,
    previous: np.ndarray | None = None,
    recentre_gain: float = 0.0,
):
    """Each image's window centre: the whole pixel whose window (``radius``) holds the most posterior mass of the
    offset (of several, the one nearest the posterior mean), which is the mode of a sharp posterior and near the mean of
    a broad one; also the posterior standard deviation (pixels) of the offset and the radius around the centre that
    holds ``hold`` of its mass.

    ``translation_mass`` ``[B, T]`` is the log posterior mass of every translation ``translations`` ``[T, 2]``;
    ``table`` is their :func:`centre_table` (None: built here). With ``previous`` ``[B, 2]`` (each image's previous
    offset in the same frame), an image keeps the window at its rounded previous offset unless the best window
    holds at least ``recentre_gain`` more of its posterior mass: a broad posterior does not move a window by noise.
    """

    from relax.relion.macros import relion_round_array

    centres, inside = centre_table(translations, radius) if table is None else table
    mass = np.asarray(translation_mass, dtype=np.float64)
    weights = np.exp(mass - mass.max(axis=1, keepdims=True))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
    mean = weights @ translations
    # Windows holding the same mass (a wide window around a sharp peak) go to the one nearest the posterior mean.
    captured = weights @ inside - 1e-9 * np.sum((centres[None, :, :] - mean[:, None, :]) ** 2, axis=2)
    choice = np.argmax(captured, axis=1)
    best = centres[choice]
    if previous is not None:
        kept_at = relion_round_array(np.asarray(previous, dtype=np.float64))
        within = np.sum((translations[None, :, :] - kept_at[:, None, :]) ** 2, axis=2) <= float(radius) ** 2 + 1e-9
        previous_mass = np.sum(np.where(within, weights, 0.0), axis=1)
        stay = captured[np.arange(mass.shape[0]), choice] < previous_mass + float(recentre_gain)
        best = np.where(stay[:, None], kept_at, best)
    spread = np.sqrt(np.maximum(np.sum(weights * np.sum((translations[None] - mean[:, None]) ** 2, axis=2), axis=1), 0))
    distance = np.sqrt(np.sum((translations[None, :, :] - best[:, None, :]) ** 2, axis=2))
    order = np.argsort(distance, axis=1)
    held = np.cumsum(np.take_along_axis(weights, order, axis=1), axis=1)
    first = np.minimum(np.sum(held < float(hold), axis=1), translations.shape[0] - 1)
    hold_radius = np.take_along_axis(distance, order, axis=1)[np.arange(mass.shape[0]), first]
    return best, spread, hold_radius


def _level_pass(
    prepared: _PreparedLevel,
    dataset,
    *,
    parents: list[np.ndarray] | None,
    children: np.ndarray | None,
    pre_shifts: np.ndarray | None,
    prior: tuple[float, float] | None,
    pruning: BnBPruning,
    coarse: bool,
    centre_radius: float,
    image_batch_size: int,
    precision: str,
    clock: dict,
):
    """One level over every image: each image's kept rotations, kept count, window centre (relative to its pre-shift),
    offset spread and hold radius, and best ``(rotation, translation)``.

    ``parents`` holds each image's kept rotations of the previous level (None at level 0, which scores every rotation);
    ``pre_shifts`` ``[N, 2]`` centre a local level's translations on each image (None: the absolute grid) and ``prior``
    ``(pixel_size, sigma_offset_angstrom)`` gives them their offset prior, at ``pre_shift + t``. Also returns every
    image's log posterior mass of the level's translations ``[N, T]`` (float32). A finer level batches the images
    in order of their parent counts: a batch scores as many candidates as its widest image.
    """

    from relax.io.batch_fetch import iter_indexed_batches

    n_images = int(dataset.n_units)
    translations = prepared.level.translations
    table = centre_table(translations, centre_radius)
    order = np.arange(n_images)
    if parents is not None:
        order = np.argsort(np.fromiter((row.size for row in parents), np.int64, count=n_images), kind="stable")
    kept_rows: list[np.ndarray | None] = [None] * n_images
    kept_counts = np.zeros(n_images, dtype=np.int32)
    centres = np.zeros((n_images, 2), dtype=np.float64)
    spread = np.zeros(n_images, dtype=np.float64)
    hold = np.zeros(n_images, dtype=np.float64)
    best_rotation = np.zeros(n_images, dtype=np.int64)
    best_translation = np.zeros((n_images, 2), dtype=np.float64)
    masses = np.zeros((n_images, translations.shape[0]), dtype=np.float32)
    start = 0
    for batch_data, _, _, _, _, _, indices in iter_indexed_batches(dataset, order, image_batch_size):
        end = start + len(indices)
        count = end - start
        batch = order[start:end]
        tick = time.time()
        weighted, image_term, weight, batch_size = _level_operands(prepared, batch_data, indices, start, end)
        if pre_shifts is not None:
            absolute = pre_shifts[batch][:, None, :] + translations[None, :, :]
            image_prior = offset_log_prior(absolute.reshape(-1, 2), pixel_size=prior[0], sigma_offset_angstrom=prior[1])
            image_prior = image_prior.reshape(count, -1)
            if batch_size > count:
                image_prior = np.concatenate([image_prior, np.repeat(image_prior[:1], batch_size - count, axis=0)])
            image_term = image_term + jnp.asarray(image_prior)
        weighted.block_until_ready()
        clock["operands"] += time.time() - tick
        tick = time.time()
        rows = None
        if parents is not None:
            parent_rows = np.full((batch_size, max(max(parents[i].size for i in batch), 1)), -1)
            for i, image in enumerate(batch):
                parent_rows[i, : parents[image].size] = parents[image]
            rows = _expand_children(parent_rows, children)
            clock["children"] += time.time() - tick
            tick = time.time()
        mass, labels, translation_mass, best = _score_level(
            prepared, weighted, image_term, weight, rows, precision=precision
        )
        mass.block_until_ready()
        clock["scores"] += time.time() - tick
        tick = time.time()
        kept_device, n_kept = _keep(
            mass, labels, pruning.mass_fraction, pruning.margin,
            max_kept=int(pruning.max_kept) if coarse else int(mass.shape[1]),
            min_kept=int(pruning.min_kept) if coarse else 0,
        )
        kept = np.asarray(kept_device)[:count]
        for image, row in zip(batch, kept):
            kept_rows[image] = np.sort(row[row >= 0]).astype(np.int32)
        kept_counts[batch] = np.asarray(n_kept)[:count]
        clock["keep"] += time.time() - tick
        tick = time.time()
        masses[batch] = np.asarray(translation_mass)[:count]
        centres[batch], spread[batch], hold[batch] = window_centres(
            masses[batch], translations, centre_radius, table=table
        )
        best_rotation[batch] = np.asarray(best[1])[:count]
        best_translation[batch] = translations[np.asarray(best[2])[:count]]
        clock["centres"] += time.time() - tick
        start = end
    return kept_rows, kept_counts, centres, spread, hold, best_rotation, best_translation, masses


def run_search(
    levels: list[BnBLevel],
    request_fields: dict,
    *,
    pruning: BnBPruning,
    image_current_size: int | None,
    centre_radius: float,
    image_batch_size: int,
    pixel_size: float,
    sigma_offset_angstrom: float,
    precision: str = "float32",
) -> BnBSearchResult:
    """Search every image of ``request_fields["experiment_dataset"]`` over ``levels`` (see the module docstring).

    ``request_fields`` are the :class:`~relax.scoring.pass1_request.Pass1Request` fields every level shares (dataset,
    noise, projector, corrections, sizes and execution settings), with ``image_pre_shifts`` None. Level 0 scores the
    absolute translation grid of ``levels[0]``; each finer level pre-shifts every image to the previous level's centre
    and scores a disk of its own step around it, wide enough for 90% of the images' 99% hold radii at the previous
    level (at least 2.5 of its step and ``centre_radius`` plus a step, at most the search radius), with the offset
    prior (``pixel_size``, ``sigma_offset_angstrom``) at the absolute offsets. ``centre_radius`` (pixels) is the
    RELION window the centres are chosen for. ``precision`` is the score products' algorithm (``"float32"`` or
    ``"tf32"``). Coarse levels keep at most ``pruning.max_kept`` rotations per image; the last level keeps every
    rotation its rules keep. A device batch holds at least ``image_batch_size`` images (:func:`image_batch`).
    """

    t0 = time.time()
    dataset = request_fields["experiment_dataset"]
    n_images = int(dataset.n_units)
    n_levels = len(levels)
    search_radius = float(np.max(np.linalg.norm(levels[0].translations, axis=1), initial=0.0))
    clock = {"plan": 0.0, "operands": 0.0, "children": 0.0, "scores": 0.0, "keep": 0.0, "centres": 0.0}
    level_scores = []
    batches = []
    kept_counts = np.zeros((n_levels, n_images), dtype=np.int32)
    parents = None
    pre_shifts = None
    hold = None
    used_levels = []
    for index, level in enumerate(levels):
        fields = dict(request_fields)
        if index > 0:
            previous_step = float(used_levels[-1].step)
            radius = float(np.quantile(hold, 0.9)) if hold.size else 0.0
            radius = min(max(radius, 2.5 * previous_step, float(centre_radius) + level.step), search_radius)
            level = replace(level, translations=translation_grid(radius, level.step))
            fields["image_pre_shifts"] = pre_shifts
        level = replace(level, translation_log_prior=(
            level.translation_log_prior if index == 0 else np.zeros(level.translations.shape[0], np.float32)
        ))
        used_levels.append(level)
        level_batch = image_batch(image_batch_size, level.translations.shape[0], level.size)
        fields["image_batch_size"] = level_batch
        batches.append(level_batch)
        tick = time.time()
        prepared = plan_level(level, fields, image_current_size=image_current_size)
        clock["plan"] += time.time() - tick
        scores_before = clock["scores"]
        (
            parents, kept_counts[index], local_centres, spread, hold, best_rotation, best_translation, masses
        ) = _level_pass(
            prepared,
            dataset,
            parents=parents,
            children=level.children,
            pre_shifts=pre_shifts,
            prior=(float(pixel_size), float(sigma_offset_angstrom)),
            pruning=pruning,
            coarse=index < n_levels - 1,
            centre_radius=centre_radius,
            image_batch_size=level_batch,
            precision=precision,
            clock=clock,
        )
        level_scores.append(round(clock["scores"] - scores_before, 1))
        base = np.zeros((n_images, 2)) if pre_shifts is None else pre_shifts
        pre_shifts = base + local_centres
        best_translation = base + best_translation
    seconds = time.time() - t0
    logger.info(
        "BnB search time: %s, scores per level %s", {name: round(value, 1) for name, value in clock.items()}, level_scores
    )
    logger.info(
        "BnB search: %d images, levels %s (sizes %s, translations %s, image batches %s), kept rotations per level "
        "median %s max %s, %.1fs",
        n_images,
        [lv.order for lv in used_levels],
        [lv.size for lv in used_levels],
        [int(lv.translations.shape[0]) for lv in used_levels],
        batches,
        [int(np.median(row)) for row in kept_counts],
        [int(np.max(row, initial=0)) for row in kept_counts],
        seconds,
    )
    return BnBSearchResult(
        centres=pre_shifts,
        candidates=parents,
        kept_counts=kept_counts,
        best_rotation=best_rotation,
        best_translation=best_translation,
        translation_spread=spread,
        hold_radius=hold,
        final_mass=masses,
        final_translations=used_levels[-1].translations,
        final_base=base,
        seconds=seconds,
    )


@partial(jax.jit, static_argnames=("max_significants",))
def _significant_mask(scores, fraction, *, max_significants: int):
    """Each image's significant scores ``[B, M]``: the highest scores until their posterior mass reaches
    ``fraction`` (at most ``max_significants`` when positive), as a mask; scores tied with the last one are kept.

    The threshold is found by bisection on the score (no sort) unless a cap asks for a count.
    """

    finite = jnp.isfinite(scores)
    log_total = jax.nn.logsumexp(scores, axis=1, keepdims=True)
    weights = jnp.where(finite, jnp.exp(scores - log_total), 0.0)
    high = jnp.max(jnp.where(finite, scores, -jnp.inf), axis=1)
    low = jnp.min(jnp.where(finite, scores, jnp.inf), axis=1)

    def bisect(_, bounds):
        low, high = bounds
        middle = 0.5 * (low + high)
        enough = jnp.sum(jnp.where(scores >= middle[:, None], weights, 0.0), axis=1) >= fraction
        return jnp.where(enough, middle, low), jnp.where(enough, high, middle)

    low, _ = jax.lax.fori_loop(0, 48, bisect, (low, high))
    mask = (scores >= low[:, None]) & finite
    if max_significants > 0:
        threshold = jax.lax.top_k(scores, int(max_significants))[0][:, -1]
        mask = mask & (scores >= threshold[:, None])
    return mask


@partial(jax.jit, static_argnames=("size", "n_window"))
def _significant_ids(mask, rotation_of_column, *, size: int, n_window: int):
    """Each image's significant samples ``rotation * n_window + w`` from its mask ``[B, M]`` over the columns
    ``(candidate, window translation)``, ascending, ``-1`` padded to ``size``."""

    columns = jax.vmap(lambda row: jnp.nonzero(row, size=size, fill_value=-1)[0])(mask)
    rotation = jnp.take_along_axis(rotation_of_column, jnp.maximum(columns, 0) // n_window, axis=1)
    return jnp.where(columns >= 0, rotation * n_window + columns % n_window, -1)


def window_significance(
    level: BnBLevel,
    request_fields: dict,
    candidates: list[np.ndarray],
    window: np.ndarray,
    window_log_prior: np.ndarray,
    *,
    image_current_size: int | None,
    adaptive_fraction: float,
    max_significants: int,
    image_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pass 1 over each image's candidate rotations: its significant samples ``rotation * W + w`` on the window.

    ``request_fields["image_pre_shifts"]`` are the window centres; ``window`` ``[W, 2]`` are the relative translations
    of pass 2's coarse grid and ``window_log_prior`` ``[N, W]`` (or ``[W]``) their offset prior. ``level`` gives the
    rotations (pass 2's coarse grid), their prior and the window size; ``candidates`` holds each image's rotations
    (sorted). Returns each image's count ``[N]`` and their samples, ascending within each image, one flat int32 array.
    """

    from relax.io.batch_fetch import iter_indexed_batches

    t0 = time.time()
    window_level = replace(
        level, translations=np.asarray(window, dtype=np.float64), translation_log_prior=np.zeros(len(window), np.float32)
    )
    image_batch_size = image_batch(image_batch_size, len(window), level.size)
    request_fields = dict(request_fields, image_batch_size=image_batch_size)
    prepared = plan_level(window_level, request_fields, image_current_size=image_current_size)
    dataset = request_fields["experiment_dataset"]
    n_images = int(dataset.n_units)
    n_window = int(np.asarray(window).shape[0])
    prior_rows = np.asarray(window_log_prior, dtype=np.float32)
    counts_per_batch: list[np.ndarray] = []
    ids_per_batch: list[np.ndarray] = []
    clock = {"operands": 0.0, "scores": 0.0, "mask": 0.0, "ids": 0.0}
    start = 0
    for batch_data, _, _, _, _, _, indices in iter_indexed_batches(dataset, np.arange(n_images), image_batch_size):
        end = start + len(indices)
        count = end - start
        tick = time.time()
        weighted, image_term, weight, batch_size = _level_operands(prepared, batch_data, indices, start, end)
        weighted.block_until_ready()
        clock["operands"] += time.time() - tick
        image_prior = prior_rows[start:end] if prior_rows.ndim == 2 else np.broadcast_to(prior_rows, (count, n_window))
        if batch_size > count:
            image_prior = np.concatenate([image_prior, np.repeat(image_prior[:1], batch_size - count, axis=0)])
        image_term = image_term + jnp.asarray(image_prior)
        width = max(int(row.size) for row in candidates[start:end])
        rows = np.full((batch_size, max(width, 1)), -1, dtype=np.int64)
        for i, row in enumerate(candidates[start:end]):
            rows[i, : row.size] = row
        tick = time.time()
        masks, labels = [], []
        for block_scores, block_labels, _ in _candidate_blocks(prepared, weighted, image_term, weight, rows):
            masks.append(jnp.swapaxes(block_scores, 1, 2).reshape(batch_size, -1))
            labels.append(block_labels)
        joined = jnp.concatenate(masks, axis=1).block_until_ready()
        clock["scores"] += time.time() - tick
        tick = time.time()
        mask = _significant_mask(joined, adaptive_fraction, max_significants=int(max_significants))
        counts = np.asarray(jnp.sum(mask, axis=1))[:count].astype(np.int32)
        clock["mask"] += time.time() - tick
        tick = time.time()
        # Columns run over ascending rotations, then window translations: each image's ids come out ascending.
        size = 1 << int(np.ceil(np.log2(max(int(counts.max(initial=1)), 1))))
        ids = np.asarray(
            _significant_ids(mask, jnp.concatenate(labels, axis=1), size=size, n_window=n_window)
        )[:count]
        ids_per_batch.append(ids[np.arange(size)[None, :] < counts[:, None]].astype(np.int32))
        counts_per_batch.append(counts)
        clock["ids"] += time.time() - tick
        start = end
    counts = np.concatenate(counts_per_batch) if counts_per_batch else np.zeros(0, np.int32)
    logger.info("BnB window significance time: %s", {name: round(value, 1) for name, value in clock.items()})
    logger.info(
        "BnB window significance: %d images, %d window translations, significant samples median %d mean %.1f max %d, "
        "%.1fs",
        n_images,
        n_window,
        int(np.median(counts)) if counts.size else 0,
        float(counts.mean()) if counts.size else 0.0,
        int(counts.max(initial=0)),
        time.time() - t0,
    )
    return counts, np.concatenate(ids_per_batch) if ids_per_batch else np.zeros(0, np.int32)
