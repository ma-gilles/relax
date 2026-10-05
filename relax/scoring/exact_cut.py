"""Direct-square scores where the coarse GEMM scorer does not decide the significance cut.

The subtomogram coarse pass scores RELION's ``d0 + 0.5 sum w |p - y|^2`` as the expansion
``d0 + 0.5 A + 0.5 C - X`` (two float32 GEMMs, :func:`relax.scoring.scoring._relion_coarse_gemm_terms`),
which rounds differently from RELION's sum of squares. A particle's diff2 is the sum of its tilt images',
thousands with a float32 unit of 2e-4, while its samples' log weights at the significance cut are closer
than that: the rounding moved samples across the cut (194 of 200 particles' kept sets matched RELION's on
etob2l_plain, iteration 2). The pass therefore takes its cut twice: on the GEMM scores, and again after the
rotations holding an undecided sample (:func:`undecided_rotations`, within the bound
:func:`gemm_error_bound` of the cut) have been scored by RELION's fused direct-square kernel
(:func:`direct_rows_diff2`). The single-particle pass 1 keeps its GEMM cut: with one image per particle it
reproduces RELION's (plain 10k K=1, iteration 2: maps within RELION-vs-RELION, jobs 14974373/14974374).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# The fused coarse projector's translation capacity (one 128-thread block).
FUSED_TRANSLATION_CAPACITY = 128

# Per-image diff2 elements ([images, rows, T] float32, 1 GiB) one block of direct_rows_diff2 computes.
DIRECT_ROWS_BLOCK_ELEMENTS = 1 << 28


def float32_unit(values):
    """The float32 spacing above each of ``values``."""

    values = jnp.abs(jnp.asarray(values, dtype=jnp.float32))
    return jnp.nextafter(values, jnp.float32(jnp.inf)) - values


def gemm_error_bound(model_energy, image_energy, initial_diff2, n_pixels: int):
    """A bound on an image's GEMM diff2 minus its direct square, over its rotations and translations.

    The GEMM scorer adds ``d0 + 0.5 A + 0.5 C - X`` (model energy, image energy, cross term), each a float32
    sum over the ``P = n_pixels`` score pixels, where the direct square adds ``P`` non-negative squares. A
    float32 sum of ``n`` products is within ``n u`` of its sum of magnitudes in any order (``u = 2^-24``),
    which for ``X`` is at most ``sqrt(A C)`` over ``2 P`` products, for ``A`` and ``C`` the energies over
    ``P`` terms of three operations, and for the direct square the diff2 over ``P`` terms of five. With the
    three closing additions the two differ by at most ``(4 P + 11) u`` times ``d0 + 0.5 A + 0.5 C +
    sqrt(A C)``, which no term exceeds. ``model_energy`` is the image's largest ``A`` over the rotations (or
    an upper bound of it), ``image_energy`` its ``C``.
    """

    return jnp.float32((4 * int(n_pixels) + 11) * 2.0**-24) * largest_term(model_energy, image_energy, initial_diff2)


def largest_term(model_energy, image_energy, initial_diff2):
    """``d0 + 0.5 A + 0.5 C + sqrt(A C)``: no term of the GEMM expansion, and no diff2, exceeds it."""

    largest = jnp.abs(initial_diff2) + 0.5 * model_energy + 0.5 * image_energy + jnp.sqrt(model_energy * image_energy)
    return largest.astype(jnp.float32)


def undecided_rotations(
    values, mask, cutoff_count, margin, diff2, smallest_diff2,
    *, n_trans: int, max_significants: int, adaptive_fraction=None, sorted_rotations=None,
):
    """The rotations ``[B, N]`` whose GEMM scores do not decide an image's significance cut, and their largest
    count over the images.

    ``values`` ``[B, N * T]`` are the log weights the cut was taken on and ``mask`` its significant samples;
    ``margin`` ``[B]`` is twice a bound on a sample's GEMM diff2 minus its direct square, so two samples'
    order is decided when their log weights are more than ``margin`` apart.

    Cut by ``max_significants`` (``cutoff_count`` reaches it): a sample more than ``margin`` above the cut's log
    weight is among the direct squares' ``max_significants`` largest and one more than ``margin`` below is not,
    whatever the others round to; the samples in between decide the cut.

    Cut by the adaptive fraction (with ``adaptive_fraction``; otherwise such an image has no undecided
    rotation): RELION drops the smallest weights whose sum stays within ``(1 - fraction)`` of the total. Each
    weight and the total move by at most ``exp(margin / 2)``, so on the sorted weights a sample is out
    whatever the rounding when it lies more than ``margin`` below the first sample whose running sum exceeds
    the tail shrunk by ``exp(-margin)``, and in when it lies above the first sample whose running sum exceeds
    the tail grown by ``exp(margin)`` (that sample too, when its lower neighbour is more than ``margin``
    away). The samples in between are undecided.

    The fraction rule sorts only the samples of each image's ``sorted_rotations`` rotations with the largest
    log weights (all of them with ``None``) and counts the other rotations' weight as lying below them. That
    is the same rule when every sample of the other rotations is out with certainty; the third result
    counts the images for which it is not, whose rows must be taken again with ``sorted_rotations=None``.

    ``diff2`` ``[B, N, T]`` and its smallest value per image ``smallest_diff2`` ``[B]`` are given when the log
    weights are RELION's ``prior + min_diff2 - diff2``: at a diff2 of thousands the first sum is rounded to the
    diff2's float32 unit, sample by sample, so another ``min_diff2`` moves the log weights by a unit each way
    and reorders them at the cut. For an image with an undecided sample at its cut, the samples within
    ``margin`` of the smallest diff2 decide ``min_diff2`` and are undecided too; an image whose cut is decided
    keeps its GEMM scores (``margin`` allows for that unit).
    """

    n_images = values.shape[0]
    capped = cutoff_count >= max_significants if max_significants > 0 else jnp.zeros(n_images, bool)
    cut = jnp.min(jnp.where(mask, values, jnp.inf), axis=1)
    scoreable = margin >= 0  # a negative margin marks an image that is not to be scored again (padding)
    low = jnp.where(capped & scoreable, cut - margin, jnp.inf)
    high = jnp.where(capped & scoreable, jnp.nextafter(cut + margin, jnp.float32(jnp.inf)), -jnp.inf)
    unresolved = jnp.zeros((), jnp.int32)
    if adaptive_fraction is not None:
        table = values.reshape(n_images, -1, n_trans)
        best = jnp.max(values, axis=1)
        n_rotations = table.shape[1]
        if sorted_rotations is None or sorted_rotations >= n_rotations:
            ordered, floor, far = jnp.sort(values, axis=1), None, 0.0
        else:
            # The rotations with the largest sample, sorted; the others' weight lies below their samples.
            rotation_best = jnp.max(table, axis=2)
            floor, kept = jax.lax.top_k(rotation_best, sorted_rotations + 1)
            floor, kept = floor[:, -1], kept[:, :-1]  # no sample of another rotation exceeds ``floor``
            ordered = jnp.sort(jnp.take_along_axis(table, kept[:, :, None], axis=1).reshape(n_images, -1), axis=1)
            rotation_weight = jnp.sum(
                jnp.where(jnp.isfinite(table), jnp.exp((table - best[:, None, None]).astype(jnp.float64)), 0.0), axis=2
            )
            is_kept = jnp.zeros(rotation_best.shape, bool).at[jnp.arange(n_images)[:, None], kept].set(True)
            far = jnp.sum(jnp.where(is_kept, 0.0, rotation_weight), axis=1)[:, None]
        weights = jnp.where(jnp.isfinite(ordered), jnp.exp((ordered - best[:, None]).astype(jnp.float64)), 0.0)
        below = far + jnp.cumsum(weights, axis=1)  # the weight of the samples up to and including each one
        tail = (1.0 - jnp.float64(jnp.float32(adaptive_fraction))) * below[:, -1]
        wide = margin.astype(jnp.float64)
        search = jax.vmap(lambda row, target: jnp.searchsorted(row, target, side="right"))
        last = ordered.shape[1] - 1
        at = lambda index: jnp.take_along_axis(ordered, jnp.clip(index, 0, last)[:, None], axis=1)[:, 0]  # noqa: E731
        first_not_out = search(below, tail * jnp.exp(-wide))
        surely_out = at(first_not_out) - margin  # below this: out
        first_in = search(below, tail * jnp.exp(wide))
        isolated = (first_in > 0) & (at(first_in) - at(first_in - 1) > margin)
        # At or above this: in.
        surely_in = jnp.where(isolated, at(first_in), jnp.nextafter(at(first_in) + margin, jnp.float32(jnp.inf)))
        if floor is not None:
            # Every sample of the unsorted rotations must be out with certainty.
            unresolved = jnp.sum(~capped & scoreable & ~(surely_out > floor), dtype=jnp.int32)
        low = jnp.where(capped | ~scoreable, low, surely_out)
        high = jnp.where(capped | ~scoreable, high, surely_in)
    undecided = (jnp.isfinite(values) & (values >= low[:, None]) & (values < high[:, None])).reshape(
        n_images, -1, n_trans
    )
    if diff2 is not None:
        at_cut = jnp.any(undecided, axis=(1, 2))
        undecided |= (diff2 <= (smallest_diff2 + margin)[:, None, None]) & at_cut[:, None, None]
    rows = jnp.any(undecided, axis=2)
    return rows, jnp.max(jnp.sum(rows, axis=1)), unresolved


def row_capacity(most: int, n_rows: int, least: int = 64) -> int:
    """The rows scored again per image: ``most`` rounded up to a power of two (at least ``least``), so few
    shapes compile."""

    return min(1 << max(int(most) - 1, int(least) - 1).bit_length(), int(n_rows))


@partial(jax.jit, static_argnames=("capacity",))
def first_rows(rows, *, capacity: int):
    """Each image's first ``capacity`` true columns of ``rows`` ``[B, N]`` (int32, ascending; zero past the image's
    count) and which of the ``capacity`` slots hold one."""

    n_images, n_rows = rows.shape
    position = jnp.cumsum(rows, axis=1, dtype=jnp.int32) - 1
    # A column that is not kept writes past the end, which the scatter drops.
    position = jnp.where(rows & (position < capacity), position, capacity)
    ids = jnp.zeros((n_images, capacity), jnp.int32).at[jnp.arange(n_images)[:, None], position].set(
        jnp.broadcast_to(jnp.arange(n_rows, dtype=jnp.int32), rows.shape), mode="drop"
    )
    count = jnp.minimum(jnp.sum(rows, axis=1, dtype=jnp.int32), capacity)
    return ids, jnp.arange(capacity, dtype=jnp.int32)[None, :] < count[:, None]


@jax.jit
def put_rows(table, ids, own, rows):
    """``table`` ``[B, N, T]`` with rows ``ids`` ``[B, M]`` replaced by ``rows`` ``[B, M, T]`` where ``own``."""

    image = jnp.arange(table.shape[0])[:, None]
    # A slot that is not written points past the end, which the scatter drops: an empty slot's id is zero and
    # would otherwise write row zero back over its new value.
    return table.at[image, jnp.where(own, ids, table.shape[1])].set(rows, mode="drop")


def direct_rows_diff2(
    projector_full,
    rotations,
    unshifted,
    pixel_weight,
    initial_diff2,
    translation_angles,
    full_to_compact,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
    padding_factor: int,
):
    """Direct-square diff2 ``[B, M, T]`` at each particle's own rotations ``[B, S, M, 3, 3]``.

    RELION's fused coarse kernel on every image ``[B, S]`` with its own phases ``[B, S, T, 2]``
    (:func:`relax.cuda.kernels.relion_coarse_diff2_projector_per_image_f32`, the translations in blocks of
    its capacity), the particle's ``S`` images added in slot order; a single-particle image is ``S = 1``.
    A rotation's diff2 does not depend on the other rotations of the call.
    """

    from relax.cuda import kernels as em_cuda_kernels

    n_particles, n_slots, n_rows = (int(n) for n in rotations.shape[:3])
    n_images, n_trans = n_particles * n_slots, int(translation_angles.shape[2])
    angles = translation_angles.reshape(n_images, n_trans, 2)
    images = rotations.reshape(n_images, n_rows, 3, 3)
    # The rows go in blocks whose per-image diff2 [images, rows, T] stays within DIRECT_ROWS_BLOCK_ELEMENTS: the
    # kernel counts its outputs in int32, and a flat K>1 posterior can leave thousands of rows undecided.
    block_rows = max(1, min(n_rows, DIRECT_ROWS_BLOCK_ELEMENTS // max(n_images * n_trans, 1)))
    totals = []
    for first in range(0, n_rows, block_rows):
        rows = images[:, first : first + block_rows]
        blocks = [
            em_cuda_kernels.relion_coarse_diff2_projector_per_image_f32(
                projector_full,
                rows,
                unshifted.reshape(n_images, -1),
                angles[:, start : start + FUSED_TRANSLATION_CAPACITY],
                pixel_weight.reshape(n_images, -1),
                initial_diff2.reshape(n_images),
                full_to_compact,
                current_size=int(current_size),
                physical_image_size=int(physical_image_size),
                model_max_r=int(model_max_r),
                padding_factor=int(padding_factor),
            )
            for start in range(0, n_trans, FUSED_TRANSLATION_CAPACITY)
        ]
        image_diff2 = (blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=2)).reshape(
            n_particles, n_slots, int(rows.shape[1]), n_trans
        )
        totals.append(_add_in_slot_order(image_diff2))
    return totals[0] if len(totals) == 1 else jnp.concatenate(totals, axis=1)


@jax.jit
def _add_in_slot_order(image_diff2):
    total = image_diff2[:, 0]
    for slot in range(1, int(image_diff2.shape[1])):
        total = total + image_diff2[:, slot]
    return total
