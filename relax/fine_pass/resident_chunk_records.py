"""Records a resident chunk's programs take and return, and how their inputs are placed.

The program spec (the static compile key of a capacity class), the chunk's row arrays, operands and stage
tables, the posterior and the M-step carry, and the M-step block inputs. A chunk constructor places its host
values with a :class:`Placement`: ``PLACE_ON_DEVICE`` (here) for the chunk loop, or the aval placement of
the compile-ahead warm-up (:mod:`relax.fine_pass.resident_chunk_programs`), so a warm-up cannot describe a
program the loop does not run. ``_DEVICE_INT32_CACHE`` holds the device row offsets for the process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Placement(NamedTuple):
    """How a chunk constructor turns host values into program inputs.

    The chunk loop places them on the device; the compile-ahead warm-up places
    their shape and dtype only. Having one constructor with two placements
    rather than two constructors means a warm-up cannot describe a program the
    loop does not run, which is the failure a hit rate would only report after
    the compile time had already been spent.
    """

    array: object
    scalar: object
    # ``{name: (value, dtype)} -> {name: placed}`` for a chunk's arrays and
    # scalars together.
    many: object


def _device_put_many(items: dict) -> dict:
    """Upload a chunk's host values in one ``device_put`` instead of one each.

    Each value becomes a NumPy array of its dtype first, so the transfer is
    the one ``jnp.asarray`` (or, for a scalar, :func:`_scalar_operand`) makes:
    the same dtype, shape and value, never weakly typed. Eleven separate puts
    per chunk took 137 s of the main thread over the full 10097 global
    iterations.
    """

    return jax.device_put({name: np.asarray(value, dtype=jnp.dtype(dtype)) for name, (value, dtype) in items.items()})


PLACE_ON_DEVICE = Placement(
    array=lambda value, dtype: jnp.asarray(value, dtype=dtype),
    # ``np.asarray`` first: a NumPy *scalar* reaches the device through one
    # eager ``convert_element_type`` per chunk, a 0-d NumPy *array* of the same
    # dtype through a plain transfer. Same dtype, shape, weak type and value
    # either way.
    scalar=lambda value, dtype: _scalar_operand(value, dtype),
    many=_device_put_many,
)


_DEVICE_INT32_CACHE: dict[int, jax.Array] = {}


def _scalar_operand(value, dtype) -> jax.Array:
    """Put a host scalar on the device without an eager conversion.

    ``jnp.asarray(np.int32(7), dtype=jnp.int32)`` dispatches a
    ``convert_element_type`` because a NumPy scalar is not an array; the same
    value wrapped in a 0-d NumPy array of the target dtype is transferred with
    no primitive at all. The chunk driver builds two of these per chunk, so on
    the early state that was 834 eager dispatches over two iterations for two
    integers whose value never leaves the host.
    """

    return jnp.asarray(np.asarray(value, dtype=jnp.dtype(dtype)))


def device_int32(value: int) -> jax.Array:
    """A device int32 scalar, made once per distinct value for the process.

    The M-step block program takes its row offset as a device operand so that
    one program serves every block of a capacity class. Building that scalar
    with ``jnp.asarray`` inside the loop would put one eager dispatch back per
    block, which is the cost this program exists to remove; the offsets are a
    handful of multiples of the block size, so they are made once and reused.
    Single-device only, which is what the EM engines run on; a multi-device
    process falls through to a fresh array.
    """

    key = int(value)
    if len(jax.devices()) != 1:
        return jnp.asarray(key, dtype=jnp.int32)
    cached = _DEVICE_INT32_CACHE.get(key)
    if cached is None:
        cached = jnp.asarray(key, dtype=jnp.int32)
        _DEVICE_INT32_CACHE[key] = cached
    return cached


@dataclass(frozen=True)
class ChunkProgramSpec:
    """Everything the chunk program is keyed on.

    Only capacity classes, pixel counts and resolved configuration appear here,
    so two chunks of one class share a program whatever their occupancy: the
    valid row and image counts travel as device scalars.
    """

    row_capacity: int
    image_capacity: int
    n_fine_trans: int
    n_score_pixels: int
    n_recon_pixels: int
    n_rect: int
    mstep_block_rows: int
    adaptive_fraction: float
    current_size: int
    mstep_current_size: int
    image_shape: tuple
    # The M-step adjoint's max_r: mstep_current_size // 2 on one grid, a ReferenceSphereClip
    # for images on another grid (relax.projection.adjoint).
    mstep_max_r: object
    recon_volume_shape: tuple
    max_adjoint_block_bytes: int
    stats_config: object
    use_rfloat_ctf_wavg: bool
    # T16: whether the M-step's weighted sums come from the translate-and-sum
    # kernel on unshifted operands, and whether its reconstruction operand takes
    # the BPref convention (a weight) or the score convention (no weight).
    use_translate_sum_kernel: bool
    bpref_recon_operand: bool
    # Whether ``ctf_probs`` comes from the kernel's fourth output instead of the
    # XLA statement the per-chunk path used. Off by default: the kernel's own
    # translation mass is bitwise against ``jnp.sum`` only at some shapes.
    kernel_ctf_probs: bool
    # Zero oversampling: the posterior keeps every weight, divides by the
    # retained coarse sum and reports the coarse winner and Pmax
    # (CoarseNormalizationReuse). The tables carry the arrays.
    reuse_coarse_normalization: bool = False
    # RELION --firstiter_cc: normalized-CC scores and a winner-take-all
    # posterior (ml_optimiser.cpp:8844-8858, :9266-9293).
    firstiter_cc: bool = False
    n_slots: int = 1
    # VDAM (--grad): BPref takes the residual shift(img) - ctf * proj, RELION's
    # cuda_kernel_backproject3D_SGD (BP.cuh:406-560); see _resident_block_residual.
    mstep_subtract_ctf_projection: bool = False
    # RELION Class3D classes; rows carry the class axis when there is more than one.
    n_classes: int = 1
    # Stable Fourier windows: the pixel axes are a physical class and the
    # M-step masks their tail past the logical window (WindowLogicalSizes).
    stable_window: bool = False
    # Union projection cache (ChunkStageTables.union_score_take): RELION's
    # native-unit division of the score rows, applied after the gather; 0 keeps
    # RECOVAR units.
    union_native_fft_size: int = 0
    # The M-step carry holds per-projection row sums instead of BPref volumes;
    # the pass backprojects them once after its chunk loop
    # (_backproject_projection_sums).
    presum_adjoint: bool = False


class ChunkRowArrays(NamedTuple):
    """One chunk's row-aligned tables and its runtime extents."""

    row_image_local: jax.Array  # int32 [C_R]
    # int32 [C_R]: the row's fine rotation; with K>1 classes its projection id
    # class * n_fine_rot + rotation; with streamed projections its cache slot.
    row_fine_rot: jax.Array
    row_log_prior: jax.Array  # float32 [C_R]
    row_mask_bits: jax.Array  # uint32 [C_R, n_mask_words], coarse-translation bitset per row
    row_mask_mode: jax.Array  # int8 [C_R]
    image_ids: jax.Array  # int32 [C_B], global image id, -1 when padded
    n_valid_rows: jax.Array  # int32 []
    n_valid_images: jax.Array  # int32 []
    segment_offsets: jax.Array  # int32 [C_B + 1], cell offsets
    image_row_start: jax.Array  # int64 [C_B], chunk-local first row of a slot
    image_row_count: jax.Array  # int64 [C_B], rows owned by a slot
    # K>1 only: the class axis of the chunk's rows.
    classes: "ChunkClassLayout | None" = None
    # More than one accumulator slot only: the slot-major M-step order.
    mstep: "ChunkMstepLayout | None" = None


class ChunkClassLayout(NamedTuple):
    """The class axis of one chunk (K>1; docs/development/resident_segments.md).

    An image slot's rows are class-major, so each (slot, class) pair owns a
    contiguous sub-segment ``s = slot * K + class`` of the slot's posterior
    segment: the per-class evidence and winner are reductions over it. The
    M-step order is :class:`ChunkMstepLayout`'s.
    """

    row_class: jax.Array  # int32 [C_R], 0 on padded rows
    row_segment: jax.Array  # int32 [C_R], slot * K + class; C_B * K on padded rows
    segment_offsets: jax.Array  # int32 [C_B * K + 1], cell offsets of each (slot, class)
    segment_row_start: jax.Array  # int64 [C_B * K], chunk-local first row of each (slot, class)


class ChunkMstepLayout(NamedTuple):
    """The M-step visits a chunk's rows slot-major, so each accumulator slot's blocks write one BPref pair.

    Slot ``a = class + K * slot_offset[unit]`` (docs/development/resident_segments.md).
    """

    row_slot: jax.Array  # int32 [C_R], 0 on padded rows


class ChunkStageOperands(NamedTuple):
    """One chunk's per-image operands, already padded to the image capacity."""

    score_input: jax.Array
    corr_img_score: jax.Array
    highres_xi2_half: jax.Array | None
    translation_prior: jax.Array
    # Exactly one reconstruction operand pair is populated. The per-chunk
    # preparation fills the pre-shifted ``[C_B, T, P]`` tiles; the once-per-half
    # preparation fills the unshifted ``[C_B, P]`` images T15's kernel takes,
    # with ``recon_weight`` set only in the exact-BPref configuration.
    shifted_recon: jax.Array | None
    shifted_noise: jax.Array | None
    recon_image: jax.Array | None
    recon_weight: jax.Array | None
    noise_image: jax.Array | None
    ctf2_over_nv_recon: jax.Array
    direct_ctf_rfloat_recon: jax.Array | None
    image_power_shells: jax.Array | None  # float64 [C_B, n_shells]
    relion_norm_high_shell: jax.Array
    raw_translated_wavg_rectangle: jax.Array
    raw_translated_wavg_for_atomic: jax.Array
    scale: jax.Array
    group_ids: jax.Array
    translation_sqdist_ang: jax.Array | None
    # int32 [C_B] optics-group row of each image, only with a [G, P] noise table.
    optics_groups: jax.Array | None = None
    # RELION --firstiter_cc only: the translated corrected score tile
    # [C_B, T, P_score] and 0.5 * |image|^2 per image (the evidence offset).
    score_shifted_cc: jax.Array | None = None
    cc_half_batch_norm: jax.Array | None = None
    # Tilt images (S4.2) only: each image's 1 / n_images of its particle. RELION divides a
    # subtomogram's noise and norm sums, not its backprojection, by the particle's image count
    # (acc_ml_optimiser_impl.h:3490-3491, :3512-3516; resident_tilts).
    image_noise_scale: jax.Array | None = None
    # CTF-premultiplied images only: the backprojected CTF weight [C_B, P]
    # (sparse_pass2_bucket_io.premultiplied_bpref_weights). None: ctf2_over_nv_recon
    # is both the backprojected weight and the noise statistics' CTF^2 / sigma2.
    bpref_ctf2_over_nv_recon: jax.Array | None = None


class CoarseNormalizationReuse(NamedTuple):
    """The coarse pass's float32 normalization, winner and Pmax, per image of the half.

    With ``--adaptive_oversampling 0`` RELION's fine pass keeps the coarse
    ``sum_weight`` and max (acc_ml_optimiser_impl.h:2868), keeps every weight
    (``significant_weight = sorted[0]``, :3590) and reports Pmax as the coarse
    ``max_weight / sum_weight`` (:3268-3269, :4223). See docs/math/zero_oversampling.md.
    """

    sum_weight: jax.Array  # float64 [n_images]
    max_posterior: jax.Array  # [n_images]
    winner_cell: jax.Array  # int64 [image capacity], segment-relative r_local * T + t


class WindowLogicalSizes(NamedTuple):
    """RELION's exact window sizes as device scalars, not program keys.

    The chunk programs are keyed on the physical pixel capacities in
    :class:`ChunkProgramSpec`; the logical current size and pixel counts the
    kernels stop at and the statistics cut at travel here, so a program is
    reused across the current sizes that share one physical class.
    """

    current_size: jax.Array  # int32 []
    recon_pixels: jax.Array  # int32 [], the logical prefix of the recon window
    rect_pixels: jax.Array  # int32 [], the logical prefix of the Wavg rectangle
    # float32 [], RELION's M-step adjoint radius; None keeps the spec's.
    mstep_max_r: jax.Array | None = None
    # int32 [], the weighted sums' image size (the norm cutoff); None is current_size.
    wsum_current_size: jax.Array | None = None


class ChunkStageTables(NamedTuple):
    """Iteration-global device tables every chunk of a half reads."""

    projection_score_cache: jax.Array
    projection_recon_cache: jax.Array
    projection_recon_abs2_cache: jax.Array
    mstep_grid: jax.Array
    coarse_parent_grid: jax.Array
    fine_translation_parent: jax.Array
    half_weights: jax.Array
    translation_angles: jax.Array
    full_to_compact: jax.Array
    noise_variance_for_noise: jax.Array
    shell_indices_noise: jax.Array
    exact_positions: jax.Array
    recon_pixel_indices: jax.Array
    relion_x_half_recon_indices: jax.Array
    shell_indices_half: jax.Array
    wavg_shell_indices: jax.Array
    wavg_scale_pixel_mask: jax.Array
    # Streamed projections: the global fine rotation id of each chunk-local
    # cache slot (rows then carry slots, not ids). None when the caches are the
    # per-iteration fine-grid caches, where the slot is the id.
    cache_slot_fine_rot: jax.Array | None = None
    # Zero oversampling only: the retained coarse normalization (None otherwise).
    coarse_reuse: CoarseNormalizationReuse | None = None
    # The logical window sizes; None takes the spec's (logical == physical).
    window_logical: WindowLogicalSizes | None = None
    # Union projection cache: projection_score_cache holds the union of the
    # score and reconstruction windows, and these take each window out of a
    # gathered row (the recon caches are then None). None for the three-cache
    # layout.
    union_score_take: jax.Array | None = None
    union_recon_take: jax.Array | None = None
    # VDAM's subtracted reference on an image window wider than the model sphere
    # (_sgd_residual_rows): the union rows' projections at the reconstruction
    # pixels the fine and wavg kernels zero, [projection, P_sgd], and those pixels'
    # positions in the reconstruction window. None when no such pixel exists.
    residual_sgd_cache: jax.Array | None = None
    residual_sgd_take: jax.Array | None = None


class ChunkPosterior(NamedTuple):
    """Stages 1-4 of one chunk."""

    row_posterior: jax.Array  # float32 [C_R, T]
    min_diff2: jax.Array  # real [C_B]
    class_log_z: jax.Array  # float64 [C_B]
    best_log_score: jax.Array  # float32 [C_B]
    best_cell_index: jax.Array  # int64 [C_B]
    max_posterior: jax.Array  # real [C_B]
    kernel_row_image_ids: jax.Array  # int32 [C_R], -1 on padded rows
    row_is_valid: jax.Array  # bool [C_R]
    # K>1 only: the (slot, class) sub-segment reductions.
    classes: "ChunkClassPosterior | None" = None


class ChunkClassPosterior(NamedTuple):
    """Each (image slot, class) sub-segment's log-Z and own winner, flat ``slot * K + class``."""

    log_z: jax.Array  # float64 [C_B * K], -inf for a class without candidates
    best_log_score: jax.Array  # float32 [C_B * K]
    best_cell_index: jax.Array  # int64 [C_B * K], sub-segment-relative r_local * T + t


class ChunkMstepCarry(NamedTuple):
    """Loop carry of the M-step block loop."""

    Ft_y: jax.Array
    Ft_ctf: jax.Array
    wavg_triplet_pixels: jax.Array  # float32 [C_B, P_rect, 3]
    noise_shells: jax.Array  # float64 [n_shells], [G, n_shells] with G optics groups
    a2_per_image: jax.Array  # real [C_B]
    xa_per_image: jax.Array  # real [C_B]
    # K>1 only: each image's scale sums, every class masked by its own
    # data_vs_prior_class > 3 (:func:`_fold_class_scale_sums`).
    scale_xa_per_image: jax.Array | None = None  # float64 [C_B]
    scale_aa_per_image: jax.Array | None = None  # float64 [C_B]


class MstepBlockInputs(NamedTuple):
    """Chunk-wide row arrays the M-step block program slices its block out of.

    ``row_fine_rot`` and ``projections`` are alternatives, and exactly one is
    populated: the global pass hands the fine-rotation ids and the program
    gathers the block's projections out of the per-iteration caches, while
    local search has no cacheable fine grid and hands the block's projections
    directly. ``None`` is a pytree structure, so the two callers key different
    programs without a flag.
    """

    row_image_local: jax.Array  # int32 [C_R]
    kernel_row_image_ids: jax.Array  # int32 [C_R]
    row_posterior: jax.Array  # float32 [C_R, T]
    row_fine_rot: jax.Array | None  # int32 [C_R]
    projections: tuple | None  # (proj, |proj|^2, M-step rotations) of one block
    # The rows [start, stop) of the accumulator slot these blocks accumulate;
    # the rows of a boundary block outside it get no weight. None where the
    # caller hands the block its rows directly (local search).
    class_row_range: jax.Array | None = None  # int32 [2]
    # Each slot's live rows in this order (:func:`_make_mstep_block_inputs`).
    slot_offsets: jax.Array | None = None  # int32 [n_slots + 1]
    # With ``spec.presum_adjoint``, each row's sum slot when it is not its projection id: a tilt slot's
    # visiting position, so the row's translation-block partials merge before one adjoint (relax#27).
    row_sum_ids: jax.Array | None = None  # int32 [C_R]
