"""Host inputs of one resident chunk: the capacity batch fetch and the reconstruction operands.

:func:`fetch_capacity_batch` reads a chunk's images into one reused capacity buffer
(``_CAPACITY_BATCH_BUFFERS``, one per thread), and
:func:`prepare_chunk_reconstruction_operands` prepares the chunk's per-chunk operands when the half's
operands are not resident. :func:`unshifted_chunk_operands` gathers a chunk from the resident per-image
operands instead, and :func:`verify_resident_chunk_operands` checks the two against each other (a
diagnostic arm).
"""

from __future__ import annotations

import logging
import threading
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass.bucket_io import _prepare_bucket_io, premultiplied_bpref_weights
from relax.fine_pass.resident_operands import (
    ResidentOperandsUnsupported,
    gather_resident_chunk_operands,
    prepare_resident_half_operands,
    require_unshifted_operand_support,
)
from relax.fine_pass.scoring import (
    _relion_native_fine_units,
    _relion_native_score_corr_img,
    _relion_powerclass_noise_terms,
)
from relax.fine_pass.wavg import _relion_cuda_translate_wavg_norm_images, image_power_shells
from relax.io.batch_fetch import fetch_indexed_batch
from relax.relion import ctf
from relax.relion.optics_groups import noise_rows, pixel_rows

logger = logging.getLogger(__name__)


# Relative band the racing shell-binning scatter is allowed in the verification
# arm: 12x the measured same-call spread of 5.8e-8, still far inside one
# float32 ulp of the accumulated shell power.
_RACING_SCATTER_RELATIVE_BAND = 7e-7


def pad_batch_to_capacity(values, capacity: int):
    """Repeat a host batch's first row up to ``capacity`` rows.

    Every per-chunk device program is keyed on its operand shapes, so a batch
    whose length is the chunk's *valid* image count traces a new program for
    every distinct occupancy. Padding the batch on the host, before anything
    is traced, gives one program per image-capacity class instead. The padded
    rows carry a duplicate image's real data and are zeroed after preparation
    by :func:`zero_padded_images`, which is itself capacity-shaped.
    """

    values = np.asarray(values)
    n = int(values.shape[0])
    if n == capacity:
        return values
    if n > capacity or n == 0:
        raise ValueError(f"cannot pad a batch of {n} rows to capacity {capacity}")
    pad = np.repeat(values[:1], capacity - n, axis=0)
    return np.concatenate([values, pad], axis=0)


# Host images are fetched in sub-batches below glibc's largest dynamic mmap
# threshold (32 MiB), so each fetch reuses heap memory, and are copied into one
# reused capacity-sized buffer. A 512-image fetch at box 800 (1.3 GB) otherwise
# faulted in fresh pages twice per chunk, for the gather and for the padding:
# about 95 s per half of host time in the EMPIAR-10202 pass-1 probe
# (py-spy, bigbox 14747301), against 5 s for the exact engine's per-image reads.
_FETCH_SUB_BATCH_BYTES = 24 << 20


# The buffer is per thread: two threads fetching same-shaped chunks at once must not share
# it (the removed --overlap_halves did, and each half scored the other's images,
# em_work/overlap_fix_20261012).
_CAPACITY_BATCH_BUFFERS = threading.local()


def fetch_capacity_batch(experiment_dataset, image_indices, capacity: int):
    """One chunk's host images padded to ``capacity`` and uploaded, with its CTF rows and indices.

    Returns ``(images, ctf_params, fetched_indices, padded_fetched_indices)``:
    the device images ``[capacity, ...]`` in fetched order, padded with the first
    fetched image as :func:`pad_batch_to_capacity` pads, the padded CTF rows,
    and the fetched indices without and with that padding. The values are the
    unpadded fetch's; only where the host copies live changes.
    """

    image_indices = np.asarray(image_indices)
    n = int(image_indices.shape[0])
    if n > capacity or n == 0:
        raise ValueError(f"cannot pad a batch of {n} rows to capacity {capacity}")
    buffer = None
    ctf_parts, fetched_parts = [], []
    start, step = 0, 1
    while start < n:
        images, ctf_params, fetched = fetch_indexed_batch(experiment_dataset, image_indices[start : start + step])
        images = np.asarray(images)
        if buffer is None:
            key = ((capacity,) + tuple(images.shape[1:]), images.dtype.str)
            if getattr(_CAPACITY_BATCH_BUFFERS, "key", None) != key:
                _CAPACITY_BATCH_BUFFERS.buffer = None
                _CAPACITY_BATCH_BUFFERS.buffer = np.empty(key[0], dtype=images.dtype)
                _CAPACITY_BATCH_BUFFERS.key = key
            buffer = _CAPACITY_BATCH_BUFFERS.buffer
            step = max(1, _FETCH_SUB_BATCH_BYTES // max(int(images[0].nbytes), 1))
        buffer[start : start + images.shape[0]] = images
        ctf_parts.append(np.asarray(ctf_params))
        fetched_parts.append(np.asarray(fetched))
        start += int(images.shape[0])
    buffer[n:] = buffer[0]
    # The upload is waited for: the buffer is refilled by the next chunk.
    device_images = jax.device_put(buffer)
    device_images.block_until_ready()
    fetched_indices = np.concatenate(fetched_parts)
    return (
        device_images,
        pad_batch_to_capacity(np.concatenate(ctf_parts), capacity),
        fetched_indices,
        pad_batch_to_capacity(fetched_indices, capacity),
    )


def reorder_permutation(fetched_indices, requested_indices, capacity: int) -> np.ndarray:
    """Host permutation from the fetched batch order to the table order.

    The dataset may return a batch in its own order. The compact engine
    reorders its bucket arrays to follow the fetch; the resident driver keeps
    the table's (RELION particle) order and permutes the operands, which is a
    pure gather and changes no arithmetic. Padded slots point at fetched row
    0; :func:`zero_padded_images` removes whatever they gathered.
    """

    fetched = np.asarray(fetched_indices).reshape(-1)
    requested = np.asarray(requested_indices).reshape(-1)
    n = int(requested.shape[0])
    position_of = {int(index): position for position, index in enumerate(requested.tolist())}
    if len(position_of) != n:
        raise ValueError("a chunk must not request the same image twice")
    order = np.zeros(capacity, dtype=np.int32)
    seen = np.zeros(n, dtype=bool)
    for slot, dataset_index in enumerate(fetched[:n].tolist()):
        position = position_of.get(int(dataset_index))
        if position is None:
            raise ValueError(f"the dataset returned image {dataset_index}, which was not requested")
        order[position] = slot
        seen[position] = True
    if not bool(seen.all()):
        raise ValueError("the dataset did not return every requested image")
    return order


class _ChunkOperandRowInputs(NamedTuple):
    """The per-chunk operands that are permuted into row order and zero-padded.

    Each optional field is ``None`` on the paths that do not produce it;
    ``None`` is a pytree structure, so those paths key their own program
    rather than carry a dead operand.
    """

    score_input: jax.Array
    corr_img_score: jax.Array
    highres_xi2_half: jax.Array | None
    shifted_recon: jax.Array
    shifted_noise: jax.Array
    ctf2_over_nv_recon: jax.Array
    direct_ctf_rfloat_recon: jax.Array | None
    processed_score_half_for_noise: jax.Array
    relion_norm_high_shell: jax.Array | None
    raw_translated_wavg_rectangle: jax.Array
    bpref_ctf2_over_nv_recon: jax.Array | None = None


@partial(jax.jit, static_argnames=("image_capacity", "n_fine_trans"))
def _chunk_operand_rows(
    arrays: _ChunkOperandRowInputs,
    permutation: jax.Array,
    valid_images: jax.Array,
    exact_positions: jax.Array,
    *,
    image_capacity: int,
    n_fine_trans: int,
) -> tuple:
    """Permute a chunk's operands into row order and zero the padded slots.

    Same statements, same order, same dtypes as the loose dispatch: two
    reshapes, ten permutation gathers, ten ``where`` masks and one rectangle
    gather. None of it is arithmetic, so no value can move; what leaves the
    host is the dispatch count. Eagerly, ``values[permutation]`` is five
    primitives rather than one, because JAX normalizes a fancy index
    (``add``, ``broadcast_in_dim``, ``select_n``) before every gather, and the
    resident local pass runs this once per chunk.
    """

    def take(values):
        return None if values is None else zero_padded_images(
            values[permutation], valid_images
        )

    raw_translated_wavg_rectangle = take(arrays.raw_translated_wavg_rectangle)
    return (
        take(arrays.score_input),
        take(arrays.corr_img_score),
        take(arrays.highres_xi2_half),
        take(arrays.shifted_recon.reshape(image_capacity, n_fine_trans, -1)),
        take(arrays.shifted_noise.reshape(image_capacity, n_fine_trans, -1)),
        take(arrays.ctf2_over_nv_recon),
        take(arrays.direct_ctf_rfloat_recon),
        take(arrays.processed_score_half_for_noise),
        take(arrays.relion_norm_high_shell),
        raw_translated_wavg_rectangle,
        raw_translated_wavg_rectangle[:, :, exact_positions],
        take(arrays.bpref_ctf2_over_nv_recon),
    )


def zero_padded_images(values, valid_images):
    """Zero the padded image slots of a capacity-shaped operand.

    ``valid_images`` is a capacity-shaped bool, so this traces one program per
    (capacity, trailing shape) pair regardless of how many slots are valid.
    """

    values = jnp.asarray(values)
    mask = jnp.asarray(valid_images, dtype=bool).reshape((-1,) + (1,) * (values.ndim - 1))
    return jnp.where(mask, values, jnp.zeros((), dtype=values.dtype))


def chunk_unshifted_operands_supported(
    bucket_io_kwargs,
    *,
    window_indices,
    recon_window_indices,
    firstiter_cc,
    relion_native_fine_units,
    relion_exact_bpref_operands,
    requested: bool,
) -> bool:
    """Whether a chunk can take its own images' unshifted operands (:func:`unshifted_chunk_operands`);
    ``requested`` is ``EngineTuning.resident_operands``.

    The checks :func:`prepare_resident_half_operands` makes, before any image
    is prepared, so the chunks are sized for the operand family they will use.
    The ``--firstiter_cc`` iteration scores translated normalized-CC tiles,
    which only the translated preparation builds.
    """

    if not requested or firstiter_cc:
        return False
    try:
        require_unshifted_operand_support(
            bucket_io_kwargs, window_indices=window_indices, recon_window_indices=recon_window_indices
        )
        if relion_native_fine_units and not relion_exact_bpref_operands:
            raise ResidentOperandsUnsupported(
                "native-unit fine scores without RELION's RFLOAT CTF operand"
            )
    except ResidentOperandsUnsupported as reason:
        logger.info("Resident pass-2 chunks keep the translated tiles: %s", reason)
        return False
    return True


def unshifted_chunk_operands(
    experiment_dataset,
    image_indices,
    *,
    image_capacity: int,
    bucket_io_kwargs,
    window_indices,
    recon_window_indices,
    rect_indices_device,
    exact_positions_device,
    translation_angles,
    noise_shell_indices_half,
    n_noise_shells: int,
    image_shape,
    current_size,
    n_fine_trans: int,
    accumulate_noise,
    source_faithful_spectrum_norm: bool,
    fine_translation_prior_2d,
    scale_corrections_np,
    group_ids_np,
    optics_groups_np,
    precision_policy,
    use_exact_relion_gaussian: bool = True,
    relion_native_fine_units: bool = False,
):
    """One chunk's unshifted per-image operands, gathered at the chunk's image capacity.

    The once-per-half preparation (T16, :func:`prepare_resident_half_operands`)
    run for the chunk's own images, then its chunk gather: the M-step
    translates these operands inside T15's translate-and-sum kernel, which
    skips zero weights, instead of reducing a pre-shifted ``[images,
    translations, pixels]`` tile. Per chunk, not per half, so the operands
    never scale with the particle count. The padded slots carry -1 and the
    gather zeroes them, as the tile path's capacity mask did. Raises
    :class:`ResidentOperandsUnsupported` for a configuration the preparation
    does not cover.
    """

    operands = prepare_resident_half_operands(
        experiment_dataset,
        image_indices,
        bucket_io_kwargs=bucket_io_kwargs,
        window_indices=window_indices,
        recon_window_indices=recon_window_indices,
        wavg_rect_indices=rect_indices_device,
        noise_shell_indices_half=noise_shell_indices_half,
        n_noise_shells=int(n_noise_shells),
        image_shape=image_shape,
        current_size=current_size,
        n_fine_trans=int(n_fine_trans),
        use_exact_relion_gaussian=bool(use_exact_relion_gaussian),
        accumulate_noise=accumulate_noise,
        source_faithful_spectrum_norm=bool(source_faithful_spectrum_norm),
        fine_translation_prior_2d=fine_translation_prior_2d,
        scale_corrections_np=scale_corrections_np,
        group_ids_np=group_ids_np,
        precision_policy=precision_policy,
        image_batch_size=int(image_capacity),
        # One batch covers the chunk, so there is nothing to read ahead.
        prefetch_depth=0,
        optics_groups_np=optics_groups_np,
        relion_native_fine_units=bool(relion_native_fine_units),
        log_summary=False,
    )
    n_images = int(np.asarray(image_indices).shape[0])
    image_slots = np.full(int(image_capacity), -1, dtype=np.int32)
    image_slots[:n_images] = np.arange(n_images, dtype=np.int32)
    return gather_resident_chunk_operands(
        operands,
        image_slots,
        translation_angles=translation_angles,
        rect_indices=rect_indices_device,
        exact_positions=exact_positions_device,
        image_shape=image_shape,
    )


def prepare_chunk_reconstruction_operands(
    *,
    chunk,
    image_indices,
    experiment_dataset,
    bucket_io_kwargs,
    windowed_prepare,
    recon_window_indices,
    score_window_indices,
    fine_translation_prior_2d,
    score_real_dtype,
    n_fine_trans,
    n_recon_windowed,
    image_shape,
    current_size,
    use_exact_relion_gaussian,
    accumulate_noise,
    source_faithful_spectrum_norm,
    relion_score_translation_angles,
    rect_indices_device,
    exact_positions_device,
    scale_corrections_np,
    group_ids_np,
    optics_groups_np=None,
    relion_native_fine_units=False,
    normalized_cc=False,
    noise_shell_indices_half=None,
    n_noise_shells=None,
):
    """Build one chunk's translated reconstruction, noise and Wavg tiles.

    The oracle path, kept selectable by
    ``RELAX_SPARSE_PASS2_RESIDENT_OPERANDS=0``. These tiles carry the
    ``(images, translations, pixels)`` axis, so they are the one operand family
    that cannot be kept resident for a whole half (17 GiB at the hp3 state);
    they are rebuilt per chunk from the same :func:`_prepare_bucket_io` call,
    with the same keyword arguments, that the compact engine makes per bucket.
    The default path instead keeps the *unshifted* per-image operands resident
    (:mod:`recovar.em.sparse_pass2.resident_operands`) and lets T15's kernel
    apply the translations inside the M-step reduction, so no tile is built at
    all.

    ``relion_native_fine_units`` gives the score operands in RELION's native
    FFT units exactly as :func:`prepare_resident_half_operands` does: the score
    image divided by N**2 and RELION's native ``corr_img``. Resident local
    search calls this without it and keeps RECOVAR units.

    ``normalized_cc`` (RELION's ``--firstiter_cc`` iteration) also returns the
    compact engine's normalized-CC operands: the translated corrected score
    tile ``[C_B, T, P]`` and half the unweighted image power, the score offset
    the compact engine reports evidence with. ``corr_img_score`` is then the
    CC pixel weight the same call returns.

    Shape stability. The batch handed to ``_prepare_bucket_io`` is padded on
    the host to the chunk's image capacity before anything is traced, so every
    chunk of one capacity class runs the same program instead of one program
    per distinct occupancy. The padded slots carry a duplicate image's real
    data through preparation and are zeroed afterwards by a capacity-shaped
    mask. Nothing downstream reads them: their posterior rows are zero and
    their image ids are -1.
    """

    image_indices = np.asarray(image_indices)

    batch_images, padded_ctf_params, fetched_indices, padded_fetched_indices = fetch_capacity_batch(
        experiment_dataset, image_indices, chunk.image_capacity
    )
    order = reorder_permutation(fetched_indices, image_indices, chunk.image_capacity)
    prepared = _prepare_bucket_io(
        experiment_dataset,
        batch_images,
        padded_ctf_params,
        padded_fetched_indices,
        return_direct_scoring_io=True,
        **bucket_io_kwargs,
    )
    (
        _shifted_score_half,
        shifted_recon_half,
        batch_norm,
        ctf2_over_nv_half,
        ctf2_over_nv_half_with_dc,
        shifted_score_half_with_dc,
        processed_score_half_for_noise,
        shifted_corrected_score_half,
        direct_score_input,
        _direct_preprocessed_score_input,
        _direct_pixel_correction,
        _direct_preprocess_normalization_factors,
        _direct_integer_pre_shifts,
        _direct_batch_image_corrections,
        direct_batch_scale_corrections,
        _direct_inverse_noise_half,
        direct_ctf_rfloat_half,
    ) = prepared

    gather_recon = jnp.asarray(recon_window_indices, dtype=jnp.int32)
    if windowed_prepare:
        shifted_recon = shifted_recon_half
        ctf2_over_nv_recon = ctf2_over_nv_half_with_dc
        shifted_noise = shifted_score_half_with_dc
    else:
        shifted_recon = shifted_recon_half[:, gather_recon]
        ctf2_over_nv_recon = ctf2_over_nv_half_with_dc[:, gather_recon]
        shifted_noise = shifted_score_half_with_dc[:, gather_recon]
    direct_ctf_rfloat_recon = (
        None if direct_ctf_rfloat_half is None else direct_ctf_rfloat_half[:, gather_recon]
    )
    bpref_ctf2_over_nv_recon = None
    ctf_premultiplied = ctf.premultiplied_ctf_rows(experiment_dataset, padded_fetched_indices, image_shape)
    if ctf_premultiplied is not None:
        # The same weights prepare_unshifted_bucket_operands gives the resident operands.
        _, premultiplied_ctf_weight = premultiplied_bpref_weights(
            jnp.asarray(direct_ctf_rfloat_half, dtype=ctf2_over_nv_recon.dtype),
            _direct_inverse_noise_half,
            (
                jnp.asarray(direct_batch_scale_corrections, dtype=ctf2_over_nv_recon.dtype)
                if bucket_io_kwargs["scale_corrections"] is not None
                else None
            ),
        )
        bpref_ctf2_over_nv_recon = jnp.where(
            jnp.asarray(ctf_premultiplied)[:, None], premultiplied_ctf_weight[:, gather_recon], ctf2_over_nv_recon
        )

    # Score-side operands come from this same call. The driver used to make a
    # second pass over the whole half for them; the preparation is per-image
    # pure (bitwise at batch 256, 128, 32, 13 and 1), so taking them here is
    # the same arithmetic with one call per chunk instead of two per image.
    if windowed_prepare:
        score_input = direct_score_input
        corr_img_score = ctf2_over_nv_half
    else:
        gather_score = jnp.asarray(score_window_indices, dtype=jnp.int32)
        score_input = direct_score_input[:, gather_score]
        corr_img_score = ctf2_over_nv_half[:, gather_score]
    if relion_native_fine_units:
        if direct_ctf_rfloat_half is None:
            raise ValueError("native-unit fine scores require RELION's RFLOAT CTF operand")
        # Same operands as the once-per-half preparation and the compact
        # engine: RELION's native corr_img (DC zeroed for half-spectrum
        # scoring) in the score window, and the unshifted score image divided
        # by N**2, which the kernel translates in-kernel.
        native_corr_img_half = _relion_native_score_corr_img(
            pixel_rows(
                noise_rows(
                    bucket_io_kwargs["noise_variance_half"],
                    bucket_io_kwargs.get("noise_optics_groups"),
                    padded_fetched_indices,
                )
            ),
            direct_ctf_rfloat_half,
            image_shape,
            (
                jnp.asarray(direct_batch_scale_corrections, dtype=jnp.float32)[:, None]
                if bucket_io_kwargs["scale_corrections"] is not None
                else None
            ),
            zero_dc=bool(bucket_io_kwargs["half_spectrum_scoring"]),
        )
        if score_window_indices is not None:
            native_corr_img_half = native_corr_img_half[
                :, jnp.asarray(score_window_indices, dtype=jnp.int32)
            ]
        corr_img_score = native_corr_img_half.astype(corr_img_score.dtype)
        score_input = _relion_native_fine_units(score_input, int(np.prod(image_shape)))

    highres_xi2_half, relion_norm_high_shell = _relion_powerclass_noise_terms(
        processed_score_half_for_noise,
        image_shape=image_shape,
        current_size=current_size,
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        accumulate_noise=accumulate_noise,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
    )
    raw_translated_wavg_rectangle = _relion_cuda_translate_wavg_norm_images(
        processed_score_half_for_noise,
        relion_score_translation_angles,
        rect_indices_device,
        image_shape,
    )

    permutation = jnp.asarray(order, dtype=jnp.int32)
    valid_images = jnp.asarray(
        np.arange(chunk.image_capacity) < chunk.n_valid_images, dtype=bool
    )
    score_shifted_cc = cc_half_batch_norm = None
    if normalized_cc:
        # The compact engine's operands (the deleted sparse_pass2_bucketed.py, the
        # normalized_cc branch): the translated corrected score tile in the
        # score window, and the -0.5 * |image|^2 evidence offset.
        tile = shifted_corrected_score_half.reshape(chunk.image_capacity, int(n_fine_trans), -1)
        if not windowed_prepare:
            tile = tile[:, :, jnp.asarray(score_window_indices, dtype=jnp.int32)]
        score_shifted_cc = zero_padded_images(tile[permutation], valid_images)
        cc_half_batch_norm = zero_padded_images(
            (0.5 * jnp.reshape(batch_norm, (chunk.image_capacity,)).real)[permutation], valid_images
        )

    (
        score_input,
        corr_img_score,
        highres_xi2_half,
        shifted_recon,
        shifted_noise,
        ctf2_over_nv_recon,
        direct_ctf_rfloat_recon,
        processed_image_half,
        relion_norm_high_shell,
        raw_translated_wavg_rectangle,
        raw_translated_wavg_for_atomic,
        bpref_ctf2_over_nv_recon,
    ) = _chunk_operand_rows(
        _ChunkOperandRowInputs(
            score_input=score_input,
            corr_img_score=corr_img_score,
            highres_xi2_half=highres_xi2_half,
            shifted_recon=shifted_recon,
            shifted_noise=shifted_noise,
            ctf2_over_nv_recon=ctf2_over_nv_recon,
            direct_ctf_rfloat_recon=direct_ctf_rfloat_recon,
            processed_score_half_for_noise=processed_score_half_for_noise,
            relion_norm_high_shell=relion_norm_high_shell,
            raw_translated_wavg_rectangle=raw_translated_wavg_rectangle,
            bpref_ctf2_over_nv_recon=bpref_ctf2_over_nv_recon,
        ),
        permutation,
        valid_images,
        exact_positions_device,
        image_capacity=int(chunk.image_capacity),
        n_fine_trans=int(n_fine_trans),
    )
    if int(shifted_recon.shape[-1]) != int(n_recon_windowed):
        raise ValueError(
            "reconstruction tile pixel count does not match the reconstruction window: "
            f"{int(shifted_recon.shape[-1])} vs {int(n_recon_windowed)}"
        )

    # Padded slots keep scale 1 so the Wavg kernel never divides by zero; their
    # posterior is zero, so the value is never observable.
    scale_chunk = np.ones(chunk.image_capacity, dtype=np.float32)
    if scale_corrections_np is not None:
        scale_chunk[:chunk.n_valid_images] = np.asarray(
            scale_corrections_np[image_indices], dtype=np.float32
        )
    group_ids_chunk = np.full(chunk.image_capacity, -1, dtype=np.int32)
    if group_ids_np is not None:
        group_ids_chunk[:chunk.n_valid_images] = np.asarray(
            group_ids_np[image_indices], dtype=np.int32
        )
    optics_groups_chunk = None
    if optics_groups_np is not None:
        optics_groups_chunk = np.zeros(chunk.image_capacity, dtype=np.int32)
        optics_groups_chunk[:chunk.n_valid_images] = np.asarray(
            optics_groups_np[image_indices], dtype=np.int32
        )

    translation_prior = jnp.asarray(
        np.zeros((chunk.image_capacity, int(n_fine_trans)), dtype=np.float32)
        if fine_translation_prior_2d is None
        else pad_batch_to_capacity(
            np.asarray(fine_translation_prior_2d)[image_indices], chunk.image_capacity
        ),
        dtype=score_real_dtype,
    )

    if noise_shell_indices_half is None or n_noise_shells is None:
        raise ValueError("the chunk operands need the noise-shell binning of the packed half")
    return {
        "score_input": score_input,
        "corr_img_score": corr_img_score,
        "highres_xi2_half": highres_xi2_half,
        "translation_prior": zero_padded_images(translation_prior, valid_images),
        "shifted_recon": shifted_recon,
        "shifted_noise": shifted_noise,
        "ctf2_over_nv_recon": ctf2_over_nv_recon,
        "direct_ctf_rfloat_recon": direct_ctf_rfloat_recon,
        "image_power_shells": image_power_shells(
            processed_image_half,
            jnp.asarray(noise_shell_indices_half, dtype=jnp.int32),
            shell_count=int(n_noise_shells),
        ),
        "relion_norm_high_shell": relion_norm_high_shell,
        "raw_translated_wavg_rectangle": raw_translated_wavg_rectangle,
        "raw_translated_wavg_for_atomic": raw_translated_wavg_for_atomic,
        "scale": jnp.asarray(scale_chunk),
        "group_ids": jnp.asarray(group_ids_chunk),
        "optics_groups": None if optics_groups_chunk is None else jnp.asarray(optics_groups_chunk),
        "score_shifted_cc": score_shifted_cc,
        "cc_half_batch_norm": cc_half_batch_norm,
        "bpref_ctf2_over_nv_recon": bpref_ctf2_over_nv_recon,
    }


def verify_resident_chunk_operands(
    resident_recon,
    reference_recon,
    *,
    translation_angles,
    recon_pixel_indices,
    image_shape,
    n_recon_pixels: int,
    bpref_recon_operand: bool,
    label: str,
) -> None:
    """Prove one real chunk's resident operands equal the per-chunk preparation.

    The two paths differ in *when* the translation is applied, so the check
    translates the resident per-image operands with the primitives
    ``_prepare_bucket_io`` uses and compares against its own pre-shifted tiles.
    A mismatch raises: this runs only in a diagnostic arm, and a silent
    difference here would be a changed reconstruction operand.
    """

    from relax.cuda import translate as cuda_translate

    image_capacity = int(np.asarray(reference_recon["shifted_recon"]).shape[0])
    angles = jnp.asarray(translation_angles, dtype=jnp.float32)
    indices = jnp.asarray(recon_pixel_indices, dtype=jnp.int32)
    if bpref_recon_operand:
        translated_recon = cuda_translate.relion_translate_bpref_f32(
            jnp.asarray(resident_recon["recon_image"], dtype=jnp.complex64),
            jnp.asarray(resident_recon["recon_weight"], dtype=jnp.float32),
            angles,
            indices,
            image_shape,
        )
    else:
        translated_recon = cuda_translate.relion_translate_score_f32(
            jnp.asarray(resident_recon["recon_image"], dtype=jnp.complex64),
            angles,
            indices,
            image_shape,
        )
    translated_noise = cuda_translate.relion_translate_score_f32(
        jnp.asarray(resident_recon["noise_image"], dtype=jnp.complex64),
        angles,
        indices,
        image_shape,
    )
    n_fine_trans = int(angles.shape[0])
    checks = {
        "shifted_recon": (
            np.asarray(translated_recon).reshape(image_capacity, n_fine_trans, n_recon_pixels),
            np.asarray(reference_recon["shifted_recon"]),
        ),
        "shifted_noise": (
            np.asarray(translated_noise).reshape(image_capacity, n_fine_trans, n_recon_pixels),
            np.asarray(reference_recon["shifted_noise"]),
        ),
    }
    for name in (
        "score_input",
        "corr_img_score",
        "highres_xi2_half",
        "translation_prior",
        "ctf2_over_nv_recon",
        "direct_ctf_rfloat_recon",
        "image_power_shells",
        "relion_norm_high_shell",
        "raw_translated_wavg_rectangle",
        "raw_translated_wavg_for_atomic",
        "scale",
        "group_ids",
        "optics_groups",
    ):
        expected = reference_recon.get(name)
        actual = resident_recon.get(name)
        if expected is None and actual is None:
            continue
        if (expected is None) != (actual is None):
            raise AssertionError(f"resident operand {name} presence differs from the per-chunk path")
        checks[name] = (np.asarray(actual), np.asarray(expected))

    # ``relion_norm_high_shell`` bins the image power with a scatter-add over
    # duplicate indices. That races: two calls on the same array in one process
    # differ by about 6e-8 relative, so no two preparations of it are bitwise,
    # including two of the per-chunk path. It is checked against that spread.
    racing_scatter = ("relion_norm_high_shell",)
    mismatched = []
    for name, (actual, expected) in checks.items():
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            mismatched.append(f"{name}: {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}")
            continue
        if np.array_equal(actual, expected):
            continue
        differing = int(np.count_nonzero(actual != expected))
        left = actual.astype(np.complex128)
        right = expected.astype(np.complex128)
        worst = float(np.max(np.abs(left - right)))
        relative = float(
            np.max(np.abs(left - right) / np.maximum(np.abs(right), 1e-30))
        )
        if name in racing_scatter and relative <= _RACING_SCATTER_RELATIVE_BAND:
            logger.info(
                "Resident pass-2 operand verification on %s: %s is inside its racing "
                "scatter-add band (%d/%d cells, max |delta| %.3e, max relative %.3e)",
                label, name, differing, actual.size, worst, relative,
            )
            continue
        mismatched.append(
            f"{name}: {differing}/{actual.size} cells differ, max |delta| {worst:.3e}, "
            f"max relative {relative:.3e}"
        )
    if mismatched:
        raise AssertionError(
            f"resident per-half operands differ from the per-chunk preparation on {label}: "
            + "; ".join(mismatched)
        )
    logger.info(
        "Resident pass-2 operand verification on %s: %d operands equal to the per-chunk "
        "preparation (BPref reconstruction operand=%d)",
        label,
        len(checks),
        int(bpref_recon_operand),
    )
