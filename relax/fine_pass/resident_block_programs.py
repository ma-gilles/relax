"""The resident M-step's device programs with a pixel axis, one per (block rows, pixel count) pair.

Each takes one fixed-size row block of a chunk: the posterior-weighted translate-and-sum
(:func:`resident_block_weighted_sums`, or the flat-row kernel), the CTF-weighted probabilities, the
residual, the noise and norm terms, the Wavg algebraic terms, and the Wavg image power of the rectangular
window.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from relax.fine_pass.local_backprojection import (
    compute_local_ctf_sums_from_probs_sum_t,
    compute_local_weighted_sums,
)
from relax.fine_pass.resident_statistics import segment_sum_by_image
from relax.fine_pass.scoring import to_relion_native_fine_units
from relax.fine_pass.wavg import relion_wavg_shifted_power
from relax.projection.projection import compute_noise_block, compute_noise_block_per_optics_group
from relax.relion.optics_groups import pixel_rows


def _flat_row_norm_and_scale_terms(
    proj,
    proj_abs2,
    summed_masked,
    ctf_probs,
    noise_variance,
    *,
    pixel_mask=None,
):
    """Per-row ``A2`` and ``XA`` partials in the flat candidate-row layout.

    Term for term this is the rectangular per-image norm residual (the test oracle
    in ``tests/oracles/projection.py``; and,
    with ``pixel_mask``, its scale-correction sibling) with the ``(B, R, P)``
    rotation axis folded into the row axis: the same masks, the same
    ``noise_variance`` broadcast and the same natural dtype promotion, summed
    over pixels only. The caller sums rows into images and forms
    ``A2 - 2 * XA`` afterwards, exactly as the host does after its
    ``axis=(1, 2)`` reduction.
    """

    ctf_has_mass = ctf_probs != 0.0
    cross_has_mass = summed_masked != 0.0
    if pixel_mask is not None:
        pixel_mask = jnp.asarray(pixel_mask, dtype=bool).reshape(-1)
        ctf_has_mass = ctf_has_mass & pixel_mask[None, :]
        cross_has_mass = cross_has_mass & pixel_mask[None, :]
    noise_variance = pixel_rows(noise_variance)
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance, 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2 * ctf_probs_raw, 0.0)
    cross_terms = jnp.where(cross_has_mass, proj * jnp.conj(summed_masked), 0.0)
    xa_terms = noise_variance * cross_terms.real
    return jnp.sum(a2_terms, axis=1), jnp.sum(xa_terms, axis=1)


@jax.jit
def resident_block_weighted_sums(
    row_posterior,  # float32 [block, T]
    row_image_local,  # int32 [block]
    shifted_recon,  # complex [C_B, T, P]
    shifted_noise,  # complex [C_B, T, P]
    ctf2_over_nv_recon,  # real [C_B, P]
):
    """Flat-row twin of the bucket layout's ``compute_local_mstep_sums`` pair (``scripts/lib/bucketed_pass2.py``).

    ``compute_local_weighted_sums`` contracts ``(B, R, T) x (B, T, N)`` with
    ``Precision.HIGHEST``; the flat-row form gathers each row's image tile and
    contracts ``(Q, T) x (Q, T, N)`` at the same precision, so the per-cell
    products and the translation reduction order are unchanged.
    ``compute_local_ctf_sums_from_probs_sum_t`` is reused verbatim, including
    its ``!= 0`` mass predicate. ``summed`` feeds the x-half BPref numerator;
    ``summed_masked`` is the noise operand the host tail builds from
    ``shifted_noise``.
    """

    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    weights = jnp.asarray(row_posterior)
    recon_tiles = jnp.asarray(shifted_recon)[row_image_local]
    noise_tiles = jnp.asarray(shifted_noise)[row_image_local]
    # ``compute_local_weighted_sums`` is called verbatim with a singleton
    # rotation axis, so the contraction keeps its pinned
    # ``Precision.HIGHEST``; only the gathered per-row tile replaces the
    # shared per-image tile.
    summed = compute_local_weighted_sums(weights[:, None, :], recon_tiles)[:, 0, :]
    summed_masked = compute_local_weighted_sums(weights[:, None, :], noise_tiles)[:, 0, :]
    probs_sum_t = jnp.sum(weights, axis=-1)
    # The rectangular helper takes ``(B, R)`` rotation sums against one
    # ``(B, P)`` CTF row per image. In the flat-row layout each row carries its
    # own gathered CTF row, so the call is made with a singleton rotation axis;
    # the ``!= 0`` mass predicate and the product order are unchanged.
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(
        probs_sum_t[:, None],
        jnp.asarray(ctf2_over_nv_recon)[row_image_local],
    )[:, 0, :]
    return summed, summed_masked, ctf_probs, probs_sum_t


def resident_block_weighted_sums_kernel(
    row_posterior,  # float32 [block, T]
    row_image_ids,  # int32 [block], -1 on a padded row
    row_image_local,  # int32 [block], the chunk-local slot the XLA gather uses
    recon_image,  # complex64 [C_B, P], unshifted
    recon_weight,  # float32 [C_B, P] (BPref weighted CTF) or None
    noise_image,  # complex64 [C_B, P], unshifted
    ctf2_over_nv_recon,  # float32 [C_B, P]
    recon_pixel_indices,  # int32 [P], centered packed-half indices
    translation_angles,  # float32 [T, 2]
    *,
    image_shape,
    n_recon_pixels: int,
):
    """T15's translate-and-sum kernel in place of the gathered-tile reduction.

    Same four outputs as :func:`resident_block_weighted_sums`, from the
    *unshifted* per-image operands: the kernel applies each translation inside
    the reduction with the phase arithmetic of the primitive that built the
    tile, so no ``[images, translations, pixels]`` tile exists. ``recon_weight``
    selects the convention pairing production uses -- BPref for the
    reconstruction operand, score for the noise operand -- and matches how
    ``prepare_bucket_io`` builds the two shifted arrays.

    Rows are bounded by their image id rather than by a row count: a padded row
    carries ``-1`` and the kernel writes it as zeros, which is the value the XLA
    path reaches through a zero posterior.

    ``ctf_probs``. The kernel can also produce it, from its own sequential
    ``probs_sum_t``. That mass is bitwise against ``jnp.sum`` only where XLA
    happens to reduce the translations sequentially too: it does at
    ``[2048, 21]``, and it does not at ``[64, 21]``, where the fourth output
    moves 53% of the cells by up to 1 relative ulp. ``ctf_probs`` feeds the
    ``Ft_ctf`` accumulator and the Wavg and noise terms, so the default keeps
    the XLA statement the per-chunk path used, on the same gathered CTF row,
    and the kernel's own mass is never used.
    """

    from relax.cuda import translate as cuda_translate

    row_posterior = jnp.asarray(row_posterior, dtype=jnp.float32)
    block_rows = int(row_posterior.shape[0])
    outputs = cuda_translate.relion_translate_sum_flat_rows_f32(
        jnp.asarray(recon_image, dtype=jnp.complex64),
        jnp.asarray(noise_image, dtype=jnp.complex64),
        jnp.asarray(row_image_ids, dtype=jnp.int32),
        row_posterior,
        jnp.asarray(translation_angles, dtype=jnp.float32),
        jnp.asarray(recon_pixel_indices, dtype=jnp.int32),
        jnp.asarray(block_rows, dtype=jnp.int32),
        jnp.asarray(int(n_recon_pixels), dtype=jnp.int32),
        recon_weight=(
            None if recon_weight is None else jnp.asarray(recon_weight, dtype=jnp.float32)
        ),
        ctf2_over_nv=None,
        image_shape=tuple(int(size) for size in image_shape),
    )
    summed, summed_masked, _kernel_mass = outputs
    ctf_probs, probs_sum_t = resident_block_ctf_probs(
        row_posterior, row_image_local, ctf2_over_nv_recon
    )
    return summed, summed_masked, ctf_probs, probs_sum_t


@jax.jit
def resident_block_ctf_probs(row_posterior, row_image_local, ctf2_over_nv_recon):
    """``ctf_probs`` and its mass, exactly as :func:`resident_block_weighted_sums` forms them.

    One program, so the kernel path costs one dispatch here rather than three.
    The statements, the gather and the ``!= 0`` mass predicate are the tile
    path's own, which is what makes the two paths bitwise on this output.
    """

    probs_sum_t = jnp.sum(jnp.asarray(row_posterior), axis=-1)
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(
        probs_sum_t[:, None],
        jnp.asarray(ctf2_over_nv_recon)[jnp.asarray(row_image_local, dtype=jnp.int32)],
    )[:, 0, :]
    return ctf_probs, probs_sum_t


@partial(jax.jit, static_argnums=1, donate_argnums=0)
def relion_native_fine_units_in_place(values, fft_size):
    """:func:`to_relion_native_fine_units` as one program that reuses the input buffer.

    The whole-grid projection cache is several GiB; the eager form holds the
    complex64 input, two float64 part arrays and the output at once (EMPIAR-10097
    VDAM at healpix 3, current size 56: a 7.77 GiB float64 temporary ran the
    device out of memory, job 14422223). The fused program has no float64
    temporaries and writes into the donated input. Same elementwise statements,
    so the same values.
    """

    return to_relion_native_fine_units(values, fft_size)


@jax.jit
def resident_block_residual(summed, probs_sum_t, proj, ctf2_over_nv_recon, row_image_local):
    """VDAM's BPref numerator: the weighted image sum minus the weighted CTF'd projection.

    RELION's ``--grad`` backprojection (``cuda_kernel_backproject3D_SGD``,
    acc/cuda/cuda_kernels/BP.cuh:406-560) accumulates ``(shift(img) - ctf * proj) * w``
    per translation. The projection does not depend on the translation, so the sum
    is ``summed - (sum_t w) * ctf^2/sigma2 * proj``: the same statement as the exact
    local engine's residual (``local_big_jit``), with its ``!= 0`` mass predicate.
    """

    frefctf_weighted = jnp.asarray(proj) * jnp.asarray(ctf2_over_nv_recon)[
        jnp.asarray(row_image_local, dtype=jnp.int32)
    ]
    probs_sum_t = jnp.asarray(probs_sum_t)
    frefctf_delta = jnp.where(
        probs_sum_t[:, None] != 0.0,
        probs_sum_t[:, None] * frefctf_weighted,
        0.0,
    )
    return summed - frefctf_delta


@partial(jax.jit, static_argnames=("n_shells", "image_capacity", "float32_bucketed_image_sums"))
def resident_block_noise_and_norm(
    proj,  # complex [block, P]
    proj_abs2,  # real [block, P]
    summed_masked,  # complex [block, P]
    ctf_probs,  # real [block, P]
    noise_variance,  # real [P], or [G, P] per optics group
    shell_indices,  # int32 [P]
    row_image_local,  # int32 [block]
    row_optics_groups=None,  # int32 [block] with a [G, P] noise table
    *,
    n_shells: int,
    image_capacity: int,
    float32_bucketed_image_sums: bool = True,
):
    """One row block's noise shells plus its per-image ``A2``/``XA`` partials.

    With a per-optics-group noise table each row uses its image's group spectrum
    and the shells come back per group, ``[G, n_shells]``. The resident passes sum
    the rows into their images with the scatter-add
    (``float32_bucketed_image_sums=False``, their statistics config's value); the
    dense GEMM coarse engine's calls keep the bucketed float32 sums.
    """

    if row_optics_groups is None:
        block_noise_shells, _, _ = compute_noise_block(
            proj,
            proj_abs2,
            summed_masked,
            ctf_probs,
            noise_variance,
            shell_indices,
            int(n_shells),
            return_split=False,
        )
        row_noise = jnp.asarray(noise_variance)
    else:
        block_noise_shells = compute_noise_block_per_optics_group(
            proj,
            proj_abs2,
            summed_masked,
            ctf_probs,
            noise_variance,
            row_optics_groups,
            shell_indices,
            int(n_shells),
        )
        row_noise = jnp.asarray(noise_variance)[row_optics_groups]
    a2_per_row, xa_per_row = _flat_row_norm_and_scale_terms(
        proj, proj_abs2, summed_masked, ctf_probs, row_noise
    )
    # These nonnegative F32 row terms can occupy thousands of candidate rows
    # per image. The bucketed sum keeps atomic addition order from losing the
    # same small contributions that the norm correction needs.
    a2_bucket = 128 if float32_bucketed_image_sums and a2_per_row.dtype == jnp.float32 else None
    xa_bucket = 128 if float32_bucketed_image_sums and xa_per_row.dtype == jnp.float32 else None
    a2_per_image = segment_sum_by_image(
        a2_per_row, row_image_local, int(image_capacity),
        float32_scalar_bucket_size=a2_bucket,
    )
    xa_per_image = segment_sum_by_image(
        xa_per_row, row_image_local, int(image_capacity),
        float32_scalar_bucket_size=xa_bucket,
    )
    return block_noise_shells.astype(jnp.float64), a2_per_image, xa_per_image


@jax.jit
def resident_block_wavg_algebraic_terms(
    proj,  # complex [block, P]
    proj_abs2,  # real [block, P]
    summed_masked,  # complex [block, P]
    ctf_probs,  # real [block, P]
    noise_variance,  # real [P]
    scale,  # real [C_B]
    row_image_local,  # int32 [block]
    row_optics_groups=None,  # int32 [block] with a [G, P] noise table
):
    """Flat-row twin of ``_relion_wavg_atomic_triplet_terms``, without its image power.

    Used when the pass does not carry RELION's RFLOAT CTF operand, which is
    the branch the host bucket tail takes when ``direct_ctf_rfloat_recon`` is
    ``None``. Term for term it is the rectangular helper with the
    ``(image, rotation)`` axes folded into the row axis: the two ``!= 0`` mass
    predicates, the per-image scale division and the float32 casts stay where
    they are. The ``diff2`` slot is ``aa - 2 xa``: the helper's per-row image
    power ``sum_t w[r, t] |x_t[p]|^2`` is summed over an image's rows by the
    Wavg accumulator, so the chunk adds it once per image from the
    translation marginal (:func:`add_wavg_rectangle_image_power`).
    """

    proj = jnp.asarray(proj, dtype=jnp.complex64)
    proj_abs2 = jnp.asarray(proj_abs2, dtype=jnp.float32)
    summed_masked = jnp.asarray(summed_masked, dtype=jnp.complex64)
    ctf_probs = jnp.asarray(ctf_probs, dtype=jnp.float32)
    noise_variance = (
        jnp.asarray(noise_variance, dtype=jnp.float32).reshape(-1)[None, :]
        if row_optics_groups is None
        else jnp.asarray(noise_variance, dtype=jnp.float32)[row_optics_groups]
    )
    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    row_scale = jnp.asarray(scale, dtype=jnp.float32).reshape(-1)[row_image_local]

    ctf_has_mass = ctf_probs != 0.0
    ctf_posterior_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance, 0.0)
    aa_raw = jnp.where(ctf_has_mass, proj_abs2 * ctf_posterior_raw, 0.0).astype(jnp.float32)
    cross_has_mass = summed_masked != 0.0
    cross = jnp.where(cross_has_mass, proj * jnp.conj(summed_masked), 0.0)
    xa_raw = (noise_variance * cross.real).astype(jnp.float32)
    safe_scale = jnp.maximum(row_scale, jnp.asarray(1e-30, dtype=jnp.float32))
    xa = (xa_raw / safe_scale[:, None]).astype(jnp.float32)
    aa = (aa_raw / (safe_scale[:, None] ** 2)).astype(jnp.float32)

    diff2 = (aa_raw - jnp.asarray(2.0, dtype=jnp.float32) * xa_raw).astype(jnp.float32)
    return jnp.stack((xa, aa, diff2), axis=-1)


@partial(jax.jit, static_argnames=("power_at_exact_positions",))
def add_wavg_rectangle_image_power(
    wavg_triplet_pixels,  # float32 [C_B, P_rect, 3]
    raw_shifted_rectangle,  # complex64 [C_B, T, P_rect]
    row_posterior,  # float32 [rows, T], a chunk's rows in any order
    row_kernel_ids,  # int32 [rows], the chunk image of each row, -1 for padding
    exact_positions,  # int32 [P_exact]
    logical_rect_pixels,  # int32 scalar
    *,
    power_at_exact_positions: bool,
):
    """Add a chunk's Wavg image power to its per-image ``diff2`` slot, once.

    The rectangular form (the test oracle ``relion_wavg_rectangle_triplet_terms``) fills the
    rectangle's ``diff2`` slot with each row's posterior-weighted image power
    and overwrites the exact-radius positions with the projected triplet; the
    M-step blocks add only their rows' exact terms
    (:func:`~relax.cuda.fine.relion_wavg_exact_atomic_flat_rows_triplet_add_f32`).
    RELION adds these per-orientation sums with atomics (wavg.cuh:147-149), so
    the grouping of the additions is not RELION's either way.

    Per image, the rows' power sums to the power of the image's translation
    marginal, ``sum_t (sum_r w[r, t]) |x_t[p]|^2``, so one contraction per
    chunk replaces the per-row (or per-block) ones. It fills the logical
    rectangle; ``power_at_exact_positions`` is False on the sequential RFLOAT
    CTF route, whose kernel already puts each row's power into its exact
    terms, and True on the algebraic route, whose exact ``diff2`` is
    ``aa - 2 xa`` (:func:`resident_block_wavg_algebraic_terms`).
    """

    image_capacity = int(wavg_triplet_pixels.shape[0])
    row_kernel_ids = jnp.asarray(row_kernel_ids, dtype=jnp.int32)
    # The marginal is a one-hot contraction rather than a segment sum: a
    # scatter-add over a chunk's rows lands its float32 additions in atomic
    # order, and the chunk's Wavg statistics repeat bitwise. Padding rows
    # (id -1) match no image.
    row_image = row_kernel_ids[:, None] == jnp.arange(image_capacity, dtype=jnp.int32)[None, :]
    # HIGHEST keeps both contractions in float32 arithmetic: at DEFAULT
    # precision a GPU dot may take TF32 operands.
    marginal = jnp.einsum(
        "rb,rt->bt",
        row_image.astype(jnp.float32),
        jnp.asarray(row_posterior, dtype=jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    image_power = jnp.einsum(
        "bt,btp->bp",
        marginal,
        relion_wavg_shifted_power(raw_shifted_rectangle),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    n_rect = int(image_power.shape[-1])
    mask = jnp.arange(n_rect, dtype=jnp.int32) < jnp.asarray(logical_rect_pixels, dtype=jnp.int32)
    if not power_at_exact_positions:
        mask = mask & jnp.ones((n_rect,), dtype=bool).at[jnp.asarray(exact_positions, dtype=jnp.int32)].set(False)
    return wavg_triplet_pixels.at[..., 2].add(
        jnp.where(mask, image_power, jnp.zeros((), image_power.dtype))
    )


def _logical_rect_pixels(tables, spec):
    """The Wavg rectangle's logical pixel count: the stable window's, else the spec's."""

    if tables.window_logical is None:
        return jnp.asarray(spec.n_rect, dtype=jnp.int32)
    return tables.window_logical.rect_pixels


def add_chunk_wavg_image_power(mstep, operands, tables, row_posterior, row_kernel_ids, *, spec):
    """:func:`add_wavg_rectangle_image_power` on a chunk's M-step carry."""

    return mstep._replace(
        wavg_triplet_pixels=add_wavg_rectangle_image_power(
            mstep.wavg_triplet_pixels,
            operands.raw_translated_wavg_rectangle,
            row_posterior,
            row_kernel_ids,
            tables.exact_positions,
            _logical_rect_pixels(tables, spec),
            power_at_exact_positions=not spec.use_rfloat_ctf_wavg,
        )
    )
