"""RELION wavg rectangle and low-shell power terms of the sparse bucketed pass 2.

The wavg rectangle that reproduces RELION's per-image weighted-average
normalization terms (atomic, sequential and direct modes), the weighted
image power shells and the low-shell noise/norm replacements derived from
them. The resident statistics stage builds the rectangle once per pass.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.fourier.fourier_window import make_fourier_window_indices_np, relion_fftw_order_for_square_score_window
from relax.fourier.half_spectrum import bin_shell_values_jax, make_relion_noise_shell_indices_half


class RelionWavgRectangle(NamedTuple):
    """Static mapping for RELION's full cropped Wavg CUDA pixel stream."""

    centered_indices: np.ndarray
    exact_positions: np.ndarray
    shell_indices: np.ndarray


@partial(jax.jit, static_argnames=("shell_count",))
def image_power_shells(processed_half, shell_indices_half, *, shell_count: int):
    """Each image's power in each noise shell, float64 ``[B, shell_count]``.

    ``sum |x|^2`` over the shell's pixels, as RELION bins one image's power
    spectrum before adding it into the noise sums (``power_img``); pixels whose
    shell index is the binning sentinel (outside ``[0, shell_count)``) are left
    out. The float32 pixel powers are reduced in float64 by a one-hot matrix
    product, which is deterministic.
    """

    pixel_power = (jnp.abs(processed_half) ** 2).astype(jnp.float64)
    shells = jnp.arange(int(shell_count), dtype=jnp.int32)
    one_hot = (jnp.asarray(shell_indices_half, dtype=jnp.int32)[:, None] == shells[None, :]).astype(jnp.float64)
    return jnp.matmul(pixel_power, one_hot, precision=jax.lax.Precision.HIGHEST)


# ``norm_unweighted_shell_cutoff`` is an operand, not a key: a Python int or a
# traced int32 scalar (the resident engine's logical cutoff); None (an empty
# pytree) means no cutoff.
@partial(
    jax.jit,
    static_argnames=(
        "include_unweighted_high_shell",
        "deterministic_norm_reduction",
    ),
)
def weighted_image_power_from_shells(
    power_shells,
    support_mass,
    norm_unweighted_high_shell,
    valid_image_mask,
    *,
    norm_unweighted_shell_cutoff: "int | jax.Array | None",
    include_unweighted_high_shell: bool,
    deterministic_norm_reduction: bool,
    high_shell_mass=None,
):
    """:func:`_weighted_image_power_shells_and_per_image_core` from per-image shell powers.

    ``power_shells`` is :func:`image_power_shells` of the processed images. The
    shell masses depend only on the shell, so weighting each image's shell sums
    gives the same noise-shell and per-image norm terms as weighting its pixels;
    the sums are grouped by shell first, per image, and accumulated in float64.
    Returns float64 shells and the per-image norm power in the norm reduction
    dtype (float64 when deterministic, float32 otherwise), as the pixel form does.

    Shells above the cutoff take each image's power with mass one (the valid images), or with
    ``high_shell_mass`` when given: a subtomogram's tilt images each carry 1 / n_images of their
    particle (acc_ml_optimiser_impl.h:3490-3491).
    """

    power_shells = jnp.asarray(power_shells, dtype=jnp.float64)
    n_shells = int(power_shells.shape[1])
    mass = jnp.asarray(support_mass).astype(jnp.float64)
    shell_mass = jnp.broadcast_to(mass[:, None], power_shells.shape)
    full_mass = (
        jnp.ones_like(mass)
        if valid_image_mask is None
        else jnp.asarray(valid_image_mask).astype(jnp.float64)
    )
    if high_shell_mass is not None:
        full_mass = full_mass * jnp.asarray(high_shell_mass).astype(jnp.float64)
    unweighted_shell = None
    if norm_unweighted_shell_cutoff is not None:
        # A Python int or a traced int32 scalar (the resident engine's logical cutoff).
        unweighted_shell = jnp.arange(n_shells) > jnp.asarray(norm_unweighted_shell_cutoff, dtype=jnp.int32)
        high_shell_mass = full_mass if include_unweighted_high_shell else jnp.zeros_like(full_mass)
        shell_mass = jnp.where(unweighted_shell[None, :], high_shell_mass[:, None], shell_mass)
    weighted_shells = jnp.sum(power_shells * shell_mass, axis=0)
    weighted_per_image = jnp.sum(power_shells * shell_mass, axis=1)
    if norm_unweighted_high_shell is not None and include_unweighted_high_shell:
        if unweighted_shell is None:
            raise ValueError("a replacement high-shell norm term requires a shell cutoff")
        replacement_high = jnp.asarray(norm_unweighted_high_shell).astype(jnp.float64)
        if replacement_high.shape != mass.shape:
            raise ValueError(
                "replacement high-shell norm term must match the particle axis, got "
                f"{replacement_high.shape} for {mass.shape}"
            )
        generic_high = jnp.sum(jnp.where(unweighted_shell[None, :], power_shells, 0.0), axis=1)
        weighted_per_image = jax.lax.optimization_barrier(weighted_per_image)
        weighted_per_image = weighted_per_image + full_mass * (replacement_high - generic_high)
    norm_reduction_dtype = jnp.float64 if deterministic_norm_reduction else jnp.float32
    return weighted_shells, weighted_per_image.astype(norm_reduction_dtype)


def _make_relion_wavg_rectangle(
    image_shape,
    current_size,
    recon_window_indices,
    *,
    reconstruction_current_size=None,
):
    """Map active reconstruction pixels into RELION's complete Wavg crop.

    ``exact_positions`` retains its historical field name, but may describe
    either the exact BackProjector disk or RELION InitialModel's rounded-shell
    Wavg support. The supplied reconstruction indices select that contract.

    RELION may remap the particle-image ``current_size`` for an optics group
    while retaining the model-coordinate radius for Projector/BackProjector.
    The rectangle and its rounded noise-shell mask therefore use the particle
    size, while the exact projected terms use ``reconstruction_current_size``.
    """

    image_shape = tuple(int(value) for value in image_shape)
    current_size = image_shape[0] if current_size is None else int(current_size)
    model_current_size = (
        current_size
        if reconstruction_current_size is None
        else int(reconstruction_current_size)
    )
    if model_current_size > current_size:
        raise ValueError(
            "RELION Wavg model support cannot exceed the particle-image crop: "
            f"model={model_current_size}, image={current_size}"
        )
    rectangle_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        square=True,
        include_dc=True,
    )
    rectangle_order = relion_fftw_order_for_square_score_window(
        image_shape,
        current_size,
        rectangle_indices,
    )
    rectangle_indices = rectangle_indices[rectangle_order]
    rounded_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        include_dc=True,
        exact_radius=False,
    )
    exact_indices, _ = make_fourier_window_indices_np(
        image_shape,
        model_current_size,
        include_dc=True,
        exact_radius=True,
    )
    full_box_unwindowed = bool(
        recon_window_indices is None
        and current_size == image_shape[0]
        and model_current_size == image_shape[0]
    )
    recon_indices = (
        np.arange(image_shape[0] * (image_shape[1] // 2 + 1), dtype=np.int32)
        if full_box_unwindowed
        else np.asarray(recon_window_indices, dtype=np.int32).reshape(-1)
    )
    # Historical K-class first-iteration CC replays use a square image crop
    # with the BackProjector radius applied inside that crop. Keep that
    # complete support when mapping projected terms into the Wavg rectangle.
    square_exact_indices, _ = make_fourier_window_indices_np(
        image_shape, model_current_size, square=True,
        include_dc=True, exact_radius=True,
    )
    exact_support = np.array_equal(np.sort(recon_indices), exact_indices)
    rounded_support = np.array_equal(np.sort(recon_indices), rounded_indices)
    square_exact_support = np.array_equal(np.sort(recon_indices), square_exact_indices)
    if not (full_box_unwindowed or exact_support or rounded_support or square_exact_support):
        raise ValueError(
            "RELION Wavg rectangle requires a complete exact-radius, square exact-radius, or rounded-shell "
            "reconstruction window: "
            f"got {recon_indices.size} pixels, expected {exact_indices.size}, "
            f"{square_exact_indices.size}, or {rounded_indices.size}"
        )

    rectangle_position = {
        int(centered_index): position
        for position, centered_index in enumerate(rectangle_indices.tolist())
    }
    try:
        exact_positions = np.asarray(
            [rectangle_position[int(index)] for index in recon_indices],
            dtype=np.int32,
        )
        rounded_positions = np.asarray(
            [rectangle_position[int(index)] for index in rounded_indices],
            dtype=np.int32,
        )
    except KeyError as error:
        raise ValueError("RELION Wavg support is not contained in its square crop") from error

    shell_indices_half = np.asarray(
        make_relion_noise_shell_indices_half(image_shape),
        dtype=np.int32,
    )
    rectangle_shells = shell_indices_half[rectangle_indices]
    valid = np.zeros(rectangle_indices.size, dtype=bool)
    valid[rounded_positions] = True
    rectangle_shells = np.where(valid, rectangle_shells, -1).astype(np.int32)
    expected_rectangle_size = current_size * (current_size // 2 + 1)
    if rectangle_indices.size != expected_rectangle_size:
        raise ValueError(
            "RELION Wavg square crop topology changed: "
            f"got {rectangle_indices.size}, expected {expected_rectangle_size}"
        )
    if np.unique(exact_positions).size != exact_positions.size:
        raise ValueError("RELION Wavg reconstruction position mapping is not bijective")
    return RelionWavgRectangle(
        centered_indices=rectangle_indices.astype(np.int32, copy=False),
        exact_positions=exact_positions,
        shell_indices=rectangle_shells,
    )


def _make_stable_relion_wavg_rectangle(image_shape, shape_plan):
    """Pack a logical Wavg rectangle before its physical-capacity tail.

    RELION's Wavg kernel walks the dense FFTW rectangle, whereas RECOVAR's
    shared projection path stores a compact disk.  Stable shapes are exact
    only when both streams retain their original logical order.  The first
    ``logical_rectangle_pixels`` entries below are therefore byte-for-byte the
    ordinary logical rectangle.  Physical-only pixels follow it and receive a
    sentinel shell; runtime CUDA bounds must never issue that tail.
    """

    logical_recon = shape_plan.packed_indices_np("recon")[
        : shape_plan.logical_reconstruction_pixels
    ]
    physical_recon = shape_plan.packed_indices_np("recon")
    logical = _make_relion_wavg_rectangle(
        image_shape,
        shape_plan.logical_current_size,
        logical_recon,
    )
    physical = _make_relion_wavg_rectangle(
        image_shape,
        shape_plan.physical_current_size,
        physical_recon,
    )

    logical_set = set(map(int, logical.centered_indices.tolist()))
    physical_tail = np.asarray(
        [
            int(index)
            for index in physical.centered_indices.tolist()
            if int(index) not in logical_set
        ],
        dtype=np.int32,
    )
    packed_rectangle = np.concatenate(
        (logical.centered_indices, physical_tail),
    ).astype(np.int32, copy=False)
    if packed_rectangle.size != shape_plan.physical_rectangle_pixels:
        raise ValueError(
            "stable RELION Wavg rectangle does not fill its physical capacity: "
            f"got {packed_rectangle.size}, expected {shape_plan.physical_rectangle_pixels}"
        )
    logical_count = shape_plan.logical_rectangle_pixels
    recon_tail_count = (
        shape_plan.physical_reconstruction_pixels
        - shape_plan.logical_reconstruction_pixels
    )
    rectangle_tail_count = packed_rectangle.size - logical_count
    if recon_tail_count > rectangle_tail_count:
        raise ValueError(
            "stable RELION Wavg rectangle tail cannot hold its reconstruction "
            f"tail: recon={recon_tail_count}, rectangle={rectangle_tail_count}"
        )
    # Some pixels newly admitted by the larger physical radius still lie in
    # the *logical* square rectangle (for example, immediately outside its
    # exact-radius disk).  Mapping those pixels by coordinate would make the
    # logical native Wavg/BPref loops consume physical-only values.  Logical
    # reconstruction rows retain their exact FFTW positions; all capacity-only
    # rows instead receive arbitrary unique storage in the inert rectangle
    # tail, whose coordinates are intentionally never issued.
    recon_positions = np.concatenate(
        (
            logical.exact_positions,
            np.arange(
                logical_count,
                logical_count + recon_tail_count,
                dtype=np.int32,
            ),
        )
    ).astype(np.int32, copy=False)
    rectangle_shells = np.full(packed_rectangle.size, -1, dtype=np.int32)
    rectangle_shells[:logical_count] = logical.shell_indices
    return RelionWavgRectangle(
        centered_indices=packed_rectangle,
        exact_positions=recon_positions,
        shell_indices=rectangle_shells,
    )


def _relion_wavg_shifted_power(raw_shifted):
    """RELION's float32 ``|x|^2``, with the barrier between its two terms.

    The barrier keeps ``real*real`` and ``+ imag*imag`` from contracting into a
    single FMA, which is what makes this expression bitwise reproducible. The
    result depends only on ``raw_shifted``, elementwise, so a caller that needs
    the power of a gathered view may square first and gather afterwards.
    """
    raw_shifted = jnp.asarray(raw_shifted, dtype=jnp.complex64)
    shifted_power = (raw_shifted.real * raw_shifted.real).astype(jnp.float32)
    shifted_power = jax.lax.optimization_barrier(shifted_power)
    return (shifted_power + raw_shifted.imag * raw_shifted.imag).astype(jnp.float32)


def _relion_wavg_rectangle_power_contraction(shifted_power, posterior):
    """Contract an already-squared rectangle against the posterior over T."""
    return jnp.einsum(
        "brt,btp->brp",
        jnp.asarray(posterior, dtype=jnp.float32),
        jnp.asarray(shifted_power, dtype=jnp.float32),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    ).astype(jnp.float32)


def _relion_wavg_rectangle_image_power(raw_shifted, posterior):
    """Shared rectangle power contraction; keep the original F32/barrier order."""
    return _relion_wavg_rectangle_power_contraction(
        _relion_wavg_shifted_power(raw_shifted), posterior
    )


@jax.jit
def _relion_wavg_rectangle_triplet_terms(
    exact_triplet_terms,
    raw_shifted_rectangle,
    posterior,
    exact_positions,
):
    """Embed exact projection terms in the full native Wavg issue stream."""

    exact_terms = jnp.asarray(exact_triplet_terms, dtype=jnp.float32)
    raw_shifted = jnp.asarray(raw_shifted_rectangle, dtype=jnp.complex64)
    posterior = jnp.asarray(posterior, dtype=jnp.float32)
    exact_positions = jnp.asarray(exact_positions, dtype=jnp.int32)
    if exact_terms.ndim != 4 or exact_terms.shape[-1] != 3:
        raise ValueError(f"exact Wavg terms must have shape (B,R,P,3), got {exact_terms.shape}")
    if raw_shifted.ndim != 3 or posterior.shape != exact_terms.shape[:2] + (raw_shifted.shape[1],):
        raise ValueError(
            "Wavg rectangle translations/posteriors do not match exact terms: "
            f"terms={exact_terms.shape}, shifted={raw_shifted.shape}, posterior={posterior.shape}"
        )
    if exact_positions.shape != (exact_terms.shape[2],):
        raise ValueError(
            "Wavg exact-position mapping does not match the projected pixel axis: "
            f"positions={exact_positions.shape}, terms={exact_terms.shape}"
        )

    image_power = _relion_wavg_rectangle_image_power(raw_shifted, posterior)
    return _embed_relion_wavg_rectangle_terms(exact_terms, image_power, exact_positions)


def _embed_relion_wavg_rectangle_terms(exact_terms, image_power, exact_positions):
    """Rectangle ``[XA, AA, diff2]`` terms: image power everywhere, exact terms on the disk."""

    exact_terms = jnp.asarray(exact_terms, dtype=jnp.float32)
    image_power = jnp.asarray(image_power, dtype=jnp.float32)
    exact_positions = jnp.asarray(exact_positions, dtype=jnp.int32)
    rectangle_terms = jnp.zeros(
        exact_terms.shape[:2] + (image_power.shape[-1], 3),
        dtype=jnp.float32,
    )
    rectangle_terms = rectangle_terms.at[..., 2].set(image_power)
    return rectangle_terms.at[:, :, exact_positions, :].set(exact_terms)


def _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
    residual_shells,
    image_power_shells,
    atomic_diff2_per_pixel,
    shell_indices,
    *,
    exclusive_shell_stop: "int | jax.Array",
    shell_count: int,
):
    """``jax.numpy`` twin of the numpy direct-residual replacement above.

    Same contract, same float32 -> float64 widening of the atomic Wavg
    ``diff2`` stream and the same shell-stop clamping as
    :func:`_replace_low_shell_noise_with_relion_wavg_direct_residual`; written
    for the device-resident statistics stage, which must keep the whole bucket
    tail inside one traced program instead of pulling per-bucket shells to the
    host (see ``recovar/em/sparse_pass2/resident_statistics.py``).

    The only difference from the numpy original is the association order of
    the float64 shell sum: the original adds one image row at a time through
    ``np.add.at``, this one sums the image axis and bins once.  Both add the
    same float64 summands, so results agree to float64 rounding and are
    bitwise identical whenever each shell receives at most one summand.
    ``shell_count`` is static because it fixes the traced output shape;
    ``residual_shells`` must already have that length.
    """

    residual = jnp.asarray(residual_shells, dtype=jnp.float64)
    image_power = jnp.asarray(image_power_shells, dtype=jnp.float64)
    atomic_diff2 = jnp.asarray(atomic_diff2_per_pixel, dtype=jnp.float32)
    shells = jnp.asarray(shell_indices, dtype=jnp.int32).reshape(-1)
    if residual.ndim != 1 or residual.shape[0] != shell_count:
        raise ValueError(
            f"noise residual shells must be a vector of length {shell_count}, got {residual.shape}"
        )
    if image_power.shape != residual.shape:
        raise ValueError(
            "noise residual and image-power shells must be matching vectors, got "
            f"{residual.shape} and {image_power.shape}"
        )
    if atomic_diff2.ndim != 2 or atomic_diff2.shape[1] != shells.shape[0]:
        raise ValueError(
            "atomic Wavg diff2 must have shape (images, pixels) matching shell indices, got "
            f"{atomic_diff2.shape} and {shells.shape}"
        )
    # A Python int or a traced int32 scalar (the resident engine's logical stop).
    shell_stop = jnp.clip(jnp.asarray(exclusive_shell_stop, dtype=jnp.int32), 0, shell_count)
    # Pixels at or above the stop shell are dropped entirely, exactly as the
    # numpy original's ``valid`` mask does, not merely overwritten afterwards.
    covered_shells = jnp.where(shells < shell_stop, shells, jnp.int32(-1))
    direct_shells = bin_shell_values_jax(
        jnp.sum(atomic_diff2.astype(jnp.float64), axis=0),
        covered_shells,
        shell_count,
    )
    replaced = jnp.arange(shell_count, dtype=jnp.int32) < shell_stop
    return (
        jnp.where(replaced, direct_shells, residual),
        jnp.where(replaced, jnp.float64(0.0), image_power),
    )


def _relion_cuda_translate_wavg_norm_images(
    processed_score_half,
    translation_angles,
    score_window_indices,
    image_shape,
):
    """Translate the raw masked image at RELION's Wavg input boundary."""

    processed_score_half = jnp.asarray(processed_score_half, dtype=jnp.complex64)
    score_window_indices = jnp.asarray(score_window_indices, dtype=jnp.int32)
    return relion_cuda_translate_wavg_norm_window(
        processed_score_half[:, score_window_indices],
        translation_angles,
        score_window_indices,
        image_shape,
    )


def relion_cuda_translate_wavg_norm_window(
    window_pixels,
    translation_angles,
    window_indices,
    image_shape,
):
    """:func:`_relion_cuda_translate_wavg_norm_images` on already gathered window pixels.

    ``window_pixels`` is ``processed_score_half[:, window_indices]``; callers
    that keep only the Wavg window of each image resident translate it here.
    ``translation_angles`` is ``[T, 2]`` or ``[B, T, 2]`` (each image's own).
    """

    from relax.cuda import kernels as em_cuda_kernels

    window_pixels = jnp.asarray(window_pixels, dtype=jnp.complex64)
    window_indices = jnp.asarray(window_indices, dtype=jnp.int32)
    translation_angles = jnp.asarray(translation_angles, dtype=jnp.float32)
    translated = em_cuda_kernels.relion_translate_score_f32(
        window_pixels,
        translation_angles,
        window_indices,
        image_shape,
    )
    return translated.reshape(
        window_pixels.shape[0],
        translation_angles.shape[-2],
        window_indices.shape[0],
    )

