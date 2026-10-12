"""EM-only symbols relocated from ``recovar.cuda_backproject`` (relax split P1, pure move).

Function bodies are AST-identical to their originals at Q (tag q-reconcile-20260923); only the module
path changed.  See pr179_coordination/relax_split_plan_20260923/PLAN.md.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from recovar.cuda_backproject import (
    custom_cuda_requested,
)
from recovar.cuda_backproject import (
    rot_to_compact as _rot_to_compact,
)

from relax.cuda import library
from relax.cuda.fine import _validate_relion_fine_diff2_inputs


@jax.jit
def relion_coarse_diff2_rectangular_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """Evaluate RELION's coarse Gaussian CUDA reduction topology.

    Shapes are ``reference=(R,N)``, ``shifted_image=(B,T,N)``,
    ``weight=(B,N)``, ``initial_diff2=(B,)``, and
    ``full_to_compact=(F,)``. The kernel uses RELION's 128-thread,
    16-orientation blocks and its native per-translation lane assignment. The
    lane partials are combined with CUDA atomics on top of the supplied
    high-resolution image term, as in RELION's coarse scorer.
    """

    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    if initial_diff2.dtype != jnp.float32:
        raise TypeError(
            f"initial_diff2 must be float32, got {initial_diff2.dtype}"
        )
    if reference.ndim != 2 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "rectangular coarse diff2 expects reference rank 2, shifted rank "
            f"3, and weight rank 2, got {reference.shape}, "
            f"{shifted_image.shape}, {weight.shape}"
        )
    if (
        shifted_image.shape[0] != weight.shape[0]
        or reference.shape[1] != shifted_image.shape[2]
        or reference.shape[1] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or shifted_image.shape[0] <= 0
        or shifted_image.shape[1] <= 0
        or shifted_image.shape[1] > 128
        or initial_diff2.shape != (shifted_image.shape[0],)
    ):
        raise ValueError(
            "rectangular coarse diff2 operands have inconsistent shapes or "
            "more than 128 translations: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}"
        )
    out_type = jax.ShapeDtypeStruct(
        (shifted_image.shape[0], reference.shape[0], shifted_image.shape[1]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_COARSE_DIFF2_RECTANGULAR_F32,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, initial_diff2, full_to_compact)


@jax.jit
def relion_coarse_normalized_cc_pairs_f32(
    shifted_image: jax.Array,
    score_weight: jax.Array,
    reference: jax.Array,
    half_weights: jax.Array,
    packed_to_compact: jax.Array,
) -> jax.Array:
    """Evaluate bounded coarse normalized-CC pairs with RELION CUDA arithmetic.

    The first three operands have shape ``(B,C,N)`` and the output has shape
    ``(B,C)``. One 128-thread CUDA block evaluates each candidate, preserving
    RELION's operand contraction, lane tree, ``sqrtf``, and 128 identical
    atomic additions. ``packed_to_compact`` restores current-size FFTW order.
    """

    for name, value, dtype in (
        ("shifted_image", shifted_image, jnp.complex64),
        ("score_weight", score_weight, jnp.float32),
        ("reference", reference, jnp.complex64),
        ("half_weights", half_weights, jnp.float32),
        ("packed_to_compact", packed_to_compact, jnp.int32),
    ):
        if value.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {value.dtype}")
    if (
        shifted_image.ndim != 3
        or score_weight.shape != shifted_image.shape
        or reference.shape != shifted_image.shape
        or half_weights.shape != (shifted_image.shape[-1],)
        or packed_to_compact.ndim != 1
        or packed_to_compact.shape[0] <= 0
        or shifted_image.shape[0] <= 0
        or shifted_image.shape[1] <= 0
        or shifted_image.shape[2] <= 0
    ):
        raise ValueError(
            "RELION coarse normalized-CC pairs have inconsistent shapes: "
            f"{shifted_image.shape}, {score_weight.shape}, {reference.shape}, "
            f"{half_weights.shape}, {packed_to_compact.shape}",
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION coarse normalized-CC pairs require a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION coarse normalized-CC pairs require the custom CUDA extension",
        )
    library._ensure_ffi()
    out_type = jax.ShapeDtypeStruct(shifted_image.shape[:-1], jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_COARSE_NORMALIZED_CC_PAIRS_F32,
        out_type,
        vmap_method="sequential",
    )(
        shifted_image,
        score_weight,
        reference,
        half_weights,
        packed_to_compact,
    )


@functools.partial(jax.jit, static_argnums=(6, 7, 8, 9))
def relion_coarse_normalized_cc_native_texture_pairs_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    shifted_image: jax.Array,
    score_weight: jax.Array,
    half_weights: jax.Array,
    packed_to_compact: jax.Array,
    current_size: int,
    padding_factor: int,
    projector_max_r: int,
    return_components: bool = False,
    translation_angles: jax.Array | None = None,
    numerator_weight: jax.Array | None = None,
) -> jax.Array:
    """Evaluate bounded normalized-CC pairs from RELION's CUDA texture.

    Each row is one candidate. Its projection is sampled from
    ``projector_full`` inside the same 128-thread CUDA block that contracts the
    normalized-CC numerator and reference norm. When ``translation_angles`` is
    supplied, the image is translated in that same block with RELION's
    coarse-kernel ``sincosf`` arithmetic. This avoids materializing a RECOVAR
    projection or translated image at a different arithmetic boundary.
    """

    projector_full = jnp.asarray(projector_full)
    rotation_matrices = jnp.asarray(rotation_matrices)
    shifted_image = jnp.asarray(shifted_image)
    score_weight = jnp.asarray(score_weight)
    half_weights = jnp.asarray(half_weights)
    packed_to_compact = jnp.asarray(packed_to_compact)
    if translation_angles is None:
        translation_angles = jnp.zeros(
            (shifted_image.shape[0], 2),
            dtype=jnp.float32,
        )
    else:
        translation_angles = jnp.asarray(translation_angles)
    if numerator_weight is None:
        numerator_weight = jnp.ones_like(score_weight, dtype=jnp.float32)
    else:
        numerator_weight = jnp.asarray(numerator_weight)
    for name, value, dtype in (
        ("projector_full", projector_full, jnp.complex64),
        ("rotation_matrices", rotation_matrices, jnp.float32),
        ("shifted_image", shifted_image, jnp.complex64),
        ("score_weight", score_weight, jnp.float32),
        ("half_weights", half_weights, jnp.float32),
        ("packed_to_compact", packed_to_compact, jnp.int32),
        ("translation_angles", translation_angles, jnp.float32),
        ("numerator_weight", numerator_weight, jnp.float32),
    ):
        if value.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {value.dtype}")
    if (
        projector_full.ndim != 3
        or projector_full.shape[0] <= 0
        or projector_full.shape[1:]
        != (projector_full.shape[0], projector_full.shape[0])
        or rotation_matrices.ndim != 3
        or rotation_matrices.shape[1:] != (3, 3)
        or rotation_matrices.shape[0] <= 0
        or shifted_image.ndim != 2
        or shifted_image.shape[0] != rotation_matrices.shape[0]
        or shifted_image.shape[1] <= 0
        or score_weight.shape != shifted_image.shape
        or translation_angles.shape != (shifted_image.shape[0], 2)
        or numerator_weight.shape != shifted_image.shape
        or half_weights.shape != (shifted_image.shape[1],)
        or packed_to_compact.shape
        != (int(current_size) * (int(current_size) // 2 + 1),)
        or int(current_size) <= 0
        or int(padding_factor) <= 0
        or int(projector_max_r) <= 0
    ):
        raise ValueError(
            "native texture normalized-CC pair operands have inconsistent shapes: "
            f"projector={projector_full.shape}, rotations={rotation_matrices.shape}, "
            f"image={shifted_image.shape}, weight={score_weight.shape}, "
            f"half_weights={half_weights.shape}, lookup={packed_to_compact.shape}",
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "RELION native texture normalized-CC pairs require a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION native texture normalized-CC pairs require the custom CUDA extension"
        )
    library._ensure_ffi()
    eulers = jnp.swapaxes(rotation_matrices, -1, -2).reshape(
        rotation_matrices.shape[0], 9
    )
    out_type = jax.ShapeDtypeStruct((shifted_image.shape[0], 3), jnp.float32)
    components = jax.ffi.ffi_call(
        library._TARGET_RELION_COARSE_NORMALIZED_CC_NATIVE_TEXTURE_PAIRS_F32,
        out_type,
        vmap_method="sequential",
    )(
        projector_full,
        eulers,
        shifted_image,
        translation_angles,
        score_weight,
        numerator_weight,
        half_weights,
        packed_to_compact,
        current_size=np.int64(current_size),
        padding_factor=np.int64(padding_factor),
        projector_max_r=np.int64(projector_max_r),
    )
    return components if return_components else components[:, 0]


def _prepare_relion_coarse_diff2_projector_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    images: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
) -> tuple[jax.Array, jax.ShapeDtypeStruct]:
    """Validate and prepare the one shared fused coarse-projector ABI."""
    if projector_full.dtype != jnp.complex64 or projector_full.ndim != 3:
        raise TypeError(
            "projector_full must be a rank-3 complex64 array, got "
            f"{projector_full.shape} {projector_full.dtype}"
        )
    if images.dtype != jnp.complex64 or images.ndim != 2:
        raise TypeError(
            f"images must be rank-2 complex64, got {images.shape} {images.dtype}"
        )
    if rotation_matrices.dtype != jnp.float32 or rotation_matrices.ndim != 3:
        raise TypeError(
            "rotation_matrices must be rank-3 float32, got "
            f"{rotation_matrices.shape} {rotation_matrices.dtype}"
        )
    if rotation_matrices.shape[1:] != (3, 3):
        raise ValueError(
            f"rotation_matrices must end in (3, 3), got {rotation_matrices.shape}"
        )
    if translation_angles.dtype != jnp.float32 or translation_angles.ndim != 2:
        raise TypeError(
            "translation_angles must be rank-2 float32, got "
            f"{translation_angles.shape} {translation_angles.dtype}"
        )
    if translation_angles.shape[1] != 2 or translation_angles.shape[0] > 128:
        raise ValueError(
            "translation_angles must have shape (T, 2) with T <= 128, got "
            f"{translation_angles.shape}"
        )
    if weight.dtype != jnp.float32 or weight.shape != images.shape:
        raise TypeError(
            f"weight must be float32 with shape {images.shape}, got {weight.shape} {weight.dtype}"
        )
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (images.shape[0],):
        raise TypeError(
            "initial_diff2 must be float32 with one value per image, got "
            f"{initial_diff2.shape} {initial_diff2.dtype}"
        )
    if full_to_compact.dtype != jnp.int32 or full_to_compact.shape != (
        int(current_size) * (int(current_size) // 2 + 1),
    ):
        raise TypeError(
            "full_to_compact must be the current-size packed int32 lookup, got "
            f"{full_to_compact.shape} {full_to_compact.dtype}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION fused coarse projector requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION fused coarse projector was requested but custom CUDA is disabled"
        )
    library._ensure_ffi()
    compact_rotations = _rot_to_compact(rotation_matrices, jnp.float32)
    out_type = jax.ShapeDtypeStruct(
        (images.shape[0], rotation_matrices.shape[0], translation_angles.shape[0]),
        jnp.float32,
    )
    return compact_rotations, out_type


def _validate_relion_coarse_single_lane_canonical(
    translation_count: int,
    *,
    canonical_reduction: bool,
    single_lane_canonical: bool,
) -> None:
    """Fail closed unless one CUDA thread owns each coarse translation."""

    if not single_lane_canonical:
        return
    if not canonical_reduction:
        raise ValueError(
            "single_lane_canonical=True requires canonical_reduction=True",
        )
    if not 65 <= int(translation_count) <= 128:
        raise ValueError(
            "single_lane_canonical=True requires 65--128 translations, got "
            f"{translation_count}",
        )


def _validate_relion_coarse_prehalf_weight(
    *,
    canonical_reduction: bool,
    single_lane_canonical: bool,
    prehalf_weight: bool,
) -> None:
    """Restrict the prehalved-weight experiment to native atomics."""

    if prehalf_weight and (canonical_reduction or single_lane_canonical):
        raise ValueError(
            "prehalf_weight=True requires canonical_reduction=False and single_lane_canonical=False",
        )


@functools.partial(
    jax.jit,
    static_argnames=(
        "current_size",
        "physical_image_size",
        "model_max_r",
        "padding_factor",
        "canonical_reduction",
        "single_lane_canonical",
        "prehalf_weight",
    ),
)
def relion_coarse_diff2_projector_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    images: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
    padding_factor: int = 1,
    canonical_reduction: bool = False,
    single_lane_canonical: bool = False,
    prehalf_weight: bool = False,
) -> jax.Array:
    """Run the parity-locked shared RELION fused coarse projector.

    ``canonical_reduction`` replaces the final atomic lane admission with a
    fixed lane-index sum.  It retains the same projector, interpolation,
    translation, pixel traversal, and per-lane arithmetic and is intended for
    source-order parity qualification at marginal adaptive cutoffs.

    ``single_lane_canonical`` is a default-off compile-time specialization for
    65--128 translations, where one CUDA thread is the sole contributor to
    each translation.  It preserves the canonical initial-plus-lane addition
    while omitting the generic lane staging buffer.

    ``prehalf_weight`` is a default-off native-atomic experiment that matches
    RELION's coarse source order by staging ``weight * 0.5`` once per pixel
    instead of multiplying every orientation's square sum by ``0.5``.  It is
    not supported by either canonical-reduction specialization.
    """

    _validate_relion_coarse_single_lane_canonical(
        translation_angles.shape[0],
        canonical_reduction=canonical_reduction,
        single_lane_canonical=single_lane_canonical,
    )
    _validate_relion_coarse_prehalf_weight(
        canonical_reduction=canonical_reduction,
        single_lane_canonical=single_lane_canonical,
        prehalf_weight=prehalf_weight,
    )

    compact_rotations, out_type = _prepare_relion_coarse_diff2_projector_f32(
        projector_full,
        rotation_matrices,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=current_size,
        physical_image_size=physical_image_size,
        model_max_r=model_max_r,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_COARSE_DIFF2_PROJECTOR_F32,
        out_type,
        vmap_method="sequential",
    )(
        projector_full,
        compact_rotations,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=np.int64(current_size),
        physical_image_size=np.int64(physical_image_size),
        model_max_r=np.int64(model_max_r),
        padding_factor=np.int64(int(padding_factor)),
        canonical_reduction=np.int64(bool(canonical_reduction)),
        single_lane_canonical=np.int64(bool(single_lane_canonical)),
        prehalf_weight=np.int64(bool(prehalf_weight)),
    )


@functools.partial(
    jax.jit,
    static_argnames=("current_size", "physical_image_size", "model_max_r", "padding_factor"),
)
def relion_coarse_diff2_projector_per_image_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    images: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
    padding_factor: int = 1,
) -> jax.Array:
    """:func:`relion_coarse_diff2_projector_f32` with one pose set per image, canonical reduction.

    Image ``b`` is scored with its own rotations ``rotation_matrices[b]`` ``[R, 3, 3]`` and phases
    ``translation_angles[b]`` ``[T, 2]`` (``T <= 128``), as a subtomogram's tilt images each have their
    own ``Aproj``; returns ``[B, R, T]``. Each image's diff2 is the per-image call's (same kernel
    arithmetic and lane reduction); the texture is staged once and all images run in one launch.
    """

    n_images, n_rotations = int(rotation_matrices.shape[0]), int(rotation_matrices.shape[1])
    if rotation_matrices.shape[2:] != (3, 3) or translation_angles.ndim != 3 or translation_angles.shape[2] != 2:
        raise ValueError(
            f"per-image poses are [B, R, 3, 3] and [B, T, 2], got {rotation_matrices.shape} {translation_angles.shape}"
        )
    if images.shape[0] != n_images or translation_angles.shape[0] != n_images:
        raise ValueError("rotations, phases and images must have one entry per image")
    compact_rotations, _ = _prepare_relion_coarse_diff2_projector_f32(
        projector_full,
        rotation_matrices.reshape(n_images * n_rotations, 3, 3),
        images,
        translation_angles[0],
        weight,
        initial_diff2,
        full_to_compact,
        current_size=current_size,
        physical_image_size=physical_image_size,
        model_max_r=model_max_r,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_COARSE_DIFF2_PROJECTOR_PER_IMAGE_F32,
        jax.ShapeDtypeStruct((n_images, n_rotations, int(translation_angles.shape[1])), jnp.float32),
        vmap_method="sequential",
    )(
        projector_full,
        compact_rotations.reshape(n_images, n_rotations, 6),
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=np.int64(current_size),
        physical_image_size=np.int64(physical_image_size),
        model_max_r=np.int64(model_max_r),
        padding_factor=np.int64(int(padding_factor)),
    )
