"""Per-image pass-2 operands prepared once per half and kept on the device (T16).

Why this module exists
----------------------
The device-resident K=1 driver used to call
:func:`~recovar.em.sparse_pass2.sparse_pass2_bucket_io._prepare_bucket_io` once
per capacity chunk. That call is per-image work -- a RELION FFT, the soft mask,
the CTF and noise algebra, the corrections and the pre-centering phase -- and it
was repeated for every chunk an image appears in, at an image occupancy of
0.38-0.66 at the early state, where a chunk is padded to 87 image slots to hold
about 33 valid images. It was 71.5% of the 135769 CUDA launches of one hp3 half
and a 43-44 ms host gap in front of every chunk, while contributing 0.32 s of
the half's 4.97 s of kernel time: the chunk loop was launch-bound, not
kernel-bound.

Every operand it produced is per-image pure except the translation tiling, so
this module runs the per-image half once for a whole set of images
(:func:`prepare_resident_half_operands`) and keeps the result on the device.
The chunk loop then only gathers rows by image id
(:func:`gather_resident_chunk_operands`). The translation tiling does not become
resident -- a ``[images, translations, pixels]`` tile for a whole half is tens
of gigabytes -- it disappears instead: T15's
``relion_translate_sum_flat_rows_f32`` applies the translation inside the M-step
reduction, so the M-step reads the *unshifted* per-image operands this module
stores.

What is stored, and in which convention
---------------------------------------
``_prepare_bucket_io`` builds two translated reconstruction operands with two
different primitives, and the kernel has to be told which:

* the reconstruction operand uses ``relion_translate_bpref_f32`` when
  ``relion_exact_bpref_operands`` is on -- RELION's BPref rotation, with the
  weighted CTF multiplied *after* the rotation. This module therefore stores the
  raw BPref image and the weighted CTF separately (``recon_image`` and
  ``recon_weight``), which is exactly the pair the kernel's BPref mode takes;
* the noise operand always uses ``relion_translate_score_f32`` (``recon_image``
  with no weight), so it is stored as one complex array.

Scope. The preparation covers the production K=1 configuration only and fails
closed elsewhere (:class:`ResidentOperandsUnsupported`), because a silently
different translate convention would change every reconstruction pixel. The
driver keeps the per-chunk path as the oracle for the configurations this module
refuses and for the ``...RESIDENT_OPERANDS=0`` comparison arm.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.helpers.batch_fetch import fetch_indexed_batch
from relax.helpers.dtype_policy import DensePrecisionPolicy
from relax.helpers.half_spectrum import make_shell_indices_half
from relax.helpers.optics_noise import pixel_rows
from relax.sparse_pass2.resident_statistics import resident_image_capacity
from relax.sparse_pass2.sparse_pass2_bucket_io import prepare_unshifted_bucket_operands
from relax.sparse_pass2.sparse_pass2_scoring import (
    _relion_native_fine_units,
    _relion_native_score_corr_img,
    _relion_powerclass_noise_terms,
    relion_powerclass_noise_presence,
)
from relax.sparse_pass2.sparse_pass2_wavg import image_power_shells, relion_cuda_translate_wavg_norm_window

logger = logging.getLogger(__name__)

RESIDENT_OPERANDS_ENV = "RELAX_SPARSE_PASS2_RESIDENT_OPERANDS"
# Cap on the resident per-image operands of one half. The operands scale with the
# half's image count, so a large particle count at a large box can outgrow the
# budget; the driver then keeps the per-chunk preparation and says so, instead
# of failing part way through. The cap is a share of what the JAX allocator can
# still hand out when the pass plans (sparse_pass2_budget.device_available_bytes,
# the reading the streamed projection budget uses); the other half stays for the
# class accumulators and the chunk working set.
_RESIDENT_OPERAND_BYTES_ENV = "RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES"
_RESIDENT_OPERAND_AVAILABLE_FRACTION = 0.5
_DEFAULT_RESIDENT_OPERAND_MAX_BYTES = 6 * 1024**3
_PREPARE_IMAGE_BATCH_ENV = "RELAX_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH"
_DEFAULT_PREPARE_IMAGE_BATCH = 256

__all__ = [
    "RESIDENT_OPERANDS_ENV",
    "ResidentHalfOperands",
    "ResidentOperandsUnsupported",
    "gather_resident_chunk_operands",
    "prepare_resident_half_operands",
    "resident_half_operand_bytes",
    "resident_image_capacity",
    "resident_operands_max_bytes",
]



class ResidentOperandsUnsupported(NotImplementedError):
    """The once-per-half preparation does not cover this pass's configuration."""


@dataclass(frozen=True)
class ResidentHalfOperands:
    """Every per-image operand of one half, on the device, in image order.

    Row ``i`` belongs to the half's image ``i``: the driver's candidate tables
    are built in that same order, so a chunk's image slots are a contiguous
    range of these rows and the per-chunk gather is a take by image id.

    Fields
    ------
    score_input, corr_img_score, highres_xi2_half, translation_prior
        The scoring operands, already gathered to the score window. These are
        what ``_prepare_bucket_io`` returned as ``direct_score_input``,
        ``ctf2_over_nv_half`` and the ``powerClass`` tail.
    recon_image, recon_weight
        The unshifted reconstruction operand of the M-step. ``recon_weight`` is
        the BPref weighted CTF and is ``None`` outside the exact-BPref
        configuration, which is also how the T15 kernel selects its translate
        convention: with a weight it reproduces ``relion_translate_bpref_f32``,
        without one ``relion_translate_score_f32``.
    noise_image
        The unshifted noise operand (``score_weighted_half`` on the
        reconstruction window), always in the score convention.
    ctf2_over_nv_recon, direct_ctf_rfloat_recon
        The M-step CTF operands on the reconstruction window.
    wavg_image_rect
        The raw processed image on RELION's Wavg rectangle, the window the Wavg
        terms translate (``processed_score_half_for_noise[:, rect_indices]``).
    image_power_shells
        The same image's power in each noise shell over the full packed half,
        float64 (:func:`~relax.sparse_pass2.sparse_pass2_wavg.image_power_shells`):
        all the image-power statistics read of it. Keeping the full half image
        instead was about 83% of a half's operand bytes at 256 pixels.
    relion_norm_high_shell, scale, group_ids
        Per-image scalars of the statistics tail. ``relion_norm_high_shell`` is
        the one operand here whose value is not reproducible bit for bit: its
        shell binning is a scatter-add over duplicate indices, which races. Two
        calls on the same array in one process differ by about 6e-8 relative,
        and a different batch size differs by the same amount, so preparing it
        once per half is inside its own run-to-run spread. Under
        ``RELAX_EM_DETERMINISTIC_REDUCTIONS=1`` the binning becomes a
        fixed-order masked reduction and every one of those comparisons is
        exact, which is how the driver's verification arm checks it.
    """

    n_images: int
    # Leading extent of every array below (:func:`resident_image_capacity`);
    # rows ``n_images..n_image_capacity-1`` are padding no chunk addresses.
    n_image_capacity: int
    n_score_pixels: int
    n_recon_pixels: int
    n_rect_pixels: int
    n_noise_shells: int
    n_fine_trans: int
    score_input: jax.Array
    corr_img_score: jax.Array
    highres_xi2_half: jax.Array | None
    translation_prior: jax.Array
    recon_image: jax.Array
    recon_weight: jax.Array | None
    noise_image: jax.Array
    ctf2_over_nv_recon: jax.Array
    direct_ctf_rfloat_recon: jax.Array | None
    wavg_image_rect: jax.Array
    image_power_shells: jax.Array
    relion_norm_high_shell: jax.Array | None
    scale: jax.Array
    group_ids: jax.Array
    # Optics-group row of each image, only with a per-group noise table
    # (relax.helpers.optics_noise); None keeps the one-group programs unchanged.
    optics_groups: jax.Array | None = None

    def __post_init__(self):
        if self.n_image_capacity < self.n_images:
            raise ValueError(f"image_capacity {self.n_image_capacity} is below n_images {self.n_images}")
        score_shape = (self.n_image_capacity, self.n_score_pixels)
        recon_shape = (self.n_image_capacity, self.n_recon_pixels)
        for name, expected in (
            ("score_input", score_shape),
            ("corr_img_score", score_shape),
            ("recon_image", recon_shape),
            ("noise_image", recon_shape),
            ("ctf2_over_nv_recon", recon_shape),
            ("wavg_image_rect", (self.n_image_capacity, self.n_rect_pixels)),
            ("image_power_shells", (self.n_image_capacity, self.n_noise_shells)),
            ("translation_prior", (self.n_image_capacity, self.n_fine_trans)),
        ):
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        for name in ("recon_weight", "direct_ctf_rfloat_recon"):
            value = getattr(self, name)
            if value is not None and tuple(value.shape) != recon_shape:
                raise ValueError(f"{name} must have shape {recon_shape}, got {tuple(value.shape)}")

    def nbytes(self) -> dict:
        """Device bytes of each resident array, plus their total."""

        parts = {}
        for name in (
            "score_input",
            "corr_img_score",
            "highres_xi2_half",
            "translation_prior",
            "recon_image",
            "recon_weight",
            "noise_image",
            "ctf2_over_nv_recon",
            "direct_ctf_rfloat_recon",
            "wavg_image_rect",
            "image_power_shells",
            "relion_norm_high_shell",
            "scale",
            "group_ids",
            "optics_groups",
        ):
            value = getattr(self, name)
            parts[name] = 0 if value is None else int(value.size) * int(value.dtype.itemsize)
        parts["total"] = sum(parts.values())
        return parts


def resident_half_operand_bytes(
    *,
    n_images: int,
    n_score_pixels: int,
    n_recon_pixels: int,
    n_rect_pixels: int,
    n_noise_shells: int,
    n_fine_trans: int,
    score_complex_bytes: int = 8,
    real_bytes: int = 4,
    rfloat_ctf_bytes: int = 8,
    norm_high_shell_bytes: int | None = None,
) -> int:
    """Host estimate of the resident operand bytes, before any device work.

    Used by the driver's admission check, so an over-budget half keeps the
    per-chunk preparation rather than allocating and failing. The arrays are
    stored at :func:`resident_image_capacity` rows, so that is what is counted.
    """

    n_images = resident_image_capacity(n_images)
    # The four per-image scalars are highres_Xi2, the norm high shell, the scale
    # and the group id. Only the norm term can be wider than the policy's real
    # dtype: source-faithful normalization accumulates it in float64.
    if norm_high_shell_bytes is None:
        norm_high_shell_bytes = real_bytes
    return int(
        n_images
        * (
            int(n_score_pixels) * (int(score_complex_bytes) + int(real_bytes))
            + int(n_recon_pixels)
            * (2 * int(score_complex_bytes) + 2 * int(real_bytes) + int(rfloat_ctf_bytes))
            + int(n_rect_pixels) * int(score_complex_bytes)
            + int(n_noise_shells) * 8
            + int(n_fine_trans) * int(real_bytes)
            + 3 * int(real_bytes)
            + int(norm_high_shell_bytes)
        )
    )


class ResidentOperandPresence(NamedTuple):
    """Which optional half operands a configuration produces.

    The real preparation learns this from the first image batch's keys. A
    caller that must know before any image exists -- the compile-ahead warm-up
    -- asks here. Each field cites the line that decides it, and the driver
    verifies the answer against the real operands as soon as they exist, so a
    drift shows up by name in the run's own log rather than as an unexplained
    compile that the warm-up failed to cover.
    """

    has_recon_weight: bool
    has_direct_ctf_rfloat: bool
    has_highres_xi2: bool
    has_relion_norm_high_shell: bool


def resident_half_operand_presence(
    *,
    relion_exact_bpref_operands,
    use_exact_relion_gaussian,
    accumulate_noise,
    current_size,
) -> ResidentOperandPresence:
    """Predict the optional operand set from the configuration flags.

    ``recon_weight`` exists when ``_batch_window_operands`` is handed a
    ``weighted_ctf_half``, and ``direct_ctf_rfloat_recon`` when it is handed a
    ``ctf_half_rfloat``; both are supplied only in the exact-BPref
    configuration (``sparse_pass2_bucket_io._prepare_bucket_io``). The two
    ``powerClass`` terms follow
    :func:`~recovar.em.sparse_pass2.sparse_pass2_scoring.relion_powerclass_noise_presence`.
    """

    has_xi2, has_norm = relion_powerclass_noise_presence(
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        accumulate_noise=accumulate_noise,
        current_size=current_size,
    )
    exact_bpref = bool(relion_exact_bpref_operands)
    return ResidentOperandPresence(
        has_recon_weight=exact_bpref,
        has_direct_ctf_rfloat=exact_bpref,
        has_highres_xi2=has_xi2,
        has_relion_norm_high_shell=has_norm,
    )


def describe_resident_operand_mismatch(predicted, real) -> str:
    """Name every field where a predicted operand tree differs from the real one.

    Returns an empty string when they agree. This is the warm-up's own check:
    the prediction is made before the preparation runs and compared after, so
    it cannot be satisfied by construction.
    """

    import dataclasses

    problems = []
    for field in dataclasses.fields(ResidentHalfOperands):
        got, want = getattr(predicted, field.name), getattr(real, field.name)
        if field.name.startswith("n_"):
            if int(got) != int(want):
                problems.append(f"{field.name}: {int(got)} vs {int(want)}")
            continue
        if (got is None) != (want is None):
            problems.append(
                f"{field.name}: predicted {'absent' if got is None else 'present'}, "
                f"really {'absent' if want is None else 'present'}"
            )
            continue
        if got is None:
            continue
        if tuple(int(d) for d in got.shape) != tuple(int(d) for d in want.shape):
            problems.append(f"{field.name}: shape {tuple(got.shape)} vs {tuple(want.shape)}")
        elif jnp.dtype(got.dtype) != jnp.dtype(want.dtype):
            problems.append(f"{field.name}: dtype {got.dtype} vs {want.dtype}")
    return "; ".join(problems)


def resident_half_operand_avals(
    *,
    n_images: int,
    n_score_pixels: int,
    n_recon_pixels: int,
    n_rect_pixels: int,
    n_noise_shells: int,
    n_fine_trans: int,
    score_complex_dtype,
    score_real_dtype,
    acc_real_dtype,
    rfloat_ctf_dtype=None,
    norm_high_shell_dtype=None,
    has_recon_weight: bool,
    has_direct_ctf_rfloat: bool,
    has_highres_xi2: bool = True,
    has_relion_norm_high_shell: bool = True,
    has_optics_groups: bool = False,
) -> "ResidentHalfOperands":
    """The half's operands as avals, without preparing them (P4-J).

    Same inputs as :func:`resident_half_operand_bytes`, which the driver's
    admission check already computes before any device work: the shapes of this
    half's operands are decided by the current size, the window and the image
    count, all of them known before pass 1 runs. This returns them as
    ``jax.ShapeDtypeStruct`` so a program that consumes the operands can be
    lowered and compiled ahead of the preparation that fills them.

    Only the SHAPES are encoded here. The dtypes are the caller's, because the
    precision policy owns them and a second copy of that decision would be a
    second place to get it wrong; ``has_*`` say which optional operands the
    configuration produces.

    ``norm_high_shell_dtype`` is separate because the norm high-shell term is
    not the policy's real dtype in production: source-faithful normalization
    accumulates it in float64 while ``highres_Xi2`` stays float32. Leave it
    ``None`` only when the caller knows the two agree; the driver takes it from
    :func:`~recovar.em.sparse_pass2.sparse_pass2_scoring.relion_powerclass_noise_dtypes`.
    Getting this wrong is what the driver's after-the-fact comparison caught on
    2026-09-20, before any measurement was believed.

    ``rfloat_ctf_dtype`` is separate and defaults to float64 because that is
    what :func:`resident_half_operand_bytes` assumes for the exact RELION CTF
    (its ``rfloat_ctf_bytes`` default is 8, and the driver leaves it at the
    default while passing the policy's dtypes for everything else). Whether the
    stored array really is float64 is not settled here: the CPU test only holds
    this function and the byte estimate to the same story, and the GPU test
    against :func:`prepare_resident_half_operands` is what decides it. If they
    disagree, the admission check is over-estimating by four bytes per
    reconstruction pixel per image, which is conservative and therefore safe,
    but this function would be wrong and the GPU test is how that surfaces.

    Nothing here allocates or touches a device buffer.
    """

    n_images = int(n_images)
    capacity = resident_image_capacity(n_images)
    score_shape = (capacity, int(n_score_pixels))
    recon_shape = (capacity, int(n_recon_pixels))
    per_image = (capacity,)

    def aval(shape, dtype):
        return jax.ShapeDtypeStruct(tuple(shape), jnp.dtype(dtype))

    return ResidentHalfOperands(
        n_images=n_images,
        n_image_capacity=capacity,
        n_score_pixels=int(n_score_pixels),
        n_recon_pixels=int(n_recon_pixels),
        n_rect_pixels=int(n_rect_pixels),
        n_noise_shells=int(n_noise_shells),
        n_fine_trans=int(n_fine_trans),
        score_input=aval(score_shape, score_complex_dtype),
        corr_img_score=aval(score_shape, score_real_dtype),
        highres_xi2_half=aval(per_image, score_real_dtype) if has_highres_xi2 else None,
        translation_prior=aval((capacity, int(n_fine_trans)), score_real_dtype),
        recon_image=aval(recon_shape, score_complex_dtype),
        recon_weight=aval(recon_shape, acc_real_dtype) if has_recon_weight else None,
        noise_image=aval(recon_shape, score_complex_dtype),
        ctf2_over_nv_recon=aval(recon_shape, acc_real_dtype),
        direct_ctf_rfloat_recon=(
            aval(recon_shape, rfloat_ctf_dtype if rfloat_ctf_dtype is not None else jnp.float64)
            if has_direct_ctf_rfloat
            else None
        ),
        wavg_image_rect=aval((capacity, int(n_rect_pixels)), score_complex_dtype),
        image_power_shells=aval((capacity, int(n_noise_shells)), jnp.float64),
        relion_norm_high_shell=(
            aval(
                per_image,
                score_real_dtype if norm_high_shell_dtype is None else norm_high_shell_dtype,
            )
            if has_relion_norm_high_shell
            else None
        ),
        scale=aval(per_image, jnp.float32),
        group_ids=aval(per_image, jnp.int32),
        optics_groups=aval(per_image, jnp.int32) if has_optics_groups else None,
    )


def resident_operands_max_bytes(available_bytes: float | None = None) -> int:
    """Budget for one half's resident per-image operands and their preparation peak.

    ``available_bytes`` is what the allocator can still hand out
    (``sparse_pass2_budget.device_available_bytes``); unknown falls back to a
    fixed 6 GiB. ``RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES`` overrides.
    """

    override = os.environ.get(_RESIDENT_OPERAND_BYTES_ENV, "").strip()
    if override:
        value = int(override)
        if value <= 0:
            raise ValueError(f"{_RESIDENT_OPERAND_BYTES_ENV} must be positive, got {value}")
        return value
    if available_bytes is None:
        return _DEFAULT_RESIDENT_OPERAND_MAX_BYTES
    return max(1, int(float(available_bytes) * _RESIDENT_OPERAND_AVAILABLE_FRACTION))


def _prepare_image_batch_size() -> int:
    raw = os.environ.get(_PREPARE_IMAGE_BATCH_ENV, "").strip()
    if not raw:
        return _DEFAULT_PREPARE_IMAGE_BATCH
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{_PREPARE_IMAGE_BATCH_ENV} must be positive, got {value}")
    return value


def _require_supported(condition: bool, message: str) -> None:
    if not condition:
        raise ResidentOperandsUnsupported(
            "the once-per-half resident operand preparation does not cover " + message
        )


@partial(jax.jit, donate_argnums=(0,))
def _place_batch(buffer: jax.Array, batch: jax.Array, start) -> jax.Array:
    """Write one preparation batch into its capacity buffer at runtime row ``start``.

    Pure data movement. ``start`` is a traced scalar, so one program serves
    every batch of a (capacity, batch size, operand) triple; a concatenation of
    the batches was keyed on their count, which follows the subset size.
    """

    return jax.lax.dynamic_update_slice_in_dim(buffer, batch, start, axis=0)


@jax.jit
def _reorder_rows(buffer: jax.Array, reorder: jax.Array) -> jax.Array:
    """Put one operand's rows in dataset order: a gather, no arithmetic."""

    return buffer[reorder]


class _BatchWindowInputs(NamedTuple):
    """The per-image-batch arrays the window gather and cast consume.

    ``weighted_ctf_half`` and ``ctf_half_rfloat`` are ``None`` on the paths
    that do not produce them; ``None`` is a pytree structure, so those paths
    key their own program instead of carrying a dead operand.
    """

    ctf2_over_nv_half: jax.Array
    sparse_score_input_half: jax.Array
    processed_score_half_for_noise: jax.Array
    recon_input_half: jax.Array
    weighted_ctf_half: jax.Array | None
    score_weighted_half: jax.Array
    ctf2_over_nv_recon_half: jax.Array
    ctf_half_rfloat: jax.Array | None
    dc_mask: jax.Array | None
    score_indices: jax.Array
    recon_indices: jax.Array
    rect_indices: jax.Array


@partial(
    jax.jit,
    static_argnames=("mask_dc", "score_real_dtype", "score_complex_dtype", "acc_real_dtype"),
)
def _batch_window_operands(
    arrays: _BatchWindowInputs,
    *,
    mask_dc: bool,
    score_real_dtype,
    score_complex_dtype,
    acc_real_dtype,
) -> dict:
    """The window gather and cast of one image batch, as one program.

    Same statements, same order, same dtypes as the loose dispatch: a DC mask
    on the score correction, seven ``values[:, indices]`` gathers and their
    casts. Nothing here is arithmetic, so the values cannot move; what leaves
    the host is the dispatch count. Each ``values[:, indices]`` was five eager
    primitives, not one -- JAX normalizes a fancy index eagerly (``add``,
    ``broadcast_in_dim``, ``select_n``) before the gather -- and the driver
    runs one of these per image batch per half, which on the early and hp3
    states was 1400 eager dispatches per iteration for a constant index array.

    ``weighted_ctf_half is None`` selects the non-BPref reconstruction operand
    and ``ctf_half_rfloat is None`` the path with no direct RFLOAT CTF, exactly
    as the Python branches did.
    """

    ctf2_score = arrays.ctf2_over_nv_half
    if mask_dc:
        ctf2_score = jnp.where(arrays.dc_mask[None, :], 0.0, ctf2_score)
    score_indices = arrays.score_indices
    recon_indices = arrays.recon_indices
    batch_arrays = {
        "score_input": arrays.sparse_score_input_half[:, score_indices],
        "corr_img_score": ctf2_score[:, score_indices].astype(score_real_dtype),
        "wavg_image_rect": arrays.processed_score_half_for_noise[:, arrays.rect_indices],
        "recon_image": jnp.asarray(
            arrays.recon_input_half[:, recon_indices], dtype=score_complex_dtype
        ),
        "noise_image": jnp.asarray(
            arrays.score_weighted_half[:, recon_indices], dtype=score_complex_dtype
        ),
        "ctf2_over_nv_recon": arrays.ctf2_over_nv_recon_half[:, recon_indices],
    }
    if arrays.weighted_ctf_half is not None:
        batch_arrays["recon_weight"] = jnp.asarray(
            arrays.weighted_ctf_half[:, recon_indices], dtype=acc_real_dtype
        )
    if arrays.ctf_half_rfloat is not None:
        batch_arrays["direct_ctf_rfloat_recon"] = arrays.ctf_half_rfloat[:, recon_indices]
    return batch_arrays


def prepare_resident_half_operands(
    experiment_dataset,
    image_indices,
    *,
    bucket_io_kwargs: dict,
    window_indices,
    recon_window_indices,
    wavg_rect_indices,
    noise_shell_indices_half,
    n_noise_shells: int,
    image_shape,
    current_size,
    n_fine_trans: int,
    use_exact_relion_gaussian: bool,
    accumulate_noise: bool,
    source_faithful_spectrum_norm: bool,
    fine_translation_prior_2d=None,
    scale_corrections_np=None,
    group_ids_np=None,
    precision_policy: DensePrecisionPolicy | None = None,
    image_batch_size: int | None = None,
    optics_groups_np=None,
    relion_native_fine_units: bool = False,
) -> ResidentHalfOperands:
    """Run the per-image preparation once for ``image_indices`` and keep it resident.

    ``bucket_io_kwargs`` is the driver's own ``_prepare_bucket_io`` keyword set,
    so this call reads exactly the configuration the per-chunk path would have
    read; the translation-only keywords are ignored here because nothing in the
    per-image half depends on them.

    ``image_batch_size`` only decides how many images one preparation call
    covers. The preparation is per image, so it does not change any value; the
    last batch is short rather than padded, unlike the per-chunk path, which had
    to pad to a capacity class to keep one program per class.

    ``relion_native_fine_units`` stores the two score operands in RELION's
    native FFT units, as the compact engine scores a fresh K=1 pass
    (:func:`~relax.sparse_pass2.sparse_pass2_scoring._relion_native_fine_units_enabled`):
    ``score_input`` divided by N**2 and ``corr_img_score`` RELION's native
    ``corr_img``. Every reconstruction and noise operand keeps RECOVAR units.

    ``wavg_rect_indices`` is RELION's Wavg rectangle and
    ``noise_shell_indices_half`` / ``n_noise_shells`` the noise-shell binning of
    the packed half the statistics use; they decide ``wavg_image_rect`` and
    ``image_power_shells``.
    """

    image_indices = np.asarray(image_indices)
    if image_indices.ndim != 1 or image_indices.size == 0:
        raise ValueError(f"image_indices must be a non-empty 1-D array, got {image_indices.shape}")
    n_images = int(image_indices.shape[0])
    position_of = {int(index): position for position, index in enumerate(image_indices.tolist())}
    if len(position_of) != n_images:
        raise ValueError("image_indices must not repeat an image")

    kwargs = dict(bucket_io_kwargs)
    relion_exact_bpref_operands = bool(kwargs.get("relion_exact_bpref_operands", False))
    score_with_masked_images = bool(kwargs.get("score_with_masked_images", False))
    half_spectrum_scoring = bool(kwargs.get("half_spectrum_scoring", False))
    use_float64_scoring = bool(kwargs.get("use_float64_scoring", False))
    score_mode = kwargs.get("score_mode", "gaussian")

    # The M-step reads the unshifted operands through T15's float32 kernel, and
    # that kernel knows two translate conventions: BPref for the reconstruction
    # operand and score for the noise operand. Without masked scoring the two
    # operands are the same array in the per-chunk path, which means the noise
    # sum is taken over the *BPref*-translated tile; the kernel cannot express
    # that pairing, so refuse it rather than change the arithmetic.
    _require_supported(score_with_masked_images, "unmasked scoring (score_with_masked_images=0)")
    _require_supported(not use_float64_scoring, "float64 scoring")
    _require_supported(score_mode == "gaussian", f"score_mode={score_mode!r}")
    _require_supported(
        kwargs.get("relion_score_translation_angles", None) is not None,
        "a pass without RELION translation angles",
    )
    _require_supported(not bool(kwargs.get("score_only", False)), "score-only preparation")
    _require_supported(window_indices is not None, "an unwindowed score spectrum")
    _require_supported(recon_window_indices is not None, "an unwindowed reconstruction spectrum")

    precision_policy = precision_policy or DensePrecisionPolicy(
        use_float64_scoring=use_float64_scoring
    )
    score_real_dtype = precision_policy.score_real_dtype
    score_complex_dtype = precision_policy.score_complex_dtype

    unshifted_kwargs = dict(
        noise_variance_half=kwargs["noise_variance_half"],
        config=kwargs["config"],
        score_with_masked_images=score_with_masked_images,
        image_corrections=kwargs["image_corrections"],
        scale_corrections=kwargs["scale_corrections"],
        image_pre_shifts=kwargs["image_pre_shifts"],
        use_float64_scoring=use_float64_scoring,
        return_direct_scoring_io=True,
        score_only=False,
        score_mode=score_mode,
        window_indices=window_indices,
        relion_exact_normalized_cc_operands=bool(
            kwargs.get("relion_exact_normalized_cc_operands", False)
        ),
        relion_exact_bpref_operands=relion_exact_bpref_operands,
        noise_optics_groups=kwargs.get("noise_optics_groups"),
    )

    score_indices = jnp.asarray(window_indices, dtype=jnp.int32)
    recon_indices = jnp.asarray(recon_window_indices, dtype=jnp.int32)
    rect_indices = jnp.asarray(wavg_rect_indices, dtype=jnp.int32)
    noise_shell_indices_half = jnp.asarray(noise_shell_indices_half, dtype=jnp.int32)
    dc_mask = None
    if half_spectrum_scoring:
        dc_mask = jnp.asarray(make_shell_indices_half(image_shape)) == 0

    # Batch outputs stay on the device, written into capacity buffers; only the
    # dataset's returned order comes back to the host, as a permutation.
    image_capacity = resident_image_capacity(n_images)
    buffers: dict[str, jax.Array] = {}
    fetched_order: list[np.ndarray] = []
    optional_available: dict[str, bool | None] = {
        "recon_weight": None,
        "direct_ctf_rfloat_recon": None,
        "highres_xi2_half": None,
        "relion_norm_high_shell": None,
    }
    batch_size = int(image_batch_size or _prepare_image_batch_size())
    buffer_rows = -(-image_capacity // batch_size) * batch_size
    native_fft_size = int(np.prod(image_shape))
    if relion_native_fine_units:
        _require_supported(
            relion_exact_bpref_operands,
            "native-unit fine scores without RELION's RFLOAT CTF operand",
        )

    for start in range(0, n_images, batch_size):
        batch_image_indices = image_indices[start : start + batch_size]
        batch_data, ctf_params, fetched_indices = fetch_indexed_batch(
            experiment_dataset, batch_image_indices
        )
        fetched_indices = np.asarray(fetched_indices)
        n_fetched = int(fetched_indices.shape[0])
        if n_fetched < batch_size:
            # The last batch repeats its first image up to the batch size, as the
            # per-chunk path pads a capacity class, so every batch of the half
            # runs the same preparation programs. The preparation is per image;
            # the repeated rows land in padding no chunk addresses.
            pad = np.concatenate([np.arange(n_fetched), np.zeros(batch_size - n_fetched, dtype=np.int64)])
            batch_data = np.asarray(batch_data)[pad]
            ctf_params = np.asarray(ctf_params)[pad]
            prepared_indices = fetched_indices[pad]
        else:
            prepared_indices = fetched_indices
        unshifted = prepare_unshifted_bucket_operands(
            experiment_dataset,
            jnp.asarray(batch_data),
            ctf_params,
            prepared_indices,
            **unshifted_kwargs,
        )
        score_corr_img_half = unshifted.ctf2_over_nv_half
        if relion_native_fine_units:
            # The native corr_img the compact engine scores with; the DC mask
            # and window gather below apply to it exactly as to the RECOVAR one.
            score_corr_img_half = _relion_native_score_corr_img(
                pixel_rows(unshifted.noise_variance_half),
                unshifted.ctf_half_rfloat,
                image_shape,
                (
                    jnp.asarray(unshifted.batch_scale_np, dtype=jnp.float32)[:, None]
                    if kwargs["scale_corrections"] is not None
                    else None
                ),
                zero_dc=half_spectrum_scoring,
            )

        batch_arrays = _batch_window_operands(
            _BatchWindowInputs(
                ctf2_over_nv_half=score_corr_img_half,
                sparse_score_input_half=unshifted.sparse_score_input_half,
                processed_score_half_for_noise=unshifted.processed_score_half_for_noise,
                recon_input_half=(
                    unshifted.recon_bpref_input_half
                    if relion_exact_bpref_operands
                    else unshifted.recon_weighted_half
                ),
                weighted_ctf_half=(
                    unshifted.weighted_ctf_half if relion_exact_bpref_operands else None
                ),
                score_weighted_half=unshifted.score_weighted_half,
                ctf2_over_nv_recon_half=unshifted.ctf2_over_nv_recon_half,
                ctf_half_rfloat=unshifted.ctf_half_rfloat,
                dc_mask=dc_mask,
                score_indices=score_indices,
                recon_indices=recon_indices,
                rect_indices=rect_indices,
            ),
            mask_dc=bool(half_spectrum_scoring and not unshifted.use_normalized_cc),
            score_real_dtype=jnp.dtype(score_real_dtype),
            score_complex_dtype=jnp.dtype(score_complex_dtype),
            acc_real_dtype=jnp.dtype(unshifted.acc_real_dtype),
        )
        if relion_native_fine_units:
            # The kernel translates this unshifted image in-kernel, so the pass
            # scores translate(image / N**2); the compact engine divides the
            # already translated image. Each is one correctly rounded division,
            # so the two agree to rounding, not bit for bit.
            batch_arrays["score_input"] = _relion_native_fine_units(
                batch_arrays["score_input"], native_fft_size
            )

        batch_arrays["image_power_shells"] = image_power_shells(
            unshifted.processed_score_half_for_noise,
            noise_shell_indices_half,
            shell_count=int(n_noise_shells),
        )

        highres_xi2_half, relion_norm_high_shell = _relion_powerclass_noise_terms(
            unshifted.processed_score_half_for_noise,
            image_shape=image_shape,
            current_size=current_size,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        )
        if highres_xi2_half is not None:
            batch_arrays["highres_xi2_half"] = highres_xi2_half
        if relion_norm_high_shell is not None:
            batch_arrays["relion_norm_high_shell"] = relion_norm_high_shell

        for name, flag in optional_available.items():
            present = name in batch_arrays
            if flag is None:
                optional_available[name] = present
            elif flag != present:
                raise ValueError(f"{name} availability changed between image batches")

        for name, value in batch_arrays.items():
            if name not in buffers:
                buffers[name] = jnp.zeros((buffer_rows,) + tuple(value.shape[1:]), dtype=value.dtype)
            buffers[name] = _place_batch(buffers[name], value, np.int32(start))
        fetched_order.append(fetched_indices)

    fetched_all = np.concatenate(fetched_order)
    if fetched_all.shape[0] != n_images:
        raise ValueError(
            f"the dataset returned {fetched_all.shape[0]} images for {n_images} requested"
        )
    destination = np.empty(n_images, dtype=np.int64)
    for order, dataset_index in enumerate(fetched_all.tolist()):
        position = position_of.get(int(dataset_index))
        if position is None:
            raise ValueError(
                f"the dataset returned image {dataset_index}, which is not in image_indices"
            )
        destination[order] = position
    inverse = np.empty(n_images, dtype=np.int64)
    inverse[destination] = np.arange(n_images, dtype=np.int64)
    if np.unique(destination).size != n_images:
        raise ValueError("the dataset did not return every requested image exactly once")
    # Rows past n_images keep their buffer rows: padding no chunk addresses.
    reorder = jnp.asarray(np.concatenate([inverse, np.arange(n_images, image_capacity, dtype=np.int64)]))

    def stack(name, required=False):
        present = name in buffers
        if required and not present:
            raise ValueError(f"the per-image preparation did not produce {name}")
        if not present or not optional_available.get(name, True):
            return None
        # Popped so each capacity buffer is freed once its reordered copy
        # exists: the preparation peaks at the operands plus one array.
        return _reorder_rows(buffers.pop(name), reorder)

    translation_prior_np = np.zeros((image_capacity, int(n_fine_trans)), dtype=score_real_dtype)
    if fine_translation_prior_2d is not None:
        translation_prior_np[:n_images] = np.asarray(fine_translation_prior_2d)[image_indices]
    translation_prior = jnp.asarray(translation_prior_np)

    scale = np.ones(image_capacity, dtype=np.float32)
    if scale_corrections_np is not None:
        scale[:n_images] = np.asarray(scale_corrections_np, dtype=np.float32)[image_indices]
    group_ids = np.full(image_capacity, -1, dtype=np.int32)
    if group_ids_np is not None:
        group_ids[:n_images] = np.asarray(group_ids_np, dtype=np.int32)[image_indices]
    optics_groups = None
    if optics_groups_np is not None:
        optics_groups = np.zeros(image_capacity, dtype=np.int32)
        optics_groups[:n_images] = np.asarray(optics_groups_np, dtype=np.int32)[image_indices]

    score_input = stack("score_input", required=True)
    recon_image = stack("recon_image", required=True)
    wavg_image_rect = stack("wavg_image_rect", required=True)
    power_shells = stack("image_power_shells", required=True)
    operands = ResidentHalfOperands(
        n_images=n_images,
        n_image_capacity=image_capacity,
        n_score_pixels=int(score_input.shape[1]),
        n_recon_pixels=int(recon_image.shape[1]),
        n_rect_pixels=int(wavg_image_rect.shape[1]),
        n_noise_shells=int(power_shells.shape[1]),
        n_fine_trans=int(n_fine_trans),
        score_input=score_input,
        corr_img_score=stack("corr_img_score", required=True),
        highres_xi2_half=stack("highres_xi2_half"),
        translation_prior=translation_prior,
        recon_image=recon_image,
        recon_weight=stack("recon_weight"),
        noise_image=stack("noise_image", required=True),
        ctf2_over_nv_recon=stack("ctf2_over_nv_recon", required=True),
        direct_ctf_rfloat_recon=stack("direct_ctf_rfloat_recon"),
        wavg_image_rect=wavg_image_rect,
        image_power_shells=power_shells,
        relion_norm_high_shell=stack("relion_norm_high_shell"),
        scale=jnp.asarray(scale),
        group_ids=jnp.asarray(group_ids),
        optics_groups=None if optics_groups is None else jnp.asarray(optics_groups),
    )
    logger.info(
        "Resident pass-2 per-half operands: %d images, %d score / %d recon / %d Wavg pixels, "
        "%.2f GiB resident (%d preparation calls)",
        operands.n_images,
        operands.n_score_pixels,
        operands.n_recon_pixels,
        operands.n_rect_pixels,
        operands.nbytes()["total"] / float(1024**3),
        (n_images + batch_size - 1) // batch_size,
    )
    # A separate line, not an addition to the one above: the census parsers key
    # on that line's text. The batch size and the last batch's remainder are
    # what set the leading extent of every program inside the preparation, so
    # P4-A's shape census could not classify those extents without them.
    logger.info(
        "Resident pass-2 per-half operand batching: image_batch_size=%d, "
        "last_batch_images=%d, image_capacity=%d",
        batch_size,
        n_images - batch_size * ((n_images - 1) // batch_size) if n_images else 0,
        image_capacity,
    )
    return operands


def _gather_rows(values, safe_slots, valid, fill=0):
    """Take one row per chunk slot, with the padded slots set to ``fill``."""

    if values is None:
        return None
    values = jnp.asarray(values)
    gathered = values[safe_slots]
    mask = valid.reshape((-1,) + (1,) * (gathered.ndim - 1))
    return jnp.where(mask, gathered, jnp.asarray(fill, dtype=gathered.dtype))


@partial(jax.jit, static_argnames=("image_shape",))
def _gather_chunk_arrays(
    image_slots,
    score_input,
    corr_img_score,
    highres_xi2_half,
    translation_prior,
    recon_image,
    recon_weight,
    noise_image,
    ctf2_over_nv_recon,
    direct_ctf_rfloat_recon,
    wavg_image_rect,
    power_shells,
    relion_norm_high_shell,
    scale,
    group_ids,
    optics_groups,
    translation_angles,
    rect_indices,
    exact_positions,
    *,
    image_shape,
):
    """One program per (half size, capacity class): the chunk's whole operand set.

    Every array is taken by image id and the padded slots are zeroed, which is
    what the per-chunk preparation's capacity mask did. ``scale`` keeps 1 on a
    padded slot so the Wavg kernel never divides by zero, and ``group_ids``
    keeps -1 so the scale accumulators drop it; both are the per-chunk path's
    own padding values, and neither is observable because a padded slot's
    posterior is zero.

    A padded slot reads the chunk's *first* image rather than image zero, which
    is what ``_pad_batch_to_capacity`` fed the per-chunk preparation. Nothing
    downstream can see the difference, since every padded row is zeroed here,
    but it keeps the two paths' intermediate batches identical.
    """

    image_slots = jnp.asarray(image_slots, dtype=jnp.int32)
    valid = image_slots >= 0
    safe_slots = jnp.where(valid, image_slots, image_slots[0])

    raw_translated_wavg_rectangle = relion_cuda_translate_wavg_norm_window(
        _gather_rows(wavg_image_rect, safe_slots, valid),
        translation_angles,
        rect_indices,
        image_shape,
    )
    return (
        _gather_rows(score_input, safe_slots, valid),
        _gather_rows(corr_img_score, safe_slots, valid),
        _gather_rows(highres_xi2_half, safe_slots, valid),
        _gather_rows(translation_prior, safe_slots, valid),
        _gather_rows(recon_image, safe_slots, valid),
        _gather_rows(recon_weight, safe_slots, valid),
        _gather_rows(noise_image, safe_slots, valid),
        _gather_rows(ctf2_over_nv_recon, safe_slots, valid),
        _gather_rows(direct_ctf_rfloat_recon, safe_slots, valid),
        _gather_rows(power_shells, safe_slots, valid),
        _gather_rows(relion_norm_high_shell, safe_slots, valid),
        _gather_rows(scale, safe_slots, valid, fill=1.0),
        _gather_rows(group_ids, safe_slots, valid, fill=-1),
        _gather_rows(optics_groups, safe_slots, valid),
        raw_translated_wavg_rectangle,
        raw_translated_wavg_rectangle[:, :, jnp.asarray(exact_positions, dtype=jnp.int32)],
    )


def gather_resident_chunk_operands(
    operands: ResidentHalfOperands,
    image_slots,
    *,
    translation_angles,
    rect_indices,
    exact_positions,
    image_shape,
) -> dict:
    """Gather one chunk's operands out of the half's resident arrays.

    ``image_slots`` is the chunk's capacity-shaped image id vector, ``-1`` on a
    padded slot. The returned keys are the ones the driver's chunk stage
    operands take, so the per-chunk preparation and this gather are
    interchangeable at the call site.
    """

    (
        score_input,
        corr_img_score,
        highres_xi2_half,
        translation_prior,
        recon_image,
        recon_weight,
        noise_image,
        ctf2_over_nv_recon,
        direct_ctf_rfloat_recon,
        power_shells,
        relion_norm_high_shell,
        scale,
        group_ids,
        optics_groups,
        raw_translated_wavg_rectangle,
        raw_translated_wavg_for_atomic,
    ) = _gather_chunk_arrays(
        jnp.asarray(image_slots, dtype=jnp.int32),
        operands.score_input,
        operands.corr_img_score,
        operands.highres_xi2_half,
        operands.translation_prior,
        operands.recon_image,
        operands.recon_weight,
        operands.noise_image,
        operands.ctf2_over_nv_recon,
        operands.direct_ctf_rfloat_recon,
        operands.wavg_image_rect,
        operands.image_power_shells,
        operands.relion_norm_high_shell,
        operands.scale,
        operands.group_ids,
        operands.optics_groups,
        jnp.asarray(translation_angles, dtype=jnp.float32),
        jnp.asarray(rect_indices, dtype=jnp.int32),
        jnp.asarray(exact_positions, dtype=jnp.int32),
        image_shape=tuple(int(size) for size in image_shape),
    )
    return {
        "score_input": score_input,
        "corr_img_score": corr_img_score,
        "highres_xi2_half": highres_xi2_half,
        "translation_prior": translation_prior,
        "recon_image": recon_image,
        "recon_weight": recon_weight,
        "noise_image": noise_image,
        "ctf2_over_nv_recon": ctf2_over_nv_recon,
        "direct_ctf_rfloat_recon": direct_ctf_rfloat_recon,
        "image_power_shells": power_shells,
        "relion_norm_high_shell": relion_norm_high_shell,
        "raw_translated_wavg_rectangle": raw_translated_wavg_rectangle,
        "raw_translated_wavg_for_atomic": raw_translated_wavg_for_atomic,
        "scale": scale,
        "group_ids": group_ids,
        "optics_groups": optics_groups,
    }
