"""RELION's norm-correction residual per image in the rectangular ``(image, rotation, pixel)`` layout.

Test oracle: moved from relax/projection/projection.py (with its broadcast helper from relax/relion/optics_groups.py),
where only tests called it; it is the independent reference the flat-row ``_flat_row_norm_and_scale_terms`` of
relax.fine_pass.resident_block_programs is compared with.
"""

import jax
import jax.numpy as jnp


def image_rotation_rows(values):
    """``[P] -> [1, 1, P]``; per-image rows ``[B, P] -> [B, 1, P]`` for ``(image, rotation, pixel)`` terms."""

    return values[None, None, :] if values.ndim == 1 else values[:, None, :]


@jax.jit
def compute_norm_residual_per_image(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
):
    """Return RELION norm-correction residual terms per image.

    This is the same ``A2 - 2*XA`` contribution as :func:`compute_noise_block`,
    but summed per image instead of binned over shells.  The caller adds the
    image-power term once per image.

    Like ``compute_noise_block``, keeps whatever real dtype the inputs
    naturally promote to -- no explicit float32 cast, which used to
    truncate this per-image residual before the caller's own accumulation
    even when the inputs were already float64 (double-precision scoring).
    """

    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * image_rotation_rows(noise_variance_half), 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    a2_per_image = jnp.sum(a2_terms, axis=(1, 2))

    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0)
    xa_terms = image_rotation_rows(noise_variance_half) * cross_terms.real
    xa_per_image = jnp.sum(xa_terms, axis=(1, 2))
    return a2_per_image - 2.0 * xa_per_image
