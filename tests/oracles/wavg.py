"""RELION's Wavg rectangle triplet terms in the rectangular ``(image, rotation, pixel)`` layout.

Test oracle: moved from relax/fine_pass/wavg.py, where only tests called it (renamed without its leading underscore);
it is the rectangular form the flat-row Wavg image-power accumulation of relax.fine_pass.resident_pass2 is compared
with.
"""

import jax
import jax.numpy as jnp

from relax.fine_pass.wavg import _relion_wavg_shifted_power


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
def relion_wavg_rectangle_triplet_terms(
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
