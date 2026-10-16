"""Per-rotation-block noise statistics for RELION's sigma2_noise update.

``compute_noise_block`` bins one rotation block's posterior-weighted ``A2 - 2*XA`` noise contribution to
resolution shells; ``compute_noise_block_per_optics_group`` does the same with one noise spectrum per optics
group. Both take projections already computed and project nothing. The definitions moved verbatim from
``relax.projection.projection``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from relax.fourier.half_spectrum import bin_shell_values_jax


@partial(jax.jit, static_argnums=(6, 7), static_argnames=("shell_count", "return_split"))
def compute_noise_block(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    shell_count,
    return_split: bool = True,
):
    """Accumulate RELION-style posterior-weighted noise for one rotation block.

    Uses the decomposition::

        E_w[|CTF*proj - img|^2] = E_w[|CTF*proj|^2] - 2*Re(E_w[conj(img)*CTF*proj]) + |img|^2
                                 =     A2            -           2*XA                  + P_img

    ``P_img`` is handled by the caller (image-only, no rotation dependence).
    This function computes the ``A2 - 2*XA`` contribution from one rotation
    block, binned to resolution shells. Inputs are un-Hermitian-weighted packed
    half spectra because RELION's noise update bins over its FFTW half-plane
    convention directly.

    The returned per-shell blocks keep whatever real dtype ``a2``/``xa``
    naturally promote to from the inputs (float32 if all inputs are float32,
    float64 if any is float64) -- no explicit cast is applied. Callers
    accumulate many of these blocks (one per rotation block/microbatch,
    across every particle) into a single running sigma2_noise sum; an
    unconditional float32 cast here used to truncate every block before that
    accumulation, compounding error across the whole reduction, unlike
    RELION's own ``wsum_model.sigma2_noise``, which stays RFLOAT (double
    under double-precision builds) through the entire per-particle
    accumulation. Pass float64 inputs (matching ``use_float64_scoring``/
    ``use_float64_projections``) to get float64 accumulation here too.
    """
    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance_half, 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    a2 = jnp.sum(a2_terms, axis=0)

    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0)
    cross = jnp.sum(cross_terms, axis=0)
    xa = jnp.where(cross.real != 0.0, noise_variance_half * cross.real, 0.0)
    block_noise = a2 - 2.0 * xa

    noise_shells = bin_shell_values_jax(block_noise, shell_indices, shell_count)
    if not return_split:
        zeros = jnp.zeros(shell_count, dtype=block_noise.dtype)
        return noise_shells, zeros, zeros
    a2_shells = bin_shell_values_jax(a2, shell_indices, shell_count)
    xa_shells = bin_shell_values_jax(xa, shell_indices, shell_count)
    return noise_shells, a2_shells, xa_shells


def compute_noise_block_per_optics_group(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_table,
    row_optics_groups,
    shell_indices,
    shell_count,
):
    """:func:`compute_noise_block` with one noise spectrum per optics group.

    Row ``r`` carries its image's group ``row_optics_groups[r]`` and is weighted by
    that group's row of ``noise_table`` ``[G, P]``. The ``A2 - 2*XA`` terms are summed
    per group, statement for statement as the one-group function sums them over all
    rows (``A2`` per row, then the group's noise times its summed cross term), and
    binned to ``[G, shell_count]``: RELION's ``wsum_model.sigma2_noise[optics_group]``.

    The rows are summed into their groups by one matrix product, the one-hot
    membership ``[G, rows]`` times the rows' ``A2`` terms and real cross terms
    ``[rows, 2P]`` (float32 products at full precision, no TF32): one pass over the
    rows for any number of groups, and an absent group's shells are exactly zero.
    The masked reduction over ``[rows, G, P]`` it replaces did G times the elementwise
    work: with 130 optics groups it was 62% of the kernel time of a real subtomogram
    final-iteration M-step block. The product sums each group's rows in another order
    than that reduction did, so the shells differ from it in float32 rounding (about
    1e-7 relative), not bitwise; cuBLAS's order is fixed (no atomics). A segment_sum is a
    scatter-add of every row onto G rows: thousands of atomics per address, 47 ms per
    8192-row M-step block (87% of the multishape Class3D iteration-2 kernel time).
    """
    n_groups = int(noise_table.shape[0])
    noise_rows = noise_table[row_optics_groups]
    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_rows, 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0).real
    n_pixels = int(a2_terms.shape[1])
    dtype = jnp.result_type(a2_terms.dtype, cross_terms.dtype)
    member = (jnp.arange(n_groups, dtype=row_optics_groups.dtype)[:, None] == row_optics_groups[None, :]).astype(dtype)
    sums = jnp.matmul(
        member,
        jnp.concatenate([a2_terms.astype(dtype), cross_terms.astype(dtype)], axis=1),
        precision=jax.lax.Precision.HIGHEST,
    )
    a2, cross = sums[:, :n_pixels], sums[:, n_pixels:]
    xa = jnp.where(cross != 0.0, noise_table * cross, 0.0)
    block_noise = a2 - 2.0 * xa
    return jax.vmap(lambda values: bin_shell_values_jax(values, shell_indices, shell_count))(
        block_noise
    )
