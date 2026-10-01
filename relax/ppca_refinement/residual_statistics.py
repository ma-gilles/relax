"""Direct expected-residual image statistics (algorithm sections 4–5 and 10).

The gradient is formed before the adjoint: a gridded latent block is a metric,
not the interpolated normal operator. All inputs share one observation metric.
"""

from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np
from recovar.ppca.triangular import unpack_tri_to_full


def full_float32(function):
    """Prevent TF32 contractions in the new production PPCA path (no FP64)."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with jax.default_matmul_precision("highest"):
            return function(*args, **kwargs)

    return wrapped


def residual_statistics_precision(function):
    """Use full float32 for residual collection, preserving legacy refinement."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        if kwargs.get("collect_residuals", False):
            return full_float32(function)(*args, **kwargs)
        return function(*args, **kwargs)

    return wrapped


@jax.jit
def residual_image_statistics(score, alpha, G_tri, logZ, Y1, ctf2_over_noise, projections):
    """Return weighted residual images, noise-power correction and embeddings.

    Y1 and ctf2_over_noise include Hermitian weights / coefficient variance.
    Noise correction is in that same weighted metric; multiply by variance and
    add measured image power before shell reduction. No division by CTF occurs.
    Arrays are bounded by a single image/rotation block.
    """
    gamma = jnp.exp(score - logZ[:, None, None])
    p = alpha.shape[-1]
    G = unpack_tri_to_full(G_tri, p)
    rhs = jnp.einsum("btr,btrp,btf->prf", gamma, alpha, Y1)
    prediction = jnp.einsum("btr,btrpq,rqf,bf->prf", gamma, G, projections, ctf2_over_noise)
    cross = jnp.einsum("btr,btrp,btf,rpf->f", gamma, alpha, Y1, projections.conj()).real
    power = jnp.einsum("btr,btrpq,rpf,rqf,bf->f", gamma, G, projections.conj(), projections, ctf2_over_noise).real
    embedding = jnp.einsum("btr,btrq->bq", gamma, alpha[..., 1:])
    return rhs - prediction, power - 2 * cross, embedding


@jax.jit
def residual_statistics_from_moment_images(rhs_images, lhs_tri_images, projections):
    """Residual images and noise-power correction from the M-step moment images.

    With ``R = sum_bt gamma alpha Y1`` and ``L = sum_bt gamma G ctf2_over_noise``
    (the augmented RHS ``(P, R, F)`` and packed upper LHS ``(tri(P), R, F)``
    images of one rotation block) and the windowed projections ``A``
    (``(P, R, F)``), the expected residual of :func:`residual_image_statistics`
    is ``R_p - sum_q L_pq A_q`` and its correction is
    ``sum_r [sum_pq L_pq Re(conj(A_p) A_q) - 2 sum_p Re(R_p conj(A_p))]``.
    Both are linear in ``(R, L)``, so the per-pose contractions are not repeated;
    the packed ``L`` is read in place, one symmetric entry at a time.
    The images carry whatever pixel metric ``Y1`` and ``ctf2_over_noise`` carry.
    """
    basis_size = rhs_images.shape[0]
    index = {}
    for k, (i, j) in enumerate(zip(*np.triu_indices(basis_size))):
        index[(int(i), int(j))] = index[(int(j), int(i))] = k
    prediction = jnp.stack(
        [sum(lhs_tri_images[index[(p, q)]] * projections[q] for q in range(basis_size)) for p in range(basis_size)]
    )
    power = sum(
        (1.0 if i == j else 2.0) * lhs_tri_images[k] * (projections[i].conj() * projections[j]).real
        for k, (i, j) in enumerate(zip(*np.triu_indices(basis_size)))
    )
    cross = (rhs_images * projections.conj()).real
    return rhs_images - prediction, jnp.sum(power, axis=0) - 2 * jnp.sum(cross, axis=(0, 1))
