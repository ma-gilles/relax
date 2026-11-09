"""Per-particle contrast as a Gaussian latent tied to the mean (``docs/math/ppca_contrast.md``, model A).

The contrast ``c = 1 + sd * e``, ``e ~ N(0, 1)``, scales the mean only: ``y = A [(1 + sd e) mu + W z] + noise``. The
engines see an ordinary PPCA with one more loading, ``sd * mu``, tied to the mean volume. :func:`with_contrast_column`
builds the model volumes ``[mu, W, sd mu]``; the statistics an engine returns for them are folded back to ``[mu, W]``
by the chain rule (:func:`fold_statistics`); :func:`split_embeddings` separates a particle's contrast estimate from its
latent coordinates; :func:`updated_sd` is the EM update of ``sd`` from the second moment of ``e``. Dense and streamed
engines, and the refinement and the initial model, share these functions.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from recovar.ppca.pose_accumulators import AugmentedPPCAStats


def with_contrast_column(mean, loadings, sd: float):
    """The model volumes ``(n_voxel, q + 2)`` columns ``[mu, W_1 .. W_q, sd mu]`` of the contrast latent model.

    ``mean`` is ``(n_voxel,)`` and ``loadings`` ``(n_voxel, q)`` (``q`` may be 0); ``sd`` is the prior sd of the
    contrast about 1. The returned columns after the mean are the ``q + 1`` loadings the engine is given.
    """
    return jnp.concatenate([jnp.reshape(mean, (-1, 1)), loadings, jnp.reshape(sd * mean, (-1, 1))], axis=1)


def accumulate_dense_with_contrast(dataset, mean, loadings, sd: float, **kwargs) -> AugmentedPPCAStats:
    """The host-mask dense engine's statistics of ``[mu, W]`` under the contrast latent.

    ``kwargs`` are those of :func:`relax.ppca_refinement.dense_dataset.accumulate_dense_ppca_statistics` (it needs
    ``collect_residuals=True`` for the residual gradient and the latent second moments); the engine runs on
    ``[mu, W, sd mu]`` and its statistics are folded to ``[mu, W]``. The contrast latent is the last embedding and the
    last entry of ``diagnostics["latent_second_moment_sum"]``, the input of :func:`updated_sd`. The refinement loop's
    dense and exact-local owners call this in place of the plain engine.
    """
    from relax.ppca_refinement.dense_dataset import accumulate_dense_ppca_statistics

    geometry = kwargs.get("geometry")
    if geometry is not None:
        kwargs["geometry"] = dataclasses.replace(geometry, q=geometry.q + 1)  # the engine now has one more loading
    columns = with_contrast_column(mean, loadings, sd)
    return fold_statistics(accumulate_dense_ppca_statistics(dataset, columns[:, 0], columns[:, 1:], **kwargs), sd)


def fold_gradient(gradient, sd: float):
    """Chain rule of a per-column gradient ``(..., q + 2)`` onto ``[mu, W]``: ``(..., q + 1)``, the mean's entry
    plus ``sd`` times the tied column's."""
    return gradient[..., :-1].at[..., 0].add(sd * gradient[..., -1])


def fold_metric(lhs_tri, sd: float):
    """Fold a packed upper metric ``(..., tri(P))`` of ``P = q + 2`` columns onto ``[mu, W]``: ``(..., tri(P - 1))``.

    This is ``T^T L T`` for the map ``T`` from ``[mu, W]`` to ``[mu, W, sd mu]``: the mean's row and column take the tied
    column's, scaled by ``sd``, and the loading block is unchanged.
    """
    size = _basis_size(lhs_tri.shape[-1])
    tied, folded = size - 1, size - 1
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(*np.triu_indices(size)))}
    kept = [index[(int(i), int(j))] for i, j in zip(*np.triu_indices(folded))]
    base = lhs_tri[..., kept]
    mean_row = jnp.stack(
        [2 * sd * lhs_tri[..., index[(0, tied)]] + sd**2 * lhs_tri[..., index[(tied, tied)]]]
        + [sd * lhs_tri[..., index[(j, tied)]] for j in range(1, folded)],
        axis=-1,
    )
    return base.at[..., :folded].add(mean_row)


def fold_statistics(stats: AugmentedPPCAStats, sd: float) -> AugmentedPPCAStats:
    """``stats`` of an engine run on ``[mu, W, sd mu]`` as the statistics of ``[mu, W]``.

    The RHS, the residual gradient and the packed metric are folded (:func:`fold_gradient`, :func:`fold_metric`); the
    embeddings keep their last column, the contrast latent ``e`` (:func:`split_embeddings`). A trace-only metric cannot
    be folded: the mean's curvature needs the cross terms.
    """
    if stats.lhs_tri is None:
        raise ValueError("The contrast latent needs the packed metric; a trace-only stream does not hold it")
    return dataclasses.replace(
        stats,
        rhs=None if stats.rhs is None else fold_gradient(stats.rhs, sd),
        lhs_tri=fold_metric(stats.lhs_tri, sd),
        residual_gradient=None if stats.residual_gradient is None else fold_gradient(stats.residual_gradient, sd),
    )


def split_embeddings(embeddings, sd: float):
    """``(u, contrast)`` from ``(n, q + 1)`` embeddings of the engine's ``q + 1`` loadings: the latent coordinates
    ``u`` ``(n, q)``, which keep the particle's contrast scale (``u = c z``), and the contrast estimate
    ``1 + sd E[e]`` ``(n,)``."""
    return embeddings[:, :-1], 1.0 + sd * embeddings[:, -1]


CONTRAST_FLOOR = 0.1


def latents_over_contrast(scaled, estimate, floor: float = CONTRAST_FLOOR):
    """``(z, n_floored)``: the estimate ``u / c`` of the latent coordinates and how many particles' contrast estimate
    was below ``floor``.

    ``scaled`` is ``u`` ``(n, q)`` and ``estimate`` the contrast estimate ``(n,)`` of :func:`split_embeddings`. A
    particle with almost no signal can have an estimate near or below zero, where ``u / c`` has no meaning: it is
    divided by ``floor`` instead, and the count is returned so the caller can report it.
    """
    estimate = np.asarray(estimate)
    return np.asarray(scaled) / np.maximum(estimate, floor)[:, None], int(np.sum(estimate < floor))


def updated_sd(sd: float, second_moment: float, *, previous_weight: float, sd_min: float, sd_max: float) -> float:
    """The next prior sd of the contrast from the batch's mean second moment of ``e`` (EM for ``sd^2 = E[delta^2]``).

    ``delta = sd e`` so the batch estimate of the variance is ``sd^2 E[e^2]``; the new variance blends it with the old
    one with weight ``previous_weight`` on the old (0 on a full-data update) and the result is clipped to
    ``[sd_min, sd_max]``.
    """
    variance = previous_weight * sd**2 + (1.0 - previous_weight) * sd**2 * float(second_moment)
    return float(min(max(np.sqrt(variance), sd_min), sd_max))


def _basis_size(n_packed: int) -> int:
    """The basis size ``P`` with ``tri(P) = n_packed``."""
    size = int((np.sqrt(8 * n_packed + 1) - 1) / 2)
    if size * (size + 1) // 2 != n_packed or size < 2:
        raise ValueError(f"{n_packed} is not the packed size of a basis of at least 2 columns")
    return size
