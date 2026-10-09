"""The posterior and the significant support of one image batch of pass 1, by contract: float32 route and generic route.

The float32 route is RELION's coarse posterior on the device in one program (:func:`float32_support`); the generic route
(``RECOVAR_K1_RELION_F32_COARSE_SUPPORT=0``) forms the posterior from the log-sum-exp and selects the support with the
shared adaptive-fraction search (:func:`generic_support`). The controller decides once which one the pass runs.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from relax.scoring.coarse_publication import coarse_support_posterior
from relax.scoring.pass1_scores import BatchScores


def _coarse_max_posterior_for_host(batch_weights, actual_batch_size):
    """Publish active row maxima without specializing on fringe batch sizes.

    Rows are independent. Reducing the physical table before slicing the
    compact host vector keeps padded rows out of the published statistics.
    """
    return np.asarray(jnp.max(batch_weights, axis=1), dtype=np.float32)[
        :actual_batch_size
    ]


@dataclass(frozen=True)
class SupportPlan:
    """What the support stage forms every batch's posterior from, fixed for the pass.

    The pass has ``n_classes`` classes of ``n_rot`` rotations and ``n_trans`` translations; the support keeps the poses
    that carry ``adaptive_fraction`` of the posterior (at most ``max_significants``). ``exact_weight_order`` (float32
    route, one class) forms RELION's log weights ``pdf_orientation + pdf_offset + min_diff2 - diff2`` from the pre-prior
    scores, the ``exact_rotation_prior`` (``[R]`` float32: the rotation prior plus the class prior) and the batch's
    translation prior. ``tie_score_ulps`` widens the float32 route's tie rule. ``return_relion_f32_normalization``
    asks the generic route for RELION's float32 normalization of the unshifted scores.
    """

    n_classes: int
    n_rot: int
    n_trans: int
    adaptive_fraction: float
    max_significants: Any
    tie_score_ulps: int
    return_relion_f32_normalization: bool
    exact_weight_order: bool
    exact_rotation_prior: Any


class SupportResult(NamedTuple):
    """One batch's posterior and support; a field a route does not produce is ``None``.

    ``weights`` is ``[B, K * R * T]`` class-major. ``mask`` is the significant poses (same shape), ``rotation_mask`` the
    ``[B, K * R]`` rotations with a significant translation, ``n_significant`` and ``cutoff_count`` the per-row counts
    with and without ties. The float32 route also gives ``sum_weight``, ``significant_weight``, ``pmax`` (the row
    maxima) and, with ``exact_weight_order``, ``winner`` (the row argmax, which RELION publishes as the coarse winner).
    The generic route gives, when asked, ``normalization_sum_weight`` and ``normalization_max_posterior``: RELION's
    float32 denominator and row maximum of the unshifted scores, host arrays of the ``actual_batch_size`` real rows.
    """

    weights: Any
    mask: Any
    rotation_mask: Any
    n_significant: Any
    cutoff_count: Any
    sum_weight: Any
    significant_weight: Any
    pmax: Any
    winner: Any
    normalization_sum_weight: Any
    normalization_max_posterior: Any


NO_SUPPORT = SupportResult(*([None] * len(SupportResult._fields)))


def exact_order_rotation_prior(class_log_priors, rotation_log_prior_padded, n_rot: int):
    """The ``[R]`` float32 rotation prior plus the class prior of class 0, as the exact weight order adds them."""

    return (
        jnp.zeros(n_rot, dtype=jnp.float32)
        if rotation_log_prior_padded is None
        else jnp.asarray(rotation_log_prior_padded[0, :n_rot], dtype=jnp.float32)
    ) + jnp.asarray(class_log_priors[0], dtype=jnp.float32)


def float32_support(plan: SupportPlan, scores: BatchScores, translation_log_prior) -> SupportResult:
    """RELION's float32 coarse posterior of ``scores`` and what the batch publishes from it, as one program.

    ``translation_log_prior`` is the batch's prior (``None``, ``[T]`` or ``[B, T]``); the exact weight order needs
    one and takes zeros when there is none.
    """

    support = coarse_support_posterior(
        scores.support_values,
        scores.raw_score_max,
        plan.exact_rotation_prior,
        (
            None
            if not plan.exact_weight_order
            else np.zeros(plan.n_trans, dtype=np.float32)
            if translation_log_prior is None
            else translation_log_prior
        ),
        exact_weight_order=plan.exact_weight_order,
        n_trans=plan.n_trans,
        adaptive_fraction=float(plan.adaptive_fraction),
        max_significants=plan.max_significants,
        tie_score_ulps=plan.tie_score_ulps,
    )
    return SupportResult(
        weights=support["weights"],
        mask=support["mask"],
        rotation_mask=support["rotation_mask"],
        n_significant=support["n_significant"],
        cutoff_count=support["cutoff_count"],
        sum_weight=support["sum_weight"],
        significant_weight=support["significant_weight"],
        pmax=support["pmax"],
        winner=support["winner"] if plan.exact_weight_order else None,
        normalization_sum_weight=None,
        normalization_max_posterior=None,
    )


def generic_support(plan: SupportPlan, scores: BatchScores, global_log_z, actual_batch_size: int) -> SupportResult:
    """The posterior of ``scores`` as ``exp(score - log Z)`` and its support by the shared adaptive-fraction search.

    ``global_log_z`` is the batch's log-sum-exp over every pose. ``scores.support_values`` are the with-prior scores.
    """

    from relax.helpers.oversampling import find_significant_rotations as _find_sig

    weights = jnp.exp(scores.support_values - global_log_z[:, None])
    normalization_sum_weight = normalization_max_posterior = None
    if plan.return_relion_f32_normalization:
        from relax.sparse_pass2.sparse_pass2_posterior import relion_f32_fine_probabilities

        # Retain the existing coarse selector and all of its outputs.
        # The symbolic fine pass needs the numeric maximum-shifted
        # denominator, not exp(logZ) or a second normalized support.
        # See docs/math/zero_oversampling.md.
        normalization_probs, _, _, _, normalization_sum, _ = relion_f32_fine_probabilities(
            scores.support_values,
            adaptive_fraction=plan.adaptive_fraction,
            keep_all=True,
        )
        normalization_sum_weight = np.asarray(normalization_sum, dtype=np.float32)[:actual_batch_size]
        normalization_max_posterior = _coarse_max_posterior_for_host(normalization_probs, actual_batch_size)
    mask, rotation_mask, n_significant, cutoff_count = _find_sig(
        weights,
        plan.n_classes * plan.n_rot,
        plan.n_trans,
        adaptive_fraction=plan.adaptive_fraction,
        max_significants=plan.max_significants,
        return_cutoff_count=True,
    )
    return SupportResult(
        weights=weights,
        mask=mask,
        rotation_mask=rotation_mask,
        n_significant=n_significant,
        cutoff_count=cutoff_count,
        sum_weight=None,
        significant_weight=None,
        pmax=None,
        winner=None,
        normalization_sum_weight=normalization_sum_weight,
        normalization_max_posterior=normalization_max_posterior,
    )
