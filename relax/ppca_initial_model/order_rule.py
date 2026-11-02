"""The accuracy-driven HEALPix order of the PPCA InitialModel: RELION's ``updateAngularSampling`` rule.

RELION refines the angular sampling while the angular step is at least 75% of the expected accuracy ``acc_rot``
(ml_optimiser.cpp ``updateAngularSampling``). Here ``acc_rot`` is the divergence-based estimate of
:mod:`relax.ppca_initial_model.pose_accuracy`. docs/math/vdam_ppca_algorithm.md sections 4.1 and 18.10.
"""

from relax.sampling import grids

# RELION: the sampling is fine enough once its step is below this share of the expected accuracy.
ACCURACY_FRACTION = 0.75


def angular_step_deg(order: int) -> float:
    """The angular step of the exhaustive HEALPix grid of ``order`` in degrees (RELION's, without oversampling)."""
    return float(grids.relion_angular_sampling_deg(order, 0))


def too_coarse(order: int, acc_rot: float) -> bool:
    """Whether the step of ``order`` is at least ``ACCURACY_FRACTION`` of the expected accuracy ``acc_rot``."""
    return angular_step_deg(order) >= ACCURACY_FRACTION * acc_rot


def next_order(order: int, acc_rot: float, *, order_max: int) -> int:
    """The order of the next iteration: one finer while the step is too coarse, up to ``order_max``; never coarser."""
    return order + 1 if order < order_max and too_coarse(order, acc_rot) else order


def final_order(order: int, acc_rot: float, *, final_order_max: int) -> int:
    """The order of the final embedding pass: the first order from ``order`` on that is fine enough, at most the cap."""
    while order < final_order_max and too_coarse(order, acc_rot):
        order += 1
    return order


def order_for(config, state, iteration: int) -> int:
    """The HEALPix order of ``iteration``'s E-step.

    The stages table gives it under the ``stages`` rule. Under ``accuracy`` the table's first order starts the run
    and the state carries the order planned after its last update; the final embeddings, which run at the iteration
    of the state itself, take the order planned after the last iteration.
    """
    if config.order_rule == "stages":
        return config.stage(iteration)[1]
    return state.next_order if state.next_order is not None else config.stage(1)[1]
