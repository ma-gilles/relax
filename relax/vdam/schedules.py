"""VDAM InitialModel's per-iteration switches: whether gradients run, the maximum significants, and the
cadences of the angular-sampling update and the expected-accuracy estimate.

The gradient schedule formulas (phase lengths, subset size, step size, tau2_fudge) are
``relax.steps.sgd_schedule``.
"""

from __future__ import annotations

from relax.vdam.state import InitialModelState

RELION_INITIALMODEL_3D_GRADIENT_MAX_SIGNIFICANTS_PER_CLASS = 100


def native_initialmodel_do_grad(
    state: InitialModelState,
    iteration: int,
    *,
    grad_em_iters: int,
) -> bool:
    return ((int(state.nr_iter) - int(iteration)) >= int(grad_em_iters)) and not bool(state.has_converged)


def active_relion_initialmodel_max_significants(state: InitialModelState, *, do_grad: bool) -> int:
    """Runtime maximum_significants used by RELION gradient InitialModel."""

    if not bool(do_grad):
        return -1
    return int(RELION_INITIALMODEL_3D_GRADIENT_MAX_SIGNIFICANTS_PER_CLASS) * int(state.K)


def relion_sampling_cadence(*, iteration: int, do_grad: bool) -> bool:
    """Every iteration without gradients, every tenth with them: the cadence of RELION's
    ``updateAngularSampling`` call (ml_optimiser.cpp:3555-3558)."""
    return not bool(do_grad) or int(iteration) % 10 == 0


def should_estimate_native_sampling_accuracy(*, iteration: int, nr_iter: int, do_grad: bool) -> bool:
    """RELION's ``calculateExpectedAngularErrors`` cadence (ml_optimiser.cpp:3541-3542): the sampling cadence,
    always in iteration 1, and in the last iteration (``iter == nr_iter``) also under gradients."""
    return (
        int(iteration) <= 1
        or int(iteration) == int(nr_iter)
        or relion_sampling_cadence(iteration=iteration, do_grad=do_grad)
    )
