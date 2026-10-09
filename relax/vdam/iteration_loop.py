"""VDAM iteration loop `run_vdam_iterations`.

Mirrors `MlOptimiser::iterate` (ml_optimiser.cpp:3458-3550) for the
gradient-refine branch:

  for iter in 1 .. nr_iter:
      do_grad = ...                       # drop grad at the EM tail
      apply_schedules(state, iter)  # stepsize, tau2_fudge, subset size
      select_subset_for_iter(state, ...)  # shuffle + prefix + stable-sort; pseudo_halfsets = do_grad
      update_image_size_and_resolution_pointers(state)
      projector_refresh_fn(state)         # projector and tau2 for the E-step
      expectation_step(state, ...)        # accumulators and the E-step's meta
      update.maximize(state, ...)         # the M-step (VDAM or momentum SGD)
      update_probabilities_from_estep, update.update_noise
      post_mstep_update(state, ...)       # solvent flattening, the input source's references
      update.update_resolution(state)
      record_iteration, iter_artifact_sink

The E-step (`expectation_step`) is a callback supplied by the caller, which owns the particle data and
the engine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Callable, Literal, Sequence

import numpy as np

from relax.helpers.convergence import _relion_optimizer_average_pmax
from relax.reconstruction import regularization_relion
from relax.reconstruction.regularization_relion import resolution_from_data_vs_prior
from relax.relion.macros import relion_round
from relax.vdam.estep_common import EstepSums, estep_sums
from relax.vdam.estep_meta_updates import (
    NonFiniteNoiseSums,
    update_noise_from_estep,
    update_probabilities_from_estep,
    with_uniform_class_direction_priors,
)
from relax.vdam.m_step import vdam_m_step, vdam_m_step_single_class
from relax.vdam.native_options import VdamEnvironment
from relax.vdam.ports import VdamObserver
from relax.vdam.schedules import (
    VdamPhaseLengths,
    compute_stepsize,
    compute_subset_size,
    compute_tau2_fudge,
)
from relax.vdam.state import InitialModelState
from relax.vdam.subset_schedule import resolve_phase_lengths, select_subset_for_iter

# Callback signatures
ExpectationStepFn = Callable[
    # (state, particle_ids, halfset_ids) -> (posterior_accumulators, posterior_meta)
    [InitialModelState, np.ndarray, np.ndarray],
    tuple,  # (List[VdamAccumulator], dict)
]
"""E-step callback. Must return `(accumulators, meta)` where accumulators
holds 2K entries when pseudo_halfsets is on (halfset-0 first, then
halfset-1) and meta holds the E-step's sums, which the M-step and the probability and noise updates read
(`estep_common.estep_sums`), and the iteration's report.
"""

IterArtifactSink = Callable[[InitialModelState, int, dict], None]
PostMstepUpdateFn = Callable[[InitialModelState, int, dict], InitialModelState]


def apply_schedules(
    state: InitialModelState,
    iter: int,
    phase_lengths: VdamPhaseLengths,
    *,
    grad_ini_subset_size: int,
    grad_fin_subset_size: int,
    nr_particles: int,
    tau2_fudge_arg: float,
    grad_em_iters: int,
    grad_stepsize: float | None = None,
) -> InitialModelState:
    """Apply the three VDAM schedules to `state.iter=iter`."""
    subset_size = compute_subset_size(
        iter=iter,
        phase_lengths=phase_lengths,
        grad_ini_subset_size=grad_ini_subset_size,
        grad_fin_subset_size=grad_fin_subset_size,
        nr_particles=nr_particles,
        nr_iter=state.nr_iter,
        grad_em_iters=grad_em_iters,
        has_converged=state.has_converged,
        grad_has_converged=state.grad_has_converged,
        nr_classes=state.K,
    )
    stepsize = compute_stepsize(
        iter=iter,
        phase_lengths=phase_lengths,
        is_3d_model=True,
        ref_dim=3,
        grad_stepsize=grad_stepsize,
    )
    tau2_fudge = compute_tau2_fudge(
        iter=iter,
        phase_lengths=phase_lengths,
        is_3d_model=True,
        ref_dim=3,
        tau2_fudge_arg=tau2_fudge_arg,
    )
    return replace(
        state, iter=iter, subset_size=subset_size, grad_current_stepsize=stepsize, tau2_fudge_factor=tau2_fudge
    )


def update_current_resolution_from_data_vs_prior(state: InitialModelState) -> InitialModelState:
    """Mirror RELION ``updateCurrentResolution`` for InitialModel/VDAM.

    Gradient InitialModel uses ``data_vs_prior_class`` produced by
    ``BackProjector::updateSSNRarrays``. The resulting ``current_resolution``
    is written in this iteration and converted to the next iteration's
    ``current_size`` when expectation setup calls
    ``updateImageSizeAndResolutionPointers``.
    """
    maxres = max(
        resolution_from_data_vs_prior(
            np.asarray(state.data_vs_prior_class[k], dtype=np.float64),
            box_size=state.box_size,
            minres_map=regularization_relion.RELION_MINRES_MAP,
        )
        for k in range(int(state.K))
    )

    return replace(
        state,
        current_resolution_shell=int(maxres),
        current_resolution=float(maxres) / (float(state.pixel_size) * float(state.box_size)),
    )


def update_image_size_and_resolution_pointers(state: InitialModelState, pilot_controls=None) -> InitialModelState:
    """Mirror the current-size part of RELION ``updateImageSizeAndResolutionPointers``."""
    maxres = relion_round(float(state.current_resolution) * float(state.pixel_size) * float(state.box_size))
    if float(state.ave_Pmax) > 0.1 and bool(state.has_high_fsc_at_limit):
        maxres += relion_round(0.25 * float(state.box_size) / 2.0)
    else:
        maxres += int(state.incr_size)
    current_size = min(max(2 * maxres, 2), int(state.box_size))
    current_size = current_size if pilot_controls is None else pilot_controls.cap_current_size(current_size)

    return replace(state, current_size=int(current_size))


def _ave_pmax(sums: EstepSums) -> float | None:
    """RELION's average Pmax, or None without per-image Pmax (the state keeps its value)."""
    if sums.pmax is None or not sums.pmax.size:
        return None
    # RELION accumulates Pmax over all VDAM pseudo-halfsets, then divides by the retained M-step
    # posterior mass (rather than the particle count). Treat the two pseudo-halfsets as one optimiser
    # population when reusing the Class3D normalization helper.
    _, average, _ = _relion_optimizer_average_pmax([sums.pmax], [float(np.sum(sums.class_mass))])
    return average


@dataclass(frozen=True)
class VdamUpdate:
    """RELION's VDAM model update: the gradient M-step, the noise blend and the data_vs_prior resolution.

    ``single_class_m_step``: each class's M-step, the run's input source's (``VdamInputSource``); ``observer``
    the run's observer, which sees each noise update.
    """

    padding_factor: int
    mstep_compute_dtype: Literal["float32", "float64"]
    observer: VdamObserver
    single_class_m_step: Callable[..., InitialModelState] = vdam_m_step_single_class

    def maximize(self, current: InitialModelState, accumulators, sums: EstepSums, meta: dict) -> InitialModelState:
        return vdam_m_step(
            current,
            accumulators=accumulators,
            grad_current_stepsize=current.grad_current_stepsize,
            tau2_fudge_factor=current.tau2_fudge_factor,
            padding_factor=self.padding_factor,
            mstep_compute_dtype=self.mstep_compute_dtype,
            average_ctf2=sums.average_ctf2,
            single_class_m_step=self.single_class_m_step,
        )

    def update_noise(self, current, sums: EstepSums, meta: dict, *, do_grad: bool, mu: float) -> InitialModelState:
        try:
            updated = update_noise_from_estep(current, sums, do_grad=do_grad, mu=mu)
        except NonFiniteNoiseSums as error:
            dump_path = self.observer.noise_sums_nonfinite(current, meta, error.summaries)
            raise ValueError(str(error) if dump_path is None else f"{error}; dump={dump_path}") from None
        if updated is not current:  # the update returns its input when there is nothing to update
            self.observer.noise_updated(current, updated, sums)
        return updated

    def update_resolution(self, current: InitialModelState) -> InitialModelState:
        return update_current_resolution_from_data_vs_prior(current)


@dataclass(frozen=True)
class MomentumSgdUpdate:
    """The opt-in momentum-SGD update; its resolution is the caller's Fourier radius schedule.

    relax.vdam.sgd reads its noise sums from the E-step's ``meta`` and adds its report there.
    """

    learning_rate: float
    padding_factor: int

    def maximize(self, current: InitialModelState, accumulators, sums: EstepSums, meta: dict) -> InitialModelState:
        from relax.vdam.sgd import sgd_m_step

        return sgd_m_step(current, accumulators, learning_rate=self.learning_rate, padding_factor=self.padding_factor, meta=meta)

    def update_noise(self, current, sums: EstepSums, meta: dict, *, do_grad: bool, mu: float) -> InitialModelState:
        from relax.vdam.sgd import update_sgd_noise

        del sums, do_grad, mu  # its own noise estimate, without VDAM's momentum
        return update_sgd_noise(current, meta)

    def update_resolution(self, current: InitialModelState) -> InitialModelState:
        return current


def run_vdam_iterations(
    state: InitialModelState,
    *,
    nr_particles: int,
    optics_group_by_particle: Sequence[int],
    grad_ini_subset_size: int,
    grad_fin_subset_size: int,
    pilot_controls=None,
    tau2_fudge_arg: float,
    grad_em_iters: int,
    random_seed: int,
    expectation_step: ExpectationStepFn,
    update: VdamUpdate | MomentumSgdUpdate,
    iter_artifact_sink: IterArtifactSink = lambda *args, **kw: None,
    record_iteration: IterArtifactSink | None = None,
    post_mstep_update: PostMstepUpdateFn | None = None,
    particle_order: Sequence[int] | None = None,
    grad_ini_frac: float,
    grad_fin_frac: float,
    phase_lengths: VdamPhaseLengths | None = None,
    grad_stepsize: float | None = None,
    mu: float,
    projector_refresh_fn: Callable[..., InitialModelState],
    projector_padding_factor: int = 1,
    start_iteration: int = 0,
    diagnostic_stop_after_iteration: int | None = None,
    fourier_radius_schedule: tuple[int, ...] | None = None,
    stochastic_all_iterations: bool = False,
    uniform_class_direction_prior: bool,
    environment: VdamEnvironment,
    observer: VdamObserver,
) -> InitialModelState:
    """Full VDAM loop; ``state`` must come from ``initialise_denovo_state`` + ``with_sigma2_noise``.

    ``update`` is the optimizer's model update (:class:`VdamUpdate` or :class:`MomentumSgdUpdate`), chosen
    once by the caller; ``observer`` the run's observer, which times each iteration's stages. ``projector_refresh_fn(state, padding_factor=...)`` runs before every E-step:
    RELION's ``MlModel::setFourierTransformMaps(!fix_tau)``, the projector and tau2 from its power
    spectrum (:meth:`relax.vdam.estep_setup.IterationProjectorContext.refresh`). ``record_iteration`` updates the caller's own run state from the completed
    iteration (the sampling controller's counters) before ``iter_artifact_sink`` writes its outputs.
    """
    phase_lengths = resolve_phase_lengths(
        int(state.nr_iter),
        float(grad_ini_frac),
        float(grad_fin_frac),
        phase_lengths,
    )
    if start_iteration < 0 or start_iteration >= int(state.nr_iter):
        raise ValueError("start_iteration must be between 0 and state.nr_iter - 1")
    if int(state.iter) != start_iteration:
        raise ValueError(
            f"state.iter must equal start_iteration ({int(state.iter)} != {start_iteration})"
        )
    final_iteration = int(state.nr_iter)
    if diagnostic_stop_after_iteration is not None:
        final_iteration = int(diagnostic_stop_after_iteration)
        if final_iteration <= start_iteration or final_iteration > int(state.nr_iter):
            raise ValueError(
                "diagnostic_stop_after_iteration must be greater than start_iteration "
                "and no greater than state.nr_iter"
            )
    current = with_uniform_class_direction_priors(state) if uniform_class_direction_prior else state

    for it in range(start_iteration + 1, final_iteration + 1):
        profile = observer.iteration_profile(it)
        do_grad = bool(stochastic_all_iterations) or (
            ((state.nr_iter - it) >= grad_em_iters) and not current.has_converged
        )

        current = apply_schedules(
            current,
            iter=it,
            phase_lengths=phase_lengths,
            grad_ini_subset_size=grad_ini_subset_size,
            grad_fin_subset_size=grad_fin_subset_size,
            nr_particles=nr_particles,
            tau2_fudge_arg=tau2_fudge_arg,
            grad_em_iters=grad_em_iters,
            grad_stepsize=grad_stepsize,
        )
        if stochastic_all_iterations:
            batch_size = min(int(pilot_controls.stochastic_batch_size), int(nr_particles))
            if batch_size >= int(nr_particles):
                raise ValueError("stochastic_all_iterations batch must be smaller than the particle count")
            current = replace(current, subset_size=batch_size)
        profile.stage("schedule")

        current = select_subset_for_iter(
            current,
            iter=it,
            nr_particles=nr_particles,
            optics_group_by_particle=optics_group_by_particle,
            random_seed=random_seed,
            do_grad=do_grad,
            particle_order=particle_order,
        )
        profile.stage("subset")

        current = update_image_size_and_resolution_pointers(current, pilot_controls)
        if fourier_radius_schedule is not None:
            radius = int(fourier_radius_schedule[it - 1])
            if radius < 1 or radius > int(state.box_size) // 2:
                raise ValueError("scheduled Fourier radius exceeds the model box")
            current = replace(
                current,
                current_size=2 * radius,
                current_resolution_shell=radius,
                current_resolution=float(radius) / (float(state.pixel_size) * float(state.box_size)),
            )
        current = projector_refresh_fn(current, padding_factor=projector_padding_factor)
        profile.stage("projector_refresh")

        # E-step: caller-supplied closure over the data loader + dense kernels
        accumulators, meta = expectation_step(
            current,
            current.subset_particle_ids,
            current.subset_halfset_ids,
        )
        if fourier_radius_schedule is not None or stochastic_all_iterations:
            subset_ids = np.ascontiguousarray(current.subset_particle_ids, dtype=np.int64)
            subset_halfsets = np.ascontiguousarray(current.subset_halfset_ids, dtype=np.int8)
            meta["subset_particle_ids_sha256"] = hashlib.sha256(subset_ids.tobytes()).hexdigest()
            meta["subset_halfset_ids_sha256"] = hashlib.sha256(subset_halfsets.tobytes()).hexdigest()
            meta["effective_estep_fourier_radius"] = int(current.current_size) // 2
        profile.stage("expectation")

        sums = estep_sums(meta)
        current = update.maximize(current, accumulators, sums, meta)
        profile.stage("mstep")
        current = update_probabilities_from_estep(
            current,
            sums,
            do_grad=do_grad,
            mu=mu,
            uniform_class_direction_prior=uniform_class_direction_prior,
        )
        full_class_sums = meta.get("class_posterior_sums_full")
        if full_class_sums is not None:
            full_class_sums = np.asarray(full_class_sums, dtype=np.float64)
            class_mass = float(np.sum(full_class_sums))
            if full_class_sums.shape == (current.K,) and np.isfinite(class_mass) and class_mass > 0.0:
                meta["class_posterior_fraction_by_class"] = (full_class_sums / class_mass).tolist()
                meta["class_posterior_fraction_source"] = "full"
        retained_class_sums = meta.get("class_posterior_sums")
        if retained_class_sums is not None:
            retained_class_sums = np.asarray(retained_class_sums, dtype=np.float64)
            retained_mass = float(np.sum(retained_class_sums))
            if retained_class_sums.shape == (current.K,) and np.isfinite(retained_mass) and retained_mass > 0.0:
                meta["class_retained_mass_fraction_by_class"] = (retained_class_sums / retained_mass).tolist()
        if uniform_class_direction_prior:
            n_directions = int(np.asarray(current.pdf_direction).shape[1])
            meta["uniform_class_direction_prior"] = True
            meta["effective_pdf_class_prior_by_class"] = np.asarray(current.pdf_class, dtype=np.float64).tolist()
            meta["effective_joint_direction_prior_per_class_direction"] = 1.0 / float(current.K * n_directions)
            meta["effective_joint_direction_count"] = n_directions
        current = update.update_noise(current, sums, meta, do_grad=do_grad, mu=mu)
        ave_pmax = _ave_pmax(sums)
        if ave_pmax is not None:
            current = replace(current, ave_Pmax=float(ave_pmax))
        if post_mstep_update is not None and (stochastic_all_iterations or not current.has_converged):
            current = post_mstep_update(current, it, meta)

        current = update.update_resolution(current)
        profile.stage("state_update")
        meta.update(
            {
                "current_size": int(current.current_size),
                "current_resolution": float(current.current_resolution),
                "current_resolution_shell": int(current.current_resolution_shell),
                "ave_Pmax": float(current.ave_Pmax),
                "subset_size": int(current.subset_size),
            }
        )
        profile.before_artifacts(meta)
        if record_iteration is not None:
            record_iteration(current, it, meta)
        iter_artifact_sink(current, it, meta)
        if pilot_controls is not None and pilot_controls.stop_requested():
            break
        profile.finish()

        # Release scratch buffers to avoid CUFFT_ALLOC_FAILED at 50k×256² (forces next-iter recompile).
        if environment.clear_jax_caches_per_iteration:
            import gc

            import jax

            jax.clear_caches()
            gc.collect()

    return current
