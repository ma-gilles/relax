"""VDAM iteration loop `run_vdam_iterations`.

Mirrors `MlOptimiser::iterate` (ml_optimiser.cpp:3458-3550) for the
gradient-refine branch:

  for iter in 1 .. nr_iter:
      schedule_update(state, iter)        # stepsize, tau2_fudge, subset
      do_grad = ...                       # drop grad at the EM tail
      pseudo_halfsets = do_grad
      select_subset_for_iter(state, ...)  # shuffle + prefix + stable-sort
      update_current_resolution(state)    # FSC-driven from iter 2
      expectation_step(state, ...)        # E-step adapter -> posteriors
      maximisation_step(state, ...)       # VDAM M-step
      post_mstep_update(state, ...)       # masks / other post-M-step hooks
      write_iter_artifacts(state, iter)

The E-step adapter (`expectation_step`) is a callback supplied by the
caller because it requires dense-path kernels + real particle data. This
module is the pure orchestrator.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from typing import Callable, Literal, Sequence

import numpy as np

from relax.helpers.convergence import _relion_optimizer_average_pmax
from relax.reconstruction.regularization_relion import resolution_from_data_vs_prior
from relax.relion.macros import relion_round
from relax.vdam.estep_common import EstepSums, estep_sums
from relax.vdam.estep_meta_updates import (
    update_noise_from_estep,
    update_probabilities_from_estep,
    with_uniform_class_direction_priors,
)
from relax.vdam.m_step import vdam_m_step
from relax.vdam.native_options import VdamEnvironment
from relax.vdam.schedules import (
    DEFAULT_GRAD_EM_ITERS,
    DEFAULT_GRAD_MU,
    VdamPhaseLengths,
    compute_stepsize,
    compute_subset_size,
    compute_tau2_fudge,
)
from relax.vdam.state import InitialModelState
from relax.vdam.subset_schedule import _resolve_phase_lengths, select_subset_for_iter

# Callback signatures
ExpectationStepFn = Callable[
    # (state, particle_ids, halfset_ids) -> (posterior_accumulators, posterior_meta)
    [InitialModelState, np.ndarray, np.ndarray],
    tuple,  # (List[VdamAccumulator], dict)
]
"""E-step callback. Must return `(accumulators, meta)` where accumulators
holds 2K entries when pseudo_halfsets is on (halfset-0 first, then
halfset-1) and meta is a free-form dict written into per-iter STAR output
(Pmax, nr_significant, best_class, best_euler, best_trans).
"""

IterArtifactSink = Callable[[InitialModelState, int, dict], None]
PostMstepUpdateFn = Callable[[InitialModelState, int, dict], InitialModelState]


def refresh_tau2_from_projector_power(
    state: InitialModelState,
    *,
    padding_factor: int = 1,
    interpolator: int = 1,
) -> InitialModelState:
    """``MlModel::setFourierTransformMaps(!fix_tau)``: tau2 from the projector setup's power spectrum.

    The same device transform the default projector context runs
    (:func:`relax.vdam.dense_adapter.prepare_relion_projector_class_inputs_and_power`),
    without keeping the scoring operands.
    """
    from relax.relion import relion_projector_setup

    _, power, _ = relion_projector_setup.reference_to_relion_projector_half_maps_and_power(
        state.Iref,
        current_size=int(state.current_size if state.current_size > 0 else state.ori_size),
        padding_factor=int(padding_factor),
        interpolator=int(interpolator),
    )
    new_tau2 = np.asarray(power, dtype=np.float64)
    return replace(state, tau2_class=new_tau2)


def default_schedule_update(
    state: InitialModelState,
    iter: int,
    phase_lengths: VdamPhaseLengths,
    *,
    grad_ini_subset_size: int,
    grad_fin_subset_size: int,
    nr_particles: int,
    tau2_fudge_arg: float,
    grad_em_iters: int = DEFAULT_GRAD_EM_ITERS,
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


def update_current_resolution_from_data_vs_prior(
    state: InitialModelState,
    *,
    minres_map: int = 5,
) -> InitialModelState:
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
            box_size=state.ori_size,
            minres_map=minres_map,
        )
        for k in range(int(state.K))
    )

    return replace(
        state,
        current_resolution_shell=int(maxres),
        current_resolution=float(maxres) / (float(state.pixel_size) * float(state.ori_size)),
    )


def update_image_size_and_resolution_pointers(state: InitialModelState, pilot_controls=None) -> InitialModelState:
    """Mirror the current-size part of RELION ``updateImageSizeAndResolutionPointers``."""
    maxres = relion_round(float(state.current_resolution) * float(state.pixel_size) * float(state.ori_size))
    if float(state.ave_Pmax) > 0.1 and bool(state.has_high_fsc_at_limit):
        maxres += relion_round(0.25 * float(state.ori_size) / 2.0)
    else:
        maxres += int(state.incr_size)
    current_size = min(max(2 * maxres, 2), int(state.ori_size))
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
    """RELION's VDAM model update: the gradient M-step, the noise blend and the data_vs_prior resolution."""

    padding_factor: int
    mstep_compute_dtype: Literal["float32", "float64"]

    def maximize(self, current: InitialModelState, accumulators, sums: EstepSums, meta: dict) -> InitialModelState:
        return vdam_m_step(
            current,
            accumulators=accumulators,
            grad_current_stepsize=current.grad_current_stepsize,
            tau2_fudge_factor=current.tau2_fudge_factor,
            padding_factor=self.padding_factor,
            mstep_compute_dtype=self.mstep_compute_dtype,
            average_ctf2=sums.average_ctf2,
        )

    def update_noise(self, current, sums: EstepSums, meta: dict, *, do_grad: bool, mu: float) -> InitialModelState:
        return update_noise_from_estep(current, sums, do_grad=do_grad, mu=mu, report=meta)

    def update_resolution(self, current: InitialModelState) -> InitialModelState:
        return update_current_resolution_from_data_vs_prior(current)


@dataclass(frozen=True)
class MomentumSgdUpdate:
    """The opt-in momentum-SGD update; its resolution is the caller's Fourier radius schedule.

    relax.sgd_initial_model reads its noise sums from the E-step's ``meta`` and adds its report there.
    """

    learning_rate: float
    padding_factor: int

    def maximize(self, current: InitialModelState, accumulators, sums: EstepSums, meta: dict) -> InitialModelState:
        from relax.sgd_initial_model.optimizer import sgd_m_step

        return sgd_m_step(current, accumulators, learning_rate=self.learning_rate, padding_factor=self.padding_factor, meta=meta)

    def update_noise(self, current, sums: EstepSums, meta: dict, *, do_grad: bool, mu: float) -> InitialModelState:
        from relax.sgd_initial_model.noise import update_sgd_noise

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
    grad_ini_frac: float = 0.3,
    grad_fin_frac: float = 0.2,
    phase_lengths: VdamPhaseLengths | None = None,
    grad_stepsize: float | None = None,
    mu: float = DEFAULT_GRAD_MU,
    refresh_tau2_from_projector: bool = True,
    projector_refresh_fn: Callable[..., InitialModelState] | None = None,
    projector_padding_factor: int = 1,
    projector_interpolator: int = 1,
    start_iteration: int = 0,
    diagnostic_stop_after_iteration: int | None = None,
    fourier_radius_schedule: tuple[int, ...] | None = None,
    stochastic_all_iterations: bool = False,
    uniform_class_direction_prior: bool = False,
    environment: VdamEnvironment = VdamEnvironment(),
) -> InitialModelState:
    """Full VDAM loop; ``state`` must come from ``initialise_denovo_state`` + ``seed_noise_from_mavg``.

    ``update`` is the optimizer's model update (:class:`VdamUpdate` or :class:`MomentumSgdUpdate`), chosen
    once by the caller. ``record_iteration`` updates the caller's own run state from the completed
    iteration (the sampling controller's counters) before ``iter_artifact_sink`` writes its outputs.
    """
    phase_lengths = _resolve_phase_lengths(
        int(state.nr_iter),
        float(grad_ini_frac),
        float(grad_fin_frac),
        phase_lengths,
    )
    start_iteration = int(start_iteration)
    if start_iteration < 0 or start_iteration >= int(state.nr_iter):
        raise ValueError("start_iteration must be between 0 and state.nr_iter - 1")
    if int(state.iter) != start_iteration:
        raise ValueError(
            f"state.iter must equal start_iteration ({int(state.iter)} != {start_iteration})"
        )
    if fourier_radius_schedule is not None and len(fourier_radius_schedule) != int(state.nr_iter):
        raise ValueError("fourier_radius_schedule must have one radius per iteration")
    if stochastic_all_iterations and (
        grad_em_iters != 0 or pilot_controls is None or pilot_controls.stochastic_batch_size is None
    ):
        raise ValueError("stochastic_all_iterations requires a fixed batch and no EM tail")
    final_iteration = int(state.nr_iter)
    if diagnostic_stop_after_iteration is not None:
        final_iteration = int(diagnostic_stop_after_iteration)
        if final_iteration <= start_iteration or final_iteration > int(state.nr_iter):
            raise ValueError(
                "diagnostic_stop_after_iteration must be greater than start_iteration "
                "and no greater than state.nr_iter"
            )
    current = with_uniform_class_direction_priors(state) if uniform_class_direction_prior else state
    profile_iterations = environment.profile

    for it in range(start_iteration + 1, final_iteration + 1):
        iteration_started = time.perf_counter()
        stage_started = iteration_started
        iteration_profile: dict[str, float] = {}

        def _record_stage(name: str) -> None:
            nonlocal stage_started
            now = time.perf_counter()
            iteration_profile[f"{name}_time_s"] = float(now - stage_started)
            stage_started = now

        do_grad = bool(stochastic_all_iterations) or (
            ((state.nr_iter - it) >= grad_em_iters) and not current.has_converged
        )

        current = default_schedule_update(
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
        if profile_iterations:
            _record_stage("schedule")

        current = select_subset_for_iter(
            current,
            iter=it,
            nr_particles=nr_particles,
            optics_group_by_particle=optics_group_by_particle,
            random_seed=random_seed,
            do_grad=do_grad,
            particle_order=particle_order,
        )
        if fourier_radius_schedule is not None or stochastic_all_iterations:
            subset_ids = np.ascontiguousarray(current.subset_particle_ids, dtype=np.int64)
            subset_halfsets = np.ascontiguousarray(current.subset_halfset_ids, dtype=np.int8)
        if profile_iterations:
            _record_stage("subset")

        current = update_image_size_and_resolution_pointers(current, pilot_controls)
        if fourier_radius_schedule is not None:
            radius = int(fourier_radius_schedule[it - 1])
            if radius < 1 or radius > int(state.ori_size) // 2:
                raise ValueError("scheduled Fourier radius exceeds the model box")
            current = replace(
                current,
                current_size=2 * radius,
                current_resolution_shell=radius,
                current_resolution=float(radius) / (float(state.pixel_size) * float(state.ori_size)),
            )
        if refresh_tau2_from_projector:
            refresh = projector_refresh_fn or refresh_tau2_from_projector_power
            current = refresh(
                current,
                padding_factor=projector_padding_factor,
                interpolator=projector_interpolator,
            )
        if profile_iterations:
            _record_stage("projector_refresh")

        # E-step: caller-supplied closure over the data loader + dense kernels
        accumulators, meta = expectation_step(
            current,
            current.subset_particle_ids,
            current.subset_halfset_ids,
        )
        if fourier_radius_schedule is not None or stochastic_all_iterations:
            meta["subset_particle_ids_sha256"] = hashlib.sha256(subset_ids.tobytes()).hexdigest()
            meta["subset_halfset_ids_sha256"] = hashlib.sha256(subset_halfsets.tobytes()).hexdigest()
            meta["effective_estep_fourier_radius"] = int(current.current_size) // 2
        if profile_iterations:
            _record_stage("expectation")
            # The local route reports its coarse (pass 1) and fine (pass 2) walls.
            passes = meta.get("sparse_pass2_profile_summary") or {}
            for key in ("pass1_time_s", "pass2_time_s"):
                if key in passes:
                    iteration_profile[f"expectation_{key}"] = float(passes[key])

        sums = estep_sums(meta)
        current = update.maximize(current, accumulators, sums, meta)
        if profile_iterations:
            _record_stage("mstep")
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
        if profile_iterations:
            _record_stage("state_update")
        meta = dict(meta)
        meta.update(
            {
                "current_size": int(current.current_size),
                "current_resolution": float(current.current_resolution),
                "current_resolution_shell": int(current.current_resolution_shell),
                "ave_Pmax": float(current.ave_Pmax),
                "subset_size": int(current.subset_size),
            }
        )
        if profile_iterations:
            iteration_profile["pre_artifact_time_s"] = float(time.perf_counter() - iteration_started)
            meta["vdam_iteration_profile_summary"] = iteration_profile
            stage_started = time.perf_counter()
        if record_iteration is not None:
            record_iteration(current, it, meta)
        iter_artifact_sink(current, it, meta)
        if pilot_controls is not None and pilot_controls.stop_requested():
            break
        if profile_iterations:
            _record_stage("artifact")
            iteration_profile["total_time_s"] = float(time.perf_counter() - iteration_started)
            print(f"VDAM iteration {it} profile: {iteration_profile}", flush=True)

        # Release scratch buffers to avoid CUFFT_ALLOC_FAILED at 50k×256² (forces next-iter recompile).
        if environment.clear_jax_caches_per_iteration:
            import gc

            import jax

            jax.clear_caches()
            gc.collect()

    return current
