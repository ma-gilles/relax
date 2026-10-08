"""RELION's own VDAM M-step primitives, step by step, as an oracle.

Each step of RELION's gradient M-step (``reweightGrad``, ``getFristMoment``,
``getSecondMoment``, ``applyMomenta``, ``updateSSNRarrays``, ``reconstructGrad``)
is RELION's code through ``relax.relion_bind``. Production never takes this path;
it runs the transaction (:func:`relax.vdam.m_step.vdam_m_step_single_class`).

``python -m relax.diagnostics.vdam_native_mstep <relax initial_model arguments>``
runs an InitialModel whose M-step is this oracle, with the parity hooks
``RELAX_MSTEP_DUMP_DIR`` / ``RELAX_MSTEP_DUMP_ITER`` (intermediate dumps) and the
``RELAX_VDAM_NATIVE_*_REPLAY_BIN`` replays (:mod:`relax.diagnostics.vdam_mstep_replay`).
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np

from relax.diagnostics import vdam_mstep_replay as replay
from relax.vdam.state import InitialModelState, VdamAccumulator, half_slot_index


def _bindings():
    try:
        from relax.relion_bind import _relion_bind_core as bind
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "the native VDAM M-step oracle needs the RELION binding: pixi run python relax/relion_bind/build.py"
        ) from error
    return bind


def native_vdam_m_step_single_class(
    state: InitialModelState,
    k: int,
    accum_h0: VdamAccumulator,
    accum_h1: VdamAccumulator | None,
    *,
    grad_current_stepsize: float,
    tau2_fudge_factor: float,
    padding_factor: int,
    r_max: int,
    min_resol_shell: float,
    dump_dir: str | None = None,
    dump_prefix: str = "",
) -> InitialModelState:
    """One class's M-step through RELION's primitives; ``dump_dir`` saves each intermediate."""

    bind = _bindings()
    ori_size = state.ori_size
    # backprojector.h:335/343 EMA defaults
    mu_first, mu_second = 0.9, 0.999
    _do_dump = dump_dir is not None
    _dump_dir = dump_dir
    _dump_prefix = dump_prefix

    def _dump(name, arr):
        if not _do_dump:
            return
        from pathlib import Path as _Path

        _Path(_dump_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{_dump_dir}/{_dump_prefix}{name}.npy", np.asarray(arr))

    _dump("accum_h0_data", accum_h0.data)
    _dump("accum_h0_weight", accum_h0.weight)
    if state.pseudo_halfsets:
        _dump("accum_h1_data", accum_h1.data)
        _dump("accum_h1_weight", accum_h1.weight)
    _dump("iref_in", state.Iref[k])
    _dump("Igrad1_in_h0", state.Igrad1[half_slot_index(k, 0, state.K, state.pseudo_halfsets)])
    if state.pseudo_halfsets:
        _dump("Igrad1_in_h1", state.Igrad1[half_slot_index(k, 1, state.K, state.pseudo_halfsets)])
    _dump("Igrad2_in", state.Igrad2[k])
    _dump("fsc_halves_in", state.fsc_halves_class[k])

    # Step 2. reweightGrad per halfset
    data_h0 = np.asarray(bind.vdam_reweight_grad(accum_h0.data, accum_h0.weight, ori_size, padding_factor, 1, r_max))
    if state.pseudo_halfsets:
        data_h1 = np.asarray(
            bind.vdam_reweight_grad(accum_h1.data, accum_h1.weight, ori_size, padding_factor, 1, r_max)
        )
    else:
        data_h1 = None
    _dump("data_h0_post_reweight", data_h0)
    if data_h1 is not None:
        _dump("data_h1_post_reweight", data_h1)

    # Step 3. getFristMoment per halfset
    slot_h0 = half_slot_index(k, 0, state.K, state.pseudo_halfsets)
    new_Igrad1 = np.array(state.Igrad1, order="C")

    new_Igrad1[slot_h0] = np.asarray(
        bind.vdam_first_moment(
            data_h0,
            state.Igrad1[slot_h0],
            ori_size,
            padding_factor,
            1,
            r_max,
            **{"lambda": mu_first},
        )
    )
    if state.pseudo_halfsets:
        slot_h1 = half_slot_index(k, 1, state.K, state.pseudo_halfsets)
        new_Igrad1[slot_h1] = np.asarray(
            bind.vdam_first_moment(
                data_h1,
                state.Igrad1[slot_h1],
                ori_size,
                padding_factor,
                1,
                r_max,
                **{"lambda": mu_first},
            )
        )
    replay_h0, replay_h1 = replay._maybe_replay_native_first_moments(
        new_Igrad1[slot_h0],
        new_Igrad1[slot_h1] if state.pseudo_halfsets else None,
        iteration=int(getattr(state, "iter", 0)),
        class_idx=k,
    )
    new_Igrad1[slot_h0] = replay_h0
    if state.pseudo_halfsets:
        new_Igrad1[slot_h1] = replay_h1
    _dump("m1_h0_post", new_Igrad1[slot_h0])
    if state.pseudo_halfsets:
        _dump("m1_h1_post", new_Igrad1[slot_h1])

    # Step 4. getSecondMoment (uses both halfset accumulators)
    new_Igrad2 = np.array(state.Igrad2, order="C")
    if state.pseudo_halfsets:
        computed_Igrad2 = np.asarray(
            bind.vdam_second_moment(
                data_h0,
                data_h1,
                state.Igrad2[k],
                ori_size,
                padding_factor,
                1,
                r_max,
                **{"lambda": mu_second},
            )
        )
        new_Igrad2[k] = replay._maybe_replay_native_second_moment(
            computed_Igrad2,
            iteration=int(getattr(state, "iter", 0)),
            class_idx=k,
        )
        _dump("m2_computed_before_replay", computed_Igrad2)
        _dump("m2_post", new_Igrad2[k])

    # Step 5. applyMomenta. Non-halfset: pass m1 twice to trigger do_half=false.
    m1_h0 = new_Igrad1[slot_h0]
    m1_h1 = new_Igrad1[slot_h1] if state.pseudo_halfsets else m1_h0
    _post_data, mom1_noise_power = bind.vdam_apply_momenta(
        data_h0, m1_h0, m1_h1, new_Igrad2[k], ori_size, padding_factor, 1, r_max
    )
    _post_data = np.asarray(_post_data)
    mom1_noise_power = np.asarray(mom1_noise_power)
    _dump("post_apply_data", _post_data)
    _dump("mom1_noise_power", mom1_noise_power)

    # Step 6. updateSSNRarrays (update_tau2_with_fsc=false in gradient mode);
    # drives updateCurrentResolution for the next expectation step.
    new_tau2_class = state.tau2_class.copy()
    new_sigma2_class = state.sigma2_class.copy()
    new_fourier_coverage_class = state.fourier_coverage_class.copy()
    new_data_vs_prior_class = state.data_vs_prior_class.copy()
    # RELION's gradient InitialModel path routes class 0 FSC into the common
    # updateSSNRarrays call, then passes per-class FSC to reconstructGrad below.
    fsc_for_ssnr = np.asarray(state.fsc_halves_class[0], dtype=np.float64)
    # RELION calls updateSSNRarrays on BPref[iclass], not on an average with
    # the pseudo-halfset BPref[iclass + nr_classes].  The latter contributes
    # to gradient moments only.  Captured native K=1 buffers confirm that
    # BPref[0].weight matches accum_h0.weight shell-by-shell.
    weight_for_ssnr = accum_h0.weight
    tau2, sigma2, data_vs_prior, fourier_coverage = bind.vdam_update_ssnr_arrays_from_bpref(
        weight_for_ssnr,
        fsc_for_ssnr,
        state.tau2_class[k],
        tau2_fudge_factor,
        ori_size,
        padding_factor,
        1,
        r_max,
        False,
        False,
        False,
    )
    new_tau2_class[k] = np.asarray(tau2, dtype=np.float64)
    new_sigma2_class[k] = np.asarray(sigma2, dtype=np.float64)
    new_data_vs_prior_class[k] = np.asarray(data_vs_prior, dtype=np.float64)
    new_fourier_coverage_class[k] = np.asarray(fourier_coverage, dtype=np.float64)
    _dump("tau2_post_ssnr", new_tau2_class[k])
    _dump("sigma2_post_ssnr", new_sigma2_class[k])
    _dump("data_vs_prior_post_ssnr", new_data_vs_prior_class[k])
    _dump("fourier_coverage_post_ssnr", new_fourier_coverage_class[k])

    # Step 7. reconstructGrad updates Iref[k] (RELION pseudo-halfset path —
    # mom1_noise_power required for correct FSC/tau weighting). Convert
    # recovar↔RELION at the boundary so the gradient is in the same frame as the accumulators.
    from recovar.utils.helpers import recovar_volume_to_relion, relion_volume_to_recovar

    iref_relion_in = recovar_volume_to_relion(np.asarray(state.Iref[k]))
    iref_relion_in = replay._maybe_replay_native_reference_input(
        iref_relion_in,
        iteration=int(getattr(state, "iter", 0)),
        class_idx=k,
    )
    _dump("iref_relion_in", iref_relion_in)
    effective_stepsize = float(grad_current_stepsize) * (
        1.0 - np.exp(-float(3 * state.K + 10) * float(np.asarray(state.pdf_class)[k]))
    )
    _dump("effective_stepsize", np.asarray([effective_stepsize], dtype=np.float64))
    new_Iref = np.array(state.Iref, order="C")
    new_Iref[k] = relion_volume_to_recovar(
        np.asarray(
            bind.vdam_reconstruct_grad(
                iref_relion_in,
                _post_data,
                accum_h0.weight,
                state.fsc_halves_class[k],
                effective_stepsize,
                tau2_fudge_factor,
                ori_size,
                padding_factor,
                1,
                r_max,
                min_resol_shell,
                False,
                True,
                mom1_noise_power,
            )
        )
    )
    _dump("iref_out_recovar_frame", new_Iref[k])
    _dump("iref_out_relion_frame", recovar_volume_to_relion(new_Iref[k]))

    new_state = replace(state)
    new_state.Iref = new_Iref
    new_state.Igrad1 = new_Igrad1
    new_state.Igrad2 = new_Igrad2
    new_state.tau2_class = new_tau2_class
    new_state.sigma2_class = new_sigma2_class
    new_state.data_vs_prior_class = new_data_vs_prior_class
    new_state.fourier_coverage_class = new_fourier_coverage_class
    return new_state


def vdam_m_step_single_class_native(
    state: InitialModelState,
    k: int,
    accum_h0: VdamAccumulator,
    accum_h1: VdamAccumulator | None,
    *,
    grad_current_stepsize: float,
    tau2_fudge_factor: float,
    grad_min_resol_shell: float | None = None,
    padding_factor: int = 1,
    mstep_compute_dtype: str = "float64",
) -> InitialModelState:
    """Drop-in for ``vdam_m_step_single_class`` running RELION's primitives (float64 only)."""

    from relax.vdam import m_step as production

    if mstep_compute_dtype != "float64":
        raise ValueError("RELION's step-by-step M-step is double precision; pass --mstep-compute-dtype float64")
    production.validate_mstep_inputs(state, k, accum_h1)
    iteration = int(getattr(state, "iter", 0))
    accum_h0, accum_h1 = replay._maybe_replay_native_bpref_accumulators(
        accum_h0, accum_h1, iteration=iteration, class_idx=k
    )
    if not production._has_relion_reconstruction_weight(state, k, accum_h0):
        return state
    dump_dir = os.environ.get("RELAX_MSTEP_DUMP_DIR")
    do_dump = dump_dir is not None and iteration == int(os.environ.get("RELAX_MSTEP_DUMP_ITER", "1"))
    return native_vdam_m_step_single_class(
        state,
        k,
        accum_h0,
        accum_h1,
        grad_current_stepsize=grad_current_stepsize,
        tau2_fudge_factor=tau2_fudge_factor,
        padding_factor=padding_factor,
        # The current-size radius, not initZeros(-1) (backprojector.cpp::initZeros).
        r_max=state.current_size // 2,
        min_resol_shell=production._grad_min_resol_shell_from_state(state, grad_min_resol_shell),
        dump_dir=dump_dir if do_dump else None,
        dump_prefix=f"c{k}_" if state.K > 1 else "",
    )


def main(argv=None) -> int:
    """``relax initial_model`` with this oracle as the M-step (replays and dumps enabled)."""

    from relax.commands import initial_model
    from relax.vdam import m_step

    # The oracle reads the replay variables itself; the production guard refuses them.
    m_step._validate_mstep_precision_route = lambda dtype: None
    m_step.vdam_m_step_single_class = vdam_m_step_single_class_native
    return initial_model.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
