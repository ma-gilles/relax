"""Single-class VDAM M-step of the native InitialModel driver.

The per-class M-step transaction, its precision-route validation, the RELION
reconstruction-weight and resolution-shell rules live here. RELION's own
step-by-step M-step with its dumps and replays is the oracle in
``diagnostics.vdam_native_mstep``; ``m_step`` keeps the multi-class driver.
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import partial
from typing import Literal, Optional

import jax.numpy as jnp
import numpy as np

from relax.relion.macros import relion_round
from relax.vdam.state import InitialModelState, VdamAccumulator, half_slot_index

XMIPP_EQUAL_ACCURACY: float = 1e-6


RELION_DEFAULT_GRAD_MIN_RESOL_ANGSTROM: float = 20.0


_MSTEP_F32_STATE_DTYPES = {
    "Iref": np.float32,
    "Igrad1": np.complex64,
    "Igrad2": np.complex64,
    "sigma2_class": np.float32,
    "data_vs_prior_class": np.float32,
    "fourier_coverage_class": np.float32,
}


def _prepare_mstep_state_precision(state, mstep_compute_dtype):
    """Convert M-owned numerical state once after bootstrap or continuation.

    FSC, authoritative tau2, noise and priors retain their existing precision.
    Bootstrap and projector refresh are not part of this F32 transaction route.
    """
    if mstep_compute_dtype == "float64":
        return state
    if mstep_compute_dtype != "float32":
        raise ValueError(f"Unknown mstep_compute_dtype: {mstep_compute_dtype!r}")
    return replace(
        state,
        **{
            name: np.asarray(getattr(state, name)).astype(dtype, copy=True)
            for name, dtype in _MSTEP_F32_STATE_DTYPES.items()
        },
    )


def _validate_mstep_precision_route(mstep_compute_dtype: Literal["float32", "float64"]) -> None:
    """Reject an unknown precision and the native parity replays, which only the diagnostics runner takes."""
    if mstep_compute_dtype not in {"float32", "float64"}:
        raise ValueError(f"Unknown mstep_compute_dtype: {mstep_compute_dtype!r}")
    requested = [name for name in NATIVE_PARITY_REPLAY_ENVS if os.environ.get(name, "").strip()]
    if requested:
        raise ValueError(
            f"{requested} replay RELION's M-step; run python -m relax.diagnostics.vdam_native_mstep instead"
        )


def _validate_mstep_state_precision(state: InitialModelState) -> None:
    for name, dtype in _MSTEP_F32_STATE_DTYPES.items():
        value = getattr(state, name)
        # Read the dtype without np.asarray, which would copy a device volume back.
        value_dtype = value.dtype if hasattr(value, "dtype") else np.asarray(value).dtype
        if np.dtype(value_dtype) != np.dtype(dtype):
            raise ValueError(f"float32 M-step requires state.{name} dtype {np.dtype(dtype)}")


# The native replays of RELION's M-step intermediates (relax.diagnostics.vdam_mstep_replay).
NATIVE_PARITY_REPLAY_ENVS = (
    "RELAX_VDAM_NATIVE_SECOND_MOMENT_REPLAY_BIN",
    "RELAX_VDAM_NATIVE_FIRST_MOMENT_REPLAY_BIN",
    "RELAX_VDAM_NATIVE_BPREF_DATA_REPLAY_BIN",
    "RELAX_VDAM_NATIVE_BPREF_WEIGHT_REPLAY_BIN",
    "RELAX_VDAM_NATIVE_IREF_INPUT_REPLAY_BIN",
)


def validate_mstep_inputs(state: InitialModelState, k: int, accum_h1) -> None:
    """The class index and the half-set accumulator layout of one class's M-step."""
    if not (0 <= k < state.K):
        raise ValueError(f"class index {k} out of range")
    if state.pseudo_halfsets and accum_h1 is None:
        raise ValueError("pseudo_halfsets=True requires accum_h1")
    if not state.pseudo_halfsets and accum_h1 is not None:
        raise ValueError("pseudo_halfsets=False must have accum_h1=None")


def _relion_resolution_shell(ori_size: int, pixel_size: float, resolution_angstrom: float) -> int:
    """RELION ``MlModel::getPixelFromResolution(1./resolution_angstrom)``."""
    if resolution_angstrom <= 0.0:
        raise ValueError(f"resolution_angstrom must be positive, got {resolution_angstrom}")
    shell = float(ori_size) * float(pixel_size) / float(resolution_angstrom)
    return relion_round(shell)


def _grad_min_resol_shell_from_state(
    state: InitialModelState,
    grad_min_resol_shell: float | None,
) -> float:
    """Default ``reconstructGrad`` min-resol shell from RELION InitialModel."""
    if grad_min_resol_shell is not None:
        return float(grad_min_resol_shell)
    return float(
        _relion_resolution_shell(
            int(state.ori_size),
            float(state.pixel_size),
            RELION_DEFAULT_GRAD_MIN_RESOL_ANGSTROM,
        )
    )


def _has_relion_reconstruction_weight(state: InitialModelState, k: int, accum_h0: VdamAccumulator) -> bool:
    """RELION only reconstructs classes with active prior and nonzero BPref weight."""
    if float(np.asarray(state.pdf_class, dtype=np.float64)[k]) <= 0.0:
        return False
    return float(np.sum(np.asarray(accum_h0.weight, dtype=np.float64))) > XMIPP_EQUAL_ACCURACY


def _run_m_step_transaction(
    transaction,
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
    mstep_compute_dtype: Literal["float32", "float64"],
) -> InitialModelState:
    """Apply one shared-layout transaction and publish the new state on the device.

    ``transaction`` takes and returns the reference in RECOVAR's axes
    (``relion_vdam_m_step_host(..., recovar_layout=True, device_volumes=True)``).
    The reference and moment volumes stay device arrays across iterations: each
    update writes its slots into a new device array, so the previous state is
    never modified, and the host reads the volumes back only to write them.
    """
    if mstep_compute_dtype == "float32":
        _validate_mstep_state_precision(state)
    slot_h0 = half_slot_index(k, 0, state.K, state.pseudo_halfsets)
    slot_h1 = half_slot_index(k, 1, state.K, True) if state.pseudo_halfsets else None
    effective_stepsize = float(grad_current_stepsize) * (
        1.0 - np.exp(-float(3 * state.K + 10) * float(np.asarray(state.pdf_class)[k]))
    )
    result = transaction(
        state.Iref[k],
        accum_h0.data,
        accum_h0.weight,
        accum_h1.data if accum_h1 is not None else None,
        accum_h1.weight if accum_h1 is not None else None,
        state.Igrad1[slot_h0],
        state.Igrad1[slot_h1] if slot_h1 is not None else None,
        state.Igrad2[k],
        state.fsc_halves_class[0],
        state.fsc_halves_class[k],
        state.tau2_class[k],
        effective_stepsize,
        tau2_fudge_factor,
        state.ori_size,
        padding_factor,
        1,
        r_max,
        min_resol_shell,
    )
    if mstep_compute_dtype == "float32":
        expected_dtypes = {
            "iref": np.float32,
            "mom1_h0": np.complex64,
            "mom2": np.complex64,
            "sigma2": np.float32,
            "data_vs_prior": np.float32,
            "fourier_coverage": np.float32,
        }
        if slot_h1 is not None:
            expected_dtypes["mom1_h1"] = np.complex64
        for name, dtype in expected_dtypes.items():
            if np.dtype(result[name].dtype) != np.dtype(dtype):
                raise ValueError(f"float32 M-step output {name} must have dtype {np.dtype(dtype)}")
        prior = np.asarray(state.tau2_class[k])
        returned_prior = np.asarray(result["tau2"])
        if (
            returned_prior.dtype != prior.dtype
            or returned_prior.shape != prior.shape
            or returned_prior.tobytes() != prior.tobytes()
        ):
            raise ValueError("float32 M-step must preserve authoritative tau2")
    out = replace(state)
    out.Iref = jnp.asarray(state.Iref).at[k].set(result["iref"])
    igrad1 = jnp.asarray(state.Igrad1).at[slot_h0].set(result["mom1_h0"])
    out.Igrad1 = igrad1 if slot_h1 is None else igrad1.at[slot_h1].set(result["mom1_h1"])
    out.Igrad2 = jnp.asarray(state.Igrad2).at[k].set(result["mom2"])
    for attribute, key in (
        ("tau2_class", "tau2"),
        ("sigma2_class", "sigma2"),
        ("data_vs_prior_class", "data_vs_prior"),
        ("fourier_coverage_class", "fourier_coverage"),
    ):
        values = getattr(state, attribute).copy()
        dtype = values.dtype if mstep_compute_dtype == "float32" else np.float64
        values[k] = np.asarray(result[key], dtype=dtype)
        setattr(out, attribute, values)
    return out


def vdam_m_step_single_class(
    state: InitialModelState,
    k: int,
    accum_h0: VdamAccumulator,
    accum_h1: Optional[VdamAccumulator],
    *,
    grad_current_stepsize: float,
    tau2_fudge_factor: float,
    grad_min_resol_shell: float | None = None,
    padding_factor: int = 1,
    mstep_compute_dtype: Literal["float32", "float64"] = "float32",
    average_ctf2=None,
) -> InitialModelState:
    """VDAM M-step for one class: the transaction of
    :func:`relax.relion.relion_vdam_mstep.relion_vdam_m_step_host`.

    RELION's own step-by-step M-step, with its intermediate dumps and parity
    replays, is the oracle in :mod:`relax.diagnostics.vdam_native_mstep`.

    Pseudo-halfsets: FSC/noise-power is derived from the halfset-data difference
    in ``applyMomenta``; ``reconstructGrad`` then uses ``mom1_noise_power``.
    ``average_ctf2``: the E-step's CTF-premultiplied average CTF^2 (SSNR tau2 correction), or None.
    """
    _validate_mstep_precision_route(mstep_compute_dtype)
    if mstep_compute_dtype == "float32":
        _validate_mstep_state_precision(state)
    validate_mstep_inputs(state, k, accum_h1)
    if not _has_relion_reconstruction_weight(state, k, accum_h0):
        return state

    # Pass the current-size radius explicitly instead of initZeros(-1)
    # (backprojector.cpp::initZeros; ml_optimiser.cpp:5846).
    r_max = state.current_size // 2
    min_resol_shell = _grad_min_resol_shell_from_state(state, grad_min_resol_shell)
    from relax.relion.relion_vdam_mstep import relion_vdam_m_step_host

    transaction = partial(
        relion_vdam_m_step_host, recovar_layout=True, device_volumes=True, average_ctf2=average_ctf2
    )
    if mstep_compute_dtype == "float32":
        transaction = partial(transaction, compute_dtype=np.float32)
    return _run_m_step_transaction(
        transaction,
        state,
        k,
        accum_h0,
        accum_h1,
        grad_current_stepsize=grad_current_stepsize,
        tau2_fudge_factor=tau2_fudge_factor,
        padding_factor=padding_factor,
        r_max=r_max,
        min_resol_shell=min_resol_shell,
        mstep_compute_dtype=mstep_compute_dtype,
    )
