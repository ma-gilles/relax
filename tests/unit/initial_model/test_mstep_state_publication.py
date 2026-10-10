"""M-step state publication: updated slots on the device, the input state untouched."""

from dataclasses import fields

import jax
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.vdam import initial_model_state_stub

from relax.vdam import m_step
from relax.vdam.state import half_slot_index

pytestmark = pytest.mark.unit
CHANGED = {"Iref", "Igrad1", "Igrad2", "tau2_class", "sigma2_class", "data_vs_prior_class", "fourier_coverage_class"}


def _case(K, pseudo, order):
    rng = np.random.default_rng(21)

    def array(shape, complex_values=False):
        values = rng.normal(size=shape)
        if complex_values:
            values = values + 1j * rng.normal(size=shape)
        # Untouched slots must retain nonfinite payloads and signed zeros too.
        words = values.view(np.uint64).reshape(-1)
        patterns = [0, 0x8000000000000000, 0x7FF8000000000021, 0x7FF0000000000000]
        words[: min(4, words.size)] = patterns[: min(4, words.size)]
        if order == "F":
            values = np.asfortranarray(values)
        elif order == "strided":
            values = values[..., ::-1]
        return values

    state = initial_model_state_stub(
        K=K,
        box_size=4,
        pseudo_halfsets=pseudo,
        Iref=array((K, 4, 4, 4)),
        Igrad1=array(((2 if pseudo else 1) * K, 4, 4, 3), True),
        Igrad2=array((K, 4, 4, 3), True),
        pdf_class=np.full(K, 1 / K),
        fsc_halves_class=np.ones((K, 3)),
    )
    for name in CHANGED - {"Iref", "Igrad1", "Igrad2"}:
        setattr(state, name, array((K, 3)))
    return state


def _call(state, k, transaction, monkeypatch):
    from relax.relion import vdam_mstep

    monkeypatch.setattr(vdam_mstep, "relion_vdam_m_step_host", lambda *args, **options: transaction(*args))
    accum = m_step.VdamAccumulator(np.zeros((4, 4, 3), dtype=np.complex128), np.ones((4, 4, 3)), k, 0)
    return m_step._run_m_step_transaction(
        state,
        k,
        accum,
        accum if state.pseudo_halfsets else None,
        grad_current_stepsize=0.3,
        tau2_fudge_factor=4.0,
        padding_factor=1,
        r_max=2,
        min_resol_shell=1.0,
        mstep_compute_dtype="float64",
        average_ctf2=None,
    )


@pytest.mark.parametrize("K,k", [(1, 0), (4, 0), (4, 1), (4, 3)])
@pytest.mark.parametrize("pseudo", [False, True])
@pytest.mark.parametrize("order", ["C", "F", "strided"])
def test_publication_writes_the_updated_slots_and_leaves_the_input_state(K, k, pseudo, order, monkeypatch):
    state = _case(K, pseudo, order)
    originals = {
        f.name: getattr(state, f.name).tobytes()
        for f in fields(state)
        if isinstance(getattr(state, f.name), np.ndarray)
    }
    # Transaction outputs deliberately alias the input state.
    result = {
        "iref": state.Iref[k],
        "mom1_h0": state.Igrad1[k],
        "mom1_h1": state.Igrad1[-1],
        "mom2": state.Igrad2[k],
        "tau2": state.tau2_class[k],
        "sigma2": state.sigma2_class[k],
        "data_vs_prior": state.data_vs_prior_class[k],
        "fourier_coverage": state.fourier_coverage_class[k],
    }
    for name in originals:
        getattr(state, name).flags.writeable = False
    calls = []

    def transaction(*args):
        calls.append(args)
        return result

    actual = _call(state, k, transaction, monkeypatch)
    assert len(calls) == 1
    # The reference is the input state with each updated slot replaced.
    h1 = half_slot_index(k, 1, K, True) if pseudo else None
    updates = {
        "Iref": {k: result["iref"]},
        "Igrad1": {k: result["mom1_h0"], **({h1: result["mom1_h1"]} if pseudo else {})},
        "Igrad2": {k: result["mom2"]},
        "tau2_class": {k: result["tau2"]},
        "sigma2_class": {k: result["sigma2"]},
        "data_vs_prior_class": {k: result["data_vs_prior"]},
        "fourier_coverage_class": {k: result["fourier_coverage"]},
    }
    assert actual is not state
    for f in fields(state):
        a, before = getattr(actual, f.name), getattr(state, f.name)
        if f.name in CHANGED:
            expected = np.array(before, order="C")
            for slot, value in updates[f.name].items():
                expected[slot] = value
            assert a.shape == expected.shape and np.dtype(a.dtype) == expected.dtype, f.name
            assert_matches(np.asarray(a), expected, err_msg=f.name, strict=True)
            if f.name in ("Iref", "Igrad1", "Igrad2"):
                assert isinstance(a, jax.Array), f.name  # the volumes stay on the device
        elif isinstance(a, np.ndarray):
            assert a is before
        else:
            assert a == before
        if isinstance(before, np.ndarray):
            assert before.tobytes() == originals[f.name], f.name
