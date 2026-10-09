"""Exact projector/spectrum reuse and the refresh-to-E-step lifetime."""
from dataclasses import replace

import numpy as np
import pytest

from relax.vdam import estep_setup as adapter
from relax.vdam import iteration_loop as loop
from relax.vdam.bootstrap_iref import initialise_denovo_state
from relax.vdam.native_options import VdamEnvironment
from relax.vdam.schedules import DEFAULT_GRAD_MU
from recovar.utils.helpers import recovar_volume_to_relion
from helpers.float_compare import assert_matches
from helpers.vdam import relative_metrics

pytestmark = pytest.mark.unit


@pytest.mark.requires_relion_bind
@pytest.mark.parametrize("classes,current_size,padding", [(1, 8, 1), (2, 12, 1), (1, 16, 2)])
def test_shared_device_projector_matches_both_native_calls(
    classes, current_size, padding, monkeypatch, tmp_path
):
    from relax.relion_bind import _relion_bind_core as bind

    state = initialise_denovo_state(
        box_size=16, pixel_size=1.0, K=classes, nr_iter=2,
        n_directions=3, pseudo_halfsets=True,
    )
    state.Iref = np.random.default_rng(17).normal(size=state.Iref.shape)
    state.current_size = current_size
    references_before = state.Iref.copy()
    native_projector = bind.compute_fourier_transform_map
    builds = []

    def counted_projector(*args):
        builds.append(1)
        return native_projector(*args)

    monkeypatch.setattr(bind, "compute_fourier_transform_map", counted_projector)
    monkeypatch.setenv(adapter._RELION_PROJECTOR_DUMP_DIR_ENV, str(tmp_path))
    inputs, power = adapter.prepare_relion_projector_class_inputs_and_power(
        state, padding_factor=padding
    )
    assert not builds  # VDAM prepares the projector on the device only.
    with np.load(tmp_path / "iter000_relion_projector_half.npz") as dumped:
        assert_matches(dumped["projector_half"], inputs[0])
        assert int(dumped["current_size"]) == current_size
        assert int(dumped["padding_factor"]) == padding
    expected_half = []
    expected_power = []
    for ref in state.Iref:
        native = np.ascontiguousarray(recovar_volume_to_relion(ref))
        half, _, _, _, radius, _, _ = bind.compute_fourier_transform_map(
            native, 16, padding, 1, current_size, True, 2
        )
        expected_half.append(np.asarray(half))
        expected_power.append(bind.vdam_projector_power_spectrum(
            native, 16, padding, 1, current_size, True, 2
        ))
        assert inputs[1] == radius
    # The slab is RELION's float texture: the double native slab rounded once
    # to complex64. The spectrum keeps the native FP64 projector contract.
    assert inputs[0].dtype == np.complex64
    assert np.all(relative_metrics(np.asarray(expected_half), inputs[0]) <= 4 * np.finfo(np.float32).eps)
    assert power.dtype == np.float64
    assert np.all(relative_metrics(np.asarray(expected_power), power) < 1e-12)
    assert_matches(state.Iref, references_before)


def test_context_builds_once_and_consumes_once(monkeypatch):
    state = initialise_denovo_state(
        box_size=8, pixel_size=1.0, K=1, nr_iter=2,
        n_directions=3, pseudo_halfsets=True,
    )
    calls = []

    def prepare(current, *, padding_factor):
        calls.append((current.iter, current.Iref.copy()))
        return (current.Iref.copy(), 4), np.full((1, 5), current.iter)

    monkeypatch.setattr(adapter, "prepare_relion_projector_class_inputs_and_power", prepare)
    ctx = adapter.IterationProjectorContext()
    for iteration in (1, 2):
        state = replace(state, iter=iteration, Iref=state.Iref + 1)
        before = state.tau2_class.copy()
        refreshed = ctx.refresh(state, padding_factor=1)
        assert_matches(state.tau2_class, before)
        assert_matches(refreshed.tau2_class, np.full((1, 5), iteration))
        inputs = ctx.take(refreshed, padding_factor=1)
        assert_matches(inputs[0], state.Iref)
        with pytest.raises(ValueError, match="no projector refresh"):
            ctx.take(refreshed, padding_factor=1)
        assert ctx.reference is None
    assert len(calls) == 2
    assert not np.array_equal(calls[0][1], calls[1][1])


@pytest.mark.parametrize("change", ["reference", "iteration", "geometry", "padding"])
def test_context_rejects_stale_handoff_and_clears(monkeypatch, change):
    state = initialise_denovo_state(
        box_size=8, pixel_size=1.0, K=1, nr_iter=2,
        n_directions=3, pseudo_halfsets=True,
    )
    monkeypatch.setattr(adapter, "prepare_relion_projector_class_inputs_and_power",
                        lambda *a, **k: ((None, 4), np.ones((1, 5))))
    ctx = adapter.IterationProjectorContext()
    current = ctx.refresh(state, padding_factor=1)
    kwargs = {"padding_factor": 1}
    if change == "reference":
        current = replace(current, Iref=current.Iref.copy())
    elif change == "iteration":
        current = replace(current, iter=1)
    elif change == "geometry":
        current = replace(current, current_size=6)
    else:
        kwargs["padding_factor"] = 2
    with pytest.raises(ValueError, match="reference or geometry"):
        ctx.take(current, **kwargs)
    assert ctx.prepared is ctx.reference is ctx.geometry is None


@pytest.mark.parametrize("mstep_compute_dtype", ["float32", "float64"])
def test_loop_callback_is_once_before_estep(monkeypatch, mstep_compute_dtype):
    state = initialise_denovo_state(
        box_size=8, pixel_size=1.0, K=1, nr_iter=2,
        n_directions=3, pseudo_halfsets=True,
    )
    events = []

    def refresh(current, *, padding_factor):
        events.append((current.iter, "refresh", current.current_size))
        assert padding_factor == 1
        return current

    def estep(current, ids, halves):
        events.append((current.iter, "estep", current.current_size))
        return [], {"max_posterior_per_image": np.ones(len(ids)), "class_posterior_sums": np.asarray([float(len(ids))])}

    def mstep(current, **kwargs):
        assert kwargs["mstep_compute_dtype"] == mstep_compute_dtype
        return current

    monkeypatch.setattr(loop, "vdam_m_step", mstep)
    loop.run_vdam_iterations(
        state, nr_particles=20, optics_group_by_particle=[0] * 20,
        grad_ini_subset_size=10, grad_fin_subset_size=10, tau2_fudge_arg=4.0,
        grad_em_iters=0, random_seed=29,
        expectation_step=estep,
        projector_refresh_fn=refresh, update=loop.VdamUpdate(padding_factor=1, mstep_compute_dtype=mstep_compute_dtype),
        grad_ini_frac=0.3,
        grad_fin_frac=0.2,
        mu=DEFAULT_GRAD_MU,
        uniform_class_direction_prior=False,
        environment=VdamEnvironment(),
    )
    expected = ["refresh", "estep"] * 2
    assert [event[1] for event in events] == expected
