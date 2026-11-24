"""The VDAM shrinkage gate (docs/math/vdam_gate_derivation.md): RELION's, the power gates and the cross-power gate."""

from dataclasses import replace

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.vdam import projector_power_refresh

from relax.commands.initial_model import _native_options_dict, make_parser
from relax.runtime.engine_tuning import DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM
from relax.steps.sgd_schedule import DEFAULT_GRAD_MU
from relax.vdam import relion_mstep
from relax.vdam.bootstrap_iref import initialise_denovo_state
from relax.vdam.gate import (
    CrossWindowGate,
    FullDataOverlapPowerGate,
    FullDataPowerGate,
    PowerGate,
    RelionGate,
    SeenOverlapPowerGate,
    WindowPowerGate,
    advance_noise_coefficients,
    advance_noise_weight,
    advance_window_coefficients,
    full_data_disagreement_scale,
    overlap_disagreement_scale,
    seen_disagreement_scale,
    vdam_gate,
    window_disagreement_scale,
)
from relax.vdam.iteration_loop import VdamUpdate, run_vdam_iterations
from relax.vdam.native_options import NativeInitialModelOptions, VdamEnvironment
from relax.vdam.ports import NoProbe, VdamObserver
from relax.vdam.relion_mstep import FIRST_MOMENT_DECAY, MStepGate, pad_moment, relion_vdam_m_step_host
from relax.vdam.state import VdamAccumulator

pytestmark = pytest.mark.unit

WEIGHT_OF_THE_NEW_DIRECTION = 1.0 - FIRST_MOMENT_DECAY


def test_the_noise_weight_is_the_sum_of_squared_moment_coefficients_over_the_batches():
    """first_t = decay^(t-1) d_1 + sum_i (1 - decay) decay^(t-i) d_i, each d_i with noise power 4 / count_i."""
    counts = [300] * 7 + [1200, 1200, 50_000, 300]
    weight = None
    for count in counts:
        weight = advance_noise_weight(weight, count)
    t = len(counts)
    explicit = FIRST_MOMENT_DECAY ** (2 * (t - 1)) * 4 / counts[0] + sum(
        WEIGHT_OF_THE_NEW_DIRECTION**2 * FIRST_MOMENT_DECAY ** (2 * (t - i)) * 4 / counts[i - 1]
        for i in range(2, t + 1)
    )
    assert_matches(weight, explicit, rtol=1e-12)
    assert advance_noise_weight(None, 300) == 4 / 300


def test_for_a_constant_batch_the_scale_is_the_batch_share_over_the_running_average_noise_reduction():
    """(b / N) / e with e = (1 - decay) / (1 + decay) = (1 - decay)^2 / (1 - decay^2), from the first iteration on."""
    weight = None
    for _ in range(400):
        weight = advance_noise_weight(weight, 300)
    reduction = (1 - FIRST_MOMENT_DECAY) / (1 + FIRST_MOMENT_DECAY)
    assert reduction == pytest.approx(WEIGHT_OF_THE_NEW_DIRECTION**2 / (1 - FIRST_MOMENT_DECAY**2), rel=1e-12)
    assert_matches(full_data_disagreement_scale(weight, 100_000), 300 / 100_000 / reduction, rtol=1e-9)
    assert full_data_disagreement_scale(advance_noise_weight(None, 300), 100_000) == pytest.approx(300 / 100_000)


def _state(*, pseudo_halfsets=True, box_size=8):
    return initialise_denovo_state(
        box_size=box_size, pixel_size=1.0, K=1, nr_iter=200, n_directions=3, pseudo_halfsets=pseudo_halfsets
    )


def test_the_full_data_gate_carries_the_weight_of_each_gradient_batch_in_the_state():
    gate, state = FullDataPowerGate(n_particles=1000), _state()
    assert state.gate_noise_weight is None
    expected = None
    for count in (100, 150, 400):
        expected = advance_noise_weight(expected, count)
        state, scale = gate.advance(replace(state, subset_particle_ids=np.arange(count)))
        assert state.gate_noise_weight == expected
        assert scale == MStepGate(full_data_disagreement_scale(expected, 1000), cross_power=False)


def test_an_iteration_without_pseudo_halfsets_leaves_the_weight_and_scales_by_one():
    gate, state = FullDataPowerGate(n_particles=1000), _state(pseudo_halfsets=False)
    tail, scale = gate.advance(replace(state, subset_particle_ids=np.arange(1000), gate_noise_weight=0.02))
    assert tail.gate_noise_weight == 0.02 and scale == MStepGate(1.0, cross_power=False)


def _halves(particle_ids):
    """RELION's pseudo-halves: the parity of the particle id."""
    return (np.asarray(particle_ids) % 2).astype(np.int8)


def test_without_repeated_particles_the_overlap_scale_is_the_independent_batch_scale():
    """Disjoint batches with halves of equal size (consecutive ids, even counts): both scales count the same noise.

    The independent-batch weight takes each half as count / 2 particles; with unequal halves it is an approximation.
    """
    n = 10_000
    order = np.arange(n)
    coefficients, weight, start = None, None, 0
    for count in (200, 200, 400, 600, 800):
        ids = order[start : start + count]
        start += count
        coefficients = advance_noise_coefficients(coefficients, ids, _halves(ids), n)
        weight = advance_noise_weight(weight, count)
        assert_matches(overlap_disagreement_scale(coefficients, n), full_data_disagreement_scale(weight, n), rtol=1e-9)


def test_when_every_batch_is_the_whole_data_set_the_running_average_reduces_nothing():
    """All particles in every iteration: each one keeps the weight 1 / n_h of its half, the full-data half map's."""
    n = 1001
    ids = np.arange(n)
    coefficients = None
    for _ in range(30):
        coefficients = advance_noise_coefficients(coefficients, ids, _halves(ids), n)
        assert_matches(overlap_disagreement_scale(coefficients, n), 1.0, rtol=1e-12)
    weight = None
    for _ in range(30):
        weight = advance_noise_weight(weight, n)
    assert full_data_disagreement_scale(weight, n) > 18  # the independent-batch scale: 1 / e = 19


@pytest.mark.parametrize("fraction", [0.02, 0.1, 0.5, 1.0])
def test_the_overlap_scale_recovers_the_noise_of_two_full_data_half_maps_when_batches_repeat_particles(fraction):
    """Monte Carlo: each particle carries one fixed noise vector; batches are random draws of ``fraction`` of them.

    The halves' first moments are built as the M-step does (first = direction, then 0.9 first + 0.1 direction, the
    direction of a half the mean of its particles), and their disagreement power times the scale is compared with that of
    two full-data half maps (the two halves' means over all particles). 3000 voxels and 40 iterations; over 5 seeds the
    largest relative error is 4.4%, so rtol is 0.1. The independent-batch scale is off by about 1 + 18 * fraction.
    """
    n, voxels, rng = 2000, 3000, np.random.default_rng(17)
    noise = rng.standard_normal((n, voxels))
    halves = _halves(np.arange(n))
    batch = int(fraction * n)
    first = coefficients = weight = None
    for _ in range(40):
        ids = np.sort(rng.choice(n, batch, replace=False))
        direction = np.stack([noise[ids[halves[ids] == h]].mean(axis=0) for h in (0, 1)])
        first = direction if first is None else 0.9 * first + 0.1 * direction
        coefficients = advance_noise_coefficients(coefficients, ids, halves[ids], n)
        weight = advance_noise_weight(weight, batch)
    measured = np.mean((first[0] - first[1]) ** 2)
    full_data = np.mean((noise[halves == 0].mean(axis=0) - noise[halves == 1].mean(axis=0)) ** 2)
    assert_matches(measured * overlap_disagreement_scale(coefficients, n), full_data, rtol=0.1)
    independent = measured * full_data_disagreement_scale(weight, n) / full_data
    assert independent == pytest.approx(1 + 18 * fraction, rel=0.15)


def test_the_overlap_gate_carries_the_coefficients_and_the_power_gate_scales_by_one():
    gate, state = FullDataOverlapPowerGate(n_particles=1000), _state()
    expected = None
    for ids in (np.arange(0, 200), np.arange(100, 400), np.arange(0, 1000)):
        expected = advance_noise_coefficients(expected, ids, _halves(ids), 1000)
        state, scale = gate.advance(replace(state, subset_particle_ids=ids, subset_halfset_ids=_halves(ids)))
        assert_matches(state.gate_noise_coefficients, expected, strict=True)
        assert scale == MStepGate(overlap_disagreement_scale(expected, 1000), cross_power=False)
    tail, scale = gate.advance(replace(state, pseudo_halfsets=False))
    assert scale == MStepGate(1.0, cross_power=False) and tail.gate_noise_coefficients is state.gate_noise_coefficients
    unchanged, scale = PowerGate().advance(state)
    assert unchanged is state and scale == MStepGate(1.0, cross_power=False)


def test_the_seen_scale_is_one_for_the_first_batch_and_the_overlap_scale_once_every_particle_is_seen():
    gate, state = SeenOverlapPowerGate(n_particles=1000), _state()
    ids = np.arange(0, 200)
    state, scale = gate.advance(replace(state, subset_particle_ids=ids, subset_halfset_ids=_halves(ids)))
    assert scale.scale == pytest.approx(1.0, rel=1e-12)
    for ids in (np.arange(100, 400), np.arange(0, 1000)):
        state, scale = gate.advance(replace(state, subset_particle_ids=ids, subset_halfset_ids=_halves(ids)))
    assert scale == MStepGate(overlap_disagreement_scale(state.gate_noise_coefficients, 1000), cross_power=False)
    assert seen_disagreement_scale(state.gate_noise_coefficients) == scale.scale


@pytest.mark.parametrize("step", [0.3, 0.5, 0.9])
def test_the_window_scale_recovers_the_noise_of_the_reference_through_the_update(step):
    """Monte Carlo of the update with the gate open and the second moment 1, particles with fixed noise.

    Each half's direction is the mean noise of its batch particles minus the reference; first = direction, then
    0.9 first + 0.1 direction; reference += step * mean(first). The reference's noise power times 4 must equal the
    measured disagreement times the window scale. 4000 voxels, 60 iterations of 10% batches; rtol .1.
    """
    n, voxels, rng = 2000, 4000, np.random.default_rng(23)
    noise = rng.standard_normal((n, voxels))
    halves = _halves(np.arange(n))
    reference = np.zeros(voxels)
    first = coefficients = moment = window = None
    for _ in range(60):
        ids = np.sort(rng.choice(n, n // 10, replace=False))
        direction = np.stack([noise[ids[halves[ids] == h]].mean(axis=0) for h in (0, 1)]) - reference
        first = direction if first is None else 0.9 * first + 0.1 * direction
        reference = reference + step * first.mean(axis=0)
        coefficients = advance_noise_coefficients(coefficients, ids, halves[ids], n)
        moment, window = advance_window_coefficients(moment, window, ids, halves[ids], n, step)
    measured = np.mean((first[0] - first[1]) ** 2)
    assert_matches(measured * window_disagreement_scale(window, coefficients), 4 * np.mean(reference**2), rtol=0.1)


def test_the_window_scale_is_the_step_squared_at_the_first_iteration_and_one_when_batches_are_all_particles():
    ids = np.arange(1000)
    coefficients = advance_noise_coefficients(None, ids, _halves(ids), 1000)
    moment, window = advance_window_coefficients(None, None, ids, _halves(ids), 1000, 0.5)
    assert window_disagreement_scale(window, coefficients) == pytest.approx(0.25, rel=1e-12)
    for _ in range(400):
        coefficients = advance_noise_coefficients(coefficients, ids, _halves(ids), 1000)
        moment, window = advance_window_coefficients(moment, window, ids, _halves(ids), 1000, 0.5)
    assert window_disagreement_scale(window, coefficients) == pytest.approx(1.0, rel=1e-6)
    gate, state = WindowPowerGate(n_particles=1000), _state()
    state, scale = gate.advance(
        replace(state, subset_particle_ids=ids, subset_halfset_ids=_halves(ids), grad_current_stepsize=0.5)
    )
    assert scale.scale == pytest.approx(0.25, rel=1e-12) and state.gate_window_reference is not None
    tail, scale = gate.advance(replace(state, pseudo_halfsets=False))
    assert scale == MStepGate(1.0, cross_power=False) and tail.gate_window_reference is state.gate_window_reference


def test_the_cross_window_gate_carries_the_window_state_and_asks_for_the_cross_power():
    ids = np.arange(1000)
    state = replace(_state(), subset_particle_ids=ids, subset_halfset_ids=_halves(ids), grad_current_stepsize=0.5)
    window_state, window = WindowPowerGate(n_particles=1000).advance(state)
    cross_state, cross = CrossWindowGate(n_particles=1000).advance(state)
    assert cross == MStepGate(window.scale, cross_power=True)
    assert_matches(cross_state.gate_window_reference, window_state.gate_window_reference, rtol=1e-12)
    assert_matches(cross_state.gate_noise_coefficients, window_state.gate_noise_coefficients, rtol=1e-12)
    tail, scale = CrossWindowGate(n_particles=1000).advance(replace(cross_state, pseudo_halfsets=False))
    assert scale == MStepGate(1.0, cross_power=True) and tail.gate_window_reference is cross_state.gate_window_reference


def test_relion_gate_takes_no_scale_and_leaves_the_state():
    state = _state()
    unchanged, scale = RelionGate().advance(state)
    assert unchanged is state and scale is None
    assert vdam_gate("relion", 1000) == RelionGate()
    assert vdam_gate("full_data_power", 1000) == FullDataPowerGate(n_particles=1000)
    assert vdam_gate("full_data_power_overlap", 1000) == FullDataOverlapPowerGate(n_particles=1000)
    assert vdam_gate("power", 1000) == PowerGate()
    assert vdam_gate("cross_window", 1000) == CrossWindowGate(n_particles=1000)
    with pytest.raises(ValueError, match="unknown VDAM gate"):
        vdam_gate("fsc", 1000)


# --- The M-step's gate against a NumPy evaluation of the formula of the note, section 4.1 step 6 and section 7 ---


def _m_step_case(size=16, radius=6):
    """A first-iteration M-step (zero moments), with random accumulators of both halves."""
    rng = np.random.default_rng(29)
    slab = 2 * (radius + 1) + 1
    shape = (slab, slab, slab // 2 + 1)
    moment_shape = (size, size, size // 2 + 1)
    return dict(
        reference_relion=rng.normal(size=(size,) * 3),
        data_h0=rng.normal(size=shape) + 1j * rng.normal(size=shape),
        weight_h0=rng.uniform(0.0, 3.0, shape),
        data_h1=rng.normal(size=shape) + 1j * rng.normal(size=shape),
        weight_h1=rng.uniform(0.0, 3.0, shape),
        mom1_h0=np.zeros(moment_shape, np.complex128),
        mom1_h1=np.zeros(moment_shape, np.complex128),
        mom2=rng.uniform(0.1, 2.0, moment_shape) + 0j,
        fsc_ssnr=np.zeros(size // 2 + 1),
        fsc_reconstruct=rng.uniform(0.0, 0.3, size // 2 + 1),
        tau2=rng.uniform(0.1, 2.0, size // 2 + 1),
        grad_stepsize=0.3,
        tau2_fudge=4.0,
        box_size=size,
        r_max=radius,
        min_resol_shell=3.0,
        stable_window_quantum=DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM,
    )


def _shell_means(values, size, radius):
    """Mean of ``values`` (a centered half volume) over the voxels of each shell inside the Fourier sphere."""
    n = values.shape[0]
    coord = np.arange(n) - n // 2
    r2 = coord[:, None, None] ** 2 + coord[None, :, None] ** 2 + np.arange(n // 2 + 1)[None, None, :] ** 2
    inside = r2 < radius**2
    shell = np.floor(np.sqrt(r2) + 0.5).astype(int)
    counts = np.bincount(shell[inside], minlength=size // 2 + 1)
    sums = np.bincount(shell[inside], weights=values[inside], minlength=size // 2 + 1)
    return sums / np.maximum(counts, 1), counts


def _expected_gate(case, projector, first_h0, first_h1, scale):
    """The gate of the note: rho = 2 fudge signal / disagreement, gate = max(rho / (1 + rho), fsc floor) <= 1."""
    size, radius = case["box_size"], case["r_max"]
    difference = np.abs(first_h0 - first_h1)
    if scale is None:
        disagreement, counts = _shell_means(difference, size, radius)
        signal, _ = _shell_means(np.abs(projector), size, radius)
    else:
        disagreement, counts = _shell_means(difference**2, size, radius)
        disagreement = disagreement * scale
        signal, _ = _shell_means(np.abs(projector) ** 2, size, radius)
    rho = np.where(disagreement > 0, 2 * case["tau2_fudge"] * signal / np.where(disagreement > 0, disagreement, 1), 0)
    gate = rho / (1 + rho)
    gate = np.minimum(np.where(gate >= case["fsc_reconstruct"], gate, case["fsc_reconstruct"]), 1.0)
    return np.where(counts > 0, gate, 0.0), disagreement, counts


@pytest.mark.parametrize("scale", [None, 0.0625, 0.5], ids=["relion amplitude", "power, scale 1/16", "power, scale 1/2"])
def test_the_m_step_gate_is_the_ratio_of_shell_means_of_the_note(scale, monkeypatch):
    """RELION's amplitude gate (no scale) and the power gate with a scale, from the returned moments and projector.

    The transaction runs with its intermediates through the host adapter; the expectation is NumPy's, on the moments
    and the reference projector it returns. float64, so rtol is 1e-9.
    """
    case = _m_step_case()
    seen = {}
    device = relion_mstep.relion_vdam_m_step_device

    def with_intermediates(*args, **options):
        seen.update(device(*args, **{**options, "return_intermediates": True}))
        return seen

    monkeypatch.setattr(relion_mstep, "relion_vdam_m_step_device", with_intermediates)
    relion_vdam_m_step_host(**case, gate=None if scale is None else MStepGate(scale, cross_power=False))

    size, radius = case["box_size"], case["r_max"]
    expected, disagreement, counts = _expected_gate(
        case, np.asarray(seen["projector"]), np.asarray(seen["mom1_h0"]), np.asarray(seen["mom1_h1"]), scale
    )
    assert counts[1] > 0 and np.all(disagreement[counts > 0] > 0)
    assert_matches(np.asarray(seen["fsc_estimate"]), expected, rtol=1e-9)
    assert_matches(np.asarray(seen["mom1_noise_power"])[counts > 0], disagreement[counts > 0], rtol=1e-9)
    # a gate strictly between the floor and 1 somewhere, so the formula and not a clamp is what is compared
    free = (expected > case["fsc_reconstruct"]) & (expected < 1)
    assert free[1 : radius - 1].any()


def test_the_scale_multiplies_the_disagreement_power_and_only_it():
    """Doubling the scale doubles ``mom1_noise_power`` and leaves the moments and the second moment as they were."""
    case = _m_step_case()
    once = relion_vdam_m_step_host(**case, gate=MStepGate(0.5, cross_power=False))
    twice = relion_vdam_m_step_host(**case, gate=MStepGate(1.0, cross_power=False))
    inside = once["mom1_noise_power"] > 0
    assert_matches(twice["mom1_noise_power"][inside], 2 * once["mom1_noise_power"][inside], rtol=1e-12)
    assert_matches(twice["mom1_h0"], once["mom1_h0"], strict=True)
    assert_matches(twice["mom2"], once["mom2"], strict=True)


def test_the_float32_transaction_takes_the_scale_in_float32():
    case = _m_step_case()
    reference = relion_vdam_m_step_host(**case, gate=MStepGate(0.25, cross_power=False))
    single = relion_vdam_m_step_host(**case, gate=MStepGate(0.25, cross_power=False), compute_dtype=np.float32)
    assert single["mom1_noise_power"].dtype == np.float32
    assert_matches(single["mom1_noise_power"], reference["mom1_noise_power"], rtol=1e-4)


def _expected_cross_gate(case, projector, first_h0, first_h1, second, scale):
    """The cross-power gate of the note (section 10): W = P / (P + V) applied as W r / (1 - W + W r), floored, <= 1."""
    size, radius = case["box_size"], case["r_max"]
    capacity = projector.shape[0]
    half0 = projector + np.asarray(pad_moment(first_h0, capacity))
    half1 = projector + np.asarray(pad_moment(first_h1, capacity))
    cross, counts = _shell_means((half0 * np.conj(half1)).real, size, radius)
    disagreement, _ = _shell_means(np.abs(first_h0 - first_h1) ** 2, size, radius)
    noise = scale * disagreement / 4
    snr = np.where(noise > 0, np.maximum(cross, 0) / np.where(noise > 0, noise, 1), 0)
    wiener = snr / (1 + snr)
    root_u, _ = _shell_means(np.sqrt(second.real), size, radius)
    gate = wiener * root_u / (1 - wiener + wiener * root_u)
    gate = np.minimum(np.where(gate >= case["fsc_reconstruct"], gate, case["fsc_reconstruct"]), 1.0)
    return np.where(counts > 0, gate, 0.0), np.where(counts > 0, wiener, 0.0)


def _cross_case():
    """The M-step case with a second moment near 3, so that the sqrt(u) conversion moves the gate."""
    rng = np.random.default_rng(41)
    case = _m_step_case()
    return dict(case, mom2=rng.uniform(2.0, 4.0, case["mom2"].shape) + 0j)


def test_the_cross_power_gate_is_the_wiener_weight_of_the_halves_converted_for_the_second_moment(monkeypatch):
    """P from the shell mean of Re (v + m0)(v + m1)^*, V = scale <|m0 - m1|^2> / 4, T = 1 (the case's fudge is 4 and
    does not enter), applied as W r / (1 - W + W r) with r the shell mean of sqrt(u). From the returned moments,
    second moment and projector; float64, so rtol is 1e-9.
    """
    case = _cross_case()
    seen = {}
    device = relion_mstep.relion_vdam_m_step_device

    def with_intermediates(*args, **options):
        seen.update(device(*args, **{**options, "return_intermediates": True}))
        return seen

    monkeypatch.setattr(relion_mstep, "relion_vdam_m_step_device", with_intermediates)
    relion_vdam_m_step_host(**case, gate=MStepGate(0.5, cross_power=True))
    expected, wiener = _expected_cross_gate(
        case, *(np.asarray(seen[name]) for name in ("projector", "mom1_h0", "mom1_h1", "mom2")), 0.5
    )
    assert_matches(np.asarray(seen["fsc_estimate"]), expected, rtol=1e-9)
    free = (expected > case["fsc_reconstruct"]) & (expected < 1)
    assert free[1 : case["r_max"] - 1].any()
    assert np.max(np.abs(expected - wiener)[free]) > 0.05  # the conversion, not W itself, is what is applied


def test_the_float32_cross_power_gate_follows_float64():
    case, gate = _cross_case(), MStepGate(0.5, cross_power=True)
    reference = relion_vdam_m_step_host(**case, gate=gate)
    single = relion_vdam_m_step_host(**case, gate=gate, compute_dtype=np.float32)
    assert_matches(np.asarray(single["iref"]), np.asarray(reference["iref"]))


def test_the_device_refuses_the_cross_power_gate_without_a_scale():
    with pytest.raises(ValueError, match="cross-power gate needs the scale"):
        relion_mstep.relion_vdam_m_step_device(*([None] * 15), box_size=16, cross_power=True)


# --- Through the iteration loop ---


def _stub_estep(size):
    """Random accumulators of each half, the same data every iteration but for the seed."""
    counter = {"n": 0}

    def estep(state, particle_ids, halfset_ids):
        counter["n"] += 1
        rng = np.random.default_rng(counter["n"])
        shape = (size, size, size // 2 + 1)
        accumulators = [
            VdamAccumulator(
                data=rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
                weight=rng.uniform(10.0, 100.0, shape),
                class_idx=0,
                halfset_idx=half,
            )
            for half in range(2 if state.pseudo_halfsets else 1)
        ]
        meta = {
            "max_posterior_per_image": np.full(len(particle_ids), 0.2, dtype=np.float32),
            "class_posterior_sums": np.full(1, float(len(particle_ids))),
        }
        return accumulators, meta

    return estep


def _run(gate, *, nr_particles=200, sizes=(50, 100), nr_iter=4, grad_em_iters=0):
    """Four gradient iterations of K=1 with real float64 M-steps; the meta of every iteration is returned too."""
    size = 16
    state = initialise_denovo_state(
        box_size=size, pixel_size=1.0, K=1, nr_iter=nr_iter, n_directions=12, pseudo_halfsets=True
    )
    state.Iref = np.ones((1, size, size, size), dtype=np.float64) * 0.1
    metas = []
    final = run_vdam_iterations(
        state,
        update=VdamUpdate(
            padding_factor=1,
            mstep_compute_dtype="float64",
            probe=NoProbe(),
            stable_window_quantum=DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM,
            gate=gate,
        ),
        nr_particles=nr_particles,
        optics_group_by_particle=[0] * nr_particles,
        grad_ini_subset_size=sizes[0],
        grad_fin_subset_size=sizes[1],
        tau2_fudge_arg=4.0,
        grad_em_iters=grad_em_iters,
        random_seed=1,
        expectation_step=_stub_estep(size),
        iter_artifact_sink=lambda current, iteration, meta: metas.append(dict(meta)),
        projector_refresh_fn=projector_power_refresh,
        grad_ini_frac=0.3,
        grad_fin_frac=0.2,
        mu=DEFAULT_GRAD_MU,
        uniform_class_direction_prior=False,
        environment=VdamEnvironment(),
        observer=VdamObserver(),
    )
    return final, metas


def test_the_loop_scales_each_iterations_disagreement_by_the_weight_of_the_batches_so_far():
    """Batches growing 50, 67, 83, 100 of 200 particles (the schedule): each logged scale is 4 / (N weight)."""
    final, metas = _run(FullDataPowerGate(n_particles=200), nr_iter=4)
    counts = [meta["subset_size"] for meta in metas]
    assert counts == [50, 67, 83, 100]
    weight = None
    for count, meta in zip(counts, metas):
        weight = advance_noise_weight(weight, count)
        report = meta["vdam_gate"]
        assert report["gate"] == "full_data_power"
        assert_matches(report["noise_weight"], weight, rtol=1e-12)
        assert_matches(report["disagreement_scale"], full_data_disagreement_scale(weight, 200), rtol=1e-12)
    assert final.gate_noise_weight == weight


def test_the_loop_scales_the_overlap_gate_by_the_coefficients_of_the_drawn_batches():
    """The scale each iteration logs is the one of the batches the loop drew (with the particles they share)."""
    batches = []
    original = FullDataOverlapPowerGate.advance

    def recording(self, state):
        batches.append((np.array(state.subset_particle_ids), np.array(state.subset_halfset_ids)))
        return original(self, state)

    FullDataOverlapPowerGate.advance = recording
    try:
        final, metas = _run(FullDataOverlapPowerGate(n_particles=200), nr_iter=4)
    finally:
        FullDataOverlapPowerGate.advance = original
    coefficients = None
    for (ids, halves), meta in zip(batches, metas):
        coefficients = advance_noise_coefficients(coefficients, ids, halves, 200)
        assert_matches(meta["vdam_gate"]["disagreement_scale"], overlap_disagreement_scale(coefficients, 200), rtol=1e-12)
    assert_matches(final.gate_noise_coefficients, coefficients, strict=True)
    assert len(set(batches[2][0]) & set(batches[3][0])) > 0  # the batches repeat particles


def test_the_two_gates_give_different_references_and_relion_reports_no_scale():
    relion, relion_metas = _run(RelionGate())
    full_data, _ = _run(FullDataPowerGate(n_particles=200))
    assert all(meta["vdam_gate"]["disagreement_scale"] is None for meta in relion_metas)
    assert relion.gate_noise_weight is None
    assert np.all(np.isfinite(np.asarray(full_data.Iref)))
    assert not np.allclose(np.asarray(relion.Iref), np.asarray(full_data.Iref), rtol=1e-3)


def test_the_cross_window_gate_runs_through_the_loop_with_the_window_scale():
    cross, metas = _run(CrossWindowGate(n_particles=200))
    window, window_metas = _run(WindowPowerGate(n_particles=200))
    relion, _ = _run(RelionGate())
    assert [meta["vdam_gate"]["gate"] for meta in metas] == ["cross_window"] * 4
    scales = [meta["vdam_gate"]["disagreement_scale"] for meta in metas]
    assert_matches(scales, [meta["vdam_gate"]["disagreement_scale"] for meta in window_metas], rtol=1e-12)
    assert np.all(np.isfinite(np.asarray(cross.Iref)))
    for other in (window, relion):
        assert not np.allclose(np.asarray(cross.Iref), np.asarray(other.Iref), rtol=1e-3)


def test_the_em_tail_runs_without_a_disagreement_and_keeps_the_weight():
    """The last two iterations are RELION's EM tail (no pseudo-halfsets): scale 1 and the weight stays."""
    _, metas = _run(FullDataPowerGate(n_particles=200), nr_iter=6, grad_em_iters=2)
    gradient, tail = metas[:4], metas[4:]
    assert [meta["vdam_gate"]["disagreement_scale"] == 1.0 for meta in tail] == [True, True]
    assert tail[0]["vdam_gate"]["noise_weight"] == gradient[-1]["vdam_gate"]["noise_weight"]
    assert tail[1]["vdam_gate"]["noise_weight"] == gradient[-1]["vdam_gate"]["noise_weight"]


# --- The option ---

_ARGS = ["--i", "unused.star", "--nr-iter", "3", "--K", "1"]


def _options(*extra, **changes):
    args = make_parser().parse_args(_ARGS + list(extra))
    return replace(NativeInitialModelOptions(**_native_options_dict(args)), **changes)


_MOMENTUM_SGD = ["--optimizer", "momentum_sgd", "--oversampling", "0", "--fourier-radius-schedule", "3x1,4x2"] + [
    "--fixed-healpix-order",
    "1",
]


def test_the_command_defaults_to_relion_s_gate_and_takes_the_full_data_power_gate():
    assert _options().vdam_gate == "relion"
    options = _options("--vdam-gate", "full_data_power")
    assert options.vdam_gate == "full_data_power"
    options.validate_run()
    _options("--vdam-gate", "cross_window").validate_run()
    with pytest.raises(SystemExit):
        make_parser().parse_args(_ARGS + ["--vdam-gate", "fsc"])


@pytest.mark.parametrize("gate", ["power", "full_data_power", "full_data_power_overlap", "cross_window"])
@pytest.mark.parametrize(
    "extra, changes",
    [
        ([], dict(nr_classes=2)),
        (_MOMENTUM_SGD, {}),
        ([], dict(diagnostic_continue_optimiser="optimiser.star", diagnostic_stop_after_iteration=2)),
    ],
    ids=["K=2", "momentum SGD", "continuation of a RELION optimiser"],
)
def test_the_power_gates_are_refused_where_they_are_not_defined(gate, extra, changes):
    _options(*extra).validate_run()  # the same run with RELION's gate is valid
    with pytest.raises(ValueError, match=f"--vdam-gate {gate} "):
        _options("--vdam-gate", gate, *extra, **changes).validate_run()
