"""Independent small-box checks for the opt-in coarse InitialModel SGD path."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.vdam import initial_model_state_stub, keep_tau2
from recovar.data_io.image_backends import _apply_relion_soft_image_mask_numpy, _centered_rfft2_numpy
from recovar.reconstruction.noise import make_radial_noise

from relax import sampling
from relax.commands.initial_model import _native_options_dict, make_parser
from relax.helpers.half_spectrum import _host_half_spectrum_plan
from relax.ppca_initial_model.vdam_controls import VdamPilotControls
from relax.relion.relion_projector_setup import setup_relion_projector, swap_relion_volume_layout
from relax.vdam.bootstrap_iref import initialise_denovo_state
from relax.vdam.estep_common import estep_sums
from relax.vdam.estep_setup import noise_variance_from_sigma2
from relax.vdam.iteration_loop import MomentumSgdUpdate, VdamUpdate, run_vdam_iterations
from relax.vdam.model_update import update_probabilities_from_estep
from relax.vdam.native_options import NativeInitialModelOptions, VdamEnvironment
from relax.vdam.native_sampling import build_sampling_plan, initial_sampling_state
from relax.vdam.ports import NoProbe, VdamObserver
from relax.vdam.schedules import DEFAULT_GRAD_MU
from relax.vdam.sgd import (
    GAMMA,
    INFLATED_PRIOR_COUNT,
    WHITE_PRIOR_COUNT,
    _bandlimit_real_map,
    _class_step,
    corner_white_sigma2,
    initialize_sgd_noise,
    sgd_m_step,
    update_sgd_noise,
)
from relax.vdam.state import VdamAccumulator

pytestmark = pytest.mark.unit


def test_scalar_curvature_step_is_corrected_forward_adjoint_on_active_band():
    """Check the actual corrected RELION forward setup, including inverse scale."""
    n, radius = 16, 6
    capacity = n + 3
    rng = np.random.default_rng(4)
    map_relion = _bandlimit_real_map(
        jnp.asarray(rng.normal(size=(n,) * 3), jnp.float32),
        radius,
        box_size=n,
        padding_factor=1,
    )
    source = jnp.asarray(rng.normal(size=(n,) * 3), jnp.float32)
    residual, _ = setup_relion_projector(
        source,
        jnp.int32(radius),
        box_size=n,
        padding_factor=1,
        compute_dtype=jnp.float32,
    )
    yz = jnp.arange(capacity) - capacity // 2
    x = jnp.arange(capacity // 2 + 1)
    active = yz[:, None, None] ** 2 + yz[None, :, None] ** 2 + x[None, None, :] ** 2 < radius**2
    residual = jnp.where(active, residual, 0)
    hermitian_weight = jnp.where((x == 0) | (x == n // 2), 1.0, 2.0)

    def forward_linear_form(volume):
        projected, _ = setup_relion_projector(
            volume,
            jnp.int32(radius),
            box_size=n,
            padding_factor=1,
            compute_dtype=jnp.float32,
        )
        return jnp.real(jnp.sum(hermitian_weight[None, None, :] * jnp.conj(residual) * projected * active))

    exact_gradient = jax.grad(forward_linear_form)(map_relion)
    expected = _bandlimit_real_map(exact_gradient, radius, box_size=n, padding_factor=1) * n
    reference = swap_relion_volume_layout(map_relion, jnp.float32)
    _, update, maximum, _, _ = _class_step(
        reference,
        jnp.zeros_like(reference),
        residual,
        active.astype(jnp.float32),
        jnp.int32(radius),
        jnp.float32(10.0),
        box_size=n,
        padding_factor=1,
    )
    actual = swap_relion_volume_layout(update, jnp.float32)
    relative_l2 = np.linalg.norm(np.asarray(actual - expected)) / np.linalg.norm(np.asarray(expected))
    assert float(maximum) == 1.0
    assert relative_l2 < 3e-6

    # The native inverse uses N times the Euclidean map-gradient scale.
    direction = _bandlimit_real_map(
        jnp.asarray(rng.normal(size=(n,) * 3), jnp.float32),
        radius,
        box_size=n,
        padding_factor=1,
    )
    epsilon = 0.01
    finite_difference = (
        forward_linear_form(map_relion + epsilon * direction) - forward_linear_form(map_relion - epsilon * direction)
    ) / (2 * epsilon)
    predicted = jnp.vdot(actual, direction) / n
    assert np.isclose(float(predicted), float(finite_difference), rtol=2e-4, atol=1e-3)


def test_scalar_step_is_duplicate_batch_invariant_and_empty_class_stable():
    n, radius, capacity = 16, 5, 19
    rng = np.random.default_rng(8)
    reference = jnp.asarray(rng.normal(size=(n,) * 3), jnp.float32)
    previous = jnp.asarray(rng.normal(size=(n,) * 3) * 0.1, jnp.float32)
    residual = jnp.asarray(rng.normal(size=(capacity, capacity, capacity // 2 + 1)), jnp.complex64)
    curvature = jnp.ones(residual.shape, jnp.float32) * 3

    def call(data, weight):
        return _class_step(
            reference,
            previous,
            data,
            weight,
            jnp.int32(radius),
            jnp.float32(0.4),
            box_size=n,
            padding_factor=1,
        )

    single = call(residual, curvature)
    duplicated = call(2 * residual, 2 * curvature)
    np.testing.assert_allclose(np.asarray(single[0]), np.asarray(duplicated[0]), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(single[1]), np.asarray(duplicated[1]), rtol=0, atol=0)
    empty = call(jnp.zeros_like(residual), jnp.zeros_like(curvature))
    np.testing.assert_array_equal(np.asarray(empty[0]), np.asarray(reference))
    np.testing.assert_array_equal(np.asarray(empty[1]), np.zeros_like(np.asarray(reference)))


def test_known_target_residual_step_reduces_projected_quadratic():
    n, radius = 16, 5
    axis = np.arange(n) - n // 2
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    target = jnp.asarray(np.exp(-(xx * xx + yy * yy + zz * zz) / 10), jnp.float32)
    target_fourier, _ = setup_relion_projector(
        target,
        jnp.int32(radius),
        box_size=n,
        padding_factor=1,
        compute_dtype=jnp.float32,
    )
    capacity = n + 3
    yz = jnp.arange(capacity) - capacity // 2
    x = jnp.arange(capacity // 2 + 1)
    active = yz[:, None, None] ** 2 + yz[None, :, None] ** 2 + x[None, None, :] ** 2 < radius**2
    residual = jnp.where(active, target_fourier, 0)
    zero = jnp.zeros((n,) * 3, jnp.float32)
    updated, _, _, _, _ = _class_step(
        zero,
        zero,
        residual,
        active.astype(jnp.float32),
        jnp.int32(radius),
        jnp.float32(0.05),
        box_size=n,
        padding_factor=1,
    )
    updated_fourier, _ = setup_relion_projector(
        swap_relion_volume_layout(updated, jnp.float32),
        jnp.int32(radius),
        box_size=n,
        padding_factor=1,
        compute_dtype=jnp.float32,
    )
    before = float(jnp.sum(jnp.abs(residual) ** 2))
    after = float(jnp.sum(jnp.abs(jnp.where(active, target_fourier - updated_fourier, 0)) ** 2))
    assert after < before


@pytest.mark.parametrize("k", [1, 4])
def test_class_pooling_uses_both_halves_without_crossing_classes(k):
    n, capacity = 16, 19
    state = initial_model_state_stub(
        K=k,
        box_size=n,
        current_size=10,
        pseudo_halfsets=True,
        Iref=jnp.zeros((k, n, n, n), jnp.float32),
        Igrad1=jnp.zeros((2 * k, n, n, n // 2 + 1), jnp.complex64),
        Igrad2=jnp.zeros((k, n, n, n // 2 + 1), jnp.complex64),
    )
    shape = (capacity, capacity, capacity // 2 + 1)
    accumulators = [
        VdamAccumulator(
            np.full(shape, 1 + class_idx, np.complex64),
            np.full(shape, 2, np.float32),
            class_idx,
            half,
        )
        for half in range(2)
        for class_idx in range(k)
    ]
    meta = {}
    updated = sgd_m_step(state, accumulators, learning_rate=0.4, padding_factor=1, meta=meta)
    assert np.asarray(updated.Iref).dtype == np.float32
    assert np.asarray(updated.sgd_previous_update).dtype == np.float32
    assert meta["sgd_max_curvature_by_class"] == [4.0] * k
    if k > 1:
        assert meta["sgd_step_norm_by_class"][1] > meta["sgd_step_norm_by_class"][0]
    empty = list(accumulators)
    empty_class = k - 1
    for half in range(2):
        index = half * k + empty_class
        empty[index] = VdamAccumulator(np.zeros(shape, np.complex64), np.zeros(shape, np.float32), empty_class, half)
    empty_updated = sgd_m_step(state, empty, learning_rate=0.4, padding_factor=1, meta={})
    np.testing.assert_array_equal(np.asarray(empty_updated.Iref[empty_class]), np.zeros((n,) * 3))


def test_corner_prior_matches_actual_masked_fft_covariance():
    n = 16
    yy, xx = np.indices((n, n))
    radial = np.sqrt((yy - n // 2) ** 2 + (xx - n // 2) ** 2)
    image_mask = np.clip((7 - radial) / 2, 0, 1)
    rng = np.random.default_rng(22)
    images = rng.normal(size=(4096, n, n)).astype(np.float32)
    measured = corner_white_sigma2(images, image_mask)
    corners = images[:, image_mask <= 1e-6].astype(np.float64)
    pixel_variance = np.mean((corners - corners.mean(axis=1, keepdims=True)) ** 2)
    # Sum the production mask/FFT response to each independent pixel impulse.
    impulses = np.eye(n * n, dtype=np.float32).reshape(n * n, n, n)
    processed = _centered_rfft2_numpy(_apply_relion_soft_image_mask_numpy(impulses, image_mask))
    expected_power = np.sum(np.abs(processed.astype(np.complex128)) ** 2, axis=0) * pixel_variance / (2 * n**4)
    shells = _host_half_spectrum_plan((n, n)).relion_noise_shell_indices.reshape(n, n // 2 + 1)
    expected = np.array([expected_power[shells == shell].mean() for shell in range(n // 2 + 1)])
    np.testing.assert_allclose(measured, expected, rtol=2e-7, atol=1e-10)
    np.testing.assert_allclose(corner_white_sigma2(images, image_mask, image_multiplier=2), 4 * measured)


def test_noise_state_is_relion_units_and_adapter_restores_engine_units():
    n = 16
    state = initial_model_state_stub(
        box_size=n,
        current_size=8,
        iter=1,
        Iref=np.zeros((1, n, n, n), np.float32),
        Igrad1=np.zeros((2, n, n, n // 2 + 1), np.complex64),
        Igrad2=np.zeros((1, n, n, n // 2 + 1), np.complex64),
        sigma2_noise=np.zeros((1, n // 2 + 1)),
    )
    baseline = np.full(n // 2 + 1, 0.002, dtype=np.float64)
    state = initialize_sgd_noise(state, baseline)
    initial_prior = (WHITE_PRIOR_COUNT + 8 * INFLATED_PRIOR_COUNT) / (WHITE_PRIOR_COUNT + INFLATED_PRIOR_COUNT)
    np.testing.assert_allclose(state.sigma2_noise[0], baseline * initial_prior)
    shells = _host_half_spectrum_plan((n, n)).relion_noise_shell_indices.reshape(-1)
    shell_counts = np.bincount(shells[shells <= n // 2], minlength=n // 2 + 1)
    batch_variance = 0.005
    mass = 20.0
    meta = {
        "wsum_sigma2_noise": np.zeros(n // 2 + 1),
        "wsum_img_power": 2 * mass * shell_counts * batch_variance * n**4,
        "noise_sumw": mass,
    }
    updated = update_sgd_noise(state, meta)
    active = np.arange(n // 2 + 1) <= state.current_size // 2
    inflated = INFLATED_PRIOR_COUNT * GAMMA
    expected = (mass * batch_variance * active + (WHITE_PRIOR_COUNT + 8 * inflated) * baseline) / (
        mass * active + WHITE_PRIOR_COUNT + inflated
    )
    np.testing.assert_allclose(updated.sigma2_noise[0], expected, rtol=2e-6)
    engine_radial = noise_variance_from_sigma2(updated.sigma2_noise, n)
    expected_radial = np.asarray(make_radial_noise(expected * n**4, (n, n))).reshape(-1)
    np.testing.assert_allclose(engine_radial, expected_radial, rtol=2e-6)
    np.testing.assert_allclose(updated.sgd_noise_count, mass * active)
    np.testing.assert_allclose(updated.sgd_noise_sum, mass * batch_variance * active, rtol=2e-6)


def test_cli_and_both_optimizers_keep_same_coarse_grid_and_terminal_subset(monkeypatch):
    common = [
        "--i",
        "unused.star",
        "--nr-iter",
        "3",
        "--K",
        "3",
        "--healpix-order",
        "1",
        "--fixed-healpix-order",
        "1",
        "--oversampling",
        "0",
        "--perturbation-factor",
        "0",
        "--fourier-radius-schedule",
        "3x1,4x2",
        "--stochastic-batch-size",
        "5",
        "--stochastic-all-iterations",
        "--grad-em-iters",
        "0",
        "--uniform-class-direction-prior",
    ]
    parser = make_parser()
    assert parser._option_string_actions["--optimizer"].choices == ("vdam", "momentum_sgd")
    opts = {}
    for optimizer in ("vdam", "momentum_sgd"):
        args = parser.parse_args(common + ["--optimizer", optimizer])
        opts[optimizer] = NativeInitialModelOptions(**_native_options_dict(args))
        opts[optimizer].validate_run()
    # The native RELION orientation binding is built separately for production.
    # This stub lets the plan test assert it requests the same parent grid.
    requested_orders = []

    def coarse_eulers(order, *, rotation_index_order):
        requested_orders.append((order, rotation_index_order))
        return np.zeros((12, 3), dtype=np.float64)

    monkeypatch.setattr(sampling, "_get_relion_rotation_grid_eulers_float64", coarse_eulers)
    grids = [
        build_sampling_plan(
            opts[name], sampling_state=initial_sampling_state(opts[name], pixel_size=1.0), iteration=iteration
        )
        for name in opts
        for iteration in (1, 3)
    ]
    # RELION's native hidden-variable order (its source angles; the matrices are their host inverses).
    assert requested_orders == [(1, "relion")] * 4
    for grid in grids:
        assert grid.oversampling == 0 and grid.healpix_order == 1
        assert grid.translation_parent is None
        np.testing.assert_array_equal(grid.rotations, grids[0].rotations)
        np.testing.assert_array_equal(grid.translations, grids[0].translations)

    from relax.vdam import iteration_loop
    from relax.vdam import sgd as sgd_noise
    from relax.vdam import sgd as sgd_optimizer

    monkeypatch.setattr(iteration_loop, "vdam_m_step", lambda state, **kwargs: state)
    monkeypatch.setattr(iteration_loop, "update_noise_from_estep", lambda state, sums, **kwargs: state)
    monkeypatch.setattr(iteration_loop, "update_current_resolution_from_data_vs_prior", lambda state: state)
    monkeypatch.setattr(sgd_optimizer, "sgd_m_step", lambda state, *args, **kwargs: state)
    monkeypatch.setattr(sgd_noise, "update_sgd_noise", lambda state, meta: state)

    def initial_state():
        return initial_model_state_stub(
            nr_iter=3,
            K=3,
            box_size=16,
            pixel_size=1,
            current_size=6,
            Iref=np.zeros((3, 16, 16, 16), np.float32),
            Igrad1=np.zeros((6, 16, 16, 9), np.complex64),
            Igrad2=np.zeros((3, 16, 16, 9), np.complex64),
            pdf_class=np.asarray([0.9, 0.09, 0.01]),
            pdf_direction=np.full((3, 4), 1.0 / 12.0) * np.asarray([2.0, 0.8, 0.2])[:, None],
        )

    histories = {}
    for optimizer in opts:
        history = []

        def fake_estep(state, particle_ids, halfset_ids):
            np.testing.assert_array_equal(state.pdf_class, np.full(3, 1.0 / 3.0))
            np.testing.assert_array_equal(state.pdf_direction, np.full((3, 4), 1.0 / 12.0))
            return [], {
                "class_posterior_sums": np.asarray([5.0, 0.0, 0.0]),
                "class_posterior_sums_full": np.asarray([3.0, 2.0, 0.0]),
                "class_direction_posterior_sums": np.full((3, 4), 5.0 / 12.0),
            }

        def sink(state, iteration, meta):
            history.append((iteration, state.subset_particle_ids.copy(), state.subset_halfset_ids.copy(), dict(meta)))

        run_vdam_iterations(
            initial_state(),
            nr_particles=20,
            optics_group_by_particle=np.zeros(20, dtype=np.int32),
            grad_ini_subset_size=5,
            grad_fin_subset_size=5,
            pilot_controls=VdamPilotControls(stochastic_batch_size=5),
            tau2_fudge_arg=1.0,
            grad_em_iters=0,
            random_seed=0,
            expectation_step=fake_estep,
            iter_artifact_sink=sink,
            post_mstep_update=lambda state, iteration, meta: replace(state, has_converged=True),
            projector_refresh_fn=keep_tau2,
            update=(
                VdamUpdate(padding_factor=1, mstep_compute_dtype="float32", probe=NoProbe())
                if optimizer == "vdam"
                else MomentumSgdUpdate(learning_rate=1.0, padding_factor=1)
            ),
            fourier_radius_schedule=(3, 4, 4),
            stochastic_all_iterations=True,
            uniform_class_direction_prior=True,
            grad_ini_frac=0.3,
            grad_fin_frac=0.2,
            mu=DEFAULT_GRAD_MU,
            environment=VdamEnvironment(),
            observer=VdamObserver(),
        )
        histories[optimizer] = history
    assert len(histories["vdam"]) == len(histories["momentum_sgd"]) == 3
    for control, candidate in zip(histories["vdam"], histories["momentum_sgd"]):
        for arm in (control, candidate):
            iteration, ids, halves, meta = arm
            assert len(ids) == len(halves) == meta["subset_size"] == 5
            assert meta["effective_estep_fourier_radius"] == (3, 4, 4)[iteration - 1]
            assert meta["subset_particle_ids_sha256"] and meta["subset_halfset_ids_sha256"]
            assert meta["class_posterior_fraction_by_class"] == [0.6, 0.4, 0.0]
            assert meta["class_posterior_fraction_source"] == "full"
            assert meta["class_retained_mass_fraction_by_class"] == [1.0, 0.0, 0.0]
            assert meta["uniform_class_direction_prior"] is True
            assert meta["effective_pdf_class_prior_by_class"] == [1.0 / 3.0] * 3
            assert meta["effective_joint_direction_prior_per_class_direction"] == 1.0 / 12.0
            assert meta["effective_joint_direction_count"] == 4
        assert control[3]["subset_particle_ids_sha256"] == candidate[3]["subset_particle_ids_sha256"]
        assert control[3]["subset_halfset_ids_sha256"] == candidate[3]["subset_halfset_ids_sha256"]


@pytest.mark.parametrize("k", [1, 4])
def test_uniform_joint_prior_is_fixed_while_offset_updates_and_default_still_learns(k):
    state = initialise_denovo_state(box_size=8, pixel_size=1.0, K=k, nr_iter=1, n_directions=3, pseudo_halfsets=True)
    state.subset_size = 50
    posterior = np.zeros(k, dtype=np.float64)
    posterior[-1] = 100.0
    direction_sums = np.zeros((k, 5), dtype=np.float64)
    direction_sums[-1, -1] = 100.0
    meta = {
        "class_posterior_sums": posterior,
        "class_direction_posterior_sums": direction_sums,
        "wsum_sigma2_offset": 500.0,
        "sigma2_offset_sumw": 80.0,
    }
    uniform = update_probabilities_from_estep(state, estep_sums(meta), do_grad=True, mu=0.9, uniform_class_direction_prior=True
    )
    learned = update_probabilities_from_estep(
        state, estep_sums(meta), do_grad=True, mu=0.9, uniform_class_direction_prior=False
    )
    np.testing.assert_array_equal(uniform.pdf_class, np.full(k, 1.0 / k))
    np.testing.assert_array_equal(uniform.pdf_direction, np.full((k, 5), 1.0 / (5 * k)))
    assert np.isclose(uniform.pdf_direction.sum(), 1.0)
    assert uniform.sigma2_offset == pytest.approx(90.3125)
    assert learned.sigma2_offset == pytest.approx(uniform.sigma2_offset)
    np.testing.assert_allclose(learned.pdf_class, state.pdf_class * 0.9 + 0.1 * posterior / posterior.sum())
    expected_direction = np.full((k, 5), 1.0 / (5 * k)) * 0.9 + 0.1 * direction_sums / 100.0
    np.testing.assert_allclose(learned.pdf_direction, expected_direction)
