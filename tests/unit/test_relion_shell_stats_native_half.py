"""RELION's shell statistics on the two public layouts of a RELION x-half accumulator.

``BackProjector::updateSSNRarrays`` and the ``reconstruct`` radial average loop over every
stored entry of the x-half accumulator ``(z, y, x >= 0)`` with weight 1, so Hermitian pairs on
the kx = 0 plane count twice and every other pair once. Normal grids reach the shell statistics
as a full volume expanded from that storage; grids of 200M voxels or more reach them repacked
to RECOVAR's native half ``(x, y, z >= 0)``. Both must give RELION's sums.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import recovar.core.fourier_transform_utils as fourier_transform_utils  # noqa: E402

from relax.helpers import half_volume_mstep  # noqa: E402
from relax.reconstruction import regularization_relion  # noqa: E402

pytestmark = pytest.mark.unit

# The axis of the public layout that is RELION's x for an x-half M-step accumulator; the
# production call sites pass it as ``full_half_axis`` for both public layouts
# (``mstep_full_half_axis=0`` in relax/refinement/half_scoring.py and iteration_loop.py).
_RELION_X_AXIS = 0

# (volume_shape, padding_factor, accumulator_volume_shape, r_max)
_GRIDS = {
    # RELION's odd BPref grid 2 * (pf * r_max + 1) + 1 at the full and at a reduced current size.
    "odd_bpref_27_full_size": ((12, 12, 12), 2, (27, 27, 27), 6),
    "odd_bpref_19_current_8": ((12, 12, 12), 2, (19, 19, 19), 4),
    "odd_bpref_27_current_8": ((12, 12, 12), 2, (27, 27, 27), 4),
    "odd_bpref_27_no_radius": ((12, 12, 12), 2, (27, 27, 27), None),
    # The even padded grid pf * N of a caller that passes no accumulator shape.
    "even_padded_24_full_size": ((12, 12, 12), 2, None, 6),
    "even_padded_24_current_8": ((12, 12, 12), 2, None, 4),
    "even_padded_24_no_radius": ((12, 12, 12), 2, None, None),
    # Padding 1: the statistics run on the native grid.
    "odd_native_13": ((13, 13, 13), 1, None, 5),
    "even_native_12": ((12, 12, 12), 1, None, 5),
    "even_native_12_no_radius": ((12, 12, 12), 1, None, None),
}


def _grid_shape(volume_shape, padding_factor, accumulator_volume_shape):
    if accumulator_volume_shape is not None:
        return tuple(accumulator_volume_shape)
    return tuple(int(s) * int(padding_factor) for s in volume_shape)


def _relion_x_half_coordinates(grid_shape):
    """Logical ``(kz, ky, kx)`` of every stored entry of the ``(z, y, xhalf)`` accumulator."""

    n = int(grid_shape[0])
    centered = np.arange(-(n // 2), n - n // 2, dtype=np.int64)
    packed = np.arange(0, n // 2 + 1, dtype=np.int64)
    return np.meshgrid(centered, centered, packed, indexing="ij")


def _anisotropic_relion_x_half_weight(grid_shape, seed):
    """A positive float32 x-half weight that changes fourfold along x and unevenly along y and z.

    The kx = 0 plane goes through the production C1 finalize, as every accumulator does before
    its layout conversion (RELION ``enforceHermitianSymmetry``). On an even grid the x Nyquist
    plane is its own Hermitian mate as well and is made symmetric here; RELION's BPref grid is
    odd and has no such plane.
    """

    n = int(grid_shape[0])
    kz, ky, kx = _relion_x_half_coordinates(grid_shape)
    half = max(n // 2, 1)
    rng = np.random.default_rng(seed)
    weight = (1.0 + 3.0 * kx / half) * (1.0 + 0.4 * np.abs(ky) / half) * (1.0 + 0.2 * (kz + ky) / half)
    weight = (weight * (0.75 + 0.5 * rng.random(weight.shape))).astype(np.float32)
    weight = half_volume_mstep.enforce_relion_half_volume_x0_hermitian_host(weight.reshape(-1), grid_shape)
    weight = np.array(weight, dtype=np.float32).reshape(kx.shape)
    if n % 2 == 0:
        partner = (n - np.arange(n)) % n
        nyquist = weight[:, :, -1]
        weight[:, :, -1] = 0.5 * (nyquist + nyquist[np.ix_(partner, partner)])
    return weight.reshape(-1)


def _relion_rule_shell_stats(x_half_weight, volume_shape, grid_shape, padding_factor, r_max, shell_rounding):
    """RELION's loop in plain NumPy: every stored x-half entry once, weight 1.

    ``updateSSNRarrays`` (backprojector.cpp:1066-1076) keeps ``r2 < ROUND(r_max * pf)^2`` and
    bins at ``ROUND(sqrt(r2) / pf)``; ``reconstruct`` (backprojector.cpp:1516-1535) bins at
    ``FLOOR``. Without a radius every entry counts and the shell is clamped to ``ori_size / 2``.
    """

    kz, ky, kx = _relion_x_half_coordinates(grid_shape)
    r2 = (kz * kz + ky * ky + kx * kx).reshape(-1)
    n_shells = int(volume_shape[0]) // 2 + 1
    scaled = np.sqrt(r2.astype(np.float64)) / float(padding_factor)
    shell = np.floor(scaled + 0.5) if shell_rounding == "round" else np.floor(scaled)
    shell = np.minimum(shell.astype(np.int64), n_shells - 1)
    if r_max is None:
        keep = np.ones(r2.shape, dtype=bool)
    else:
        max_r = int(np.floor(float(r_max) * float(padding_factor) + 0.5))
        keep = r2 < max_r * max_r
    values = np.asarray(x_half_weight, dtype=np.float64).reshape(-1)
    shell_sum = np.bincount(shell[keep], weights=values[keep], minlength=n_shells)
    shell_count = np.bincount(shell[keep], minlength=n_shells)
    return shell_sum, shell_count


def _full_route(x_half_weight, grid_shape):
    return np.asarray(half_volume_mstep.relion_x_half_volume_to_full(x_half_weight, grid_shape))


def _native_half_route(x_half_weight, grid_shape, monkeypatch):
    monkeypatch.setenv("RELAX_RELION_X_HALF_TO_NATIVE_HALF", "1")
    native_half = half_volume_mstep.relion_x_half_volume_to_public_layout(x_half_weight, grid_shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(grid_shape)
    assert native_half.shape == (int(np.prod(half_shape)),)
    return native_half


def _production_shell_stats(weight, grid, shell_rounding):
    volume_shape, padding_factor, accumulator_volume_shape, r_max = grid
    stats = regularization_relion.compute_relion_weight_shell_stats(
        weight,
        volume_shape,
        padding_factor=padding_factor,
        r_max=r_max,
        shell_rounding=shell_rounding,
        full_half_axis=_RELION_X_AXIS,
        accumulator_volume_shape=accumulator_volume_shape,
    )
    return {key: np.asarray(value) for key, value in stats.items()}


def _assert_relion_shell_stats(stats, expected_sum, expected_count):
    assert_matches(stats["shell_count"].astype(np.int64), expected_count)
    assert_matches(stats["shell_count"], expected_count.astype(np.float64))
    assert_matches(stats["shell_sum"], expected_sum)
    expected_average = np.where(expected_count > 0, expected_sum / np.maximum(expected_count, 1), 0.0)
    assert_matches(stats["avg_weight_shells"], expected_average)


@pytest.fixture(params=["device", "host"])
def reducer(request, monkeypatch):
    """Both reductions of the padded-grid statistics; the host one serves the largest grids."""

    if request.param == "host":
        monkeypatch.setattr(regularization_relion, "_RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS", 1)
    return request.param


@pytest.mark.parametrize("shell_rounding", ["round", "floor"])
@pytest.mark.parametrize("grid_name", sorted(_GRIDS))
def test_full_route_shell_stats_follow_relion_stored_half(grid_name, shell_rounding, reducer):
    grid = _GRIDS[grid_name]
    volume_shape, padding_factor, accumulator_volume_shape, r_max = grid
    grid_shape = _grid_shape(volume_shape, padding_factor, accumulator_volume_shape)
    x_half_weight = _anisotropic_relion_x_half_weight(grid_shape, seed=3)
    expected_sum, expected_count = _relion_rule_shell_stats(
        x_half_weight, volume_shape, grid_shape, padding_factor, r_max, shell_rounding
    )

    stats = _production_shell_stats(_full_route(x_half_weight, grid_shape), grid, shell_rounding)

    _assert_relion_shell_stats(stats, expected_sum, expected_count)


@pytest.mark.parametrize("shell_rounding", ["round", "floor"])
@pytest.mark.parametrize("grid_name", sorted(_GRIDS))
def test_native_half_route_shell_stats_follow_relion_stored_half(grid_name, shell_rounding, reducer, monkeypatch):
    grid = _GRIDS[grid_name]
    volume_shape, padding_factor, accumulator_volume_shape, r_max = grid
    grid_shape = _grid_shape(volume_shape, padding_factor, accumulator_volume_shape)
    x_half_weight = _anisotropic_relion_x_half_weight(grid_shape, seed=3)
    expected_sum, expected_count = _relion_rule_shell_stats(
        x_half_weight, volume_shape, grid_shape, padding_factor, r_max, shell_rounding
    )

    native_half = _native_half_route(x_half_weight, grid_shape, monkeypatch)
    stats = _production_shell_stats(native_half, grid, shell_rounding)

    _assert_relion_shell_stats(stats, expected_sum, expected_count)
    full_stats = _production_shell_stats(_full_route(x_half_weight, grid_shape), grid, shell_rounding)
    for key in ("shell_sum", "shell_count", "avg_weight_shells"):
        assert_matches(stats[key], full_stats[key])


@pytest.mark.parametrize("grid_name", ["odd_bpref_27_full_size", "odd_bpref_27_current_8", "even_padded_24_current_8"])
def test_native_half_route_tau2_and_data_vs_prior_match_full_route(grid_name, reducer, monkeypatch):
    """The two consumers of the statistics, called as relax/refinement/iteration_loop.py calls them."""

    volume_shape, padding_factor, accumulator_volume_shape, r_max = _GRIDS[grid_name]
    grid_shape = _grid_shape(volume_shape, padding_factor, accumulator_volume_shape)
    n_shells = volume_shape[0] // 2 + 1
    current_size = 2 * r_max
    fsc = np.linspace(0.95, 0.25, n_shells).astype(np.float32)
    tau2 = np.linspace(2.0, 0.5, n_shells).astype(np.float32)
    x_half_weight = _anisotropic_relion_x_half_weight(grid_shape, seed=5)

    def consumers(weight):
        prior, _, details = regularization_relion.compute_relion_tau2_from_weights(
            weight,
            weight,
            fsc,
            volume_shape,
            padding_factor=padding_factor,
            r_max=current_size // 2,
            return_details=True,
            full_half_axis=_RELION_X_AXIS,
            accumulator_volume_shape=accumulator_volume_shape,
        )
        data_vs_prior = regularization_relion.compute_data_vs_prior(
            weight,
            tau2,
            volume_shape,
            padding_factor=padding_factor,
            current_size=current_size,
            full_half_axis=_RELION_X_AXIS,
            accumulator_volume_shape=accumulator_volume_shape,
        )
        return np.asarray(details["prior_shells"]), np.asarray(details["sigma2_shells"]), np.asarray(data_vs_prior)

    full = consumers(_full_route(x_half_weight, grid_shape))
    native = consumers(_native_half_route(x_half_weight, grid_shape, monkeypatch))

    for native_values, full_values in zip(native, full):
        assert np.any(full_values > 0)
        assert_matches(native_values, full_values)


@pytest.mark.parametrize("grid_name", ["odd_bpref_27_full_size", "even_padded_24_full_size", "odd_native_13"])
def test_recovar_native_half_accumulator_counts_every_stored_entry(grid_name, reducer):
    """A half accumulator whose packed axis is RELION's stored axis (the default ``full_half_axis``)
    is RELION's storage itself: every entry once."""

    volume_shape, padding_factor, accumulator_volume_shape, r_max = _GRIDS[grid_name]
    grid_shape = _grid_shape(volume_shape, padding_factor, accumulator_volume_shape)
    x_half_weight = _anisotropic_relion_x_half_weight(grid_shape, seed=7)
    expected_sum, expected_count = _relion_rule_shell_stats(
        x_half_weight, volume_shape, grid_shape, padding_factor, r_max, "round"
    )

    stats = regularization_relion.compute_relion_weight_shell_stats(
        x_half_weight,
        volume_shape,
        padding_factor=padding_factor,
        r_max=r_max,
        shell_rounding="round",
        accumulator_volume_shape=accumulator_volume_shape,
    )

    _assert_relion_shell_stats({key: np.asarray(value) for key, value in stats.items()}, expected_sum, expected_count)
