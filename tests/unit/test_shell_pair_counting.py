"""The ``shell_pair_counting`` option: RELION's stored-half loops or every Hermitian pair once.

RELION's 3-D shell loops visit every stored entry of a half volume ``(z, y, x >= 0)`` with weight 1.
On the ``x = 0`` plane both members of each Hermitian pair are stored, so those pairs count twice
(``updateSSNRarrays``, ``calculateDownSampledFourierShellCorrelation``,
``Projector::computeFourierTransformMap``, ``getSpectrum``). ``"once"`` counts every pair once,
which makes each statistic the one of the whole Fourier grid, every voxel once. The oracles below
are plain numpy on the raw x-half storage or on the whole grid; they share no code with production.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.tiny_refinement import record_calls, run_tiny_refinement

from relax.helpers import half_volume_mstep
from relax.reconstruction import regularization_relion as rr
from relax.refinement import final_reconstruction, full_refinement, mean_helpers
from relax.refinement.refinement_options import RelionConsistencyOptions
from relax.relion import reference_initialization
from relax.relion import relion_projector_setup as setup

pytestmark = pytest.mark.unit

COUNTINGS = ["relion", "once"]
# The axis of both public layouts that is RELION's x for an x-half M-step accumulator.
RELION_X_AXIS = 0
# (volume_shape, padding_factor, accumulator_volume_shape, r_max): RELION's odd BPref grid
# 2 * (pf * r_max + 1) + 1 at the full and at a cropped current size, for an even and an odd box, the even
# padded grid of a caller that passes no accumulator shape, and the unpadded native grid.
GRIDS = {
    "odd_bpref_19_full_size": ((8, 8, 8), 2, (19, 19, 19), 4),
    "odd_bpref_19_current_4": ((8, 8, 8), 2, (19, 19, 19), 2),
    "odd_bpref_11_current_4": ((8, 8, 8), 2, (11, 11, 11), 2),
    "odd_bpref_15_box_6": ((6, 6, 6), 2, (15, 15, 15), 3),
    "even_padded_16_full_size": ((8, 8, 8), 2, None, 4),
    "even_padded_16_current_4": ((8, 8, 8), 2, None, 2),
    "odd_native_9": ((9, 9, 9), 1, None, 3),
    "even_native_8": ((8, 8, 8), 1, None, 3),
}


def _grid_size(grid):
    volume_shape, padding_factor, accumulator_volume_shape, _ = grid
    return int(accumulator_volume_shape[0]) if accumulator_volume_shape else int(volume_shape[0]) * padding_factor


def _mirror(size):
    """Index of the frequency ``-k`` on a centered axis (an even axis' Nyquist is its own mirror)."""
    return (2 * (size // 2) - np.arange(size)) % size


def _own_mate_columns(size):
    """x-half columns whose Hermitian mates are stored too: ``x = 0``, and an even axis' Nyquist."""
    return [0] if size % 2 else [0, size // 2]


def _x_half(size, seed, *, complex_values=False):
    """A random RELION x-half array ``a[z, y, x >= 0]`` whose own-mate planes are Hermitian."""
    rng = np.random.default_rng(seed)
    shape = (size, size, size // 2 + 1)
    kx = np.arange(size // 2 + 1)
    values = (1.0 + 3.0 * kx / max(size // 2, 1)) * (0.5 + rng.random(shape))  # anisotropic along x
    if complex_values:
        values = values * np.exp(2j * np.pi * rng.random(shape))
    mirror = _mirror(size)
    for column in _own_mate_columns(size):
        plane = values[:, :, column]
        values[:, :, column] = 0.5 * (plane + np.conj(plane[np.ix_(mirror, mirror)]))
    return values


def _whole_grid(x_half):
    """Every voxel of the centered Fourier grid once, expanded from the x-half by Hermitian symmetry."""
    size = x_half.shape[0]
    mirror = _mirror(size)
    full = np.empty((size, size, size), dtype=x_half.dtype)
    center = size // 2
    for index in range(size):
        k = index - center
        if k >= 0:
            full[:, :, index] = x_half[:, :, k]
        elif size % 2 == 0 and k == -center:
            full[:, :, index] = x_half[:, :, center]
        else:
            full[:, :, index] = np.conj(x_half[:, :, -k][np.ix_(mirror, mirror)])
    return full


def _coordinates(size, *, half):
    centered = np.arange(size) - size // 2
    return np.meshgrid(centered, centered, np.arange(size // 2 + 1) if half else centered, indexing="ij")


def _shells(coordinates, grid, shell_rounding):
    """updateSSNRarrays' rule: keep r2 < ROUND(r_max pf)^2, bin at ROUND (or FLOOR) of r / pf."""
    volume_shape, padding_factor, _, r_max = grid
    r2 = sum(c.astype(np.int64) ** 2 for c in coordinates)
    scaled = np.sqrt(r2.astype(np.float64)) / padding_factor
    shell = np.floor(scaled + 0.5) if shell_rounding == "round" else np.floor(scaled)
    shell = np.minimum(shell.astype(np.int64), volume_shape[0] // 2)
    keep = r2 < int(np.floor(r_max * padding_factor + 0.5)) ** 2
    return shell, keep


def _stored_half_stats(x_half, grid, shell_rounding, counting):
    """Shell sums over the stored x-half: every entry once (RELION), or 1/2 where the mate is stored too."""
    size = x_half.shape[0]
    shell, keep = _shells(_coordinates(size, half=True), grid, shell_rounding)
    pair_weight = np.ones(x_half.shape)
    if counting == "once":
        pair_weight[:, :, _own_mate_columns(size)] = 0.5
    n_shells = grid[0][0] // 2 + 1
    shell_sum = np.bincount(shell[keep], weights=(x_half.real * pair_weight)[keep], minlength=n_shells)
    shell_count = np.bincount(shell[keep], weights=pair_weight[keep], minlength=n_shells)
    return shell_sum, shell_count


def _whole_grid_stats(x_half, grid, shell_rounding):
    """Shell sums over the whole Fourier grid, every voxel once."""
    full = _whole_grid(x_half)
    shell, keep = _shells(_coordinates(full.shape[0], half=False), grid, shell_rounding)
    n_shells = grid[0][0] // 2 + 1
    return (
        np.bincount(shell[keep], weights=full.real[keep], minlength=n_shells),
        np.bincount(shell[keep], minlength=n_shells).astype(np.float64),
    )


def _public_layouts(x_half, monkeypatch):
    """The two layouts an x-half accumulator reaches the statistics in: full, and (large grids) native half."""
    size = x_half.shape[0]
    flat = np.ascontiguousarray(x_half).reshape(-1)
    full = np.asarray(half_volume_mstep.relion_x_half_volume_to_full(flat, (size,) * 3))
    with monkeypatch.context() as patch:
        patch.setenv("RELAX_RELION_X_HALF_TO_NATIVE_HALF", "1")
        native_half = np.asarray(half_volume_mstep.relion_x_half_volume_to_public_layout(flat, (size,) * 3))
    assert native_half.size == size * size * (size // 2 + 1) and full.size == size**3
    return {"full": full, "native_half": native_half}


@pytest.fixture(params=["device", "host"])
def reducer(request, monkeypatch):
    """Both reductions of the padded-grid statistics; the host one serves the largest grids."""
    if request.param == "host":
        monkeypatch.setattr(rr, "_RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS", 1)
    return request.param


# --- weight shell statistics: sigma2, tau2, data-vs-prior ----------------------------------------


@pytest.mark.parametrize("shell_rounding", ["round", "floor"])
@pytest.mark.parametrize("counting", COUNTINGS)
@pytest.mark.parametrize("grid_name", sorted(GRIDS))
def test_weight_shell_statistics_follow_the_counting_rule_on_both_layouts(
    grid_name, counting, shell_rounding, reducer, monkeypatch
):
    jax.config.update("jax_enable_x64", True)
    grid = GRIDS[grid_name]
    volume_shape, padding_factor, accumulator_volume_shape, r_max = grid
    x_half = _x_half(_grid_size(grid), seed=3)
    expected_sum, expected_count = _stored_half_stats(x_half, grid, shell_rounding, counting)

    for layout, weight in _public_layouts(x_half, monkeypatch).items():
        stats = rr._compute_relion_weight_shell_stats(
            weight,
            volume_shape,
            padding_factor=padding_factor,
            r_max=r_max,
            shell_rounding=shell_rounding,
            full_half_axis=RELION_X_AXIS,
            accumulator_volume_shape=accumulator_volume_shape,
            shell_pair_counting=counting,
        )
        assert_matches(np.asarray(stats["shell_count"]), expected_count, rtol=1e-12, err_msg=layout)
        assert_matches(np.asarray(stats["shell_sum"]), expected_sum, rtol=1e-12, err_msg=layout)
        average = np.where(expected_count > 0, expected_sum / np.maximum(expected_count, 0.5), 0.0)
        assert_matches(np.asarray(stats["avg_weight_shells"]), average, rtol=1e-12, err_msg=layout)


@pytest.mark.parametrize("shell_rounding", ["round", "floor"])
@pytest.mark.parametrize("grid_name", sorted(GRIDS))
def test_once_is_the_whole_grid_statistic_and_relion_doubles_the_zero_plane(grid_name, shell_rounding):
    """The weighting rule itself: half the whole-grid sums, origin and Nyquist planes included."""
    grid = GRIDS[grid_name]
    x_half = _x_half(_grid_size(grid), seed=5)
    whole_sum, whole_count = _whole_grid_stats(x_half, grid, shell_rounding)
    once_sum, once_count = _stored_half_stats(x_half, grid, shell_rounding, "once")
    relion_sum, relion_count = _stored_half_stats(x_half, grid, shell_rounding, "relion")
    assert_matches(once_sum, 0.5 * whole_sum, rtol=1e-12)
    assert_matches(once_count, 0.5 * whole_count, rtol=1e-12)
    # RELION counts the own-mate planes once more: its shell mean leans towards x = 0.
    assert np.all(relion_count >= once_count) and np.any(relion_count > once_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        relion_mean, once_mean = relion_sum / relion_count, once_sum / once_count
    populated = once_count > 1
    # The test weight grows along x, so the mean that double-counts x = 0 is the smaller one.
    assert np.all(relion_mean[populated] <= once_mean[populated] * (1 + 1e-12))
    assert np.any(relion_mean[populated] < once_mean[populated] * (1 - 1e-3))


@pytest.mark.parametrize("counting", COUNTINGS)
def test_packed_half_on_relions_own_axis_follows_the_counting_rule(counting, reducer):
    """A half accumulator whose packed axis is RELION's stored axis (the default ``full_half_axis``)."""
    jax.config.update("jax_enable_x64", True)
    grid = GRIDS["odd_bpref_19_current_4"]
    volume_shape, padding_factor, accumulator_volume_shape, r_max = grid
    x_half = _x_half(_grid_size(grid), seed=7)
    expected_sum, expected_count = _stored_half_stats(x_half, grid, "round", counting)
    stats = rr._compute_relion_weight_shell_stats(
        x_half.reshape(-1),
        volume_shape,
        padding_factor=padding_factor,
        r_max=r_max,
        accumulator_volume_shape=accumulator_volume_shape,
        shell_pair_counting=counting,
    )
    assert_matches(np.asarray(stats["shell_sum"]), expected_sum, rtol=1e-12)
    assert_matches(np.asarray(stats["shell_count"]), expected_count, rtol=1e-12)


def test_default_counting_is_relions_and_an_unknown_rule_is_refused():
    jax.config.update("jax_enable_x64", True)
    grid = GRIDS["odd_bpref_19_current_4"]
    volume_shape, padding_factor, accumulator_volume_shape, r_max = grid
    weight = half_volume_mstep.relion_x_half_volume_to_full(_x_half(19, seed=3).reshape(-1), (19,) * 3)
    kwargs = dict(
        padding_factor=padding_factor, r_max=r_max, full_half_axis=RELION_X_AXIS,
        accumulator_volume_shape=accumulator_volume_shape,
    )
    default = rr._compute_relion_weight_shell_stats(weight, volume_shape, **kwargs)
    relion = rr._compute_relion_weight_shell_stats(weight, volume_shape, shell_pair_counting="relion", **kwargs)
    for key in ("shell_sum", "shell_count", "avg_weight_shells"):
        assert_matches(np.asarray(default[key]), np.asarray(relion[key]))
    with pytest.raises(ValueError, match="shell_pair_counting"):
        rr._compute_relion_weight_shell_stats(weight, volume_shape, shell_pair_counting="twice", **kwargs)


@pytest.mark.parametrize("counting", COUNTINGS)
@pytest.mark.parametrize("grid_name", ["odd_bpref_19_full_size", "odd_bpref_11_current_4", "even_padded_16_current_4"])
def test_tau2_and_data_vs_prior_use_the_counted_shell_mean(grid_name, counting, reducer, monkeypatch):
    """sigma2 = 1 / (pf^3 mean weight), tau2 = SSNR sigma2 and data_vs_prior = mean weight tau2 pf^3."""
    jax.config.update("jax_enable_x64", True)
    grid = GRIDS[grid_name]
    volume_shape, padding_factor, accumulator_volume_shape, r_max = grid
    n_shells = volume_shape[0] // 2 + 1
    fsc = np.linspace(0.95, 0.25, n_shells)
    tau2 = np.linspace(2.0, 0.5, n_shells)
    x_half = _x_half(_grid_size(grid), seed=11)
    shell_sum, shell_count = _stored_half_stats(x_half, grid, "round", counting)
    mean_weight = np.where(shell_count > 0, shell_sum / np.maximum(shell_count, 0.5), 0.0)
    with np.errstate(divide="ignore"):
        sigma2 = np.where(mean_weight > 0, 1.0 / (padding_factor**3 * mean_weight), 0.0)
    ssnr = fsc / (1.0 - fsc)
    data_vs_prior = mean_weight * tau2 * padding_factor**3
    data_vs_prior[r_max + 1 :] = 0.0

    results = {}
    for layout, weight in _public_layouts(x_half, monkeypatch).items():
        _, _, details = rr.compute_relion_tau2_from_weights(
            weight, weight, fsc, volume_shape,
            padding_factor=padding_factor, r_max=r_max, return_details=True, full_half_axis=RELION_X_AXIS,
            accumulator_volume_shape=accumulator_volume_shape, output_dtype=jnp.float64,
            shell_pair_counting=counting,
        )
        dvp = rr.compute_data_vs_prior(
            weight, tau2, volume_shape,
            padding_factor=padding_factor, current_size=2 * r_max, full_half_axis=RELION_X_AXIS,
            accumulator_volume_shape=accumulator_volume_shape, shell_pair_counting=counting,
        )
        populated = mean_weight > 0
        assert_matches(np.asarray(details["sigma2_shells"]), sigma2, rtol=1e-12)
        assert_matches(np.asarray(details["prior_shells"])[populated], (ssnr * sigma2)[populated], rtol=1e-12)
        assert_matches(np.asarray(dvp), data_vs_prior, rtol=1e-12)
        results[layout] = np.asarray(details["prior_shells"])
    assert_matches(results["native_half"], results["full"], rtol=1e-12)


# --- gold-standard FSC -------------------------------------------------------------------------


def _round_away(values):
    return (np.sign(values) * np.floor(np.abs(values) + 0.5)).astype(np.int64)


def _downsampled_fsc(data, weights, grid, counting):
    """getDownsampledAverage + calculateDownSampledFourierShellCorrelation on raw x-half storage.

    Each padded voxel lands on the native cell ROUND(k / pf); a cell's average is sum(data) / sum(weight).
    The correlation sums the stored cells with exact radius R <= r_max in shells ROUND(R); ``once``
    weights the cells of x = 0, whose mates are stored too, by 1/2.
    """
    volume_shape, padding_factor, _, r_max = grid
    size = data[0].shape[0]
    kz, ky, kx = _coordinates(size, half=True)
    dz, dy, dx = (_round_away(c / padding_factor) for c in (kz, ky, kx))
    radius = r_max + 1
    inside = (np.abs(dz) <= radius) & (np.abs(dy) <= radius) & (dx <= radius)
    side = 2 * radius + 1
    label = ((dz + radius) * side + (dy + radius)) * (radius + 1) + dx
    n_cells = side * side * (radius + 1)
    averages = []
    for values, weight in zip(data, weights):
        total = np.bincount(label[inside], weights=values.real[inside], minlength=n_cells) + 1j * np.bincount(
            label[inside], weights=values.imag[inside], minlength=n_cells
        )
        weight_total = np.bincount(label[inside], weights=weight.real[inside], minlength=n_cells)
        averages.append(np.where(weight_total > 0, total / np.where(weight_total > 0, weight_total, 1.0), 0.0))
    cell = np.arange(-radius, radius + 1)
    cz, cy, cx = np.meshgrid(cell, cell, np.arange(radius + 1), indexing="ij")
    r = np.sqrt((cz**2 + cy**2 + cx**2).astype(np.float64)).reshape(-1)
    keep = r <= r_max
    shell = np.floor(r + 0.5).astype(np.int64)[keep]
    pair_weight = np.where(cx.reshape(-1)[keep] == 0, 0.5, 1.0) if counting == "once" else np.ones(keep.sum())
    a, b = averages[0][keep], averages[1][keep]
    n_shells = volume_shape[0] // 2 + 1
    cross = np.bincount(shell, weights=(np.conj(a) * b).real * pair_weight, minlength=n_shells)
    power_a = np.bincount(shell, weights=np.abs(a) ** 2 * pair_weight, minlength=n_shells)
    power_b = np.bincount(shell, weights=np.abs(b) ** 2 * pair_weight, minlength=n_shells)
    fsc = np.zeros(n_shells)
    populated = power_a * power_b > 0
    fsc[populated] = cross[populated] / np.sqrt(power_a[populated] * power_b[populated])
    fsc[0] = 1.0
    return fsc


def _half_accumulators(size, seed):
    """Two half sets sharing a signal: complex data (Hermitian x = 0 plane) and positive weights."""
    signal = _x_half(size, seed, complex_values=True)
    data, weights = [], []
    for half in range(2):
        weight = _x_half(size, seed + 10 + half)
        noise = 0.6 * _x_half(size, seed + 20 + half, complex_values=True)
        data.append((signal + noise) * weight)
        weights.append(weight)
    return data, weights


@pytest.mark.parametrize("counting", COUNTINGS)
@pytest.mark.parametrize(
    "grid_name", ["odd_bpref_19_full_size", "odd_bpref_19_current_4", "odd_bpref_11_current_4", "odd_bpref_15_box_6"]
)
def test_backprojector_fsc_follows_the_counting_rule_on_both_layouts(grid_name, counting, monkeypatch):
    grid = GRIDS[grid_name]
    volume_shape, padding_factor, accumulator_volume_shape, r_max = grid
    data, weights = _half_accumulators(_grid_size(grid), seed=13)
    expected = _downsampled_fsc(data, weights, grid, counting)

    layouts = [_public_layouts(values, monkeypatch) for values in (*data, *weights)]
    for layout in ("full", "native_half"):
        y0, y1, w0, w1 = (values[layout] for values in layouts)
        fsc = rr.compute_relion_fsc_from_backprojector(
            y0, y1, w0, w1, volume_shape,
            padding_factor=padding_factor, r_max=r_max, accumulator_volume_shape=accumulator_volume_shape,
            output_dtype=jnp.float64, shell_pair_counting=counting,
        )
        assert_matches(np.asarray(fsc), expected, rtol=1e-12, err_msg=layout)
    # The final pass's route: a Hermitian full layout reduced through its packed half.
    y0, y1, w0, w1 = (values["full"] for values in layouts)
    fsc = rr.compute_relion_fsc_from_backprojector(
        y0, y1, w0, w1, volume_shape,
        padding_factor=padding_factor, r_max=r_max, accumulator_volume_shape=accumulator_volume_shape,
        output_dtype=jnp.float64, full_is_hermitian=True, shell_pair_counting=counting,
    )
    assert_matches(np.asarray(fsc), expected, rtol=1e-12)


def test_backprojector_fsc_counting_changes_the_curve_and_the_default_is_relions():
    grid = GRIDS["odd_bpref_19_full_size"]
    volume_shape, padding_factor, accumulator_volume_shape, r_max = grid
    data, weights = _half_accumulators(19, seed=13)
    relion = _downsampled_fsc(data, weights, grid, "relion")
    once = _downsampled_fsc(data, weights, grid, "once")
    assert np.max(np.abs(relion[1:] - once[1:])) > 1e-4
    y0, y1, w0, w1 = (
        np.asarray(half_volume_mstep.relion_x_half_volume_to_full(values.reshape(-1), (19,) * 3))
        for values in (*data, *weights)
    )
    kwargs = dict(
        padding_factor=padding_factor, r_max=r_max, accumulator_volume_shape=accumulator_volume_shape,
        output_dtype=jnp.float64,
    )
    default = rr.compute_relion_fsc_from_backprojector(y0, y1, w0, w1, volume_shape, **kwargs)
    assert_matches(np.asarray(default), relion, rtol=1e-12)
    with pytest.raises(ValueError, match="shell_pair_counting"):
        rr.compute_relion_fsc_from_backprojector(y0, y1, w0, w1, volume_shape, shell_pair_counting="twice", **kwargs)


# --- reference power spectra: the Class3D tau2 and the start-up tau2 ------------------------------


def _reference(n, seed=0):
    jax.config.update("jax_enable_x64", True)
    return np.random.default_rng(seed).normal(size=(n, n, n))


def _radial_window(n, pf):
    c = np.arange(n) - n // 2
    r = np.sqrt(c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2)
    return np.sinc(r / (n * pf)) ** 2


def _projector_power_oracle(reference, r_max, pf, counting):
    """computeFourierTransformMap's power spectrum from a whole-grid numpy transform.

    The padded, gridding-corrected reference is transformed on the whole grid; the shell mean of
    |F|^2 / 2 inside the sphere of pf r_max is taken over every voxel once (``once``) or over the
    stored half x >= 0, each stored coefficient once (``relion``).
    """
    n = reference.shape[0]
    m = pf * n
    padded = np.zeros((m, m, m))
    lo = m // 2 - n // 2
    padded[lo : lo + n, lo : lo + n, lo : lo + n] = reference / _radial_window(n, pf)
    transform = np.fft.fftn(np.fft.ifftshift(padded)) / m**3 * (pf**3 * n)
    power = np.abs(transform) ** 2 / 2.0
    k = np.fft.fftfreq(m, d=1.0 / m).astype(np.int64)
    kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")
    r2 = kz**2 + ky**2 + kx**2
    inside = r2 <= (pf * r_max) ** 2
    if counting == "relion":
        # The stored half holds x = 0 .. M/2; the transform's -M/2 is that Nyquist column.
        inside = inside & ((kx >= 0) | (kx == -(m // 2)))
    shell = np.minimum(np.floor(np.sqrt(r2.astype(np.float64)) / pf + 0.5).astype(np.int64), n // 2)
    n_shells = n // 2 + 1
    total = np.bincount(shell[inside], weights=power[inside], minlength=n_shells)
    count = np.bincount(shell[inside], minlength=n_shells)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


@pytest.mark.parametrize("counting", COUNTINGS)
@pytest.mark.parametrize("n,r_max", [(8, 4), (8, 2), (6, 3), (6, 2)])
def test_projector_power_spectrum_follows_the_counting_rule(counting, n, r_max):
    """Full and cropped radius, two box sizes; at the full radius the Nyquist column is stored too."""
    reference = _reference(n)
    _, power = setup.setup_relion_projector_on_host(
        reference, r_max, ori_size=n, padding_factor=2, shell_pair_counting=counting
    )
    assert_matches(np.asarray(power), _projector_power_oracle(reference, r_max, 2, counting), rtol=1e-11)


def test_projector_data_does_not_depend_on_the_counting_and_the_default_is_relions():
    reference = _reference(8)
    default = setup.setup_relion_projector_on_host(reference, 3, ori_size=8, padding_factor=2)
    relion = setup.setup_relion_projector_on_host(reference, 3, ori_size=8, padding_factor=2, shell_pair_counting="relion")
    once = setup.setup_relion_projector_on_host(reference, 3, ori_size=8, padding_factor=2, shell_pair_counting="once")
    assert_matches(default[1], relion[1])
    assert_matches(once[0], relion[0])
    assert np.max(np.abs(once[1] - relion[1])) > 1e-3 * np.max(np.abs(relion[1]))
    with pytest.raises(ValueError, match="shell_pair_counting"):
        setup.setup_relion_projector_on_host(reference, 3, ori_size=8, padding_factor=2, shell_pair_counting="twice")


@pytest.mark.parametrize("counting", COUNTINGS)
def test_class_tau2_from_the_reference_uses_the_counted_power(counting):
    """compute_relion_tau2_from_iref_power_spectrum builds its own spectrum with the requested counting."""
    from recovar.core import fourier_transform_utils as ftu
    from recovar.utils.helpers import recovar_volume_to_relion

    reference = _reference(8, seed=2)
    reference_ft = np.asarray(ftu.get_dft3(jnp.asarray(reference))).reshape(-1)
    _, details = rr.compute_relion_tau2_from_iref_power_spectrum(
        reference_ft, (8, 8, 8), padding_factor=2, current_size=6, return_details=True, shell_pair_counting=counting
    )
    relion_frame = np.asarray(recovar_volume_to_relion(reference), dtype=np.float64)
    power = _projector_power_oracle(relion_frame, 3, 2, counting)
    # ReferenceTau2 scale: the data_dim-2 projector power / N^2, times N^2 pf^3 / 8.
    assert_matches(np.asarray(details["tau2_shells"]), (power * 2**3 / 8.0).astype(np.float32), rtol=1e-5)


@pytest.mark.parametrize("counting", COUNTINGS)
@pytest.mark.parametrize("n", [8, 7])
def test_start_up_power_spectrum_follows_the_counting_rule(counting, n):
    """getSpectrum of the start-up reference (initialiseDataVersusPrior), even and odd box.

    The oracle sums the FFTW half: every stored coefficient once (RELION), or 1/2 on the columns
    whose mates are stored too (kx = 0, and an even box's Nyquist kx).
    """
    volume = _reference(n, seed=4)
    n_shells = n // 2 + 1
    spectrum = reference_initialization._POWER_SPECTRUM_3D[counting](volume, n_shells)

    half = np.fft.rfftn(volume) / volume.size
    k = np.fft.fftfreq(n, d=1.0 / n)
    kx = np.arange(n // 2 + 1)
    shell = np.floor(np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kx[None, None, :] ** 2) + 0.5).astype(int)
    pair_weight = np.ones(half.shape)
    if counting == "once":
        pair_weight[:, :, _own_mate_columns(n)] = 0.5
    keep = shell < n_shells
    expected = np.bincount(shell[keep], weights=(np.abs(half) ** 2 * pair_weight)[keep], minlength=n_shells) / np.bincount(
        shell[keep], weights=pair_weight[keep], minlength=n_shells
    )
    assert_matches(spectrum, expected, rtol=1e-12)


def test_start_up_tau2_and_data_vs_prior_take_the_counting(monkeypatch):
    volume = _reference(8, seed=4)
    sigma2 = np.linspace(2.0, 1.0, 5)
    kwargs = dict(tau2_fudge=1.0, avg_sigma2_noise=sigma2, nr_particles=10)
    default = reference_initialization.relion_initial_tau2_and_data_vs_prior(volume, **kwargs)
    relion = reference_initialization.relion_initial_tau2_and_data_vs_prior(volume, shell_pair_counting="relion", **kwargs)
    once = reference_initialization.relion_initial_tau2_and_data_vs_prior(volume, shell_pair_counting="once", **kwargs)
    assert_matches(default[0], relion[0])
    # tau2 = fudge * spectrum * N^2 / 2 and data_vs_prior follows it shell by shell.
    for counting, (tau2, data_vs_prior) in (("relion", relion), ("once", once)):
        spectrum = reference_initialization._POWER_SPECTRUM_3D[counting](volume, 5)
        assert_matches(tau2, spectrum * 8.0**2 / 2.0, rtol=1e-12)
        assert_matches(data_vs_prior[1:] / tau2[1:], 10 / sigma2[1:] / (2.0 * np.arange(1, 5)), rtol=1e-12)
    assert np.max(np.abs(once[0] - relion[0])) > 1e-3 * np.max(relion[0])
    # The command's start-up helper forwards the option.
    recorded = record_calls(monkeypatch, reference_initialization, "relion_initial_tau2_and_data_vs_prior")
    full_refinement._relion_start_tau2_and_data_vs_prior(
        volume, sigma2 * 8.0**4, grid_size=8, volume_shape=(8, 8, 8), tau2_fudge=1.0, nr_particles=10,
        shell_pair_counting="once",
    )
    assert recorded[0][1]["shell_pair_counting"] == "once"


# --- the controller ----------------------------------------------------------------------------


@pytest.mark.parametrize("counting", COUNTINGS)
def test_k1_refinement_hands_every_shell_statistic_the_counting(monkeypatch, counting):
    """Numbered iterations (half-map FSC, per-half tau2) and the final all-data pass."""
    fsc_calls = record_calls(monkeypatch, rr, "compute_relion_fsc_from_backprojector")
    tau2_calls = record_calls(monkeypatch, rr, "compute_relion_tau2_from_weights")
    stats_calls = record_calls(monkeypatch, rr, "_compute_relion_weight_shell_stats")

    result = run_tiny_refinement(monkeypatch, consistency=RelionConsistencyOptions(shell_pair_counting=counting))

    assert result["final_all_data_ran"]
    assert len(fsc_calls) == 3 and len(tau2_calls) == 5  # two iterations and the final pass
    for calls in (fsc_calls, tau2_calls, stats_calls):
        assert calls and {kwargs["shell_pair_counting"] for _, kwargs in calls} == {counting}


@pytest.mark.parametrize("counting", COUNTINGS)
def test_class3d_refinement_hands_every_shell_statistic_the_counting(monkeypatch, counting):
    """Class tau2 from the reference power, the weight statistics and data-vs-prior."""
    power_calls = record_calls(monkeypatch, rr, "compute_relion_tau2_from_iref_power_spectrum")
    dvp_calls = record_calls(monkeypatch, rr, "compute_data_vs_prior")
    stats_calls = record_calls(monkeypatch, rr, "_compute_relion_weight_shell_stats")

    run_tiny_refinement(
        monkeypatch, n_classes=2, consistency=RelionConsistencyOptions(shell_pair_counting=counting)
    )

    assert len(power_calls) == 4 and len(dvp_calls) == 4  # two classes, two iterations
    for calls in (power_calls, dvp_calls, stats_calls):
        assert {kwargs["shell_pair_counting"] for _, kwargs in calls} == {counting}
    # The scoring projector's spectrum counts as RELION does; another counting builds its own.
    supplied = [kwargs["projector_power_spectrum"] is not None for _, kwargs in power_calls]
    assert not any(supplied) if counting == "once" else any(supplied)


@pytest.mark.parametrize("counting", COUNTINGS)
def test_final_class_priors_take_the_counting(monkeypatch, counting):
    recorded = []

    def estimate(references, denominators, *, settings, **kwargs):
        recorded.append(settings.shell_pair_counting)
        return mean_helpers.ClassPriorEstimate(
            variance=jnp.ones(4), shells=jnp.ones(3), relion_shells=jnp.ones(3), data_vs_prior=jnp.ones(3),
            details={key: np.ones(3) for key in mean_helpers._CLASS_TAU2_DETAIL_KEYS}, weight_shells={},
        )

    monkeypatch.setattr(mean_helpers, "estimate_class_prior", estimate)
    settings = mean_helpers.ReconstructionSettings(
        grid_size=8, voxel_size=1.0, volume_shape=(8, 8, 8), padding_factor=2, projection_padding_factor=2,
        minres_map=5, width_mask_edge=5, fmask_edge=2, tau2_fudge=1.0, particle_diameter_angstrom=None,
        first_iteration_lowpass_angstrom=None, shell_pair_counting=counting,
    )
    final_reconstruction.compute_final_class_priors(
        None, None, projector=None, n_classes=2, settings=settings, current_size=8, accumulator_shape=None,
        full_half_axis=0,
    )
    assert recorded == [counting, counting]
