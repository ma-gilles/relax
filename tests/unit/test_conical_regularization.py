"""The conical-FSC prior (relax.reconstruction.conical_regularization): RELION's rule per cone, and its reduction
to RELION's shell prior when every cone agrees."""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.reconstruction_settings import reconstruction_settings
from helpers.run_options import reconstruction_programs

from relax.reconstruction import conical_regularization as cr
from relax.reconstruction import regularization
from relax.reconstruction.half_volume_mstep import relion_backprojector_volume_shape

BOX = 32
PAD = 2
CURRENT_SIZE = 30
SETTINGS = cr.ConicalPriorSettings(half_angle_deg=30.0, n_axes=40)


def _accumulator_shape(current_size=CURRENT_SIZE):
    return relion_backprojector_volume_shape((BOX,) * 3, PAD, current_size=current_size)


def _frequencies(shape):
    return np.meshgrid(*[np.arange(n) - n // 2 for n in shape], indexing="ij")


def _hermitian_pair(shape, rng, *, weight_of, signal_scale=1.0, noise_scale=0.3, correlated_of=None):
    """Two Hermitian full padded accumulator pairs (data, weight) sharing a signal, with independent noise.

    ``weight_of(k0, k1, k2)`` gives the weight; ``correlated_of`` (a boolean of the same grid) keeps the signal in
    half 2 only where it is true (independent signal elsewhere).
    """
    k = _frequencies(shape)
    weight = weight_of(*k).astype(np.float64)

    def hermitian(values):
        mate = np.conj(values[::-1, ::-1, ::-1])  # odd grids: index i and n-1-i are k and -k
        return 0.5 * (values + mate)

    def complex_noise():
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    signal = hermitian(signal_scale * complex_noise())
    other = hermitian(signal_scale * complex_noise())
    signal2 = signal if correlated_of is None else np.where(correlated_of, signal, other)
    data1 = weight * (signal + hermitian(noise_scale * complex_noise()))
    data2 = weight * (signal2 + hermitian(noise_scale * complex_noise()))
    return (data1, data2), (weight.copy(), weight.copy())


def _packed(values, shape):
    return np.asarray(regularization._packed_half_of_hermitian_full(values, shape))


def _unit_prior(tau2_shells, shape, *, whole=False, solve_dtype=np.float64, current_size=CURRENT_SIZE):
    """A ConePrior whose ratios are 1 everywhere: RELION's shell prior."""
    cube = current_size // 2 + 1
    side = 2 * cube + 1
    n_sets = len(tau2_shells)
    zeros = np.zeros((SETTINGS.n_axes, current_size // 2 + 1))
    sums = cr.ConeShellSums(
        zeros.astype(np.int64), zeros, np.stack([zeros, zeros]), np.stack([zeros] * n_sets),
        zeros[0].astype(np.int64), zeros[0], np.stack([zeros[0]] * 2), np.stack([zeros[0]] * n_sets),
    )
    return cr.ConePrior(
        settings=SETTINGS, whole=whole, volume_shape=(BOX,) * 3, accumulator_shape=shape, padding_factor=PAD,
        current_size=current_size, tau2_fudge=1.0, minres_map=2, tau2_shells=tuple(tau2_shells), sums=sums,
        ratios=np.ones((n_sets, SETTINGS.n_axes, current_size // 2 + 1)),
        measured=np.zeros((SETTINGS.n_axes, current_size // 2 + 1), dtype=bool),
        shrinkage=np.zeros(current_size // 2 + 1),
        ratio_cubes=np.ones((n_sets, side, side, side)), solve_dtype=np.dtype(solve_dtype),
    )


def _settings():
    return reconstruction_settings(
        box_size=BOX, voxel_size=2.0, volume_shape=(BOX,) * 3, padding_factor=PAD, projection_padding_factor=2,
        minres_map=2, width_mask_edge=3, fmask_edge=2, tau2_fudge=1.0, particle_diameter_angstrom=None,
        first_iteration_lowpass_angstrom=None, programs=reconstruction_programs(),
    )


@pytest.mark.parametrize("packed", [False, True])
def test_unit_ratios_reproduce_relion_shell_reconstruction(packed):
    """With every ratio 1 the prior term added to the weight and a solve without a shell prior give RELION's
    regularized map: the term is RECOVAR's 1 / (p^3 T tau2[ires]) on RELION's MAP support."""
    rng = np.random.default_rng(3)
    shape = _accumulator_shape()
    (data, _), (weight, _) = _hermitian_pair(shape, rng, weight_of=lambda a, b, c: 1.0 + 0.1 * (a * a + b * b + c * c))
    tau2 = rng.uniform(0.5, 2.0, size=BOX // 2 + 1) * 1e-2
    if packed:
        data, weight = _packed(data, shape), _packed(weight, shape)
    else:
        data, weight = data.reshape(-1), weight.reshape(-1)
    settings = _settings()
    relion_map = settings.reconstruct(
        weight, data, tau=np.asarray(tau2, dtype=np.float64), current_size=CURRENT_SIZE, accumulator_volume_shape=shape,
        tau_is_1d=True,
    )
    prior = _unit_prior([tau2], shape)
    cone_map = settings.reconstruct(
        prior.regularized_weight(0, weight), data, tau=None, current_size=CURRENT_SIZE, accumulator_volume_shape=shape,
        tau_is_1d=True,
    )
    assert_matches(np.asarray(cone_map), np.asarray(relion_map))


@pytest.mark.parametrize(
    "layout, current_size",
    [("full", CURRENT_SIZE), ("packed_host", CURRENT_SIZE), ("packed_host", BOX), ("packed_device", CURRENT_SIZE)],
)
def test_unit_ratios_reproduce_relion_shell_reconstruction_on_the_large_grid_path(monkeypatch, layout, current_size):
    """On RECOVAR's large-grid single-precision path (forced on a small grid by lowering its voxel threshold) the
    prior's weight is float32, and with every ratio 1 the cone map is RELAX's shell-prior map on that path.

    The packed host layout takes the host-staged solve: at current size 32 the 67^3 accumulator exceeds the 64^3
    reconstruction grid, the Stage-A branch that takes only a float32 weight. A device weight is not donated.
    """
    import jax.numpy as jnp

    monkeypatch.setenv("RECOVAR_RELION_POSTPROCESS_LARGE_GRID_SINGLE_PRECISION", "auto")
    monkeypatch.setenv("RECOVAR_RELION_POSTPROCESS_SINGLE_PRECISION_MIN_VOXELS", "1000")
    rng = np.random.default_rng(5)
    shape = _accumulator_shape(current_size)
    (data, data2), (weight, weight2) = _hermitian_pair(
        shape, rng, weight_of=lambda a, b, c: 1.0 + 0.1 * (a * a + b * b + c * c)
    )
    estimated = cr.estimate_cone_prior(
        (data, data2), (weight, weight2), [np.full(BOX // 2 + 1, 0.05)] * 2, settings=SETTINGS,
        volume_shape=(BOX,) * 3, accumulator_shape=shape, padding_factor=PAD, current_size=current_size,
        tau2_fudge=1.0, minres_map=2, whole=False, solve_dtype=np.float64,
    )
    assert cr.large_grid_single_precision((BOX,) * 3, shape, PAD)
    assert estimated.solve_dtype == np.float32
    tau2 = rng.uniform(0.5, 2.0, size=BOX // 2 + 1) * 1e-2
    data, weight = data.astype(np.complex64), weight.astype(np.float32)
    if layout == "full":
        data, weight = data.reshape(-1), weight.reshape(-1)
    else:
        data, weight = _packed(data, shape), _packed(weight, shape)
    if layout == "packed_device":
        data, weight = jnp.asarray(data), jnp.asarray(weight)
    settings = _settings()
    relion_map = settings.reconstruct(
        weight, data, tau=np.asarray(tau2, dtype=np.float64), current_size=current_size,
        accumulator_volume_shape=shape, tau_is_1d=True,
    )
    unit = _unit_prior([tau2], shape, solve_dtype=estimated.solve_dtype, current_size=current_size)
    regularized = unit.regularized_weight(0, weight)
    assert regularized.dtype == np.float32 and regularized.shape == weight.shape
    assert isinstance(regularized, np.ndarray) == isinstance(weight, np.ndarray)
    if layout == "packed_device":
        assert not weight.is_deleted()
    cone_map = settings.reconstruct(
        regularized, data, tau=None, current_size=current_size, accumulator_volume_shape=shape, tau_is_1d=True,
    )
    assert_matches(np.asarray(cone_map), np.asarray(relion_map))


def test_the_default_grid_path_keeps_the_callers_dtype():
    """Below RECOVAR's large-grid threshold the prior keeps the caller's dtype (float64 for the numbered solves)."""
    rng = np.random.default_rng(6)
    shape = _accumulator_shape()
    data, weights = _hermitian_pair(shape, rng, weight_of=lambda a, b, c: 1.0 + np.abs(a))
    assert not cr.large_grid_single_precision((BOX,) * 3, shape, PAD)
    prior = cr.estimate_cone_prior(
        data, weights, [np.full(BOX // 2 + 1, 0.05)] * 2, settings=SETTINGS, volume_shape=(BOX,) * 3,
        accumulator_shape=shape, padding_factor=PAD, current_size=CURRENT_SIZE, tau2_fudge=1.0, minres_map=2,
        whole=False, solve_dtype=np.float64,
    )
    assert prior.solve_dtype == np.float64
    assert prior.regularized_weight(0, weights[0].reshape(-1)).dtype == np.float64


def test_shell_fsc_matches_relion_backprojector_fsc_with_pairs_counted_once():
    """The native averages are RELION's downsampled averages: their shell FSC is the backprojector FSC that counts
    every Hermitian pair once."""
    rng = np.random.default_rng(5)
    shape = _accumulator_shape()
    data, weights = _hermitian_pair(
        shape, rng, weight_of=lambda a, b, c: 1.0 + 0.2 * np.abs(c), noise_scale=1.0
    )
    r_max = CURRENT_SIZE // 2
    native = cr.native_half_averages(data, weights, accumulator_shape=shape, padding_factor=PAD, r_max=r_max)
    sums = cr.cone_shell_sums(native, native.weights, SETTINGS.axes(), SETTINGS.cos_half_angle)
    relion = regularization.compute_relion_fsc_from_backprojector(
        *(d.reshape(-1) for d in data), *(w.reshape(-1) for w in weights), (BOX,) * 3, padding_factor=PAD,
        r_max=r_max, accumulator_volume_shape=shape, output_dtype=np.float64, shell_pair_counting="once",
        packed_stream_min_elements=1 << 60, fsc_dump=None,
    )
    assert_matches(sums.shell_fsc()[1:], np.asarray(relion)[1 : r_max + 1])


def test_packed_and_full_accumulators_give_one_prior():
    """A packed half and its full Hermitian expansion give the same statistics and the same term on shared voxels."""
    rng = np.random.default_rng(7)
    shape = _accumulator_shape()
    data, weights = _hermitian_pair(shape, rng, weight_of=lambda a, b, c: 1.0 + np.abs(a) + 0.3 * np.abs(c))
    tau2 = [np.full(BOX // 2 + 1, 0.05), np.full(BOX // 2 + 1, 0.07)]
    common = dict(
        settings=SETTINGS, volume_shape=(BOX,) * 3, accumulator_shape=shape, padding_factor=PAD,
        current_size=CURRENT_SIZE, tau2_fudge=1.0, minres_map=2, whole=False, solve_dtype=np.float64,
    )
    full = cr.estimate_cone_prior(data, weights, tau2, **common)
    packed = cr.estimate_cone_prior(
        [_packed(d, shape) for d in data], [_packed(w, shape) for w in weights], tau2, **common
    )
    assert_matches(packed.ratios, full.ratios)
    zero_weight = np.zeros(shape)
    full_term = full.regularized_weight(1, zero_weight.reshape(-1))
    packed_term = packed.regularized_weight(1, _packed(zero_weight, shape))
    assert_matches(packed_term, _packed(full_term.reshape(shape), shape))


def _cone_tilts(axes):
    return np.degrees(np.arccos(np.abs(axes[:, 2])))


def test_a_better_sampled_cone_gets_a_smaller_tau2():
    """At an FSC that saturates in every cone the ratio is the shell's mean weight over the cone's: cones along a
    better-sampled axis get a ratio below 1, the others above."""
    rng = np.random.default_rng(11)
    shape = _accumulator_shape()

    def weight_of(a, b, c):
        r = np.sqrt(a * a + b * b + c * c) + 1e-9
        return np.where(np.abs(c) >= 0.8 * r, 20.0, 1.0)

    data, weights = _hermitian_pair(shape, rng, weight_of=weight_of, noise_scale=1e-4)
    tau2 = [np.full(BOX // 2 + 1, 0.05)] * 2
    prior = cr.estimate_cone_prior(
        data, weights, tau2, settings=SETTINGS, volume_shape=(BOX,) * 3, accumulator_shape=shape,
        padding_factor=PAD, current_size=CURRENT_SIZE, tau2_fudge=1.0, minres_map=2, whole=False,
        solve_dtype=np.float64,
    )
    tilt = _cone_tilts(SETTINGS.axes())
    measured_shells = np.flatnonzero(prior.measured.any(axis=0))
    assert measured_shells.size > 0
    s = int(measured_shells[-1])
    assert np.all(prior.ratios[0, tilt < 15.0, s] < 1.0)
    assert np.all(prior.ratios[0, tilt > 75.0, s] > 1.0)


def test_a_cone_without_shared_signal_gets_a_small_tau2():
    """Equal weights, signal shared between the halves only away from the z axis: the cones along z get a ratio far
    below 1 (RELION's rule in a cone whose FSC is near 0), the equatorial cones above 1."""
    rng = np.random.default_rng(13)
    shape = _accumulator_shape()
    k = _frequencies(shape)
    r = np.sqrt(sum(x.astype(np.float64) ** 2 for x in k)) + 1e-9
    shared = np.abs(k[2]) < 0.5 * r
    data, weights = _hermitian_pair(
        shape, rng, weight_of=lambda a, b, c: np.ones(a.shape), noise_scale=0.2, correlated_of=shared
    )
    tau2 = [np.full(BOX // 2 + 1, 0.05)] * 2
    prior = cr.estimate_cone_prior(
        data, weights, tau2, settings=SETTINGS, volume_shape=(BOX,) * 3, accumulator_shape=shape,
        padding_factor=PAD, current_size=CURRENT_SIZE, tau2_fudge=1.0, minres_map=2, whole=False,
        solve_dtype=np.float64,
    )
    tilt = _cone_tilts(SETTINGS.axes())
    s = int(np.flatnonzero(prior.measured.any(axis=0))[-1])
    assert np.all(prior.ratios[0, tilt < 15.0, s] < 0.2)
    assert np.all(prior.ratios[0, tilt > 75.0, s] > 1.0)
    # The voxel ratio averages the cones that hold the voxel: smaller along z than in the equatorial plane.
    cube = prior.ratio_cubes[0]
    centre = CURRENT_SIZE // 2 + 1
    assert cube[centre, centre, centre + s] < cube[centre + s, centre, centre]


def test_cones_that_differ_by_noise_alone_keep_relions_value():
    """Equal weights and an isotropic signal at a low SNR: the cone FSCs scatter by noise only, the shrinkage stays
    small and the ratios stay near 1 (RELION's prior) instead of following each cone's noise. Over seeds 23, 29 and 31
    the median |log ratio| falls from 0.5-0.85 without shrinkage to about 0.035 with it; the test asks for a factor 5."""
    rng = np.random.default_rng(23)
    shape = _accumulator_shape()
    data, weights = _hermitian_pair(
        shape, rng, weight_of=lambda a, b, c: np.ones(a.shape), signal_scale=0.3, noise_scale=1.0
    )
    sums_prior = cr.estimate_cone_prior(
        data, weights, [np.full(BOX // 2 + 1, 0.05)] * 2, settings=SETTINGS, volume_shape=(BOX,) * 3,
        accumulator_shape=shape, padding_factor=PAD, current_size=CURRENT_SIZE, tau2_fudge=1.0, minres_map=2,
        whole=False, solve_dtype=np.float64,
    )
    measured_shells = np.flatnonzero(sums_prior.measured.any(axis=0))
    assert measured_shells.size > 0
    assert np.all(sums_prior.shrinkage[measured_shells] < 0.5)
    measured = sums_prior.measured
    raw = cr.relion_ssnr(sums_prior.sums.cone_fsc(), whole=False) / cr.relion_ssnr(
        sums_prior.sums.shell_fsc(), whole=False
    )[None, :]
    raw_spread = np.median(np.abs(np.log(raw[measured])))
    shrunk_spread = np.median(np.abs(np.log(sums_prior.ratios[0][measured])))
    assert raw_spread > 0.3
    assert shrunk_spread < 0.2 * raw_spread


def test_whole_data_prior_has_one_weight_set():
    """The final M-step's prior takes the joined halves' one shell curve and the summed weights."""
    rng = np.random.default_rng(17)
    shape = _accumulator_shape()
    data, weights = _hermitian_pair(shape, rng, weight_of=lambda a, b, c: 1.0 + np.abs(b))
    prior = cr.estimate_cone_prior(
        data, weights, [np.full(BOX // 2 + 1, 0.05)], settings=SETTINGS, volume_shape=(BOX,) * 3,
        accumulator_shape=shape, padding_factor=PAD, current_size=CURRENT_SIZE, tau2_fudge=1.0, minres_map=2,
        whole=True, solve_dtype=np.float32,
    )
    assert prior.ratios.shape[0] == 1 and prior.ratio_cubes.shape[0] == 1
    assert prior.regularized_weight(0, weights[0].reshape(-1)).dtype == np.float32


def test_relion_ssnr_matches_the_shell_rule():
    """The SSNR is RELION's: clamp, whole-map transform, F / (1 - F)."""
    fsc = np.array([-0.2, 0.0005, 0.3, 0.9995])
    clamped = np.array([0.001, 0.001, 0.3, 0.999])
    assert_matches(cr.relion_ssnr(fsc, whole=False), clamped / (1 - clamped))
    whole = np.minimum(np.sqrt(2 * np.maximum(fsc, 0.001) / (np.maximum(fsc, 0.001) + 1)), 0.999)
    assert_matches(cr.relion_ssnr(fsc, whole=True), whole / (1 - whole))


def test_cones_must_cover_the_sphere():
    """Too few or too narrow cones leave directions outside every cone, which the settings refuse."""
    with pytest.raises(ValueError, match="outside every cone"):
        cr.ConicalPriorSettings(half_angle_deg=10.0, n_axes=20)
    assert cr.largest_axis_distance_deg(cr.cone_axes(100)) < 20.0
    cr.ConicalPriorSettings()  # the defaults cover it


def test_refinement_with_unmeasured_cones_gives_relions_maps(monkeypatch):
    """A tiny K=1 run (its cone-shell bins hold fewer than MIN_CONE_VOXELS voxels, so every ratio is 1) estimates the
    conical-FSC prior in each numbered M-step and the final one and gives RELION's maps."""
    from helpers.tiny_refinement import record_calls, run_tiny_refinement

    relion = run_tiny_refinement(monkeypatch)
    estimates = record_calls(monkeypatch, cr, "estimate_cone_prior")
    cone = run_tiny_refinement(monkeypatch, conical_prior=cr.ConicalPriorSettings())
    assert [kwargs["whole"] for _, kwargs in estimates] == [False, False, True]
    assert_matches(np.asarray(cone.maps.mean), np.asarray(relion.maps.mean))
    for cone_half, relion_half in zip(cone.maps.means, relion.maps.means, strict=True):
        assert_matches(np.asarray(cone_half), np.asarray(relion_half))


def test_star_tables_write_as_model_star_blocks(tmp_path):
    """Each weight set's statistics become the two extra model-STAR blocks, one row per cone-shell bin, which the
    STAR reader parses back by name."""
    from relax.refinement import run_files
    from relax.star.blocks import read_star_blocks

    rng = np.random.default_rng(19)
    shape = _accumulator_shape()
    data, weights = _hermitian_pair(shape, rng, weight_of=lambda a, b, c: 1.0 + np.abs(c))
    prior = cr.estimate_cone_prior(
        data, weights, [np.full(BOX // 2 + 1, 0.05)] * 2, settings=SETTINGS, volume_shape=(BOX,) * 3,
        accumulator_shape=shape, padding_factor=PAD, current_size=CURRENT_SIZE, tau2_fudge=1.0, minres_map=2,
        whole=False, solve_dtype=np.float64,
    )
    general, bins = prior.star_tables(1)
    path = tmp_path / "model.star"
    path.write_text(
        run_files._list_block(run_files.CONE_PRIOR_BLOCK, general)
        + run_files._loop_block(run_files.CONE_PRIOR_BINS_BLOCK, bins)
    )
    blocks = read_star_blocks(path)
    table = blocks[run_files.CONE_PRIOR_BINS_BLOCK]
    n_rows = SETTINGS.n_axes * (CURRENT_SIZE // 2)
    assert len(np.asarray(table["rlnRelaxCfscTau2Ratio"])) == n_rows
    assert_matches(np.asarray(table["rlnRelaxCfscTau2Ratio"], dtype=np.float64), bins["rlnRelaxCfscTau2Ratio"])
