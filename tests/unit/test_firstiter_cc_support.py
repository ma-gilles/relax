"""The ``firstiter_cc_support`` option: the pixels the first-iteration normalized CC sums.

RELION's CC kernels apply no mask (ml_optimiser.cpp:7414-7429, 6847-6856; diff2.cuh CC kernels with
a uniform ``corr_img``): every pixel of the cropped FFTW rectangle enters, so both ``kx = 0`` copies,
the DC pixel and the corners of the rectangle are summed, unlike the Gaussian iterations (``Mresol > 0``:
rounded shells 1 .. cs/2, without ``jp == 0, ip < 0``). ``"gaussian"`` weights the CC numerator, its
reference norm and the image power ``Xi2`` on the Gaussian support. The oracles enumerate RELION's
FFTW labels in plain numpy.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.run_options import stand_in
from helpers.tiny_refinement import MockHalfSet, engine_stage_kwargs, run_tiny_refinement

from relax.fine_pass.window import _pass2_half_weights
from relax.fourier import half_spectrum
from relax.fourier.fourier_window import make_fourier_window_spec
from relax.refinement.refinement_options import RelionConsistencyOptions

pytestmark = pytest.mark.unit


def _layout_index(ip, jp, box):
    """Flat index of RELION's (ip, jp) in relax's packed half layout of a ``box`` image."""
    ky = np.where(ip == box // 2, -(box // 2), ip)  # RELION's +N/2 row is this layout's -N/2
    return (ky + box // 2) * (box // 2 + 1) + jp


def _cropped_labels(current_size):
    """(ip, jp) of the cropped FFTW half image: rows 0 .. cs/2, -(cs/2 - 1) .. -1, columns 0 .. cs/2."""
    rows = np.arange(current_size)
    ip = np.where(rows < current_size // 2 + 1, rows, rows - current_size)
    return np.meshgrid(ip, np.arange(current_size // 2 + 1), indexing="ij")


def _rectangle(box, current_size):
    """RELION's CC support: every pixel of the cropped rectangle."""
    ip, jp = _cropped_labels(current_size)
    return np.sort(_layout_index(ip, jp, box).ravel())


def _gaussian_support(box, current_size):
    """RELION's Gaussian support (Mresol_fine > 0): shells 1 .. cs/2 without jp == 0, ip < 0."""
    ip, jp = _cropped_labels(current_size)
    shell = np.floor(np.sqrt((ip * ip + jp * jp).astype(np.float64)) + 0.5).astype(np.int64)
    keep = (shell > 0) & (shell <= current_size // 2) & ~((jp == 0) & (ip < 0))
    return np.sort(_layout_index(ip[keep], jp[keep], box))


CASES = [(16, 8), (16, 12), (32, 16), (32, 30), (16, 16), (32, 32)]


@pytest.mark.parametrize("box,current_size", CASES)
def test_gaussian_support_weights_are_relions_gaussian_support(box, current_size):
    """Cropped and full current size."""
    weights = half_spectrum.gaussian_support_weights((box, box), current_size)
    assert set(np.unique(weights)) <= {0.0, 1.0}
    assert_matches(np.flatnonzero(weights), _gaussian_support(box, current_size))
    if current_size == box:
        assert_matches(half_spectrum.gaussian_support_weights((box, box), None), weights)


def _cc_window(box, current_size):
    # The normalized-CC pass keeps its rectangular score window (sparse_pass2_window._pass2_window_setup).
    return make_fourier_window_spec(
        (box, box), current_size, box * (box // 2 + 1), score_square=True, score_include_dc=True, window_at_box=True
    )


@pytest.mark.parametrize("box,current_size", CASES)
def test_cc_weights_default_to_every_pixel_of_the_rectangle_and_gaussian_to_the_gaussian_support(box, current_size):
    spec = _cc_window(box, current_size)
    window = np.arange(box * (box // 2 + 1)) if spec.score_indices_np is None else np.asarray(spec.score_indices_np)
    assert_matches(np.sort(window), _rectangle(box, current_size))
    kwargs = dict(half_spectrum_scoring=True, relion_firstiter_score_mode="normalized_cc", use_float64_scoring=False)
    _, relion = _pass2_half_weights((box, box), spec, **kwargs)
    _, default = _pass2_half_weights((box, box), spec, firstiter_cc_support="relion", current_size=current_size, **kwargs)
    _, gaussian = _pass2_half_weights(
        (box, box), spec, firstiter_cc_support="gaussian", current_size=current_size, **kwargs
    )
    assert np.all(np.asarray(relion) == 1.0)
    assert_matches(np.asarray(default), np.asarray(relion))
    support = _gaussian_support(box, current_size)
    assert_matches(np.asarray(gaussian), np.isin(window, support).astype(np.float32))
    # What leaves the sums: the kx = 0, ky < 0 copies, the DC pixel and the rectangle's corners.
    assert support.size < window.size and np.all(np.isin(support, window))


def test_gaussian_iterations_are_not_touched_by_the_option():
    spec = make_fourier_window_spec((16, 16), 8, 16 * 9)
    kwargs = dict(half_spectrum_scoring=True, relion_firstiter_score_mode="gaussian", use_float64_scoring=False)
    _, relion = _pass2_half_weights((16, 16), spec, **kwargs)
    _, gaussian = _pass2_half_weights((16, 16), spec, firstiter_cc_support="gaussian", current_size=8, **kwargs)
    assert_matches(np.asarray(gaussian), np.asarray(relion))


def _cc_operands(box=8, current_size=6, **kwargs):
    from recovar.core.configs import ForwardModelConfig

    from relax.fine_pass.bucket_io import prepare_unshifted_bucket_operands

    dataset = MockHalfSet(3, np.random.default_rng(0))
    config = ForwardModelConfig.from_dataset(dataset, disc_type="linear_interp", process_fn=dataset.process_images)
    batch, _, _, ctf_params, _, _, indices = next(dataset.iter_batches(3))
    spec = _cc_window(box, current_size)
    operands = prepare_unshifted_bucket_operands(
        dataset, batch, ctf_params, indices,
        noise_variance_half=jnp.ones(box * (box // 2 + 1), dtype=jnp.float32), config=config,
        score_with_masked_images=False, image_corrections=None, scale_corrections=None, image_pre_shifts=None,
        use_float64_scoring=False, return_direct_scoring_io=True, score_mode="normalized_cc",
        window_indices=spec.score_indices, **kwargs,
    )
    return operands, np.asarray(spec.score_indices_np)


def test_image_power_is_summed_on_the_support_the_score_counts():
    """Xi2 of pass 2: RELION's sum over the rectangle, or over the Gaussian support."""
    relion, window = _cc_operands()
    support = _gaussian_support(8, 6)
    weights = np.isin(window, support).astype(np.float32)
    gaussian, _ = _cc_operands(cc_power_weights=weights)
    power = np.abs(np.asarray(relion.processed_score_half_raw, dtype=np.complex128)) ** 2
    assert_matches(np.asarray(relion.batch_norm)[:, 0], power[:, _rectangle(8, 6)].sum(axis=1), rtol=1e-6)
    assert_matches(np.asarray(gaussian.batch_norm)[:, 0], power[:, support].sum(axis=1), rtol=1e-6)
    assert np.all(np.asarray(gaussian.batch_norm) < np.asarray(relion.batch_norm))


def test_coarse_image_power_takes_the_weights():
    """Xi2 of pass 1 (the exact coarse operands)."""
    from relax.scoring.coarse_operands import relion_cc_inverse_power_from_processed

    rng = np.random.default_rng(1)
    processed = rng.normal(size=(2, 8 * 5)) + 1j * rng.normal(size=(2, 8 * 5))
    window = _rectangle(8, 6)
    weights = np.isin(window, _gaussian_support(8, 6)).astype(np.float64)
    power = np.abs(processed) ** 2
    relion = relion_cc_inverse_power_from_processed(processed, window)
    gaussian = relion_cc_inverse_power_from_processed(processed, window, weights)
    assert_matches(np.asarray(relion)[:, 0], 1.0 / power[:, window].sum(axis=1), rtol=1e-12)
    assert_matches(np.asarray(gaussian)[:, 0], 1.0 / (power[:, window] * weights).sum(axis=1), rtol=1e-12)


@pytest.mark.parametrize("box,current_size", [(16, 8), (16, 12), (16, 16)])
def test_fine_cc_score_is_the_normalized_cc_over_the_weighted_support(box, current_size):
    """The fine CC reduction with the two weight sets against the plain numpy normalized CC."""
    from relax.fine_pass.scoring import _relion_cuda_fine_normalized_cc_score

    spec = _cc_window(box, current_size)
    window = np.arange(box * (box // 2 + 1)) if spec.score_indices_np is None else np.asarray(spec.score_indices_np)
    rng = np.random.default_rng(2)
    reference = (rng.normal(size=window.size) + 1j * rng.normal(size=window.size)).astype(np.complex128)
    image = (rng.normal(size=window.size) + 1j * rng.normal(size=window.size)).astype(np.complex128)
    for support in ("relion", "gaussian"):
        _, weights = _pass2_half_weights(
            (box, box), spec, half_spectrum_scoring=True, relion_firstiter_score_mode="normalized_cc",
            use_float64_scoring=True, firstiter_cc_support=support, current_size=current_size,
        )
        counted = np.ones(window.size, dtype=bool) if support == "relion" else np.isin(window, _gaussian_support(box, current_size))
        inverse_power = 1.0 / np.sum(np.abs(image[counted]) ** 2)
        score = _relion_cuda_fine_normalized_cc_score(
            jnp.asarray(reference), jnp.asarray(image), jnp.full(window.size, inverse_power), jnp.asarray(weights)
        )
        expected = np.sum((np.conj(reference) * image).real[counted]) / np.sqrt(
            np.sum(np.abs(reference[counted]) ** 2) * np.sum(np.abs(image[counted]) ** 2)
        )
        assert_matches(float(score), expected, rtol=1e-12)
        assert abs(expected) <= 1.0


@pytest.mark.parametrize("support", ["relion", "gaussian"])
def test_engine_stages_receive_the_support(monkeypatch, support):
    """Coarse CC pass and the resident pass 2, as the adaptive engine calls them."""
    stages = engine_stage_kwargs(monkeypatch, **({} if support == "relion" else {"firstiter_cc_support": support}))
    assert stages["pass1_cc"]["score_mode"] == "normalized_cc"
    assert {kwargs["firstiter_cc_support"] for kwargs in stages.values()} == {support}


@pytest.mark.parametrize("support", ["relion", "gaussian"])
def test_refinement_hands_the_engine_the_support(monkeypatch, support):
    """The numbered iterations (the first is the CC iteration) of the real controller on the stand-in engine."""
    engine_calls = []
    run_tiny_refinement(
        monkeypatch,
        engine_calls=engine_calls,
        parity=dict(emulate_relion_firstiter_cc=True, relion_firstiter_ini_high_angstrom=8.0),
        consistency=RelionConsistencyOptions(firstiter_cc_support=support),
    )
    numbered = engine_calls[:4]
    assert [call["kwargs"]["relion_firstiter_score_mode"] for call in numbered] == ["normalized_cc"] * 2 + ["gaussian"] * 2
    # Every call states the support, the default included.
    assert {call["kwargs"]["firstiter_cc_support"] for call in numbered} == {support}


def test_a_run_without_the_cc_iteration_refuses_the_option():
    from relax.refinement import command_options
    from relax.refinement.refinement_options import require_consistency_route

    gaussian = RelionConsistencyOptions(firstiter_cc_support="gaussian")
    with pytest.raises(NotImplementedError, match="no CC iteration"):
        require_consistency_route(stand_in.options(consistency=gaussian), subtomograms=False, several_image_shapes=False)
    with_cc = stand_in.options(consistency=gaussian, parity=stand_in.parity(emulate_relion_firstiter_cc=True))
    assert require_consistency_route(with_cc, subtomograms=False, several_image_shapes=False) is gaussian
    base = ["--data_dir", "data", "--output", "out", "--firstiter_cc_support", "gaussian"]
    assert command_options.resolve_consistency_options(command_options.parse_refinement_args(base)) == gaussian
    with pytest.raises(SystemExit, match="needs --firstiter_cc"):
        command_options.resolve_consistency_options(command_options.parse_refinement_args([*base, "--no-firstiter_cc"]))
