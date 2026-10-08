"""The ``nyquist_column_counting`` option: the Hermitian pairs of a full-size image's Nyquist column.

RELION's per-image support at the box (``Mresol_fine``: ``ROUND(r) <= N/2`` without ``jp == 0 && ip < 0``,
ml_optimiser.cpp:5785-5811) keeps both ``(N/2, ip)`` and ``(N/2, -ip)``, which are one Hermitian pair,
so those pairs count twice in the Gaussian score, the image power (``power_img``), the noise, norm and
scale sums and ``Npix_per_shell``. ``"once"`` drops the ``ip < 0`` members, as RELION drops
``jp == 0, ip < 0``. The oracles enumerate RELION's FFTW labels in plain numpy.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.tiny_refinement import MockHalfSet, engine_stage_kwargs, record_calls, run_tiny_refinement

from relax.helpers import half_spectrum
from relax.reconstruction import noise_relion
from relax.refinement.refinement_options import RefinementOptions, RelionConsistencyOptions

pytestmark = pytest.mark.unit


def _layout_labels(size):
    """RELION's (ip, jp) of every pixel of relax's packed half layout (centred rows, Nyquist column last).

    Row ``ky = -N/2`` is RELION's ``ip = +N/2``; the last column is ``jp = N/2``.
    """
    ky = np.arange(size) - size // 2
    ip = np.where(ky == -(size // 2), size // 2, ky)
    return np.broadcast_to(ip[:, None], (size, size // 2 + 1)), np.broadcast_to(np.arange(size // 2 + 1), (size, size // 2 + 1))


def _relion_support(size):
    """RELION's full-size support: rounded shells up to N/2 without jp == 0, ip < 0 (DC kept here)."""
    ip, jp = _layout_labels(size)
    shell = np.floor(np.sqrt((ip * ip + jp * jp).astype(np.float64)) + 0.5).astype(np.int64)
    return (shell <= size // 2) & ~((jp == 0) & (ip < 0)), shell


def _mates_both_counted(counted, size):
    """Counted pixels whose distinct Hermitian mate (-ip, -jp), taken modulo N, is counted too."""
    ip, jp = _layout_labels(size)
    position = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(ip.ravel(), jp.ravel()))}
    doubled = np.zeros(counted.size, dtype=bool)
    for k in np.flatnonzero(counted.ravel()):
        a, b = int(ip.ravel()[k]), int(jp.ravel()[k])
        mate_a = a if a == size // 2 else -a  # +N/2 and -N/2 are the same frequency
        mate_b = b if b == size // 2 else -b
        mate = position.get((mate_a, mate_b))
        doubled[k] = mate is not None and mate != k and bool(counted.ravel()[mate])
    return doubled.reshape(counted.shape)


@pytest.mark.parametrize("size", [8, 16, 32, 64])
def test_relion_counts_the_nyquist_column_pairs_twice_and_once_counts_every_pair_once(size):
    support, _ = _relion_support(size)
    doubled = _mates_both_counted(support, size)
    ip, jp = _layout_labels(size)
    # Under RELION's rule the only pairs with both members counted lie on the Nyquist column.
    assert doubled.any() and np.all(jp[doubled] == size // 2)
    redundant = np.asarray(half_spectrum.redundant_nyquist_column_pixels((size, size))).reshape(size, -1)
    assert_matches(redundant & support, doubled & (ip < 0))
    assert not _mates_both_counted(support & ~redundant, size).any()


@pytest.mark.parametrize("size,in_column,redundant_count", [(128, 17, 8), (256, 23, 11)])
def test_the_doubled_pixels_of_the_nyquist_shell(size, in_column, redundant_count):
    """The audit's numbers: 17 pixels of shell 64 lie in that column at box 128 (8 redundant), 23 at 256."""
    support, shell = _relion_support(size)
    _, jp = _layout_labels(size)
    assert int(np.count_nonzero(support & (jp == size // 2) & (shell == size // 2))) == in_column
    redundant = np.asarray(half_spectrum.redundant_nyquist_column_pixels((size, size))).reshape(size, -1)
    assert int(np.count_nonzero(redundant & support)) == redundant_count


def test_an_odd_width_has_no_nyquist_column():
    assert not np.asarray(half_spectrum.redundant_nyquist_column_pixels((9, 9))).any()


@pytest.mark.parametrize("size", [8, 16, 32])
def test_gaussian_weights_drop_the_redundant_members_only(size):
    relion = np.asarray(half_spectrum.make_scoring_half_image_weights((size, size), relion_half_sum=True))
    default = np.asarray(
        half_spectrum.make_scoring_half_image_weights((size, size), relion_half_sum=True, nyquist_column_counting="relion")
    )
    once = np.asarray(
        half_spectrum.make_scoring_half_image_weights((size, size), relion_half_sum=True, nyquist_column_counting="once")
    )
    assert_matches(default, relion)
    ip, jp = (labels.ravel() for labels in _layout_labels(size))
    assert_matches(relion, (~((jp == 0) & (ip < 0))).astype(np.float32))
    assert_matches(once, (~((jp == 0) & (ip < 0)) & ~((jp == size // 2) & (ip < 0))).astype(np.float32))
    with pytest.raises(ValueError, match="nyquist_column_counting"):
        half_spectrum.make_scoring_half_image_weights((size, size), relion_half_sum=True, nyquist_column_counting="x")


def test_first_iteration_cc_weights_keep_relions_rule():
    """The normalized-CC kernels apply no mask (option firstiter_cc_support)."""
    weights = half_spectrum.make_scoring_half_image_weights(
        (16, 16), relion_half_sum=True, exclude_relion_redundant_x0=False, nyquist_column_counting="once"
    )
    assert np.all(np.asarray(weights) == 1.0)


@pytest.mark.parametrize("size,current_size", [(16, 16), (16, None), (32, 32), (32, 16), (64, 32)])
def test_pixel_counts_per_shell_follow_the_counting_rule(size, current_size):
    """Npix_per_shell at the box, and with the summed count below it: only the Nyquist shell changes."""
    support, shell = _relion_support(size)
    relion_count = np.bincount(shell[support], minlength=size // 2 + 1)
    redundant = np.asarray(half_spectrum.redundant_nyquist_column_pixels((size, size))).reshape(size, -1)
    once_count = np.bincount(shell[support & ~redundant], minlength=size // 2 + 1)
    assert np.flatnonzero(relion_count != once_count).tolist() == [size // 2]

    sums = np.ones(size // 2 + 1)
    for counting, expected in (("relion", relion_count), ("once", once_count)):
        sigma2 = np.asarray(
            noise_relion.normalize_wsum_to_sigma2_noise(
                sums, np.zeros_like(sums), 0.5, (size, size), nyquist_column_counting=counting
            )
        )
        assert_matches(1.0 / sigma2, expected.astype(np.float64), rtol=1e-6)
        summed = noise_relion.summed_noise_pixels_per_shell((size, size), current_size, counting)
        low = slice(0, (size if current_size is None else current_size) // 2 + 1)
        if current_size is None or current_size == size:
            assert_matches(np.asarray(summed, dtype=np.int64), expected)
        else:
            assert int(summed[size // 2]) == int(expected[size // 2])
            assert np.all(np.asarray(summed)[low] <= relion_count[low])


def _unshifted_operands(**kwargs):
    import jax.numpy as jnp
    from recovar.core.configs import ForwardModelConfig

    from relax.sparse_pass2.sparse_pass2_bucket_io import prepare_unshifted_bucket_operands

    dataset = MockHalfSet(3, np.random.default_rng(0))
    config = ForwardModelConfig.from_dataset(dataset, disc_type="linear_interp", process_fn=dataset.process_images)
    batch, _, _, ctf_params, _, _, indices = next(dataset.iter_batches(3))
    return prepare_unshifted_bucket_operands(
        dataset, batch, ctf_params, indices,
        noise_variance_half=jnp.ones(8 * 5, dtype=jnp.float32), config=config, score_with_masked_images=False,
        image_corrections=None, scale_corrections=None, image_pre_shifts=None, use_float64_scoring=False,
        return_direct_scoring_io=True, **kwargs,
    )


def test_the_noise_image_loses_the_redundant_members_and_the_score_image_keeps_them():
    """The one array every noise, power, norm and scale sum of pass 2 reads."""
    relion, default, once = (
        _unshifted_operands(),
        _unshifted_operands(nyquist_column_counting="relion"),
        _unshifted_operands(nyquist_column_counting="once"),
    )
    redundant = np.asarray(half_spectrum.redundant_nyquist_column_pixels((8, 8)))
    assert redundant.sum() == 3
    noise_image = np.asarray(relion.processed_score_half_for_noise)
    assert np.all(noise_image[:, redundant] != 0)
    assert_matches(np.asarray(default.processed_score_half_for_noise), noise_image)
    assert_matches(np.asarray(once.processed_score_half_for_noise), np.where(redundant[None, :], 0.0, noise_image))
    for name in ("sparse_score_input_half", "score_weighted_half", "recon_weighted_half", "processed_recon_half_raw"):
        assert_matches(np.asarray(getattr(once, name)), np.asarray(getattr(relion, name)))


def test_image_power_of_the_noise_image_counts_each_nyquist_pair_once():
    """Shell power of the masked image against the oracle's pair-once sum over RELION's labels."""
    once = _unshifted_operands(nyquist_column_counting="once")
    relion = _unshifted_operands()
    support, shell = _relion_support(8)
    redundant = np.asarray(half_spectrum.redundant_nyquist_column_pixels((8, 8))).reshape(8, -1)
    image = np.asarray(relion.processed_score_half_for_noise).reshape(3, 8, 5)
    power = np.abs(image) ** 2
    keep = support & ~redundant
    expected = np.stack([np.bincount(shell[keep], weights=p[keep], minlength=5) for p in power])
    table = np.asarray(half_spectrum.make_relion_noise_shell_indices_half((8, 8)))
    masked_power = np.abs(np.asarray(once.processed_score_half_for_noise)) ** 2
    got = np.stack([half_spectrum.bin_shell_values_np(p, table, 5) for p in masked_power])
    assert_matches(got, expected, rtol=1e-6)


def test_pass2_weight_builder_takes_the_rule():
    from relax.helpers.fourier_window import make_fourier_window_spec
    from relax.sparse_pass2.sparse_pass2_window import _pass2_half_weights

    spec = make_fourier_window_spec((16, 16), 16, 16 * 9, window_at_box=True)
    kwargs = dict(half_spectrum_scoring=True, relion_firstiter_score_mode="gaussian", use_float64_scoring=False)
    _, relion = _pass2_half_weights((16, 16), spec, **kwargs)
    _, once = _pass2_half_weights((16, 16), spec, nyquist_column_counting="once", **kwargs)
    redundant = np.asarray(half_spectrum.redundant_nyquist_column_pixels((16, 16)))[spec.score_indices_np]
    assert redundant.sum() == 2 and np.all(np.asarray(relion) == 1.0)
    assert_matches(np.asarray(once), (~redundant).astype(np.float32))


@pytest.mark.parametrize("counting", ["relion", "once"])
def test_engine_stages_receive_the_rule(monkeypatch, counting):
    """Coarse pass (Gaussian and CC routes) and the resident pass 2, as the adaptive engine calls them."""
    stages = engine_stage_kwargs(monkeypatch, **({} if counting == "relion" else {"nyquist_column_counting": counting}))
    assert set(stages) == {"pass1", "pass1_cc", "pass2"}
    assert {kwargs["nyquist_column_counting"] for kwargs in stages.values()} == {counting}


@pytest.mark.parametrize("counting", ["relion", "once"])
def test_refinement_hands_the_engine_and_the_noise_update_the_rule(monkeypatch, counting):
    """Numbered iterations and the final pass of the real controller on the stand-in engine."""
    engine_calls = []
    noise = record_calls(monkeypatch, noise_relion, "normalize_wsum_to_sigma2_noise")
    run_tiny_refinement(
        monkeypatch, engine_calls=engine_calls, consistency=RelionConsistencyOptions(nyquist_column_counting=counting)
    )
    assert len(engine_calls) == 6  # two halves, two iterations and the final pass
    received = {call["kwargs"].get("nyquist_column_counting", "relion") for call in engine_calls}
    assert received == {counting}
    # The default adds no keyword to the engine call.
    assert all(("nyquist_column_counting" in call["kwargs"]) == (counting == "once") for call in engine_calls)
    assert noise and {kwargs["nyquist_column_counting"] for _, kwargs in noise} == {counting}


def test_local_search_policies_carry_the_rule():
    from relax.refinement.half_scoring import LocalExecutionPolicy
    from relax.refinement.local_search_iteration import LocalSearchKernelPolicy
    from relax.refinement.refinement_options import LocalAdaptivePass2Support

    pruned = LocalAdaptivePass2Support(full_parent=False, rotation_only=False, denominator_mode=None)
    assert LocalExecutionPolicy(
        disc_type="x", disable_adjoint_y=False, disable_adjoint_ctf=False, relion_x_half_mstep=False, adaptive_pass2=pruned
    ).nyquist_column_counting == "relion"
    assert LocalSearchKernelPolicy(disc_type="x", current_size=8).nyquist_column_counting == "relion"


def test_resident_pass_refuses_the_rule_for_subtomograms_and_the_full_grid_gemm_engine():
    from relax.sparse_pass2 import resident_pass2

    for refused in (dict(tilt=object()), dict(dense_gemm_full_grid=True)):
        with pytest.raises(NotImplementedError, match="single-particle resident pass 2"):
            resident_pass2._resident_pass2(
                None, None, None, None, None, 0, "linear_interp",
                oversampling_order=1, current_size=8, translation_step=None, rotation_log_prior=None,
                score_with_masked_images=True, return_stats=True, translation_log_prior=None, accumulate_noise=True,
                half_spectrum_scoring=True, projection_padding_factor=2, reconstruction_padding_factor=2,
                image_corrections=None, scale_corrections=None, image_pre_shifts=None, use_float64_scoring=False,
                random_perturbation=0.0, nyquist_column_counting="once", **refused,
            )


def test_gemm_dense_coarse_engine_refuses_the_rule():
    from relax.refinement.refinement_options import AdaptiveOptions, require_consistency_route

    options = RefinementOptions(
        consistency=RelionConsistencyOptions(nyquist_column_counting="once"),
        adaptive=AdaptiveOptions(coarse_engine="gemm_dense"),
    )
    with pytest.raises(NotImplementedError, match="gemm_dense"):
        require_consistency_route(options, subtomograms=False, several_image_shapes=False)
