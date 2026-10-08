"""Program-count reductions in the coarse pass that must not move any number.

Three changes are covered, all of them program-count changes only:

* ``RELAX_COARSE_PAD_FINAL_IMAGE_BATCH`` pads a half set's last coarse image
  batch up to ``image_batch_size`` by repeating image row zero, so the coarse
  pass, significance and image preprocessing see a single image extent per run
  instead of one per remainder. The repeated rows are dropped from every
  science output.
* ``RELAX_EM_JIT_STAGE_GLUE`` runs the post-transform preprocessing chain as one
  jitted program, instead of one XLA program per primitive per extent.
* ``_collate_batch_to_jax`` builds the host array before the device transfer,
  which removes the ``convert_element_type`` program JAX compiles for every
  distinct Python-list length.

Each test compares the changed path against the unchanged one within the float
band (discrete outputs exactly).
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.unit

def _significance_call(monkeypatch, n_classes=2):
    """Arguments for a coarse significance call whose last batch is a remainder.

    Pass 1 runs on the CPU exact-operand harness (helpers.exact_pass1_harness).
    """

    from helpers.exact_pass1_harness import ExactPass1Dataset, coded_class_projectors, install_exact_pass1_mocks

    install_exact_pass1_mocks(monkeypatch)
    dataset = ExactPass1Dataset(np.arange(7))
    rotations = np.tile(np.eye(3, dtype=np.float32), (5, 1, 1))
    rotations[:, 0, 1] = np.asarray([0.0, 0.3, 0.1, 0.4, 0.2], dtype=np.float32)
    args = (
        dataset,
        jnp.ones(dataset.image_size, dtype=jnp.float32),
        rotations,
        jnp.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 0.0]], dtype=jnp.float32),
    )
    kwargs = dict(
        class_log_priors=np.log(np.arange(1, n_classes + 1) / sum(range(1, n_classes + 1))),
        rotation_log_prior=np.linspace(0.0, -0.4, 5, dtype=np.float32),
        translation_log_prior=np.linspace(
            0.0, -0.3, 7 * 3, dtype=np.float32
        ).reshape(7, 3),
        adaptive_fraction=0.9,
        max_significants=6,
        # 7 images in batches of 3 leaves a 1-image tail batch.
        image_batch_size=3,
        rotation_block_size=2,
        current_size=4,
        half_spectrum_scoring=True,
        return_class_best=True,
        relion_projector_half=coded_class_projectors(n_classes),
        relion_projector_r_max=1,
        relion_projector_texture_interp=True,
    )
    return args, kwargs


def _assert_significance_results_match(candidate, control):
    for actual, expected in zip(candidate[:4], control[:4]):
        assert_matches(np.asarray(actual), np.asarray(expected))
    for actual_class, expected_class in zip(candidate[4], control[4]):
        for actual, expected in zip(actual_class, expected_class):
            if actual is None or expected is None:
                assert actual is expected
            else:
                assert_matches(np.asarray(actual), np.asarray(expected))
    assert set(candidate[5]) == set(control[5])
    for key, expected in control[5].items():
        actual = candidate[5][key]
        if isinstance(expected, np.ndarray):
            assert_matches(actual, expected)
            assert actual.dtype == expected.dtype


# P3-B measured this path's own floor on a GPU: repeating the identical
# configuration moves 0 to 3 of 10080 entries, each by one float32 ulp, always
# in a reporting field and never in a mask, a count, a sample-index array or an
# assignment (its report, section 5a). On a GPU the batch extent changes the
# reduction shape, so the padded path moves inside that measured band.
#
# Measured on a claimed A100 at this commit, this fixture compares 77 float
# entries, its null control (the identical configuration twice in one process)
# moves 0, and the padded path moves 6, all of them in four ``full_stats``
# reporting fields and none by more than one float32 ulp (1.2e-7 relative).
# Those fields are float64 accumulators over float32 posteriors, so their band
# is the float32 one, not the float64 default.
_REPORTING_FIELD_RTOL = 1e-6


def _assert_significance_results_within_null_band(candidate, control):
    """Default band everywhere except reporting fields, which get the float32 band."""

    def compare(actual, expected, label, *, reporting):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        assert actual.shape == expected.shape, label
        assert actual.dtype == expected.dtype, label
        # masks, counts, sample indices and assignments stay exact (discrete).
        rtol = _REPORTING_FIELD_RTOL if reporting and actual.dtype.kind in "fc" else None
        assert_matches(actual, expected, err_msg=label, rtol=rtol)

    for i, (actual, expected) in enumerate(zip(candidate[:4], control[:4])):
        compare(actual, expected, f"result[{i}]", reporting=False)
    for c, (actual_class, expected_class) in enumerate(zip(candidate[4], control[4])):
        for i, (actual, expected) in enumerate(zip(actual_class, expected_class)):
            if actual is None or expected is None:
                assert actual is expected, f"class[{c}][{i}]"
                continue
            compare(actual, expected, f"class[{c}][{i}]", reporting=False)
    assert set(candidate[5]) == set(control[5])
    for key, expected in control[5].items():
        actual = candidate[5][key]
        if isinstance(expected, np.ndarray):
            compare(actual, expected, f"full_stats[{key!r}]", reporting=True)


def _run_padding_pair(monkeypatch):
    """The unpadded control and the padded candidate, in this process."""

    from relax.scoring import significance

    args, kwargs = _significance_call(monkeypatch)
    monkeypatch.setenv("RELAX_COARSE_PAD_FINAL_IMAGE_BATCH", "0")
    control = significance._compute_k_class_significance_batched(*args, **kwargs)
    monkeypatch.setenv("RELAX_COARSE_PAD_FINAL_IMAGE_BATCH", "1")
    candidate = significance._compute_k_class_significance_batched(*args, **kwargs)
    return candidate, control


def test_coarse_pad_env_flag_preserves_every_significance_output(monkeypatch):
    """The padded tail batch must reproduce the unpadded outputs within the default band.

    CPU-only: on a GPU the padding changes the coarse reduction's shape, and
    the reporting fields move inside the wider null band the GPU sibling below asserts.
    """

    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only contract; the GPU band is the sibling test")
    candidate, control = _run_padding_pair(monkeypatch)
    _assert_significance_results_match(candidate, control)


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="the null band is a GPU measurement"
)
def test_coarse_pad_env_flag_stays_inside_the_null_band_on_gpu(monkeypatch):
    """On a GPU the padded path must stay inside P3-B's measured floor."""

    candidate, control = _run_padding_pair(monkeypatch)
    _assert_significance_results_within_null_band(candidate, control)


def test_coarse_pad_env_flag_gives_every_batch_one_image_extent(monkeypatch):
    """With the flag on, preprocessing sees ``image_batch_size`` rows every time."""

    from relax.scoring import pass1_operands, significance

    args, kwargs = _significance_call(monkeypatch, n_classes=1)
    original = pass1_operands._process_relion_exact_coarse_half_image
    seen = []

    def record(experiment_dataset, batch, *rest, **batch_kwargs):
        seen.append(int(np.asarray(batch).shape[0]))
        return original(experiment_dataset, batch, *rest, **batch_kwargs)

    monkeypatch.setattr(pass1_operands, "_process_relion_exact_coarse_half_image", record)

    monkeypatch.setenv("RELAX_COARSE_PAD_FINAL_IMAGE_BATCH", "0")
    significance._compute_k_class_significance_batched(*args, **kwargs)
    unpadded = list(seen)

    seen.clear()
    monkeypatch.setenv("RELAX_COARSE_PAD_FINAL_IMAGE_BATCH", "1")
    significance._compute_k_class_significance_batched(*args, **kwargs)
    padded = list(seen)

    assert unpadded == [3, 3, 1]
    assert padded == [3, 3, 3]
    assert len(set(padded)) == 1


def test_preprocess_batch_jitted_elementwise_matches_eager():
    """The jitted elementwise chain matches the eager one within the float band."""

    from relax.helpers import preprocessing

    rng = np.random.default_rng(3)
    n_images, n_half_pixels, n_trans = 4, 9, 3
    processed = jnp.asarray(
        rng.standard_normal((n_images, n_half_pixels))
        + 1j * rng.standard_normal((n_images, n_half_pixels)),
        dtype=jnp.complex64,
    )
    ctf = jnp.asarray(rng.uniform(0.3, 1.7, (n_images, n_half_pixels)), dtype=jnp.float32)
    noise = jnp.asarray(rng.uniform(0.5, 2.0, n_half_pixels), dtype=jnp.float32)
    phases = jnp.asarray(
        rng.standard_normal((n_trans, n_half_pixels))
        + 1j * rng.standard_normal((n_trans, n_half_pixels)),
        dtype=jnp.complex64,
    )
    weights = jnp.asarray(rng.uniform(1.0, 2.0, n_half_pixels), dtype=jnp.float32)

    common = dict(
        score_complex_dtype=jnp.complex64,
        score_real_dtype=jnp.float32,
        norm_real_dtype=None,
    )
    eager = preprocessing._preprocess_batch_elementwise(
        processed, ctf, noise, phases, weights, **common
    )
    jitted = preprocessing._preprocess_batch_elementwise_jit(
        processed, ctf, noise, phases, weights, **common
    )
    assert len(eager) == len(jitted) == 4
    for actual, expected in zip(jitted, eager):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert_matches(np.asarray(actual), np.asarray(expected))


def test_collate_list_batch_makes_no_program_and_same_array():
    """``_collate_batch_to_jax`` must stop compiling one program per list length."""

    import jax._src.dispatch as dispatch
    from recovar.data_io.image_backends import _collate_batch_to_jax

    batches = [
        [int(value) for value in range(n)] for n in (5, 6, 7)
    ]
    expected = [jnp.asarray(batch) for batch in batches]

    original = dispatch.xla_primitive_callable
    calls = []

    def counting(*args, **kwargs):
        calls.append(args[0] if args else None)
        return original(*args, **kwargs)

    dispatch.xla_primitive_callable = counting
    try:
        actual = [_collate_batch_to_jax(batch) for batch in batches]
    finally:
        dispatch.xla_primitive_callable = original

    assert calls == [], f"collation compiled {len(calls)} eager programs: {calls}"
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype and got.shape == want.shape
        assert got.weak_type == want.weak_type
        assert_matches(np.asarray(got), np.asarray(want))


def test_collate_keeps_device_arrays_on_device():
    """A list of device arrays must not be pulled back through the host."""

    from recovar.data_io.image_backends import _collate_batch_to_jax

    batch = [jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(2.0, dtype=jnp.float32)]
    collated = _collate_batch_to_jax(batch)
    assert isinstance(collated, jax.Array)
    assert_matches(np.asarray(collated), np.asarray([1.0, 2.0], dtype=np.float32))


@pytest.mark.parametrize(
    "name, reader, unset",
    [
        (
            "RELAX_COARSE_PAD_FINAL_IMAGE_BATCH",
            "relax.scoring.significance:_coarse_pad_final_image_batch_enabled",
            True,  # on by default since the K=1 resident flip
        ),
        (
            "RELAX_EM_JIT_STAGE_GLUE",
            "relax.helpers.preprocessing:jit_stage_glue_enabled",
            False,  # the K=1 entry points set it; the shared default stays off
        ),
    ],
)
def test_stage_glue_flags_fail_closed_on_bad_tokens(monkeypatch, name, reader, unset):
    import importlib

    module_name, attribute = reader.split(":")
    read = getattr(importlib.import_module(module_name), attribute)

    monkeypatch.delenv(name, raising=False)
    assert read() is unset
    monkeypatch.setenv(name, "1")
    assert read() is True
    monkeypatch.setenv(name, "0")
    assert read() is False
    monkeypatch.setenv(name, "maybe")
    with pytest.raises(ValueError, match=name):
        read()


def test_source_star_ctf_pads_with_the_rest_of_the_coarse_batch():
    """The firstiter-CC operand rebuilt from the source STAR must pad too.

    ``RELAX_COARSE_PAD_FINAL_IMAGE_BATCH`` pads the coarse batch's images,
    CTF parameters, pre-shifts, corrections and scales, but the normalized-CC
    tree-rescore branch of ``_compute_k_class_significance_batched`` rebuilds one
    more per-image operand from the source STAR at the *unpadded* ``indices``.
    With the flag on, the first iteration of a K=1 end-to-end died there:

        TypeError: div got incompatible shapes for broadcasting:
                   (250, 1), (216, 33024)

    (250 = the padded batch scale, 216 = the half set's last batch). The branch
    itself needs RELION CUDA preprocessing and a real source STAR, so the
    end-to-end is its regression check; this test pins the padding contract the
    fix relies on: the repeat-padded operand keeps every live row, repeats row
    zero, and broadcasts against the padded per-image scale.
    """

    from relax.relion.relion_coarse_operands import _repeat_pad_batch_axis
    from relax.sparse_pass2.sparse_pass2_scoring import (
        _relion_cuda_pixel_correction_from_rfloat_ctf,
    )

    actual, padded_size, pixels = 216, 250, 12
    rng = np.random.default_rng(20260920)
    ctf = jnp.asarray(rng.uniform(0.5, 1.5, (actual, pixels)), dtype=jnp.float32)
    padded = jnp.asarray(_repeat_pad_batch_axis(ctf, padded_size))

    assert padded.shape == (padded_size, pixels)
    assert_matches(np.asarray(padded[:actual]), np.asarray(ctf))
    assert_matches(
        np.asarray(padded[actual:]),
        np.repeat(np.asarray(ctf[:1]), padded_size - actual, axis=0),
    )

    scale = jnp.asarray(rng.uniform(0.9, 1.1, (padded_size, 1)), dtype=jnp.float32)
    correction = _relion_cuda_pixel_correction_from_rfloat_ctf(scale, padded)
    assert correction.shape == (padded_size, pixels)
    # The live rows are the unpadded answer: padding may not move a science row.
    unpadded_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
        scale[:actual], ctf
    )
    assert_matches(
        np.asarray(correction[:actual]), np.asarray(unpadded_correction)
    )
