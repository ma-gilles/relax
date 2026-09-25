"""Fresh K=1 fine diff2 is scored in RELION's native FFT units.

RELION evaluates the fine-pass diff2 on unnormalised FFT coefficients. RECOVAR's
shifted image and projected reference carry an extra ``N**2`` and its score
``corr_img`` an extra ``N**-4``. The factors cancel over the reals, and in
float32 whenever ``N**2`` is a power of two, but not otherwise. Final Q scored
the fresh K=1 exact-Gaussian pass in native units
(recovar ``RECOVAR_K1_RELION_NATIVE_FINE_SCORE_UNITS``, default on,
aa0eccbfd4); recovar 06d5dbea1 restored it and this pins that behaviour as the
single production path of the compact engine.

Float comparisons are relative bands, never bitwise. Where two float paths
agree by construction (one correctly rounded binary64 division, or a
power-of-two box where the unit factors are exact), the band is
``_EXACT_BY_CONSTRUCTION_RTOL``, below one float32 ULP; these comparisons were
measured bit-identical when the test was written.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import recovar.core.fourier_transform_utils as ftu
from helpers.em_arrays import _hermitian_volume, _raw_real_image_2d

from relax.helpers.half_spectrum import make_shell_indices_half
from relax.sparse_pass2.sparse_pass2_scoring import (
    _relion_cuda_corr_img_from_native_noise_variance,
    _relion_cuda_native_corr_img_from_noise_variance,
    _relion_native_fine_units,
    _score_pass2_bucket_relion_gpu_diff2_raw,
)
from helpers.float_compare import assert_matches

# Below one float32 ULP (2**-23 ~ 1.19e-7): tight enough to catch a different
# rounding path at most values, without asserting bitwise equality.
_EXACT_BY_CONSTRUCTION_RTOL = 1.0e-7


def _random_complex(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def _divide_correctly_rounded(values, fft_size):
    values = np.asarray(values, dtype=np.complex64)
    real = (values.real.astype(np.float64) / np.float64(fft_size)).astype(np.float32)
    imag = (values.imag.astype(np.float64) / np.float64(fft_size)).astype(np.float32)
    return (real + np.complex64(1j) * imag).astype(np.complex64)


@pytest.mark.parametrize("image_size", [6, 8, 300])
def test_score_corr_img_is_the_native_corr_img_scaled_once(image_size):
    rng = np.random.default_rng(image_size)
    n_half = image_size * (image_size // 2 + 1)
    noise_variance = rng.uniform(0.5, 2.0, (1, n_half))
    ctf = rng.uniform(-1.0, 1.0, (3, n_half))
    scale = rng.uniform(0.8, 1.2, (3, 1)).astype(np.float32)
    shape = (image_size, image_size)

    native = _relion_cuda_native_corr_img_from_noise_variance(noise_variance, ctf, shape, scale)
    canonical = _relion_cuda_corr_img_from_native_noise_variance(noise_variance, ctf, shape, scale)

    assert native.dtype == jnp.float32
    expected = (np.asarray(native, dtype=np.float64) / np.float64(image_size) ** 4).astype(np.float32)
    # Measured exact: the same native operand divided once in binary64.
    np.testing.assert_allclose(np.asarray(canonical), expected, rtol=_EXACT_BY_CONSTRUCTION_RTOL, atol=0.0)

    # Each part is divided once and correctly rounded, whatever the device or
    # fusion context; XLA's complex64 division by a real scalar is neither.
    values = _random_complex(rng, (3, n_half)) * np.float32(10.0) ** rng.uniform(-3, 5, (3, n_half)).astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(_relion_native_fine_units(values, image_size * image_size)),
        _divide_correctly_rounded(values, image_size * image_size),
        rtol=_EXACT_BY_CONSTRUCTION_RTOL,
        atol=0.0,
    )


def test_native_units_are_bitwise_inert_for_power_of_two_boxes():
    """Box sizes with N**2 a power of two score identically in either unit system."""

    rng = np.random.default_rng(20260923)
    image_size = 8
    fft_scale = np.float32(image_size * image_size)
    n_pixels = 13
    shifted_native = _random_complex(rng, (2, 3, n_pixels))
    proj_native = _random_complex(rng, (2, 4, n_pixels))
    corr_native = rng.uniform(1.0e5, 1.0e6, (2, n_pixels)).astype(np.float32)
    weights = np.ones(n_pixels, dtype=np.float32)

    native = _score_pass2_bucket_relion_gpu_diff2_raw(
        jnp.asarray(shifted_native), jnp.asarray(corr_native), jnp.asarray(proj_native), jnp.asarray(weights)
    )
    recovar_units = _score_pass2_bucket_relion_gpu_diff2_raw(
        jnp.asarray(shifted_native * fft_scale),
        jnp.asarray((corr_native.astype(np.float64) / np.float64(fft_scale) ** 2).astype(np.float32)),
        jnp.asarray(proj_native * fft_scale),
        jnp.asarray(weights),
    )
    # Measured exact: scaling by a power of two is exact in float32.
    np.testing.assert_allclose(
        np.asarray(recovar_units), np.asarray(native), rtol=_EXACT_BY_CONSTRUCTION_RTOL, atol=0.0
    )


def _nearest_relative_distance(values, candidates):
    """Relative distance of each value to its nearest candidate value."""

    values = np.asarray(values, dtype=np.complex128).reshape(-1)
    candidates = np.unique(np.asarray(candidates, dtype=np.complex128).reshape(-1))
    distance = np.abs(values[:, None] - candidates[None, :]).min(axis=1)
    return distance / np.maximum(np.abs(values), np.finfo(np.float32).tiny)


class _Captured(Exception):
    pass


class _NativeUnitsDataset:
    def __init__(self, image_size, n_images=2, seed=20260923):
        self.image_shape = (image_size, image_size)
        self.image_size = image_size * image_size
        self.grid_size = image_size
        self.volume_shape = (image_size,) * 3
        self.volume_size = image_size**3
        self.n_images = n_images
        self.n_units = n_images
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((n_images, 9), dtype=np.float32)
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)
        self.premultiplied_ctf = False
        rng = np.random.default_rng(seed)
        self._images = np.stack(
            [_raw_real_image_2d(self.image_shape, seed=int(rng.integers(10000))) for _ in range(n_images)]
        )

        class _Backend:
            relion_fourier_backend = "relion_cuda"

        class _ImageSource:
            backend = _Backend()

        self.image_source = _ImageSource()

    def ctf_evaluator(self, params, image_shape=None, voxel_size=None, *, half_image=False):
        h, w = self.image_shape
        size = h * (w // 2 + 1) if half_image else h * w
        return jnp.ones((params.shape[0], size), dtype=jnp.float32)

    def process_images(self, batch, apply_image_mask=False, **kwargs):
        images = jnp.asarray(batch)
        return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)

    def process_images_half(self, batch, apply_image_mask=False, **kwargs):
        images = jnp.asarray(batch)
        processed = ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)
        normalization = kwargs.get("relion_normalization_factors")
        if normalization is None:
            return processed
        return processed * jnp.asarray(normalization, dtype=jnp.float32)[:, None]

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        indices = np.arange(self.n_images) if indices is None else np.asarray(indices)
        for start in range(0, len(indices), max(1, batch_size)):
            idx = np.asarray(indices[start : start + max(1, batch_size)])
            yield (
                jnp.asarray(self._images[idx]),
                jnp.asarray(self.rotation_matrices[idx]),
                jnp.asarray(self.translations[idx]),
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)


def _capture_fresh_k1_fine_score_operands(monkeypatch, image_size, current_size):
    from relax.cuda import kernels as em_cuda_kernels
    from relax.relion import relion_ctf
    from relax.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod
    from relax.sparse_pass2 import sparse_pass2_policy, sparse_pass2_projection_blocks
    from relax.sparse_pass2.dispatch import compute_pass2_stats_sparse

    for name in (
        "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
        "RELAX_BPREF_CONTRIBUTION_DUMP_DIR",
        "RELAX_BPREF_MEMBERSHIP_DUMP_DIR",
        "RELAX_BPREF_ACCUMULATOR_DELTA_DUMP_DIR",
        "RELAX_PASS2_DUMP_DIR",
        "RELAX_SPARSE_PASS2_RESIDENT",
        sparse_pass2_policy._BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    dataset = _NativeUnitsDataset(image_size)
    shape = dataset.image_shape
    n_half = image_size * (image_size // 2 + 1)
    ctf_rfloat = np.linspace(0.2, 1.1, 2 * n_half, dtype=np.float64).reshape(2, n_half)
    monkeypatch.setattr(
        relion_ctf,
        "_relion_exact_ctf_half_from_source_star",
        lambda _dataset, indices, _shape: jnp.asarray(ctf_rfloat[np.asarray(indices)]),
    )
    monkeypatch.setattr(
        bucketed_mod,
        "_relion_cuda_score_translation_angles_if_available",
        lambda translations, _shape, **_kwargs: jnp.zeros((len(translations), 2), dtype=jnp.float32),
    )
    translated_inputs = []

    def fake_translate_score(images, angles, _pixel_indices, _shape):
        translated_inputs.append(np.asarray(images))
        return jnp.repeat(jnp.asarray(images), int(angles.shape[0]), axis=0)

    monkeypatch.setattr(em_cuda_kernels, "relion_translate_score_f32", fake_translate_score)
    monkeypatch.setattr(
        em_cuda_kernels,
        "relion_translate_bpref_f32",
        lambda images, weights, angles, _pixel_indices, _shape: jnp.repeat(
            jnp.asarray(images) * jnp.asarray(weights), int(angles.shape[0]), axis=0
        ),
    )

    projected_rows = []

    def fake_projector(_projector, rotations, image_shape, **kwargs):
        pixel_indices = kwargs.get("pixel_indices")
        n_pixels = n_half if pixel_indices is None else int(np.asarray(pixel_indices).size)
        row = (jnp.linspace(0.25, 1.25, n_pixels, dtype=jnp.float32) * (1.0 + 0.5j)).astype(jnp.complex64)
        projected_rows.append(np.asarray(row))
        projections = jnp.broadcast_to(row, (int(rotations.shape[0]), n_pixels))
        return projections, (jnp.abs(projections) ** 2 if kwargs.get("return_abs2", True) else None)

    monkeypatch.setattr(sparse_pass2_projection_blocks, "_compute_relion_projector_projections_block", fake_projector)
    captured = {}

    def capture(shifted, corr_img, proj, half_weights, *args, **kwargs):
        captured.update(shifted=np.asarray(shifted), corr_img=np.asarray(corr_img), proj=np.asarray(proj))
        raise _Captured

    monkeypatch.setattr(bucketed_mod, "_score_pass2_bucket_relion_gpu_diff2_raw", capture)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    with pytest.raises(_Captured):
        compute_pass2_stats_sparse(
            experiment_dataset=dataset,
            volume=_hermitian_volume(dataset.volume_shape, seed=20260923),
            mean_variance=jnp.ones(dataset.volume_size, dtype=jnp.float32),
            noise_variance=jnp.ones(dataset.image_size, dtype=jnp.float32),
            translations=jnp.zeros((1, 2), dtype=jnp.float32),
            significant_sample_indices=[np.asarray([0, 1], dtype=np.int32)] * 2,
            nside_level=0,
            disc_type="linear_interp",
            oversampling_order=0,
            current_size=current_size,
            half_spectrum_scoring=True,
            fine_rotations_override=fine_rotations,
            fine_rotation_parent_override=np.asarray([0, 1], dtype=np.int64),
            fine_translations_override=np.zeros((1, 2), dtype=np.float32),
            fine_translation_parent_override=np.asarray([0], dtype=np.int32),
            relion_x_half_mstep=True,
            relion_firstiter_winner_take_all=False,
            relion_exact_fine_gaussian=True,
            relion_projector_half=np.zeros((3, 3, 2), dtype=np.complex64),
            relion_projector_r_max=image_size // 2,
            preserve_bpref_particle_order=True,
            source_faithful_spectrum_norm=True,
        )
    return captured, translated_inputs, projected_rows, ctf_rfloat, shape


@pytest.mark.parametrize("current_size", [None, 4], ids=["full-box", "windowed"])
def test_fresh_k1_fine_diff2_receives_native_unit_operands(monkeypatch, current_size):
    image_size = 6
    captured, translated_inputs, projected_rows, ctf_rfloat, shape = _capture_fresh_k1_fine_score_operands(
        monkeypatch, image_size, current_size
    )
    fft_size = image_size * image_size
    n_score = captured["corr_img"].shape[-1]

    native_corr = np.array(
        _relion_cuda_native_corr_img_from_noise_variance(
            np.ones((1, ctf_rfloat.shape[1]), dtype=np.float32), ctf_rfloat, shape
        )
    )
    native_corr[:, np.asarray(make_shell_indices_half(shape)) == 0] = 0.0
    if current_size is None:
        assert n_score == ctf_rfloat.shape[1]
        expected_corr = native_corr
    else:
        from relax.helpers.fourier_window import make_fourier_window_spec

        window = make_fourier_window_spec(shape, current_size, ctf_rfloat.shape[1], square=False)
        expected_corr = native_corr[:, np.asarray(window.score_indices_np)]
    # The zero origin is a discrete mask and stays exact; the values are a
    # measured-exact band.
    assert_matches(captured["corr_img"] == 0.0, expected_corr == 0.0)
    np.testing.assert_allclose(captured["corr_img"], expected_corr, rtol=_EXACT_BY_CONSTRUCTION_RTOL, atol=0.0)

    # The fake translation repeats its input, so the shifted operand is the
    # corrected score input divided by N**2 once, in float32.
    translated = translated_inputs[-1]
    expected_shifted = _divide_correctly_rounded(translated, fft_size)
    np.testing.assert_allclose(
        captured["shifted"].reshape(expected_shifted.shape),
        expected_shifted,
        rtol=_EXACT_BY_CONSTRUCTION_RTOL,
        atol=0.0,
    )
    # The scored reference is a gather of the projector's pixels, each divided
    # by N**2 once; RECOVAR-unit values are N**2 larger and never match.
    native_projection_values = _divide_correctly_rounded(np.concatenate(projected_rows), fft_size)
    assert _nearest_relative_distance(captured["proj"], native_projection_values).max() <= (
        _EXACT_BY_CONSTRUCTION_RTOL
    )
    recovar_unit_values = np.concatenate(projected_rows)
    assert _nearest_relative_distance(captured["proj"], recovar_unit_values).min() > 0.5


@pytest.mark.parametrize(
    ("fresh", "image_size", "in_scope"),
    [(True, 300, False), (True, 384, False), (True, 256, True), (False, 300, True), (True, None, True)],
)
def test_resident_driver_leaves_non_power_of_two_fresh_passes_to_the_compact_engine(fresh, image_size, in_scope):
    from relax.sparse_pass2 import resident_pass2 as rp

    reason = rp.resident_pass2_out_of_scope_reason(
        relion_firstiter_score_mode="gaussian",
        relion_firstiter_winner_take_all=False,
        source_faithful_spectrum_norm=fresh,
        image_size=image_size,
    )
    if in_scope:
        assert reason is None
    else:
        assert reason is not None and "native-unit" in reason and str(image_size) in reason
