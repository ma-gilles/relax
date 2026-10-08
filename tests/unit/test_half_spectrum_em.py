"""Half-spectrum scoring equivalence tests (Phase 1 of RELION-parity plan).

Verifies that the half-spectrum weights, shell binning and block scores reproduce the
full-spectrum reference. The dense ``run_em`` engine these once covered end to end was
removed on 2026-10-03; its passes are checked against the RELION E-step reference in
``test_resident_relion_reference``.

Tests:
1. test_half_inner_product_correctness: weighted half-spectrum dot product == full
2. test_e_step_half_matches_full: half-spectrum E-step scores match full-spectrum
"""


import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
import recovar.core.fourier_transform_utils as ftu
from helpers.dense_block_scores import _e_step_block_scores
from helpers.em_arrays import _hermitian_volume, _raw_real_image_2d
from recovar import core
from recovar.core.configs import ForwardModelConfig

from relax.helpers.half_spectrum import (
    bin_shell_values_jax,
    bin_shell_values_np,
    make_half_image_weights,
    make_relion_noise_shell_indices_half,
    make_scoring_half_image_weights,
    make_shell_indices_half,
)
from relax.helpers.preprocessing import preprocess_batch as _preprocess_batch
from relax.helpers.projection import compute_projections_block as _compute_projections_block
from relax.scoring.scoring import _update_logsumexp

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Test constants -- small sizes for unit tests (no GPU required for tiny data)
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512
H, W = IMAGE_SHAPE
N_HALF = H * (W // 2 + 1)  # 8 * 5 = 40
N_FULL = H * W  # 8 * 8 = 64
N_ROTATIONS = 5
N_TRANSLATIONS = 3
N_IMAGES = 4
SEED = 42


def test_relion_scoring_half_weights_drop_redundant_negative_kx0_rows():
    weights = np.asarray(
        make_scoring_half_image_weights(IMAGE_SHAPE, relion_half_sum=True)
    ).reshape(IMAGE_SHAPE[0], IMAGE_SHAPE[1] // 2 + 1)

    expected = np.ones_like(weights)
    expected[1 : IMAGE_SHAPE[0] // 2, 0] = 0.0
    assert_matches(weights, expected)

    # RELION treats the Nyquist boundary row as +N/2 and keeps it.
    assert weights[0, 0] == 1.0
    assert weights[IMAGE_SHAPE[0] // 2, 0] == 1.0


def test_relion_normalized_cc_half_weights_keep_rectangular_x0_rows():
    weights = np.asarray(
        make_scoring_half_image_weights(
            (8, 8),
            relion_half_sum=True,
            exclude_relion_redundant_x0=False,
        )
    ).reshape(8, 5)

    assert_matches(weights, np.ones((8, 5), dtype=np.float32))


def test_non_relion_scoring_half_weights_keep_hermitian_multiplicity():
    actual = make_scoring_half_image_weights(IMAGE_SHAPE, relion_half_sum=False)
    assert_matches(np.asarray(actual), np.asarray(make_half_image_weights(IMAGE_SHAPE)))


@pytest.mark.parametrize(
    "image_shape",
    [(4, 6), (6, 4), (7, 7), (8, 8), (127, 127), (128, 128)],
)
def test_host_planned_shell_geometry_is_byte_exact_to_jax_reference(image_shape):
    expected = np.asarray(
        ftu.get_grid_of_radial_distances_real(
            image_shape,
            voxel_size=1,
            scaled=False,
            frequency_shift=0,
            rounded=True,
        ),
        dtype=np.int32,
    ).reshape(-1)
    actual = np.asarray(make_shell_indices_half(image_shape), dtype=np.int32)

    assert_matches(actual, expected)

    height, width = image_shape
    half_width = width // 2 + 1
    n_shells = height // 2 + 1
    coords = np.asarray(
        ftu.get_k_coordinate_of_each_pixel_half(
            image_shape,
            voxel_size=1,
            scaled=False,
        ),
    ).reshape(height, half_width, 2)
    kx = np.rint(coords[..., 0]).astype(np.int32)
    ky = np.rint(coords[..., 1]).astype(np.int32)
    expected_grid = expected.reshape(height, half_width)
    vertical_nyquist = (height % 2 == 0) & (ky == -(height // 2))
    redundant_x0 = (kx == 0) & (ky < 0) & ~vertical_nyquist
    expected_noise_shells = np.where(
        (expected_grid < n_shells) & ~redundant_x0,
        expected_grid,
        n_shells,
    ).reshape(-1)

    assert_matches(
        np.asarray(make_relion_noise_shell_indices_half(image_shape)),
        expected_noise_shells,
    )


def test_relion_shell_binning_drops_sentinel_indices_under_jit():
    shell_count = IMAGE_SHAPE[0] // 2 + 1
    shell_indices = np.asarray(make_relion_noise_shell_indices_half(IMAGE_SHAPE), dtype=np.int32)
    assert np.any(shell_indices == shell_count)

    values = np.arange(1, shell_indices.size + 1, dtype=np.float32)
    expected = bin_shell_values_np(values, shell_indices, shell_count)

    bin_jit = jax.jit(lambda vals, inds: bin_shell_values_jax(vals, inds, shell_count))
    actual = np.asarray(bin_jit(jnp.asarray(values), jnp.asarray(shell_indices)))

    assert_matches(actual, expected)


def test_shell_binning_maps_arbitrary_out_of_range_indices_to_drop_bin():
    shell_count = 5
    shell_indices = np.asarray([-1, 0, 2, 4, 5, 99], dtype=np.int32)
    values = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32)

    actual_jax = np.asarray(
        bin_shell_values_jax(jnp.asarray(values), jnp.asarray(shell_indices), shell_count)
    )
    actual_np = bin_shell_values_np(values, shell_indices, shell_count)

    assert_matches(actual_jax, np.asarray([2, 0, 3, 0, 4], dtype=np.float32))
    assert_matches(actual_np, np.asarray([2, 0, 3, 0, 4], dtype=np.float32))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hermitian_image_2d(image_shape, seed=42):
    """Generate a Hermitian-symmetric 2D spectrum (DFT of real data), centered."""
    ft = np.fft.fftshift(np.fft.fft2(_raw_real_image_2d(image_shape, seed=seed)))
    return jnp.array(ft, dtype=jnp.complex64)


def _make_rotations(n, seed=42):
    """Generate n rotation matrices via QR decomposition."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return jnp.array(q, dtype=jnp.float32)


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    """CTF that returns ones (identity)."""
    if half_image:
        h, w = image_shape if image_shape is not None else IMAGE_SHAPE
        sz = h * (w // 2 + 1)
    else:
        sz = IMAGE_SIZE
    return jnp.ones((params.shape[0], sz), dtype=jnp.float32)


def _raw_real_process(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _constant_half_noise_variance(noise_variance):
    noise_variance = jnp.asarray(noise_variance)
    if noise_variance.shape[-1] == N_HALF:
        return noise_variance
    values = np.asarray(noise_variance).reshape(-1)
    if values.size != IMAGE_SIZE or not np.allclose(values, values[0]):
        raise ValueError("test helper only supports scalar full-image noise variance")
    return jnp.full((N_HALF,), values[0], dtype=noise_variance.dtype)


def _preprocess_test_batch(
    dataset,
    batch_data,
    ctf_params,
    noise_variance,
    translations,
    config,
    *,
    score_with_masked_images=False,
):
    return _preprocess_batch(
        dataset,
        batch_data,
        ctf_params,
        _constant_half_noise_variance(noise_variance),
        translations,
        config,
        score_with_masked_images=score_with_masked_images,
        score_complex_dtype=jnp.complex64,
        score_real_dtype=jnp.float32,
        norm_real_dtype=jnp.float64,
    )


def _compute_jax_projections_block(volume, rotations, image_shape, volume_shape, disc_type):
    return _compute_projections_block(
        volume,
        rotations,
        image_shape,
        volume_shape,
        disc_type,
        relion_texture_interp=False,
    )


class MockDataset:
    """Minimal dataset for equivalence tests."""

    def __init__(self, rng):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = N_IMAGES
        self.n_units = N_IMAGES
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((N_IMAGES, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)

        self._images = np.zeros((N_IMAGES, *IMAGE_SHAPE), dtype=np.float32)
        for i in range(N_IMAGES):
            self._images[i] = _raw_real_image_2d(IMAGE_SHAPE, seed=rng.integers(10000))

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        _ = kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        for chunk_start in range(0, len(indices), max(1, batch_size)):
            chunk_end = min(chunk_start + max(1, batch_size), len(indices))
            idx = np.asarray(indices[chunk_start:chunk_end])
            yield (
                jnp.asarray(self._images[idx]),
                None,
                None,
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def mock_dataset(rng):
    return MockDataset(rng)


@pytest.fixture
def seeded_inputs(rng, mock_dataset):
    """Deterministic inputs for equivalence tests."""
    volume = _hermitian_volume(VOLUME_SHAPE, seed=42)
    rotations = _make_rotations(N_ROTATIONS, seed=12)
    translations = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    noise_variance = jnp.ones(IMAGE_SIZE, dtype=jnp.float32)

    config = ForwardModelConfig.from_dataset(
        mock_dataset,
        disc_type="linear_interp",
        process_fn=mock_dataset.process_images,
    )

    return {
        "volume": volume,
        "rotations": rotations,
        "translations": translations,
        "noise_variance": noise_variance,
        "config": config,
        "dataset": mock_dataset,
    }


# ===========================================================================
# Test 1: Half-spectrum inner product correctness
# ===========================================================================


class TestHalfInnerProductCorrectness:
    """Verify that weighted half-spectrum inner product matches full inner product.

    This is the fundamental identity that makes all half-spectrum GEMMs correct.
    """

    def test_hermitian_pair(self):
        """Re<a, b>_full == Re[sum(conj(a_half) * w * b_half)] for Hermitian data."""
        a_2d = _hermitian_image_2d(IMAGE_SHAPE, seed=10)
        b_2d = _hermitian_image_2d(IMAGE_SHAPE, seed=20)

        a_flat = a_2d.reshape(N_FULL)
        b_flat = b_2d.reshape(N_FULL)

        # Full inner product
        ip_full = jnp.sum(jnp.conj(a_flat) * b_flat).real

        # Half inner product with weights
        a_half = ftu.full_image_to_half_image(a_flat[None, :], IMAGE_SHAPE).reshape(-1)
        b_half = ftu.full_image_to_half_image(b_flat[None, :], IMAGE_SHAPE).reshape(-1)
        w = make_half_image_weights(IMAGE_SHAPE)
        ip_half = jnp.sum(jnp.conj(a_half) * w * b_half).real

        np.testing.assert_allclose(float(ip_full), float(ip_half), rtol=1e-5)

    def test_matrix_inner_product(self):
        """Batched: conj(A) @ B^T matches conj(A_half) @ (B_half * w)^T."""
        rng = np.random.default_rng(42)
        n_a, n_b = 4, 5

        A = jnp.stack([_hermitian_image_2d(IMAGE_SHAPE, seed=i).reshape(-1) for i in range(n_a)])
        B = jnp.stack([_hermitian_image_2d(IMAGE_SHAPE, seed=100 + i).reshape(-1) for i in range(n_b)])

        # Full GEMM
        full_result = (jnp.conj(A) @ B.T).real  # (n_a, n_b)

        # Half GEMM with weights absorbed into B
        w = make_half_image_weights(IMAGE_SHAPE)
        A_half = ftu.full_image_to_half_image(A, IMAGE_SHAPE)
        B_half = ftu.full_image_to_half_image(B, IMAGE_SHAPE)
        B_half_weighted = B_half * w
        half_result = (jnp.conj(A_half) @ B_half_weighted.T).real  # (n_a, n_b)

        np.testing.assert_allclose(
            np.array(full_result),
            np.array(half_result),
            rtol=1e-5,
            atol=1e-4,
            err_msg="Batched half-spectrum GEMM does not match full GEMM",
        )

    def test_weight_values(self):
        """Verify make_half_image_weights returns correct values."""
        w = make_half_image_weights(IMAGE_SHAPE)
        w_2d = w.reshape(H, W // 2 + 1)

        # DC column (0) should be 1
        assert_matches(np.array(w_2d[:, 0]), np.ones(H))
        # Nyquist column (-1) should be 1
        assert_matches(np.array(w_2d[:, -1]), np.ones(H))
        # Interior columns should be 2
        if W // 2 + 1 > 2:
            assert_matches(
                np.array(w_2d[:, 1:-1]),
                2.0 * np.ones((H, W // 2 - 1)),
            )


# ===========================================================================
# Test 2: E-step half matches full
# ===========================================================================


class TestEStepHalfMatchesFull:
    """Verify that the half-spectrum E-step produces the same scores as the
    full-spectrum reference (compute_dot_products + norm term)."""

    def test_scores_match(self, seeded_inputs):
        """E-step scores from half-spectrum path match full-spectrum scores."""
        s = seeded_inputs
        config = s["config"]
        volume = s["volume"]
        rotations = s["rotations"]
        translations = s["translations"]
        noise_variance = s["noise_variance"]
        ds = s["dataset"]

        n_images = ds.n_units
        n_trans = translations.shape[0]

        # Get batch data
        batch_data = jnp.asarray(ds._images)
        ctf_params = jnp.asarray(ds.CTF_params)

        # === FULL-SPECTRUM reference (from em.core) ===
        from oracles import core as em_core

        # Full-spectrum projections
        proj_full = core.slice_volume(volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=False)
        proj_abs2_full = jnp.abs(proj_full) ** 2

        # Cross-term via existing function
        cross_term = em_core.compute_dot_products(
            config, proj_full, batch_data, translations, ctf_params, noise_variance
        )
        # Norm-term
        norm_term = em_core.compute_ctf_projection_norms(config, proj_abs2_full, ctf_params, noise_variance)
        # Full residual: scores = -0.5 * (cross + norm)
        scores_full = -0.5 * (cross_term + norm_term[..., None])

        # === HALF-SPECTRUM path ===
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_test_batch(
            ds,
            batch_data,
            ctf_params,
            noise_variance,
            translations,
            config,
        )

        proj_half, proj_abs2_half = _compute_jax_projections_block(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp"
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        proj_half_weighted = proj_half * half_weights
        proj_abs2_weighted = proj_abs2_half * half_weights

        scores_half = _e_step_block_scores(
            shifted_half,
            ctf2_over_nv_half,
            proj_half_weighted,
            proj_abs2_weighted,
            n_images,
            n_trans,
        )

        score_offset = 0.5 * np.asarray(batch_norm).reshape(n_images, 1, 1)
        np.testing.assert_allclose(
            np.array(scores_full) + score_offset,
            np.array(scores_half),
            atol=1e-3,
            rtol=5e-4,
            err_msg="E-step scores differ between half-spectrum and full-spectrum paths",
        )

    def test_probabilities_match(self, seeded_inputs):
        """Probabilities (softmax of scores) match between half and full paths."""
        s = seeded_inputs
        config = s["config"]
        volume = s["volume"]
        rotations = s["rotations"]
        translations = s["translations"]
        noise_variance = s["noise_variance"]
        ds = s["dataset"]

        n_images = ds.n_units
        n_trans = translations.shape[0]
        n_rot = rotations.shape[0]

        batch_data = jnp.asarray(ds._images)
        ctf_params = jnp.asarray(ds.CTF_params)

        # Full-spectrum scores
        from oracles import core as em_core

        proj_full = core.slice_volume(volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=False)
        proj_abs2_full = jnp.abs(proj_full) ** 2
        cross_term = em_core.compute_dot_products(
            config, proj_full, batch_data, translations, ctf_params, noise_variance
        )
        norm_term = em_core.compute_ctf_projection_norms(config, proj_abs2_full, ctf_params, noise_variance)
        scores_full = -0.5 * (cross_term + norm_term[..., None])
        # Softmax
        scores_flat = scores_full.reshape(n_images, -1)
        log_Z_full = jax.scipy.special.logsumexp(scores_flat, axis=1)
        probs_full = jnp.exp(scores_full - log_Z_full[:, None, None])

        # Half-spectrum scores
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_test_batch(
            ds,
            batch_data,
            ctf_params,
            noise_variance,
            translations,
            config,
        )
        proj_half, proj_abs2_half = _compute_jax_projections_block(
            volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp"
        )
        half_weights = make_half_image_weights(IMAGE_SHAPE)
        proj_half_weighted = proj_half * half_weights
        proj_abs2_weighted = proj_abs2_half * half_weights

        scores_half = _e_step_block_scores(
            shifted_half,
            ctf2_over_nv_half,
            proj_half_weighted,
            proj_abs2_weighted,
            n_images,
            n_trans,
        )
        scores_h_flat = scores_half.reshape(n_images, -1)
        log_Z_half = jax.scipy.special.logsumexp(scores_h_flat, axis=1)
        probs_half = jnp.exp(scores_half - log_Z_half[:, None, None])

        np.testing.assert_allclose(
            np.array(probs_full),
            np.array(probs_half),
            atol=1e-5,
            err_msg="Probabilities differ between half-spectrum and full-spectrum paths",
        )


# ===========================================================================
# Test 3: M-step half matches full
# ===========================================================================


# ===========================================================================
# Test 4: Full iteration half matches (run_em)
# ===========================================================================


# ===========================================================================
# Test 5: make_half_image_weights shape and dtype
# ===========================================================================


class TestMakeHalfImageWeights:
    """Verify make_half_image_weights produces correct shape and values."""

    @pytest.mark.parametrize("shape", [(4, 4), (8, 8), (16, 16), (64, 64), (128, 128)])
    def test_shape(self, shape):
        """Output shape is (H * (W//2 + 1),)."""
        H, W = shape
        w = make_half_image_weights(shape)
        expected_len = H * (W // 2 + 1)
        assert w.shape == (expected_len,), f"Expected ({expected_len},), got {w.shape}"

    @pytest.mark.parametrize("shape", [(4, 4), (8, 8), (128, 128)])
    def test_sum(self, shape):
        """Sum of weights should equal H * W (total pixels in full image)."""
        H, W = shape
        w = make_half_image_weights(shape)
        expected_sum = H * W
        np.testing.assert_allclose(float(jnp.sum(w)), expected_sum, rtol=1e-5)

    def test_dtype(self):
        """Weights should be float32."""
        w = make_half_image_weights(IMAGE_SHAPE)
        assert w.dtype == jnp.float32


# ===========================================================================
# Test 6: Streaming logsumexp correctness
# ===========================================================================


class TestStreamingLogsumexp:
    """Verify that streaming logsumexp over blocks matches direct computation."""

    def test_streaming_matches_direct(self):
        """Streaming logsumexp over 3 blocks matches jax.scipy.special.logsumexp."""
        rng = np.random.default_rng(42)
        n_images = 4
        n_rot_per_block = 3
        n_trans = 2

        # Create 3 blocks of scores
        blocks = [
            jnp.array(rng.standard_normal((n_images, n_rot_per_block, n_trans)).astype(np.float32)) for _ in range(3)
        ]

        # Direct: concatenate all and compute
        all_scores = jnp.concatenate([b.reshape(n_images, -1) for b in blocks], axis=1)
        log_Z_direct = jax.scipy.special.logsumexp(all_scores, axis=1)

        # Streaming
        max_s = jnp.full(n_images, -jnp.inf)
        sum_exp = jnp.zeros(n_images)
        for block in blocks:
            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, block)
        log_Z_streaming = max_s + jnp.log(sum_exp)

        np.testing.assert_allclose(
            np.array(log_Z_direct),
            np.array(log_Z_streaming),
            atol=1e-5,
            err_msg="Streaming logsumexp does not match direct computation",
        )
