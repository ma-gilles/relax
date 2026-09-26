"""The device-resident pass 2 against the NumPy RELION E-step reference.

The reference (``tests/helpers/relion_estep_reference.py``) restates RELION's GPU fine pass
from the pass's inputs, sharing no code with relax's engines. These tests are the independent
check of the resident driver's conventions; before them the driver was pinned only against the
compact engine, which cannot validate a convention both engines share.

The fixture is the 8x8 production-shaped K=1 pass of ``test_resident_pass2_driver`` with a
RELION ``PPref`` projector, at a windowed current size and at the full box. Two settings keep
it free of float ties that are not conventions:

- The noise variance is 1e30 on the four axis pixels of radius ``current_size/2``. Those pixels
  sit exactly on the BackProjector sphere (and, at the box, the projector's), so whether a
  float32 rotation keeps them is a last-bit tie.
- ``RELAX_RELION_PROJECTOR_TEXTURE_INTERP=0``: the CUDA texture unit interpolates with 8-bit
  fixed-point weights, a hardware approximation the float64 reference does not model.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from helpers import relion_estep_reference as ref

pytestmark = pytest.mark.unit

N = 8
# The driver scores and reduces in float32; the reference is float64.
F32 = 1e-6


def _tie_free_noise(current_size: int, noise: float) -> np.ndarray:
    """Centered ``[ky, kx]`` noise variance with the four radius-``current_size/2`` axis pixels removed."""

    k = np.arange(N) - N // 2
    ky, kx = np.meshgrid(k, k, indexing="ij")
    r = current_size // 2
    tie = ((np.abs(kx) == r) & (ky == 0)) | ((np.abs(ky) == r) & (kx == 0))
    return np.where(tie, 1e30, noise).astype(np.float32)


def _ppref(volume_ft):
    """RELION's padded projector slab (padding 2, the box's radius) of a centered FFT volume."""

    import recovar.core.fourier_transform_utils as ftu

    from relax.relion.relion_projector_setup import reference_to_relion_projector_half_maps

    volume_real = np.asarray(ftu.get_idft3(np.asarray(volume_ft).reshape(N, N, N)).real, dtype=np.float64)
    halves, r_max = reference_to_relion_projector_half_maps(
        volume_real[None], current_size=N, padding_factor=2, interpolator=1, projector_setup_backend="jax"
    )
    return np.asarray(halves[0]), int(r_max)


def _fftw_images(dataset, *, masked=False):
    """The dataset's Fourier images in RELION's FFTW row order."""

    centered = np.asarray(dataset.process_images_half(dataset._images, apply_image_mask=masked), dtype=np.complex128)
    centered = centered.reshape(-1, N, N // 2 + 1)
    # RECOVAR's centered rows hold ky + N/2; the ky = -N/2 row is RELION's +N/2 row.
    return centered[:, [(ky + N // 2) % N for ky in ref.fftw_rows(N)]]


def _reference_pass(args, current_size, noise_full, **fields):
    """The reference pass on the driver's own inputs (images, grids, priors and groups)."""

    images = _fftw_images(args["experiment_dataset"])
    sigma2 = np.empty((N, N // 2 + 1))
    for row, ky in enumerate(ref.fftw_rows(N)):
        sigma2[row] = noise_full[(ky + N // 2) % N, (np.arange(N // 2 + 1) + N // 2) % N]
    return ref.Pass(
        images=images,
        ctf=np.ones(images.shape),
        sigma2=sigma2,
        fine_rotations=np.asarray(args["fine_rotations_override"], dtype=np.float64),
        rotation_parent=np.asarray(args["fine_rotation_parent_override"]),
        fine_translations=np.asarray(args["fine_translations_override"], dtype=np.float64),
        translation_parent=np.asarray(args["fine_translation_parent_override"]),
        translation_log_prior=(
            None
            if args["translation_log_prior"] is None
            else np.asarray(args["translation_log_prior"], dtype=np.float64)
        ),
        current_size=current_size,
        reconstruction_padding=int(args["reconstruction_padding_factor"]),
        adaptive_fraction=float(args["adaptive_fraction"]),
        group_ids=None if args.get("group_ids") is None else np.asarray(args["group_ids"]),
        **fields,
    )


def _case(current_size: int, noise: float, base_args=None, **fields):
    """A K=1 resident driver's arguments and the reference pass built from the same inputs.

    ``base_args`` is a driver fixture (default ``test_resident_pass2_driver``'s); ``fields``
    are extra reference ``Pass`` fields for the configuration it exercises.
    """

    from test_resident_pass2_driver import _driver_fixture_args

    args = _driver_fixture_args() if base_args is None else base_args
    half, r_max = _ppref(args["volume"])
    noise_full = _tie_free_noise(current_size, noise)
    args.update(
        current_size=current_size,
        relion_projector_half=jnp.asarray(half, dtype=jnp.complex64),
        relion_projector_r_max=r_max,
        noise_variance=jnp.asarray(noise_full.reshape(-1)),
    )
    dvp = args.get("scale_correction_data_vs_prior")
    reference = _reference_pass(
        args,
        current_size,
        noise_full,
        projector={"data": half.astype(np.complex128), "pad": 2, "r_max": r_max},
        coarse_support=list(args["significant_sample_indices"]),
        rotation_log_prior=np.asarray(args["rotation_log_prior"], dtype=np.float64),
        data_vs_prior=None if dvp is None else np.asarray(dvp),
        **fields,
    )
    return args, reference


def _k_class_case(n_classes: int, current_size: int, noise: float):
    """The K-class resident pass's arguments and the reference pass with one model per class."""

    from test_resident_k_class_pass2 import _k_class_args

    args, volumes, supports, priors = _k_class_args(n_classes)
    slabs = [_ppref(volumes[k]) for k in range(n_classes)]
    r_max = slabs[0][1]
    noise_full = _tie_free_noise(current_size, noise)
    args.update(
        current_size=current_size,
        relion_projector_half=jnp.asarray(np.stack([half for half, _ in slabs]), dtype=jnp.complex64),
        relion_projector_r_max=r_max,
        noise_variance=jnp.asarray(noise_full.reshape(-1)),
    )
    dvp = np.asarray(args["scale_correction_data_vs_prior"])
    classes = [
        ref.ClassModel(
            projector={"data": slabs[k][0].astype(np.complex128), "pad": 2, "r_max": r_max},
            coarse_support=list(supports[k]),
            rotation_log_prior=np.asarray(priors[k], dtype=np.float64),
            data_vs_prior=dvp[k],
        )
        for k in range(n_classes)
    ]
    reference = _reference_pass(
        args,
        current_size,
        noise_full,
        projector=classes[0].projector,
        coarse_support=classes[0].coarse_support,
        rotation_log_prior=classes[0].rotation_log_prior,
        classes=classes,
    )
    return args, volumes, supports, priors, reference


def _public_full(half: np.ndarray) -> np.ndarray:
    """RELION's ``(z, y, x_half)`` BPref array in relax's public full ``(x, y, z)`` flat layout."""

    c = half.shape[0] // 2
    full = np.zeros((half.shape[0],) * 3, dtype=half.dtype)
    full[:, :, c:] = half
    mirrored = half[::-1, ::-1, 1:][:, :, ::-1]
    full[:, :, :c] = np.conj(mirrored) if np.iscomplexobj(half) else mirrored
    return full.transpose(2, 1, 0).reshape(-1)


def _rel_l2(actual, desired) -> float:
    """Relative L2 distance; absolute when the reference is zero (a class with no retained mass)."""

    actual, desired = np.asarray(actual), np.asarray(desired)
    scale = float(np.linalg.norm(desired))
    return float(np.linalg.norm(actual - desired) / scale) if scale else float(np.linalg.norm(actual))


# ---------------------------------------------------------------------------
# CPU: the reference's own RELION rules
# ---------------------------------------------------------------------------


def test_significant_weight_is_relions_cumulative_crossing():
    # Ascending [0.1, 0.2, 0.3, 0.4]: cumulative [0.1, 0.3, 0.6, 1.0]. A threshold of 0.25
    # is crossed between 0.1 and 0.3, so RELION keeps 0.2 and above.
    weights = np.array([0.3, 0.1, 0.4, 0.2])
    assert ref.significant_weight(weights, adaptive_fraction=0.75) == pytest.approx(0.2)
    # Below the first cumulative value the index stays 0 and every weight is kept.
    assert ref.significant_weight(weights, adaptive_fraction=0.95) == pytest.approx(0.1)


def test_mresol_fine_drops_the_duplicated_half_column_and_the_corners():
    resol = ref.mresol_fine(8, 6)
    rows = ref.fftw_rows(8)
    for row, ky in enumerate(rows):
        for kx in range(5):
            radius = np.floor(np.hypot(kx, ky) + 0.5)
            expected = (kx == 0 and ky < 0) or radius > 3 or ky < -2 or ky > 3 or kx > 3
            assert (resol[row, kx] < 0) == expected, (ky, kx)


def test_hermitian_x0_enforcement_sums_each_pair_once():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(5, 5, 3)) + 1j * rng.normal(size=(5, 5, 3))
    weight = rng.uniform(size=(5, 5, 3))
    before_data, before_weight = data.copy(), weight.copy()
    ref.enforce_hermitian_x0(data, weight)
    c = 2
    for iz in range(-c, c + 1):
        for iy in range(-c, c + 1):
            if (iz, iy) == (0, 0):
                continue
            a, b = (iz + c, iy + c, 0), (-iz + c, -iy + c, 0)
            assert data[a] == pytest.approx(before_data[a] + np.conj(before_data[b]))
            assert weight[a] == pytest.approx(before_weight[a] + before_weight[b])
    assert data[c, c, 0] == before_data[c, c, 0]


def test_reference_posteriors_are_normalized_and_keep_the_adaptive_fraction():
    _, reference = _case(6, 200.0)
    for post in ref.posteriors(reference):
        assert post.weight.sum() == pytest.approx(1.0, rel=1e-12)
        assert post.weight[post.kept].sum() >= reference.adaptive_fraction - 1e-12


# ---------------------------------------------------------------------------
# GPU: the resident driver against the reference
# ---------------------------------------------------------------------------


def _gpu_available():
    if jax.default_backend() != "gpu":
        return False
    from recovar import cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    return bool(
        cuda_backproject.custom_cuda_requested()
        and em_cuda_kernels.sparse_pass2_segmented_supported()
        and em_cuda_kernels.relion_wavg_sequential_runtime_flat_rows_triplet_f32_supported()
        and em_cuda_kernels.relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32_supported()
    )


requires_resident_gpu = pytest.mark.skipif(not _gpu_available(), reason="every resident stage is a CUDA FFI target")


@pytest.fixture
def _resident_env(monkeypatch):
    monkeypatch.setenv("RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_SCALE_AA", "1")
    monkeypatch.setenv("RELAX_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY", "1")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES", "256,1024,4096")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_IMAGE_CAPACITIES", "4,16,64")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS", "128")
    monkeypatch.setenv("RELAX_RELION_PROJECTOR_TEXTURE_INTERP", "0")


@requires_resident_gpu
@pytest.mark.parametrize(
    "current_size, noise",
    [
        (6, 200.0),  # windowed, broad posteriors: the M-step sums see many hypotheses
        (8, 200.0),  # the full box: RELION's window still cuts the corners
        (6, 0.8),  # windowed, peaked posteriors
    ],
)
def test_resident_driver_is_relions_fine_pass(_resident_env, current_size, noise):
    """Scores, posterior, M-step accumulators and noise statistics against the reference.

    Measured on an H100 (job 14478445): scores and log-evidence agree to float32 rounding,
    |dlogZ| 1.4e-6 at noise 200 and 1.5e-3 at noise 0.8, where the diff2 sums reach 6000
    (one float32 ULP there is 5e-4); pmax 6.7e-7 and 2.6e-4; accumulators 1.2e-6 to 2.6e-6
    relative L2; noise, norm and scale sums at most 3.1e-7.
    """

    from relax.sparse_pass2 import resident_pass2 as rp

    args, reference = _case(current_size, noise)
    _assert_k1_pass(rp.compute_pass2_stats_resident(**args), reference)


def _assert_k1_pass(out, reference, *, posterior=True, scale_sums=True, best=None, stats_rtol=1e-6):
    """A K=1 driver result against the reference: discrete state, posterior, BPref and statistics."""

    posts = ref.posteriors(reference)
    mstep = ref.mstep(reference, posts)

    if best is None:
        best = [post.cells[np.argmax(post.log_weight)] for post in posts]
    assert_matches(np.asarray(out.best_rotation_indices), np.array([b[1] for b in best]))
    assert_matches(
        np.asarray(out.best_translations, dtype=np.float64),
        reference.fine_translations[np.array([b[2] for b in best])],
        rtol=F32,
    )

    log_z = np.array([np.logaddexp.reduce(post.log_weight) for post in posts])
    best_score = np.array([post.log_weight.max() for post in posts])
    stats = out.relion_stats
    # A posterior weight's relative error is its diff2's absolute error: float32 rounding of a
    # sum as large as the image's |log Z|.
    posterior_atol = 4.0 * float(np.finfo(np.float32).eps) * float(np.max(np.abs(log_z)))
    if posterior:
        assert_matches(np.asarray(stats.log_evidence_per_image), log_z, rtol=F32)
        assert_matches(np.asarray(stats.best_log_score_per_image), best_score, rtol=F32)
        np.testing.assert_allclose(
            np.asarray(stats.max_posterior_per_image, dtype=np.float64),
            [post.weight.max() for post in posts],
            rtol=0.0,
            atol=posterior_atol,
        )
    if posterior:
        np.testing.assert_allclose(
            np.asarray(stats.rotation_posterior_sums),
            ref.rotation_posterior_sums(reference, posts),
            rtol=0.0,
            atol=posterior_atol * len(posts),
        )
    else:
        assert _rel_l2(stats.rotation_posterior_sums, ref.rotation_posterior_sums(reference, posts)) < 1e-5

    assert _rel_l2(out.Ft_y, _public_full(mstep["data"])) < 1e-5
    assert _rel_l2(out.Ft_ctf, _public_full(mstep["weight"])) < 1e-5

    noise_stats = out.noise_stats
    # relax keeps the image-power tail beyond current_size/2 in wsum_img_power; RELION adds it
    # to both sums (acc_ml_optimiser_impl.h:4947-4948).
    total_noise = np.asarray(noise_stats.wsum_sigma2_noise) + np.asarray(noise_stats.wsum_img_power)
    assert _rel_l2(total_noise, mstep["wsum_sigma2_noise"]) < 1e-6
    assert_matches(np.asarray(noise_stats.wsum_img_power), mstep["wsum_img_power"], rtol=F32)
    assert _rel_l2(noise_stats.wsum_norm_correction, mstep["wsum_norm_correction"]) < stats_rtol
    if scale_sums:
        for name in ("wsum_scale_correction_xa", "wsum_scale_correction_aa"):
            assert _rel_l2(getattr(noise_stats, name), mstep[name]) < stats_rtol, name
    assert float(noise_stats.sumw) == pytest.approx(mstep["sumw"], rel=1e-6)


@requires_resident_gpu
@pytest.mark.parametrize("n_coarse_trans", [9, 33])
def test_wide_translation_grids_are_relions_fine_pass(_resident_env, n_coarse_trans):
    """More coarse translations than one 32-bit mask word, and a partial second word."""

    from test_resident_wide_translation_mask import _wide_driver_args

    from relax.sparse_pass2 import resident_pass2 as rp

    args, reference = _case(6, 200.0, base_args=_wide_driver_args(n_coarse_trans))
    _assert_k1_pass(rp.compute_pass2_stats_resident(**args), reference)


@requires_resident_gpu
@pytest.mark.parametrize("groups", [True, False])
def test_vdam_residual_backprojection_is_relions_grad_pass(_resident_env, groups):
    """VDAM's ``--grad`` pass backprojects ``shifted - CTF P``; without groups the scale is 1."""

    from test_resident_vdam_estep import _vdam_args

    from relax.sparse_pass2 import resident_pass2 as rp

    args, reference = _case(6, 200.0, base_args=_vdam_args(residual=True, groups=groups), subtract_reference=True)
    _assert_k1_pass(rp.compute_pass2_stats_resident(**args), reference, scale_sums=groups)


@requires_resident_gpu
def test_zero_oversampling_reuses_relions_coarse_normalization(_resident_env):
    """At zero oversampling the fine pass reuses the coarse sum and keeps every hypothesis."""

    import dataclasses

    from test_resident_zero_oversampling import _os0_driver_args

    from relax.sparse_pass2 import resident_pass2 as rp

    base = _os0_driver_args()
    args, reference = _case(6, 200.0, base_args=base)
    # The coarse sum a real coarse pass would publish: sum_j exp(s_j - max s + 50) over the same
    # hypotheses (at zero oversampling the fine hypotheses are the coarse ones). The fixture's
    # random sums (1 to 20) put every weight near 1e20, where float32 rounding is no longer a
    # test of the arithmetic.
    coarse_sum = np.array([np.exp(post.log_weight - post.log_weight.max()).sum() for post in ref.posteriors(reference)])
    coarse_sum = coarse_sum * np.exp(50.0)
    args["relion_f32_normalization_sum_weight"] = coarse_sum
    reference = dataclasses.replace(reference, coarse_sum_weight=coarse_sum)
    # The pose is the coarse winner (op.max_index is not recomputed). The winners and Pmax are
    # synthetic, so the per-image posterior fields have no RELION statement to check; the M-step
    # and the statistics do.
    n_coarse_trans = int(np.asarray(base["translations"]).shape[0])
    winners = [(0, *divmod(int(w), n_coarse_trans)) for w in base["relion_coarse_hard_assignment"]]
    # Every hypothesis is kept here (no significance pruning), so each image's float32 Wavg sums
    # add all of its cells. Rounding error of a float32 sum of n terms grows like sqrt(n) eps32,
    # so the statistics bound is derived from the largest per-image cell count, with a factor of
    # 2: it is 1.1e-5 here (up to 2304 cells). Measured 3.2e-6 relative L2 on the per-image norm
    # sums (A100, job 14487449). The pruned passes above keep the 1e-6 bound (at most 3.1e-7).
    n_cells = max(len(post.cells) for post in ref.posteriors(reference))
    stats_rtol = 2.0 * np.sqrt(n_cells) * float(np.finfo(np.float32).eps)
    _assert_k1_pass(
        rp.compute_pass2_stats_resident(**args), reference, posterior=False, best=winners, stats_rtol=stats_rtol
    )


@requires_resident_gpu
def test_c4_pass_is_relions_fine_pass_on_the_asymmetric_unit(_resident_env):
    """C4: the E-step on the asymmetric-unit grid; the BPref as RELION's BackProjector holds it."""

    from test_resident_symmetry import _c4_driver_args

    from relax.sparse_pass2 import resident_pass2 as rp

    args, reference = _case(6, 200.0, base_args=_c4_driver_args(), symmetry="C4")
    _assert_k1_pass(rp.compute_pass2_stats_resident(**args), reference)


def _accumulator(actual, half):
    """The reference BPref array in the layout ``actual`` uses (public full, or x-half as stored)."""

    actual = np.asarray(actual).reshape(-1)
    return _public_full(half) if actual.size == half.shape[0] ** 3 else half.reshape(-1)


@requires_resident_gpu
@pytest.mark.parametrize("n_classes, current_size", [(2, 6), (3, 6), (2, 8)])
def test_resident_k_class_pass_is_relions_class3d_fine_pass(_resident_env, n_classes, current_size):
    """The joint class-by-pose posterior, per-class BPrefs and the summed statistics."""

    from test_resident_k_class_pass2 import _resident

    args, volumes, supports, priors, reference = _k_class_case(n_classes, current_size, 200.0)
    out = _resident(args, volumes, supports, priors)
    posts = ref.posteriors(reference)
    mstep = ref.mstep(reference, posts)
    n_fine_trans = reference.fine_translations.shape[0]

    log_z = np.array([np.logaddexp.reduce(post.log_weight) for post in posts])
    posterior_atol = 4.0 * float(np.finfo(np.float32).eps) * float(np.max(np.abs(log_z)))
    assert_matches(np.asarray(out.stats.log_evidence_per_image), log_z, rtol=F32)
    np.testing.assert_allclose(
        np.asarray(out.stats.max_posterior_per_image, dtype=np.float64),
        [post.weight.max() for post in posts],
        rtol=0.0,
        atol=posterior_atol,
    )
    for k in range(n_classes):
        class_log_z, class_best, class_hard = [], [], []
        for post in posts:
            mine = post.cells[:, 0] == k
            class_log_z.append(np.logaddexp.reduce(post.log_weight[mine]) if mine.any() else -np.inf)
            class_best.append(post.log_weight[mine].max() if mine.any() else -np.inf)
            best = post.cells[mine][np.argmax(post.log_weight[mine])] if mine.any() else None
            class_hard.append(-1 if best is None else best[1] * n_fine_trans + best[2])
        has = np.isfinite(class_log_z)
        assert_matches(np.asarray(out.class_log_evidence_per_image[k])[has], np.asarray(class_log_z)[has], rtol=F32)
        assert_matches(np.asarray(out.class_best_log_score_per_image[k])[has], np.asarray(class_best)[has], rtol=F32)
        assert_matches(np.asarray(out.per_class_hard_assignments[k]), np.asarray(class_hard))
        np.testing.assert_allclose(
            np.asarray(out.class_rotation_posterior_sums[k]),
            ref.rotation_posterior_sums(reference, posts, k),
            rtol=0.0,
            atol=posterior_atol * len(posts),
            err_msg=f"class {k}",
        )
        assert _rel_l2(out.Ft_y[k], _accumulator(out.Ft_y[k], mstep["class_data"][k])) < 1e-5, k
        assert _rel_l2(out.Ft_ctf[k], _accumulator(out.Ft_ctf[k], mstep["class_weight"][k])) < 1e-5, k
    # Scaled by the largest class: a nearly empty class carries the float32 rounding of the total.
    assert_matches(np.asarray(out.class_reconstruction_posterior_sums), mstep["class_sumw"], rtol=F32)
    noise_stats = out.noise_stats
    total_noise = np.asarray(noise_stats.wsum_sigma2_noise) + np.asarray(noise_stats.wsum_img_power)
    assert _rel_l2(total_noise, mstep["wsum_sigma2_noise"]) < 1e-6
    for name in ("wsum_norm_correction", "wsum_scale_correction_xa", "wsum_scale_correction_aa"):
        assert _rel_l2(getattr(noise_stats, name), mstep[name]) < 1e-6, name


# ---------------------------------------------------------------------------
# Local search: the resident local fine pass
# ---------------------------------------------------------------------------


def _local_case(current_size: int, noise: float):
    """``test_resident_local_pass2``'s layout and images with a padding-2 projector, and the reference.

    Padding 2, RELION's default, rather than that file's padding 1: RELION's projector keeps a
    point while ``int(|A^-1 k pad|^2) <= (r_max pad)^2`` (acc_projectorkernel_impl.h:177-179),
    so at padding 1 a pixel with ``|k|^2 = r_max^2 + 1`` is a float tie of that truncation.
    """

    from test_resident_local_pass2 import _case

    case = _case()
    half, r_max = _ppref(case["volume"])
    case.update(projector_half=jnp.asarray(half, dtype=jnp.complex64), r_max=r_max)
    noise_full = _tie_free_noise(current_size, noise)
    case["noise_variance"] = jnp.asarray(noise_full.reshape(-1))
    layout = case["layout"]
    n_trans = int(layout.translation_grid.shape[0])
    image_cells = []
    for image in range(layout.n_images):
        rows = np.arange(layout.rotation_offsets[image], layout.rotation_offsets[image + 1])
        mask = layout.sample_mask_rows(int(rows[0]), int(rows[-1]) + 1)
        if mask is None:
            mask = np.ones((rows.size, n_trans), dtype=bool)
        row, trans = np.nonzero(mask)
        image_cells.append(np.column_stack([rows[row], trans]))
    args = {
        "experiment_dataset": case["dataset"],
        "fine_rotations_override": np.asarray(layout.rotations_flat),
        "fine_rotation_parent_override": np.asarray(layout.rotation_posterior_ids_flat),
        "fine_translations_override": np.asarray(layout.translation_grid),
        "fine_translation_parent_override": np.zeros(n_trans, dtype=np.int64),
        "translation_log_prior": None,
        "reconstruction_padding_factor": 2,
        "adaptive_fraction": 0.999,
        "group_ids": np.zeros(layout.n_images, dtype=np.int64),
    }
    reference = _reference_pass(
        args,
        current_size,
        noise_full,
        projector={"data": half.astype(np.complex128), "pad": 2, "r_max": r_max},
        coarse_support=[None] * layout.n_images,
        rotation_log_prior=np.zeros(int(layout.n_global_rotations)),
        data_vs_prior=np.full(case["n_shells"], 5.0),
        image_cells=image_cells,
        fine_rotation_log_prior=np.asarray(layout.rotation_log_priors_flat, dtype=np.float64),
        fine_translation_log_prior=np.asarray(layout.translation_log_priors, dtype=np.float64),
        masked_images=_fftw_images(case["dataset"], masked=True),
    )
    return case, reference


def _run_local(case, current_size: int):
    """The production local fine pass (``_run_local_search_iteration``) at padding 2."""

    from test_resident_local_pass2 import N_IMAGES, OVERSAMPLING, PARENT_ORDER, _prior_eulers

    from relax.refinement import local_search_iteration

    return local_search_iteration._run_local_search_iteration(
        case["dataset"],
        case["volume"],
        case["noise_variance"],
        _prior_eulers(N_IMAGES, 20260919),
        None,
        PARENT_ORDER + OVERSAMPLING,
        0.35,
        0.35,
        case["translations"],
        np.zeros((N_IMAGES, 2), dtype=np.float32),
        3.0,
        "linear_interp",
        image_batch_size=4,
        rotation_block_size=64,
        current_size=current_size,
        reconstruction_current_size=current_size,
        accumulate_noise=True,
        projection_padding_factor=2,
        reconstruction_padding_factor=2,
        half_spectrum_scoring=True,
        relion_exact_score_translation=True,
        projection_relion_texture_interp=None,
        relion_projector_half=case["projector_half"],
        relion_projector_r_max=case["r_max"],
        do_gridding_correction=True,
        square_window=False,
        group_ids=np.zeros(N_IMAGES, dtype=np.int32),
        scale_correction_group_count=1,
        scale_correction_data_vs_prior=np.full(case["n_shells"], 5.0, dtype=np.float64),
        mstep_relion_x_half=True,
        reconstruct_significant_only=True,
        stats_use_reconstruction_probs=True,
        adaptive_fraction=0.999,
        max_significants=-1,
        return_best_pose_details=True,
        pass2_layout=case["layout"],
    )


@requires_resident_gpu
@pytest.mark.parametrize("noise", [200.0, 2.0])
def test_resident_local_pass_is_relions_local_fine_pass(monkeypatch, noise):
    """The resident local fine pass (current size 6 of 8) against the reference."""

    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT", "1")
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES", "64,256,1024")
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES", "2,4,8")
    monkeypatch.setenv("RELAX_RELION_PROJECTOR_TEXTURE_INTERP", "0")
    case, reference = _local_case(6, noise)
    out = _run_local(case, 6)
    posts = ref.posteriors(reference)
    mstep = ref.mstep(reference, posts)

    best = [post.cells[np.argmax(post.log_weight)] for post in posts]
    assert_matches(
        np.asarray(out.best_pose_rotations, dtype=np.float64),
        reference.fine_rotations[np.array([b[1] for b in best])],
        rtol=F32,
    )
    assert_matches(
        np.asarray(out.best_pose_translations, dtype=np.float64),
        reference.fine_translations[np.array([b[2] for b in best])],
        rtol=F32,
    )
    log_z = np.array([np.logaddexp.reduce(post.log_weight) for post in posts])
    best_score = np.array([post.log_weight.max() for post in posts])
    stats = out.relion_stats
    posterior_atol = 4.0 * float(np.finfo(np.float32).eps) * float(np.max(np.abs(log_z)))
    assert_matches(np.asarray(stats.log_evidence_per_image), log_z, rtol=F32)
    assert_matches(np.asarray(stats.best_log_score_per_image), best_score, rtol=F32)
    np.testing.assert_allclose(
        np.asarray(stats.max_posterior_per_image, dtype=np.float64),
        [post.weight.max() for post in posts],
        rtol=0.0,
        atol=posterior_atol,
    )
    np.testing.assert_allclose(
        np.asarray(stats.rotation_posterior_sums),
        ref.rotation_posterior_sums(reference, posts),
        rtol=0.0,
        atol=posterior_atol * len(posts),
    )
    assert _rel_l2(out.Ft_y, _accumulator(out.Ft_y, mstep["data"])) < 1e-5
    assert _rel_l2(out.Ft_ctf, _accumulator(out.Ft_ctf, mstep["weight"])) < 1e-5
    noise_stats = out.noise_stats
    total_noise = np.asarray(noise_stats.wsum_sigma2_noise) + np.asarray(noise_stats.wsum_img_power)
    assert _rel_l2(total_noise, mstep["wsum_sigma2_noise"]) < 1e-6
    for name in ("wsum_norm_correction", "wsum_scale_correction_xa", "wsum_scale_correction_aa"):
        assert _rel_l2(getattr(noise_stats, name), mstep[name]) < 1e-6, name
    assert float(noise_stats.sumw) == pytest.approx(mstep["sumw"], rel=1e-6)
