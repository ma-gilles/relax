"""CTF-premultiplied particles (``rlnCtfDataAreCtfPremultiplied``) against RELION's formulas.

RELION 5.0.1 squares the CTF image of a premultiplied particle right after
``CTF::getFftwImage`` (ml_optimiser.cpp:6486-6492, acc_ml_optimiser_impl.h:840-847)
and uses it in the ordinary scoring kernels; its BPref kernels then weight the image
by ``Minvsigma2`` alone and the backprojected CTF by ``Fctf * scale * Minvsigma2``
(BP.cuh, ``CTF_PREMULTIPLIED``). These tests hold relax's exact CTF rows and its
backprojection weights to those statements.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.relion import relion_ctf
from relax.relion.relion_metadata import refuse_unsupported_optics


# Moved from relax/relion/relion_ctf.py (PLAN e1): no relax module uses it, only this test file.
def require_no_premultiplied_ctf(experiment_dataset, image_indices, image_shape, *, where: str) -> None:
    """Refuse CTF-premultiplied images on a path that backprojects them as ordinary ones."""

    flags = relion_ctf.premultiplied_ctf_rows(experiment_dataset, image_indices, image_shape)
    if flags is not None and flags.any():
        raise NotImplementedError(
            f"{where} does not implement CTF-premultiplied images; they run on the resident sparse pass 2"
        )


BOX = 16
PARTICLES = (
    # defocus U, V, angle, optics group
    (21000.0, 20500.0, 30.0, 1),
    (15000.0, 15800.0, 110.0, 2),
    (18000.0, 18000.0, 0.0, 2),
    (25000.0, 24000.0, 75.0, 1),
)
OPTICS = (
    # group, voltage, Cs, Q0, pixel size, premultiplied
    (1, 300.0, 2.7, 0.07, 2.0, 0),
    (2, 200.0, 2.0, 0.10, 2.0, 1),
)


def _write_star(path, premultiplied=(0, 1)):
    lines = [
        "data_optics",
        "loop_",
        "_rlnOpticsGroup",
        "_rlnOpticsGroupName",
        "_rlnVoltage",
        "_rlnSphericalAberration",
        "_rlnAmplitudeContrast",
        "_rlnImagePixelSize",
        "_rlnImageSize",
        "_rlnImageDimensionality",
        "_rlnCtfDataAreCtfPremultiplied",
    ]
    for (group, voltage, cs, q0, pixel, _), flag in zip(OPTICS, premultiplied):
        lines.append(f"{group} optics{group} {voltage} {cs} {q0} {pixel} {BOX} 2 {flag}")
    lines += [
        "",
        "data_particles",
        "loop_",
        "_rlnImageName",
        "_rlnDefocusU",
        "_rlnDefocusV",
        "_rlnDefocusAngle",
        "_rlnOpticsGroup",
    ]
    for index, (du, dv, angle, group) in enumerate(PARTICLES, start=1):
        lines.append(f"{index}@stack.mrcs {du} {dv} {angle} {group}")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def star_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    star = _write_star(tmp_path / "particles.star")
    return SimpleNamespace(particles_file=str(star), image_shape=(BOX, BOX))


@pytest.mark.unit
def test_premultiplied_rows_follow_the_optics_group_flag(star_dataset, tmp_path, monkeypatch):
    flags = relion_ctf.premultiplied_ctf_rows(star_dataset, np.asarray([3, 1, 0, 2]), (BOX, BOX))
    assert_matches(flags, np.asarray([False, True, False, True]))
    assert relion_ctf.dataset_has_premultiplied_ctf(star_dataset, (BOX, BOX))
    require_no_premultiplied_ctf(star_dataset, np.asarray([0, 3]), (BOX, BOX), where="a test path")
    with pytest.raises(NotImplementedError, match="a test path"):
        require_no_premultiplied_ctf(star_dataset, np.asarray([0, 2]), (BOX, BOX), where="a test path")

    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    ordinary = SimpleNamespace(
        particles_file=str(_write_star(tmp_path / "ordinary.star", premultiplied=(0, 0))), image_shape=(BOX, BOX)
    )
    assert relion_ctf.premultiplied_ctf_rows(ordinary, np.arange(4), (BOX, BOX)) is None
    assert not relion_ctf.refuse_generic_ctf_for_optics(ordinary)


@pytest.mark.unit
def test_premultiplied_questions_refuse_a_dataset_whose_optics_are_unknown():
    """A non-STAR source leaves the optics table unknown: raise rather than answer "not premultiplied"."""

    dataset = SimpleNamespace(particles_file="particles.mrcs", image_shape=(BOX, BOX), n_units=4)
    with pytest.raises(ValueError, match="SimpleNamespace is not a STAR-backed dataset .* CTF-premultiplied"):
        relion_ctf.premultiplied_ctf_rows(dataset, np.arange(4), (BOX, BOX))
    with pytest.raises(ValueError, match="not a STAR-backed dataset"):
        relion_ctf.dataset_has_premultiplied_ctf(dataset, (BOX, BOX))
    with pytest.raises(ValueError, match="not a STAR-backed dataset"):
        relion_ctf.premultiplied_average_ctf2([(dataset, None, BOX, 1.0)], BOX, 4.0)
    with pytest.raises(AttributeError, match="particles_file"):
        relion_ctf.dataset_has_premultiplied_ctf(SimpleNamespace(image_shape=(BOX, BOX)), (BOX, BOX))


@pytest.mark.unit
def test_a_dataset_built_in_memory_has_no_optics_table_unless_it_flags_premultiplied_images():
    in_memory = SimpleNamespace(particles_file=None, image_shape=(BOX, BOX), n_units=4, premultiplied_ctf=False)
    assert relion_ctf.premultiplied_ctf_rows(in_memory, np.arange(4), (BOX, BOX)) is None
    assert not relion_ctf.dataset_has_premultiplied_ctf(in_memory, (BOX, BOX))
    assert relion_ctf.premultiplied_average_ctf2([(in_memory, None, BOX, 1.0)], BOX, 4.0) is None
    in_memory.premultiplied_ctf = True
    with pytest.raises(ValueError, match="built in memory .* CTF-premultiplied"):
        relion_ctf.dataset_has_premultiplied_ctf(in_memory, (BOX, BOX))


@pytest.mark.unit
def test_generic_ctf_fails_closed_on_premultiplied_data(star_dataset):
    star_dataset._ctf_evaluator = object()
    assert relion_ctf.refuse_generic_ctf_for_optics(star_dataset)
    with pytest.raises(NotImplementedError, match="premultipli"):
        star_dataset._ctf_evaluator(np.zeros((1, 9)), (BOX, BOX), 2.0, half_image=True)


@pytest.mark.unit
def test_exact_ctf_rows_hold_relion_fctf_squared_for_premultiplied_groups(star_dataset):
    relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")
    order = np.asarray([2, 0, 3, 1])
    rows = relion_ctf.relion_exact_ctf_half_from_source_star_host(star_dataset, order, (BOX, BOX))

    groups = {group: (voltage, cs, q0, pixel, flag) for group, voltage, cs, q0, pixel, flag in OPTICS}
    expected = []
    for index in order:
        du, dv, angle, group = PARTICLES[index]
        voltage, cs, q0, pixel, flag = groups[group]
        fctf = np.asarray(
            relion_bind.get_ctf_image(
                du, dv, angle, voltage, cs, q0, 0.0, pixel, BOX, BOX, False, False, False, 0.0, 1.0
            )
        )
        if flag:
            fctf = fctf * fctf  # ml_optimiser.cpp:6486-6492
        # RECOVAR's frame: centered rows and the opposite CTF sign (relion_ctf._evaluate_exact_ctf_rows).
        expected.append(-np.fft.fftshift(fctf, axes=0).reshape(-1))
    assert_matches(rows, np.stack(expected), rtol=1e-13)
    assert np.all(rows[[0, 3]] <= 0.0)  # -CTF^2 on the premultiplied images


def _relion_bp_sums(image, fctf, minvsigma2, weight, scale, premultiplied):
    """RELION's BPref sums for one image and one pose (BP.cuh cuda_kernel_backproject3D).

    ``fctf`` is RELION's CTF operand (``CTF^2`` for a premultiplied image) before the
    scale correction, which RELION folds into ``ctfs`` (acc_ml_optimiser_impl.h:3090).
    """

    ctf = fctf * scale
    if premultiplied:
        w = weight * minvsigma2
        return image * w, w * ctf
    w = weight * ctf * minvsigma2
    return image * w, w * ctf


@pytest.mark.unit
def test_premultiplied_bpref_weights_match_relion_bp_kernel_in_recovar_frame():
    from relax.sparse_pass2.sparse_pass2_bucket_io import premultiplied_bpref_weights

    rng = np.random.default_rng(7)
    n_images, n_pixels = 3, 11
    ctf = rng.uniform(-1.0, 1.0, (n_images, n_pixels))
    fctf = ctf * ctf  # a premultiplied image's RELION Fctf
    image = rng.standard_normal((n_images, n_pixels)) + 1j * rng.standard_normal((n_images, n_pixels))
    minvsigma2 = rng.uniform(0.5, 2.0, n_pixels)
    scale = rng.uniform(0.8, 1.2, n_images)
    weight = rng.uniform(0.0, 1.0, n_images)

    image_weight, ctf_weight = premultiplied_bpref_weights(-fctf, minvsigma2, scale)
    image_weight, ctf_weight = np.asarray(image_weight), np.asarray(ctf_weight)
    data = weight[:, None] * image * image_weight
    bp_weight = weight[:, None] * ctf_weight
    relion_data, relion_weight = _relion_bp_sums(
        image, fctf, minvsigma2[None, :], weight[:, None], scale[:, None], premultiplied=True
    )
    # RECOVAR's reconstruction is the negated RELION map: same weight, negated data.
    assert_matches(bp_weight, relion_weight, rtol=1e-13)
    assert_matches(data, -relion_data, rtol=1e-13)

    # An ordinary image through the ordinary operands (ctf * minvsigma2 * scale, and its square).
    ordinary_data = weight[:, None] * image * (-ctf) * minvsigma2 * scale[:, None]
    ordinary_weight = weight[:, None] * ((-ctf) * minvsigma2 * (-ctf)) * scale[:, None] ** 2
    relion_data, relion_weight = _relion_bp_sums(
        image, ctf, minvsigma2[None, :], weight[:, None], scale[:, None], premultiplied=False
    )
    assert_matches(ordinary_weight, relion_weight, rtol=1e-13)
    assert_matches(ordinary_data, -relion_data, rtol=1e-13)

    no_scale_image, no_scale_ctf = premultiplied_bpref_weights(-fctf, minvsigma2, None)
    assert_matches(np.asarray(no_scale_image), image_weight)
    assert_matches(np.asarray(no_scale_ctf), fctf * minvsigma2, rtol=1e-13)


@pytest.mark.unit
def test_refinement_accepts_premultiplied_optics_only_where_supported():
    import pandas as pd

    optics = pd.DataFrame({"_rlnOpticsGroup": [1, 2], "_rlnCtfDataAreCtfPremultiplied": [1, 0]})
    refuse_unsupported_optics(optics, source="refine3d.star", supported={"ctf_premultiplied"})
    with pytest.raises(NotImplementedError, match="rlnCtfDataAreCtfPremultiplied"):
        refuse_unsupported_optics(optics, source="class3d.star")


@pytest.mark.unit
def test_dense_preprocessing_scores_premultiplied_batches_with_the_exact_rows(star_dataset, tmp_path, monkeypatch):
    from relax.helpers.preprocessing import _dense_batch_ctf_half

    generic_calls = []
    config = SimpleNamespace(image_shape=(BOX, BOX), compute_ctf_half=lambda params: generic_calls.append(params))
    order = np.asarray([2, 0, 3, 1])
    rows = np.asarray(_dense_batch_ctf_half(star_dataset, np.zeros((4, 9)), config, np.float32, order))
    assert rows.dtype == np.float32 and not generic_calls
    expected = relion_ctf.relion_exact_ctf_half_from_source_star_host(star_dataset, order, (BOX, BOX))
    assert_matches(rows, expected, rtol=1e-6)  # the float32 cast

    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    ordinary = SimpleNamespace(
        particles_file=str(_write_star(tmp_path / "ordinary.star", premultiplied=(0, 0))), image_shape=(BOX, BOX)
    )
    _dense_batch_ctf_half(ordinary, np.zeros((4, 9)), config, np.float32, order)
    assert len(generic_calls) == 1


def _relion_sumw_ctf2(fctf_fftw, window):
    """RELION's per-image sumw_ctf2 term, literally: Mresol_fine over the window of Fctf."""

    size = fctf_fftw.shape[0]
    sums = np.zeros(window // 2 + 1)
    for i in range(window):
        ip = i if i < window // 2 + 1 else i - window
        for jp in range(window // 2 + 1):
            ires = int(np.floor(np.sqrt(ip * ip + jp * jp) + 0.5))
            if ires < window // 2 + 1 and not (jp == 0 and ip < 0):
                sums[ires] += fctf_fftw[ip % size, jp]
    return sums


def _relion_npix_per_shell(ori_size):
    npix = np.zeros(ori_size // 2 + 1)
    for i in range(ori_size):
        ip = i if i < ori_size // 2 + 1 else i - ori_size
        for jp in range(ori_size // 2 + 1):
            ires = int(np.floor(np.sqrt(ip * ip + jp * jp) + 0.5))
            if ires < ori_size // 2 + 1 and not (jp == 0 and ip < 0):
                npix[ires] += 1
    return npix


@pytest.mark.unit
@pytest.mark.parametrize("window", [BOX, 10])
def test_average_ctf2_is_relions_set_average_ctf2(star_dataset, window):
    """setAverageCTF2 (ml_optimiser.cpp:4885-4926) with the storeWeightedSums sumw_ctf2 term, written out
    literally. The denominator is sumw_group, the images' significant posterior mass (relax#60): below
    the image count when the fine pass keeps only the top adaptive_fraction of each image's mass."""

    star_dataset.n_units = len(PARTICLES)
    scales = np.asarray([1.2, 0.0005, 0.9, 1.1])
    rows = relion_ctf.relion_exact_ctf_half_from_source_star_host(star_dataset, np.arange(4), (BOX, BOX))
    # RECOVAR frame -> RELION's FFTW Fctf (relion_ctf._evaluate_exact_ctf_rows).
    fctf = -np.fft.ifftshift(np.asarray(rows).reshape(4, BOX, BOX // 2 + 1), axes=1)
    premultiplied = [group == 2 for *_, group in PARTICLES]
    numerator = sum(
        max(0.001, scales[p]) * _relion_sumw_ctf2(fctf[p], window) for p in range(4) if premultiplied[p]
    )
    npix = _relion_npix_per_shell(BOX)
    expected = np.zeros(BOX // 2 + 1)
    expected[: window // 2 + 1] = numerator
    sumw = 4 * 0.999  # four images, each keeping 0.999 of its mass
    expected = expected / (sumw * npix)

    sums = relion_ctf.premultiplied_ctf2_shell_sums(star_dataset, np.arange(4), (BOX, BOX), window)
    for p in range(4):
        assert_matches(sums[p], _relion_sumw_ctf2(fctf[p], window), rtol=1e-12)
    average = relion_ctf.premultiplied_average_ctf2([(star_dataset, scales, window, 1.0)], BOX, sumw)
    assert_matches(average, expected, rtol=1e-12)
    assert np.all(average[: window // 2 + 1] > 0)
    # Counting images instead of their weight (before relax#60) is off by the kept mass.
    by_count = relion_ctf.premultiplied_average_ctf2([(star_dataset, scales, window, 1.0)], BOX, 4.0)
    assert_matches(by_count[: window // 2 + 1] / average[: window // 2 + 1], 0.999, rtol=1e-12)
    with pytest.raises(ValueError, match="needs the E-step's sumw"):
        relion_ctf.premultiplied_average_ctf2([(star_dataset, scales, window, 1.0)], BOX, 0.0)


@pytest.mark.unit
def test_average_ctf2_is_none_without_premultiplied_images(tmp_path, monkeypatch):
    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    ordinary = SimpleNamespace(
        particles_file=str(_write_star(tmp_path / "ordinary.star", premultiplied=(0, 0))),
        image_shape=(BOX, BOX),
        n_units=len(PARTICLES),
    )
    assert relion_ctf.premultiplied_average_ctf2([(ordinary, None, BOX, 1.0)], BOX, 4.0) is None


def _shape_class_half(star, scale_b):
    """A two-shape half over the four STAR rows: class A rows (0, 3) on the model grid, class B rows (2, 1) at s_g."""

    from relax.refinement.optics_shapes import MultiShapeHalf, ShapeClass

    def dataset(rows):
        return SimpleNamespace(
            particles_file=str(star), image_shape=(BOX, BOX), n_units=len(rows), dataset_indices=np.asarray(rows)
        )

    classes = (
        ShapeClass(dataset([0, 3]), np.asarray([0, 3]), BOX, 2.0, 1.0, 1.0),
        ShapeClass(dataset([2, 1]), np.asarray([1, 2]), BOX, 2.0 * scale_b, scale_b, 1.0 / scale_b),
    )
    return MultiShapeHalf(classes, image_shape=(BOX, BOX), volume_shape=(BOX,) * 3, voxel_size=2.0)


@pytest.mark.unit
def test_average_ctf2_of_several_image_shapes_is_relions_remapped_sumw_ctf2(tmp_path, monkeypatch):
    """Each shape class sums Fctf on its own image current size, then storeWeightedSums adds
    shell i onto model shell ROUND(i / s_g) (acc_ml_optimiser_impl.h:3612-3626)."""

    from relax.refinement.noise_updates import datasets_store_premultiplied_ctf
    from relax.refinement.optics_shapes import average_ctf2_parts

    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    star = _write_star(tmp_path / "particles.star")
    scale_b, current_size = 0.8, 12
    half = _shape_class_half(star, scale_b)
    half_scales = np.asarray([1.2, 0.0005, 0.9, 1.1])  # in the half's image order

    full = SimpleNamespace(particles_file=str(star), image_shape=(BOX, BOX))
    rows = relion_ctf.relion_exact_ctf_half_from_source_star_host(full, np.arange(4), (BOX, BOX))
    fctf = -np.fft.ifftshift(np.asarray(rows).reshape(4, BOX, BOX // 2 + 1), axes=1)
    window_b = 2 * int(np.ceil(0.5 * scale_b * current_size))  # 10; its shells 0-5 land on 0, 1, 3, 4, 5, 6
    expected = np.zeros(BOX // 2 + 1)
    # Class B holds the premultiplied STAR rows 2 and 1, at the half's images 1 and 2.
    for star_row, image in ((2, 1), (1, 2)):
        term = max(0.001, half_scales[image]) * _relion_sumw_ctf2(fctf[star_row], window_b)
        for i, value in enumerate(term):
            i_resam = int(np.floor(i / scale_b + 0.5))
            if i_resam < expected.size:
                expected[i_resam] += value
    expected /= 3.5 * _relion_npix_per_shell(BOX)

    parts = average_ctf2_parts(half, half_scales, current_size=current_size, image_current_size=current_size)
    assert [part[2] for part in parts] == [current_size, window_b]
    average = relion_ctf.premultiplied_average_ctf2(parts, BOX, 3.5)
    assert_matches(average, expected, rtol=1e-12)
    assert datasets_store_premultiplied_ctf([half])
    # The half itself has no single source STAR: asking it directly is refused, not answered "no".
    with pytest.raises(AttributeError, match="several image shapes"):
        relion_ctf.dataset_has_premultiplied_ctf(half, (BOX, BOX))


@pytest.mark.unit
def test_subtomogram_average_ctf2_divides_by_the_particles_sumw_not_by_a_count(monkeypatch):
    """relax#63: for 2D-stack subtomograms RELION's numerator adds one term per tilt image
    (acc_ml_optimiser_impl.h:3590-3600) and its denominator, sumw_group, one posterior mass per particle
    (:2842). The Class3D prior hands premultiplied_average_ctf2 the tilt images as parts and the halves'
    noise sumw as the denominator: neither the tilt count nor the particle count."""

    from relax.helpers.types import NoiseStats
    from relax.refinement import priors
    from relax.refinement.tomo_half import TomoHalf

    class _Seen(Exception):
        pass

    seen = {}

    def average_ctf2(parts, box_size, sumw):
        seen.update(parts=parts, box_size=box_size, sumw=sumw)
        raise _Seen

    monkeypatch.setattr(priors.relion_ctf, "premultiplied_average_ctf2", average_ctf2)
    monkeypatch.setattr(
        priors, "average_ctf2_parts", lambda dataset, scales, **_: [(dataset, scales, BOX, 1.0)]
    )

    def half(tilts_per_particle, scales, kept_mass):
        tomo = object.__new__(TomoHalf)
        tomo.unit_image_offsets = np.concatenate([[0], np.cumsum(tilts_per_particle)])
        tomo.images = SimpleNamespace(n_units=int(np.sum(tilts_per_particle)))
        stats = NoiseStats(
            wsum_sigma2_noise=np.zeros(1), wsum_img_power=np.zeros(1), wsum_sigma2_offset=0.0,
            sumw=np.asarray(kept_mass, dtype=np.float64),  # per optics group
        )
        return SimpleNamespace(dataset=tomo, scale_corrections=np.asarray(scales)), stats

    half1, stats1 = half([3, 4], [0.9, 1.1], [0.999, 0.998])  # 2 particles, 7 tilt images
    half2, stats2 = half([2], [1.0], [0.0, 0.997])  # 1 particle, 2 tilt images
    with pytest.raises(_Seen):
        priors.estimate_class_priors(
            None, None, None, SimpleNamespace(box_size=BOX),
            half_denominators=None, halves=(half1, half2), noise_stats_per_half=(stats1, stats2), n_classes=2,
            iteration=0, current_size=BOX, image_current_size=BOX, accumulator_shape=None, full_half_axis=-1,
            projector_power_spectrum=None, class_tau2=SimpleNamespace(source=None), scoring_dtype=np.float32, log=None,
        )

    n_tilts, n_particles = 9, 3
    assert seen["sumw"] == pytest.approx(0.999 + 0.998 + 0.997, rel=1e-12)
    assert seen["sumw"] != n_tilts and seen["sumw"] != n_particles
    # The numerator's parts are the tilt images, each with its particle's scale.
    assert [part[0].n_units for part in seen["parts"]] == [7, 2]
    assert_matches(seen["parts"][0][1], [0.9, 0.9, 0.9, 1.1, 1.1, 1.1, 1.1], rtol=0)
