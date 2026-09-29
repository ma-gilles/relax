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
    relion_ctf.require_no_premultiplied_ctf(star_dataset, np.asarray([0, 3]), (BOX, BOX), where="a test path")
    with pytest.raises(NotImplementedError, match="a test path"):
        relion_ctf.require_no_premultiplied_ctf(star_dataset, np.asarray([0, 2]), (BOX, BOX), where="a test path")

    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    ordinary = SimpleNamespace(
        particles_file=str(_write_star(tmp_path / "ordinary.star", premultiplied=(0, 0))), image_shape=(BOX, BOX)
    )
    assert relion_ctf.premultiplied_ctf_rows(ordinary, np.arange(4), (BOX, BOX)) is None
    assert not relion_ctf.refuse_generic_ctf_for_premultiplied(ordinary)
    assert relion_ctf.premultiplied_ctf_rows(SimpleNamespace(particles_file=None), np.arange(4), (BOX, BOX)) is None


@pytest.mark.unit
def test_generic_ctf_fails_closed_on_premultiplied_data(star_dataset):
    star_dataset._ctf_evaluator = object()
    assert relion_ctf.refuse_generic_ctf_for_premultiplied(star_dataset)
    with pytest.raises(NotImplementedError, match="premultipli"):
        star_dataset._ctf_evaluator(np.zeros((1, 9)), (BOX, BOX), 2.0, half_image=True)


@pytest.mark.unit
def test_exact_ctf_rows_hold_relion_fctf_squared_for_premultiplied_groups(star_dataset):
    relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")
    order = np.asarray([2, 0, 3, 1])
    rows = relion_ctf._relion_exact_ctf_half_from_source_star_host(star_dataset, order, (BOX, BOX))

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
    refuse_unsupported_optics(optics, source="refine3d.star", ctf_premultiplied_supported=True)
    with pytest.raises(NotImplementedError, match="rlnCtfDataAreCtfPremultiplied"):
        refuse_unsupported_optics(optics, source="class3d.star")
