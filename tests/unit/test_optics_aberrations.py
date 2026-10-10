"""Beam tilt and Zernike aberrations against RELION's ObservationModel.

relax evaluates the odd Zernike phase (beam tilt included) and the even Zernike
gamma offset of an optics group in Python (``relax.relion.optics_aberrations``);
RELION's own ``ObservationModel::getPhaseCorrection`` / ``getGammaOffset``,
reached through the binding's ``optics_phase_correction`` / ``optics_gamma_offset``,
is the reference. Images are demodulated by the conjugate phase, as
``ObservationModel::demodulatePhase`` does.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import starfile
from helpers.float_compare import assert_matches

from relax.relion import ctf as relion_ctf
from relax.relion import optics_aberrations as oa
from relax.relion.metadata import (
    IMPLEMENTED_OPTICS_FEATURES,
    INITIAL_MODEL_OPTICS_FEATURES,
    refuse_unsupported_optics,
)
from relax.vdam import bootstrap_iref

BOX = 24
PIXEL = 1.4
ODD = "[0.12,-0.25,0.05,0.3,0.02,-0.04]"
EVEN = "[0,0.2,-0.1,0.3,0.05,-0.02,0.01,0.1,0.02]"


def _write_star(path, *, tilt=(0.9, -0.6), odd=ODD, even=None, mag=None, mag_both=False):
    labels = [
        "_rlnOpticsGroup",
        "_rlnOpticsGroupName",
        "_rlnVoltage",
        "_rlnSphericalAberration",
        "_rlnAmplitudeContrast",
        "_rlnImagePixelSize",
        "_rlnImageSize",
        "_rlnImageDimensionality",
    ]
    ordinary = ["1", "og1", "300", "2.7", "0.07", str(PIXEL), str(BOX), "2"]
    aberrated = ["2", "og2", "200", "2.0", "0.1", str(PIXEL), str(BOX), "2"]
    if tilt is not None:
        labels += ["_rlnBeamTiltX", "_rlnBeamTiltY"]
        ordinary += ["0", "0"]
        aberrated += [str(tilt[0]), str(tilt[1])]
    if odd is not None:
        labels.append("_rlnOddZernike")
        ordinary.append("[0,0,0,0,0,0]")
        aberrated.append(odd)
    if even is not None:
        labels.append("_rlnEvenZernike")
        ordinary.append("[0,0,0,0,0,0,0,0,0]")
        aberrated.append(even)
    if mag is not None:
        labels += ["_rlnMagMat00", "_rlnMagMat01", "_rlnMagMat10", "_rlnMagMat11"]
        ordinary += [str(v) for v in np.asarray(mag if mag_both else np.eye(2)).reshape(-1)]
        aberrated += [str(v) for v in np.asarray(mag).reshape(-1)]
    lines = ["data_optics", "", "loop_", *labels, " ".join(ordinary), " ".join(aberrated), ""]
    lines += ["data_particles", "", "loop_", "_rlnImageName", "_rlnDefocusU", "_rlnDefocusV"]
    lines += ["_rlnDefocusAngle", "_rlnOpticsGroup"]
    for index, group in enumerate((2, 1, 2), start=1):
        lines.append(f"{index}@stack.mrcs 20000 19000 40 {group}")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    monkeypatch.setattr(oa, "_ODD_PHASE_CACHE", {})


def _optics_row(star):
    return starfile.read(star, always_dict=True)["optics"].iloc[1]


@pytest.mark.unit
@pytest.mark.parametrize(
    "tilt, odd, mag",
    [
        ((0.9, -0.6), None, None),
        (None, ODD, None),
        ((0.9, -0.6), ODD, None),
        ((0.9, -0.6), ODD, [[1.01, 0.004], [-0.003, 0.995]]),
    ],
)
def test_odd_phase_matches_relion_phase_correction(tmp_path, tilt, odd, mag):
    relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")
    if not hasattr(relion_bind, "optics_phase_correction"):
        pytest.skip("the RELION binding predates the optics bindings")
    star = _write_star(tmp_path / "particles.star", tilt=tilt, odd=odd, mag=mag)
    row = _optics_row(star)
    coefficients = oa.optics_group_odd_coefficients(row, has_odd=odd is not None, has_tilt=tilt is not None)
    phase = oa.zernike_phase_fftw_half(
        coefficients, oa.odd_index_to_mn, BOX, PIXEL, BOX, None if mag is None else oa.optics_group_mag_matrix(row)
    )
    relion = np.asarray(relion_bind.optics_phase_correction(str(star), 2, BOX))
    assert np.abs(phase).max() > 0.1  # the aberration is not negligible on this grid
    assert_matches(np.exp(1j * phase), relion, rtol=1e-13)
    assert_matches(np.asarray(relion_bind.optics_phase_correction(str(star), 1, BOX)), np.ones_like(relion))


@pytest.mark.unit
def test_even_gamma_offset_matches_relion(tmp_path):
    relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")
    if not hasattr(relion_bind, "optics_gamma_offset"):
        pytest.skip("the RELION binding predates the optics bindings")
    star = _write_star(tmp_path / "particles.star", tilt=None, odd=None, even=EVEN)
    gamma = oa.zernike_phase_fftw_half(oa.parse_relion_vector(EVEN), oa.even_index_to_mn, BOX, PIXEL, BOX)
    assert_matches(gamma, np.asarray(relion_bind.optics_gamma_offset(str(star), 2, BOX)), rtol=1e-13)


@pytest.mark.unit
def test_optics_ctf_equals_the_plain_ctf_for_an_ordinary_group(tmp_path):
    relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")
    if not hasattr(relion_bind, "optics_ctf_images_batch"):
        pytest.skip("the RELION binding predates the optics bindings")
    star = _write_star(tmp_path / "particles.star", tilt=None, odd=None, even=EVEN)
    params = np.asarray([[20000.0, 19000.0, 40.0, 0.0, 1.0, 0.0, 1], [20000.0, 19000.0, 40.0, 0.0, 1.0, 0.0, 2]])
    optics = np.asarray(relion_bind.optics_ctf_images_batch(str(star), params, BOX, BOX, False, 2))
    plain = np.asarray(
        relion_bind.get_ctf_images_batch(
            np.asarray([[20000.0, 19000.0, 40.0, 300.0, 2.7, 0.07, 0.0, PIXEL, 0.0, 1.0]]),
            BOX,
            BOX,
            False,
            False,
            False,
            1,
        )
    )
    assert_matches(optics[0], plain[0], rtol=1e-13)
    assert np.abs(optics[1] - plain[0]).max() > 0.05  # other kV, Cs, Q0 and the even terms


@pytest.mark.unit
def test_beam_tilt_insertion_follows_tilt_helper():
    coefficients = oa.insert_beam_tilt([0.0] * 6, 0.9, -0.6, 2.7, oa.relion_wavelength(300.0))
    scale = 2.7 * 20000 * oa.relion_wavelength(300.0) ** 2 * 3.141592654
    z3x, z3y = -scale * 0.9 / 3.0, -scale * -0.6 / 3.0
    assert_matches(np.asarray(coefficients), np.asarray([2 * z3y, 2 * z3x, 0.0, z3y, z3x, 0.0]), rtol=1e-13)
    # A shorter coefficient list is extended to the five coma/trefoil terms.
    assert len(oa.insert_beam_tilt([0.1, 0.2], 0.1, 0.0, 2.7, 0.02)) == 5


@pytest.mark.unit
def test_refinement_accepts_the_implemented_optics_features():
    optics = pd.DataFrame({"_rlnOpticsGroup": [1], "_rlnBeamTiltX": [0.4], "_rlnOddZernike": ["[0.1,0]"]})
    refuse_unsupported_optics(optics, source="refine3d.star", supported=IMPLEMENTED_OPTICS_FEATURES)
    with pytest.raises(NotImplementedError, match="rlnBeamTiltX"):
        refuse_unsupported_optics(optics, source="class3d.star")
    refuse_unsupported_optics(
        pd.DataFrame({"_rlnOpticsGroup": [1], "_rlnEvenZernike": ["[0,0.1]"]}),
        source="refine3d.star",
        supported=IMPLEMENTED_OPTICS_FEATURES,
    )
    refuse_unsupported_optics(
        pd.DataFrame({"_rlnOpticsGroup": [1], "_rlnMagMat00": [1.01]}),
        source="refine3d.star",
        supported=IMPLEMENTED_OPTICS_FEATURES,
    )
    with pytest.raises(NotImplementedError, match="rlnMagMat00"):
        refuse_unsupported_optics(
            pd.DataFrame({"_rlnOpticsGroup": [1], "_rlnMagMat00": [1.01]}), source="class3d.star"
        )


def _relion_ctf_reference(du, dv, angle, voltage, cs, q0, size, pixel, gamma_offset, mag=None):
    """CTF::initialise + getCTF (ctf.cpp:211-262, ctf.h:184-257) on the FFTW half grid, no damping."""

    lam = 12.2643247 / np.sqrt(voltage * 1e3 * (1.0 + voltage * 1e3 * 0.978466e-6))
    k1 = np.pi / 2 * 2 * lam
    k2 = np.pi / 2 * cs * 1e7 * lam**3
    k3 = np.arctan(q0 / np.sqrt(1 - q0 * q0))
    az = np.deg2rad(angle)
    q = np.array([[np.cos(az), np.sin(az)], [-np.sin(az), np.cos(az)]])
    a = q.T @ np.diag([-du, -dv]) @ q
    half = size // 2 + 1
    rows = np.arange(size)
    y = np.where(rows <= size // 2, rows, rows - size)[:, None] / (size * pixel)
    x = np.arange(half)[None, :] / (size * pixel)
    x, y = np.broadcast_to(x, (size, half)), np.broadcast_to(y, (size, half))
    if mag is not None:
        x, y = mag[0, 0] * x + mag[0, 1] * y, mag[1, 0] * x + mag[1, 1] * y
    u2 = x * x + y * y
    gamma = k1 * (a[0, 0] * x * x + 2 * a[0, 1] * x * y + a[1, 1] * y * y) + k2 * u2 * u2 - k3 + gamma_offset
    ctf = -np.sin(gamma)
    return np.where(np.abs(ctf) < 1e-8, np.sign(ctf) * 1e-8, ctf)


@pytest.mark.unit
def test_exact_ctf_rows_carry_the_even_zernike_gamma_offset(tmp_path):
    relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")
    if not hasattr(relion_bind, "optics_ctf_images_batch"):
        pytest.skip("the RELION binding predates the optics bindings")
    star = _write_star(tmp_path / "particles.star", tilt=None, odd=None, even=EVEN)
    dataset = SimpleNamespace(particles_file=str(star), image_shape=(BOX, BOX))
    rows = relion_ctf.relion_exact_ctf_half_from_source_star_host(dataset, np.asarray([0, 1]), (BOX, BOX))
    gamma = oa.zernike_phase_fftw_half(oa.parse_relion_vector(EVEN), oa.even_index_to_mn, BOX, PIXEL, BOX)
    expected = [
        _relion_ctf_reference(20000.0, 19000.0, 40.0, 200.0, 2.0, 0.1, BOX, PIXEL, gamma),  # group 2
        _relion_ctf_reference(20000.0, 19000.0, 40.0, 300.0, 2.7, 0.07, BOX, PIXEL, 0.0),  # group 1
    ]
    # RECOVAR's frame: centered rows, opposite sign (relion_ctf._evaluate_exact_ctf_rows).
    assert_matches(rows, np.stack([-np.fft.fftshift(e, axes=0).reshape(-1) for e in expected]), rtol=1e-12)


MAG = np.asarray([[1.012, 0.006], [-0.004, 0.991]])


@pytest.mark.unit
def test_exact_ctf_rows_carry_the_magnification(tmp_path):
    relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")
    if not hasattr(relion_bind, "optics_ctf_images_batch"):
        pytest.skip("the RELION binding predates the optics bindings")
    star = _write_star(tmp_path / "particles.star", tilt=None, odd=None, even=EVEN, mag=MAG)
    dataset = SimpleNamespace(particles_file=str(star), image_shape=(BOX, BOX))
    rows = relion_ctf.relion_exact_ctf_half_from_source_star_host(dataset, np.asarray([0]), (BOX, BOX))
    gamma = oa.zernike_phase_fftw_half(oa.parse_relion_vector(EVEN), oa.even_index_to_mn, BOX, PIXEL, BOX, MAG)
    expected = _relion_ctf_reference(20000.0, 19000.0, 40.0, 200.0, 2.0, 0.1, BOX, PIXEL, gamma, mag=MAG)
    assert_matches(rows[0], -np.fft.fftshift(expected, axes=0).reshape(-1), rtol=1e-12)
    assert oa.dataset_needs_exact_ctf(dataset)
    # Projection matrices are transformed per scoring call: a dataset spanning groups with
    # different magnifications must be split into shape classes, one matrix each (relax#48).
    with pytest.raises(ValueError, match="split into shape classes"):
        oa.dataset_projection_magnification(dataset)
    # Particles 1 and 3 are optics group 2 (MAG), particle 2 group 1 (identity).
    group_2 = SimpleNamespace(**vars(dataset), n_units=2, dataset_indices=np.asarray([0, 2]))
    group_1 = SimpleNamespace(**vars(dataset), n_units=1, dataset_indices=np.asarray([1]))
    assert_matches(oa.dataset_projection_magnification(group_2), oa.relax_projection_magnification(MAG))
    assert oa.dataset_projection_magnification(group_1) is None
    assert oa.dataset_magnification_is_anisotropic(group_2)
    assert not oa.dataset_magnification_is_anisotropic(group_1)
    shared = SimpleNamespace(
        particles_file=str(_write_star(tmp_path / "shared.star", tilt=None, odd=None, mag=MAG, mag_both=True)),
        image_shape=(BOX, BOX),
    )
    assert_matches(oa.dataset_projection_magnification(shared), oa.relax_projection_magnification(MAG))


@pytest.mark.unit
def test_magnified_projection_matrices_match_relion_left_matrix():
    """relax projects with inv(A)^T; RELION's magnified matrix is inv(M3) A (applyAnisoMag)."""

    relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")
    rng = np.random.default_rng(11)
    eulers = rng.uniform([-180, 0, -180], [180, 180, 180], size=(7, 3))
    m3 = np.eye(3)
    m3[:2, :2] = MAG
    left = np.broadcast_to(np.linalg.inv(m3), (7, 3, 3)).copy()
    relion = np.swapaxes(np.asarray(relion_bind.euler_angles_to_inverse_matrices(eulers, left)), 1, 2)
    plain = np.swapaxes(np.asarray(relion_bind.euler_angles_to_inverse_matrices(eulers)), 1, 2)
    magnified = oa.projection_rotations(plain, 1.0, oa.relax_projection_magnification(MAG), dtype=np.float64)
    assert_matches(magnified, relion, rtol=1e-13)
    assert_matches(oa.reported_rotations(magnified, 1.0, oa.relax_projection_magnification(MAG)), plain, rtol=1e-13)
    # With the optics scale: applyScaleDifference multiplies inv(M3) A by s (obs_model.cpp:1332-1339).
    scaled = np.swapaxes(np.asarray(relion_bind.euler_angles_to_inverse_matrices(eulers, 1.1 * left)), 1, 2)
    assert_matches(
        oa.projection_rotations(plain, 1.1, oa.relax_projection_magnification(MAG), dtype=np.float64),
        scaled,
        rtol=1e-13,
    )


@pytest.mark.unit
def test_expected_accuracy_with_caller_ctfs_is_the_plain_one_for_an_ordinary_group():
    relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")
    box = 16
    rng = np.random.default_rng(5)
    common = dict(
        references=rng.standard_normal((1, box, box, box)),
        eulers_deg=np.asarray([[10.0, 40.0, 70.0], [100.0, 120.0, -30.0]]),
        particle_ids=np.asarray([0, 1]),
        class_ids=np.asarray([0, 0], dtype=np.int32),
        pdf_class=np.asarray([1.0]),
        sigma2_noise=np.full(box // 2 + 1, 0.5),
        defU=np.asarray([15000.0, 21000.0]),
        defV=np.asarray([14000.0, 20500.0]),
        defAngle=np.asarray([30.0, 80.0]),
        phase_shift=np.zeros(2),
        voltage=300.0,
        Cs=2.7,
        Q0=0.07,
        pixel_size=3.0,
        ori_size=box,
        current_image_size=12,
    )
    if "trial_ctf" not in relion_bind.vdam_expected_angular_errors.__doc__:
        pytest.skip("the RELION binding predates caller CTFs")
    plain = relion_bind.vdam_expected_angular_errors(**common)
    rows = np.asarray(
        [
            [15000.0, 14000.0, 30.0, 300.0, 2.7, 0.07, 0.0, 1.0, 0.0],
            [21000.0, 20500.0, 80.0, 300.0, 2.7, 0.07, 0.0, 1.0, 0.0],
        ]
    )
    trial_ctf = relion_ctf.relion_ctf_fftw_half(rows, box, 3.0)  # relax's CTF, in RELION's frame
    caller = relion_bind.vdam_expected_angular_errors(**common, trial_ctf=trial_ctf, projection_left=np.eye(3))
    assert_matches(np.asarray([caller["acc_rot"], caller["acc_trans"]]), np.asarray([plain["acc_rot"], plain["acc_trans"]]))


@pytest.mark.unit
@pytest.mark.parametrize("even, mag", [(None, None), (EVEN, None), (None, MAG), (EVEN, MAG)])
def test_relax_ctf_rows_match_relion_observation_model(tmp_path, even, mag):
    """relax's CTF (relion_ctf.relion_ctf_fftw_half) against RELION's CTF::getFftwImage via ObservationModel."""

    relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")
    if not hasattr(relion_bind, "optics_ctf_images_batch"):
        pytest.skip("the RELION binding predates the optics bindings")
    star = _write_star(tmp_path / "particles.star", tilt=None, odd=None, even=even, mag=mag, mag_both=True)
    dataset = SimpleNamespace(particles_file=str(star), image_shape=(BOX, BOX))
    rows = relion_ctf.relion_exact_ctf_half_from_source_star_host(dataset, np.asarray([0, 1, 2]), (BOX, BOX))
    params = np.asarray([[20000.0, 19000.0, 40.0, 0.0, 1.0, 0.0, g] for g in (2, 1, 2)])
    relion = np.asarray(relion_bind.optics_ctf_images_batch(str(star), params, BOX, BOX, False, 1))
    assert_matches(rows, np.stack([-np.fft.fftshift(r, axes=0).reshape(-1) for r in relion]), rtol=1e-12)


@pytest.mark.unit
def test_initial_model_takes_identity_magnification_columns_and_refuses_every_other_matrix(tmp_path):
    """An optics table whose ``rlnMagMat`` columns hold the exact identity is the same data as one without
    them: InitialModel's bootstrap gamma offsets are equal. Every other matrix is refused, also one inside
    ``Matrix2D::isIdentity``'s 1e-6: that tolerance only decides whether RELION magnifies the projections
    (acc_ml_optimiser_impl.h:1100), its CTF still applies the stored matrix (relax#66)."""

    def offsets(name, mag):
        star = _write_star(tmp_path / name, tilt=None, odd=None, even=EVEN, mag=mag)
        optics = starfile.read(star, always_dict=True)["optics"]
        refuse_unsupported_optics(optics, source=name, supported=INITIAL_MODEL_OPTICS_FEATURES)
        dataset = SimpleNamespace(particles_file=str(star), image_shape=(BOX, BOX))
        return bootstrap_iref._bootstrap_gamma_offsets(dataset, np.arange(3), BOX)

    groups, gamma = offsets("plain.star", None)
    identity_groups, identity_gamma = offsets("identity.star", np.eye(2))
    np.testing.assert_array_equal(identity_groups, groups)
    assert np.abs(gamma[2]).max() > 0.01 and gamma[1] is None and identity_gamma[1] is None
    assert_matches(identity_gamma[2], gamma[2], rtol=1e-13)
    refused = (("near.star", np.eye(2) + 5e-7), ("beyond.star", np.eye(2) + 2e-6), ("magnified.star", MAG))
    for name, mag in refused:
        with pytest.raises(NotImplementedError, match="rlnMagMat.*only identity magnification matrices"):
            offsets(name, mag)
    with pytest.raises(NotImplementedError, match=r"found \{'rlnMagMat00': \[1.0, 1.012\]"):
        offsets("values.star", MAG)
