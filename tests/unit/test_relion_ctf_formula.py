"""relax's host CTF (``relion_ctf.relion_ctf_fftw_half``) against RELION's own CTF code.

relax evaluates RELION's ``CTF::getFftwImage`` in its own code; the RELION binding is
the oracle here only. Covered: the plain CTF (``get_ctf_images_batch``: CTF::setValues),
its square (the Fctf RELION uses for CTF-premultiplied images, ml_optimiser.cpp:6486-6492),
and the optics-table terms through RELION's ObservationModel (``optics_ctf_images_batch``:
CTF::setValuesByGroup) with even Zernike terms, anisotropic magnification and both.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.relion import relion_ctf

relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")

# RELION's gamma reaches hundreds of radians at the edge of the grid; one double
# rounding of it is about 1e-13 there, so the CTF agrees to about 1e-12 of its range.
RTOL = 1e-11
EVEN = [0.0, 0.2, -0.1, 0.3, 0.05, -0.02, 0.01, 0.1, 0.02]
MAG = np.asarray([[1.012, 0.006], [-0.004, 0.991]])


def _params(rng, n):
    """defU, defV, defAng, voltage, Cs, Q0, Bfac, scale, phase shift."""

    du = rng.uniform(5000.0, 35000.0, n)
    return np.stack(
        [
            du,
            du + rng.uniform(-1500.0, 1500.0, n),
            rng.uniform(-180.0, 180.0, n),
            rng.choice([200.0, 300.0], n),
            rng.choice([0.01, 2.0, 2.7], n),
            rng.choice([0.07, 0.1], n),
            rng.choice([0.0, 0.0, 60.0, 150.0], n),
            rng.uniform(0.8, 1.2, n),
            rng.choice([0.0, 0.0, 35.0], n),
        ],
        axis=1,
    )


def _relion_plain(params, box, pixel):
    batch = np.stack(
        [
            params[:, 0],
            params[:, 1],
            params[:, 2],
            params[:, 3],
            params[:, 4],
            params[:, 5],
            params[:, 6],
            np.full(len(params), pixel),
            params[:, 8],
            params[:, 7],
        ],
        axis=1,
    )
    return np.asarray(relion_bind.get_ctf_images_batch(batch, box, box, False, False, True, 1))


@pytest.mark.unit
@pytest.mark.parametrize("box, pixel", [(16, 3.0), (64, 1.3), (128, 0.85)])
def test_plain_ctf_matches_relion(box, pixel):
    params = _params(np.random.default_rng(box), 12)
    assert_matches(relion_ctf.relion_ctf_fftw_half(params, box, pixel), _relion_plain(params, box, pixel), rtol=RTOL)


@pytest.mark.unit
def test_premultiplied_fctf_is_relions_square():
    params = _params(np.random.default_rng(3), 8)
    relax = relion_ctf.relion_ctf_fftw_half(params, 64, 1.3)
    relion = _relion_plain(params, 64, 1.3)
    assert_matches(relax * relax, relion * relion, rtol=RTOL)


def _optics_star(path, box, pixel, *, even=None, mag=None):
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
    values = ["1", "og1", "300", "2.7", "0.07", str(pixel), str(box), "2"]
    if even is not None:
        labels.append("_rlnEvenZernike")
        values.append("[" + ",".join(str(c) for c in even) + "]")
    if mag is not None:
        labels += ["_rlnMagMat00", "_rlnMagMat01", "_rlnMagMat10", "_rlnMagMat11"]
        values += [str(v) for v in np.asarray(mag).reshape(-1)]
    path.write_text("data_optics\n\nloop_\n" + "\n".join(labels) + "\n" + " ".join(values) + "\n")
    return path


@pytest.mark.unit
@pytest.mark.parametrize("even, mag", [(None, None), (EVEN, None), (None, MAG), (EVEN, MAG)])
def test_optics_table_ctf_matches_relion_observation_model(tmp_path, even, mag):
    if not hasattr(relion_bind, "optics_ctf_images_batch"):
        pytest.skip("the RELION binding predates the optics oracles")
    box, pixel = 48, 1.4
    star = _optics_star(tmp_path / "optics.star", box, pixel, even=even, mag=mag)
    params = _params(np.random.default_rng(7), 6)
    params[:, 3:6] = (300.0, 2.7, 0.07)  # the optics group's
    gamma = None if even is None else np.asarray(relion_bind.optics_gamma_offset(str(star), 1, box))
    relax = relion_ctf.relion_ctf_fftw_half(params, box, pixel, gamma_offset=gamma, mag_matrix=mag)
    by_group = np.stack(
        [params[:, 0], params[:, 1], params[:, 2], params[:, 6], params[:, 7], params[:, 8], np.ones(6)], axis=1
    )
    relion = np.asarray(relion_bind.optics_ctf_images_batch(str(star), by_group, box, box, True, 1))
    assert_matches(relax, relion, rtol=RTOL)
    if even is not None or mag is not None:
        assert np.abs(relax - relion_ctf.relion_ctf_fftw_half(params, box, pixel)).max() > 1e-3


@pytest.mark.unit
def test_exact_ctf_rows_are_relions_in_recovar_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    box, pixel = 32, 2.0
    star = tmp_path / "particles.star"
    star.write_text(
        "data_optics\n\nloop_\n_rlnOpticsGroup\n_rlnVoltage\n_rlnSphericalAberration\n_rlnAmplitudeContrast\n"
        f"_rlnImagePixelSize\n_rlnImageSize\n1 300 2.7 0.07 {pixel} {box}\n2 200 2.0 0.1 {pixel} {box}\n\n"
        "data_particles\n\nloop_\n_rlnImageName\n_rlnDefocusU\n_rlnDefocusV\n_rlnDefocusAngle\n_rlnPhaseShift\n"
        "_rlnCtfScalefactor\n_rlnCtfBfactor\n_rlnOpticsGroup\n"
        "1@s.mrcs 21000 20000 30 0 1.0 0 1\n2@s.mrcs 15000 15800 110 20 0.9 80 2\n"
    )
    dataset = SimpleNamespace(particles_file=str(star))
    rows = relion_ctf._relion_exact_ctf_half_from_source_star_host(dataset, np.asarray([1, 0]), (box, box))
    params = np.asarray(
        [
            [15000.0, 15800.0, 110.0, 200.0, 2.0, 0.1, 80.0, 0.9, 20.0],
            [21000.0, 20000.0, 30.0, 300.0, 2.7, 0.07, 0.0, 1.0, 0.0],
        ]
    )
    relion = _relion_plain(params, box, pixel)
    # RECOVAR's frame: centered rows, opposite sign (relion_ctf._evaluate_exact_ctf_rows).
    assert_matches(rows, -np.fft.fftshift(relion, axes=1).reshape(2, -1), rtol=RTOL)
