"""RELION 5 subtomogram input: per-tilt rows, particle index and the tomo CTF.

The dataset is RECOVAR's simulated RELION 5 project (``relion_tomo``), whose images a
RELION 5.0.1 Refine3D recovers (the cryo-ET S1 gate), so its CTF is an independent
reference for relax's exact RELION CTF operand with dose damping.
"""

from types import SimpleNamespace

import mrcfile
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.data_io.starfile import read_star, star_column
from recovar.simulation import relion_tomo
from scipy.spatial.transform import Rotation

from relax.relion import relion_ctf, tomo_input

GRID = 32
VOXEL = 4.0


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("relion_tomo_input")
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    vol = np.exp(-((xx - 12) ** 2 + yy**2 + zz**2) / 200) + 0.5 * np.exp(-(xx**2 + (yy + 16) ** 2 + zz**2) / 100)
    with mrcfile.new(root / "vol0000.mrc") as mrc:
        mrc.set_data(vol.astype(np.float32))
        mrc.voxel_size = VOXEL
    out = root / "project"
    relion_tomo.generate_relion5_tomo_dataset(
        str(out),
        str(root / "vol"),
        VOXEL,
        n_particles=6,
        grid_size=GRID,
        n_tomograms=2,
        max_tilt=30.0,
        tilt_step=10.0,
        tomogram_size=(512, 512, 128),
        hidden_tilt_fraction=0.3,
        snr=0.05,
        seed=3,
    )
    particles, tomograms = tomo_input.read_optimisation_set(out / "optimisation_set.star")
    flat = tomo_input.flatten_relion5_tomo(particles, tomograms, root / "relax" / "particles_tilts.star")
    return out, flat


@pytest.mark.unit
def test_relion_tomo_damping_matches_relion_formula():
    freq_sq = np.array([0.0, 1e-4, 1e-2, 0.03])
    ne = 0.245 * np.power(freq_sq[1:], -0.8325) + 2.81
    expected = np.r_[1.0, np.exp(-0.5 * 12.0 / ne)]
    np.testing.assert_allclose(tomo_input.relion_tomo_damping(freq_sq, 12.0), expected, rtol=1e-15)
    np.testing.assert_allclose(
        tomo_input.relion_tomo_damping(freq_sq, 12.0, bfactor_per_electron_dose=4.0),
        np.exp(-0.25 * 48.0 * freq_sq),
        rtol=1e-15,
    )


@pytest.mark.unit
def test_particle_index_follows_particles_star(project):
    out, flat = project
    rows, optics = read_star(str(flat))
    index = tomo_input.tomo_particle_index(rows)
    particles, _ = read_star(str(out / "particles.star"))

    names = np.asarray(star_column(particles, "rlnTomoParticleName"))
    order = {name: p for p, name in enumerate(names)}
    visible = [
        sum(int(v) for v in str(frames).strip("[]").split(","))
        for frames in star_column(particles, "rlnTomoVisibleFrames")
    ]
    assert index.n_particles == len(names)
    assert index.n_images == len(rows) == sum(visible)
    for p, name in enumerate(index.particle_names):
        source = order[name]
        assert np.diff(index.image_offsets)[p] == visible[source]
        assert index.half_set[p] == int(star_column(particles, "rlnRandomSubset").iloc[source])
        assert index.optics_group[p] == int(star_column(particles, "rlnOpticsGroup").iloc[source])
    dose = np.asarray(star_column(rows, "rlnMicrographPreExposure"), dtype=np.float64)
    for p in range(index.n_particles):
        assert np.all(np.diff(dose[index.image_offsets[p] : index.image_offsets[p + 1]]) > 0)
    assert len(optics) == 2


@pytest.mark.unit
def test_exact_ctf_matches_simulated_tomo_ctf(project, monkeypatch):
    """relax's exact CTF of each tilt row equals the CTF the images were made with."""

    pytest.importorskip("relax.relion_bind._relion_bind_core")
    out, flat = project
    rows, optics = read_star(str(flat))
    monkeypatch.setenv("RELAX_K1_RELION_EXACT_CTF_STAR", str(flat))
    relion_ctf.clear_exact_ctf_result_cache()
    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    dataset = SimpleNamespace(particles_file=str(flat))
    indices = np.arange(len(rows), dtype=np.int64)
    got = relion_ctf._relion_exact_ctf_half_from_source_star_host(dataset, indices, (GRID, GRID))

    # The simulator's CTF parameters for the same rows, in recovar's CTF layout.
    ctf_params = np.zeros((len(rows), 11))
    group = np.asarray(star_column(rows, "rlnOpticsGroup"), dtype=np.int64)
    by_group = {int(g): r for g, (_, r) in zip(star_column(optics, "rlnOpticsGroup"), optics.iterrows())}

    def optics_col(label):
        return np.array([float(by_group[g][f"_{label}"]) for g in group])

    from recovar.core import CTFParamIndex

    ctf_params[:, CTFParamIndex.DFU] = np.asarray(star_column(rows, "rlnDefocusU"), dtype=np.float64)
    ctf_params[:, CTFParamIndex.DFV] = np.asarray(star_column(rows, "rlnDefocusV"), dtype=np.float64)
    ctf_params[:, CTFParamIndex.DFANG] = np.asarray(star_column(rows, "rlnDefocusAngle"), dtype=np.float64)
    ctf_params[:, CTFParamIndex.VOLT] = optics_col("rlnVoltage")
    ctf_params[:, CTFParamIndex.CS] = optics_col("rlnSphericalAberration")
    ctf_params[:, CTFParamIndex.W] = optics_col("rlnAmplitudeContrast")
    ctf_params[:, CTFParamIndex.CONTRAST] = np.asarray(star_column(rows, "rlnCtfScalefactor"), dtype=np.float64)
    ctf_params[:, CTFParamIndex.DOSE] = np.asarray(star_column(rows, "rlnMicrographPreExposure"), dtype=np.float64)
    expected = relion_tomo.relion_tomo_ctf(ctf_params, (GRID, GRID), VOXEL, half_image=True)
    assert expected.dtype == np.float64
    expected = np.asarray(expected)
    assert got.shape == expected.shape
    # The Nyquist row and column differ by convention, not by dose: RELION's FFTW grid
    # puts +N/2 there, RECOVAR's half grid -N/2, and the astigmatic CTF is not
    # symmetric under one sign flip. Compare every other pixel.
    import recovar.core.fourier_transform_utils as ftu

    freqs = np.asarray(ftu.get_k_coordinate_of_each_pixel_half((GRID, GRID), VOXEL, scaled=True))
    inside = np.all(np.abs(freqs) < 0.5 / VOXEL, axis=-1)
    assert inside.sum() == (GRID - 1) * (GRID // 2)
    np.testing.assert_allclose(got[:, inside], expected[:, inside], rtol=0, atol=1e-10)


def _gl_rotation(axis, degrees):
    """gravis t3Matrix::rotation (t3Matrix.h:478-496), written out as RELION does."""
    n = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    a = np.deg2rad(degrees)
    s = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    nnt = np.outer(n, n)
    return nnt + np.cos(a) * (np.eye(3) - nnt) + np.sin(a) * s


@pytest.mark.unit
def test_tilt_rotations_follow_relion_projection_matrix():
    x, y, z = np.array([3.0, -1.5]), np.array([-60.0, 42.0]), np.array([85.3, -91.2])
    expected = np.stack(
        [
            _gl_rotation((0, 0, 1), z[i]) @ _gl_rotation((0, 1, 0), y[i]) @ _gl_rotation((1, 0, 0), x[i])
            for i in range(2)
        ]
    )
    assert_matches(tomo_input.relion_tilt_rotations(x, y, z), expected)


def _geometry(project, particles_star=None):
    out, flat = project
    rows, _ = read_star(str(flat))
    index = tomo_input.tomo_particle_index(rows)
    particles = particles_star or out / "particles.star"
    return rows, index, tomo_input.relion_image_geometry(rows, index, particles, out / "tomograms.star")


@pytest.mark.unit
def test_image_geometry_is_relion_frame_order(project):
    out, _ = project
    rows, index, geometry = _geometry(project)
    particles, _ = read_star(str(out / "particles.star"))
    tomograms, _ = read_star(str(out / "tomograms.star"))
    names = list(star_column(particles, "rlnTomoParticleName"))
    series_file = dict(zip(star_column(tomograms, "rlnTomoName"), star_column(tomograms, "rlnTomoTiltSeriesStarFile")))
    reordered = False
    for p, name in enumerate(index.particle_names):
        start, stop = index.image_offsets[p], index.image_offsets[p + 1]
        s = names.index(name)
        visible = [
            f
            for f, v in enumerate(str(star_column(particles, "rlnTomoVisibleFrames").iloc[s]).strip("[]").split(","))
            if int(v) == 1
        ]
        assert geometry.frames[start:stop].tolist() == visible
        assert sorted(geometry.rows[start:stop].tolist()) == list(range(start, stop))
        reordered |= geometry.rows[start:stop].tolist() != list(range(start, stop))
        table, _ = read_star(str(out / series_file[star_column(particles, "rlnTomoName").iloc[s]]))
        # The converter's own tilt rotations (scipy, Rz Ry Rx) are an independent reference.
        tilts = Rotation.from_euler(
            "ZYX",
            np.stack(
                [
                    np.asarray(star_column(table, "rlnTomoZRot"), dtype=np.float64),
                    np.asarray(star_column(table, "rlnTomoYTilt"), dtype=np.float64),
                    np.asarray(star_column(table, "rlnTomoXTilt"), dtype=np.float64),
                ],
                axis=1,
            ),
            degrees=True,
        ).as_matrix()
        assert_matches(geometry.projections[start:stop], tilts[visible])
        micrographs = np.asarray(star_column(rows, "rlnMicrographName")).astype(str)[geometry.rows[start:stop]]
        assert micrographs.tolist() == [str(m) for m in np.asarray(star_column(table, "rlnMicrographName"))[visible]]
    # The simulated tilt scheme is dose-symmetric, so frame order differs from the per-tilt STAR's dose order.
    assert reordered


@pytest.mark.unit
def test_image_geometry_applies_the_subtomogram_matrix(project, tmp_path):
    from recovar.data_io import starfile

    from relax.sampling import _relion_euler_angles_to_matrix

    out, _ = project
    particles, optics = read_star(str(out / "particles.star"))
    angles = np.array([[10.0, 35.0, -20.0]]) + np.arange(len(particles))[:, None]
    for label, column in zip(("Rot", "Tilt", "Psi"), angles.T):
        particles[f"_rlnTomoSubtomogram{label}"] = column
    star = tmp_path / "particles_subtomo.star"
    starfile.write_star(str(star), particles, data_optics=optics)
    _, index, plain = _geometry(project)
    _, _, rotated = _geometry(project, star)
    names = list(star_column(particles, "rlnTomoParticleName"))
    for p, name in enumerate(index.particle_names):
        start, stop = index.image_offsets[p], index.image_offsets[p + 1]
        a = _relion_euler_angles_to_matrix(angles[names.index(name)])[0]
        assert_matches(rotated.projections[start:stop], plain.projections[start:stop] @ a)
