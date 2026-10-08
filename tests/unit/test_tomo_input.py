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
def test_command_particle_loading_preserves_tilt_particle_identity(project, tmp_path, monkeypatch):
    from relax.refinement import particle_loading

    out, flat = project
    monkeypatch.setenv("RECOVAR_CACHE_DIR", "")
    monkeypatch.setenv("RELAX_USE_FLOAT64_SCORING", "0")
    args = SimpleNamespace(
        data_dir=str(out), output=str(tmp_path), preread_images=False,
        scratch_dir="", keep_free_scratch_gb=0.0, particle_diameter_ang=120.0,
        width_mask_edge_px=5.0, relion_softmask_reduction="control",
        image_fourier_backend="host_numpy", relion_init_dir=None, init_noise_from_npz=None,
    )
    loaded = particle_loading.load_particle_inputs(args)
    original_rows, _ = read_star(str(flat))
    original_index = tomo_input.tomo_particle_index(original_rows)
    particles, _ = read_star(str(out / "particles.star"))
    particle_names = np.asarray(star_column(particles, "rlnTomoParticleName"))
    visible_counts = [
        sum(int(value) for value in str(frames).strip("[]").split(","))
        for frames in star_column(particles, "rlnTomoVisibleFrames")
    ]
    assert loaded.tomographic and loaded.shape_class_rows is None
    assert loaded.dataset.n_units == original_index.n_particles
    assert loaded.dataset.images.n_units == original_index.n_images
    assert np.array_equal(loaded.dataset.particle_names, particle_names)
    assert np.array_equal(loaded.dataset.unit_image_offsets, np.r_[0, np.cumsum(visible_counts)])
    assert_matches(loaded.dataset.images.voxel_size, VOXEL)
    assert_matches(loaded.mask_parameters, (120.0, 5.0))


@pytest.mark.unit
def test_command_particle_loading_stages_subtomogram_stacks_to_scratch_dir(project, tmp_path, monkeypatch):
    """--scratch_dir copies the stacks of the per-tilt STAR; the particle STAR has no _rlnImageName (relax #15)."""
    import os
    import signal

    from relax.refinement import particle_loading

    out, _ = project
    monkeypatch.delenv("RECOVAR_CACHE_DIR", raising=False)
    monkeypatch.setenv("RELAX_USE_FLOAT64_SCORING", "0")
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    scratch_dir = tmp_path / "local"
    scratch_dir.mkdir()

    def load(name, scratch):
        args = SimpleNamespace(
            data_dir=str(out), output=str(tmp_path / name), preread_images=False,
            scratch_dir=scratch, keep_free_scratch_gb=0.0, particle_diameter_ang=120.0,
            width_mask_edge_px=5.0, relion_softmask_reduction="control",
            image_fourier_backend="host_numpy", relion_init_dir=None, init_noise_from_npz=None,
        )
        return particle_loading.load_particle_inputs(args).dataset

    streamed = load("streamed", "")
    staged = load("staged", str(scratch_dir))
    (run_dir,) = os.listdir(scratch_dir)
    assert run_dir.startswith("relax_volatile_")
    stacks = staged.images.image_source.backend.source.stack_files()
    assert stacks and all(path.startswith(str(scratch_dir / run_dir) + os.sep) for path in stacks)
    assert not any(path.startswith(str(scratch_dir)) for path in streamed.images.image_source.backend.source.stack_files())
    assert np.array_equal(staged.image_rows, streamed.image_rows)
    for unit in range(staged.n_units):
        assert_matches(staged.unit_images(unit), streamed.unit_images(unit))


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

    from relax.healpix_sampling import euler_angles_to_matrix

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
        a = euler_angles_to_matrix(angles[names.index(name)])[0]
        assert_matches(rotated.projections[start:stop], plain.projections[start:stop] @ a)


def test_subtomogram_runs_take_the_first_iteration_cross_correlation():
    """Subtomogram Refine3D and Class3D run RELION's CC iteration (score_tomo_half normalized_cc)."""

    from relax.refinement.command_options import validate_tomo_args

    args = SimpleNamespace(
        relion_init_dir=None, init_noise_from_npz=None, relion_softmask_reduction="control", firstiter_cc=False
    )
    validate_tomo_args(args, None, False)
    for n_classes in (1, 2):
        validate_tomo_args(SimpleNamespace(**{**vars(args), "firstiter_cc": True, "n_classes": n_classes}), None, False)
    with pytest.raises(SystemExit, match="soft-mask"):
        validate_tomo_args(SimpleNamespace(**{**vars(args), "relion_softmask_reduction": "probe"}), None, False)


@pytest.mark.unit
def test_subtomogram_runs_refuse_unqualified_optics_features(project):
    """Subtomogram Refine3D/Class3D refuse each optics feature until its per-tilt path is qualified.

    relion_refine applies all four to every tilt image; ignoring one silently gave wrong maps.
    """

    from relax.refinement.particle_loading import validate_particle_optics
    from relax.relion.relion_metadata import OPTICS_FEATURE_LABELS, TOMO_OPTICS_FEATURES

    out, _ = project
    _, optics = read_star(str(out / "particles.star"))
    validate_particle_optics(optics, tomographic=True)
    values = {
        "ctf_premultiplied": {"_rlnCtfDataAreCtfPremultiplied": 1},
        "odd_aberrations": {"_rlnBeamTiltX": 0.5},
        "even_aberrations": {"_rlnEvenZernike": "[0,0,0,0,40]"},
        "magnification": {"_rlnMagMat00": 1.01, "_rlnMagMat01": 0.0, "_rlnMagMat10": 0.0, "_rlnMagMat11": 1.0},
    }
    assert set(values) == set(OPTICS_FEATURE_LABELS)
    for feature, columns in values.items():
        table = optics.copy()
        for label, value in columns.items():
            table[label] = value
        validate_particle_optics(table, tomographic=False)
        if feature in TOMO_OPTICS_FEATURES:
            validate_particle_optics(table, tomographic=True)
        else:
            with pytest.raises(NotImplementedError, match="does not implement"):
                validate_particle_optics(table, tomographic=True)


@pytest.mark.unit
def test_flat_star_keeps_the_optics_features(project, tmp_path):
    """The per-tilt STAR carries each tilt's optics-group features, as relion_refine reads them per image."""

    from recovar.data_io import starfile

    out, _ = project
    particles, optics = read_star(str(out / "particles.star"))
    groups = np.asarray(star_column(optics, "rlnOpticsGroup"), dtype=np.int64)
    optics["_rlnCtfDataAreCtfPremultiplied"] = 1
    optics["_rlnOddZernike"] = [f"[0,0,{10 * g},0,0,-5]" for g in groups]
    optics["_rlnMagMat00"] = 1.0 + 0.01 * groups
    optics["_rlnTomoUnrelatedLabel"] = 7.0
    star = tmp_path / "particles_optics.star"
    starfile.write_star(str(star), particles, data_optics=optics)
    flat = tomo_input.flatten_relion5_tomo(star, out / "tomograms.star", tmp_path / "flat.star")
    _, flat_optics = read_star(str(flat))
    flat_groups = np.asarray(star_column(flat_optics, "rlnOpticsGroup"), dtype=np.int64)
    position = {int(g): i for i, g in enumerate(groups)}
    rows = [position[int(g)] for g in flat_groups]
    for label in ("rlnCtfDataAreCtfPremultiplied", "rlnOddZernike", "rlnMagMat00"):
        assert list(map(str, star_column(flat_optics, label))) == list(
            map(str, np.asarray(star_column(optics, label))[rows])
        )
    assert star_column(flat_optics, "rlnTomoUnrelatedLabel") is None


ABERRATED_OPTICS = dict(
    voltage=300.0,
    cs=2.7,
    amp_contrast=0.1,
    noise_scale=1.0,
    beam_tilt=(1.6, -1.2),
    odd_zernike=[0, 0, 40, 0, 0, -30],
    even_zernike=[0, 0, 0, 0, 400, 0, 0, 0, -300],
    mag_matrix=[[1.015, 0.004], [0.004, 0.99]],
)


@pytest.fixture(scope="module")
def aberrated_project(tmp_path_factory):
    """A simulated project whose second optics group has every aberration, CTF-premultiplied."""

    root = tmp_path_factory.mktemp("relion_tomo_optics")
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    with mrcfile.new(root / "vol0000.mrc") as mrc:
        mrc.set_data(np.exp(-((xx - 12) ** 2 + yy**2 + zz**2) / 200).astype(np.float32))
        mrc.voxel_size = VOXEL
    plain = {key: ABERRATED_OPTICS[key] for key in ("voltage", "cs", "amp_contrast", "noise_scale")}
    out = root / "project"
    relion_tomo.generate_relion5_tomo_dataset(
        str(out),
        str(root / "vol"),
        VOXEL,
        n_particles=4,
        grid_size=GRID,
        n_tomograms=2,
        optics_groups=[plain, ABERRATED_OPTICS],
        max_tilt=30.0,
        tilt_step=10.0,
        tomogram_size=(512, 512, 128),
        premultiplied_ctf=True,
        snr=0.05,
        seed=4,
    )
    particles, tomograms = tomo_input.read_optimisation_set(out / "optimisation_set.star")
    return tomo_input.flatten_relion5_tomo(particles, tomograms, root / "relax" / "particles_tilts.star")


def _inside_nyquist_half():
    import recovar.core.fourier_transform_utils as ftu

    freqs = np.asarray(ftu.get_k_coordinate_of_each_pixel_half((GRID, GRID), VOXEL, scaled=True), dtype=np.float64)
    return freqs, np.all(np.abs(freqs) < 0.5 / VOXEL, axis=-1)


@pytest.mark.unit
def test_exact_ctf_of_aberrated_tilts_matches_the_simulator(aberrated_project, monkeypatch):
    """Per tilt image: relax's exact CTF (even Zernike, magnification, premultiplied CTF^2) is the simulated one."""

    from recovar.core import CTFParamIndex

    flat = aberrated_project
    rows, _ = read_star(str(flat))
    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    relion_ctf.clear_exact_ctf_result_cache()
    dataset = SimpleNamespace(particles_file=str(flat))
    got = relion_ctf._relion_exact_ctf_half_from_source_star_host(dataset, np.arange(len(rows)), (GRID, GRID))
    group = np.asarray(star_column(rows, "rlnOpticsGroup"), dtype=np.int64)
    params = np.zeros((len(rows), 11))
    for column, label in (
        (CTFParamIndex.DFU, "rlnDefocusU"),
        (CTFParamIndex.DFV, "rlnDefocusV"),
        (CTFParamIndex.DFANG, "rlnDefocusAngle"),
        (CTFParamIndex.CONTRAST, "rlnCtfScalefactor"),
        (CTFParamIndex.DOSE, "rlnMicrographPreExposure"),
    ):
        params[:, column] = np.asarray(star_column(rows, label), dtype=np.float64)
    params[:, CTFParamIndex.VOLT], params[:, CTFParamIndex.CS], params[:, CTFParamIndex.W] = 300.0, 2.7, 0.1
    _, inside = _inside_nyquist_half()
    for g, optics in (
        (1, {}),
        (2, dict(mag_matrix=ABERRATED_OPTICS["mag_matrix"], even_zernike=ABERRATED_OPTICS["even_zernike"])),
    ):
        sel = group == g
        expected = np.asarray(relion_tomo.relion_tomo_ctf(params[sel], (GRID, GRID), VOXEL, half_image=True, **optics))
        # RELION squares its CTF row for premultiplied images; relax stores RELION's rows negated.
        np.testing.assert_allclose(got[sel][:, inside], -(expected**2)[:, inside], rtol=0, atol=1e-9)


@pytest.mark.unit
def test_odd_demodulation_of_aberrated_tilts_undoes_the_simulated_phase(aberrated_project, monkeypatch):
    """relax demodulates each tilt image by exp(-i phase) of its group's beam tilt and odd Zernike terms."""

    from relax.relion import optics_aberrations

    flat = aberrated_project
    rows, _ = read_star(str(flat))
    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    monkeypatch.setattr(optics_aberrations, "_ODD_PHASE_CACHE", {})
    dataset = SimpleNamespace(particles_file=str(flat), image_shape=(GRID, GRID))
    got = np.asarray(optics_aberrations.odd_demodulation_rows(dataset, np.arange(len(rows))))
    group = np.asarray(star_column(rows, "rlnOpticsGroup"), dtype=np.int64)
    freqs, inside = _inside_nyquist_half()
    odd = relion_tomo.odd_coefficients_with_beam_tilt(
        ABERRATED_OPTICS["odd_zernike"], ABERRATED_OPTICS["beam_tilt"], 2.7, 300.0
    )
    phase = relion_tomo.zernike_phase(
        odd, relion_tomo.odd_index_to_mn, freqs @ np.asarray(ABERRATED_OPTICS["mag_matrix"]).T
    )
    assert_matches(got[group == 1], np.ones_like(got[group == 1]))
    expected = np.broadcast_to(np.exp(-1j * phase)[inside], got[group == 2][:, inside].shape)
    np.testing.assert_allclose(got[group == 2][:, inside], expected, rtol=0, atol=1e-9)


@pytest.mark.unit
def test_uncached_fftw_ctf_rows_are_the_exact_rows(aberrated_project, monkeypatch):
    """relion_fftw_ctf_rows gives the exact operands' rows in RELION's frame, and the plain CTF on request."""

    flat = aberrated_project
    rows, _ = read_star(str(flat))
    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    relion_ctf.clear_exact_ctf_result_cache()
    dataset = SimpleNamespace(particles_file=str(flat))
    indices = np.arange(len(rows))[::-1]
    cached = relion_ctf._relion_exact_ctf_half_from_source_star_host(dataset, indices, (GRID, GRID))
    squared = relion_ctf.relion_fftw_ctf_rows(dataset, indices, (GRID, GRID))
    assert_matches(-np.fft.fftshift(squared, axes=1).reshape(len(rows), -1), cached)
    plain = relion_ctf.relion_fftw_ctf_rows(dataset, indices, (GRID, GRID), square_premultiplied=False)
    assert_matches(plain * plain, squared)
