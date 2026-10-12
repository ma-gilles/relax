"""relax reads a subtomogram project in the forms RELION's own tools write (relax#79).

Every tilt-series project relax was tested on came from RECOVAR's simulator, which writes each STAR block
as a ``loop_`` with every label the reader uses. RELION 5's ``relion_tomo_subtomo`` and tilt-series jobs
write ``data_general`` as label-value pairs, no ``rlnRandomSubset``, and tilt-series STARs without the
labels that sit at their default (``rlnCtfScalefactor``, ``rlnTomoXTilt``). The first real project
(the RELION-5 tutorial, tests/fixtures/relion5_tomo_tutorial) was not recognised as subtomograms at all.

Two inputs: RELION's own text from that project (no images: the per-tilt STAR needs none), and a
simulated project rewritten in RELION's forms (tests/helpers/relion_tomo_form.py), which must flatten to
the same per-tilt STAR as the simulated one.
"""

import shutil
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from helpers.relion_tomo_form import RELION_GENERAL, write_relion_form
from recovar.data_io.starfile import read_star, star_column
from recovar.simulation import relion_tomo

from relax.refinement.tomo_half import is_relion5_2d_stack_star
from relax.star import tomo_input

TUTORIAL = Path(__file__).resolve().parents[1] / "fixtures" / "relion5_tomo_tutorial"
# The particle STAR as relion_tomo_subtomo wrote it: RELION 5.0-beta-2 (shipped) and 5.0.1 (re-extracted).
RELION_PARTICLES = {"5.0-beta-2": "Extract/job010/particles.star", "5.0.1": "Extract_relion_5.0.1/job010/particles.star"}
GRID, VOXEL = 32, 4.0


def _visible(particles) -> list[int]:
    return [sum(int(v) for v in str(frames).strip("[]").split(",")) for frames in star_column(particles, "rlnTomoVisibleFrames")]


@pytest.fixture
def tutorial_project(tmp_path):
    """A data directory as the user guide lays it out: RELION's two STAR files and its project tree."""

    def build(version):
        root = tmp_path / f"project_{version}"
        shutil.copytree(TUTORIAL / "Denoise", root / "Denoise")
        shutil.copy(TUTORIAL / RELION_PARTICLES[version], root / "particles.star")
        shutil.copy(TUTORIAL / "Denoise" / "job008" / "tomograms.star", root / "tomograms.star")
        return root

    return build


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    root = tmp_path_factory.mktemp("relion_form")
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    with mrcfile.new(root / "vol0000.mrc") as mrc:
        mrc.set_data((np.exp(-((xx - 12) ** 2 + yy**2 + zz**2) / 200)).astype(np.float32))
        mrc.voxel_size = VOXEL
    relion_tomo.generate_relion5_tomo_dataset(
        str(root / "project"), str(root / "vol"), VOXEL, n_particles=6, grid_size=GRID, n_tomograms=2,
        max_tilt=30.0, tilt_step=10.0, tomogram_size=(512, 512, 128), hidden_tilt_fraction=0.3, snr=0.05, seed=3,
    )  # fmt: skip
    return root / "project", write_relion_form(root / "project", root / "relion_form")


@pytest.mark.unit
@pytest.mark.parametrize("version", sorted(RELION_PARTICLES))
def test_relions_own_particle_star_is_recognised_as_2d_stacks(version):
    """``data_general`` as label-value pairs under ``# version 50001``, RELION's text unchanged."""

    path = TUTORIAL / RELION_PARTICLES[version]
    assert path.read_text().startswith(RELION_GENERAL)
    assert is_relion5_2d_stack_star(path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "general, expected",
    [
        ("data_general\n\n_rlnTomoSubTomosAre2DStacks                       1\n \n", True),
        ("data_general\n\n_rlnTomoSubTomosAre2DStacks 0\n\n", False),
        ("data_general\n\nloop_\n_rlnTomoSubTomosAre2DStacks #1\n1\n\n", True),
        ("data_general\n\nloop_\n_rlnTomoSubTomosAre2DStacks\n0\n\n", False),
        ("data_general\n\n_rlnSomethingElse 1\n\n", False),
        ("", False),
    ],
)
def test_2d_stack_flag_is_read_from_a_list_and_from_a_loop(tmp_path, general, expected):
    star = tmp_path / "particles.star"
    star.write_text(
        "\n# version 50001\n\n" + general + "\n# version 50001\n\ndata_particles\n\nloop_ \n"
        "_rlnTomoName #1 \n_rlnImageName #2 \n TS_01 Subtomograms/TS_01/1_stack2d.mrcs \n \n"
    )
    assert is_relion5_2d_stack_star(star) is expected


@pytest.mark.unit
@pytest.mark.parametrize("version", sorted(RELION_PARTICLES))
def test_relions_own_project_flattens_to_one_row_per_visible_tilt(tutorial_project, tmp_path, version):
    """RELION's particle, tomogram and tilt-series STARs go through the reader as written.

    The particle STAR has no ``rlnRandomSubset``; TS_03's tilt-series STAR has no ``rlnCtfScalefactor``
    (1 in RELION, TomogramSet::loadTomogram), TS_01's has it; the tomogram STAR is tab-separated text
    of the ``starfile`` package with columns the reader does not know.
    """

    root = tutorial_project(version)
    before = {path: path.read_bytes() for path in root.rglob("*.star")}
    flat = tomo_input.flatten_relion5_tomo(root / "particles.star", root / "tomograms.star", tmp_path / "run" / "flat.star")
    assert {path: path.read_bytes() for path in root.rglob("*.star")} == before  # nothing written into the project

    rows, optics = read_star(str(flat))
    particles, _ = read_star(str(root / "particles.star"))
    index = tomo_input.tomo_particle_index(rows)
    names = np.asarray(star_column(particles, "rlnTomoParticleName")).astype(str)
    assert sorted(index.particle_names.astype(str)) == sorted(names) and names.size == 6
    visible = dict(zip(names, _visible(particles)))
    assert [int(n) for n in np.diff(index.image_offsets)] == [visible[str(name)] for name in index.particle_names]
    assert index.n_images == len(rows) == sum(visible.values())
    assert np.all(index.half_set == 0)  # RELION's "not assigned": the run draws the halves

    scale = np.asarray(star_column(rows, "rlnCtfScalefactor"), dtype=np.float64)
    tomo_of_row = np.repeat([str(name).split("/")[0] for name in index.particle_names], np.diff(index.image_offsets))
    assert np.all(scale[tomo_of_row == "TS_03"] == 1.0)
    series, _ = read_star(str(root / "Denoise/job008/tilt_series/TS_01.star"))
    written = set(np.round(np.asarray(star_column(series, "rlnCtfScalefactor"), dtype=np.float64), 6))
    assert set(np.round(scale[tomo_of_row == "TS_01"], 6)) <= written and scale[tomo_of_row == "TS_01"].min() < 1.0
    assert float(np.asarray(star_column(optics, "rlnCtfDataAreCtfPremultiplied"))[0]) == 1.0

    geometry = tomo_input.relion_image_geometry(rows, index, root / "particles.star", root / "tomograms.star")
    assert geometry.projections.shape == (index.n_images, 3, 3)
    assert np.allclose(np.einsum("nij,nkj->nik", geometry.projections, geometry.projections), np.eye(3), atol=1e-12)


@pytest.mark.unit
def test_reader_copies_write_out_relions_defaults_and_nothing_else(tutorial_project, tmp_path):
    root = tutorial_project("5.0-beta-2")
    particles, tomograms = tomo_input._reader_inputs(
        root / "particles.star", root / "tomograms.star", tmp_path / "run" / "flat.star"
    )
    assert particles.parent == tomograms.parent == tmp_path / "run"

    def rows(path, block):
        lines = Path(path).read_text().splitlines()
        labels, first, end = tomo_input._star_loop_bounds(lines, block)
        return [lines[i].split()[0] for i in range(labels, first)], [lines[i].split() for i in range(first, end)]

    labels, data = rows(particles, "particles")
    source_labels, source = rows(root / "particles.star", "particles")
    assert labels == source_labels + ["_rlnRandomSubset"] and [row[:-1] for row in data] == source
    assert {row[-1] for row in data} == {"0"}

    _, tomo_rows = rows(tomograms, "global")
    _, source_rows = rows(root / "tomograms.star", "global")
    series_column = 8  # rlnTomoTiltSeriesStarFile, the ninth label of the shipped file
    for row, source_row in zip(tomo_rows, source_rows, strict=True):
        assert row[:series_column] + row[series_column + 1 :] == source_row[:series_column] + source_row[series_column + 1 :]
        assert Path(row[series_column]).is_absolute() and Path(row[series_column]).is_file()
    ts01, ts03 = Path(tomo_rows[0][series_column]), Path(tomo_rows[1][series_column])
    # TS_01's STAR has every label, so the original is named; TS_03's copy gets rlnCtfScalefactor 1.
    assert ts01 == (root / "Denoise/job008/tilt_series/TS_01.star").resolve()
    labels, data = rows(ts03, "TS_03")
    source_labels, source = rows(root / "Denoise/job008/tilt_series/TS_03.star", "TS_03")
    assert labels == source_labels + ["_rlnCtfScalefactor"] and [row[:-1] for row in data] == source
    assert {float(row[-1]) for row in data} == {1.0}


@pytest.mark.unit
def test_complete_project_is_read_in_place(simulated, tmp_path):
    project, _ = simulated
    out = tmp_path / "run" / "flat.star"
    particles, tomograms = tomo_input._reader_inputs(project / "particles.star", project / "tomograms.star", out)
    assert (particles, tomograms) == (project / "particles.star", project / "tomograms.star")
    assert not out.parent.exists() or not list(out.parent.iterdir())


@pytest.mark.unit
def test_simulated_project_in_relions_forms_flattens_to_the_same_rows(simulated, tmp_path):
    """The simulated project rewritten as RELION writes it (list ``data_general``, no ``rlnRandomSubset``,
    no ``rlnTomoXTilt``) gives the per-tilt STAR of the simulated one, apart from the half sets."""

    project, relion_form = simulated
    text = (relion_form / "particles.star").read_text()
    assert text.startswith(RELION_GENERAL) and "_rlnRandomSubset" not in text
    assert all("_rlnTomoXTilt" not in path.read_text() for path in (relion_form / "tilt_series").glob("*.star"))
    assert is_relion5_2d_stack_star(relion_form / "particles.star")

    flat = {}
    for name, root in (("simulated", project), ("relion", relion_form)):
        flat[name] = read_star(
            str(tomo_input.flatten_relion5_tomo(root / "particles.star", root / "tomograms.star", tmp_path / name / "flat.star"))
        )
    (rows, optics), (relion_rows, relion_optics) = flat["simulated"], flat["relion"]
    assert list(rows.columns) == list(relion_rows.columns) and len(rows) == len(relion_rows)
    for column in rows.columns:
        if str(column).lstrip("_") == "rlnRandomSubset":
            assert set(np.asarray(relion_rows[column]).astype(int)) == {0}
        else:
            assert np.array_equal(np.asarray(rows[column]), np.asarray(relion_rows[column])), column
    assert optics.equals(relion_optics)


@pytest.mark.unit
def test_simulated_project_in_relions_forms_loads_the_same_dataset(simulated, tmp_path):
    """What the engine is handed from RELION's forms is, bit for bit, what it is handed from the simulated
    project: the units and their tilt images, RELION's ``Aproj`` and left matrix per image, the frames,
    tomograms and optics groups, the CTF rows and every pixel."""

    from relax.refinement.tomo_half import load_tomo_dataset

    project, relion_form = simulated
    loaded = {
        name: load_tomo_dataset(
            root / "particles.star", root / "tomograms.star", tmp_path / name / "particles_2d.star", datadir=str(root), lazy=True
        )
        for name, root in (("simulated", project), ("relion", relion_form))
    }
    a, b = loaded["simulated"], loaded["relion"]
    assert (a.n_units, a.image_shape, a.volume_shape, a.voxel_size) == (b.n_units, b.image_shape, b.volume_shape, b.voxel_size)
    for field in (
        "unit_image_offsets", "image_rows", "image_projections", "image_left", "image_frames",
        "unit_tomogram", "unit_optics_group", "particle_names",
    ):  # fmt: skip
        assert np.array_equal(getattr(a, field), getattr(b, field)), field
    assert np.array_equal(np.asarray(a.images.CTF_params), np.asarray(b.images.CTF_params))
    for unit in range(a.n_units):
        assert np.array_equal(np.asarray(a.unit_images(unit)), np.asarray(b.unit_images(unit))), unit
