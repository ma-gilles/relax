"""relax's exact RELION CTF rows against RELION's own CTF code, on real particle tables.

The rows come from one float64 JAX program (``relion_ctf._relion_ctf_program``), evaluated per request and
kept by no cache. The oracle is the RELION binding (``optics_ctf_images_batch``: ``CTF::setValuesByGroup``
and ``getFftwImage`` with RELION's ObservationModel reading the same STAR), which shares no code with
relax. The tables are EMPIAR-10097's particle STAR and the per-tilt STAR of a RELION 5 subtomogram
project; the optics terms no pinned real table carries (anisotropic magnification, even Zernike terms, a
CTF-premultiplied group) are added to EMPIAR-10097's optics table.

Two tolerances. Against RELION (``RTOL``): RELION's gamma reaches hundreds of radians at the edge of the
grid and its compiled code may round it once differently (a contracted multiply-add), about 1e-13 there,
so the CTF agrees to about 1e-12 of its range; test_relion_ctf_formula.py uses the same value. Against a
strict IEEE evaluation of the same expression with glibc (``helpers/relion_ctf_numpy_reference.py``) the
program differs only in the last unit of ``sin`` and ``exp``: measured at most 7.4e-16 relative on a GPU
and 5.8e-16 on the CPU backend, checked here at 1e-15 on whichever backend the test runs.
"""

from types import SimpleNamespace

import jax
import numpy as np
import pandas as pd
import pytest
from helpers import relion_ctf_numpy_reference as reference
from helpers.em_fixtures import fixture_file
from helpers.float_compare import assert_matches
from recovar.data_io.starfile import read_star, star_column

from relax.relion import relion_ctf, tomo_input

relion_bind = pytest.importorskip("relax.relion_bind._relion_bind_core")

pytestmark = [pytest.mark.unit, pytest.mark.requires_relion_bind]

# See the module docstring.
RTOL = 1e-11
EVEN = [0.0, 0.2, -0.1, 0.3, 0.05, -0.02, 0.01, 0.1, 0.02]
MAG = {"rlnMagMat00": 1.012, "rlnMagMat01": 0.006, "rlnMagMat10": -0.004, "rlnMagMat11": 0.991}


def _plain(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(columns=lambda label: str(label).lstrip("_"))


def _write_star(path, optics: pd.DataFrame, particles: pd.DataFrame):
    """A RELION STAR of these optics groups and particle rows."""

    def block(name, table):
        lines = [f"data_{name}", "", "loop_"] + [f"_{label}" for label in table.columns]
        lines += [" ".join(str(value) for value in row) for row in table.itertuples(index=False)]
        return "\n".join(lines) + "\n\n"

    path.write_text(block("optics", optics) + block("particles", particles))
    return path


def _column(table, label, default):
    return np.asarray(table[label], dtype=np.float64) if label in table else np.full(len(table), float(default))


def _relion_rows(star, particles, indices, size, *, tomo=False):
    """RELION's ``Fctf`` of these particles: getFftwImage with damping, squared for a premultiplied group
    (ml_optimiser.cpp:6461-6492); a tilt image is damped by its dose, or by ``BfactorPerElectronDose * dose``
    as a B-factor when that is positive (exp_model.cpp:228-237)."""

    rows = particles.iloc[np.asarray(indices)]
    bfactor, dose = _column(rows, "rlnCtfBfactor", 0.0), np.full(len(rows), -1.0)
    if tomo:
        exposure = _column(rows, "rlnMicrographPreExposure", 0.0)
        per_dose = _column(rows, "rlnCtfBfactorPerElectronDose", 0.0)
        bfactor = np.where(per_dose > 0.0, per_dose * exposure, 0.0)
        dose = np.where(per_dose > 0.0, -999.0, exposure)
    params = np.stack(
        [
            _column(rows, "rlnDefocusU", np.nan),
            _column(rows, "rlnDefocusV", np.nan),
            _column(rows, "rlnDefocusAngle", np.nan),
            bfactor,
            _column(rows, "rlnCtfScalefactor", 1.0),
            _column(rows, "rlnPhaseShift", 0.0),
            _column(rows, "rlnOpticsGroup", np.nan),
            dose,
        ],
        axis=1,
    )
    ctf = np.asarray(relion_bind.optics_ctf_images_batch(str(star), params, size, size, True, 1))
    _, optics = read_star(str(star))
    flags = star_column(optics, "rlnCtfDataAreCtfPremultiplied")
    if flags is not None:
        premultiplied = dict(zip(np.asarray(star_column(optics, "rlnOpticsGroup"), dtype=int), np.asarray(flags, dtype=float)))
        squared = np.asarray([premultiplied[int(group)] != 0.0 for group in params[:, 6]])
        ctf = np.where(squared[:, None, None], ctf * ctf, ctf)
    return ctf


def _recovar_frame(fftw_rows):
    """RECOVAR's half rows of RELION's: centred rows, opposite sign, flattened."""

    return -np.fft.fftshift(fftw_rows, axes=1).reshape(fftw_rows.shape[0], -1)


@pytest.fixture
def source(monkeypatch):
    """``source(star)``: a dataset reading that STAR through an empty source cache."""

    monkeypatch.setattr(relion_ctf, "_RELION_EXACT_CTF_SOURCE_CACHE", {})
    monkeypatch.delenv("RELAX_K1_RELION_EXACT_CTF_STAR", raising=False)
    return lambda star: SimpleNamespace(particles_file=str(star))


@pytest.fixture(scope="module")
def empiar_10097():
    """``(star, particles, optics, size)`` of EMPIAR-10097's particle STAR."""

    star = fixture_file("empiar_10097_hp3_state", "input/particles.star")
    particles, optics = read_star(str(star))
    return star, _plain(particles), _plain(optics), int(star_column(optics, "rlnImageSize", required=True)[0])


@pytest.fixture(scope="module")
def tilt_table(tmp_path_factory):
    """``(particles, optics, size)``: the per-tilt rows of a RELION 5 subtomogram project, with each tilt
    image's cumulative dose."""

    flat = tomo_input.flatten_relion5_tomo(
        fixture_file("cryoet_et09_box64_data", "particles.star"),
        fixture_file("cryoet_et09_box64_data", "tomograms.star"),
        tmp_path_factory.mktemp("tilts") / "particles_tilts.star",
    )
    particles, optics = read_star(str(flat))
    return _plain(particles), _plain(optics), int(star_column(optics, "rlnImageSize", required=True)[0])


def _optics_variant(empiar_10097, tmp_path, *, even=None, mag=None, premultiplied_second_group=False, n=12):
    """EMPIAR-10097 particles under its optics table with the given terms added."""

    _, particles, optics, size = empiar_10097
    rows = particles.iloc[np.random.default_rng(4).choice(len(particles), n, replace=False)].reset_index(drop=True)
    rows = rows[["rlnImageName", "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnPhaseShift", "rlnOpticsGroup"]]
    optics = optics.copy()
    if even is not None:
        optics["rlnEvenZernike"] = "[" + ",".join(str(c) for c in even) + "]"
    if mag is not None:
        for label, value in mag.items():
            optics[label] = value
    if premultiplied_second_group:
        second = optics.copy()
        second["rlnOpticsGroup"], second["rlnOpticsGroupName"] = 2, "opticsGroup2"
        optics["rlnCtfDataAreCtfPremultiplied"], second["rlnCtfDataAreCtfPremultiplied"] = 0, 1
        optics = pd.concat([optics, second], ignore_index=True)
        rows.loc[rows.index % 2 == 1, "rlnOpticsGroup"] = 2
    return _write_star(tmp_path / "particles.star", optics, rows), rows, size


def test_plain_spa_rows_match_relion(empiar_10097, source):
    star, particles, _, size = empiar_10097
    indices = np.random.default_rng(4).choice(len(particles), 24, replace=False)
    dataset = source(star)

    rows = relion_ctf.relion_fftw_ctf_rows(dataset, indices, (size, size))
    centred = relion_ctf.relion_exact_ctf_half_from_source_star(dataset, indices, (size, size))

    relion = _relion_rows(star, particles, indices, size)
    assert_matches(rows, relion, rtol=RTOL)
    assert isinstance(centred, jax.Array) and centred.dtype == np.float64
    assert_matches(centred, _recovar_frame(relion), rtol=RTOL)
    # The float32 operands derived from the rows: the cast and CTF^2.
    assert_matches(rows.astype(np.float32), relion.astype(np.float32))
    assert_matches((rows * rows).astype(np.float32), (relion * relion).astype(np.float32))


def test_rows_are_the_strict_ieee_evaluation_to_the_last_unit(empiar_10097, source):
    """On the backend this test runs on (the CPU backend, or a GPU): every value within 1e-15 relative of the
    one-operation-at-a-time glibc evaluation, so no product and sum were contracted (see the program)."""

    star, particles, _, size = empiar_10097
    indices = np.random.default_rng(8).choice(len(particles), 24, replace=False)
    dataset = source(star)
    rows = relion_ctf.relion_fftw_ctf_rows(dataset, indices, (size, size))
    _, cache = relion_ctf.exact_ctf_source_cache(dataset, (size, size))
    params = relion_ctf._relion_ctf_batch_params(cache, indices)
    strict = reference.ctf_fftw_half(params[:, [0, 1, 2, 3, 4, 5, 6, 9, 8]], size, params[0, 7])
    # Per value, not per row: a contracted gamma shows as 1e-7 relative next to a CTF zero.
    assert np.max(np.abs(rows - strict) / np.abs(strict)) < 1e-15


@pytest.mark.parametrize(
    "even, mag", [(None, MAG), (EVEN, None), (EVEN, MAG)], ids=["magnification", "gamma_offset", "both"]
)
def test_magnified_and_aberrated_rows_match_relion(empiar_10097, source, tmp_path, even, mag):
    star, particles, size = _optics_variant(empiar_10097, tmp_path, even=even, mag=mag)
    indices = np.arange(len(particles))[::-1]
    dataset = source(star)

    rows = relion_ctf.relion_fftw_ctf_rows(dataset, indices, (size, size))

    assert_matches(rows, _relion_rows(star, particles, indices, size), rtol=RTOL)
    plain_star, plain_particles, _ = _optics_variant(empiar_10097, tmp_path)
    assert np.abs(rows - _relion_rows(plain_star, plain_particles, indices, size)).max() > 1e-3


def test_premultiplied_group_rows_are_relions_squared_ctf(empiar_10097, source, tmp_path):
    star, particles, size = _optics_variant(empiar_10097, tmp_path, premultiplied_second_group=True)
    indices = np.arange(len(particles))
    dataset = source(star)

    scoring = relion_ctf.relion_fftw_ctf_rows(dataset, indices, (size, size))
    centred = relion_ctf.relion_exact_ctf_half_from_source_star(dataset, indices, (size, size))
    startup = relion_ctf.relion_fftw_ctf_rows(dataset, indices, (size, size), square_premultiplied=False)

    relion = _relion_rows(star, particles, indices, size)
    assert_matches(scoring, relion, rtol=RTOL)
    assert_matches(centred, _recovar_frame(relion), rtol=RTOL)
    # RELION's start-up bootstrap keeps the plain CTF of a premultiplied group (ml_optimiser.cpp:3036-3051).
    plain_star = _write_star(tmp_path / "plain.star", read_star(str(star))[1].pipe(_plain).drop(columns="rlnCtfDataAreCtfPremultiplied"), particles)
    assert_matches(startup, _relion_rows(plain_star, particles, indices, size), rtol=RTOL)
    assert np.all(scoring[1::2] > 0) and np.any(scoring[0::2] < 0)


@pytest.mark.parametrize("bfactor_per_dose", [False, True], ids=["dose", "bfactor_per_dose"])
def test_tilt_rows_are_damped_as_relion_damps_them(tilt_table, source, tmp_path, bfactor_per_dose):
    particles, optics, size = tilt_table
    chosen = np.random.default_rng(6).choice(len(particles), 40, replace=False)
    rows_table = particles.iloc[chosen].reset_index(drop=True)
    rows_table = rows_table[
        ["rlnImageName", "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnCtfScalefactor", "rlnMicrographPreExposure", "rlnOpticsGroup"]
    ]
    assert rows_table["rlnMicrographPreExposure"].astype(float).max() > 50.0
    if bfactor_per_dose:
        rows_table["rlnCtfBfactorPerElectronDose"] = np.where(rows_table.index % 2 == 0, 4.0, 0.0)
    star = _write_star(tmp_path / "tilts.star", optics, rows_table)
    indices = np.arange(len(rows_table))
    dataset = source(star)

    rows = relion_ctf.relion_fftw_ctf_rows(dataset, indices, (size, size))

    assert_matches(rows, _relion_rows(star, rows_table, indices, size, tomo=True), rtol=RTOL)
    undamped = _relion_rows(star, rows_table.assign(rlnMicrographPreExposure=0.0), indices, size, tomo=True)
    assert np.abs(rows - undamped).max() > 1e-2


def test_tomo_damping_agrees_on_numpy_and_jax(tilt_table):
    """One expression on two array backends: the same values to 1e-15 on a real dose table."""

    import jax.numpy as jnp

    particles, optics, size = tilt_table
    dose = np.unique(np.asarray(particles["rlnMicrographPreExposure"], dtype=np.float64))
    freq_sq = tomo_input.fftw_half_freq_sq(size, size, float(optics["rlnImagePixelSize"][0])).reshape(-1)
    for per_dose in (np.zeros_like(dose), np.where(np.arange(dose.size) % 2 == 0, 4.0, 0.0)):
        host = tomo_input.relion_tomo_damping(freq_sq, dose, per_dose)
        device = tomo_input.relion_tomo_damping(jnp.asarray(freq_sq), jnp.asarray(dose), jnp.asarray(per_dose))
        assert isinstance(host, np.ndarray) and isinstance(device, jax.Array)
        assert host.dtype == device.dtype == np.float64 and host.shape == device.shape == (dose.size, freq_sq.size)
        assert_matches(device, host, rtol=1e-15)


def test_rows_keep_order_duplicates_and_requested_pixels(empiar_10097, source, tmp_path):
    star, particles, size = _optics_variant(empiar_10097, tmp_path, premultiplied_second_group=True)
    dataset = source(star)
    width = size // 2 + 1
    indices = np.asarray([3, 0, 3, 10, 1, 7], dtype=np.int64)
    # Unsorted, with a repeat: column order and duplication are kept.
    pixels = np.asarray([7, 0, 7, size * width - 1, 40, 3 * width], dtype=np.int64)

    full = relion_ctf.relion_exact_ctf_half_from_source_star_host(dataset, indices, (size, size))
    some = relion_ctf.relion_exact_ctf_half_from_source_star_host(dataset, indices, (size, size), pixel_indices=pixels)

    relion = _recovar_frame(_relion_rows(star, particles, indices, size))
    assert full.shape == (indices.size, size * width) and some.shape == (indices.size, pixels.size)
    assert full.flags.writeable and some.flags.writeable
    assert_matches(full, relion, rtol=RTOL)
    assert_matches(some, relion[:, pixels], rtol=RTOL)
    with pytest.raises(ValueError, match="outside"):
        relion_ctf.relion_exact_ctf_half_from_source_star(
            dataset, indices, (size, size), pixel_indices=np.asarray([size * width])
        )


def test_a_request_is_evaluated_in_blocks_without_changing_its_rows(empiar_10097, source, monkeypatch):
    star, particles, _, size = empiar_10097
    indices = np.random.default_rng(2).choice(len(particles), 37, replace=False)
    dataset = source(star)
    pixels = np.arange(0, size * (size // 2 + 1), 97)
    whole = relion_ctf.relion_exact_ctf_half_from_source_star_host(dataset, indices, (size, size), pixel_indices=pixels)
    # Four rows per program call: full blocks and a last short block padded to a power of two.
    monkeypatch.setattr(relion_ctf, "_CTF_BLOCK_ELEMENTS", 4 * pixels.size)
    blocked = relion_ctf.relion_exact_ctf_half_from_source_star_host(dataset, indices, (size, size), pixel_indices=pixels)
    assert_matches(blocked, whole)
    assert_matches(whole, _recovar_frame(_relion_rows(star, particles, indices, size))[:, pixels], rtol=RTOL)


def test_block_rows_are_powers_of_two():
    """A bounded set of compiled programs per pixel count: every program call has a power of two of rows."""

    calls = []
    program = relion_ctf._relion_ctf_program

    def record(constants, *args, **kwargs):
        calls.append(int(constants.shape[0]))
        return program(constants, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relion_ctf, "_relion_ctf_program", record)
        for n in (1, 3, 37, 256, 300, 1000):
            rows = relion_ctf._relion_ctf_rows(np.ones((n, 9)), 8, 12.0)
            assert rows.shape == (n, 8 * 5)
    assert set(calls) <= {1 << k for k in range(9)}


def test_nothing_of_a_request_is_kept(empiar_10097, source):
    """RELION evaluates a particle's CTF in every expectation; relax keeps no row on the host or the device."""

    star, particles, _, size = empiar_10097
    dataset = source(star)
    indices = np.arange(16)
    first = relion_ctf.relion_exact_ctf_half_from_source_star_host(dataset, indices, (size, size))
    second = relion_ctf.relion_exact_ctf_half_from_source_star_host(dataset, indices, (size, size))
    assert first is not second and not np.shares_memory(first, second)
    assert_matches(first, second)

    def arrays(value):
        if isinstance(value, dict):
            return [array for item in value.values() for array in arrays(item)]
        return [value] if isinstance(value, (np.ndarray, jax.Array)) else []

    # What the parsed tables keep per particle is scalars, nine CTF constants at most: never a value per pixel.
    (cache,) = relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE.values()
    kept = arrays(cache)
    assert kept and max(array.size for array in kept) <= 9 * len(particles)
    assert all(size * (size // 2 + 1) not in array.shape for array in kept)


def test_absent_ctf_columns_take_relions_defaults(empiar_10097, source, tmp_path):
    """A STAR without rlnPhaseShift/rlnCtfBfactor/rlnCtfScalefactor gets RELION's 0/0/1, and an optics-level
    value wins over the default: CTF::readValue (ctf.cpp:71-91) reads the particle, its group, the default."""

    _, particles, optics, size = empiar_10097
    rows = particles.iloc[:4][["rlnImageName", "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnOpticsGroup"]]
    star = _write_star(tmp_path / "particles.star", optics.assign(rlnCtfScalefactor=0.75), rows)
    ctf = relion_ctf.relion_fftw_ctf_rows(source(star), np.arange(4), (size, size))
    assert_matches(ctf, _relion_rows(star, rows.assign(rlnCtfScalefactor=0.75), np.arange(4), size), rtol=RTOL)
