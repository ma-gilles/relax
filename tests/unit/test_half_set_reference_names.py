"""A split-halves refinement reads RELION's per-half-set reference (relax#93).

RELION's MPI refinement passes ``--ref`` through ``FileName::getHalf`` for each half-set rank
(ml_optimiser_mpi.cpp:726-731, filename.cpp:474-489): a reference named ``half1`` gives half set 2 the
``half2`` file and the other way round; only the base name is changed.
"""

import logging

import numpy as np
import pytest

from relax.refinement import startup_references
from relax.refinement.startup_references import relion_half_reference_path

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("path", "half1", "half2"),
    [
        ("Reconstruct/job072/half1.mrc", "Reconstruct/job072/half1.mrc", "Reconstruct/job072/half2.mrc"),
        ("Reconstruct/job072/half2.mrc", "Reconstruct/job072/half1.mrc", "Reconstruct/job072/half2.mrc"),
        ("Refine3D/job074/run_half1_class001_unfil.mrc", "Refine3D/job074/run_half1_class001_unfil.mrc",
         "Refine3D/job074/run_half2_class001_unfil.mrc"),
        # Every occurrence in the base name; the directory is never changed.
        ("/data/half1_maps/a_half1_b_half1.mrc", "/data/half1_maps/a_half1_b_half1.mrc",
         "/data/half1_maps/a_half2_b_half2.mrc"),
        ("/data/half1_maps/merged.mrc", "/data/half1_maps/merged.mrc", "/data/half1_maps/merged.mrc"),
        ("half1.mrc", "half1.mrc", "half2.mrc"),
        # A name with both: half set 2 replaces half1, half set 1 replaces half2.
        ("half1_vs_half2.mrc", "half1_vs_half1.mrc", "half2_vs_half2.mrc"),
    ],
)  # fmt: skip
def test_relion_half_reference_path_follows_getHalf(path, half1, half2):
    assert relion_half_reference_path(path, 1) == half1
    assert relion_half_reference_path(path, 2) == half2


def _write_map(path, volume, pixel_size=2.0):
    import mrcfile

    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.asarray(volume, dtype=np.float32))
        handle.header.cella = (np.float32(pixel_size * volume.shape[0]),) * 3
    return str(path)


def _load(path, **kwargs):
    return startup_references.load_k1_reference(
        path, volume_shape=(8, 8, 8), ini_high=None, real_for_projector=True, real_dtype=np.float32,
        complex_dtype=np.complex64, log=logging.getLogger(__name__), **kwargs,
    )  # fmt: skip


def test_each_half_set_starts_from_its_own_named_reference(tmp_path):
    rng = np.random.default_rng(3)
    maps = [rng.standard_normal((8, 8, 8)) for _ in range(2)]
    first = _write_map(tmp_path / "ref_half1.mrc", maps[0])
    _write_map(tmp_path / "ref_half2.mrc", maps[1])
    refs = _load(first)
    alone = [_load(_write_map(tmp_path / f"alone{k}.mrc", maps[k])) for k in range(2)]
    np.testing.assert_array_equal(refs.fourier.half1, alone[0].fourier.half1)
    np.testing.assert_array_equal(refs.fourier.half2, alone[1].fourier.half1)
    np.testing.assert_array_equal(refs.real_for_projector.half2, alone[1].real_for_projector.half1)
    # The start-up tau2 and data-vs-prior are half 1's.
    np.testing.assert_array_equal(refs.prior_source, alone[0].prior_source)


def test_a_reference_without_a_half_name_is_shared(tmp_path):
    refs = _load(_write_map(tmp_path / "merged.mrc", np.ones((8, 8, 8))))
    assert refs.fourier.half2 is refs.fourier.half1


def test_a_continued_run_reads_the_named_file_for_both_halves(tmp_path):
    refs = _load(_write_map(tmp_path / "ref_half1.mrc", np.ones((8, 8, 8))), split_halves=False)
    assert refs.fourier.half2 is refs.fourier.half1


def test_a_missing_half_set_reference_stops_the_run(tmp_path):
    with pytest.raises(SystemExit, match="ref_half2.mrc does not exist"):
        _load(_write_map(tmp_path / "ref_half1.mrc", np.ones((8, 8, 8))))


def test_half_set_references_with_different_pixel_sizes_are_refused(tmp_path):
    first = _write_map(tmp_path / "ref_half1.mrc", np.ones((8, 8, 8)), 2.0)
    _write_map(tmp_path / "ref_half2.mrc", np.ones((8, 8, 8)), 2.1)
    with pytest.raises(SystemExit, match="different header pixel sizes"):
        _load(first)


def _start(reference_real, counts=(160, 140)):
    return startup_references.relion_start_half_sets(
        reference_real, np.full(5, 8.0**4), counts, box_size=8, volume_shape=(8, 8, 8), tau2_fudge=1.0,
        shell_pair_counting="relion",
    )  # fmt: skip


def test_each_half_set_starts_from_its_own_tau2_and_data_vs_prior(tmp_path):
    # Each RELION rank runs initialiseDataVersusPrior on its own Iref and particle count (ml_optimiser_mpi.cpp:771).
    from relax.steps.half_inputs import HalfPair

    rng = np.random.default_rng(4)
    first = _write_map(tmp_path / "ref_half1.mrc", rng.standard_normal((8, 8, 8)))
    _write_map(tmp_path / "ref_half2.mrc", 3.0 * rng.standard_normal((8, 8, 8)))
    refs = _load(first)
    mean_variance, (curve_1, curve_2) = _start(refs.reference_real)
    assert mean_variance.shape[0] == 2
    for k, (count, reference) in enumerate(zip((160, 140), refs.reference_real)):
        own_tau2, (own_curve, _) = _start(HalfPair.shared(reference), counts=(count, count))
        np.testing.assert_array_equal(np.asarray(mean_variance[k]), np.asarray(own_tau2))
        np.testing.assert_array_equal((curve_1, curve_2)[k], own_curve)
    assert not np.allclose(np.asarray(mean_variance[0]), np.asarray(mean_variance[1]))


def test_a_shared_reference_has_one_tau2_and_each_half_sets_own_particle_count(tmp_path):
    refs = _load(_write_map(tmp_path / "merged.mrc", np.random.default_rng(6).standard_normal((8, 8, 8))))
    mean_variance, (curve_1, curve_2) = _start(refs.reference_real)
    assert np.ndim(mean_variance) == 1
    # RELION's evidence is nr_particles * pdf_class / sigma2: half set 2's curve is half set 1's times 140 / 160.
    np.testing.assert_allclose(curve_2[1:], curve_1[1:] * 140 / 160, rtol=1e-12)
