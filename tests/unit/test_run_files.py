"""RELION run_itNNN files: a snapshot written and read back is the same snapshot."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.core import fourier_transform_utils as ftu

from relax.helpers.convergence import RefinementState
from relax.reconstruction import regularization
from relax.refinement import iteration_snapshot as iteration_snapshot_module
from relax.refinement import setup_checks
from relax.refinement.half_inputs import initialize_halfsets
from relax.refinement.iteration_snapshot import (
    REFINEMENT_STATE_SCALAR_FIELDS,
    IterationSnapshot,
    SnapshotCapture,
    radial_shell_volume,
    refinement_state_fields,
    tau2_mean_variance,
)
from relax.refinement.run_files import (
    RunFileWriter,
    RunSettings,
    read_run_files,
    read_star_blocks,
)
from relax.sampling.orientation_priors import DirectionPrior

pytestmark = pytest.mark.unit

BOX = 16
N_SHELLS = BOX // 2 + 1


def test_snapshot_capture_has_a_staged_run_lifecycle():
    assert tuple(field.name for field in dataclasses.fields(SnapshotCapture)) == (
        "n_classes", "box_size", "voxel_size", "tau2_fudge", "consistency",
    )
    assert tuple(inspect.signature(SnapshotCapture.begin).parameters) == (
        "self", "relion_iteration", "state", "sigma_offset_angstrom_per_half",
        "current_size", "incr_size", "has_high_fsc_at_limit", "random_perturbation",
        "acc_rot_per_class", "acc_trans_per_class_angstrom",
    )
    assert not hasattr(SnapshotCapture, "capture_maps_and_spectra")
    assert not hasattr(SnapshotCapture, "capture_priors")
    assert not hasattr(SnapshotCapture, "capture_particles")
    for name in (
        "SnapshotRunSpec", "SnapshotReferenceState", "SnapshotSamplingState",
        "SnapshotParticleState", "capture_iteration_snapshot",
    ):
        assert not hasattr(iteration_snapshot_module, name)


@pytest.mark.parametrize("n_classes", [1, 2])
def test_controller_publishes_each_checkpoint_from_one_run_capture(monkeypatch, n_classes):
    """One SnapshotCapture per run; each due iteration begins, finishes and publishes its snapshot."""
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.refinement.refinement_options import CheckpointOptions

    class Writer:
        def due(self, relion_iteration):
            return True

        def wants_unfiltered_maps(self, relion_iteration, *, n_classes):
            return False

        def __call__(self, snapshot):
            published.append(snapshot)

    published = []
    trace = CallTrace(monkeypatch)
    trace.wrap(setup_checks, "SnapshotCapture", "capture")
    trace.wrap(SnapshotCapture, "begin")
    trace.wrap(SnapshotCapture, "finish")
    run_tiny_refinement(
        monkeypatch, n_classes=n_classes, final_after_max_iter=False, checkpoint=CheckpointOptions(writer=Writer()),
    )
    assert trace.labels("capture", "begin", "finish") == ["capture"] + 2 * ["begin", "finish"]
    (capture,) = trace.calls("capture")
    assert all(call.args[0] is capture.result for call in trace.calls_seen if call.label != "capture")
    assert published == [call.result for call in trace.calls("finish")]
    assert [call.args[1] for call in trace.calls("finish")] == [call.result for call in trace.calls("begin")]


def test_staged_snapshot_capture_copies_complete_k1_state():
    capture = SnapshotCapture(
        n_classes=1,
        box_size=BOX,
        voxel_size=2.5,
        tau2_fudge=1.25,
        consistency={},
    )
    state = RefinementState(
        iteration=2,
        healpix_order=3,
        adaptive_oversampling=1,
        translation_range=4.0,
        translation_step=1.0,
    )
    snapshot = capture.begin(
        3,
        state,
        sigma_offset_angstrom_per_half=(2.0, 3.0),
        current_size=12,
        incr_size=10,
        has_high_fsc_at_limit=True,
        random_perturbation=0.125,
        acc_rot_per_class=np.array([2.25]),
        acc_trans_per_class_angstrom=np.array([1.125]),
    )
    means = [np.ones(8, dtype=np.complex64), np.full(8, 2, dtype=np.complex64)]
    unfiltered = [np.full(8, 3, dtype=np.complex64), np.full(8, 4, dtype=np.complex64)]
    tau2 = [np.arange(5, dtype=np.float32), np.arange(5, dtype=np.float32) + 1]
    noise = [np.arange(5, dtype=np.float64) + 2, np.arange(5, dtype=np.float64) + 3]
    direction_prior = [np.full(12, 0.25), np.full(12, 0.5)]
    half_inputs = initialize_halfsets(
        (None, None),
        previous_best_translations=[np.zeros((2, 2), dtype=np.float32)] * 2,
        previous_best_rotation_eulers=[np.zeros((2, 3), dtype=np.float64)] * 2,
        image_corrections=[np.ones(2, dtype=np.float32)] * 2,
        scale_corrections=[np.ones(2, dtype=np.float32)] * 2,
        group_ids=[None, np.ones(2, dtype=np.int64)],
        group_count=None,
        optics_group_ids=(None, None),
    )
    result = capture.finish_k1(
        snapshot, means, unfiltered, tau2, np.arange(5, dtype=np.float32), noise,
        fsc=np.linspace(1.0, 0.0, 5),
        fsc_for_growth=np.linspace(0.9, 0.1, 5),
        direction_priors=[DirectionPrior(values, 1) for values in direction_prior],
        half_inputs=half_inputs,
        max_posterior=[np.full(2, 0.7), np.full(2, 0.8)],
        significant_counts=[np.ones(2, dtype=np.int32), np.full(2, 2, dtype=np.int32)],
        avg_norm_correction=(0.95, None),
    )

    assert result.relion_iteration == 3 and result.n_classes == 1
    assert_matches(result.sigma_offset_angstrom, (2.0, 3.0))
    # A half without norm correction records RELION's 1.0 in relax's frame.
    assert_matches(result.avg_norm_correction, (0.95, float(BOX) ** 2))
    assert_matches(result.acc_rot_per_class, [2.25])
    assert_matches(result.acc_trans_per_class_angstrom, [1.125])
    assert result.extra["direction_prior_order_half1"] == 1
    assert result.extra["direction_prior_order_half2"] == 1
    assert result.group_ids[0].shape == (2,) and np.all(result.group_ids[0] == 0)
    assert result.class_assignments is None and result.class_weights is None
    assert not np.shares_memory(result.means[0], means[0])
    assert not np.shares_memory(result.direction_prior[0], direction_prior[0])

_PARTICLES = """
# version 30001

data_optics

loop_
_rlnOpticsGroup #1
_rlnOpticsGroupName #2
_rlnImagePixelSize #3
_rlnImageSize #4
_rlnVoltage #5
1 opticsGroup1 2.5 16 300.0

# version 30001

data_particles

loop_
_rlnImageName #1
_rlnMicrographName #2
_rlnDefocusU #3
_rlnDefocusV #4
_rlnOpticsGroup #5
_rlnAngleRot #6
"""


def _write_input_star(directory: Path, n_particles: int) -> Path:
    rows = [
        f"{i + 1:06d}@Particles/stack.mrcs mic_{i % 2}.mrc 12345.678901 12001.5 1 0.000000" for i in range(n_particles)
    ]
    path = directory / "particles.star"
    path.write_text(_PARTICLES + "\n".join(rows) + "\n")
    return path


def _fourier(real):
    return np.asarray(ftu.get_dft3(jnp.asarray(real))).astype(np.complex64).reshape(-1)


def _state_fields():
    state = RefinementState(
        iteration=4,
        healpix_order=4,
        adaptive_oversampling=1,
        translation_range=3.25,
        translation_step=0.8125,
        current_resolution=7.123456789,
        voxel_size_angstrom=2.5,
    )
    state = dataclasses.replace(
        state,
        previous_resolution=7.5,
        nr_iter_wo_resol_gain=1,
        has_fine_enough_angular_sampling=True,
        ave_Pmax=0.4123456789,
        acc_rot=2.25,
        acc_trans=float("inf"),
        current_changes_optimal_orientations=1.0 / 3.0,
        smallest_changes_optimal_orientations=0.1 + 0.2,
        suppress_hidden_variable_increment_once=True,
    )
    return refinement_state_fields(state)


def _k1_snapshot(half_sizes, rng):
    reals = [rng.standard_normal((BOX, BOX, BOX)).astype(np.float32) for _ in range(4)]
    n_groups = 2
    group_scale = np.array([0.93, 1.07], dtype=np.float32)
    group_ids = [rng.integers(0, n_groups, size=n) for n in half_sizes]
    scale = [group_scale[g] for g in group_ids]
    return IterationSnapshot(
        relion_iteration=5,
        n_classes=1,
        box_size=BOX,
        pixel_size=2.5,
        tau2_fudge=1.0,
        means=[_fourier(reals[0]), _fourier(reals[1])],
        unfiltered_means=[_fourier(reals[2]), _fourier(reals[3])],
        tau2_shells=rng.random((2, N_SHELLS)) * 1e3,
        data_vs_prior=rng.random(N_SHELLS) * 50,
        fsc=np.linspace(1.0, 0.01, N_SHELLS),
        fsc_for_growth=np.linspace(0.999, 0.02, N_SHELLS),
        noise_shells=[rng.random(N_SHELLS) * 1e-3 + 1e-6, rng.random(N_SHELLS) * 1e-3 + 1e-6],
        sigma_offset_angstrom=(3.141592653589793, 2.718281828459045),
        current_size=12,
        incr_size=11,
        has_high_fsc_at_limit=True,
        random_perturbation=-0.123456789012345,
        state_fields=_state_fields(),
        rotation_eulers=[rng.uniform(-180, 180, (n, 3)) for n in half_sizes],
        translations=[rng.uniform(-3, 3, (n, 2)).astype(np.float32) for n in half_sizes],
        image_corrections=[(s * rng.uniform(0.8, 1.2, s.size)).astype(np.float32) for s in scale],
        scale_corrections=scale,
        group_ids=group_ids,
        class_weights=None,
        direction_prior=[rng.random(48), rng.random(48)],
        class_assignments=None,
        max_posterior=[rng.random(n) for n in half_sizes],
        significant_counts=[rng.integers(1, 500, n) for n in half_sizes],
        avg_norm_correction=(0.9876543210987654 * BOX**2, 1.0123456789 * BOX**2),
        acc_rot_per_class=np.array([2.25]),
        acc_trans_per_class_angstrom=np.array([1.125]),
        extra={"euler_dtype": "float64", "translation_dtype": "float32", "correction_dtype": "float32"},
    )


def _run_settings(output_root, **fields):
    """RunSettings of a 25-iteration K=1 run with relion_refine's defaults where the test sets nothing."""
    return RunSettings(
        **{
            "output_root": str(output_root),
            "random_seed": 17,
            "nr_iter": 25,
            "particle_diameter": 200.0,
            "width_mask_edge": 5,
            "adaptive_oversampling": 1,
            "adaptive_fraction": 0.999,
            "low_resol_join_halves": 40.0,
            "auto_local_healpix_order": 4,
            "strict_highres_exp": -1.0,
            "do_solvent_fsc": False,
            "solvent_mask_name": "None",
            "max_significants": -1,
            "symmetry": "C1",
            "healpix_order_original": 2,
            "offset_range_original_angstrom": 0.0,
            "offset_step_original_angstrom": 0.0,
            "perturbation_factor": 0.5,
            "padding_factor": 2.0,
            "command_line": "",
            "mode": "relion",
            **fields,
        }
    )


def _writer(tmp_path, input_star, half_rows, **kwargs):
    kwargs.setdefault("background", False)
    return RunFileWriter(
        tmp_path / "out",
        settings=_run_settings(tmp_path / "out" / "run"),
        input_star=input_star,
        half_rows=half_rows,
        **kwargs,
    )


def _assert_snapshots_match(read, written):
    for name in (
        "relion_iteration",
        "n_classes",
        "box_size",
        "current_size",
        "incr_size",
        "has_high_fsc_at_limit",
    ):
        assert getattr(read, name) == getattr(written, name), name
    for name in (
        "pixel_size",
        "tau2_fudge",
        "tau2_shells",
        "data_vs_prior",
        "fsc",
        "fsc_for_growth",
        "sigma_offset_angstrom",
        "random_perturbation",
        "class_weights",
        "avg_norm_correction",
        "acc_rot_per_class",
        "acc_trans_per_class_angstrom",
    ):
        expected = getattr(written, name)
        if expected is None:
            assert getattr(read, name) is None, name
        else:
            assert_matches(np.asarray(getattr(read, name)), np.asarray(expected), err_msg=name)
    assert set(read.state_fields) == set(REFINEMENT_STATE_SCALAR_FIELDS)
    for name, value in written.state_fields.items():
        assert type(read.state_fields[name]) is type(value), name
        assert_matches(np.asarray(read.state_fields[name]), np.asarray(value), err_msg=name)
    for h in range(2):
        assert_matches(read.noise_shells[h], written.noise_shells[h], err_msg="noise")
        assert_matches(read.rotation_eulers[h], written.rotation_eulers[h], err_msg="eulers")
        for name in ("translations", "image_corrections", "scale_corrections"):
            a, b = getattr(read, name)[h], getattr(written, name)[h]
            assert a.dtype == b.dtype, name
            assert_matches(a, b, err_msg=name)
        assert np.array_equal(read.group_ids[h], written.group_ids[h])
        if written.max_posterior is not None:
            assert_matches(read.max_posterior[h], written.max_posterior[h])
            assert np.array_equal(read.significant_counts[h], written.significant_counts[h])
        # Maps pass through RELION's float32 real-space MRC.
        assert_matches(read.means[h], written.means[h], rtol=1e-6)
    if written.direction_prior is None:
        assert read.direction_prior is None
    else:
        for h in range(2):
            assert_matches(read.direction_prior[h], written.direction_prior[h])


def test_auto_refine_run_files_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    input_star = _write_input_star(tmp_path, 7)
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    snapshot = _k1_snapshot([4, 3], rng)
    writer = _writer(tmp_path, input_star, half_rows)
    optimiser = writer(snapshot)
    assert optimiser.name == "run_it005_optimiser.star"
    for suffix in (
        "half1_model.star",
        "half2_model.star",
        "data.star",
        "sampling.star",
        "half1_class001.mrc",
        "half2_class001.mrc",
        "half1_class001_unfil.mrc",
        "half2_class001_unfil.mrc",
    ):
        assert (tmp_path / "out" / f"run_it005_{suffix}").exists(), suffix

    names = read_star_blocks(input_star)["particles"]["rlnImageName"]
    read = read_run_files(optimiser, image_names=names, half_rows=half_rows)
    _assert_snapshots_match(read, snapshot)
    for h in range(2):
        assert_matches(read.unfiltered_means[h], snapshot.unfiltered_means[h], rtol=1e-6)


def test_optimiser_star_records_mode_relax_and_relions_mode_writes_nothing(tmp_path):
    rng = np.random.default_rng(0)
    input_star = _write_input_star(tmp_path, 7)
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    snapshot = _k1_snapshot([4, 3], rng)
    names = read_star_blocks(input_star)["particles"]["rlnImageName"]
    for mode, recorded in (("relion", None), ("relax", "relax")):
        writer = RunFileWriter(
            tmp_path / mode,
            settings=_run_settings(tmp_path / mode / "run", mode=mode),
            input_star=input_star,
            half_rows=half_rows,
            background=False,
        )
        optimiser = writer(snapshot)
        assert read_star_blocks(optimiser)["relax_state"].get("relax_mode") == recorded
        read = read_run_files(optimiser, image_names=names, half_rows=half_rows)
        _assert_snapshots_match(read, snapshot)
        assert "mode" not in read.extra


def test_subtomogram_run_files_round_trip_3d_offsets(tmp_path):
    """Subtomogram particles' 3D offsets go to rlnOriginXAngst/Y/Z and come back (--continue)."""

    import dataclasses

    rng = np.random.default_rng(4)
    input_star = _write_input_star(tmp_path, 7)
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    snapshot = _k1_snapshot([4, 3], rng)
    snapshot = dataclasses.replace(
        snapshot, translations=[rng.uniform(-3, 3, (n, 3)).astype(np.float32) for n in (4, 3)]
    )
    optimiser = _writer(tmp_path, input_star, half_rows)(snapshot)
    data = read_star_blocks(tmp_path / "out" / "run_it005_data.star")["particles"]
    assert "rlnOriginZAngst" in data
    names = read_star_blocks(input_star)["particles"]["rlnImageName"]
    read = read_run_files(optimiser, image_names=names, half_rows=half_rows)
    for h in range(2):
        assert read.translations[h].shape == (half_rows[h].size, 3)
        assert_matches(read.translations[h], snapshot.translations[h])


def test_unnormalized_run_files_carry_relion_norm_and_spectra_past_current_size(tmp_path):
    """A run without norm correction (subtomogram) writes RELION's rlnNormCorrection(Average) = 1.0; auto-refine
    writes tau2 = 0 and the FSC-floor SSNR past the current size, as RELION's reader expects, and reads back the
    loop's floors."""

    import dataclasses

    from recovar import jax_config

    rng = np.random.default_rng(6)
    input_star = _write_input_star(tmp_path, 7)
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    snapshot = _k1_snapshot([4, 3], rng)
    past = np.arange(N_SHELLS) > snapshot.current_size // 2
    tau2 = np.where(past, jax_config.EPSILON, snapshot.tau2_shells)
    data_vs_prior = np.where(past, 0.0, snapshot.data_vs_prior)
    snapshot = dataclasses.replace(
        snapshot,
        avg_norm_correction=(float(BOX) ** 2, float(BOX) ** 2),
        image_corrections=[np.asarray(s, dtype=np.float32) for s in snapshot.scale_corrections],
        tau2_shells=tau2,
        data_vs_prior=data_vs_prior,
    )
    optimiser = _writer(tmp_path, input_star, half_rows)(snapshot)

    out = tmp_path / "out"
    data = read_star_blocks(out / "run_it005_data.star")["particles"]
    np.testing.assert_allclose(np.asarray(data["rlnNormCorrection"], dtype=np.float64), 1.0, rtol=1e-6)
    for h in (1, 2):
        model = read_star_blocks(out / f"run_it005_half{h}_model.star")
        assert float(model["model_general"]["rlnNormCorrectionAverage"]) == pytest.approx(1.0, rel=1e-6)
        table = model["model_class_1"]
        written_tau2 = np.asarray(table["rlnReferenceTau2"], dtype=np.float64)
        written_ssnr = np.asarray(table["rlnSsnrMap"], dtype=np.float64)
        # Past the current size the written spectrum is zero-filled, not computed: a structural check.
        assert np.all(written_tau2[past] == 0.0) and np.all(written_tau2[~past] > 0.0)
        np.testing.assert_allclose(written_ssnr[past], 0.001 / 0.999, rtol=1e-12)

    names = read_star_blocks(input_star)["particles"]["rlnImageName"]
    read = read_run_files(optimiser, image_names=names, half_rows=half_rows)
    _assert_snapshots_match(read, snapshot)


def _unestimated_norm_snapshot(snapshot, input_norms, rng):
    """``snapshot`` after a pass whose norm correction is off, for particles that entered with ``input_norms``.

    The corrections go through the production path: the start-up corrections of the input column, the
    norm/scale update with norm correction off, and the snapshot's frame conversion of its average.
    """

    from types import SimpleNamespace

    from relax.relion.input_poses import _initial_corrections_from_norm
    from relax.relion.normalization import update_relion_norm_scale_corrections
    from relax.types import NoiseStats

    image, scale = _initial_corrections_from_norm(input_norms)
    stats = [
        NoiseStats(
            wsum_sigma2_noise=np.zeros(1, np.float32),
            wsum_img_power=np.zeros(1, np.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(norms.size),
            wsum_scale_correction_xa=np.asarray([0.93, 1.07]) * 5.0,
            wsum_scale_correction_aa=np.full(2, 5.0),
        )
        for norms in input_norms
    ]
    update = update_relion_norm_scale_corrections(
        noise_stats_per_half=stats,
        image_corrections_per_half=image,
        scale_corrections_per_half=scale,
        group_ids_per_half=snapshot.group_ids,
        group_count_per_half=[2, 2],
        do_norm_correction=False,
    )
    assert update.avg_norm_correction_per_half == [None, None]
    halves = [
        SimpleNamespace(
            rotation_eulers=snapshot.rotation_eulers[h],
            translations=snapshot.translations[h],
            image_corrections=np.asarray(update.image_corrections_per_half[h]),
            scale_corrections=np.asarray(update.scale_corrections_per_half[h]),
            group_ids=snapshot.group_ids[h],
        )
        for h in range(2)
    ]
    state = iteration_snapshot_module._host_particle_state(
        halves, snapshot.max_posterior, snapshot.significant_counts, update.avg_norm_correction_per_half,
        direction_priors=[], box_size=BOX, consistency={},
    )
    return dataclasses.replace(
        snapshot,
        image_corrections=state["image_corrections"],
        scale_corrections=state["scale_corrections"],
        avg_norm_correction=state["avg_norm_correction"],
    )


@pytest.mark.parametrize("input_norm", [1.0, 0.25, 1.0 / BOX**2])
def test_unestimated_norm_column_is_written_back_and_does_not_compound(tmp_path, input_norm):
    """A run without norm correction (subtomograms; relion_refine switches it off, ml_optimiser.cpp:2448-2453)
    writes each particle's input rlnNormCorrection back and rlnNormCorrectionAverage 1, whatever the input
    value; a second pass started from the written column writes the same column, and the files read back
    (--continue) and written again are unchanged."""

    rng = np.random.default_rng(64)
    input_star = _write_input_star(tmp_path, 7)
    names = read_star_blocks(input_star)["particles"]["rlnImageName"]
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    base = _k1_snapshot([4, 3], rng)
    column = np.full(7, input_norm) * np.where(np.arange(7) % 2, 1.0, 0.5 if input_norm != 1.0 else 1.0)

    written_columns = []
    for chained_pass in range(2):
        out = tmp_path / f"pass{chained_pass}"
        snapshot = _unestimated_norm_snapshot(base, [column[rows] for rows in half_rows], rng)
        writer = RunFileWriter(
            out, settings=_run_settings(out / "run"), input_star=input_star, half_rows=half_rows, background=False
        )
        optimiser = writer(snapshot)
        data = read_star_blocks(out / "run_it005_data.star")["particles"]
        written = np.asarray(data["rlnNormCorrection"], dtype=np.float64)
        np.testing.assert_allclose(written, column, rtol=1e-6)
        for h in (1, 2):
            model = read_star_blocks(out / f"run_it005_half{h}_model.star")
            assert float(model["model_general"]["rlnNormCorrectionAverage"]) == pytest.approx(1.0, rel=1e-12)
        written_columns.append(written)

        # --continue: read the files and write them again.
        read = read_run_files(optimiser, image_names=names, half_rows=half_rows)
        again = tmp_path / f"pass{chained_pass}_again"
        RunFileWriter(
            again, settings=_run_settings(again / "run"), input_star=input_star, half_rows=half_rows, background=False
        )(read)
        rewritten = read_star_blocks(again / "run_it005_data.star")["particles"]["rlnNormCorrection"]
        np.testing.assert_allclose(np.asarray(rewritten, dtype=np.float64), written, rtol=1e-6)
        column = written  # the next pass starts from this pass's data.star

    np.testing.assert_allclose(written_columns[1], written_columns[0], rtol=1e-6)


def test_data_star_writes_the_input_general_block_back(tmp_path):
    """A subtomogram STAR's data_general (rlnTomoSubTomosAre2DStacks) is written back, as relion_refine does:
    without it the output reads as single particles and cannot start the next job. A single-particle STAR has
    no such block and gets none."""

    rng = np.random.default_rng(65)
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    plain = _write_input_star(tmp_path, 7)
    _writer(tmp_path, plain, half_rows)(_k1_snapshot([4, 3], rng))
    assert "general" not in read_star_blocks(tmp_path / "out" / "run_it005_data.star")

    for name, block in (
        ("loop", "\n# version 50001\n\ndata_general\n\nloop_\n_rlnTomoSubTomosAre2DStacks #1\n1\n"),
        ("list", "\n# version 50001\n\ndata_general\n\n_rlnTomoSubTomosAre2DStacks 1\n"),
    ):
        directory = tmp_path / name
        directory.mkdir()
        tomo = directory / "particles.star"
        tomo.write_text(block + plain.read_text())
        _writer(directory, tomo, half_rows)(_k1_snapshot([4, 3], rng))
        blocks = read_star_blocks(directory / "out" / "run_it005_data.star")
        assert list(blocks)[0] == "general"
        assert blocks["general"] == {"rlnTomoSubTomosAre2DStacks": ["1"]}  # a loop, as relion_refine writes it
        assert blocks["particles"]["rlnImageName"] == read_star_blocks(plain)["particles"]["rlnImageName"]


@pytest.mark.parametrize("looped", [True, False])
def test_initial_model_data_star_starts_a_subtomogram_refinement(tmp_path, looped):
    """The chain InitialModel -> Class3D/Refine3D: InitialModel's data.star of a subtomogram input keeps
    data_general (rlnTomoSubTomosAre2DStacks), so the file reads as 2D stacks, and the refinement writer started
    from it, untouched, writes the block again. A single-particle input gets no such block from either writer."""

    from recovar.data_io.starfile import read_star

    from relax.refinement import run_files
    from relax.refinement.tomo_half import is_relion5_2d_stack_star
    from relax.relion import initial_model_io
    from relax.vdam.state import NativeParticleState

    rng = np.random.default_rng(66)
    plain = _write_input_star(tmp_path, 7)
    block = (
        "\n# version 50001\n\ndata_general\n\nloop_\n_rlnTomoSubTomosAre2DStacks #1\n1\n"
        if looped
        else "\n# version 50001\n\ndata_general\n\n_rlnTomoSubTomosAre2DStacks 1\n"
    )
    tomo = tmp_path / "tomo_particles.star"
    tomo.write_text(block + plain.read_text())
    state = NativeParticleState(
        translation_offsets=rng.uniform(-2, 2, (7, 3)).astype(np.float32),
        class_assignments=np.zeros(7, dtype=np.int32),
        max_posterior=rng.random(7).astype(np.float32),
    )
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]

    for input_star, is_tomo in ((tomo, True), (plain, False)):
        main, optics = read_star(str(input_star))
        assert (run_files.input_general_block(input_star) is not None) == is_tomo
        # A subtomogram dataset (TomoDataset) names its particle STAR; a single-particle dataset has none.
        dataset = SimpleNamespace(voxel_size=2.5, n_images=7, **({"particles_star": str(input_star)} if is_tomo else {}))
        vdam_out = tmp_path / f"vdam_{input_star.stem}" / "run_it003_data.star"
        initial_model_io._write_data_star(str(vdam_out), main, optics, dataset, state)
        assert is_relion5_2d_stack_star(str(vdam_out)) == is_tomo
        written, written_optics = read_star(str(vdam_out))
        assert written_optics is not None and len(written) == 7
        assert sorted(written["_rlnImageName"]) == sorted(main["_rlnImageName"])
        assert "_rlnOriginZAngst" in written

        # The next job starts from InitialModel's file as written.
        out = tmp_path / f"refine_{input_star.stem}"
        rows = [np.sort(r) for r in half_rows]
        RunFileWriter(out, settings=_run_settings(out / "run"), input_star=vdam_out, half_rows=rows, background=False)(
            _k1_snapshot([4, 3], rng)
        )
        assert is_relion5_2d_stack_star(str(out / "run_it005_data.star")) == is_tomo
        refined, initial = read_star_blocks(out / "run_it005_data.star"), read_star_blocks(vdam_out)
        assert ("general" in refined) == ("general" in initial) == is_tomo
        if is_tomo:
            # The two writers cannot drift: the same block, first in the file, from either input form.
            assert list(refined)[0] == list(initial)[0] == "general"
            assert refined["general"] == initial["general"] == {"rlnTomoSubTomosAre2DStacks": ["1"]}
            direct = tmp_path / f"direct_{looped}"
            RunFileWriter(
                direct, settings=_run_settings(direct / "run"), input_star=tomo, half_rows=rows, background=False
            )(_k1_snapshot([4, 3], rng))
            assert read_star_blocks(direct / "run_it005_data.star")["general"] == initial["general"]


def test_healpix_cap_round_trips_and_the_continuing_run_keeps_its_own(tmp_path):
    """An uncapped run writes max_healpix_order as None and reads it back; a continued run
    keeps its own command-line limit (a restored cap of 7 overrode --max_healpix_order 12,
    bigbox 14557970)."""

    rng = np.random.default_rng(1)
    input_star = _write_input_star(tmp_path, 7)
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    names = read_star_blocks(input_star)["particles"]["rlnImageName"]
    for cap in (None, 7):
        snapshot = _k1_snapshot([4, 3], rng)
        snapshot.state_fields["max_healpix_order"] = cap
        optimiser = _writer(tmp_path, input_star, half_rows)(snapshot)
        read = read_run_files(optimiser, image_names=names, half_rows=half_rows)
        assert read.state_fields["max_healpix_order"] == cap
        for limit in (None, 12):
            restored = read.refinement_state(RefinementState(max_healpix_order=limit))
            assert restored.max_healpix_order == limit
            assert restored.healpix_order == snapshot.state_fields["healpix_order"]


def test_v1_files_written_before_the_subtomogram_field_read_as_single_particle(tmp_path):
    """relax.run_files.v1 files from before subtomogram refinement (1cdf2bb) have no
    relax_state_subtomogram; they are single-particle runs (EMPIAR-10202 it026 state,
    bigbox 14605497). Any other missing state field is still an error."""

    rng = np.random.default_rng(2)
    input_star = _write_input_star(tmp_path, 7)
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    names = read_star_blocks(input_star)["particles"]["rlnImageName"]
    optimiser = _writer(tmp_path, input_star, half_rows)(_k1_snapshot([4, 3], rng))
    text = Path(optimiser).read_text()
    lines = text.splitlines(keepends=True)
    older = "".join(line for line in lines if not line.startswith("_relax_state_subtomogram "))
    assert older != text
    Path(optimiser).write_text(older)
    assert read_run_files(optimiser, image_names=names, half_rows=half_rows).state_fields["subtomogram"] is False
    Path(optimiser).write_text("".join(line for line in lines if not line.startswith("_relax_state_healpix_order ")))
    with pytest.raises(KeyError, match="relax_state_healpix_order"):
        read_run_files(optimiser, image_names=names, half_rows=half_rows)


def test_run_files_use_relion_blocks_and_keep_input_columns(tmp_path):
    rng = np.random.default_rng(1)
    input_star = _write_input_star(tmp_path, 7)
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    optimiser = _writer(tmp_path, input_star, half_rows)(_k1_snapshot([4, 3], rng))

    general = read_star_blocks(optimiser)["optimiser_general"]
    assert general["rlnModelStarFile"] == "run_it005_half1_model.star"
    assert general["rlnModelStarFile2"] == "run_it005_half2_model.star"
    assert general["rlnDoSplitRandomHalves"] == "1" and general["rlnDoAutoRefine"] == "1"
    assert general["rlnRandomSeed"] == "17"
    model = read_star_blocks(tmp_path / "out" / "run_it005_half1_model.star")
    assert {"model_general", "model_classes", "model_class_1", "model_groups", "model_optics_group_1"} <= set(model)
    assert model["model_classes"]["rlnReferenceImage"] == ["run_it005_half1_class001.mrc"]
    # model.star carries Angstrom: MlModel::write stores 1./current_resolution of its reciprocal-Angstrom
    # member (ml_model.cpp:633), and readStar inverts it back (ml_model.cpp:202).
    assert float(model["model_general"]["rlnCurrentResolution"]) == pytest.approx(7.123456789)

    data = read_star_blocks(tmp_path / "out" / "run_it005_data.star")
    particles = data["particles"]
    assert particles["rlnDefocusU"] == ["12345.678901"] * 7  # untouched input text
    assert particles["rlnMicrographName"][:2] == ["mic_0.mrc", "mic_1.mrc"]
    assert particles["rlnRandomSubset"] == ["1", "2", "1", "2", "1", "2", "1"]
    assert data["optics"]["rlnImagePixelSize"] == ["2.5"]


def test_class3d_run_files_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    n, k = 6, 3
    input_star = _write_input_star(tmp_path, n)
    half_rows = [np.array([3, 1, 5, 0, 2, 4]), np.array([], dtype=np.int64)]
    stack = np.stack([_fourier(rng.standard_normal((BOX, BOX, BOX)).astype(np.float32)) for _ in range(k)])
    group_ids = [np.zeros(n, dtype=np.int64), np.zeros(0, dtype=np.int64)]
    snapshot = IterationSnapshot(
        relion_iteration=3,
        n_classes=k,
        box_size=BOX,
        pixel_size=2.5,
        tau2_fudge=4.0,
        means=[stack, stack],
        tau2_shells=rng.random((k, N_SHELLS)),
        data_vs_prior=rng.random((k, N_SHELLS)),
        noise_shells=[rng.random(N_SHELLS)] * 2,
        sigma_offset_angstrom=(4.5, 4.5),
        current_size=10,
        incr_size=10,
        has_high_fsc_at_limit=False,
        random_perturbation=0.25,
        state_fields=_state_fields(),
        rotation_eulers=[rng.uniform(-180, 180, (n, 3)), np.zeros((0, 3))],
        translations=[rng.uniform(-3, 3, (n, 2)).astype(np.float32), np.zeros((0, 2), np.float32)],
        image_corrections=[rng.uniform(0.8, 1.2, n).astype(np.float32), np.zeros(0, np.float32)],
        scale_corrections=[np.ones(n, np.float32), np.zeros(0, np.float32)],
        group_ids=group_ids,
        class_weights=np.array([0.5, 0.3, 0.2]),
        direction_prior=[rng.random((k, 48))] * 2,
        class_assignments=[rng.integers(0, k, n), np.zeros(0, dtype=np.int64)],
        fsc=None,
        fsc_for_growth=None,
        max_posterior=None,
        significant_counts=None,
        unfiltered_means=None,
        avg_norm_correction=(0.75 * BOX**2, 0.75 * BOX**2),
        acc_rot_per_class=np.array([3.5, 999.0, 7.25]),
        acc_trans_per_class_angstrom=np.array([1.5, 999.0, 2.75]),
        extra={"euler_dtype": "float64", "translation_dtype": "float32", "correction_dtype": "float32"},
    )
    optimiser = _writer(tmp_path, input_star, half_rows)(snapshot)
    out = tmp_path / "out"
    # RELION's conventions in the files: per-class accuracies, norm corrections over
    # ori_size**2, and pdf_direction rows summing to the class fraction.
    model = read_star_blocks(out / "run_it003_model.star")
    assert [float(v) for v in model["model_classes"]["rlnAccuracyRotations"]] == [3.5, 999.0, 7.25]
    assert [float(v) for v in model["model_classes"]["rlnAccuracyTranslationsAngst"]] == [1.5, 999.0, 2.75]
    assert float(model["model_general"]["rlnNormCorrectionAverage"]) == pytest.approx(0.75)
    for c in range(k):
        orient = np.asarray(model[f"model_pdf_orient_class_{c + 1}"]["rlnOrientationDistribution"], dtype=np.float64)
        assert_matches(orient, snapshot.direction_prior[0][c] * snapshot.class_weights[c])
    norm = np.asarray(read_star_blocks(out / "run_it003_data.star")["particles"]["rlnNormCorrection"], dtype=np.float64)
    assert_matches(norm[half_rows[0]], 0.75 / snapshot.image_corrections[0].astype(np.float64))
    assert (out / "run_it003_model.star").exists() and not (out / "run_it003_half1_model.star").exists()
    assert [p.name for p in sorted(out.glob("run_it003_class*.mrc"))] == [
        "run_it003_class001.mrc",
        "run_it003_class002.mrc",
        "run_it003_class003.mrc",
    ]
    names = read_star_blocks(input_star)["particles"]["rlnImageName"]
    read = read_run_files(optimiser, image_names=names, half_rows=half_rows)
    _assert_snapshots_match(read, snapshot)
    assert np.array_equal(read.class_assignments[0], snapshot.class_assignments[0])
    assert_matches(read.means[0], stack, rtol=1e-6)


def test_writer_frequency_and_unfiltered_maps(tmp_path):
    input_star = _write_input_star(tmp_path, 4)
    writer = _writer(tmp_path, input_star, [np.array([0, 1]), np.array([2, 3])], write_every=3)
    assert [writer.due(i) for i in range(1, 7)] == [False, False, True, False, False, True]
    assert writer.wants_unfiltered_maps(3, n_classes=1)
    assert not writer.wants_unfiltered_maps(3, n_classes=4)
    quiet = _writer(tmp_path, input_star, [np.array([0, 1]), np.array([2, 3])], write_unfiltered_maps=False)
    assert not quiet.wants_unfiltered_maps(1, n_classes=1)
    with pytest.raises(ValueError, match="partition"):
        _writer(tmp_path, input_star, [np.array([0, 1]), np.array([1, 3])])


def test_reader_refuses_files_without_relax_state(tmp_path):
    path = tmp_path / "run_it002_optimiser.star"
    path.write_text("data_optimiser_general\n\n_rlnCurrentIteration 2\n")
    with pytest.raises(ValueError, match="not written by relax"):
        read_run_files(path, image_names=[], half_rows=[np.array([]), np.array([])])


def test_reader_refuses_a_different_particle_list(tmp_path):
    rng = np.random.default_rng(3)
    input_star = _write_input_star(tmp_path, 7)
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    optimiser = _writer(tmp_path, input_star, half_rows)(_k1_snapshot([4, 3], rng))
    names = list(read_star_blocks(input_star)["particles"]["rlnImageName"])
    names[2], names[3] = names[3], names[2]
    with pytest.raises(ValueError, match="input order"):
        read_run_files(optimiser, image_names=names, half_rows=half_rows)


def test_tau2_volume_matches_the_mstep_volume():
    """The continued loop rebuilds mean_variance from shells exactly as the M-step builds it."""

    rng = np.random.default_rng(4)
    shape = (BOX, BOX, BOX)
    fsc = np.linspace(0.99, 0.05, N_SHELLS)
    volumes, shells = [], []
    for _ in range(2):
        weights = jnp.asarray(rng.random(shape).astype(np.float32).reshape(-1))
        prior, _, details = regularization.compute_relion_tau2_from_weights(
            weights, weights, fsc, shape, r_max=BOX // 2, return_details=True
        )
        volumes.append(prior)
        shells.append(np.asarray(details["prior_shells"], dtype=np.float64))
    for volume, shell in zip(volumes, shells):
        assert_matches(np.asarray(radial_shell_volume(shell, shape, dtype=jnp.float32)), np.asarray(volume))
    snapshot = _k1_snapshot([1, 1], rng)
    snapshot.tau2_shells = np.stack(shells)
    assert_matches(
        np.asarray(tau2_mean_variance(snapshot, shape, dtype=jnp.float32)),
        np.asarray(0.5 * (volumes[0] + volumes[1])),
    )


def test_growth_state_update_is_idempotent():
    """The files hold incr_size after the FSC update; the loop applies it once more."""

    fsc = np.concatenate([np.ones(20), np.linspace(0.9, 0.0, 45)])
    once = regularization.update_relion_growth_state_from_fsc(fsc, 60, incr_size=10, has_high_fsc_at_limit=False)
    twice = regularization.update_relion_growth_state_from_fsc(
        fsc, 60, incr_size=once[0], has_high_fsc_at_limit=once[1]
    )
    assert once == twice


def test_host_map_transforms_follow_recovar_convention():
    """Maps are transformed on the host with recovar's centered DFT convention."""

    from relax.refinement.run_files import _fourier_from_map, _real_from_fourier

    rng = np.random.default_rng(5)
    real = rng.standard_normal((BOX, BOX, BOX)).astype(np.float32)
    expected_ft = np.asarray(ftu.get_dft3(jnp.asarray(real))).reshape(-1)
    assert_matches(_fourier_from_map(real, (BOX, BOX, BOX)), expected_ft.astype(np.complex64))
    back = np.asarray(ftu.get_idft3(jnp.asarray(expected_ft.reshape(BOX, BOX, BOX)))).real
    assert_matches(_real_from_fourier(expected_ft, (BOX, BOX, BOX)), back.astype(np.float32))


def test_background_writer_hands_over_and_reports_errors(tmp_path):
    rng = np.random.default_rng(6)
    input_star = _write_input_star(tmp_path, 7)
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    writer = _writer(tmp_path, input_star, half_rows, background=True)
    assert writer.background
    snapshot = _k1_snapshot([4, 3], rng)
    optimiser = writer(snapshot)
    writer.wait()
    assert optimiser.exists() and not optimiser.with_name(optimiser.name + ".partial").exists()
    names = read_star_blocks(input_star)["particles"]["rlnImageName"]
    _assert_snapshots_match(read_run_files(optimiser, image_names=names, half_rows=half_rows), snapshot)

    broken = _k1_snapshot([4, 3], rng)
    broken.relion_iteration = 6
    broken.noise_shells = None  # the thread fails; the next call or wait() reports it
    writer(broken)
    with pytest.raises(RuntimeError, match="run files failed"):
        writer.wait()
    assert not (tmp_path / "out" / "run_it006_optimiser.star").exists()


def test_keep_iterations_removes_older_run_files_only(tmp_path):
    rng = np.random.default_rng(7)
    input_star = _write_input_star(tmp_path, 7)
    half_rows = [np.array([4, 0, 2, 6]), np.array([5, 1, 3])]
    out = tmp_path / "out"
    out.mkdir()
    (out / "final_merged.mrc").write_text("x")
    (out / "run_it009_optimiser.star").write_text("left by an earlier run")
    writer = _writer(tmp_path, input_star, half_rows, keep_iterations=2)
    for it in (1, 2, 3):
        snapshot = _k1_snapshot([4, 3], rng)
        snapshot.relion_iteration = it
        writer(snapshot)
    names = sorted(p.name for p in out.iterdir())
    assert not any(n.startswith("run_it001_") for n in names)
    for it in (2, 3):
        assert f"run_it{it:03d}_optimiser.star" in names and f"run_it{it:03d}_half1_class001_unfil.mrc" in names
    assert "final_merged.mrc" in names and "run_it009_optimiser.star" in names
    with pytest.raises(ValueError, match="keep_iterations"):
        _writer(tmp_path, input_star, half_rows, keep_iterations=-1)


def test_estimated_resolution_is_the_last_shell_before_data_vs_prior_drops_below_one():
    """``rlnEstimatedResolution`` of a class follows MlModel::calculateTotalFourierCoverage (ml_model.cpp:1646-1657)."""

    from types import SimpleNamespace

    from relax.refinement.run_files import _estimated_resolution_angstrom as estimate

    snapshot = SimpleNamespace(pixel_size=2.0, box_size=64)
    # Shells 0-3 are at or above 1, shell 4 is the first below: maxres 3, 128 A / 3. A later shell above 1 is ignored.
    assert estimate(np.array([999.0, 8.0, 2.0, 1.0, 0.5, 3.0]), snapshot) == pytest.approx(128.0 / 3.0)
    assert estimate(np.array([999.0, 8.0, 2.0]), snapshot) == pytest.approx(64.0)
    assert np.isposinf(estimate(np.array([999.0, 0.2, 5.0]), snapshot))
