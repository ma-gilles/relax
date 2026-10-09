"""Focused lifetime guards for K=1 references between EM iterations."""

import gc
import weakref

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.em_arrays import _hermitian_volume
from helpers.float_compare import assert_matches
from helpers.tiny_refinement import VOLUME_SHAPE, CallTrace, frame_holds, run_tiny_refinement

from relax.dense.score_outputs import PerHalfOutputs
from relax.diagnostics.observers import IntermediatesObserver
from relax.reconstruction import regularization_relion
from relax.refinement import finalization, iteration_loop, maximization, mean_helpers
from relax.refinement.iteration_snapshot import IterationSnapshot
from relax.refinement.refinement_options import CheckpointOptions

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128], ids=["production-f32", "diagnostic-f64"])
def test_snapshot_and_release_previous_k1_means_owns_host_copies(complex_dtype):
    first = np.arange(12, dtype=np.float64).astype(complex_dtype).reshape(3, 4)
    second = (first + complex_dtype(2.0 + 3.0j)).copy()
    means = [first, second]
    snapshots = mean_helpers._snapshot_and_release_previous_k1_means(means)

    assert means == [None, None]
    for snapshot, original in zip(snapshots, (first, second), strict=True):
        assert type(snapshot) is np.ndarray
        assert snapshot.flags.owndata
        assert not np.shares_memory(snapshot, original)
        assert_matches(snapshot, original)

    first[...] = complex_dtype(-9.0 + 4.0j)
    assert not np.array_equal(snapshots[0], first)


@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128], ids=["production-f32", "diagnostic-f64"])
def test_snapshot_of_device_k1_means_is_their_host_value(complex_dtype):
    """A device map's snapshot is its one host transfer: a NumPy array of the same dtype and values."""
    expected = np.arange(12, dtype=np.float64).astype(complex_dtype)
    means = [jnp.asarray(expected), None]
    snapshots = mean_helpers._snapshot_and_release_previous_k1_means(means)

    assert means == [None, None]
    assert snapshots[1] is None
    assert type(snapshots[0]) is np.ndarray
    assert snapshots[0].dtype == jnp.asarray(expected).dtype
    assert_matches(snapshots[0], np.asarray(jnp.asarray(expected)))


def test_k1_mean_release_precedes_tau_and_reconstruction(monkeypatch):
    """The controller drops the start-up volume before the first M-step, and each M-step releases the
    previous references from the reference model before the prior update and the reconstruction."""
    trace = CallTrace(monkeypatch)
    start_volume = _hermitian_volume(VOLUME_SHAPE, seed=42)
    entry = {}

    def maximization_starts(call):
        entry["model"] = call.args[0]
        assert all(value is not None for value in call.args[0].maps)
        assert not frame_holds(iteration_loop.refine_single_volume, start_volume)

    def previous_means_released(call):
        assert entry["model"].maps == [None, None]

    trace.wrap(maximization, "k1_maximization", before=maximization_starts)
    trace.wrap(maximization, "_snapshot_and_release_previous_k1_means")
    trace.wrap(maximization, "estimate_split_half_prior", before=previous_means_released)
    trace.wrap(maximization, "reconstruct_numbered_k1_halfmaps", before=previous_means_released)
    trace.wrap(regularization_relion, "compute_relion_fsc_from_backprojector", "fsc")
    trace.wrap(regularization_relion, "compute_relion_tau2_from_weights", "tau2")
    run_tiny_refinement(monkeypatch, init_volume=start_volume)

    m_steps = [call for call in trace.calls_seen if call.label == "k1_maximization" or "k1_maximization" in call.inside]
    assert [call.label for call in m_steps] == 2 * [
        "k1_maximization", "_snapshot_and_release_previous_k1_means", "estimate_split_half_prior", "fsc", "tau2",
        "tau2", "reconstruct_numbered_k1_halfmaps",
    ]
    # Each half's tau2 is computed from the FSC curve of this iteration's halves.
    for at in (3, 10):
        fsc = m_steps[at].result
        for tau2 in m_steps[at + 1 : at + 3]:
            assert any(value is fsc for value in (*tau2.args, *tau2.kwargs.values()))


def test_final_pass_starts_without_the_last_iterations_pass_outputs_or_snapshot(monkeypatch):
    """The numbered loop's pass outputs (both halves' accumulators) and its run-files snapshot are released at the
    iteration boundary, so the final all-data pass does not carry them (41 GB of host memory at EMPIAR-10202's box 800)."""
    import sys

    trace = CallTrace(monkeypatch)
    held = {}

    def final_pass_starts(call):
        function = iteration_loop.refine_single_volume
        code = getattr(function, "__wrapped__", function).__code__
        frame = sys._getframe(1)
        while frame is not None and frame.f_code is not code:
            frame = frame.f_back
        assert frame is not None
        held["types"] = {type(value) for value in frame.f_locals.values()}

    trace.wrap(finalization, "run_final_all_data", before=final_pass_starts)
    written = []

    class Writer:
        def wants_unfiltered_maps(self, *args, **kwargs):
            return True

        def due(self, relion_iteration):
            return True

        def __call__(self, snapshot):
            written.append(type(snapshot))

    run_tiny_refinement(monkeypatch, checkpoint=CheckpointOptions(writer=Writer()))

    assert written and set(written) == {IterationSnapshot}

    assert "types" in held
    assert PerHalfOutputs not in held["types"]
    assert IterationSnapshot not in held["types"]


def test_rotation_posterior_trajectory_is_kept_only_for_diagnostic_runs(monkeypatch, tmp_path):
    """Each iteration's float64 posterior copy (2.6 GB at EMPIAR-10202's fine orders) is a diagnostic record."""
    default = run_tiny_refinement(monkeypatch)
    diagnostic = run_tiny_refinement(monkeypatch, observer=IntermediatesObserver(tmp_path))

    assert default.history.rotation_posterior_trajectory_per_half == []
    assert len(diagnostic.history.rotation_posterior_trajectory_per_half) == 2


def test_production_runner_leaves_cold_start_host_owned_until_normalization(monkeypatch, tmp_path):
    """The command hands the controller the start-up reference as a host array; the controller normalizes it."""
    from helpers.tiny_main import controller_inputs

    inputs = controller_inputs(monkeypatch, tmp_path, "refine")
    volume, mean_variance, _ = inputs["startup"].take()
    assert type(volume) is np.ndarray
    assert type(mean_variance) is np.ndarray


def test_normalize_initial_means_reuses_immutable_shared_reference():
    import jax.numpy as jnp

    shared = jnp.arange(64, dtype=jnp.complex64)
    got = mean_helpers._normalize_initial_means(shared, n_classes=1)

    assert got[0] is got[1]
    assert_matches(np.asarray(got[0]), np.asarray(shared))
    got[0] = got[0].at[0].set(jnp.complex64(-1.0))
    assert got[0] is not got[1]
    assert_matches(np.asarray(got[1]), np.asarray(shared))


@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128], ids=["production-f32", "diagnostic-f64"])
def test_reference_owner_keeps_no_extra_map_alias_after_k1_release(complex_dtype):
    expected = np.arange(12, dtype=np.float64).astype(complex_dtype)
    model = mean_helpers.ReferenceModel(
        maps=[expected.copy(), expected.copy()], tau2=None, tau2_per_half=[None, None],
    )
    references = [weakref.ref(value) for value in model.maps]
    previous = mean_helpers._snapshot_and_release_previous_k1_means(model.maps)
    assert model.maps == [None, None]
    assert all(reference() is None for reference in references)
    for value in previous:
        assert_matches(value, expected)


def test_startup_handoff_gives_its_arrays_once_and_keeps_none():
    """``take`` hands over the start-up arrays and drops the holder's references; a second take raises."""
    from relax.refinement.startup_references import StartupHandoff

    volume, prior, real = np.ones(8, np.complex64), np.ones(8), np.ones((2, 2, 2))
    startup = StartupHandoff(volume, prior, real)
    assert startup.hands_reference_real

    taken = startup.take()
    assert all(got is want for got, want in zip(taken, (volume, prior, real), strict=True))
    held = weakref.ref(volume)
    del volume, taken
    assert held() is None
    with pytest.raises(RuntimeError, match="already taken"):
        startup.take()
    assert not StartupHandoff(None, None).hands_reference_real


def test_command_keeps_no_startup_array_once_the_controller_takes_them(monkeypatch, tmp_path):
    """main hands the start-up volume, tau2 and projector maps over in a holder and names none of them, so the
    controller frees them after the start-up instead of at return (relax#26: 12.3 GB at EMPIAR-10202's box 800)."""
    from helpers.tiny_main import _run_main, _stand_in_device, main_frame_arrays, write_tiny_data_dir

    seen = {}

    class Reached(Exception):
        pass

    def controller(*, startup, **kwargs):
        taken = [weakref.ref(array) for array in startup.take() if array is not None]
        main_frame_arrays()  # refreshes main's frame snapshot of its locals (Python 3.11)
        gc.collect()
        seen["alive"] = [ref() is not None for ref in taken]
        raise Reached

    _stand_in_device(monkeypatch)
    monkeypatch.setattr(iteration_loop, "refine_single_volume", controller)
    data = write_tiny_data_dir(tmp_path / "data")
    with pytest.raises(Reached):
        _run_main(monkeypatch, "refine", data, tmp_path / "out", [])

    # The volume, the tau2 and (K=1 --firstiter_cc default) the first projector's real maps: all freed.
    assert seen["alive"] == [False, False, False]
