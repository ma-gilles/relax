"""Each controller environment variable is read once, by ``RefinementEnvironment.from_environ`` when the command
builds the run's options (code rule 5), and reaches its consumer as an argument."""

from __future__ import annotations

import os

import pytest
from helpers.tiny_main import controller_inputs
from helpers.tiny_refinement import run_tiny_refinement

from relax.refinement.refinement_options import RefinementEnvironment

pytestmark = pytest.mark.unit


def _from_environ(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    return RefinementEnvironment.from_environ()


def _unset_relax_environment(monkeypatch):
    """Delete every RELAX_ and RECOVAR_ variable for the test."""
    for name in list(os.environ):
        if name.startswith(("RELAX_", "RECOVAR_")):
            monkeypatch.delenv(name)


def _command_environment(monkeypatch, tmp_path, name, value):
    """The ``options.environment`` ``relax refine`` hands the controller with ``name=value`` set."""
    monkeypatch.setenv(name, value)
    return controller_inputs(monkeypatch, tmp_path, "refine")["options"].environment


def _received(monkeypatch, module, function, keyword, environment, **run):
    """The ``keyword`` arguments ``module.function`` receives in a tiny run with ``environment``."""
    original = getattr(module, function)
    seen = []

    def wrapped(*args, **kwargs):
        seen.append(kwargs[keyword])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, function, wrapped)
    run = {"max_iter": 1, **run}
    run_tiny_refinement(monkeypatch, final_after_max_iter=False, environment=environment, **run)
    return seen


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("  ", None), ("0", False), ("off", False), ("1", True), ("yes", True), ("anything", True)],
)
def test_firstiter_fused_bpref_is_read_once(monkeypatch, value, expected):
    env = _from_environ(monkeypatch, "RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", value)
    assert env.firstiter_fused_bpref is expected
    if value is None:
        assert env == RefinementEnvironment()


def test_firstiter_fused_bpref_reaches_the_command(monkeypatch, tmp_path):
    env = _command_environment(monkeypatch, tmp_path, "RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", "0")
    assert env.firstiter_fused_bpref is False


def test_firstiter_fused_bpref_reaches_the_batch_plan(monkeypatch):
    from relax.refinement import expectation

    monkeypatch.delenv("RELAX_K1_RELION_FIRSTITER_FUSED_BPREF", raising=False)
    seen = _received(monkeypatch, expectation, "prepare_half_batches", "bpref_half",
                     RefinementEnvironment(firstiter_fused_bpref=False))
    assert seen and {half.fused_request for half in seen} == {False}


@pytest.mark.parametrize(("value", "expected"), [(None, frozenset()), (" ", frozenset()), ("4, 7", {4, 7}), ("3,", {3})])
def test_diagnostic_float64_pass2_iterations_are_read_once(monkeypatch, value, expected):
    env = _from_environ(monkeypatch, "RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", value)
    assert env.diagnostic_float64_pass2_iterations == frozenset(expected)


def test_malformed_float64_pass2_iterations_are_refused(monkeypatch):
    with pytest.raises(ValueError, match="comma-separated integers"):
        _from_environ(monkeypatch, "RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "4,bad")


def test_diagnostic_float64_pass2_iterations_reach_the_command(monkeypatch, tmp_path):
    env = _command_environment(monkeypatch, tmp_path, "RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "1")
    assert env.diagnostic_float64_pass2_iterations == frozenset({1})


def test_diagnostic_float64_pass2_iterations_reach_the_expectation(monkeypatch):
    from relax.refinement import expectation

    monkeypatch.delenv("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", raising=False)
    original = expectation._diagnostic_float64_pass2_matches
    seen = []
    monkeypatch.setattr(
        expectation, "_diagnostic_float64_pass2_matches",
        lambda iteration, iterations: seen.append(iterations) or original(iteration, iterations),
    )
    run_tiny_refinement(monkeypatch, max_iter=1, final_after_max_iter=False,
                        environment=RefinementEnvironment(diagnostic_float64_pass2_iterations=frozenset({99})))
    assert seen and set(seen) == {frozenset({99})}


def test_low_pmax_refine_guard_is_read_once(monkeypatch, caplog):
    from relax.refinement.refinement_state import LowPmaxRefineGuard

    for name in ("RELAX_EM_LOW_PMAX_REFINE_MAX_AVE_PMAX", "RELAX_EM_LOW_PMAX_REFINE_MIN_RES_STALL",
                 "RELAX_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL"):
        monkeypatch.delenv(name, raising=False)
    assert _from_environ(monkeypatch, "RELAX_EM_LOW_PMAX_REFINE_GUARD", None).low_pmax_refine_guard is None
    assert _from_environ(monkeypatch, "RELAX_EM_LOW_PMAX_REFINE_GUARD", "0").low_pmax_refine_guard is None
    on = _from_environ(monkeypatch, "RELAX_EM_LOW_PMAX_REFINE_GUARD", "1").low_pmax_refine_guard
    assert on == LowPmaxRefineGuard(max_ave_pmax=0.15, min_res_stall=3, require_local_search=True)
    monkeypatch.setenv("RELAX_EM_LOW_PMAX_REFINE_MAX_AVE_PMAX", "0.2")
    monkeypatch.setenv("RELAX_EM_LOW_PMAX_REFINE_MIN_RES_STALL", "5")
    monkeypatch.setenv("RELAX_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL", "no")
    assert RefinementEnvironment.from_environ().low_pmax_refine_guard == LowPmaxRefineGuard(
        max_ave_pmax=0.2, min_res_stall=5, require_local_search=False)
    # An unrecognised value warns and keeps its default, as before.
    monkeypatch.setenv("RELAX_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL", "maybe")
    monkeypatch.setenv("RELAX_EM_LOW_PMAX_REFINE_MIN_RES_STALL", "x")
    guard = RefinementEnvironment.from_environ().low_pmax_refine_guard
    assert guard.require_local_search is True and guard.min_res_stall == 3
    assert "RELAX_EM_LOW_PMAX_REFINE_REQUIRE_LOCAL" in caplog.text


def test_low_pmax_refine_guard_reaches_the_command(monkeypatch, tmp_path):
    env = _command_environment(monkeypatch, tmp_path, "RELAX_EM_LOW_PMAX_REFINE_GUARD", "1")
    assert env.low_pmax_refine_guard is not None


def test_low_pmax_refine_guard_reaches_the_sampling_update(monkeypatch):
    from relax.refinement import numbered_transitions
    from relax.refinement.refinement_state import LowPmaxRefineGuard

    monkeypatch.delenv("RELAX_EM_LOW_PMAX_REFINE_GUARD", raising=False)
    guard = LowPmaxRefineGuard(max_ave_pmax=0.5, min_res_stall=2, require_local_search=False)
    seen = _received(monkeypatch, numbered_transitions, "advance_expectation_sampling", "low_pmax_guard",
                     RefinementEnvironment(low_pmax_refine_guard=guard), max_iter=2)
    assert seen and set(seen) == {guard}
    seen = _received(monkeypatch, numbered_transitions, "update_refinement_state", "low_pmax_guard",
                     RefinementEnvironment(low_pmax_refine_guard=guard), max_iter=2)
    assert seen and set(seen) == {guard}


@pytest.mark.parametrize(("value", "expected"), [(None, 0), ("12", 12)])
def test_fsc_packed_stream_threshold_is_read_once(monkeypatch, value, expected):
    from relax.reconstruction.volume_solver import ReconstructionPrograms

    if value is None:
        monkeypatch.delenv("RELAX_RELION_FSC_PACKED_STREAM_MIN_ELEMENTS", raising=False)
    else:
        monkeypatch.setenv("RELAX_RELION_FSC_PACKED_STREAM_MIN_ELEMENTS", value)
    assert ReconstructionPrograms.from_environ().fsc_packed_stream_min_elements == expected


def test_malformed_fsc_packed_stream_threshold_is_refused(monkeypatch):
    from relax.reconstruction.volume_solver import ReconstructionPrograms

    monkeypatch.setenv("RELAX_RELION_FSC_PACKED_STREAM_MIN_ELEMENTS", "many")
    with pytest.raises(ValueError):
        ReconstructionPrograms.from_environ()


def test_fsc_packed_stream_threshold_reaches_the_command(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_RELION_FSC_PACKED_STREAM_MIN_ELEMENTS", "12")
    options = controller_inputs(monkeypatch, tmp_path, "refine")["options"]
    assert options.variants.reconstruction.fsc_packed_stream_min_elements == 12


def test_fsc_packed_stream_threshold_reaches_the_fsc(monkeypatch):
    from dataclasses import replace

    from relax.reconstruction import regularization
    from relax.refinement.refinement_options import ScoringVariants

    variants = ScoringVariants.from_environ()
    variants = replace(variants, reconstruction=replace(variants.reconstruction, fsc_packed_stream_min_elements=7))
    seen = _received(monkeypatch, regularization, "compute_relion_fsc_from_backprojector", "packed_stream_min_elements",
                     RefinementEnvironment(), variants=variants)
    assert seen and set(seen) == {7}


@pytest.mark.parametrize(("forced", "minimum", "expected_forced", "expected_minimum"),
                         [(None, None, None, 200_000_000), ("always", "5", True, 5), ("never", None, False, 200_000_000),
                          ("auto", None, None, 200_000_000)])
def test_lowres_join_route_is_read_once(monkeypatch, forced, minimum, expected_forced, expected_minimum):
    from relax.reconstruction.volume_solver import ReconstructionPrograms

    for name, value in (("RELAX_LOWRES_JOIN_HOST_FALLBACK", forced), ("RELAX_LOWRES_JOIN_HOST_FALLBACK_MIN_ELEMENTS", minimum)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    route = ReconstructionPrograms.from_environ().lowres_join
    assert (route.forced_host, route.host_min_elements) == (expected_forced, expected_minimum)


def test_lowres_join_route_reaches_the_command(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_LOWRES_JOIN_HOST_FALLBACK", "always")
    options = controller_inputs(monkeypatch, tmp_path, "refine")["options"]
    assert options.variants.reconstruction.lowres_join.forced_host is True


def test_lowres_join_route_reaches_the_join(monkeypatch):
    from dataclasses import replace

    from relax.reconstruction import regularization
    from relax.refinement.refinement_options import ScoringVariants

    route = regularization.LowResolutionJoinRoute(forced_host=False, host_min_elements=11)
    variants = ScoringVariants.from_environ()
    variants = replace(variants, reconstruction=replace(variants.reconstruction, lowres_join=route))
    seen = _received(monkeypatch, regularization, "join_halves_at_low_resolution", "route",
                     RefinementEnvironment(), variants=variants, parity={"low_resol_join_halves_angstrom": 40.0})
    assert seen and set(seen) == {route}


@pytest.mark.parametrize(("value", "expected"), [(None, None), ("  ", None), (" /cache ", "/cache")])
def test_projector_cache_dir_is_read_once(monkeypatch, value, expected):
    env = _from_environ(monkeypatch, "RELAX_RELION_PROJECTOR_CACHE_DIR", value)
    assert env.projector_files.cache_dir == expected


def test_projector_cache_dir_reaches_the_command(monkeypatch, tmp_path):
    env = _command_environment(monkeypatch, tmp_path, "RELAX_RELION_PROJECTOR_CACHE_DIR", str(tmp_path / "cache"))
    assert env.projector_files.cache_dir == str(tmp_path / "cache")


def test_projector_files_reach_the_numbered_projectors(monkeypatch, tmp_path):
    from relax.steps import projector_preparation
    from relax.steps.projector_preparation import ProjectorFiles

    monkeypatch.delenv("RELAX_RELION_PROJECTOR_CACHE_DIR", raising=False)
    files = ProjectorFiles(cache_dir=str(tmp_path / "cache"))
    seen = _received(monkeypatch, projector_preparation, "prepare_scoring_projector", "files",
                     RefinementEnvironment(projector_files=files))
    assert seen and set(seen) == {files}


@pytest.mark.parametrize(("value", "expected"), [(None, None), ("", None), ("/dump", "/dump")])
def test_projector_dump_dir_is_read_once(monkeypatch, value, expected):
    env = _from_environ(monkeypatch, "RELAX_RELION_PROJECTOR_DUMP_DIR", value)
    assert env.projector_files.dump_dir == expected


def test_projector_dump_dir_reaches_the_command(monkeypatch, tmp_path):
    env = _command_environment(monkeypatch, tmp_path, "RELAX_RELION_PROJECTOR_DUMP_DIR", str(tmp_path / "dump"))
    assert env.projector_files.dump_dir == str(tmp_path / "dump")


def test_projector_dump_dir_writes_the_numbered_projectors(monkeypatch, tmp_path):
    from relax.steps.projector_preparation import ProjectorFiles

    monkeypatch.delenv("RELAX_RELION_PROJECTOR_DUMP_DIR", raising=False)
    run_tiny_refinement(monkeypatch, max_iter=1, final_after_max_iter=False,
                        environment=RefinementEnvironment(projector_files=ProjectorFiles(dump_dir=str(tmp_path))))
    assert sorted(path.name for path in tmp_path.glob("iter000_half*_relion_projector_half.npz")) == [
        "iter000_half0_relion_projector_half.npz", "iter000_half1_relion_projector_half.npz"]


def test_fsc_dump_is_read_once(monkeypatch):
    from relax.reconstruction.regularization import FscDump
    from relax.reconstruction.volume_solver import ReconstructionPrograms

    for name in ("RELAX_MSTEP_FSC_DUMP_TAG", "RELAX_MSTEP_FSC_DUMP_AVG"):
        monkeypatch.delenv(name, raising=False)
    for unset in (None, ""):
        if unset is None:
            monkeypatch.delenv("RELAX_MSTEP_FSC_DUMP_DIR", raising=False)
        else:
            monkeypatch.setenv("RELAX_MSTEP_FSC_DUMP_DIR", unset)
        assert ReconstructionPrograms.from_environ().fsc_dump is None
    monkeypatch.setenv("RELAX_MSTEP_FSC_DUMP_DIR", "/fsc")
    assert ReconstructionPrograms.from_environ().fsc_dump == FscDump(directory="/fsc", tag="recovar", average=False)
    monkeypatch.setenv("RELAX_MSTEP_FSC_DUMP_TAG", "relax")
    monkeypatch.setenv("RELAX_MSTEP_FSC_DUMP_AVG", "TRUE")
    assert ReconstructionPrograms.from_environ().fsc_dump == FscDump(directory="/fsc", tag="relax", average=True)


@pytest.mark.parametrize(("value", "expected"), [(None, True), ("0", False), ("1", True)])
def test_stable_reconstruction_windows_are_read_once(monkeypatch, value, expected):
    from relax.reconstruction.volume_solver import ReconstructionPrograms

    if value is None:
        monkeypatch.delenv("RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS", raising=False)
    else:
        monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_STABLE_WINDOWS", value)
    assert ReconstructionPrograms.from_environ().stable_windows is expected


@pytest.mark.parametrize(
    ("value", "expected"), [(None, None), ("auto", None), ("never", False), ("0", False), ("always", True), ("1", True)]
)
def test_host_irfft_route_is_read_once(monkeypatch, value, expected):
    from relax.reconstruction.volume_solver import ReconstructionPrograms

    if value is None:
        monkeypatch.delenv("RELAX_RELION_HOST_IRFFT", raising=False)
    else:
        monkeypatch.setenv("RELAX_RELION_HOST_IRFFT", value)
    assert ReconstructionPrograms.from_environ().host_irfft is expected


def test_host_fft_workers_are_read_once(monkeypatch):
    from relax.reconstruction.volume_solver import ReconstructionPrograms

    monkeypatch.delenv("RELAX_RELION_HOST_FFT_WORKERS", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert ReconstructionPrograms.from_environ().host_fft_workers == 1
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")
    assert ReconstructionPrograms.from_environ().host_fft_workers == 5
    monkeypatch.setenv("RELAX_RELION_HOST_FFT_WORKERS", "3")
    assert ReconstructionPrograms.from_environ().host_fft_workers == 3


def test_reconstruction_programs_helper_is_the_unset_environment(monkeypatch):
    """``helpers.run_options.reconstruction_programs()`` is what the command reads with no variable set and one host
    FFT thread (no ``SLURM_CPUS_PER_TASK``); tests pass it, or it with the fields they need, for the programs."""
    from helpers.run_options import reconstruction_programs

    from relax.reconstruction.volume_solver import ReconstructionPrograms

    _unset_relax_environment(monkeypatch)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert ReconstructionPrograms.from_environ() == reconstruction_programs()


def test_fsc_dump_reaches_the_command(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_MSTEP_FSC_DUMP_DIR", str(tmp_path / "fsc"))
    options = controller_inputs(monkeypatch, tmp_path, "refine")["options"]
    assert options.variants.reconstruction.fsc_dump.directory == str(tmp_path / "fsc")


def test_fsc_dump_writes_the_split_half_fsc(monkeypatch, tmp_path):
    from dataclasses import replace

    from relax.reconstruction.regularization import FscDump
    from relax.refinement.refinement_options import ScoringVariants

    monkeypatch.delenv("RELAX_MSTEP_FSC_DUMP_DIR", raising=False)
    variants = ScoringVariants.from_environ()
    variants = replace(variants, reconstruction=replace(
        variants.reconstruction, fsc_dump=FscDump(directory=str(tmp_path), tag="tiny", average=False)))
    run_tiny_refinement(monkeypatch, max_iter=1, final_after_max_iter=False, environment=RefinementEnvironment(),
                        variants=variants)
    assert (tmp_path / "tiny_downsampled_fsc.txt").exists()


def test_engine_tuning_is_read_once(monkeypatch):
    from relax.runtime.compile_ahead import CompileAheadConfig
    from relax.runtime.engine_tuning import EngineTuning

    monkeypatch.delenv("RELAX_EM_COMPILE_AHEAD_MAX_PROGRAMS", raising=False)
    assert _from_environ(monkeypatch, "RELAX_EM_COMPILE_AHEAD", None).engine == EngineTuning()
    assert _from_environ(monkeypatch, "RELAX_EM_COMPILE_AHEAD", "on").engine.compile_ahead == CompileAheadConfig(
        enabled=True)


def test_engine_tuning_reaches_the_command(monkeypatch, tmp_path):
    env = _command_environment(monkeypatch, tmp_path, "RELAX_EM_COMPILE_AHEAD", "1")
    assert env.engine.compile_ahead.enabled is True


def test_engine_tuning_defaults_are_the_unset_environment(monkeypatch):
    """``EngineTuning()`` is what the command reads with no variable set: tests pass it (or a ``replace`` of it)
    for the command's default engine settings."""
    from relax.runtime.engine_tuning import EngineTuning

    _unset_relax_environment(monkeypatch)
    assert EngineTuning.from_environ() == EngineTuning()


def test_scoring_overrides_defaults_are_the_unset_environment(monkeypatch):
    """``ScoringOverrides()`` is what the command reads with no variable set: the production scoring path."""
    from relax.runtime.scoring_overrides import ScoringOverrides

    _unset_relax_environment(monkeypatch)
    assert ScoringOverrides.from_environ() == ScoringOverrides()


def test_refinement_environment_defaults_are_the_unset_environment(monkeypatch):
    """``RefinementEnvironment()`` is what the command reads with no variable set; the tiny refinement runs with it
    unless a test gives another."""
    _unset_relax_environment(monkeypatch)
    assert RefinementEnvironment.from_environ() == RefinementEnvironment()


def test_engine_tuning_reaches_the_dense_engine(monkeypatch):
    from relax.runtime.compile_ahead import CompileAheadConfig
    from relax.runtime.engine_tuning import EngineTuning

    monkeypatch.delenv("RELAX_EM_COMPILE_AHEAD", raising=False)
    tuning = EngineTuning(compile_ahead=CompileAheadConfig(enabled=True, max_programs=5))
    calls = []
    run_tiny_refinement(monkeypatch, max_iter=1, final_after_max_iter=False, engine_calls=calls,
                        environment=RefinementEnvironment(engine=tuning))
    assert calls and all(call["kwargs"]["engine_tuning"] is tuning for call in calls)


def test_bpref_symmetry_chunk_is_read_once(monkeypatch):
    from relax.runtime.engine_tuning import EngineTuning

    monkeypatch.delenv("RELAX_RELION_BPREF_SYMMETRY_CHUNK_VOXELS", raising=False)
    assert EngineTuning.from_environ().bpref_symmetry_chunk_voxels == 16 * 1024 * 1024
    assert EngineTuning().bpref_symmetry_chunk_voxels == 16 * 1024 * 1024
    monkeypatch.setenv("RELAX_RELION_BPREF_SYMMETRY_CHUNK_VOXELS", "37")
    assert EngineTuning.from_environ().bpref_symmetry_chunk_voxels == 37
    monkeypatch.setenv("RELAX_RELION_BPREF_SYMMETRY_CHUNK_VOXELS", "many")
    with pytest.raises(ValueError):
        EngineTuning.from_environ()


def test_local_selector_chunking_is_read_once_at_the_boundary(monkeypatch):
    """A non-blank RELAX_LOCAL_SELECTOR_CHUNK_SIZE wins and the maximum is not read; otherwise
    RELAX_LOCAL_SELECTOR_MAX_ELEMENTS (default 16M); int() refuses text, as the old reader did."""
    import pytest

    from relax.runtime.engine_tuning import EngineTuning

    monkeypatch.delenv("RELAX_LOCAL_SELECTOR_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("RELAX_LOCAL_SELECTOR_MAX_ELEMENTS", raising=False)
    tuning = EngineTuning.from_environ()
    assert (tuning.local_selector_chunk_size, tuning.local_selector_max_elements) == (None, 16_000_000)
    monkeypatch.setenv("RELAX_LOCAL_SELECTOR_MAX_ELEMENTS", "1000")
    assert EngineTuning.from_environ().local_selector_max_elements == 1000
    monkeypatch.setenv("RELAX_LOCAL_SELECTOR_MAX_ELEMENTS", "many")
    with pytest.raises(ValueError):
        EngineTuning.from_environ()
    monkeypatch.setenv("RELAX_LOCAL_SELECTOR_CHUNK_SIZE", "7")
    assert EngineTuning.from_environ().local_selector_chunk_size == 7
    monkeypatch.setenv("RELAX_LOCAL_SELECTOR_CHUNK_SIZE", "seven")
    with pytest.raises(ValueError):
        EngineTuning.from_environ()
