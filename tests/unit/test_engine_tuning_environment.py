"""``EngineTuning.from_environ``: each engine variable the command reads once, and how it is parsed.

The command reads the engine variables once, into ``EngineTuning`` (code rule 5); every other test builds the
record with explicit values (rule 22). These are the record's parse tests, the only tests besides the other
records' parse tests that set an engine variable and call ``from_environ``.
"""

import pytest

from relax.runtime import compile_ahead as ca
from relax.runtime.engine_tuning import (
    DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM,
    FINITE_CHECK_ENV,
    FINITE_CHECK_WARN_ENV,
    STABLE_FOURIER_WINDOW_QUANTUM_ENV,
    VDAM_STABLE_FOURIER_WINDOW_QUANTUM,
    EngineTuning,
    XHalfLargeGrid,
)

pytestmark = pytest.mark.unit


def test_stable_window_runtime_quantum_is_diagnostic_and_fail_closed(monkeypatch):
    monkeypatch.delenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, raising=False)
    assert EngineTuning.from_environ().stable_window_quantum == DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM == 8

    monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, "16")
    assert EngineTuning.from_environ().stable_window_quantum == 16

    for invalid in ("0", "3", "nope"):
        monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, invalid)
        with pytest.raises(ValueError, match=STABLE_FOURIER_WINDOW_QUANTUM_ENV):
            EngineTuning.from_environ()


@pytest.mark.parametrize("raw", ["", "0", "-1", "nan", "lots"])
def test_coarse_gemm_projection_cache_budget_refusals_at_the_boundary(monkeypatch, raw):
    """A set RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB must be a finite positive GiB count (blank too)."""
    variable = "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB"
    monkeypatch.setenv(variable, raw)
    with pytest.raises(ValueError, match=f"{variable} must be a finite positive number"):
        EngineTuning.from_environ()


def test_coarse_gemm_transient_budget_is_read_once_at_the_boundary(monkeypatch):
    """RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB: unset is 2 GiB; a finite positive GiB count (stripped)
    is taken; blank, zero, inf and text are refused, as the old reader did."""

    variable = "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB"
    monkeypatch.delenv(variable, raising=False)
    assert EngineTuning.from_environ().coarse_gemm_transient_max_bytes == 2 * 1024**3
    monkeypatch.setenv(variable, " 0.5 ")
    assert EngineTuning.from_environ().coarse_gemm_transient_max_bytes == 512 * 1024**2
    for raw in ("", "0", "inf", "two"):
        monkeypatch.setenv(variable, raw)
        with pytest.raises(ValueError, match=f"{variable} must be a finite positive number"):
            EngineTuning.from_environ()


@pytest.mark.parametrize("name", ["RELAX_DENSE_GEMM_SCORE_CACHE", "RELAX_DENSE_GEMM_CLASS_BATCH_SCORES"])
def test_dense_gemm_layout_switches_are_read_once_at_the_boundary(monkeypatch, name):
    """Unset is off; exactly 0 or 1 is taken; anything else (blank and " 1" too) is refused, as before."""
    field = {"RELAX_DENSE_GEMM_SCORE_CACHE": "dense_gemm_score_cache",
             "RELAX_DENSE_GEMM_CLASS_BATCH_SCORES": "dense_gemm_class_batch_scores"}[name]
    monkeypatch.delenv(name, raising=False)
    assert getattr(EngineTuning.from_environ(), field) is False
    for raw, expected in (("0", False), ("1", True)):
        monkeypatch.setenv(name, raw)
        assert getattr(EngineTuning.from_environ(), field) is expected
    for raw in ("", " 1", "true", "2"):
        monkeypatch.setenv(name, raw)
        with pytest.raises(ValueError, match=f"{name} must be 0 or 1"):
            EngineTuning.from_environ()


@pytest.mark.parametrize("value", ["eighty", "80GB", "0", "-4", "nan", "inf", " "])
def test_a_malformed_device_memory_override_is_refused(monkeypatch, value):
    """deep2 V2-4: RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB is a positive number or an error, never ignored."""
    monkeypatch.setenv("RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB", value)
    with pytest.raises(ValueError, match="RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB must be a positive number"):
        EngineTuning.from_environ()


@pytest.mark.parametrize("raw", ["0", "-0.1", "inf", "nan", "big"])
def test_relion_em_batch_sizing_projection_budget_override_rejects_invalid(monkeypatch, raw):
    monkeypatch.setenv("RELAX_RELION_EM_BATCH_PROJECTION_FRACTION", raw)

    with pytest.raises(ValueError, match="RELAX_RELION_EM_BATCH_PROJECTION_FRACTION must be a positive finite float"):
        EngineTuning.from_environ()


def test_default_depth_is_two_and_zero_turns_it_off(monkeypatch):
    monkeypatch.delenv("RELAX_EM_PREFETCH_BATCHES", raising=False)
    assert EngineTuning.from_environ().prefetch_depth == 2
    monkeypatch.setenv("RELAX_EM_PREFETCH_BATCHES", " 0 ")
    assert EngineTuning.from_environ().prefetch_depth == 0
    monkeypatch.setenv("RELAX_EM_PREFETCH_BATCHES", "")
    assert EngineTuning.from_environ().prefetch_depth == 2


@pytest.mark.parametrize("value", ["-1", "invalid", "1.5"])
def test_invalid_depth_rejected(monkeypatch, value):
    monkeypatch.setenv("RELAX_EM_PREFETCH_BATCHES", value)
    with pytest.raises(ValueError, match="non-negative integer"):
        EngineTuning.from_environ()


def _from_environ():
    """The warm-up the command reads (``EngineTuning.from_environ``)."""
    return EngineTuning.from_environ().compile_ahead


def test_default_is_off(monkeypatch):
    monkeypatch.delenv(ca.COMPILE_AHEAD_ENV, raising=False)
    assert _from_environ() == ca.CompileAheadConfig(enabled=False)
    assert EngineTuning().compile_ahead == _from_environ()


@pytest.mark.parametrize("token,expected", [
    ("1", True), ("on", True), ("true", True), ("YES", True),
    ("0", False), ("off", False), ("", False), ("no", False),
])
def test_env_tokens(monkeypatch, token, expected):
    monkeypatch.setenv(ca.COMPILE_AHEAD_ENV, token)
    assert _from_environ().enabled is expected


def test_the_program_cap_is_read_with_the_switch(monkeypatch):
    monkeypatch.setenv(ca.COMPILE_AHEAD_ENV, "1")
    monkeypatch.delenv(ca.COMPILE_AHEAD_MAX_ENV, raising=False)
    assert _from_environ().max_programs == ca.DEFAULT_MAX_PROGRAMS
    monkeypatch.setenv(ca.COMPILE_AHEAD_MAX_ENV, "3")
    assert _from_environ().max_programs == 3
    monkeypatch.setenv(ca.COMPILE_AHEAD_MAX_ENV, "-1")
    with pytest.raises(ValueError, match="must not be negative"):
        _from_environ()


def test_a_bad_token_is_refused(monkeypatch):
    monkeypatch.setenv(ca.COMPILE_AHEAD_ENV, "sometimes")
    with pytest.raises(ValueError):
        _from_environ()


@pytest.mark.parametrize("name", [
    "RELAX_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES", "RELAX_SPARSE_PASS2_RESIDENT_IMAGE_CAPACITIES",
    "RELAX_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES", "RELAX_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES",
])
@pytest.mark.parametrize("raw, expected", [
    (None, None), ("", None), ("  ", None),
    ("1, 2  4", (1, 2, 4)), ("2,2,4", (2, 2, 4)),
])
def test_capacity_ladders_are_read_once_at_the_boundary(monkeypatch, name, raw, expected):
    """Unset or blank is None (the engine's own ladder); equal adjacent values are kept."""
    field = {
        "RELAX_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES": "resident_row_capacities",
        "RELAX_SPARSE_PASS2_RESIDENT_IMAGE_CAPACITIES": "resident_image_capacities",
        "RELAX_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES": "local_row_capacities",
        "RELAX_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES": "local_image_capacities",
    }[name]
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert getattr(EngineTuning.from_environ(), field) == expected


@pytest.mark.parametrize("raw", [",", "0,2", "-1,2", "4,2", "two", "1;2"])
def test_capacity_ladder_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES", raw)
    with pytest.raises(ValueError):
        EngineTuning.from_environ()


def test_the_mode_is_read_once_and_is_off_unless_the_flag_is_set(monkeypatch):
    """Production must pay nothing: unset, the mode is "off" and no caller runs a check; the warn switch alone
    does not arm them."""

    monkeypatch.delenv(FINITE_CHECK_ENV, raising=False)
    monkeypatch.delenv(FINITE_CHECK_WARN_ENV, raising=False)
    assert EngineTuning.from_environ().finite_check == "off" == EngineTuning().finite_check
    monkeypatch.setenv(FINITE_CHECK_WARN_ENV, "1")
    assert EngineTuning.from_environ().finite_check == "off"
    monkeypatch.setenv(FINITE_CHECK_ENV, "1")
    assert EngineTuning.from_environ().finite_check == "warn"
    monkeypatch.delenv(FINITE_CHECK_WARN_ENV)
    assert EngineTuning.from_environ().finite_check == "raise"


def test_stage_glue_flags_fail_closed_on_bad_tokens(monkeypatch):
    """RELAX_COARSE_PAD_FINAL_IMAGE_BATCH (EngineTuning.coarse_pad_final_image_batch): on by default since the K=1
    resident flip; 1 and 0 are taken, anything else is refused."""
    name = "RELAX_COARSE_PAD_FINAL_IMAGE_BATCH"

    def read():
        return EngineTuning.from_environ().coarse_pad_final_image_batch

    monkeypatch.delenv(name, raising=False)
    assert read() is True
    monkeypatch.setenv(name, "1")
    assert read() is True
    monkeypatch.setenv(name, "0")
    assert read() is False
    monkeypatch.setenv(name, "maybe")
    with pytest.raises(ValueError, match=name):
        read()


@pytest.mark.parametrize("raw", ["0", "-5", "1e9", "lots"])
def test_firstiter_cc_budget_env_override_rejects_invalid(monkeypatch, raw):
    monkeypatch.setenv("RELAX_RELION_FIRSTITER_RECON_COMPLEX_BUDGET", raw)

    with pytest.raises(ValueError, match="RELAX_RELION_FIRSTITER_RECON_COMPLEX_BUDGET must be a positive integer"):
        EngineTuning.from_environ()


@pytest.mark.parametrize("name", ["TO_NATIVE_HALF", "FULL_HOST", "HOST_X0"])
def test_large_grid_switches_read_once_at_the_boundary(monkeypatch, name):
    """Each switch: a set flag is on unless it is a false token, and a blank one forces nothing (as
    env_flags.parse_env_flag); a blank minimum keeps the default and one that is not an integer is refused (rule 12)."""

    field = name.lower()
    variable = f"RELAX_RELION_X_HALF_{name}"
    monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(f"{variable}_MIN_VOXELS", "1000")
    switch = getattr(EngineTuning.from_environ().x_half_large_grid, field)
    assert (switch.forced, switch.on(999), switch.on(1000)) == (None, False, True)
    monkeypatch.setenv(f"{variable}_MIN_VOXELS", " ")
    assert getattr(EngineTuning.from_environ().x_half_large_grid, field) == getattr(XHalfLargeGrid(), field)
    monkeypatch.setenv(f"{variable}_MIN_VOXELS", "lots")
    with pytest.raises(ValueError, match=f"{variable}_MIN_VOXELS must be an integer, got 'lots'"):
        EngineTuning.from_environ()
    monkeypatch.setenv(f"{variable}_MIN_VOXELS", "1000")
    for raw, forced in (("", False), (" ", False), (" OFF ", False), ("no", False), ("0", False), ("anything", True)):
        monkeypatch.setenv(variable, raw)
        assert getattr(EngineTuning.from_environ().x_half_large_grid, field).on(1) is forced


def test_the_override_is_read_once(monkeypatch):
    """EngineTuning.from_environ reads RELAX_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES: unset or empty is None, a
    non-negative integer is taken, anything else is refused."""
    name = "RELAX_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES"
    monkeypatch.delenv(name, raising=False)
    assert EngineTuning.from_environ().projection_max_bytes is None
    monkeypatch.setenv(name, "")
    assert EngineTuning.from_environ().projection_max_bytes is None
    for raw, value in (("0", 0), ("654321", 654321)):
        monkeypatch.setenv(name, raw)
        assert EngineTuning.from_environ().projection_max_bytes == value
    for bad in ("-3", "lots"):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            EngineTuning.from_environ()


def test_glue_jit_is_on_by_default_and_reads_the_environment(monkeypatch):
    name = "RELAX_SPARSE_PASS2_RESIDENT_GLUE_JIT"
    monkeypatch.delenv(name, raising=False)
    assert EngineTuning.from_environ().resident_glue_jit is True is EngineTuning().resident_glue_jit
    monkeypatch.setenv(name, "0")
    assert EngineTuning.from_environ().resident_glue_jit is False
    monkeypatch.setenv(name, "1")
    assert EngineTuning.from_environ().resident_glue_jit is True


def test_local_block_row_program_is_read_once_at_the_boundary(monkeypatch):
    """RELAX_LOCAL_SEARCH_RESIDENT_BLOCK_ROW_PROGRAM (env_flags.parse_env_flag): unset or blank on, false tokens off."""
    for raw, expected in ((None, True), ("", True), ("0", False), ("off", False), ("1", True)):
        if raw is None:
            monkeypatch.delenv("RELAX_LOCAL_SEARCH_RESIDENT_BLOCK_ROW_PROGRAM", raising=False)
        else:
            monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_BLOCK_ROW_PROGRAM", raw)
        assert EngineTuning.from_environ().local_block_row_program is expected


@pytest.mark.parametrize(("value", "expected"), [(None, None), ("  ", None), ("128", 128), (" 16 ", 16)])
def test_mstep_block_rows_are_read_once(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS", raising=False)
    else:
        monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS", value)
    assert EngineTuning.from_environ().mstep_block_rows == expected


@pytest.mark.parametrize("value", ["0", "-4", "12", "rows"])
def test_malformed_mstep_block_rows_are_refused(monkeypatch, value):
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS", value)
    with pytest.raises(ValueError):
        EngineTuning.from_environ()


def test_resident_operands_verify_is_read_once_at_the_boundary(monkeypatch):
    """RELAX_SPARSE_PASS2_RESIDENT_OPERANDS_VERIFY (env_flags.parse_env_flag): unset and blank off, set on."""
    for raw, expected in ((None, False), ("", False), ("1", True), ("off", False)):
        if raw is None:
            monkeypatch.delenv("RELAX_SPARSE_PASS2_RESIDENT_OPERANDS_VERIFY", raising=False)
        else:
            monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_OPERANDS_VERIFY", raw)
        assert EngineTuning.from_environ().resident_operands_verify is expected


def test_coarse_gemm_projection_cache_request_is_read_once_at_the_boundary(monkeypatch):
    """RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE: unset is None (the pass asks for the cache, not explicitly);
    the true and false tokens are explicit; anything else is refused, as the strict flag did."""
    variable = "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE"
    monkeypatch.delenv(variable, raising=False)
    assert EngineTuning.from_environ().coarse_gemm_projection_cache is None
    for raw, expected in (("0", False), (" OFF ", False), ("no", False), ("1", True), ("Yes", True), ("on", True)):
        monkeypatch.setenv(variable, raw)
        assert EngineTuning.from_environ().coarse_gemm_projection_cache is expected
    for raw in ("automatic", ""):
        monkeypatch.setenv(variable, raw)
        with pytest.raises(ValueError, match=variable):
            EngineTuning.from_environ()


def test_a_device_memory_override_is_taken_and_an_empty_one_is_unset(monkeypatch):
    """The command reads RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB once (EngineTuning.device_memory_bytes)."""
    monkeypatch.setenv("RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB", "1.5")
    assert EngineTuning.from_environ().device_memory_bytes == int(1.5 * 1024**3)
    monkeypatch.setenv("RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB", "")
    assert EngineTuning.from_environ().device_memory_bytes is None is EngineTuning().device_memory_bytes


def test_relion_em_batch_projection_fraction_is_read_once(monkeypatch):
    """RELAX_RELION_EM_BATCH_PROJECTION_FRACTION: unset or blank is the default fraction."""
    monkeypatch.delenv("RELAX_RELION_EM_BATCH_PROJECTION_FRACTION", raising=False)
    assert EngineTuning.from_environ().em_batch_projection_fraction == EngineTuning().em_batch_projection_fraction
    monkeypatch.setenv("RELAX_RELION_EM_BATCH_PROJECTION_FRACTION", " ")
    assert EngineTuning.from_environ().em_batch_projection_fraction == 0.20


def test_firstiter_cc_budget_is_read_once(monkeypatch):
    """RELAX_RELION_FIRSTITER_RECON_COMPLEX_BUDGET: unset or blank is the default; a positive integer is taken."""
    monkeypatch.setenv("RELAX_RELION_FIRSTITER_RECON_COMPLEX_BUDGET", " ")
    budget = EngineTuning.from_environ().firstiter_cc_recon_complex_budget
    assert budget == EngineTuning().firstiter_cc_recon_complex_budget == 268_435_456

    monkeypatch.setenv("RELAX_RELION_FIRSTITER_RECON_COMPLEX_BUDGET", str(3 * 268_435_456))
    assert EngineTuning.from_environ().firstiter_cc_recon_complex_budget == 3 * 268_435_456


@pytest.mark.parametrize(
    "name",
    [
        "RELAX_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES",
        "RELAX_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES",
        "RELAX_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS",
    ],
)
def test_pass2_memory_overrides_are_read_once(monkeypatch, name):
    """Each override is read by EngineTuning.from_environ: unset or empty is None, a positive integer is taken,
    anything else is refused."""
    field = {
        "RELAX_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES": "translation_tile_max_bytes",
        "RELAX_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES": "adjoint_block_max_bytes",
        "RELAX_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS": "max_projected_rotations",
    }[name]
    monkeypatch.delenv(name, raising=False)
    assert getattr(EngineTuning.from_environ(), field) is None
    monkeypatch.setenv(name, "")
    assert getattr(EngineTuning.from_environ(), field) is None
    monkeypatch.setenv(name, "4096")
    assert getattr(EngineTuning.from_environ(), field) == 4096
    for bad in ("0", "-3", "lots"):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError, match="must be a positive integer"):
            EngineTuning.from_environ()


def test_the_local_projector_call_cap_is_read_once(monkeypatch):
    """RELAX_LOCAL_SEARCH_RESIDENT_PROJECTION_CALL_MAX_BYTES replaces the 4 GiB cap; a value that is not positive
    is refused."""
    gib = 1024**3
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_PROJECTION_CALL_MAX_BYTES", str(gib))
    assert EngineTuning.from_environ().local_projection_call_max_bytes == gib
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_PROJECTION_CALL_MAX_BYTES", "0")
    with pytest.raises(ValueError, match="must be positive"):
        EngineTuning.from_environ()


def test_local_chunk_profile_is_read_once_at_the_boundary(monkeypatch):
    """RELAX_LOCAL_SEARCH_RESIDENT_CHUNK_PROFILE (env_flags.parse_env_flag, default off)."""
    monkeypatch.delenv("RELAX_LOCAL_SEARCH_RESIDENT_CHUNK_PROFILE", raising=False)
    assert EngineTuning.from_environ().local_chunk_profile is False
    monkeypatch.setenv("RELAX_LOCAL_SEARCH_RESIDENT_CHUNK_PROFILE", "1")
    assert EngineTuning.from_environ().local_chunk_profile is True


def test_the_resident_operand_budget_is_read_once(monkeypatch):
    """RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES: a positive byte count (stripped) is taken; anything else is
    refused."""
    gib = 1024**3
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES", f" {3 * gib} ")
    assert EngineTuning.from_environ().resident_operand_max_bytes == 3 * gib
    for bad in ("0", "-5", "lots"):
        monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES", bad)
        with pytest.raises(ValueError):
            EngineTuning.from_environ()


def test_the_preparation_batch_is_read_once(monkeypatch):
    """RELAX_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH: unset is 256; a positive integer is taken; anything else
    is refused."""
    monkeypatch.delenv("RELAX_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH", raising=False)
    assert EngineTuning.from_environ().prepare_image_batch == 256 == EngineTuning().prepare_image_batch
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH", "64")
    assert EngineTuning.from_environ().prepare_image_batch == 64
    for bad in ("0", "x"):
        monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH", bad)
        with pytest.raises(ValueError):
            EngineTuning.from_environ()


def test_the_resident_diagnostics_are_read_once(monkeypatch):
    """A set diagnostic dump directory or flag is named in EngineTuning.resident_diagnostics; blank directories
    and false flags are not."""
    from relax.runtime.engine_tuning import RESIDENT_DIAGNOSTIC_DIR_ENVS, RESIDENT_DIAGNOSTIC_FLAG_ENVS

    for name in RESIDENT_DIAGNOSTIC_DIR_ENVS + RESIDENT_DIAGNOSTIC_FLAG_ENVS:
        monkeypatch.delenv(name, raising=False)
    assert EngineTuning.from_environ().resident_diagnostics == ()
    monkeypatch.setenv("RELAX_BPREF_MEMBERSHIP_DUMP_DIR", "  ")
    monkeypatch.setenv("RELAX_SPARSE_PASS2_LOG_CANDIDATE_DENSITY", "0")
    assert EngineTuning.from_environ().resident_diagnostics == ()
    monkeypatch.setenv("RELAX_PASS2_DUMP_DIR", "/tmp/does-not-matter")
    monkeypatch.setenv("RELAX_RELION_X_HALF_BP_FUSED_ATOMICS", "1")
    requested = ("RELAX_PASS2_DUMP_DIR", "RELAX_RELION_X_HALF_BP_FUSED_ATOMICS")
    assert EngineTuning.from_environ().resident_diagnostics == requested


def test_chunk_timing_is_read_once_at_the_boundary(monkeypatch):
    """RELAX_SPARSE_PASS2_RESIDENT_CHUNK_TIMING (env_flags.parse_env_flag, default off)."""
    monkeypatch.delenv("RELAX_SPARSE_PASS2_RESIDENT_CHUNK_TIMING", raising=False)
    assert EngineTuning.from_environ().chunk_timing is False
    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_CHUNK_TIMING", "1")
    assert EngineTuning.from_environ().chunk_timing is True
    for raw, expected in (("", False), ("off", False), ("yes", True)):
        monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_CHUNK_TIMING", raw)
        assert EngineTuning.from_environ().chunk_timing is expected


def test_resident_operands_switch_is_read_once_at_the_boundary(monkeypatch):
    """RELAX_SPARSE_PASS2_RESIDENT_OPERANDS (env_flags.parse_env_flag, default on)."""
    for raw, expected in ((None, True), ("", True), ("0", False), ("1", True)):
        if raw is None:
            monkeypatch.delenv("RELAX_SPARSE_PASS2_RESIDENT_OPERANDS", raising=False)
        else:
            monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT_OPERANDS", raw)
        assert EngineTuning.from_environ().resident_operands is expected


def test_the_pass2_projection_cache_request_is_read_once(monkeypatch):
    """RELAX_SPARSE_PASS2_PROJECTION_CACHE: unset and "auto" are None (the default), "on" True, "off" False;
    anything else is refused."""

    def requested(value):
        if value is None:
            monkeypatch.delenv("RELAX_SPARSE_PASS2_PROJECTION_CACHE", raising=False)
        else:
            monkeypatch.setenv("RELAX_SPARSE_PASS2_PROJECTION_CACHE", value)
        return EngineTuning.from_environ().pass2_projection_cache

    assert requested(None) is None and EngineTuning().pass2_projection_cache is None
    assert requested("on") is True and requested("off") is False
    assert requested("AUTO") is None and requested("Yes") is True
    with pytest.raises(ValueError, match="must be 'auto', 'on', or 'off'"):
        requested("invalid")


def test_the_quantum_override_is_read_once_at_the_boundary(monkeypatch):
    """RELAX_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM: unset or blank is the command's own quantum; a set value
    replaces it for every command; odd, small and text values are refused, as before."""
    for raw in (None, " "):
        if raw is None:
            monkeypatch.delenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, raising=False)
        else:
            monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, raw)
        assert EngineTuning.from_environ().stable_window_quantum == DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM
        assert EngineTuning.from_environ().stable_window_quantum == EngineTuning().stable_window_quantum
        vdam = EngineTuning.from_environ(stable_window_quantum=VDAM_STABLE_FOURIER_WINDOW_QUANTUM)
        assert vdam.stable_window_quantum == VDAM_STABLE_FOURIER_WINDOW_QUANTUM
        assert EngineTuning.from_environ(stable_window_quantum=24).stable_window_quantum == 24
    monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, " 16 ")
    assert EngineTuning.from_environ().stable_window_quantum == 16
    assert EngineTuning.from_environ(stable_window_quantum=VDAM_STABLE_FOURIER_WINDOW_QUANTUM).stable_window_quantum == 16
    for bad in ("7", "0", "eight"):
        monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, bad)
        with pytest.raises(ValueError, match="must be an even integer >= 2"):
            EngineTuning.from_environ()

