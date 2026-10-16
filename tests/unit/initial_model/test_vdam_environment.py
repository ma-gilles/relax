"""``VdamEnvironment.from_environ``: each variable the InitialModel command reads once, and how it is parsed.

The command reads its environment once, into ``VdamEnvironment`` (code rule 5); every other test builds the
record with explicit values, ``VdamEnvironment()`` or a ``replace`` of it (rule 22). These are the record's parse
tests and its guard against the unset environment.
"""

import pytest

from relax.vdam import native_options

pytestmark = pytest.mark.unit


def test_accumulator_dump_directory_is_read_once(monkeypatch, tmp_path):
    """RELAX_INITIAL_MODEL_ACCUM_DUMP_DIR is read into VdamEnvironment as set; unset or empty is None."""
    monkeypatch.delenv("RELAX_INITIAL_MODEL_ACCUM_DUMP_DIR", raising=False)
    assert native_options.VdamEnvironment.from_environ().accum_dump_dir is None
    monkeypatch.setenv("RELAX_INITIAL_MODEL_ACCUM_DUMP_DIR", "")
    assert native_options.VdamEnvironment.from_environ().accum_dump_dir is None
    monkeypatch.setenv("RELAX_INITIAL_MODEL_ACCUM_DUMP_DIR", str(tmp_path))
    assert native_options.VdamEnvironment.from_environ().accum_dump_dir == str(tmp_path)


def test_projector_dump_directory_is_read_once(monkeypatch, tmp_path):
    """RELAX_INITIAL_MODEL_PROJECTOR_DUMP_DIR is read once into VdamEnvironment, stripped; unset or blank is None."""
    for unset in (None, "  "):
        if unset is None:
            monkeypatch.delenv("RELAX_INITIAL_MODEL_PROJECTOR_DUMP_DIR", raising=False)
        else:
            monkeypatch.setenv("RELAX_INITIAL_MODEL_PROJECTOR_DUMP_DIR", unset)
        assert native_options.VdamEnvironment.from_environ().projector_dump_dir is None
    assert native_options.VdamEnvironment().projector_dump_dir is None
    monkeypatch.setenv("RELAX_INITIAL_MODEL_PROJECTOR_DUMP_DIR", f" {tmp_path} ")
    assert native_options.VdamEnvironment.from_environ().projector_dump_dir == str(tmp_path)


@pytest.mark.parametrize("name, field", [("RELAX_INITIALMODEL_SKIP_EXPECTED_ACCURACY", "skip_expected_accuracy")])
def test_expected_accuracy_diagnostic_switches_are_explicit_and_strict(monkeypatch, name, field):
    monkeypatch.delenv(name, raising=False)
    assert getattr(native_options.VdamEnvironment.from_environ(), field) is False

    monkeypatch.setenv(name, "1")
    assert getattr(native_options.VdamEnvironment.from_environ(), field) is True

    monkeypatch.setenv(name, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        native_options.VdamEnvironment.from_environ()


def test_engine_diagnostic_switches_are_read_once_by_from_environ(monkeypatch):
    monkeypatch.delenv("RELAX_ADAPTIVE_FRACTION", raising=False)
    monkeypatch.delenv("RELAX_DISABLE_SUBTRACT_PROJECTED_REFERENCE", raising=False)
    environment = native_options.VdamEnvironment.from_environ()
    assert (environment.adaptive_fraction, environment.subtract_projected_reference) == (None, True)

    monkeypatch.setenv("RELAX_ADAPTIVE_FRACTION", "")  # empty is unset, as the E-step read it
    monkeypatch.setenv("RELAX_DISABLE_SUBTRACT_PROJECTED_REFERENCE", "0")  # a false token leaves it on
    environment = native_options.VdamEnvironment.from_environ()
    assert (environment.adaptive_fraction, environment.subtract_projected_reference) == (None, True)
    monkeypatch.setenv("RELAX_DISABLE_SUBTRACT_PROJECTED_REFERENCE", "1")
    environment = native_options.VdamEnvironment.from_environ()
    assert (environment.adaptive_fraction, environment.subtract_projected_reference) == (None, False)

    monkeypatch.setenv("RELAX_ADAPTIVE_FRACTION", "0.99")
    environment = native_options.VdamEnvironment.from_environ()
    monkeypatch.delenv("RELAX_ADAPTIVE_FRACTION")  # read when the command built it, not later
    assert environment.adaptive_fraction == 0.99

    monkeypatch.setenv("RELAX_ADAPTIVE_FRACTION", "most")
    with pytest.raises(ValueError):
        native_options.VdamEnvironment.from_environ()


@pytest.mark.parametrize(
    "name, field, on",
    [
        ("RELAX_CLEAR_JAX_CACHES_PER_ITER", "clear_jax_caches_per_iteration", True),
        ("RELAX_DISABLE_SUBTRACT_PROJECTED_REFERENCE", "subtract_projected_reference", False),
    ],
)
def test_environment_flags_take_the_shared_tokens_and_refuse_a_malformed_value(monkeypatch, name, field, on):
    """deep2 V2-4: one flag parser (env_flags.parse_env_strict_flag); ``=0`` is off and a typo is an error."""
    monkeypatch.delenv(name, raising=False)
    assert getattr(native_options.VdamEnvironment.from_environ(), field) is (not on)
    for token, expected in (("1", on), ("yes", on), ("True", on), ("0", not on), ("false", not on), ("off", not on)):
        monkeypatch.setenv(name, token)
        assert getattr(native_options.VdamEnvironment.from_environ(), field) is expected
    monkeypatch.setenv(name, "maybe")
    with pytest.raises(ValueError, match=name):
        native_options.VdamEnvironment.from_environ()


def test_library_settings_are_read_once_by_from_environ(monkeypatch):
    """TF_GPU_ALLOCATOR, JAX_COMPILATION_CACHE_DIR and JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS as they are set;
    unset: "default", None and None."""
    names = ("TF_GPU_ALLOCATOR", "JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS")
    fields = ("cuda_allocator", "jax_compilation_cache_dir", "jax_persistent_cache_min_compile_time_secs")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    environment = native_options.VdamEnvironment.from_environ()
    assert tuple(getattr(environment, field) for field in fields) == ("default", None, None)
    for name, value in zip(names, ("cuda_malloc_async", "/cache/dir", "0.5"), strict=True):
        monkeypatch.setenv(name, value)
    environment = native_options.VdamEnvironment.from_environ()
    assert tuple(getattr(environment, field) for field in fields) == ("cuda_malloc_async", "/cache/dir", "0.5")


def test_vdam_environment_defaults_are_the_unset_environment(monkeypatch):
    """``VdamEnvironment()`` is what the command reads with no variable set; tests pass it, or it with the fields
    they need."""
    import os

    for name in list(os.environ):
        if name.startswith(("RELAX_", "RECOVAR_")):
            monkeypatch.delenv(name)
    for name in ("TF_GPU_ALLOCATOR", "JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"):
        monkeypatch.delenv(name, raising=False)
    assert native_options.VdamEnvironment.from_environ() == native_options.VdamEnvironment()


def test_float64_scoring_switch_is_refused_by_from_environ(monkeypatch):
    # The resident E-step never took it: InitialModel ignored RELAX_USE_FLOAT64_SCORING until 2026-10-06.
    monkeypatch.setenv("RELAX_USE_FLOAT64_SCORING", "1")
    with pytest.raises(NotImplementedError, match="float32 only"):
        native_options.VdamEnvironment.from_environ()
