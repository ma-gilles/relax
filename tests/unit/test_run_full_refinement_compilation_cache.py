"""Relax entry points activate recovar's persistent compilation cache, which stays bounded."""

import os

import jax
import pytest

import relax.refinement.full_refinement as run_full_refinement
from relax.helpers import compilation_cache

pytestmark = pytest.mark.unit


@pytest.fixture
def restore_cache_config():
    saved = (jax.config.jax_compilation_cache_dir, jax.config.jax_persistent_cache_min_compile_time_secs)
    yield
    jax.config.update("jax_compilation_cache_dir", saved[0])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", saved[1])


class _StopAfterActivation(Exception):
    pass


def test_refinement_main_sets_the_live_cache_config_before_parsing(monkeypatch, tmp_path, restore_cache_config):
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", raising=False)
    jax.config.update("jax_compilation_cache_dir", None)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    def stop():
        raise _StopAfterActivation

    monkeypatch.setattr(run_full_refinement, "_parse_args", stop)
    with pytest.raises(_StopAfterActivation):
        run_full_refinement.main()
    assert jax.config.jax_compilation_cache_dir == str(tmp_path)
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.01


def test_no_cache_directory_leaves_the_config_alone(monkeypatch, restore_cache_config):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    before = jax.config.jax_compilation_cache_dir
    assert compilation_cache.activate_recovar_compilation_cache() is None
    assert jax.config.jax_compilation_cache_dir == before


def _write_entries(directory, n):
    for index in range(n):
        for suffix in ("-cache", "-atime"):
            path = directory / f"jit_f-{index:04d}{suffix}"
            path.write_bytes(b"x")
            os.utime(path, ns=(10**18 + index, 10**18 + index))


def test_prune_keeps_the_most_recently_written_entries(tmp_path):
    _write_entries(tmp_path, 10)
    (tmp_path / "unrelated").write_bytes(b"y")
    assert compilation_cache.prune_compilation_cache(tmp_path, max_entries=4) == 6
    kept = sorted(path.name for path in tmp_path.iterdir())
    assert kept == sorted(
        ["unrelated"] + [f"jit_f-{index:04d}{suffix}" for index in range(6, 10) for suffix in ("-cache", "-atime")]
    )
    assert compilation_cache.prune_compilation_cache(tmp_path, max_entries=4) == 0


def test_background_prune_runs_at_most_once_per_interval(tmp_path):
    _write_entries(tmp_path, 5)
    thread = compilation_cache.start_background_prune(tmp_path, max_entries=2)
    assert thread is not None
    thread.join(timeout=30)
    assert len(list(tmp_path.glob("*-cache"))) == 2
    _write_entries(tmp_path, 5)
    assert compilation_cache.start_background_prune(tmp_path, max_entries=2) is None
    assert len(list(tmp_path.glob("*-cache"))) == 5
