"""``relax build_cuda`` renames a fresh build over its output instead of rewriting it in place (fake make, CPU)."""

from __future__ import annotations

import pathlib
import sys

import pytest
from recovar import cuda_build

pytestmark = pytest.mark.unit


def test_build_cuda_renames_a_fresh_build_over_the_output(tmp_path, monkeypatch):
    from relax.commands import build_cuda
    from relax.cuda import kernels

    output = tmp_path / "librelax_cuda.so"
    output.write_text("old build")
    calls = []

    def fake_make(cmd, *, env):
        calls.append(cmd)
        pathlib.Path(cmd[-1].removeprefix("LIB=")).write_text("new build")

    monkeypatch.setattr(cuda_build.subprocess, "check_call", fake_make)
    monkeypatch.setattr(sys, "argv", ["relax build_cuda", "--output", str(output)])
    with open(output) as reader:  # a process still using the old library
        build_cuda.main()
        assert reader.read() == "old build"

    assert output.read_text() == "new build"
    assert len(calls) == 1 and calls[0][:3] == ["make", "-B", "-C"] and calls[0][-1] != f"LIB={output}"
    assert cuda_build.built_from(output, kernels._LIBRARY.source_digest())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["librelax_cuda.so", "librelax_cuda.so.sources.sha256"]


def test_relax_loader_uses_a_pinned_library_without_rebuilding(tmp_path, monkeypatch):
    from relax.cuda import kernels

    pinned = tmp_path / "librelax_cuda.so"
    pinned.write_text("pinned build")  # no recorded digest, as for a library built with make directly
    monkeypatch.setenv(kernels._LIBRARY.lib_env, str(pinned))
    monkeypatch.setattr(kernels._LIBRARY, "missing_required_symbol", lambda _path: None)
    monkeypatch.setattr(cuda_build.subprocess, "check_call", lambda *a, **k: pytest.fail("must not build"))
    assert kernels._LIBRARY.existing_path() == pinned.resolve()

    monkeypatch.setenv(kernels._LIBRARY.lib_env, str(tmp_path / "absent.so"))
    with pytest.raises(cuda_build.PinnedLibraryError, match="RELAX_CUDA_LIB="):
        kernels._LIBRARY.ensure_path()
