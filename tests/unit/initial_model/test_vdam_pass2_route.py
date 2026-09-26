"""The VDAM E-step route per class count: K>1 defaults to the resident adaptive route, K=1 to exact-local."""

from types import SimpleNamespace

import pytest

from relax.sparse_pass2.engine_record import record_pass_engine, take_pass_engines
from relax.sparse_pass2.sparse_pass2_policy import ResidentConfigurationUnsupported
from relax.vdam.dense_adapter import _run_vdam_pass2_route, vdam_pass2_route


@pytest.fixture(autouse=True)
def _clear_engine_record():
    take_pass_engines()
    yield
    take_pass_engines()


@pytest.mark.parametrize(
    ("engine", "n_classes", "expected"),
    [
        ("auto", 1, ("local", False)),
        ("auto", 2, ("adaptive", False)),
        ("auto", 4, ("adaptive", False)),
        ("adaptive", 1, ("adaptive", True)),
        ("adaptive", 4, ("adaptive", True)),
        ("local", 4, ("local", True)),
        ("local_segmented", 2, ("local", True)),
    ],
)
def test_route_per_engine_and_class_count(engine, n_classes, expected):
    assert vdam_pass2_route(engine, n_classes) == expected


def _adaptive_ok():
    record_pass_engine("global", "resident")
    return SimpleNamespace(meta={})


def _adaptive_refused():
    raise ResidentConfigurationUnsupported("device-resident pass 2 does not implement this configuration: test")


def _local_run(calls):
    def run_local():
        calls.append("local")
        return SimpleNamespace(meta={})

    return run_local


def test_auto_k2_runs_the_resident_adaptive_route():
    calls = []
    result = _run_vdam_pass2_route("auto", 2, _adaptive_ok, _local_run(calls))
    assert calls == []
    assert result.meta["pass2_engine"] == "adaptive"
    assert result.meta["pass2_engines"] == ["global:resident"]


def test_auto_k1_runs_the_exact_local_route():
    calls = []
    result = _run_vdam_pass2_route("auto", 1, _adaptive_refused, _local_run(calls))
    assert calls == ["local"]
    assert result.meta["pass2_engine"] == "local"
    assert result.meta["pass2_engines"] == ["global:local"]


def test_auto_k2_refusal_falls_back_to_exact_local_with_the_reason():
    calls = []
    result = _run_vdam_pass2_route("auto", 4, _adaptive_refused, _local_run(calls))
    assert calls == ["local"]
    assert result.meta["pass2_engine"] == "local"
    (entry,) = result.meta["pass2_engines"]
    assert entry.startswith("global:local (resident adaptive route refused: ")


def test_explicit_adaptive_refusal_is_an_error():
    calls = []
    with pytest.raises(ResidentConfigurationUnsupported):
        _run_vdam_pass2_route("adaptive", 4, _adaptive_refused, _local_run(calls))
    assert calls == []
