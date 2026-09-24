"""Fail closed on absent, mismatched, or cross-capture K4 fixtures."""
from pathlib import Path

import pytest
from helpers import em_parity_oracles as fixtures


def _environment():
    return {
        "EM_PARITY_FAST_K4_H1_OS0_RELION_DIR": "/exact/oracle",
        "EM_PARITY_FAST_K4_H1_OS0_DISPATCH_SCHEDULE": "/exact/schedule.npz",
        "EM_PARITY_FAST_K4_RELION_DIR": "/wrong/oracle",
        "EM_PARITY_FAST_K4_DISPATCH_SCHEDULE": "/wrong/schedule.npz",
    }


def _metadata(monkeypatch, order=1, oversampling=0):
    monkeypatch.setattr(fixtures.starfile, "read", lambda path, **kw: {
        "sampling_general": {"rlnHealpixOrder": order},
        "optimiser_general": {"rlnAdaptiveOversampleOrder": oversampling},
    })


def test_case_pair_takes_precedence_and_verifies_same_capture(monkeypatch):
    from relax.relion import relion_worker_scale as worker

    _metadata(monkeypatch)
    marker = object()
    loaded, verified = [], []
    monkeypatch.setattr(worker, "load_relion_dispatch_schedule", lambda p: loaded.append(p) or marker)
    monkeypatch.setattr(worker, "verify_relion_dispatch_schedule_oracle", lambda s, p: verified.append((s, p)))
    oracle, args = fixtures.k4_oracle(1, 0, environ=_environment())
    assert oracle == Path("/exact/oracle")
    assert args == ["--relion-dispatch-schedule", "/exact/schedule.npz"]
    assert loaded == [Path("/exact/schedule.npz")]
    assert verified == [(marker, oracle)]


@pytest.mark.parametrize("grid", [(2, 0), (1, 1)])
def test_wrong_grid_fails_before_loading_schedule(monkeypatch, grid):
    _metadata(monkeypatch, *grid)
    with pytest.raises(ValueError, match="differs from requested"):
        fixtures.k4_oracle(1, 0, environ=_environment())


def test_partial_case_pair_cannot_mix_with_global_pair():
    env = _environment()
    del env["EM_PARITY_FAST_K4_H1_OS0_DISPATCH_SCHEDULE"]
    with pytest.raises(ValueError, match="Missing matched"):
        fixtures.k4_oracle(1, 0, environ=env)


def test_cross_capture_error_propagates(monkeypatch):
    from relax.relion import relion_worker_scale as worker

    _metadata(monkeypatch)
    monkeypatch.setattr(worker, "load_relion_dispatch_schedule", lambda p: object())
    def reject(*args):
        raise ValueError("oracle manifest mismatch")
    monkeypatch.setattr(worker, "verify_relion_dispatch_schedule_oracle", reject)
    with pytest.raises(ValueError, match="oracle manifest mismatch"):
        fixtures.k4_oracle(1, 0, environ=_environment())


def test_manifest_capture_is_the_default(monkeypatch):
    from helpers import em_fixtures

    from relax.relion import relion_worker_scale as worker

    _metadata(monkeypatch, 2, 1)
    monkeypatch.setattr(em_fixtures, "fixture_dir", lambda name: Path("/manifest") / name)
    monkeypatch.setattr(worker, "load_relion_dispatch_schedule", lambda p: object())
    monkeypatch.setattr(worker, "verify_relion_dispatch_schedule_oracle", lambda s, p: None)
    oracle, args = fixtures.k4_oracle(2, 1, environ={})
    assert oracle == Path("/manifest/k4_5k128_oracle_h2_os1")
    assert args == ["--relion-dispatch-schedule", "/manifest/k4_5k128_oracle_h2_os1/dispatch_schedule.npz"]


def test_grid_without_manifest_capture_needs_explicit_pair():
    with pytest.raises(ValueError, match="Missing matched"):
        fixtures.k4_oracle(1, 0, environ={})
