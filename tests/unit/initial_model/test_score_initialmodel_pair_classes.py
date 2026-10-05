"""K > 1 arm pairs: a class that sits in its own frame is matched through its own rigid fit (2026-10-05, et15 ogtomo)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "score_initialmodel_maps.py"


def _module():
    spec = importlib.util.spec_from_file_location("score_initialmodel_maps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _blob(rng, n=16):
    """A smooth, asymmetric test volume."""
    v = np.zeros((n, n, n))
    for _ in range(6):
        c = rng.uniform(4, n - 4, size=3)
        z, y, x = np.indices(v.shape)
        v += np.exp(-((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) / rng.uniform(2, 6))
    return v


@pytest.mark.unit
def test_rotated_class_two_is_matched_through_its_own_fit():
    m = _module()
    rng = np.random.default_rng(0)
    a_vols = [_blob(rng), _blob(rng)]
    sh = m.Shells(16)
    a_fts = [sh.ft(v) for v in a_vols]
    # b after the pair's consensus transform: class 1 aligned with a's class 1, class 2 still rotated in its own frame
    raw_b = [a_vols[0].copy(), np.rot90(a_vols[1], k=1, axes=(1, 2)).copy()]
    cons_b_vols = [v.copy() for v in raw_b]
    via_ref = [[0.99, 0.1], [0.1, 0.99]]  # both arms' reference-frame maps match class 1-1 and 2-2
    calls = []

    def fit_class(moving, i):
        calls.append(i)
        return [("class_fit_proper", np.rot90(moving, k=-1, axes=(1, 2)) if i == 1 else moving)]

    before = m.auc(sh.fsc(a_fts[1], sh.ft(cons_b_vols[1])))
    matches = m._pair_class_matches(sh, a_fts, cons_b_vols, raw_b, via_ref, fit_class)
    by = {(i, j): (a, src) for i, j, a, src, _ in matches}
    assert sorted(by) == [(0, 0), (1, 1)] and sorted(calls) == [0, 1]
    assert by[(1, 1)][1] == "class_fit_proper" and by[(1, 1)][0] > 0.999 > before + 0.2
    assert by[(0, 0)][0] > 0.999


@pytest.mark.unit
def test_consensus_kept_when_the_class_fit_is_not_better():
    m = _module()
    rng = np.random.default_rng(1)
    a_vols = [_blob(rng), _blob(rng)]
    sh = m.Shells(16)
    a_fts = [sh.ft(v) for v in a_vols]
    raw_b = [v.copy() for v in a_vols]
    matches = m._pair_class_matches(
        sh, a_fts, raw_b, raw_b, [[1, 0], [0, 1]], lambda moving, i: [("class_fit_proper", np.roll(moving, 3, axis=0))]
    )
    assert [src for *_, src, _ in matches] == ["consensus", "consensus"]
