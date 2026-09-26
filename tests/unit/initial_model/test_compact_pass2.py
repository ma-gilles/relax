"""Focused contracts for the single InitialModel pass-2 engine."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from relax.helpers.types import make_relion_stats
from relax.vdam.sparse_pass2_estep import _resolve_pass2_engine

pytestmark = pytest.mark.unit


class _StatsResult(NamedTuple):
    stats: object
    per_class_stats: tuple[object, ...]


def _stats(rotation_sums):
    return make_relion_stats(
        log_evidence_per_image=np.zeros(1),
        best_log_score_per_image=np.zeros(1),
        max_posterior_per_image=np.ones(1),
        rotation_posterior_sums=np.asarray(rotation_sums, dtype=np.float64),
        rotation_dtype=jnp.float64,
    )


def test_one_engine_serves_k1_and_kclass():
    """The compact sparse pass-2 route is gone: every selector resolves to the exact-local
    engine, which scores all classes in one pass over class-segmented rows."""
    for token in ("auto", "local", "local_segmented", " AUTO "):
        assert _resolve_pass2_engine(token) in {"auto", "local", "local_segmented"}


def test_compact_engine_is_no_longer_selectable():
    with pytest.raises(ValueError, match="must be one of 'auto', 'local' or 'local_segmented'"):
        _resolve_pass2_engine("compact")
