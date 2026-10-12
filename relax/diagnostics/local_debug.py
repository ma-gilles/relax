"""The score-dump label scope of the exact-local and dense engine dumps."""

from __future__ import annotations

import os
from contextlib import contextmanager


@contextmanager
def score_dump_label(label: str, *, local: bool = False):
    """Append a phase/class label and restore the process environment on exit.

    Local scores and fused posteriors retain their separate caller prefixes.
    Call sites create a fresh scope for each engine invocation.
    """
    names = (
        ("RELAX_LOCAL_SCORE_DUMP_LABEL", "RELAX_LOCAL_FUSED_POSTERIOR_DUMP_LABEL")
        if local else ("RELAX_DEBUG_PER_POSE_DUMP_LABEL",)
    )
    previous = {}
    for name in names:
        old = os.environ.get(name)
        previous[name] = old
        os.environ[name] = f"{old}_{label}" if old else label
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


