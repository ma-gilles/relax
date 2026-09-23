"""The EM entry points opt in to the EM-scoped XLA default before importing jax.

Split from recovar tests/unit/test_em_xla_defaults.py (relax split): the policy
(``recovar.jax_config.em_xla_flag_additions``) stays tested in recovar; the entry
points that set ``RECOVAR_EM_XLA_DEFAULTS`` live here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
# Every EM entry point must opt in, so the entry assertions below run over all
# of them rather than over one named script.
ENTRIES = (
    REPO / "scripts" / "run_full_refinement.py",
    REPO / "relax" / "commands" / "initial_model.py",
)
MARKER = "RECOVAR_EM_XLA_DEFAULTS"


@pytest.mark.parametrize("ENTRY", ENTRIES, ids=lambda p: p.name)
def test_the_em_entry_sets_the_marker_before_importing_jax(ENTRY):
    """`XLA_FLAGS` is read when jax is imported, so the order is load-bearing."""

    lines = ENTRY.read_text().splitlines()
    marker_at = [i for i, l in enumerate(lines)
                 if MARKER in l and "setdefault" in l]
    assert marker_at, f"{ENTRY.name} does not opt in to the EM XLA defaults"
    first_import = next(
        i for i, l in enumerate(lines)
        if re.match(r"^(import jax|from jax|import recovar|from recovar|import relax|from relax)", l)
    )
    assert marker_at[0] < first_import, (
        f"the marker is set at line {marker_at[0] + 1}, after the first jax or "
        f"recovar or relax import at line {first_import + 1}; XLA_FLAGS would already have been read"
    )


@pytest.mark.parametrize("ENTRY", ENTRIES, ids=lambda p: p.name)
def test_the_entry_uses_setdefault_so_an_explicit_zero_wins(ENTRY):
    text = ENTRY.read_text()
    assert f'os.environ.setdefault("{MARKER}", "1")' in text, (
        "the entry must use setdefault, or RECOVAR_EM_XLA_DEFAULTS=0 in the "
        "environment could not turn the default off"
    )
