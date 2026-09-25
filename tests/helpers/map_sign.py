"""Sign and axis-convention check of relax map files against RELION's (relax.helpers.map_io).

relax writes its maps in RELION's convention, so the raw file arrays of a relax map and the
RELION map of the same run correlate positively; a sign or convention flip gives about -1
(matching maps measured |corr| >= 0.9915 on seven medium runs, 0.9999-1.0 at 01817da). This
checks the convention only, with a wide margin; map quality is gated by FSC
(tests/tiers/fsc_thresholds.json), never by this correlation.
"""

from __future__ import annotations

import mrcfile
import numpy as np

SIGN_CORRELATION_MIN = 0.5  # a flip reads about -1; matching maps read above 0.99


def file_correlation(lhs_path, rhs_path) -> float:
    """Pearson correlation of two MRC files' raw arrays."""
    arrays = []
    for path in (lhs_path, rhs_path):
        with mrcfile.open(str(path), permissive=True) as handle:
            arrays.append(np.asarray(handle.data, dtype=np.float64).ravel())
    a, b = (x - x.mean() for x in arrays)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def assert_same_sign_convention(relax_path, relion_path) -> float:
    """The relax file has RELION's sign and axis convention (sign and convention only)."""
    corr = file_correlation(relax_path, relion_path)
    assert corr >= SIGN_CORRELATION_MIN, (
        f"{relax_path} correlates {corr:.5f} with RELION's {relion_path} as files; a sign or "
        f"convention flip reads about -1 (sign check >= {SIGN_CORRELATION_MIN})"
    )
    return corr
