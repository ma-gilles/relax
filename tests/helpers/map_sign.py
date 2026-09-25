"""Raw-file sign check of relax maps against RELION maps (relax.helpers.map_io).

relax writes its maps in RELION's convention, so the file arrays of a relax map and the RELION
map of the same run correlate positively voxel for voxel; the pre-convention writer gave about
-1. The bound comes from seven medium-tier runs on the K1 5k/128 fixture before the change
(14375481 onwards, maps negated there): |corr| >= 0.99150 (e2e unfiltered half 2), >= 0.99999 for
the three-iteration fast cases. Correlation checks the sign only; FSC gates the quality.
"""

from __future__ import annotations

import mrcfile
import numpy as np

FILE_CORRELATION_MIN = 0.98


def file_correlation(lhs_path, rhs_path) -> float:
    """Pearson correlation of two MRC files' raw arrays."""
    arrays = []
    for path in (lhs_path, rhs_path):
        with mrcfile.open(str(path), permissive=True) as handle:
            arrays.append(np.asarray(handle.data, dtype=np.float64).ravel())
    a, b = (x - x.mean() for x in arrays)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def assert_relion_file_sign(relax_path, relion_path) -> float:
    corr = file_correlation(relax_path, relion_path)
    assert corr >= FILE_CORRELATION_MIN, (
        f"{relax_path} correlates {corr:.5f} with RELION's {relion_path} as files; "
        f"relax maps must carry RELION's sign (>= {FILE_CORRELATION_MIN})"
    )
    return corr
