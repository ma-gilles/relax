"""Rows of the local-search candidate mask decoded from the resident local table.

Test oracle: moved from relax/local_search/resident_layout.py, where only tests called it; it is the independent
host decoding ``tables_from_local_layout`` and ``materialize_local_chunk`` are checked against.
"""

import numpy as np

from relax.local_search.resident_layout import ResidentLocalTables


def expand_local_mask_rows(tables: ResidentLocalTables, start: int, stop: int) -> np.ndarray:
    """Reference (numpy) dense mask for rows ``[start, stop)``.

    Exactly ``LocalHypothesisLayout.sample_mask_rows(start, stop)``, including
    the ``None`` case, which the layout means as full support and the bucketed
    engine materializes as an all-true ``local_sample_mask``.
    """

    n_rows = stop - start
    if tables.row_mask_bits is None:
        return np.ones((n_rows, tables.n_trans), dtype=bool)
    return np.unpackbits(
        tables.row_mask_bits[start:stop],
        axis=1,
        count=int(tables.n_trans),
        bitorder="little",
    ).view(np.bool_)
