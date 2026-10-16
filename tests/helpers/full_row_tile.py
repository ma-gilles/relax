"""One-tile call of the full-row accumulator, for tests that compare a single image tile."""

from relax.ppca_refinement.full_row_kernels import FullRowStream
from relax.ppca_refinement.full_row_stream import accumulate_full_row_tiles
from relax.ppca_refinement.residual_statistics import full_float32


@full_float32
def accumulate_full_row_tile(stream: FullRowStream, image_indices, significant_rows, *, enforce_x0: bool = True):
    """The statistics of one tile: ``accumulate_full_row_tiles`` of a single ``(image_indices, significant_rows)``."""
    return accumulate_full_row_tiles(stream, [(image_indices, significant_rows)], enforce_x0=enforce_x0)[0]
