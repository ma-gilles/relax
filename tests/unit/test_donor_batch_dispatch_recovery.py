"""Donor memory policy and metadata-only batch shape regressions."""
import numpy as np
from relax.scoring.significance import _pad_significance_preprocess_inputs


class ShapeOnlyBatch:
    shape = (3, 8, 8)
    def __array__(self, *args, **kwargs):
        raise AssertionError("Shape inspection copied the image batch")


def test_unpadded_batch_shape_does_not_copy_to_host():
    batch = ShapeOnlyBatch()
    ctf = np.ones((3, 9))
    out = _pad_significance_preprocess_inputs(batch, ctf, None, None, None, None, target_size=3)
    assert out[0] is batch and out[1] is ctf
