import numpy as np
from helpers.float_compare import matches

from scripts.extract_recovar_bpref_factor_operands import _compact_indices


def test_compact_indices_preserve_column_and_center_rows():
    image_shape = np.asarray((8, 8), dtype=np.int32)
    half_width = 5
    window = np.asarray((0, 1, 5, 6, 20, 21, 35, 36), dtype=np.int32)

    returned_window, centered = _compact_indices(
        {"image_shape": image_shape, "window_indices": window}
    )

    expected_rows = (window // half_width - image_shape[0] // 2) % image_shape[0]
    expected = expected_rows * half_width + window % half_width
    assert matches(returned_window, window)
    assert matches(centered, expected)
    assert matches(centered % half_width, window % half_width)


def test_compact_indices_do_not_silently_use_fftw_rows():
    window = np.asarray((0, 5, 10, 15), dtype=np.int32)
    _, centered = _compact_indices(
        {"image_shape": np.asarray((8, 8), dtype=np.int32), "window_indices": window}
    )

    assert not np.array_equal(centered, window)
