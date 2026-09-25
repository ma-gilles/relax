"""Exact full-rotation-row streaming support versus the independent local layout."""

import numpy as np

from relax import sampling
from relax.local.local_layout import build_pass2_hypothesis_layout
from relax.ppca_initial_model.iteration_loop import _full_fine_mask_tile


def test_streamed_full_rotation_rows_preserve_local_support_and_order():
    translations = np.asarray([[-2, 0], [0, 0], [2, 0]], np.float32)
    rotation_prior = np.asarray([-0.5, -1.0, -1.5], np.float32)
    significant = [np.asarray([0, 1, 3, 8], np.int32), None, np.asarray([0, 4, 6], np.int32)]
    kwargs = dict(
        n_coarse_rotations=3, n_coarse_translations=3, nside_level=1,
        translations=translations, oversampling_order=1, translation_step=2.0,
        rotation_log_prior=rotation_prior, rotation_index_order="relion", allow_empty=False,
    )
    reference = build_pass2_hypothesis_layout(significant, **kwargs)
    shared = build_pass2_hypothesis_layout([None], **kwargs)
    fine_translations, parent = sampling.get_oversampled_translation_grid(
        translations, 2.0, oversampling_order=1,
    )
    assert np.array_equal(np.asarray(fine_translations, np.float32), shared.translation_grid)
    masks = _full_fine_mask_tile(significant, 3, 3, 8, parent)
    assert masks.shape == (3, 24, 12)
    assert masks.flags.c_contiguous
    for image in range(3):
        begin, end = map(int, reference.rotation_offsets[image:image + 2])
        assert np.array_equal(reference.rotations_flat[begin:end], shared.rotations_flat)
        assert np.array_equal(reference.rotation_ids_flat[begin:end], shared.rotation_ids_flat)
        assert np.array_equal(reference.rotation_log_priors_flat[begin:end], shared.rotation_log_priors_flat)
        assert np.array_equal(reference.rotation_posterior_ids_flat[begin:end], shared.rotation_posterior_ids_flat)
        assert np.array_equal(reference.sample_mask_rows(begin, end), masks[image])
