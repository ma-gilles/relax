"""Class3D local searches: one local row set per image, repeated per class (expand_local_layout_classes)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.fine_pass.resident_candidates import CapacityChunk
from relax.local_search.layout import (
    build_local_adaptive_pass2_hypothesis_layout,
    build_local_hypothesis_layout,
    expand_local_layout_classes,
)
from relax.local_search.resident_layout import materialize_local_chunk, tables_from_local_layout
from relax.sampling import build_local_search_grid_metadata

pytestmark = pytest.mark.unit

ORDER = 1
N_IMAGES = 5
TRANSLATIONS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)


def _parent():
    rng = np.random.default_rng(7)
    eulers = np.column_stack(
        [rng.uniform(0, 360, N_IMAGES), rng.uniform(20, 160, N_IMAGES), rng.uniform(0, 360, N_IMAGES)]
    )
    return build_local_hypothesis_layout(
        eulers, None, 0.35, 0.35, ORDER, TRANSLATIONS, np.zeros((N_IMAGES, 2), dtype=np.float32), 3.0, None, 1.0,
        grid_metadata=build_local_search_grid_metadata(ORDER), translation_prior_reference_translations=TRANSLATIONS,
        dtype=np.float32,
    )


def _image_rows(layout, image):
    return slice(int(layout.rotation_offsets[image]), int(layout.rotation_offsets[image + 1]))


def test_expansion_repeats_each_image_rows_class_major():
    parent = _parent()
    expanded = expand_local_layout_classes(parent, 3)
    assert expanded.n_classes == 3
    assert_matches(np.asarray(expanded.rotation_counts), 3 * np.asarray(parent.rotation_counts))
    for image in range(N_IMAGES):
        rows, source = _image_rows(expanded, image), _image_rows(parent, image)
        n = source.stop - source.start
        assert_matches(expanded.row_class_flat[rows], np.repeat(np.arange(3, dtype=np.int32), n))
        for name in ("rotation_ids_flat", "rotations_flat", "rotation_log_priors_flat", "mstep_rotations_flat"):
            if getattr(parent, name) is None:
                continue
            assert_matches(np.asarray(getattr(expanded, name))[rows], np.tile(np.asarray(getattr(parent, name))[source], (3,) + (1,) * (np.ndim(getattr(parent, name)) - 1)))
    assert_matches(np.asarray(expanded.translation_log_priors), np.asarray(parent.translation_log_priors))
    with pytest.raises(ValueError, match="already expanded"):
        expand_local_layout_classes(expanded, 2)


def _class_samples(parent, rng, n_classes):
    """Per image and class, a random subset of (rotation, translation) parents, in each encoding."""

    n_global, t = int(parent.n_global_rotations), int(TRANSLATIONS.shape[0])
    per_class, encoded = [], []
    for image in range(N_IMAGES):
        ids = np.asarray(parent.rotation_ids_flat[_image_rows(parent, image)], dtype=np.int64)
        pairs = (ids[:, None] * t + np.arange(t)).reshape(-1)
        chosen = [np.sort(rng.choice(pairs, size=max(2, pairs.size // (3 + k)), replace=False)) for k in range(n_classes)]
        per_class.append(chosen)
        encoded.append(np.concatenate([(k * n_global + c // t) * t + c % t for k, c in enumerate(chosen)]))
    return per_class, encoded


def test_adaptive_pass2_keeps_each_class_own_parents():
    parent = _parent()
    per_class, encoded = _class_samples(parent, np.random.default_rng(3), 2)
    kwargs = dict(oversampling_order=1, random_perturbation=0.0, dtype=np.float32)
    joint = build_local_adaptive_pass2_hypothesis_layout(expand_local_layout_classes(parent, 2), encoded, ORDER, **kwargs)
    assert joint.n_classes == 2
    for k in range(2):
        alone = build_local_adaptive_pass2_hypothesis_layout(parent, [c[k] for c in per_class], ORDER, **kwargs)
        for image in range(N_IMAGES):
            rows = np.arange(_image_rows(joint, image).start, _image_rows(joint, image).stop)
            rows = rows[joint.row_class_flat[rows] == k]
            ref = _image_rows(alone, image)
            for name in ("rotation_ids_flat", "rotations_flat", "rotation_log_priors_flat", "rotation_posterior_ids_flat"):
                assert_matches(np.asarray(getattr(joint, name))[rows], np.asarray(getattr(alone, name))[ref])
            assert_matches(
                np.unpackbits(joint.sample_mask_bits[rows], axis=1, bitorder="little"),
                np.unpackbits(alone.sample_mask_bits[ref], axis=1, bitorder="little"),
            )
    # Image-major, then class-major.
    for image in range(N_IMAGES):
        assert np.all(np.diff(joint.row_class_flat[_image_rows(joint, image)]) >= 0)


def test_single_class_pass2_layout_is_unchanged_by_the_class_decode():
    parent = _parent()
    per_class, _ = _class_samples(parent, np.random.default_rng(5), 1)
    layout = build_local_adaptive_pass2_hypothesis_layout(
        parent, [c[0] for c in per_class], ORDER, oversampling_order=1, random_perturbation=0.0, dtype=np.float32
    )
    assert layout.n_classes == 1 and layout.row_class_flat is None


def test_resident_tables_bin_the_posterior_per_class_and_carry_the_row_class():
    parent = _parent()
    expanded = expand_local_layout_classes(parent, 2)
    tables = tables_from_local_layout(expanded)
    n_global = int(parent.n_global_rotations)
    assert tables.n_classes == 2 and tables.n_posterior_bins == 2 * n_global
    assert_matches(
        tables.row_posterior_id.astype(np.int64),
        expanded.row_class_flat.astype(np.int64) * n_global + np.asarray(expanded.rotation_ids_flat, dtype=np.int64),
    )
    assert tables_from_local_layout(parent).row_class is None
    rows = int(tables.row_offsets[2])
    chunk = CapacityChunk(
        image_start=0, image_stop=2, row_start=0, row_stop=rows, row_capacity=rows + 3, image_capacity=4
    )
    host = materialize_local_chunk(tables, chunk)
    assert_matches(host["row_class"][:rows], expanded.row_class_flat[:rows])
    assert not np.any(host["row_class"][rows:])
