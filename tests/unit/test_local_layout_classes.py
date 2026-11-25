"""Class3D local searches: one local row set per image, repeated per class (expand_local_layout_classes), each
class's rows weighted by its own direction prior when one is given (with_class_rotation_log_priors)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.fine_pass.resident_candidates import CapacityChunk
from relax.local_search.layout import (
    build_local_adaptive_pass2_hypothesis_layout,
    build_local_hypothesis_layout,
    drop_local_layout_classes,
    expand_local_layout_classes,
    restrict_local_layout_classes,
    with_class_rotation_log_priors,
)
from relax.local_search.resident_layout import materialize_local_chunk, tables_from_local_layout
from relax.runtime.engine_tuning import EngineTuning
from relax.sampling.grids import build_local_search_grid_metadata

pytestmark = pytest.mark.unit

ORDER = 1
N_IMAGES = 5
TRANSLATIONS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)


def _parent(rotation_log_prior=None):
    rng = np.random.default_rng(7)
    eulers = np.column_stack(
        [rng.uniform(0, 360, N_IMAGES), rng.uniform(20, 160, N_IMAGES), rng.uniform(0, 360, N_IMAGES)]
    )
    return build_local_hypothesis_layout(
        eulers, None, 0.35, 0.35, ORDER, TRANSLATIONS, np.zeros((N_IMAGES, 2), dtype=np.float32), 3.0, None, 1.0,
        grid_metadata=build_local_search_grid_metadata(ORDER), translation_prior_reference_translations=TRANSLATIONS,
        rotation_log_prior=rotation_log_prior,
        dtype=np.float32,
        sigma_tilt_deg=0.0,
        engine_tuning=EngineTuning(),
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


def _class_tables(n_classes: int) -> np.ndarray:
    """A different log prior on every rotation of the parent grid for each class."""
    n_global = int(_parent().n_global_rotations)
    return np.log(np.random.default_rng(11).dirichlet(np.ones(n_global), size=n_classes)).astype(np.float32)


def test_one_table_per_class_all_equal_is_the_k1_prior():
    """Every class given the K=1 table: the rows K=1 builds with that table, repeated per class."""
    table = _class_tables(1)[0]
    weighted = with_class_rotation_log_priors(expand_local_layout_classes(_parent(), 3), np.stack([table] * 3))
    reference = expand_local_layout_classes(_parent(table), 3)
    assert_matches(np.asarray(weighted.rotation_log_priors_flat), np.asarray(reference.rotation_log_priors_flat))
    assert_matches(np.asarray(weighted.rotation_ids_flat), np.asarray(reference.rotation_ids_flat))


def test_each_class_rows_take_their_class_prior_through_restriction_and_drop():
    tables = _class_tables(3)
    weighted = with_class_rotation_log_priors(expand_local_layout_classes(_parent(), 3), tables)
    ids = np.asarray(weighted.rotation_ids_flat, dtype=np.int64)
    row_class = np.asarray(weighted.row_class_flat, dtype=np.int64)
    assert_matches(np.asarray(weighted.rotation_log_priors_flat), tables[row_class, ids])
    assert np.asarray(weighted.rotation_log_priors_flat).dtype == np.float32
    seeds = np.arange(N_IMAGES) % 3
    for layout in (restrict_local_layout_classes(weighted, seeds), drop_local_layout_classes(weighted, [1])):
        ids = np.asarray(layout.rotation_ids_flat, dtype=np.int64)
        row_class = np.asarray(layout.row_class_flat, dtype=np.int64)
        assert_matches(np.asarray(layout.rotation_log_priors_flat), tables[row_class, ids])


def test_adaptive_pass2_children_inherit_their_class_parent_prior():
    """Pass 2 from per-class weighted parents: each class's children carry what K=1 children of a parent layout
    built with that class's table carry."""
    tables = _class_tables(2)
    per_class, encoded = _class_samples(_parent(), np.random.default_rng(3), 2)
    kwargs = dict(oversampling_order=1, random_perturbation=0.0, dtype=np.float32)
    weighted = with_class_rotation_log_priors(expand_local_layout_classes(_parent(), 2), tables)
    joint = build_local_adaptive_pass2_hypothesis_layout(weighted, encoded, ORDER, **kwargs)
    for k in range(2):
        alone = build_local_adaptive_pass2_hypothesis_layout(_parent(tables[k]), [c[k] for c in per_class], ORDER, **kwargs)
        for image in range(N_IMAGES):
            rows = np.arange(_image_rows(joint, image).start, _image_rows(joint, image).stop)
            rows = rows[joint.row_class_flat[rows] == k]
            assert_matches(
                np.asarray(joint.rotation_log_priors_flat)[rows],
                np.asarray(alone.rotation_log_priors_flat)[_image_rows(alone, image)],
            )


def test_class_priors_need_class_expanded_parent_rows_and_one_table_per_class():
    expanded = expand_local_layout_classes(_parent(), 2)
    with pytest.raises(ValueError, match="only a class-expanded layout"):
        with_class_rotation_log_priors(_parent(), _class_tables(1))
    with pytest.raises(ValueError, match="must be"):
        with_class_rotation_log_priors(expanded, _class_tables(3))
    with pytest.raises(ValueError, match="oversampled child rows"):
        with_class_rotation_log_priors(replace(expanded, oversampled_rows=True), _class_tables(2))



@pytest.mark.parametrize("weighted", [False, True])
def test_a_class3d_local_iteration_weights_each_class_rows_by_its_prior(monkeypatch, weighted):
    """The local iteration expands its own layout over the classes and, given one direction prior per class, hands
    the engine each class's rows weighted by it; without one, RELION's Gaussian log priors."""
    from types import SimpleNamespace

    from helpers.refinement_specs import local_iteration_owners
    from scipy.spatial.transform import Rotation

    from relax.refinement import local_half
    from relax.types import LocalEMResult

    seen = {}

    def run_local(data, layout, *args, **kwargs):
        seen["layout"] = layout
        return LocalEMResult(
            Ft_y=np.zeros(8, dtype=np.complex64),
            Ft_ctf=np.ones(8, dtype=np.float32),
            hard_assignments=np.zeros(layout.rotation_counts.size, dtype=np.int32),
            stats=object(),
        )

    monkeypatch.setattr(local_half, "compute_local_search_resident", run_local)
    order, n_images = 2, 3
    prior = Rotation.random(n_images, random_state=7).as_matrix()
    translations = np.zeros((n_images, 2), dtype=np.float32)
    n_global = 12 * 4**order * int(build_local_search_grid_metadata(order)["n_psi"])
    tables = np.log(np.random.default_rng(2).dirichlet(np.ones(n_global), size=2)).astype(np.float32)
    local_half._run_local_search_iteration(*local_iteration_owners(
        SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 2), voxel_size=1.0, particles_file=None),
        None, None, prior, None,
        healpix_order=order, sigma_rot=0.2, sigma_psi=0.2,
        translations=translations[:1], prior_translations=translations,
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp",
        current_size=2,
        n_classes=2,
        class_direction_log_prior=tables if weighted else None,
    ))
    layout = seen["layout"]
    assert layout.n_classes == 2 and layout.rotation_counts.sum() > 0
    ids = np.asarray(layout.rotation_ids_flat, dtype=np.int64)
    row_class = np.asarray(layout.row_class_flat, dtype=np.int64)
    gaussian = expand_local_layout_classes(
        build_local_hypothesis_layout(
            prior, None, 0.2, 0.2, order, translations[:1], translations, 1.0, None, 1.0,
            grid_metadata=build_local_search_grid_metadata(order), sigma_tilt_deg=0.0, engine_tuning=EngineTuning(),
        ),
        2,
    )
    expected = tables[row_class, ids] if weighted else np.asarray(gaussian.rotation_log_priors_flat)
    assert_matches(np.asarray(layout.rotation_log_priors_flat), expected)
