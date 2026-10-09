"""Optics groups remap their current size against RELION's model pixel size, the reference header's (relax#56).

RELION's model pixel size is the reference map's header value (ml_model.cpp:944-962): a 1.4 A map reads as
float32 1.399999976. Each optics group's image current size is
``2 CEIL(0.5 (box_g pix_g) / (ori model_pix) current_size)`` (ml_optimiser.cpp:6917-6923), so a 1.40 A group
scores at 52 for current size 50, and its weighted sums take direct residuals through shell 26 and the image
power from shell 27 (acc_ml_optimiser_impl.h:4940-4951).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from relax.helpers import optics_scale
from relax.refinement import optics_shapes
from relax.sparse_pass2.resident_statistics import resolve_statistics_config

pytestmark = pytest.mark.unit

STAR_PIXEL = 1.4
HEADER_PIXEL = float(np.float32(1.4))  # 1.399999976...
DIAMOND = 1.36


def _classes(model_pixel):
    pairs = [
        (SimpleNamespace(image_shape=(128, 128), voxel_size=pixel), np.arange(i, i + 1))
        for i, pixel in enumerate((STAR_PIXEL, DIAMOND))
    ]
    return optics_shapes.make_shape_classes(pairs, model_box_size=128, ref_pixel=STAR_PIXEL, model_pixel=model_pixel)


def _noise_shells(current_size):
    config = resolve_statistics_config(
        n_shells=65, n_fine_trans=1, n_images=1, n_coarse_rot=1, n_scale_groups=1, current_size=current_size
    )
    # (last direct-residual shell, first image-power shell)
    return config.direct_noise_exclusive_shell_stop - 1, config.direct_noise_exclusive_shell_stop


def test_reference_group_remaps_against_the_header_pixel():
    reference, diamond = _classes(HEADER_PIXEL)
    sizes = [optics_scale.group_current_size(50, c.box_size, c.scale) for c in (reference, diamond)]
    assert sizes == [52, 50]
    assert _noise_shells(sizes[0]) == (26, 27)
    assert _noise_shells(sizes[1]) == (25, 26)


def test_translations_stay_on_the_star_pixels():
    reference, diamond = _classes(HEADER_PIXEL)
    assert (reference.translation_factor, diamond.translation_factor) == (1.0, STAR_PIXEL / DIAMOND)


def test_an_exact_header_pixel_keeps_the_reference_group_at_the_model_size():
    reference, _ = _classes(None)
    assert reference.scale == 1.0
    assert optics_scale.group_current_size(50, reference.box_size, reference.scale) == 50


def test_halves_scale_against_the_dataset_model_pixel():
    datasets = [
        SimpleNamespace(
            image_shape=(128, 128),
            volume_shape=(128,) * 3,
            voxel_size=pixel,
            n_units=2,
            subset=lambda rows, pixel=pixel: SimpleNamespace(image_shape=(128, 128), voxel_size=pixel),
        )
        for pixel in (STAR_PIXEL, DIAMOND)
    ]
    dataset = optics_shapes.MultiShapeDataset(datasets, [np.array([0, 2]), np.array([1, 3])])
    plain = dataset.subset(np.arange(4)).classes[0].scale
    header = dataset.with_model_pixel_size(HEADER_PIXEL).subset(np.arange(4)).classes[0].scale
    assert plain == 1.0
    assert header == pytest.approx(STAR_PIXEL / HEADER_PIXEL, rel=1e-15) and header > 1.0


@pytest.mark.parametrize("model_pixel", [HEADER_PIXEL, STAR_PIXEL])
def test_single_and_several_shape_paths_give_a_group_the_same_window_and_noise_split(model_pixel):
    """A 1.40 A group is scored, and its noise shells split, the same whether it is the dataset's only shape or the
    reference-grid class of several: the single-shape window (plan_expectation_windows) and the class window
    (class_kwargs) agree, so their per-group sigma2 sums cover the same pixels (relax#56: 52 vs 50 split the
    shells 26/27 vs 25/26 on EMPIAR-10299)."""
    import logging

    from relax.helpers.resolution import ImageGeometry
    from relax.refinement.iteration_planning import RunOptics, plan_expectation_windows

    single = plan_expectation_windows(
        50,
        RunOptics(
            image_geometry=ImageGeometry(image_shape=(128, 128), pixel_size_angstrom=STAR_PIXEL),
            model_pixel_size=model_pixel,
            optics_image_sizes=np.array([128]),
            optics_pixel_sizes=np.array([STAR_PIXEL]),
            multi_shape_halves=False,
        ),
        log=logging.getLogger("test"),
    ).image_current_size
    reference, _ = _classes(model_pixel)
    several = optics_shapes.class_kwargs({"cs_for_engine": 50}, reference, 0)["cs_for_engine"]
    assert single == several
    assert _noise_shells(single) == _noise_shells(several)
