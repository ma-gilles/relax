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

from relax.fine_pass.resident_statistics import resolve_statistics_config
from relax.refinement import optics_shapes, shape_class_scoring
from relax.relion import optics_scale

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

    from relax.fourier.resolution import ImageGeometry
    from relax.refinement.image_size_plans import RunOptics, plan_expectation_windows

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
    several = shape_class_scoring.class_kwargs({"cs_for_engine": 50}, reference, 0)["cs_for_engine"]
    assert single == several
    assert _noise_shells(single) == _noise_shells(several)


CELL9_STAR_PIXEL = 1.6375  # EMPIAR-10076 Class3D K4 (cell 9): STAR optics pixel
CELL9_HEADER_PIXEL = float(np.float32(1.6374975))  # its reference header, 1.63749754...


def _single_shape_windows(current_size, *, model_pixel, star_pixel, box=256, optics=True):
    import logging

    from relax.fourier.resolution import ImageGeometry
    from relax.refinement.image_size_plans import RunOptics, plan_expectation_windows

    return plan_expectation_windows(
        current_size,
        RunOptics(
            image_geometry=ImageGeometry(image_shape=(box, box), pixel_size_angstrom=star_pixel),
            model_pixel_size=model_pixel,
            optics_image_sizes=np.array([box]) if optics else None,
            optics_pixel_sizes=np.array([star_pixel]) if optics else None,
            multi_shape_halves=False,
        ),
        log=logging.getLogger("test"),
    )


def test_class3d_half_sets_carry_the_input_star_optics(monkeypatch):
    """K>1 remaps its current size like K=1, so its half sets carry the input STAR's optics geometry."""
    import pandas as pd

    from relax.refinement import particle_loading
    from relax.relion import input_particle_table

    monkeypatch.setattr(input_particle_table, "prepare_class3d_particle_layout", lambda *a, **k: "layout")
    star = {
        "optics": pd.DataFrame({"rlnOpticsGroup": [1], "rlnImageSize": [256], "rlnImagePixelSize": [CELL9_STAR_PIXEL]}),
        "particles": pd.DataFrame({"rlnOpticsGroup": [1, 1]}),
    }
    half_sets = particle_loading.split_half_sets(
        star, SimpleNamespace(n_units=2), halfset_path=None, n_classes=4, seed=29, init_relion_iteration=0,
        fresh_auto_refine_order=False, noise_order_needed=False, tomographic=False,
    )
    assert half_sets.layout == "layout"
    assert half_sets.optics_image_sizes.tolist() == [256]
    assert half_sets.optics_pixel_sizes.tolist() == [CELL9_STAR_PIXEL]


@pytest.mark.parametrize("current_size", [34, 36, 80, 254])
def test_a_non_exact_header_pixel_widens_a_single_shape_window_by_two(current_size):
    """Cell 9: s = 1.6375 / 1.63749754 = 1 + 1.5e-6, so 2 CEIL(0.5 s cs) = cs + 2 for every even cs below the box
    (RELION it1: 36 against model size 34)."""
    windows = _single_shape_windows(current_size, model_pixel=CELL9_HEADER_PIXEL, star_pixel=CELL9_STAR_PIXEL)
    assert windows.model_size == current_size
    assert windows.image_current_size == current_size + 2


@pytest.mark.parametrize("pixel", [4.25, 2.125, 1.0])
def test_an_exact_pixel_plans_the_same_windows_with_and_without_optics(pixel):
    """Synthetic fixtures store an exact pixel, so the K>1 optics geometry leaves every window as it was."""
    for current_size in range(2, 257, 2):
        with_optics = _single_shape_windows(current_size, model_pixel=float(np.float32(pixel)), star_pixel=pixel)
        without = _single_shape_windows(current_size, model_pixel=pixel, star_pixel=pixel, optics=False)
        assert with_optics == without


def test_the_trial_grid_uses_the_model_pixel_and_stored_offsets_the_star_pixels():
    """relax#57: RELION's trial offsets are Angstrom from the model pixel (ml_optimiser.cpp:590, 597), divided by the
    class's STAR pixel (getTranslationsInPixel); stored offsets convert between STAR pixels (#52)."""
    reference, diamond = _classes(HEADER_PIXEL)
    grid = np.array([[1.0, -2.0]])
    stored = np.array([[0.5, 1.5], [-3.0, 2.0]])  # one stored offset per image of the half
    for shape_class in (reference, diamond):
        out = shape_class_scoring.class_kwargs(
            {"current_translations": grid, "trans_prior_center": stored, "translation_step": 1.0}, shape_class, 2
        )
        assert out["current_translations"] == pytest.approx(grid * HEADER_PIXEL / shape_class.pixel_size, rel=1e-15)
        assert out["translation_step"] == pytest.approx(HEADER_PIXEL / shape_class.pixel_size, rel=1e-15)
        expected = stored[shape_class.image_indices] * STAR_PIXEL / shape_class.pixel_size
        assert out["trans_prior_center"] == pytest.approx(expected, rel=1e-15)
    exact, _ = _classes(None)
    assert exact.trial_grid_factor() == exact.translation_factor == 1.0


@pytest.fixture
def magnified(monkeypatch):
    """Make the one-shape test halves anisotropically magnified (the engines' reference-sphere-clip predicate)."""
    from relax.refinement import dense_half

    monkeypatch.setattr(dense_half, "dataset_magnification_is_anisotropic", lambda dataset: True)


@pytest.fixture
def unmagnified(monkeypatch):
    from relax.refinement import dense_half

    monkeypatch.setattr(dense_half, "dataset_magnification_is_anisotropic", lambda dataset: False)


def _local_grid(model_pixel, model_support_size=38):
    from relax.local_search.sampling import LocalSampling
    from relax.refinement import dense_half

    scale = optics_scale.scale_difference(128, STAR_PIXEL, 128, model_pixel)
    sampling = LocalSampling(
        search=None, rotations=None, translations=None, base_translations=None, perturbation=0.0, angular_step_deg=None,
        image_window_size=optics_scale.group_current_size(38, 128, scale), coarse_image_window_size=None,
        model_support_size=model_support_size,
    )
    dataset = SimpleNamespace(image_shape=(128, 128))
    return dense_half.single_shape_reconstruction_grid(dataset, sampling, shape_class_scoring.OpticsSpec.single_shape(scale))


def _dense_grid(model_pixel, model_support_size=28):
    from relax.refinement import dense_half

    scale = optics_scale.scale_difference(128, STAR_PIXEL, 128, model_pixel)
    sampling = dense_half.DenseSamplingSpec(
        effective_rotations=None, current_translations=None, base_translations=None, current_healpix_order=2,
        oversampling_order=1, translation_step=1.0, random_perturbation=0.0,
        image_window_size=optics_scale.group_current_size(28, 128, scale), coarse_engine="auto", symmetry="C1",
        model_support_size=model_support_size,
    )
    dataset = SimpleNamespace(image_shape=(128, 128))
    return dense_half.single_shape_reconstruction_grid(dataset, sampling, shape_class_scoring.OpticsSpec.single_shape(scale))


def test_a_magnified_single_shape_local_half_reconstructs_on_the_remapped_window(magnified):
    """relax#69: one magnified 1.40 A group against the header pixel scores at 40 for model size 38. Its M-step
    window is the rounded shell of 40 (653 pixels) with the reference kept at 38, as for the same group among
    several shapes; the rounded shell of 38 (597 pixels) is not a window the Wavg rectangle of the 40 crop takes."""
    from relax.fine_pass.wavg import _make_relion_wavg_rectangle
    from relax.fourier.fourier_window import make_fourier_window_indices_np

    sampling, optics = _local_grid(HEADER_PIXEL)
    assert (sampling.image_window_size, sampling.model_support_size, optics.reference_current_size) == (40, 40, 38)
    reference, _ = _classes(HEADER_PIXEL)
    several = shape_class_scoring.class_kwargs({"cs_for_engine": 38}, reference, 0)
    assert (several["model_current_size_for_engine"], several["reference_current_size"]) == (40, 38)
    assert shape_class_scoring.reconstruction_image_radius(optics.reference_current_size, optics.projection_scale) == 19.0

    window, count = make_fourier_window_indices_np((128, 128), sampling.model_support_size, include_dc=True)
    assert int(count) == 653
    _make_relion_wavg_rectangle((128, 128), 40, window, reconstruction_current_size=sampling.model_support_size)
    unremapped, count = make_fourier_window_indices_np((128, 128), 38, include_dc=True)
    assert int(count) == 597
    with pytest.raises(ValueError, match="got 597 pixels"):
        _make_relion_wavg_rectangle((128, 128), 40, unremapped, reconstruction_current_size=38)


def test_a_magnified_single_shape_dense_half_reconstructs_on_the_remapped_window(magnified):
    """The dense path (global searches, Class3D) takes the same grid: cell C group 1 at Class3D's model size 28
    scores at 30, and its rounded shell of 28 (330 pixels) is refused on the 30 crop as the local one of 38 is."""
    from relax.fine_pass.wavg import _make_relion_wavg_rectangle
    from relax.fourier.fourier_window import make_fourier_window_indices_np

    sampling, optics = _dense_grid(HEADER_PIXEL)
    assert (sampling.image_window_size, sampling.model_support_size, optics.reference_current_size) == (30, 30, 28)
    window, count = make_fourier_window_indices_np((128, 128), 30, include_dc=True)
    assert int(count) == 372
    _make_relion_wavg_rectangle((128, 128), 30, window, reconstruction_current_size=30)
    unremapped, count = make_fourier_window_indices_np((128, 128), 28, include_dc=True)
    assert int(count) == 330
    with pytest.raises(ValueError, match="got 330 pixels"):
        _make_relion_wavg_rectangle((128, 128), 30, unremapped, reconstruction_current_size=28)


def test_an_unmagnified_single_shape_half_keeps_the_exact_window_of_the_model_size(unmagnified):
    """Without magnification a rounding scale changes nothing on either entry: the engine gets the model's size
    (it builds the exact-radius window, 565 pixels at 38 and 307 at 28, which the rectangle of the remapped crop
    takes) and no reference clip."""
    from relax.fine_pass.wavg import _make_relion_wavg_rectangle
    from relax.fourier.fourier_window import make_fourier_window_indices_np

    for grid, model_size, image_size, pixels in ((_local_grid, 38, 40, 565), (_dense_grid, 28, 30, 307)):
        sampling, optics = grid(HEADER_PIXEL)
        assert (sampling.image_window_size, sampling.model_support_size) == (image_size, model_size)
        assert optics.reference_current_size is None
        exact, count = make_fourier_window_indices_np((128, 128), model_size, include_dc=True, exact_radius=True)
        assert int(count) == pixels
        _make_relion_wavg_rectangle((128, 128), image_size, exact, reconstruction_current_size=model_size)


@pytest.mark.parametrize("fixture", ["magnified", "unmagnified"])
def test_a_single_shape_half_on_the_model_grid_keeps_its_window(fixture, request):
    """Scale exactly 1 (an exact header pixel), or no explicit model support: nothing is remapped."""
    request.getfixturevalue(fixture)
    for grid, model_size in ((_local_grid, 38), (_dense_grid, 28)):
        for model_support_size in (model_size, None):
            sampling, optics = grid(STAR_PIXEL, model_support_size)
            assert (sampling.image_window_size, sampling.model_support_size) == (model_size, model_support_size)
            assert optics.reference_current_size is None
        sampling, optics = grid(HEADER_PIXEL, None)
        assert sampling.model_support_size is None and optics.reference_current_size is None
