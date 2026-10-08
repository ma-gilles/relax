"""Final local preparation preserves deferred grids and parent window sizing."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from relax.refinement import local_sampling

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("pixel_size", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_particle_spacing_rejected_even_with_model_override(pixel_size):
    from relax.refinement.iteration_loop import refine_single_volume
    from relax.refinement.refinement_options import RefinementOptions, RelionParityOptions

    half = SimpleNamespace(voxel_size=pixel_size, image_shape=(32, 32), volume_shape=(32, 32, 32))
    options = RefinementOptions(parity=RelionParityOptions(relion_model_pixel_size=1.5))
    with pytest.raises(ValueError, match="Particle pixel size must be finite and positive"):
        refine_single_volume([half, half], None, None, None, None, options=options)


def _optics(optics_pixel_sizes=None, optics_image_sizes=None, model_pixel_size=1.5):
    from relax.helpers.resolution import ImageGeometry
    from relax.refinement.iteration_planning import RunOptics

    return RunOptics(
        image_geometry=ImageGeometry(image_shape=(128, 128), pixel_size_angstrom=1.5), model_pixel_size=model_pixel_size,
        optics_image_sizes=optics_image_sizes, optics_pixel_sizes=optics_pixel_sizes, multi_shape_halves=False,
    )


@pytest.fixture
def preparation(monkeypatch):
    calls = []
    monkeypatch.setattr(local_sampling.sampling, "relion_angular_sampling_deg", lambda *a, **kw: 3.0)
    monkeypatch.setattr(local_sampling, "healpix_angular_step", lambda order: 6.0)

    def size_parent(**kwargs):
        calls.append(kwargs)
        return 32

    monkeypatch.setattr(local_sampling, "relion_local_pass1_current_size", size_parent)
    monkeypatch.setattr(local_sampling, "_precompute_exact_local_fine_grid_enabled", lambda *a, **kw: False)
    inputs = dict(
        search=local_sampling.LocalSearchSettings(healpix_order=2, oversampling_order=0, sigma_rot=0.1, sigma_psi=0.2),
        optics=_optics(),
        translations=object(), base_translations=object(), image_window_size=64,
        particle_diameter_angstrom=100.0,
        perturbation=None, rotation_dtype=object(),
    )
    return inputs, calls


@pytest.mark.parametrize("oversampling", [0, 1])
def test_deferred_grid_sizes_parent_only_when_expanded(preparation, oversampling):
    inputs, calls = preparation
    inputs["search"] = replace(inputs["search"], healpix_order=2 + oversampling, oversampling_order=oversampling)
    result = local_sampling.prepare_final_local_sampling(**inputs)
    assert result.rotations is None
    assert result.mstep_rotations is None
    assert result.search.healpix_order == 2 + oversampling
    assert result.coarse_image_window_size == (32 if oversampling else 64)
    assert len(calls) == oversampling
    assert (result.coarse_angular_step_deg is None) == (oversampling == 0)
    assert result.translations is inputs["translations"]
    assert result.search is inputs["search"]


@pytest.mark.parametrize("perturbation", [None, 0.0, 0.125])
def test_eager_grid_uses_explicit_dtype(preparation, monkeypatch, perturbation):
    inputs, calls = preparation
    inputs["perturbation"] = perturbation
    fine, mstep = object(), object()
    observed = []
    monkeypatch.setattr(local_sampling, "_precompute_exact_local_fine_grid_enabled", lambda *a, **kw: True)

    def exact_grid(**kwargs):
        observed.append(kwargs)
        return fine, object(), mstep

    monkeypatch.setattr(local_sampling.sampling, "_exact_local_fine_grid", exact_grid)
    result = local_sampling.prepare_final_local_sampling(**inputs)
    assert result.rotations is fine
    assert result.mstep_rotations is mstep
    assert observed[0]["dtype"] is inputs["rotation_dtype"]
    assert observed[0]["random_perturbation"] is perturbation
    assert result.coarse_image_window_size == 64
    assert not calls


def test_parent_expansion_never_materializes_exhaustive_grid(preparation, monkeypatch):
    inputs, calls = preparation
    inputs["search"] = replace(inputs["search"], healpix_order=3, oversampling_order=1)

    def unexpected_precompute(*args, **kwargs):
        raise AssertionError("Parent expansion should generate particle neighborhoods")

    monkeypatch.setattr(local_sampling, "_precompute_exact_local_fine_grid_enabled", unexpected_precompute)
    result = local_sampling.prepare_final_local_sampling(**inputs)
    assert result.rotations is None
    assert len(calls) == 1


@pytest.mark.parametrize("parent_order", [6, 7])
def test_final_sampling_uses_resolved_parent_order(preparation, parent_order):
    # Final STAR metadata may advance hp6 to hp7; oversampling must follow it.
    inputs, calls = preparation
    inputs["search"] = replace(inputs["search"], healpix_order=parent_order + 1, oversampling_order=1)
    result = local_sampling.prepare_final_local_sampling(**inputs)
    assert result.search.healpix_order == parent_order + 1
    assert calls[0]["pre_update_healpix_order"] == parent_order


@pytest.mark.parametrize(
    ("optics_pixel_sizes", "optics_image_sizes", "model_pixel_size", "expected"),
    [(None, None, 1.2, (1.5, 128)), ([1.4999999], [128], 1.5, (1.4999999, 128))],
)
def test_final_and_numbered_pass1_windows_use_the_first_optics_groups_image_geometry(
    preparation, optics_pixel_sizes, optics_image_sizes, model_pixel_size, expected
):
    # RELION sizes pass 1 from remap_sizes * ori_size * mymodel.pixel_size = the optics group's box times its
    # STAR pixel size (ml_optimiser.cpp:6941), in the numbered and the final iteration alike. The final pass
    # used the image geometry instead of the optics group's STAR values, and the numbered fallback without
    # optics sizes used the model pixel size.
    inputs, calls = preparation
    optics = _optics(optics_pixel_sizes, optics_image_sizes, model_pixel_size)
    assert optics.first_optics_group_geometry() == expected
    inputs["optics"] = optics
    inputs["search"] = replace(inputs["search"], healpix_order=3, oversampling_order=1)
    local_sampling.prepare_final_local_sampling(**inputs)
    assert (calls[-1]["pixel_size"], calls[-1]["box_size"]) == expected
