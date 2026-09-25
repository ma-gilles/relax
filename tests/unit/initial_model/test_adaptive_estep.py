"""The adaptive-route InitialModel E-step keeps VDAM's trial geometry and priors."""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax import sampling
from relax.helpers.types import make_relion_stats
from relax.vdam import adaptive_estep, native_options, native_sampling

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("order", [0, 1, 2])
def test_recovar_order_permutation_selects_the_same_rotations(order):
    relion = sampling._get_relion_rotation_grid_eulers_float64(order, rotation_index_order="relion")
    recovar = sampling._get_relion_rotation_grid_eulers_float64(order, rotation_index_order="recovar")
    key = adaptive_estep.relion_order_of_recovar_rotations(order)
    assert sorted(key.tolist()) == list(range(relion.shape[0]))
    assert_matches(relion[key], recovar)


@pytest.mark.parametrize("order,oversampling", [(1, 0), (1, 1), (2, 1)])
@pytest.mark.parametrize("perturbation", [0.0, -0.375])
def test_route_translations_match_the_vdam_sampling_plan(order, oversampling, perturbation):
    opts = native_options.NativeInitialModelOptions(
        fn_img="particles.star",
        healpix_order=order,
        oversampling=oversampling,
        random_perturbation=perturbation,
    )
    plan = native_sampling._build_sampling_plan(opts, iteration=3, defer_fine_rotations=True)
    route = adaptive_estep.adaptive_route_grids(
        healpix_order=order,
        oversampling_order=oversampling,
        random_perturbation=perturbation,
        coarse_base_translations=plan.coarse_base_translations,
        translation_step=plan.offset_step_px,
    )
    assert_matches(route.grids.coarse_translations, plan.coarse_translations)
    assert_matches(route.grids.fine_translations.astype(np.float32), plan.translations)
    n_rot = sampling.rotation_grid_size(order)
    assert route.grids.fine_rotations.shape[0] == n_rot * 8**oversampling
    assert route.fine_source_eulers.shape == (route.grids.fine_rotations.shape[0], 3)


def test_rotation_priors_move_to_recovar_order():
    key = adaptive_estep.relion_order_of_recovar_rotations(1)
    prior_relion = np.arange(2 * key.size, dtype=np.float32).reshape(2, key.size)
    prior = adaptive_estep._recovar_order_prior(prior_relion, key)
    assert_matches(prior[:, 5], prior_relion[:, key[5]])
    with pytest.raises(ValueError):
        adaptive_estep._recovar_order_prior(prior_relion[:, :-1], key)


@pytest.mark.parametrize("fine", [False, True])
def test_direction_sums_bin_recovar_order_rotations(fine):
    order = 0
    n_psi = sampling.rotation_grid_n_in_planes(order)
    n_rot = sampling.rotation_grid_size(order)
    n_dir = n_rot // n_psi
    rng = np.random.default_rng(0)
    coarse = rng.random(n_rot)
    parent = np.repeat(np.arange(n_rot), 8)
    sums = np.repeat(coarse / 8.0, 8) if fine else coarse
    stats = make_relion_stats(
        log_evidence_per_image=np.zeros(1),
        best_log_score_per_image=np.zeros(1),
        max_posterior_per_image=np.ones(1),
        rotation_posterior_sums=sums,
        host_arrays=True,
    )
    result = SimpleNamespace(per_class_stats=(stats,))
    result._replace = lambda **kw: SimpleNamespace(**{**result.__dict__, **kw})
    binned = adaptive_estep._direction_posterior_stats(result, n_coarse_rot=n_rot, rot_parent_map=parent, n_psi=n_psi)
    expected = coarse.reshape(n_psi, n_dir).sum(axis=0)
    assert_matches(np.asarray(binned.per_class_stats[0].rotation_posterior_sums), expected, rtol=1e-13)
