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


class _Dataset:
    n_images = 4

    def subset(self, image_indices):
        n_images = int(np.asarray(image_indices).size)
        return SimpleNamespace(n_images=n_images, n_units=n_images)


class _ReplaceableNamespace(SimpleNamespace):
    def _replace(self, **updates):
        return type(self)(**{**vars(self), **updates})


def test_pseudo_halfsets_are_one_pass_with_accumulator_slots(monkeypatch):
    """VDAM K=2: one adaptive pass over the subset; each particle's half is its slot group.

    RELION backprojects particle p of class k into BPref[k + (part_id % 2) * K]
    (acc_ml_optimiser_impl.h:4800-4804); the subset schedule's halves go straight
    to the resident engine as ``reconstruction_group_ids``.
    """

    from relax.vdam.estep_common import DenseInitialModelEstepConfig
    from relax.vdam.init import initialise_denovo_state

    calls = []
    opts = native_options.NativeInitialModelOptions(fn_img="particles.star", healpix_order=1, oversampling=1)
    plan = native_sampling._build_sampling_plan(opts, iteration=2, defer_fine_rotations=True)
    key = adaptive_estep.relion_order_of_recovar_rotations(1)
    prior_relion = np.arange(2 * key.size, dtype=np.float32).reshape(2, key.size)

    def fake_route(dataset, *args, **kwargs):
        calls.append(dict(kwargs, n_images=int(dataset.n_images)))
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(dataset.n_images),
            best_log_score_per_image=np.zeros(dataset.n_images),
            max_posterior_per_image=np.ones(dataset.n_images),
            rotation_posterior_sums=np.ones(key.size),
            host_arrays=True,
        )
        return _ReplaceableNamespace(
            Ft_y=[np.zeros(2), np.ones(2)], Ft_ctf=[np.zeros(2), np.ones(2)], per_class_stats=(stats, stats)
        )

    accumulator_calls = []
    monkeypatch.setattr(adaptive_estep, "run_dense_k_class_em_adaptive", fake_route)
    monkeypatch.setattr(adaptive_estep, "uses_relion_cuda_image_preprocessing", lambda dataset: True)
    monkeypatch.setattr(
        adaptive_estep, "_arrays_to_accumulators", lambda *a, **kw: accumulator_calls.append(kw) or []
    )
    monkeypatch.setattr(adaptive_estep, "_sparse_pass2_estep_meta", lambda results, selected: {})
    monkeypatch.setattr(adaptive_estep, "_add_accumulator_weight_meta", lambda meta, acc, K: None)
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=2, nr_iter=4, n_directions=4, pseudo_halfsets=True)
    config = DenseInitialModelEstepConfig(
        noise_variance=np.ones(64, dtype=np.float32),
        rotations=None,
        translations=plan.translations,
        relion_bpref_frame=True,
        engine_kwargs={},
    )
    engine_kwargs = {
        "healpix_order": 1,
        "oversampling_order": 1,
        "random_perturbation": plan.random_perturbation,
        "coarse_translations": plan.coarse_translations,
        "coarse_base_translations": plan.coarse_base_translations,
        "translation_step": plan.offset_step_px,
        "max_significants": 200,
        "class_rotation_log_prior": prior_relion,
        "current_size": 8,
        "reconstruction_subtract_projected_reference": True,
        "score_with_masked_images": True,
        "half_spectrum_scoring": True,
        "projection_padding_factor": 1,
        "reconstruction_padding_factor": 1,
        "projection_mask_current_image_disk": False,
    }
    result = adaptive_estep.run_adaptive_initial_model_estep(
        _Dataset(),
        state,
        config,
        class_log_priors=np.zeros(2),
        joint_particle_ids=np.array([3, 0, 2], dtype=np.int64),
        joint_halfset_ids=np.array([1, 0, 1], dtype=np.int32),
        means=None,
        mean_variance=None,
        relion_projector_half_by_class=np.zeros((2, 1)),
        relion_projector_r_max=1,
        engine_kwargs=engine_kwargs,
    )
    assert len(calls) == 1 and calls[0]["n_images"] == 3
    np.testing.assert_array_equal(calls[0]["reconstruction_group_ids"], [1, 0, 1])
    assert calls[0]["reconstruction_group_count"] == 2
    assert calls[0]["max_significants"] == 200
    assert calls[0]["mstep_subtract_ctf_projection"] is True
    assert_matches(calls[0]["class_rotation_log_prior"], prior_relion[:, key])
    assert accumulator_calls[0]["halfset_idx"] is None and accumulator_calls[0]["reconstruction_group_count"] == 2
    assert result.meta["halfset_ids"] == (0, 1)
