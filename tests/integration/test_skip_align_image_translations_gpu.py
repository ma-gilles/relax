"""RELION --skip_align on the resident pass: an image translated as it is prepared scores as that translation does.

``--skip_align`` gives each particle one translation, its own offset remainder; relax scores the
one zero translation of images that pass 2 translated as it prepared them
(``image_translations``, :func:`relax.classification.given_poses.given_pose_grids`). The same
pass with the remainders as an ordinary translation grid and each image's one sample on its own
translation must give the same classes, maps and noise sums.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _k2_inputs(monkeypatch):
    """The K=2 adaptive E-step fixture of tests/integration/test_pass1_deferred_preprocess_check_gpu.py."""
    from helpers.em_arrays import _hermitian_volume
    from helpers.sparse_pass2_mock import VOLUME_SHAPE

    from relax.relion import relion_ctf

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "unit"))
    from integration.test_dense_gemm_coarse_engine_gpu import _install_native_preprocessing, _variable_ctf_rows
    from unit.test_resident_pass2_driver import _driver_fixture_args
    from unit.test_resident_relion_reference import _ppref, _tie_free_noise

    monkeypatch.setenv("RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    args = _driver_fixture_args(seed=20260929)
    dataset = args.pop("experiment_dataset")
    _install_native_preprocessing(dataset)
    monkeypatch.setattr(relion_ctf, "_relion_exact_ctf_half_from_source_star", _variable_ctf_rows)
    for name in (
        "volume",
        "noise_variance",
        "significant_sample_indices",
        "normalization_other_score_log_z",
        "normalization_score_mode",
        "relion_x_half_mstep",
        "return_score_log_z",
        "preserve_bpref_particle_order",
        "fine_rotations_override",
        "fine_rotation_parent_override",
        "fine_translations_override",
        "fine_translation_parent_override",
        "translations",
        "translation_log_prior",
        "rotation_log_prior",
        "nside_level",
        "oversampling_order",
    ):
        args.pop(name)
    noise = jnp.asarray(_tie_free_noise(6, 200.0).reshape(-1), jnp.float32)
    volumes = jnp.stack([_hermitian_volume(VOLUME_SHAPE, seed=17 + i) for i in range(2)])
    slabs = [_ppref(volumes[i]) for i in range(2)]
    args.update(
        score_with_masked_images=True,
        relion_projector_half=np.stack([slab for slab, _ in slabs]),
        relion_projector_r_max=slabs[0][1],
        mstep_relion_x_half=True,
    )
    return dataset, volumes, noise, args


def _given_pose_pass(dataset, volumes, noise, args, grids, supports, translations, **given):
    from relax.classification import k_class

    keywords = dict(args)
    disc_type = keywords.pop("disc_type")
    return k_class.run_dense_k_class_em_adaptive(
        dataset,
        volumes,
        None,
        noise,
        grids.rotations,
        translations,
        grids.rotations,
        translations,
        grids.rotation_parent_map,
        np.arange(translations.shape[0]),
        disc_type,
        class_log_priors=np.log(np.array([0.4, 0.6])),
        accumulate_noise=keywords.pop("accumulate_noise"),
        coarse_healpix_order=grids.healpix_order,
        oversampling_order=0,
        fine_current_size=keywords["current_size"],
        given_supports=supports,
        **given,
        **keywords,
    )


@pytest.mark.gpu
def test_image_translations_score_as_their_translation_samples(monkeypatch):
    assert jax.default_backend() == "gpu"
    from helpers.float_compare import assert_trees_match
    from scipy.spatial.transform import Rotation

    from relax.classification.given_poses import given_pose_grids

    dataset, volumes, noise, args = _k2_inputs(monkeypatch)
    n = int(dataset.n_units)
    rng = np.random.default_rng(5)
    # float32 values, so both passes see the same remainders.
    remainders = rng.uniform(-0.5, 0.5, size=(n, 2)).astype(np.float32).astype(np.float64)
    remainders[0] = 0.0  # one image without a remainder
    rotations = Rotation.random(n, random_state=6).as_matrix().astype(np.float32)
    grids = given_pose_grids(rotations, remainders)
    # The prior centers move by the remainder, so both passes sum the same |offset - center|^2.
    centers = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)

    # Each remainder an ordinary translation sample: image i's one sample is (rotation i, translation i).
    searched = _given_pose_pass(
        dataset,
        volumes,
        noise,
        dict(args, translation_prior_centers=centers),
        grids,
        [np.asarray([i * n + i], dtype=np.int32) for i in range(n)],
        remainders.astype(np.float32),
    )
    # The remainders applied to the images as they are prepared, on the one zero translation.
    given = _given_pose_pass(
        dataset,
        volumes,
        noise,
        dict(args, translation_prior_centers=(centers - remainders).astype(np.float32)),
        grids,
        grids.supports,
        grids.translations.astype(np.float32),
        given_image_translations=grids.image_translations,
    )

    # The host phase (float64, rounded once) and the kernel's float32 sincosf of the same translation differ by
    # float32 rounding: measured 1.8e-7 at most (wsum_norm_correction; Ft_y 1.2e-7, job 15054173). The float32
    # band, also for the sums the engine keeps in float64.
    rtol = 1e-6
    for name in ("class_responsibilities", "class_posterior_sums", "Ft_y", "Ft_ctf"):
        assert_trees_match(
            np.asarray(getattr(given, name)), np.asarray(getattr(searched, name)), rtol=rtol, err_msg=name
        )
    assert_trees_match(given.stats._asdict(), searched.stats._asdict(), rtol=rtol, err_msg="stats")
    assert given.aggregate_noise_stats is not None and searched.aggregate_noise_stats is not None
    assert_trees_match(
        given.aggregate_noise_stats._asdict(), searched.aggregate_noise_stats._asdict(), rtol=rtol, err_msg="noise"
    )
    assert np.array_equal(np.asarray(given.class_assignments), np.asarray(searched.class_assignments))
    # The rotation is image i's either way; the translation id is the zero one against image i's own.
    n_trans = remainders.shape[0]
    assert np.array_equal(
        np.asarray(given.per_class_hard_assignments), np.asarray(searched.per_class_hard_assignments) // n_trans
    )
