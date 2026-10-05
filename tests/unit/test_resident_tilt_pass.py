"""The resident pass with subtomogram units (S4.2) in its one-image case equals the SPA pass (GPU).

A particle with one image, an identity left matrix and the SPA phases is an SPA image: RELION's GPU path
then keeps the SPA matrices (isIdentity, acc_ml_optimiser_impl.h:1100-1104) and divides nothing by the
image count. So ``_resident_pass2(tilt=...)``, which scores and backprojects through the tilt chunk runner,
must reproduce ``_resident_pass2`` on the SPA driver fixture: the discrete state exactly, the scores in the
float band, and the maps and sums inside the resident driver's own repeat band.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_resident_pass2_driver import (
    _driver_fixture_args,
    _resident_production_env,  # noqa: F401 (fixture)
    requires_resident_gpu,
)

pytestmark = pytest.mark.unit

# The float32 BPref atomics' repeat band for the SPA-vs-tilt accumulator comparisons (Ft_y, Ft_ctf).
# The SPA pass repeated alone reaches relL2 1.0e-7 (local A100 probe, 2026-10-02: 3 SPA + 3 tilt runs,
# SPA vs SPA 6.1e-8-1.0e-7, SPA vs tilt 3.5e-8-7.4e-8); bindw's job 14869734 measured Ft_ctf up to
# 7.4e-8 at main; medium 14864305 failed at 1.21e-7 (Ft_y) and ppcaspeed's run at 1.03e-7. Twice the
# repeat band, user decision of 2026-10-02.
_BPREF_ATOMIC_BAND = 2e-7


def _one_image_tilt_inputs(args):
    from relax.sparse_pass2.resident_tilts import TiltPassInputs
    from relax.sparse_pass2.sparse_pass2_bucket_io import _relion_cuda_score_translation_angles_if_available
    from relax.sparse_pass2.sparse_pass2_window import _fine_translation_prior_2d

    n_images = args["experiment_dataset"].n_units
    fine_translations = np.asarray(args["fine_translations_override"])
    angles = np.asarray(
        _relion_cuda_score_translation_angles_if_available(
            fine_translations, args["experiment_dataset"].image_shape, enabled=True
        ),
        dtype=np.float32,
    )
    prior = np.asarray(
        _fine_translation_prior_2d(
            args["translation_log_prior"],
            np.asarray(args["fine_translation_parent_override"]),
            n_images=n_images,
            n_fine_trans=fine_translations.shape[0],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    return TiltPassInputs(
        unit_image_offsets=np.arange(n_images + 1, dtype=np.int64),
        image_left=np.tile(np.eye(3), (n_images, 1, 1)),
        image_angles=np.broadcast_to(angles, (n_images,) + angles.shape).copy(),
        image_noise_scale=np.ones(n_images, dtype=np.float32),
        unit_translation_prior=np.broadcast_to(prior, (n_images, fine_translations.shape[0])).copy()
        if prior.ndim == 1
        else prior,
        unit_translation_sqdist_ang=None,
        unit_optics_groups=None,
        fine_source_eulers=None,
        fine_rotations=np.asarray(args["fine_rotations_override"]),
        slot_capacity=1,
    )


@requires_resident_gpu
def test_one_image_particles_reproduce_the_spa_pass(_resident_production_env):  # noqa: F811
    from relax.sparse_pass2 import resident_pass2 as rp

    # The once-per-half resident operands, which the tilt runner needs, cover masked scoring only.
    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    spa = rp._resident_pass2(**args)
    # The tilt pass takes the offset prior per particle, from the tilt inputs.
    tomo = rp._resident_pass2(**dict(args, translation_log_prior=None), tilt=_one_image_tilt_inputs(args))
    _assert_tilt_pass_matches_spa(spa, tomo)


@requires_resident_gpu
def test_tilt_chunks_prepare_their_own_operands_when_the_half_does_not_fit(_resident_production_env, monkeypatch):  # noqa: F811
    """A half whose resident operands exceed the budget: each tilt chunk prepares its own images' operands."""

    from relax.sparse_pass2 import resident_pass2 as rp

    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    spa = rp._resident_pass2(**args)
    monkeypatch.setattr(rp, "_resident_operands_fit", lambda *a, **k: False)
    tomo = rp._resident_pass2(**dict(args, translation_log_prior=None), tilt=_one_image_tilt_inputs(args))
    _assert_tilt_pass_matches_spa(spa, tomo)


def _assert_tilt_pass_matches_spa(spa, tomo):
    for field in ("hard_assignment", "best_fine_rotation_indices"):
        assert_matches(getattr(spa.finalized, field), getattr(tomo.finalized, field), err_msg=field)
    for field in (
        "log_evidence_per_image",
        "best_log_score_per_image",
        "max_posterior_per_image",
        "rotation_posterior_sums",
    ):
        assert_matches(
            np.asarray(getattr(spa.finalized, field)), np.asarray(getattr(tomo.finalized, field)), err_msg=field
        )

    def rel_l2(a, b):
        a, b = np.asarray(a), np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    # The float32 BPref atomics' repeat band (_BPREF_ATOMIC_BAND).
    assert rel_l2(spa.Ft_y[0], tomo.Ft_y[0]) < _BPREF_ATOMIC_BAND
    assert rel_l2(spa.Ft_ctf[0], tomo.Ft_ctf[0]) < _BPREF_ATOMIC_BAND
    for field in (
        "wsum_sigma2_noise",
        "wsum_img_power",
        "wsum_norm_correction",
        "wsum_scale_correction_xa",
        "wsum_scale_correction_aa",
    ):
        assert rel_l2(getattr(spa.noise_stats, field), getattr(tomo.noise_stats, field)) < 1e-7, field
    assert abs(float(spa.noise_stats.sumw) - float(tomo.noise_stats.sumw)) <= 1e-6 * abs(float(spa.noise_stats.sumw))


# ---------------------------------------------------------------------------
# K>1 classes (RELION subtomogram Class3D)
# ---------------------------------------------------------------------------


@requires_resident_gpu
def test_one_image_particles_reproduce_the_spa_k_class_pass(_resident_production_env, monkeypatch):  # noqa: F811
    """Class3D through the tilt runner, one image per particle, is the SPA K-class pass.

    Both score every class in the particle's one segment and backproject each class into its own
    BPref. The SPA pass backprojects every row here (no per-projection sums), as the tilt runner does.
    """

    from test_resident_k_class_pass2 import _k_class_args, _resident

    from relax.sparse_pass2 import resident_pass2 as rp

    monkeypatch.setattr(rp, "_PRESUM_ADJOINT_FREE_FRACTION", 0.0)
    args, volumes, supports, priors = _k_class_args(2)
    args = dict(args, score_with_masked_images=True)
    spa = _resident(args, volumes, supports, priors)
    tomo = _resident(
        dict(args, translation_log_prior=None, tilt=_one_image_tilt_inputs(args)), volumes, supports, priors
    )

    assert_matches(spa.per_class_hard_assignments, tomo.per_class_hard_assignments)
    for field in (
        "class_log_evidence_per_image",
        "class_best_log_score_per_image",
        "class_rotation_posterior_sums",
        "class_reconstruction_posterior_sums",
    ):
        assert_matches(np.asarray(getattr(spa, field)), np.asarray(getattr(tomo, field)), err_msg=field)
    for field in ("log_evidence_per_image", "best_log_score_per_image", "max_posterior_per_image"):
        assert_matches(np.asarray(getattr(spa.stats, field)), np.asarray(getattr(tomo.stats, field)), err_msg=field)
    for k in range(2):
        assert _rel_l2(spa.Ft_y[k], tomo.Ft_y[k]) < _BPREF_ATOMIC_BAND, f"Ft_y class {k}"
        assert _rel_l2(spa.Ft_ctf[k], tomo.Ft_ctf[k]) < _BPREF_ATOMIC_BAND, f"Ft_ctf class {k}"
    _assert_noise_stats_match(spa.noise_stats, tomo.noise_stats)


def _two_image_particles(args):
    """The driver fixture's 12 images as 6 particles of 2 images (identity left matrices)."""

    one_image = _one_image_tilt_inputs(args)
    n_images = args["experiment_dataset"].n_units
    n_units = n_images // 2
    return one_image._replace(
        unit_image_offsets=np.arange(0, n_images + 1, 2, dtype=np.int64),
        image_noise_scale=np.full(n_images, 0.5, dtype=np.float32),
        unit_translation_prior=one_image.unit_translation_prior[::2].copy(),
        slot_capacity=2,
    ), n_units


@requires_resident_gpu
def test_duplicated_class_of_tilt_particles_is_the_k1_pass(_resident_production_env):  # noqa: F811
    """Two copies of one class at prior 1/2 each: every particle's K=1 posterior split in half.

    With two images per particle, each class's BPref is half the K=1 BPref, its mass half of
    ``sumw``, its evidence ``log 1/2`` below the K=1 evidence, and the per-image noise, norm and
    scale sums (added over the classes' M-step passes) are the K=1 ones.
    """

    from test_resident_k_class_pass2 import _resident

    from relax.sparse_pass2 import resident_pass2 as rp

    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    tilt, n_units = _two_image_particles(args)
    support = args.pop("significant_sample_indices")[:n_units]
    args["translation_log_prior"] = None
    single = rp.compute_tilt_pass2_stats_resident(tilt=tilt, significant_sample_indices=support, **args)

    k_args = dict(args, tilt=tilt)
    prior = k_args.pop("rotation_log_prior")
    volume = k_args.pop("volume")
    for name in ("normalization_other_score_log_z", "normalization_score_mode"):
        k_args.pop(name)
    half = (prior + np.float32(np.log(0.5))).astype(np.float32)
    doubled = _resident(k_args, jnp.stack([volume, volume]), [support, support], [half, half])

    for k in range(2):
        assert_matches(doubled.per_class_best_pose_rotation_ids[k], single.best_rotation_indices)
        # log 1/2 is folded into the float32 row prior: compare at float32 (test_resident_k_class_pass2).
        assert_matches(
            np.float32(doubled.class_log_evidence_per_image[k]),
            np.float32(np.asarray(single.relion_stats.log_evidence_per_image, dtype=np.float64) + np.log(0.5)),
        )
        assert _rel_l2(0.5 * np.asarray(single.Ft_y), doubled.Ft_y[k]) < 1e-6, f"Ft_y class {k}"
        assert _rel_l2(0.5 * np.asarray(single.Ft_ctf), doubled.Ft_ctf[k]) < 1e-6, f"Ft_ctf class {k}"
    assert_matches(
        np.float32(doubled.class_reconstruction_posterior_sums),
        np.float32(np.full(2, 0.5 * float(single.noise_stats.sumw))),
    )
    # The SPA duplicated-class bounds (test_resident_k_class_pass2.test_duplicated_class_is_the_k1_pass): the Wavg
    # residual cancels most of its magnitude, so its reduction order shows at 1e-4; the other sums at 1e-6
    # (measured on 2aeae7f, job 14696158: at most 8.5e-8).
    for field, bound in (
        ("wsum_sigma2_noise", 1e-4),
        ("wsum_img_power", 1e-6),
        ("wsum_norm_correction", 1e-6),
        ("wsum_scale_correction_xa", 1e-6),
        ("wsum_scale_correction_aa", 1e-6),
    ):
        measured = _rel_l2(getattr(single.noise_stats, field), getattr(doubled.noise_stats, field))
        print(f"duplicated tilt class {field} rel L2 {measured:.3e}")
        assert measured < bound, field
    assert abs(float(single.noise_stats.sumw) - float(doubled.noise_stats.sumw)) <= 1e-6 * float(
        single.noise_stats.sumw
    )


def _rel_l2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    den = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / den) if den else float(np.linalg.norm(b))


def _assert_noise_stats_match(want, got):
    for field in (
        "wsum_sigma2_noise",
        "wsum_img_power",
        "wsum_norm_correction",
        "wsum_scale_correction_xa",
        "wsum_scale_correction_aa",
    ):
        measured = _rel_l2(getattr(want, field), getattr(got, field))
        print(f"{field} rel L2 {measured:.3e}")
        # The resident driver's repeat band, as for K=1 (_assert_tilt_pass_matches_spa); measured at most
        # 3.5e-9 on 2aeae7f (job 14696158).
        assert measured < 1e-7, field
    assert abs(float(want.sumw) - float(got.sumw)) <= 1e-6 * abs(float(want.sumw))


# ---------------------------------------------------------------------------
# InitialModel (VDAM): residual backprojection into pseudo-halfset slots
# ---------------------------------------------------------------------------


def _vdam_options(n_units):
    """VDAM's E-step options: residual BPref, each particle's pseudo-halfset (RELION's part_id % 2)."""

    return dict(
        mstep_subtract_ctf_projection=True,
        reconstruction_group_ids=(np.arange(n_units) % 2).astype(np.int32),
        reconstruction_group_count=2,
    )


def _assert_accumulators_match(spa_Ft_y, spa_Ft_ctf, tomo_Ft_y, tomo_Ft_ctf):
    assert len(spa_Ft_y) == len(tomo_Ft_y) == 2
    for slot in range(2):
        # A residual BPref cancels most of the image sum, so its float32 atomic repeat spread is wider than the
        # plain BPref's (_BPREF_ATOMIC_BAND): on a local H100 (2026-09-30) a same-code repeat moved one slot by
        # 2.9e-8 and the slot-blocked pass differed by 3.8e-8 and 1.1e-7 in two runs; the CTF weights keep
        # the plain band.
        assert _rel_l2(spa_Ft_y[slot], tomo_Ft_y[slot]) < 3e-7, f"Ft_y slot {slot}"
        assert _rel_l2(spa_Ft_ctf[slot], tomo_Ft_ctf[slot]) < _BPREF_ATOMIC_BAND, f"Ft_ctf slot {slot}"


@requires_resident_gpu
def test_one_image_particles_reproduce_the_spa_vdam_pass(_resident_production_env, monkeypatch):  # noqa: F811
    """VDAM's E-step through the tilt runner, one image per particle, is the SPA VDAM pass.

    Each image backprojects its residual into its pseudo-halfset's BPref slot; the SPA pass backprojects
    every row here (no per-projection sums), as the tilt runner does. The residual must change the map.
    """

    from relax.sparse_pass2 import resident_pass2 as rp

    monkeypatch.setattr(rp, "_PRESUM_ADJOINT_FREE_FRACTION", 0.0)
    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    n_images = args["experiment_dataset"].n_units
    vdam = dict(args, **_vdam_options(n_images))
    spa = rp._resident_pass2(**vdam)
    tomo = rp._resident_pass2(**dict(vdam, translation_log_prior=None), tilt=_one_image_tilt_inputs(args))
    for field in ("hard_assignment", "best_fine_rotation_indices"):
        assert_matches(getattr(spa.finalized, field), getattr(tomo.finalized, field), err_msg=field)
    _assert_accumulators_match(spa.Ft_y, spa.Ft_ctf, tomo.Ft_y, tomo.Ft_ctf)
    _assert_noise_stats_match(spa.noise_stats, tomo.noise_stats)

    plain = rp._resident_pass2(
        **dict(args, translation_log_prior=None, reconstruction_group_ids=None, reconstruction_group_count=None),
        tilt=_one_image_tilt_inputs(args),
    )
    # The two slots partition the particles; the residual moves Ft_y and leaves the CTF weights.
    assert _rel_l2(plain.Ft_ctf[0], np.asarray(tomo.Ft_ctf[0]) + np.asarray(tomo.Ft_ctf[1])) < 1e-6
    assert _rel_l2(plain.Ft_y[0], np.asarray(tomo.Ft_y[0]) + np.asarray(tomo.Ft_y[1])) > 1e-3


@requires_resident_gpu
def test_one_image_particles_reproduce_the_spa_k_class_vdam_pass(_resident_production_env, monkeypatch):  # noqa: F811
    """K=2 VDAM through the tilt runner, one image per particle, is the SPA K-class VDAM pass (slots class + K * half)."""

    from test_resident_k_class_pass2 import _k_class_args, _resident

    from relax.sparse_pass2 import resident_pass2 as rp

    monkeypatch.setattr(rp, "_PRESUM_ADJOINT_FREE_FRACTION", 0.0)
    args, volumes, supports, priors = _k_class_args(2)
    args = dict(args, score_with_masked_images=True)
    vdam = dict(args, **_vdam_options(args["experiment_dataset"].n_units))
    spa = _resident(vdam, volumes, supports, priors)
    tomo = _resident(
        dict(vdam, translation_log_prior=None, tilt=_one_image_tilt_inputs(args)), volumes, supports, priors
    )
    assert_matches(spa.per_class_hard_assignments, tomo.per_class_hard_assignments)
    for k in range(2):
        _assert_accumulators_match(spa.Ft_y[k], spa.Ft_ctf[k], tomo.Ft_y[k], tomo.Ft_ctf[k])
    _assert_noise_stats_match(spa.noise_stats, tomo.noise_stats)


@requires_resident_gpu
def test_slot_blocked_tilt_projections_are_the_one_block_pass(_resident_production_env, monkeypatch):  # noqa: F811
    """A chunk projected one image slot at a time (its projections do not fit at once) is the one-block pass.

    The running diff2 keeps its slot order across blocks and each accumulator its slot order, so the
    discrete state is exact, the scores equal and the maps and sums inside the repeat band. K=2 with
    VDAM's pseudo-halfsets exercises every accumulator slot.
    """

    from test_resident_k_class_pass2 import _k_class_args, _resident

    from relax.sparse_pass2 import resident_pass2 as rp
    from relax.sparse_pass2 import resident_tilts

    monkeypatch.setattr(rp, "_PRESUM_ADJOINT_FREE_FRACTION", 0.0)
    args, volumes, supports, priors = _k_class_args(2)
    args = dict(args, score_with_masked_images=True)
    tilt, n_units = _two_image_particles(args)
    supports = [class_supports[:n_units] for class_supports in supports]
    vdam = dict(args, translation_log_prior=None, tilt=tilt, **_vdam_options(n_units))
    whole = _resident(vdam, volumes, supports, priors)
    monkeypatch.setattr(resident_tilts, "tilt_projection_slot_block", lambda *a, **k: 1)
    blocked = _resident(vdam, volumes, supports, priors)

    assert_matches(whole.per_class_hard_assignments, blocked.per_class_hard_assignments)
    for field in ("class_log_evidence_per_image", "class_best_log_score_per_image", "class_rotation_posterior_sums"):
        assert_matches(np.asarray(getattr(whole, field)), np.asarray(getattr(blocked, field)), err_msg=field)
    for k in range(2):
        _assert_accumulators_match(whole.Ft_y[k], whole.Ft_ctf[k], blocked.Ft_y[k], blocked.Ft_ctf[k])
    _assert_noise_stats_match(whole.noise_stats, blocked.noise_stats)


@requires_resident_gpu
def test_each_particles_own_translations_are_the_whole_grid_mstep(_resident_production_env, monkeypatch):  # noqa: F811
    """The M-step at each particle's translations with mass (unit_mstep_translations) is the whole-grid M-step.

    The translations it drops carry exactly zero posterior, so only the order of the image-power
    reduction over translations differs: the discrete state is exact, the scores equal and the maps and
    sums inside the repeat band. The whole grid multiplies a non-finite translated tile by a zero
    posterior into NaN where the sparse M-step drops it, so both runs must be finite (non-finite images
    stop the run at preprocessing, relax#16). K=2 with VDAM's pseudo-halfsets exercises every
    accumulator slot.
    """

    from test_resident_k_class_pass2 import _k_class_args, _resident

    from relax.sparse_pass2 import resident_tilts

    args, volumes, supports, priors = _k_class_args(2)
    args = dict(args, score_with_masked_images=True)
    tilt, n_units = _two_image_particles(args)
    supports = [class_supports[:n_units] for class_supports in supports]
    vdam = dict(args, translation_log_prior=None, tilt=tilt, **_vdam_options(n_units))
    calls = []
    sparse_tables = resident_tilts.unit_mstep_translations

    def recorded(*a, **k):
        out = sparse_tables(*a, **k)
        calls.append(out)
        return out

    monkeypatch.setattr(resident_tilts, "unit_mstep_translations", recorded)
    sparse = _resident(vdam, volumes, supports, priors)
    n_fine = np.asarray(args["fine_translations_override"]).shape[0]
    assert calls
    print(f"particle translation tables {sorted({out.index.shape[1] for out in calls})} of {n_fine}")

    def whole_grid(row_posterior, row_unit, *, unit_capacity, n_fine_trans, minimum=4):
        index = np.tile(np.arange(int(n_fine_trans), dtype=np.int64), (int(unit_capacity), 1))
        return resident_tilts.UnitMstepTranslations(index=index, valid=np.ones_like(index, dtype=bool))

    monkeypatch.setattr(resident_tilts, "unit_mstep_translations", whole_grid)
    dense = _resident(vdam, volumes, supports, priors)

    assert_matches(dense.per_class_hard_assignments, sparse.per_class_hard_assignments)
    for field in ("class_log_evidence_per_image", "class_best_log_score_per_image", "class_rotation_posterior_sums"):
        assert_matches(np.asarray(getattr(dense, field)), np.asarray(getattr(sparse, field)), err_msg=field)
    for k in range(2):
        for run in (dense, sparse):
            assert np.all(np.isfinite(np.asarray(run.Ft_y[k]))) and np.all(np.isfinite(np.asarray(run.Ft_ctf[k])))
        _assert_accumulators_match(dense.Ft_y[k], dense.Ft_ctf[k], sparse.Ft_y[k], sparse.Ft_ctf[k])
    _assert_noise_stats_match(dense.noise_stats, sparse.noise_stats)


# ---------------------------------------------------------------------------
# The --firstiter_cc iteration (normalized CC, winner takes all)
# ---------------------------------------------------------------------------


@requires_resident_gpu
def test_one_image_particles_reproduce_the_spa_firstiter_cc_pass(_resident_production_env):  # noqa: F811
    """The CC iteration through the tilt runner, one image per particle, is the SPA CC pass.

    The tilt runner translates each image's resident CC operands per slot (RELION's score translation) where
    the SPA pass builds its translated CC tiles per chunk; both score with RELION's fine CC reduction and keep
    each unit's best cell, then backproject it with the Gaussian M-step.
    """

    from relax.sparse_pass2 import resident_pass2 as rp

    args = dict(
        _driver_fixture_args(),
        score_with_masked_images=True,
        relion_firstiter_score_mode="normalized_cc",
        relion_firstiter_winner_take_all=True,
        relion_exact_fine_normalized_cc=True,
    )
    spa = rp._resident_pass2(**args)
    tomo = rp._resident_pass2(**dict(args, translation_log_prior=None), tilt=_one_image_tilt_inputs(args))
    for field in ("hard_assignment", "best_fine_rotation_indices"):
        assert_matches(getattr(spa.finalized, field), getattr(tomo.finalized, field), err_msg=field)
    # The CC scores are float32 (the evidence offset is added in float64): compare them at float32.
    for field in ("best_log_score_per_image", "max_posterior_per_image"):
        assert_matches(
            np.float32(getattr(spa.finalized, field)), np.float32(getattr(tomo.finalized, field)), err_msg=field
        )
    assert _rel_l2(spa.Ft_y[0], tomo.Ft_y[0]) < _BPREF_ATOMIC_BAND
    assert _rel_l2(spa.Ft_ctf[0], tomo.Ft_ctf[0]) < _BPREF_ATOMIC_BAND
