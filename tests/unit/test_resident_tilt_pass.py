"""The resident pass with subtomogram units (S4.2) in its one-image case equals the SPA pass (GPU).

A particle with one image, an identity left matrix and the SPA phases is an SPA image: RELION's GPU path
then keeps the SPA matrices (isIdentity, acc_ml_optimiser_impl.h:1100-1104) and divides nothing by the
image count. So ``_resident_pass2(
    tilt=..., engine_tuning=RESIDENT_PRODUCTION_ENGINE,
    scoring_overrides=ScoringOverrides(),
)``, which scores and backprojects through the tilt chunk runner,
must reproduce ``_resident_pass2`` on the SPA driver fixture: the discrete state exactly, the scores in the
float band, and the maps and sums inside the resident driver's own repeat band.
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.engine_settings import RESIDENT_PRODUCTION_ENGINE
from helpers.float_compare import RTOL, assert_matches
from test_resident_pass2_driver import (
    _driver_fixture_args,
    requires_resident_gpu,
)

from relax.runtime.scoring_overrides import ScoringOverrides

pytestmark = pytest.mark.unit

# The float32 BPref atomics' repeat band for the SPA-vs-tilt accumulator comparisons (Ft_y, Ft_ctf).
# The SPA pass repeated alone reaches relL2 1.0e-7 (local A100 probe, 2026-10-02: 3 SPA + 3 tilt runs,
# SPA vs SPA 6.1e-8-1.0e-7, SPA vs tilt 3.5e-8-7.4e-8); bindw's job 14869734 measured Ft_ctf up to
# 7.4e-8 at main; medium 14864305 failed at 1.21e-7 (Ft_y) and ppcaspeed's run at 1.03e-7. Twice the
# repeat band, user decision of 2026-10-02.
_BPREF_ATOMIC_BAND = 2e-7


def _one_image_tilt_inputs(args):
    from relax.fine_pass.bucket_io import relion_cuda_score_translation_angles_if_available
    from relax.fine_pass.resident_tilts import TiltPassInputs
    from relax.fine_pass.window import fine_translation_log_prior_2d

    n_images = args["experiment_dataset"].n_units
    fine_translations = np.asarray(args["fine_translations_override"])
    angles = np.asarray(
        relion_cuda_score_translation_angles_if_available(
            fine_translations, args["experiment_dataset"].image_shape, enabled=True
        ),
        dtype=np.float32,
    )
    prior = np.asarray(
        fine_translation_log_prior_2d(
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
def test_one_image_particles_reproduce_the_spa_pass():
    from relax.fine_pass import resident_pass2 as rp

    # The once-per-half resident operands, which the tilt runner needs, cover masked scoring only.
    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    spa = rp._resident_pass2(**args, engine_tuning=RESIDENT_PRODUCTION_ENGINE, scoring_overrides=ScoringOverrides())
    # The tilt pass takes the offset prior per particle, from the tilt inputs.
    tomo = rp._resident_pass2(**dict(args, translation_log_prior=None), tilt=_one_image_tilt_inputs(
        args), engine_tuning=RESIDENT_PRODUCTION_ENGINE,
        scoring_overrides=ScoringOverrides(),
    )
    _assert_tilt_pass_matches_spa(spa, tomo)


@requires_resident_gpu
def test_tilt_driver_accumulates_through_the_partial_pair(monkeypatch):
    """The tilt M-step goes through the partial pair (relax#65, production since relax#92).

    With one image per particle the tilt pass must still reproduce the SPA pass.
    """

    from relax.fine_pass import resident_pass2 as rp
    from relax.reconstruction import half_volume_mstep as hvm

    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    spa = rp._resident_pass2(**args, engine_tuning=RESIDENT_PRODUCTION_ENGINE, scoring_overrides=ScoringOverrides())
    allocated = []
    real_allocate = rp.allocate_row_partial

    def allocate(*a, **kw):
        allocated.append(real_allocate(*a, **kw))
        return allocated[-1]

    monkeypatch.setattr(rp, "allocate_row_partial", allocate)
    tomo = rp._resident_pass2(**dict(args, translation_log_prior=None), tilt=_one_image_tilt_inputs(
        args), engine_tuning=RESIDENT_PRODUCTION_ENGINE,
        scoring_overrides=ScoringOverrides(),
    )
    assert allocated and all(isinstance(p, hvm.RowPartial) for p in allocated), "the tilt pass must own a partial pair"
    _assert_tilt_pass_matches_spa(spa, tomo)


@requires_resident_gpu
def test_tilt_chunks_prepare_their_own_operands_when_the_half_does_not_fit(monkeypatch):
    """A half whose resident operands exceed the budget: each tilt chunk prepares its own images' operands."""

    from relax.fine_pass import resident_pass2 as rp
    from relax.fine_pass import resident_projections

    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    spa = rp._resident_pass2(**args, engine_tuning=RESIDENT_PRODUCTION_ENGINE, scoring_overrides=ScoringOverrides())
    monkeypatch.setattr(resident_projections, "resident_operands_fit", lambda *a, **k: False)
    tomo = rp._resident_pass2(**dict(args, translation_log_prior=None), tilt=_one_image_tilt_inputs(
        args), engine_tuning=RESIDENT_PRODUCTION_ENGINE,
        scoring_overrides=ScoringOverrides(),
    )
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
def test_one_image_particles_reproduce_the_spa_k_class_pass(monkeypatch):
    """Class3D through the tilt runner, one image per particle, is the SPA K-class pass.

    Both score every class in the particle's one segment and backproject each class into its own
    BPref. The SPA pass backprojects every row here (no per-projection sums), as the tilt runner does.
    """

    from test_resident_k_class_pass2 import _k_class_args, _resident

    from relax.fine_pass import resident_projections

    monkeypatch.setattr(resident_projections, "PRESUM_ADJOINT_FREE_FRACTION", 0.0)
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
def test_duplicated_class_of_tilt_particles_is_the_k1_pass():
    """Two copies of one class at prior 1/2 each: every particle's K=1 posterior split in half.

    With two images per particle, each class's BPref is half the K=1 BPref, its mass half of
    ``sumw``, its evidence ``log 1/2`` below the K=1 evidence, and the per-image noise, norm and
    scale sums (added over the classes' M-step passes) are the K=1 ones.
    """

    from test_resident_k_class_pass2 import _resident

    from relax.fine_pass import resident_pass2 as rp

    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    tilt, n_units = _two_image_particles(args)
    support = args.pop("significant_sample_indices")[:n_units]
    args["translation_log_prior"] = None
    single = rp.compute_tilt_pass2_stats_resident(
        tilt=tilt, significant_sample_indices=support, **args, engine_tuning=RESIDENT_PRODUCTION_ENGINE,
        scoring_overrides=ScoringOverrides(),
    )

    k_args = dict(args, tilt=tilt)
    prior = k_args.pop("rotation_log_prior")
    volume = k_args.pop("volume")
    k_args.pop("normalization_other_score_log_z")
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
def test_one_image_particles_reproduce_the_spa_vdam_pass(monkeypatch):
    """VDAM's E-step through the tilt runner, one image per particle, is the SPA VDAM pass.

    Each image backprojects its residual into its pseudo-halfset's BPref slot; the SPA pass backprojects
    every row here (no per-projection sums), as the tilt runner does. The residual must change the map.
    """

    from relax.fine_pass import resident_pass2 as rp
    from relax.fine_pass import resident_projections

    monkeypatch.setattr(resident_projections, "PRESUM_ADJOINT_FREE_FRACTION", 0.0)
    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    n_images = args["experiment_dataset"].n_units
    vdam = dict(args, **_vdam_options(n_images))
    spa = rp._resident_pass2(**vdam, engine_tuning=RESIDENT_PRODUCTION_ENGINE, scoring_overrides=ScoringOverrides())
    tomo = rp._resident_pass2(**dict(vdam, translation_log_prior=None), tilt=_one_image_tilt_inputs(
        args), engine_tuning=RESIDENT_PRODUCTION_ENGINE,
        scoring_overrides=ScoringOverrides(),
    )
    for field in ("hard_assignment", "best_fine_rotation_indices"):
        assert_matches(getattr(spa.finalized, field), getattr(tomo.finalized, field), err_msg=field)
    _assert_accumulators_match(spa.Ft_y, spa.Ft_ctf, tomo.Ft_y, tomo.Ft_ctf)
    _assert_noise_stats_match(spa.noise_stats, tomo.noise_stats)

    plain = rp._resident_pass2(
        **dict(args, translation_log_prior=None, reconstruction_group_ids=None, reconstruction_group_count=None),
        tilt=_one_image_tilt_inputs(args),
        engine_tuning=RESIDENT_PRODUCTION_ENGINE,
        scoring_overrides=ScoringOverrides(),
    )
    # The two slots partition the particles; the residual moves Ft_y and leaves the CTF weights.
    assert _rel_l2(plain.Ft_ctf[0], np.asarray(tomo.Ft_ctf[0]) + np.asarray(tomo.Ft_ctf[1])) < 1e-6
    assert _rel_l2(plain.Ft_y[0], np.asarray(tomo.Ft_y[0]) + np.asarray(tomo.Ft_y[1])) > 1e-3


@requires_resident_gpu
def test_one_image_particles_reproduce_the_spa_k_class_vdam_pass(monkeypatch):
    """K=2 VDAM through the tilt runner, one image per particle, is the SPA K-class VDAM pass (slots class + K * half)."""

    from test_resident_k_class_pass2 import _k_class_args, _resident

    from relax.fine_pass import resident_projections

    monkeypatch.setattr(resident_projections, "PRESUM_ADJOINT_FREE_FRACTION", 0.0)
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


def test_slot_blocks_split_the_slots_evenly_whatever_the_free_bytes():
    """The slot block is set by the number of blocks the fitting count needs, so nearby budgets give the same
    blocks (and the same M-step programs), each at most the count that fits."""

    from relax.fine_pass.resident_tilts import tilt_projection_slot_block

    def block(fit):
        return tilt_projection_slot_block(10, slot_capacity=41, slot_row_bytes=1, budget_bytes=2 * 10 * fit)

    assert block(41) == block(100) == 41
    assert {block(fit) for fit in (21, 25, 30, 31, 32, 40)} == {21}
    assert {block(fit) for fit in (14, 17, 20)} == {14}
    assert block(1) == 1
    for fit in range(1, 42):
        assert block(fit) <= fit
    assert tilt_projection_slot_block(10, slot_capacity=41, slot_row_bytes=1, budget_bytes=None) == 41


@requires_resident_gpu
def test_slot_blocked_tilt_projections_are_the_one_block_pass(monkeypatch):
    """A chunk projected one image slot at a time (its projections do not fit at once) is the one-block pass.

    The running diff2 keeps its slot order across blocks and each accumulator its slot order, so the
    discrete state is exact, the scores equal and the maps and sums inside the repeat band. K=2 with
    VDAM's pseudo-halfsets exercises every accumulator slot.
    """

    from test_resident_k_class_pass2 import _k_class_args, _resident

    from relax.fine_pass import resident_projections, resident_tilts

    monkeypatch.setattr(resident_projections, "PRESUM_ADJOINT_FREE_FRACTION", 0.0)
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
def test_only_the_overflow_tilt_chunks_take_the_largest_particles_slot_block(monkeypatch):
    """A particle past the row classes splits its own chunk into slot blocks, not every chunk of the pass (relax#115).

    The row class sits between the fixture's smallest and largest particle and a chunk past it fits one slot at a
    time: the row-class chunks keep both slots, the overflow chunks take one, and the pass is the one-block pass
    (exact discrete state, maps and sums inside the repeat band). The overflow chunks also run at unit class 1, their
    one particle's slots, not the regular chunks' class (relax#127).
    """

    from test_resident_k_class_pass2 import _k_class_args, _resident

    from relax.fine_pass import resident_tilts
    from relax.runtime.shape_buckets import pow2_floor

    args, volumes, supports, priors = _k_class_args(2)
    args = dict(args, score_with_masked_images=True)
    tilt, n_units = _two_image_particles(args)
    supports = [class_supports[:n_units] for class_supports in supports]
    vdam = dict(args, translation_log_prior=None, tilt=tilt, **_vdam_options(n_units))
    real_chunk, particle_rows, chunk_blocks = resident_tilts.run_tilt_chunk, [], []

    def chunk(tilt_chunk, *a, **k):
        particle_rows.extend(np.diff(np.asarray(k["tables"].row_offsets)).tolist())
        chunk_blocks.append((int(tilt_chunk.row_capacity), int(k["slot_block"]), int(tilt_chunk.image_capacity)))
        return real_chunk(tilt_chunk, *a, **k)

    monkeypatch.setattr(resident_tilts, "run_tilt_chunk", chunk)
    whole = _resident(vdam, volumes, supports, priors)
    row_class = pow2_floor(max(particle_rows) - 1)
    assert min(particle_rows) <= row_class < max(particle_rows), sorted(set(particle_rows))
    chunk_blocks.clear()

    monkeypatch.setattr(
        resident_tilts, "tilt_capacity_ladders", lambda rows, images, **k: ((row_class,), tuple(sorted(set(images))))
    )
    monkeypatch.setattr(
        resident_tilts,
        "tilt_projection_slot_block",
        lambda rows, *, slot_capacity, slot_row_bytes, budget_bytes: 1 if int(rows) > row_class else int(slot_capacity),
    )
    blocked = _resident(vdam, volumes, supports, priors)

    assert {block for rows, block, _ in chunk_blocks if rows <= row_class} == {2}
    assert {block for rows, block, _ in chunk_blocks if rows > row_class} == {1}
    assert {units for rows, _, units in chunk_blocks if rows > row_class} == {1}
    assert_matches(whole.per_class_hard_assignments, blocked.per_class_hard_assignments)
    for field in ("class_log_evidence_per_image", "class_best_log_score_per_image", "class_rotation_posterior_sums"):
        assert_matches(np.asarray(getattr(whole, field)), np.asarray(getattr(blocked, field)), err_msg=field)
    for k in range(2):
        _assert_accumulators_match(whole.Ft_y[k], whole.Ft_ctf[k], blocked.Ft_y[k], blocked.Ft_ctf[k])
    _assert_noise_stats_match(whole.noise_stats, blocked.noise_stats)


def test_an_overflow_chunk_is_planned_at_its_own_projection_bytes():
    """The plan counts a particle past the row classes at ``overflow_row_bytes`` and the row classes at ``row_bytes``.

    At hn_k2_s0150 iteration 30 a 37k-row particle fits two of its 41 image slots at a time while a 512-row chunk
    fits all 41 (relax#115): planned at one row size, either the overflow chunk is refused or every chunk is.
    """

    from relax.fine_pass import resident_chunk_memory
    from relax.fine_pass.policy import ResidentConfigurationUnsupported

    gib = 1024**3
    slot_bytes = int(126.4 * 1024)
    kwargs = dict(
        row_capacity_ladder=(256, 512),
        image_capacity_ladder=(32, 128),
        mstep_block_rows=256,
        row_bytes=41 * slot_bytes,
        n_fine_trans=41,
        n_recon_pixels=6427,
        budget_bytes=int(40.88 * gib),
        rows_live_during_prepare=True,
        max_image_rows=37344,
    )
    overflow_rows = resident_chunk_memory.overflow_row_capacity(37344, (256, 512))
    assert overflow_rows * 41 * slot_bytes > kwargs["budget_bytes"]
    with pytest.raises(ResidentConfigurationUnsupported, match="37344 rows, run alone"):
        resident_chunk_memory.plan_resident_chunk_memory(**kwargs)
    plan = resident_chunk_memory.plan_resident_chunk_memory(**kwargs, overflow_row_bytes=2 * slot_bytes)
    assert plan.row_capacity_ladder == (256, 512)
    assert plan.peak_bytes >= max(512 * 41, overflow_rows * 2) * slot_bytes
    alone = resident_chunk_memory.resident_chunk_bytes(
        row_capacity=overflow_rows,
        image_capacity=32,
        mstep_block_rows=plan.mstep_block_rows,
        row_bytes=2 * slot_bytes,
        n_fine_trans=41,
        n_recon_pixels=6427,
        rows_live_during_prepare=True,
    )
    assert plan.peak_bytes >= alone


def _lone_tilt_runs(monkeypatch, args, tilt):
    """One pass with the particles' overflow chunks held whole, and one where they do not fit the budget.

    A 16-row ladder puts every particle of the fixture past the largest row class. The second pass has a chunk
    budget between what its largest particle needs in row blocks and what it needs whole,
    so the whole-chunk plan is refused and the row-blocked one taken.
    Returns ``(whole, blocked, row blocks of each of the blocked pass's chunks)``.
    """

    from relax.fine_pass import resident_chunk_memory, resident_tilts
    from relax.fine_pass import resident_pass2 as rp

    monkeypatch.setattr(
        resident_tilts, "tilt_capacity_ladders", lambda rows, images, **k: ((16,), tuple(sorted(set(images))))
    )
    engine_tuning = replace(RESIDENT_PRODUCTION_ENGINE, mstep_block_rows=16)
    plans, lone_rows = [], []
    real_plan, real_chunk = resident_chunk_memory.plan_resident_chunk_memory, resident_tilts.run_tilt_chunk

    def plan(**kwargs):
        plans.append(kwargs)
        return real_plan(**kwargs)

    def chunk(tilt_chunk, *a, **k):
        # Row blocks the chunk runs in (0: held whole).
        lone_rows.append(int(tilt_chunk.row_capacity) // k["lone_block_rows"] if k.get("lone_block_rows") else 0)
        return real_chunk(tilt_chunk, *a, **k)

    def slot_tables(unit_slot_images, **k):
        # A row-blocked particle's M-step holds its one unit (relax#82): its tables' unit axis.
        if int(k["row_capacity"]) == 16:
            assert np.asarray(unit_slot_images).shape[1] == 1
            mstep_units.append(1)
        return real_tables(unit_slot_images, **k)

    mstep_units, real_tables = [], resident_tilts.slot_mstep_tables
    monkeypatch.setattr(resident_chunk_memory, "plan_resident_chunk_memory", plan)
    monkeypatch.setattr(resident_tilts, "run_tilt_chunk", chunk)
    monkeypatch.setattr(resident_tilts, "slot_mstep_tables", slot_tables)
    whole = rp._resident_pass2(**dict(args, translation_log_prior=None), tilt=tilt, engine_tuning=engine_tuning, scoring_overrides=ScoringOverrides())
    assert not mstep_units
    assert lone_rows and not any(lone_rows), "a particle that fits whole must not run in row blocks"
    assert resident_chunk_memory.overflow_row_capacity(plans[-1]["max_image_rows"], (16,)) > 16, "the fixture has no overflow particle"

    # Shrink the budget until the whole-particle plan is refused (the planner first shrinks the slot block and
    # the M-step block); the row-blocked plan must then fit, or the pass itself refuses and the test fails.
    whole_peak = real_plan(**dict(plans[-1], budget_bytes=None)).peak_bytes
    for fraction in (0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05):
        budget = int(whole_peak * fraction)
        monkeypatch.setattr(resident_chunk_memory, "resident_chunk_budget_bytes", lambda budget=budget, **k: budget)
        del lone_rows[:]
        blocked = rp._resident_pass2(**dict(args, translation_log_prior=None), tilt=tilt, engine_tuning=engine_tuning, scoring_overrides=ScoringOverrides())
        if any(lone_rows):
            assert mstep_units, "the row-blocked particles' M-step tables were not built for one unit"
            print(f"row-blocked at a budget of {fraction} of the whole-particle peak ({whole_peak} bytes)")
            return whole, blocked, list(lone_rows)
    raise AssertionError("no budget made the whole-particle plan refuse while the row-blocked one fit")


@requires_resident_gpu
def test_lone_tilt_particle_in_row_blocks_is_the_one_chunk_pass(monkeypatch):
    """A particle whose rows' projections exceed the chunk budget runs alone in row blocks (a real ribosome
    tilt-series refinement refused at its first HEALPix-3 iteration: 294,912 rows, 82.8 GiB against 42.1).

    Each row's diff2 keeps its slot order and the posterior is formed over all of the particle's rows, so the
    discrete state is exact and the scores equal; the maps and sums differ only by the grouping of the M-step
    blocks, inside the SPA lone-chunk tests' band. Two tilt images per particle exercise the slot axis.
    """

    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    tilt, n_units = _two_image_particles(args)
    # The fixture's per-image supports and normalization, one per particle.
    args = dict(
        args,
        significant_sample_indices=args["significant_sample_indices"][:n_units],
        normalization_other_score_log_z=args["normalization_other_score_log_z"][:n_units],
    )
    whole, blocked, lone_rows = _lone_tilt_runs(monkeypatch, args, tilt)
    # A flat-posterior particle is many blocks long (294,912 rows at HEALPix 3, millions at finer orders), not
    # just past one: the fixture's largest runs in several 16-row blocks.
    print("row blocks per chunk:", lone_rows)
    assert max(lone_rows) >= 4, lone_rows
    _assert_tilt_pass_matches_spa(whole, blocked)


def _lean_lone_run(monkeypatch, args, tilt, *, block_rows=None):
    """The blocked pass again with its row-blocked particles in their lean form (relax#118).

    Called after :func:`_lone_tilt_runs`, whose budget stays patched: the full lone form's per-row bytes are made
    too large, so the planner takes the lean form (resident_tilt_lone). ``block_rows`` replaces the lean blocks'
    rows at the chunk call. Returns ``(pass, lean_lone flags and block rows of the row-blocked chunks)``.
    """

    from relax.fine_pass import resident_pass2 as rp
    from relax.fine_pass import resident_tilts

    real_row_bytes = resident_tilts.tilt_lone_row_bytes
    monkeypatch.setattr(
        resident_tilts,
        "tilt_lone_row_bytes",
        lambda *a, lean=False, **k: real_row_bytes(*a, lean=True, **k) if lean else 1 << 50,
    )
    real_chunk, calls = resident_tilts.run_tilt_chunk, []

    def chunk(tilt_chunk, *a, **k):
        if k.get("lone_block_rows"):
            if block_rows is not None:
                k = dict(k, lone_block_rows=int(block_rows))
            calls.append((bool(k.get("lean_lone")), int(k["lone_block_rows"])))
        return real_chunk(tilt_chunk, *a, **k)

    monkeypatch.setattr(resident_tilts, "run_tilt_chunk", chunk)
    engine_tuning = replace(RESIDENT_PRODUCTION_ENGINE, mstep_block_rows=16)
    lean = rp._resident_pass2(
        **dict(args, translation_log_prior=None), tilt=tilt, engine_tuning=engine_tuning, scoring_overrides=ScoringOverrides()
    )
    return lean, calls


@requires_resident_gpu
def test_lean_lone_tilt_particle_is_the_full_lone_pass(monkeypatch):
    """A row-blocked particle whose full posterior arrays do not fit takes the lean form (relax#118): its scores
    built in place a block of rows at a time and the reconstruction-only posterior handler.

    At the same row blocks every value is the full lone form's: each cell is computed by the same operations and
    the particle's minimum, normalization and significance cut are taken over all of its cells. At smaller row
    blocks (a 16 GB card) only the grouping of the projector calls and the M-step blocks changes, as between the
    full lone form and the one-chunk pass.
    """

    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    tilt, n_units = _two_image_particles(args)
    args = dict(
        args,
        significant_sample_indices=args["significant_sample_indices"][:n_units],
        normalization_other_score_log_z=args["normalization_other_score_log_z"][:n_units],
    )
    whole, blocked, lone_rows = _lone_tilt_runs(monkeypatch, args, tilt)
    lean, calls = _lean_lone_run(monkeypatch, args, tilt)
    assert calls and all(lean_form and rows == 16 for lean_form, rows in calls), calls
    # The full lone form's values (one float32 operation sequence per cell; assert_matches' float32 default).
    for field in ("hard_assignment", "best_fine_rotation_indices"):
        assert_matches(getattr(blocked.finalized, field), getattr(lean.finalized, field), err_msg=field)
    for field in ("log_evidence_per_image", "best_log_score_per_image", "max_posterior_per_image", "rotation_posterior_sums"):
        assert_matches(np.asarray(getattr(blocked.finalized, field)), np.asarray(getattr(lean.finalized, field)), err_msg=field)
    _assert_tilt_pass_matches_spa(blocked, lean)
    _assert_tilt_pass_matches_spa(whole, lean)

    smaller, smaller_calls = _lean_lone_run(monkeypatch, args, tilt, block_rows=4)
    assert smaller_calls and all(lean_form and rows == 4 for lean_form, rows in smaller_calls), smaller_calls
    _assert_tilt_pass_matches_spa(whole, smaller)


def test_lean_lone_blocks_halve_to_a_fitting_block_that_divides_the_particle():
    """The lean plan halves the block from the largest row class until the plan fits, and refuses a block that
    does not divide the particle's rows when it reaches one, naming that cause (not the first-iteration CC pass)."""

    from relax.fine_pass.policy import ResidentConfigurationUnsupported
    from relax.fine_pass.resident_tilts import plan_lean_lone_blocks

    def planner(fits_rows):
        def plan_memory(lone_row_bytes, projection_factor, lone_block_rows=None):
            if lone_block_rows > fits_rows:
                raise ResidentConfigurationUnsupported(f"{lone_block_rows} rows do not fit")
            return ("plan", lone_row_bytes, lone_block_rows)

        return plan_memory

    kwargs = dict(lean_row_bytes=100, slot_capacity=4, slot_row_bytes=1, budget_bytes=None, gathered_row_copies=2)
    plan, slot_block, rows = plan_lean_lone_blocks(planner(64), lone_rows=512, largest_row_class=256, **kwargs)
    assert (plan, slot_block, rows) == (("plan", 100, 64), 4, 64)
    with pytest.raises(ResidentConfigurationUnsupported, match="16 rows do not fit"):
        plan_lean_lone_blocks(planner(8), lone_rows=512, largest_row_class=256, **kwargs)
    # 66 rows halve to 33, which divides the particle's 132, then to 16, which does not.
    with pytest.raises(ResidentConfigurationUnsupported, match="divide its 132 rows.*reached 16 rows"):
        plan_lean_lone_blocks(planner(8), lone_rows=132, largest_row_class=66, **kwargs)


def test_lean_lone_plan_fits_the_16gb_refusal():
    """The budget arithmetic of the 16 GB refusal (relax#118, job 15929725): 294,912 rows, 41 image slots of
    300,808 bytes per projected row, 1,176 translations, a 6.71 GiB chunk budget.

    The full lone form needs 14.24 GiB (its rows' posterior arrays alone 11.1 GiB); the lean form keeps 13 bytes
    per cell and fits in blocks of 128 rows, not of 256.
    """

    from relax.fine_pass import resident_chunk_memory
    from relax.fine_pass.policy import ResidentConfigurationUnsupported
    from relax.fine_pass.resident_tilts import tilt_lone_row_bytes

    gib = 1024**3
    kwargs = dict(
        row_capacity_ladder=(256,),
        image_capacity_ladder=(32,),
        mstep_block_rows=128,
        row_bytes=41 * 300808,
        n_fine_trans=41,
        n_recon_pixels=7475,
        budget_bytes=7206227933,
        rows_live_during_prepare=True,
        fixed_bytes=130772880,
        max_image_rows=294912,
        held_tile_pixels=17135,
        prepare_tile_pixels=17135,
        mstep_tile_pixels=0,
    )
    full = tilt_lone_row_bytes(1176, slot_capacity=41)
    assert full == 40512
    with pytest.raises(ResidentConfigurationUnsupported, match="needs 14.24 GiB"):
        resident_chunk_memory.plan_resident_chunk_memory(**kwargs, lone_row_bytes=full)
    lean = tilt_lone_row_bytes(1176, slot_capacity=41, lean=True)
    assert lean == 1176 * 13 + 64 + 41 * 40
    with pytest.raises(ResidentConfigurationUnsupported):
        resident_chunk_memory.plan_resident_chunk_memory(**kwargs, lone_row_bytes=lean)
    plan = resident_chunk_memory.plan_resident_chunk_memory(
        **dict(kwargs, row_bytes=41 * 300808), lone_row_bytes=lean, lone_block_rows=128
    )
    assert plan.peak_bytes <= kwargs["budget_bytes"]
    assert 128 * 41 * 300808 + 294912 * lean <= plan.peak_bytes
    print(f"lean lone plan at 128-row blocks: {plan.peak_bytes / gib:.2f} GiB of {kwargs['budget_bytes'] / gib:.2f}")


def test_lone_tilt_plan_fits_the_refused_real_configuration():
    """The budget arithmetic of the refused pass: 294,912 rows, 41 image slots, 37 translations, 146.9 KiB per
    projected row and two live copies of it, 7475 reconstruction pixels, a 42.14 GiB chunk budget.

    Held whole, the particle's rows cost rows x slot block x row bytes even at one slot (82.8 GiB); in row
    blocks it costs one block's chunk plus tilt_lone_row_bytes for each of its rows (under 1 GiB of them).
    """

    from relax.fine_pass import resident_chunk_memory
    from relax.fine_pass.policy import ResidentConfigurationUnsupported
    from relax.fine_pass.resident_tilts import tilt_lone_row_bytes

    gib = 1024**3
    row_bytes = 2 * int(146.9 * 1024)
    kwargs = dict(
        row_capacity_ladder=(256, 1024, 4096),
        image_capacity_ladder=(1, 2, 4),
        mstep_block_rows=256,
        row_bytes=row_bytes,
        n_fine_trans=41,
        n_recon_pixels=7475,
        budget_bytes=int(42.14 * gib),
        rows_live_during_prepare=True,
        max_image_rows=294912,
    )
    assert 294912 * row_bytes > 82 * gib
    with pytest.raises(ResidentConfigurationUnsupported, match="294912 rows, run alone"):
        resident_chunk_memory.plan_resident_chunk_memory(**kwargs)
    lone_row_bytes = tilt_lone_row_bytes(37, slot_capacity=41)
    assert lone_row_bytes == 37 * (resident_chunk_memory._LONE_CHUNK_CELL_BYTES + 4) + 64 + 41 * 40
    assert 294912 * lone_row_bytes < gib
    plan = resident_chunk_memory.plan_resident_chunk_memory(**kwargs, lone_row_bytes=lone_row_bytes)
    assert plan.peak_bytes <= int(42.14 * gib)
    assert max(plan.row_capacity_ladder) * row_bytes + 294912 * lone_row_bytes <= plan.peak_bytes


@requires_resident_gpu
def test_each_particles_own_translations_are_the_whole_grid_mstep(monkeypatch):
    """The M-step at each particle's translations with mass (unit_mstep_translations) is the whole-grid M-step.

    The translations it drops carry exactly zero posterior, so only the order of the image-power
    reduction over translations differs: the discrete state is exact, the scores equal and the maps and
    sums inside the repeat band. The whole grid multiplies a non-finite translated tile by a zero
    posterior into NaN where the sparse M-step drops it, so both runs must be finite (non-finite images
    stop the run at preprocessing, relax#16). K=2 with VDAM's pseudo-halfsets exercises every
    accumulator slot.
    """

    from test_resident_k_class_pass2 import _k_class_args, _resident

    from relax.fine_pass import resident_tilt_mstep, resident_tilts

    args, volumes, supports, priors = _k_class_args(2)
    args = dict(args, score_with_masked_images=True)
    tilt, n_units = _two_image_particles(args)
    supports = [class_supports[:n_units] for class_supports in supports]
    vdam = dict(args, translation_log_prior=None, tilt=tilt, **_vdam_options(n_units))
    calls = []
    sparse_tables = resident_tilt_mstep.unit_mstep_translations

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
        return resident_tilt_mstep.UnitMstepTranslations(index=index, valid=np.ones_like(index, dtype=bool))

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


@requires_resident_gpu
def test_translation_blocked_tilt_mstep_is_the_one_block_mstep(monkeypatch):
    """An M-step split into one-translation blocks is the one-block M-step (relax#27).

    The blocks merge each row's partials before its one adjoint, so only float32 reassociation within a
    row and the atomics' order differ: the discrete state is exact, the scores equal and the maps and
    sums inside the repeat band. K=2 with VDAM's pseudo-halfsets exercises every accumulator slot.
    """

    from test_resident_k_class_pass2 import _k_class_args, _resident

    from relax.fine_pass import resident_tilt_mstep, resident_tilts

    args, volumes, supports, priors = _k_class_args(2)
    args = dict(args, score_with_masked_images=True)
    tilt, n_units = _two_image_particles(args)
    supports = [class_supports[:n_units] for class_supports in supports]
    vdam = dict(args, translation_log_prior=None, tilt=tilt, **_vdam_options(n_units))
    whole = _resident(vdam, volumes, supports, priors)
    blocks = resident_tilt_mstep.mstep_translation_blocks
    n_blocks = []

    def one_translation_each(kept, *, bytes_per_translation, tile_budget_bytes, wide_tile_budget_bytes):
        out = blocks(kept, bytes_per_translation=1, tile_budget_bytes=1, wide_tile_budget_bytes=1)
        n_blocks.append(len(out))
        return out

    monkeypatch.setattr(resident_tilts, "mstep_translation_blocks", one_translation_each)
    blocked = _resident(vdam, volumes, supports, priors)
    assert max(n_blocks) > 1

    assert_matches(whole.per_class_hard_assignments, blocked.per_class_hard_assignments)
    for field in ("class_log_evidence_per_image", "class_best_log_score_per_image", "class_rotation_posterior_sums"):
        assert_matches(np.asarray(getattr(whole, field)), np.asarray(getattr(blocked, field)), err_msg=field)
    for k in range(2):
        _assert_accumulators_match(whole.Ft_y[k], whole.Ft_ctf[k], blocked.Ft_y[k], blocked.Ft_ctf[k])
    _assert_noise_stats_match(whole.noise_stats, blocked.noise_stats)


@requires_resident_gpu
def test_mstep_at_the_chunks_valid_units_is_the_unit_capacity_mstep(monkeypatch):
    """An M-step sized at the chunk's valid units is the M-step sized at its unit capacity (relax#102).

    Both runs get a tile budget of two translations at the unit capacity, so the capacity-sized M-step
    runs in two-translation blocks and the valid-units one in fewer, larger blocks. The blocks merge each
    row's partials before its one adjoint, so only float32 reassociation within a row and the atomics'
    order differ: the discrete state is exact, the scores equal and the maps and sums inside the repeat
    band. K=2 with VDAM's pseudo-halfsets exercises every accumulator slot.
    """

    from test_resident_k_class_pass2 import _k_class_args, _resident

    from relax.fine_pass import resident_tilt_mstep, resident_tilts

    args, volumes, supports, priors = _k_class_args(2)
    args = dict(args, score_with_masked_images=True)
    tilt, n_units = _two_image_particles(args)
    supports = [class_supports[:n_units] for class_supports in supports]
    vdam = dict(args, translation_log_prior=None, tilt=tilt, **_vdam_options(n_units))
    units, blocks = resident_tilt_mstep.tilt_mstep_units, resident_tilt_mstep.mstep_translation_blocks
    sizing, n_blocks = [], []

    def recorded_units(n_valid_units, *, unit_capacity, capacity_sized):
        valid_units = units(n_valid_units, unit_capacity=unit_capacity)
        sizing.append((int(unit_capacity) if capacity_sized else valid_units, valid_units, int(unit_capacity)))
        return sizing[-1][0]

    def two_capacity_translations(kept, *, bytes_per_translation, tile_budget_bytes, wide_tile_budget_bytes):
        mstep_units, _, unit_capacity = sizing[-1]
        capacity_bytes = bytes_per_translation // mstep_units * unit_capacity
        out = blocks(
            kept, bytes_per_translation=bytes_per_translation, tile_budget_bytes=2 * capacity_bytes,
            wide_tile_budget_bytes=2 * capacity_bytes,
        )
        n_blocks[-1] += len(out)
        return out

    monkeypatch.setattr(resident_tilts, "mstep_translation_blocks", two_capacity_translations)
    # One row class that holds every particle: the fixture's particles run as one regular chunk below its unit class
    # (past the row classes, a particle runs alone at unit class 1, where its valid units are the capacity).
    monkeypatch.setattr(
        resident_tilts, "tilt_capacity_ladders", lambda rows, images, **k: ((16384,), tuple(sorted(set(images))))
    )
    runs = []
    for capacity_sized in (True, False):
        n_blocks.append(0)
        monkeypatch.setattr(
            resident_tilts,
            "tilt_mstep_units",
            lambda n, *, unit_capacity, capacity_sized=capacity_sized: recorded_units(
                n, unit_capacity=unit_capacity, capacity_sized=capacity_sized
            ),
        )
        runs.append(_resident(vdam, volumes, supports, priors))
    capacity, valid = runs
    # The fixture's chunks hold fewer units than their capacity, so the valid-units M-step has fewer blocks.
    assert any(valid_units < unit_capacity for _, valid_units, unit_capacity in sizing)
    assert n_blocks[1] < n_blocks[0]

    assert_matches(capacity.per_class_hard_assignments, valid.per_class_hard_assignments)
    for field in ("class_log_evidence_per_image", "class_best_log_score_per_image", "class_rotation_posterior_sums"):
        assert_matches(np.asarray(getattr(capacity, field)), np.asarray(getattr(valid, field)), err_msg=field)
    for k in range(2):
        _assert_accumulators_match(capacity.Ft_y[k], capacity.Ft_ctf[k], valid.Ft_y[k], valid.Ft_ctf[k])
    _assert_noise_stats_match(capacity.noise_stats, valid.noise_stats)


@requires_resident_gpu
def test_row_partials_merged_before_the_adjoint_keep_their_float32_weight():
    """The swamping toy of relax#27: many rows into one voxel, each row's weight split over translation blocks.

    200k rows of weight 0.5-1 plus four tails (one per block) of up to 2e-3 of it, all at the DC voxel.
    Adjoined block by block, each tail lands on a float32 total ~1e5 times the row and is lost (-4e-3);
    merged per row first (the tilt M-step's row sums, ``resident_tilt_mstep.tilt_mstep_program``), they enter
    at the row's own scale and the voxel keeps the one-block total (three runs on a local GPU: one block
    -1.6e-6 to -9.9e-6, merged -2.0e-6 to -9.4e-6, blocks -4.0e-3 against the float64 sum; the float32
    atomics' order moves the first two between runs).
    """

    from types import SimpleNamespace

    from relax.fine_pass import resident_mstep_blocks

    n, n_rows = 8, 200_000
    spec = SimpleNamespace(
        stable_window=False,
        image_shape=(n, n),
        recon_volume_shape=(n, n, n),
        mstep_max_r=float(n // 2),
        max_adjoint_block_bytes=1 << 30,
    )
    # The window's one pixel is the DC term: every row adds into the DC voxel.
    tables = SimpleNamespace(relion_x_half_recon_indices=jnp.zeros(1, dtype=jnp.int32), window_logical=None)
    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n_rows, 3, 3))
    rng = np.random.default_rng(27)
    weight = rng.uniform(0.5, 1.0, n_rows).astype(np.float32)
    tails = (rng.uniform(0.0, 2e-3, (4, n_rows)) * weight).astype(np.float32)
    volume_size = n * n * (n // 2 + 1)

    def backproject(rows, volume):
        rows = jnp.asarray(rows, dtype=jnp.float32)[:, None]
        _, Ft_ctf = resident_mstep_blocks.backproject_block_rows(
            rows.astype(jnp.complex64),
            rows,
            rotations,
            jnp.zeros(volume_size, jnp.complex64),
            volume,
            tables=tables,
            spec=spec,
        )
        return Ft_ctf

    zero = jnp.zeros(volume_size, dtype=jnp.float32)
    one_block = backproject(weight + tails.sum(axis=0), zero)
    per_block = backproject(weight, zero)
    for tail in tails:
        per_block = backproject(tail, per_block)
    row_sums = jnp.zeros(n_rows, dtype=jnp.float32).at[jnp.arange(n_rows)].add(weight)
    for tail in tails:
        row_sums = row_sums.at[jnp.arange(n_rows)].add(tail)
    merged = backproject(row_sums, zero)

    total = float(np.sum(weight, dtype=np.float64) + np.sum(tails, dtype=np.float64))
    dc = int(np.argmax(np.asarray(one_block)))
    error = {
        name: float(np.asarray(v)[dc]) / total - 1.0
        for name, v in (("one block", one_block), ("merged", merged), ("per block", per_block))
    }
    print("DC weight relative to the float64 sum", error)
    assert abs(error["one block"]) < 5e-5 and abs(error["merged"]) < 5e-5
    assert error["per block"] < -1e-3


# ---------------------------------------------------------------------------
# The --firstiter_cc iteration (normalized CC, winner takes all)
# ---------------------------------------------------------------------------


@requires_resident_gpu
def test_one_image_particles_reproduce_the_spa_firstiter_cc_pass():
    """The CC iteration through the tilt runner, one image per particle, is the SPA CC pass.

    The tilt runner translates each image's resident CC operands per slot (RELION's score translation) where
    the SPA pass builds its translated CC tiles per chunk; both score with RELION's fine CC reduction and keep
    each unit's best cell, then backproject it with the Gaussian M-step.
    """

    from relax.fine_pass import resident_pass2 as rp

    args = dict(
        _driver_fixture_args(),
        score_with_masked_images=True,
        relion_firstiter_score_mode="normalized_cc",
        relion_firstiter_winner_take_all=True,
        relion_exact_fine_normalized_cc=True,
    )
    spa = rp._resident_pass2(**args, engine_tuning=RESIDENT_PRODUCTION_ENGINE, scoring_overrides=ScoringOverrides())
    tomo = rp._resident_pass2(**dict(args, translation_log_prior=None), tilt=_one_image_tilt_inputs(
        args), engine_tuning=RESIDENT_PRODUCTION_ENGINE,
        scoring_overrides=ScoringOverrides(),
    )
    for field in ("hard_assignment", "best_fine_rotation_indices"):
        assert_matches(getattr(spa.finalized, field), getattr(tomo.finalized, field), err_msg=field)
    # The CC scores are float32 (the evidence offset is added in float64): compare them at float32.
    for field in ("best_log_score_per_image", "max_posterior_per_image"):
        assert_matches(
            np.float32(getattr(spa.finalized, field)), np.float32(getattr(tomo.finalized, field)), err_msg=field
        )
    assert _rel_l2(spa.Ft_y[0], tomo.Ft_y[0]) < _BPREF_ATOMIC_BAND
    assert _rel_l2(spa.Ft_ctf[0], tomo.Ft_ctf[0]) < _BPREF_ATOMIC_BAND


@requires_resident_gpu
def test_compiled_tilt_fold_is_the_eager_fold(monkeypatch):
    """accumulate_tilt_chunk_terms compiled (one program, the optics groups vmapped) gives the eager function's
    statistics, for the chunk of a real tilt pass and for the same chunk split over three optics groups.

    ``sumw`` is compared at the float32 band: it is a float32 atomic scatter-add of the rows' posterior
    (segment_sum_by_image) summed in float32 and cast to float64, so it is not reproducible to float64 even
    eager against eager (2^-24 on 1.0 in 2 of 10 back-to-back eager runs on an H100, job 15669435). Every other
    leaf keeps the float64 default.
    """

    from dataclasses import replace as dc_replace

    import jax

    from relax.fine_pass import resident_pass2 as rp
    from relax.fine_pass import resident_tilts

    calls = []
    real_fold = resident_tilts.accumulate_tilt_chunk_terms

    def spy(stats, **kwargs):
        calls.append((stats, kwargs))
        return real_fold(stats, **kwargs)

    monkeypatch.setattr(resident_tilts, "accumulate_tilt_chunk_terms", spy)
    args = dict(_driver_fixture_args(), score_with_masked_images=True)
    rp._resident_pass2(
        **dict(args, translation_log_prior=None), tilt=_one_image_tilt_inputs(args),
        engine_tuning=RESIDENT_PRODUCTION_ENGINE, scoring_overrides=ScoringOverrides(),
    )
    assert calls
    stats, kwargs = calls[0]

    float32_atomic = {".sumw"}

    def assert_same(compiled, eager):
        leaves_c, tree_c = jax.tree_util.tree_flatten_with_path(compiled)[0], jax.tree_util.tree_structure(compiled)
        leaves_e, tree_e = jax.tree_util.tree_leaves(eager), jax.tree_util.tree_structure(eager)
        assert tree_c == tree_e
        for (path, got), want in zip(leaves_c, leaves_e, strict=True):
            got, want = np.asarray(got), np.asarray(want)
            if got.dtype.kind in "fc":
                band = {"rtol": RTOL[np.dtype(np.float32)]} if jax.tree_util.keystr(path) in float32_atomic else {}
                assert_matches(got, want, **band)
            else:
                np.testing.assert_array_equal(got, want)

    assert_same(real_fold(stats, **kwargs), real_fold.__wrapped__(stats, **kwargs))

    # The same chunk over three optics groups: each image's group, each group's noise shells.
    n_groups = 3
    spec = kwargs["spec"]
    spec = dc_replace(spec, stats_config=dc_replace(spec.stats_config, n_optics_groups=n_groups))
    n_images = int(kwargs["image_ids"].shape[0])
    image_groups = jnp.asarray(np.arange(n_images) % n_groups, dtype=jnp.int32)
    mstep = kwargs["mstep"]
    mstep = mstep._replace(
        noise_shells=jnp.stack([jnp.asarray(mstep.noise_shells) * f for f in (1.0, 0.5, 2.0)])
    )
    n_shells = int(spec.stats_config.n_shells)
    grouped = stats._replace(
        wsum_sigma2_noise=jnp.zeros((n_groups, n_shells), jnp.float64),
        wsum_img_power=jnp.zeros((n_groups, n_shells), jnp.float64),
        sumw=jnp.zeros(n_groups, jnp.float64),
    )
    n_units = int(kwargs["unit_ids"].shape[0])
    grouped_kwargs = dict(
        kwargs,
        spec=spec,
        mstep=mstep,
        operands=kwargs["operands"]._replace(optics_groups=image_groups),
        unit_optics_groups=jnp.asarray(np.arange(n_units) % n_groups, dtype=jnp.int32),
    )
    compiled = real_fold(grouped, **grouped_kwargs)
    assert compiled.wsum_sigma2_noise.shape == (n_groups, n_shells)
    assert_same(compiled, real_fold.__wrapped__(grouped, **grouped_kwargs))


@requires_resident_gpu
def test_a_compact_grid_at_its_rotation_capacity_is_the_exact_length_pass(monkeypatch):
    """A local search's compact coarse grid sizes its rotation-mass accumulator at the capacity class
    (resolve_statistics_config rotation_capacity), so one chunk fold serves every iteration's grid size. The padded
    slots take no mass: the pass equals the one whose accumulator is the grid's exact length, with the discrete
    state, scores and rotation masses exact and the maps and sums in the repeat band."""

    from relax.fine_pass import resident_pass2 as rp
    from relax.fine_pass import resident_statistics

    args = dict(_driver_fixture_args(), score_with_masked_images=True, return_stats=True)
    tilt, n_units = _two_image_particles(args)
    support = args.pop("significant_sample_indices")[:n_units]
    n_coarse_rot = int(np.asarray(args.pop("rotation_log_prior")).size)
    args["translation_log_prior"] = None
    resolve = resident_statistics.resolve_statistics_config
    lengths = []

    def run(rotation_capacity):
        def resolved(**kwargs):
            config = resolve(**dict(kwargs, rotation_capacity=rotation_capacity))
            lengths.append(int(config.n_coarse_rot))
            return config

        monkeypatch.setattr(resident_statistics, "resolve_statistics_config", resolved)
        return rp.compute_tilt_pass2_stats_resident(
            tilt=tilt, coarse_rotation_ids=np.arange(n_coarse_rot), significant_sample_indices=support,
            rotation_log_prior=None, **args, engine_tuning=RESIDENT_PRODUCTION_ENGINE,
            scoring_overrides=ScoringOverrides(),
        )

    capacity, exact = run(True), run(False)
    assert lengths == [resident_statistics.resident_image_capacity(n_coarse_rot), n_coarse_rot]
    assert lengths[0] > n_coarse_rot
    np.testing.assert_array_equal(capacity.best_rotation_indices, exact.best_rotation_indices)
    np.testing.assert_array_equal(capacity.hard_assignment, exact.hard_assignment)
    for field in ("log_evidence_per_image", "best_log_score_per_image", "max_posterior_per_image"):
        np.testing.assert_array_equal(getattr(capacity.relion_stats, field), getattr(exact.relion_stats, field))
    rotation_sums = np.asarray(capacity.relion_stats.rotation_posterior_sums)
    assert rotation_sums.shape == (n_coarse_rot,)
    assert_matches(rotation_sums, np.asarray(exact.relion_stats.rotation_posterior_sums))
    assert _rel_l2(exact.Ft_y, capacity.Ft_y) < _BPREF_ATOMIC_BAND
    assert _rel_l2(exact.Ft_ctf, capacity.Ft_ctf) < _BPREF_ATOMIC_BAND
    _assert_noise_stats_match(exact.noise_stats, capacity.noise_stats)
