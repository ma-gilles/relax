"""The resident pass with subtomogram units (S4.2) in its one-image case equals the SPA pass (GPU).

A particle with one image, an identity left matrix and the SPA phases is an SPA image: RELION's GPU path
then keeps the SPA matrices (isIdentity, acc_ml_optimiser_impl.h:1614-1618) and divides nothing by the
image count. So ``_resident_pass2(tilt=...)``, which scores and backprojects through the tilt chunk runner,
must reproduce ``_resident_pass2`` on the SPA driver fixture: the discrete state exactly, the scores in the
float band, and the maps and sums inside the resident driver's own repeat band.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_resident_pass2_driver import (
    _driver_fixture_args,
    _resident_production_env,  # noqa: F401 (fixture)
    requires_resident_gpu,
)

pytestmark = pytest.mark.unit


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

    args = _driver_fixture_args()
    spa = rp._resident_pass2(**args)
    tomo = rp._resident_pass2(**args, tilt=_one_image_tilt_inputs(args))

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

    # The resident driver's repeat band (test_resident_driver_repeats_itself): float32 BPref atomics.
    assert rel_l2(spa.Ft_y[0], tomo.Ft_y[0]) < 1e-7
    assert rel_l2(spa.Ft_ctf[0], tomo.Ft_ctf[0]) < 1e-7
    for field in (
        "wsum_sigma2_noise",
        "wsum_img_power",
        "wsum_norm_correction",
        "wsum_scale_correction_xa",
        "wsum_scale_correction_aa",
    ):
        assert rel_l2(getattr(spa.noise_stats, field), getattr(tomo.noise_stats, field)) < 1e-7, field
    assert abs(float(spa.noise_stats.sumw) - float(tomo.noise_stats.sumw)) <= 1e-6 * abs(float(spa.noise_stats.sumw))
