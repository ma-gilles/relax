"""The K=1 and K-class dense routes share one adaptive engine call owner."""

import numpy as np

from relax.classification import k_class
from relax.refinement import half_scoring


def test_the_dense_pass2_switches_are_gone_and_refused():
    assert not hasattr(k_class, "_sparse_pass2_selected")
    import json
    from pathlib import Path

    import relax

    retired = json.loads((Path(relax.__file__).parent / "renamed_environment.json").read_text())
    text = json.dumps(retired)
    for env_name in ("RELAX_K1_DENSE_PASS2", "RELAX_K_CLASS_DENSE_PASS2"):
        assert env_name in text


def _grids(fine_mstep):
    return half_scoring.AdaptivePass2Grids(
        coarse_rotations=np.zeros((2, 3, 3), dtype=np.float32),
        coarse_translations=np.zeros((1, 2), dtype=np.float32),
        fine_rotations=np.zeros((4, 3, 3), dtype=np.float32),
        fine_translations=np.zeros((3, 2), dtype=np.float32),
        rotation_parent_map=np.zeros(4, dtype=np.int64),
        translation_parent_map=np.zeros(3, dtype=np.int64),
        fine_mstep_rotations=fine_mstep,
        coarse_translation_phase_source=np.zeros((1, 2), dtype=np.float64),
        n_fine_translations=3,
    )


def test_common_engine_keywords_are_the_sparse_pass2_keywords():
    fine_mstep = np.ones((4, 3, 3), dtype=np.float32)
    def common_kwargs(*, max_significants):
        return half_scoring._adaptive_engine_common_kwargs(
            _grids(fine_mstep),
            half_scoring.DensePriorSpec(
                rotation_log_prior_k=None,
                class_rotation_log_prior_k=None,
                translation_log_prior=None,
                translation_search_base=None,
                trans_prior_center_for_engine=None,
                class_log_priors="priors",
            ),
            half_scoring.DenseBatchPolicy(
                image_batch_size=1,
                safe_batch_sizes=lambda *_args, **_kwargs: (1, 1),
                max_significants=max_significants,
            ),
            half_scoring.DenseSamplingSpec(
                effective_rotations=None,
                current_translations=None,
                base_translations=None,
                current_healpix_order=np.int64(2),
                oversampling_order=0,
                translation_step=1.0,
                random_perturbation=0.0,
                cs_for_engine=None,
            ),
            half_scoring.DenseExecutionPolicy(
                disc_type="linear_interp",
                disable_adjoint_y=False,
                disable_adjoint_ctf=False,
                return_best_pose_details=True,
                bpref_device_signature_active=False,
                debug_iteration=3,
            ),
        )

    sparse = common_kwargs(max_significants=None)
    capped = common_kwargs(max_significants=5)
    expected_keys = {
        "class_log_priors", "accumulate_noise", "adaptive_fraction", "max_significants",
        "relion_fine_mstep_prune", "coarse_healpix_order", "fine_mstep_rotations_override",
        "return_best_pose_details", "bpref_device_signature_active", "debug_iteration",
    }
    assert set(sparse) == set(capped) == expected_keys
    assert sparse["accumulate_noise"] is True
    assert sparse["adaptive_fraction"] == half_scoring.RELION_ADAPTIVE_FRACTION
    assert sparse["max_significants"] == -1 and capped["max_significants"] == 5
    assert sparse["relion_fine_mstep_prune"] is True and capped["relion_fine_mstep_prune"] is True
    assert sparse["fine_mstep_rotations_override"] is fine_mstep
    assert type(sparse["coarse_healpix_order"]) is int and sparse["coarse_healpix_order"] == 2
    assert sparse["class_log_priors"] == "priors" and sparse["debug_iteration"] == 3


def test_coarse_pose_assignments_need_a_fine_pass():
    from relax.dense.score_outputs import _collapse_fine_pose_assignments_to_coarse as collapse

    ha = np.array([0, 5], dtype=np.int32)
    assert collapse(ha, rot_parent_map=None, trans_parent_map=None, n_trans_coarse=1, n_trans_fine=None) is None
    assert collapse(ha, rot_parent_map=[0, 0], trans_parent_map=[0, 0, 0], n_trans_coarse=1, n_trans_fine=None) is None
    assert collapse(ha, rot_parent_map=[0, 0], trans_parent_map=None, n_trans_coarse=1, n_trans_fine=3) is None
    # Fine pose 5 = rotation 1, translation 2 of 3; parents rotation 1, translation 0 of 2 -> coarse 1 * 2 + 0.
    coarse = collapse(ha, rot_parent_map=[0, 1], trans_parent_map=[0, 1, 0], n_trans_coarse=2, n_trans_fine=3)
    assert coarse.dtype == np.int32 and coarse.tolist() == [0, 2]
