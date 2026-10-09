"""The dense pass-2 switches are retired; fine pose assignments collapse onto the coarse grid.

The keywords both adaptive dense routes pass the engine are pinned by
``test_firstiter_cc_batch_budget.test_kclass_nonfirstiter_adaptive_dispatch_sizes_actual_fine_grid``.
"""

import numpy as np

from relax.classification import k_class


def test_the_dense_pass2_switches_are_gone_and_refused():
    assert not hasattr(k_class, "_sparse_pass2_selected")
    import json
    from pathlib import Path

    import relax

    retired = json.loads((Path(relax.__file__).parent / "renamed_environment.json").read_text())
    text = json.dumps(retired)
    for env_name in ("RELAX_K1_DENSE_PASS2", "RELAX_K_CLASS_DENSE_PASS2"):
        assert env_name in text


def test_coarse_pose_assignments_need_a_fine_pass():
    from relax.dense.score_outputs import _collapse_fine_pose_assignments_to_coarse as collapse

    ha = np.array([0, 5], dtype=np.int32)
    assert collapse(ha, rot_parent_map=None, trans_parent_map=None, n_trans_coarse=1, n_trans_fine=None) is None
    assert collapse(ha, rot_parent_map=[0, 0], trans_parent_map=[0, 0, 0], n_trans_coarse=1, n_trans_fine=None) is None
    assert collapse(ha, rot_parent_map=[0, 0], trans_parent_map=None, n_trans_coarse=1, n_trans_fine=3) is None
    # Fine pose 5 = rotation 1, translation 2 of 3; parents rotation 1, translation 0 of 2 -> coarse 1 * 2 + 0.
    coarse = collapse(ha, rot_parent_map=[0, 1], trans_parent_map=[0, 1, 0], n_trans_coarse=2, n_trans_fine=3)
    assert coarse.dtype == np.int32 and coarse.tolist() == [0, 2]
