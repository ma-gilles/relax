import numpy as np
import pytest

from relax.ppca_refinement.initialization import real_volume_to_centered_fourier_half
from relax.ppca_refinement.schedule import (
    PPCARefinementScheduleState,
    evaluate_halfset_resolution_gate,
    loading_subspace_agreement,
)


pytestmark = pytest.mark.unit


def _passing_state(**changes):
    state = PPCARefinementScheduleState(
        current_size=32,
        healpix_order=2,
        best_pose_indices=np.array([1, 2, 3], dtype=np.int32),
        previous_best_pose_indices=np.array([1, 2, 3], dtype=np.int32),
        halfset_means_aligned=True,
        halfset_resolution_supports=True,
        no_halfset_drift=True,
        kclass_schedule_allows=True,
        pmax_mean=0.9,
        q=1,
    )
    return state.replace(**changes)


def test_fsc_good_but_best_poses_changed_blocks_resolution_increase():
    decision = evaluate_halfset_resolution_gate(
        _passing_state(previous_best_pose_indices=np.array([1, 9, 3], dtype=np.int32)),
        pose_stability_threshold=0.0,
    )
    assert not decision.allow_increase
    assert "best poses changed" in decision.reasons


def test_best_poses_stable_but_halfset_means_misaligned_blocks_resolution_increase():
    decision = evaluate_halfset_resolution_gate(_passing_state(halfset_means_aligned=False))
    assert not decision.allow_increase
    assert "halfset means not aligned" in decision.reasons


def test_w_sign_flip_with_same_subspace_is_stable_diagnostic():
    W = np.eye(3, 5, dtype=np.float32)
    assert loading_subspace_agreement(W, -W) > 1.0 - 1e-6


def test_q_zero_schedule_allows_when_homogeneous_gates_pass():
    decision = evaluate_halfset_resolution_gate(_passing_state(q=0), pose_stability_threshold=0.0)
    assert decision.allow_increase
    assert decision.reasons == ()


def test_loading_subspace_agreement_on_half_fourier_data_matches_real_space_ground_truth():
    """Fourier-domain agreement must match the real-space value under a Parseval-consistent FFT.

    Parseval's theorem preserves inner-product ratios (hence principal
    angles) between a real-space signal and its Fourier transform, so the
    subspace agreement computed directly on two real volumes is the ground
    truth a correctly Hermitian-weighted comparison of their packed
    half-Fourier representations (``volume_shape`` given) must reproduce. An
    unweighted half-space comparison (``volume_shape`` omitted) does not.
    """
    rng = np.random.default_rng(0)
    volume_shape = (4, 4, 4)
    q = 2
    A_real = rng.standard_normal((q, *volume_shape))
    B_real = rng.standard_normal((q, *volume_shape))

    ground_truth = loading_subspace_agreement(A_real.reshape(q, -1), B_real.reshape(q, -1))
    A_half = np.stack([real_volume_to_centered_fourier_half(A_real[i]) for i in range(q)])
    B_half = np.stack([real_volume_to_centered_fourier_half(B_real[i]) for i in range(q)])

    weighted = loading_subspace_agreement(A_half, B_half, volume_shape=volume_shape)
    naive = loading_subspace_agreement(A_half, B_half)

    assert weighted == pytest.approx(ground_truth, abs=1e-5)
    assert naive != pytest.approx(ground_truth, abs=1e-3)
