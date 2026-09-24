"""Per-optics-group noise in the exact local engine's big-JIT bucket route (CPU).

Local search runs its pass-1 parent probe (score-only) and the full-box final pass
on this route. Two optics groups with identical noise rows must reproduce the
one-group result, and different rows must change it.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.local.local_em_engine import run_local_em_exact

GROUPS = np.array([0, 1, 1], dtype=np.int32)


def _case():
    import test_refine_relion_mode as fixture

    rng = np.random.default_rng(0)
    _, mean, noise, layout = fixture._sparse_big_jit_local_case(rng)
    dataset = fixture.MockDataset(3, rng)
    noise = jnp.linspace(0.5, 2.0, noise.size, dtype=jnp.float32)
    return dataset, mean, noise, layout


COMMON = dict(
    image_batch_size=3,
    rotation_block_size=8,
    projection_padding_factor=2,
    reconstruction_padding_factor=2,
    do_gridding_correction=True,
    half_spectrum_scoring=True,
    score_with_masked_images=True,
    adaptive_fraction=0.999,
    reconstruct_significant_only=True,
)
PROBE = dict(
    current_size=6,
    accumulate_noise=False,
    disable_adjoint_y=True,
    disable_adjoint_ctf=True,
    return_reconstruction_sample_indices=True,
    return_profile=True,
    max_significants=3,
    score_only=True,
)
FULL_BOX = dict(
    current_size=8,
    accumulate_noise=True,
    stats_use_reconstruction_probs=True,
    return_best_pose_details=True,
)


def _run(noise, *, groups=None, **kwargs):
    dataset, mean, _, layout = _case()
    extra = {} if groups is None else {"optics_group_ids": groups}
    return run_local_em_exact(dataset, mean, noise, layout, "linear_interp", **COMMON, **kwargs, **extra)


@pytest.mark.unit
@pytest.mark.parametrize("kwargs", [PROBE, FULL_BOX], ids=["parent_probe", "full_box"])
def test_identical_groups_reproduce_one_group(kwargs):
    _, _, noise, _ = _case()
    one = _run(noise, **kwargs)
    two = _run(jnp.stack([noise, noise]), groups=GROUPS, **kwargs)
    assert_matches(two.hard_assignments, one.hard_assignments)
    np.testing.assert_allclose(
        np.asarray(two.stats.log_evidence_per_image), np.asarray(one.stats.log_evidence_per_image), rtol=1e-6
    )
    if kwargs["accumulate_noise"]:
        np.testing.assert_allclose(np.asarray(two.Ft_y), np.asarray(one.Ft_y), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(two.Ft_ctf), np.asarray(one.Ft_ctf), rtol=1e-5, atol=1e-6)
        one_noise, two_noise = one.noise_stats, two.noise_stats
        assert np.asarray(two_noise.wsum_sigma2_noise).shape[0] == 2
        np.testing.assert_allclose(
            np.asarray(two_noise.wsum_sigma2_noise).sum(axis=0),
            np.asarray(one_noise.wsum_sigma2_noise),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(two_noise.wsum_img_power).sum(axis=0), np.asarray(one_noise.wsum_img_power), rtol=1e-5
        )
        np.testing.assert_allclose(np.sum(two_noise.sumw), one_noise.sumw, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(two_noise.wsum_norm_correction), np.asarray(one_noise.wsum_norm_correction), rtol=1e-5
        )


@pytest.mark.unit
def test_group_rows_are_used():
    _, _, noise, _ = _case()
    one = _run(noise, **PROBE)
    two = _run(jnp.stack([noise, 4.0 * noise]), groups=GROUPS, **PROBE)
    assert not np.allclose(
        np.asarray(two.stats.log_evidence_per_image)[GROUPS == 1],
        np.asarray(one.stats.log_evidence_per_image)[GROUPS == 1],
    )
    np.testing.assert_allclose(
        np.asarray(two.stats.log_evidence_per_image)[GROUPS == 0],
        np.asarray(one.stats.log_evidence_per_image)[GROUPS == 0],
        rtol=1e-6,
    )
