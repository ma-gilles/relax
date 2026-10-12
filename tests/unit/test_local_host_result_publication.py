"""Exact publication contracts; the same tests also run on a pinned GPU."""

from dataclasses import fields, is_dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.classification import k_class_results
from relax.types import _stats_array, make_noise_stats, make_relion_stats

pytestmark = pytest.mark.unit


def _same_bytes(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
    elif is_dataclass(actual):
        assert type(actual) is type(expected)
        for field in fields(actual):
            _same_bytes(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(actual, tuple):
        assert type(actual) is type(expected)
        for a, b in zip(actual, expected, strict=True):
            _same_bytes(a, b)
    else:
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert_matches(a, b, strict=True)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("dtype", [None, np.float32, np.float64])
@pytest.mark.parametrize("device_input", [False, True])
def test_host_factory_cast_matches_jax_bytes(x64, dtype, device_input):
    with jax.enable_x64(x64):
        values = np.asarray(
            [0.0, -0.0, 1.0, -2.5, 1 + 2**-24, 1 + 3 * 2**-24, 2**-149, -(2**-149), 2**-126, np.inf, -np.inf, np.nan],
            dtype=np.float64,
        )
        if device_input:
            values = jnp.asarray(values)
        actual = _stats_array(values, dtype, True)
        assert isinstance(actual, np.ndarray)
        _same_bytes(actual, _stats_array(values, dtype, False))


def _noise(values, *, host=False, optional=True):
    return make_noise_stats(
        wsum_sigma2_noise=values,
        wsum_img_power=values,
        wsum_sigma2_offset=-0.0,
        sumw=-0.0,
        **(
            {
                name: values
                for name in (
                    "wsum_noise_a2",
                    "wsum_noise_xa",
                    "wsum_norm_correction",
                    "wsum_scale_correction_xa",
                    "wsum_scale_correction_aa",
                )
            }
            if optional
            else {}
        ),
        host_arrays=host,
    )


@pytest.mark.parametrize("optional", [False, True])
def test_single_class_reduction_preserves_payloads_and_scalar_sum(optional):
    values = np.asarray(
        [0, 0x80000000, 0x7FC00021, 0xFFC00022, 0x7F800000, 0xFF800000, 1, 0x80000001, 0x3F800000],
        dtype=np.uint32,
    ).view(np.float32)
    stats = _noise(values, optional=optional)
    actual = k_class_results._sum_noise_stats((stats,), host_arrays=True)
    expected = k_class_results._sum_noise_stats((stats,))
    _same_bytes(actual, expected)
    assert isinstance(actual.wsum_sigma2_noise, np.ndarray)
    _same_bytes(np.asarray(jnp.sum(jnp.stack([jnp.asarray(values)]), axis=0)), values)


@pytest.mark.parametrize("host", [False, True])
def test_complete_result_publication_matches_device_result(host):
    values = np.asarray([0.0, -0.0, 0.25], dtype=np.float32)
    stats = make_relion_stats(
        log_evidence_per_image=[4.0, 5.0, 6.0],
        best_log_score_per_image=[3.0, 4.0, 5.0],
        max_posterior_per_image=[0.5, 0.8, 0.9],
        rotation_posterior_sums=values,
        host_arrays=host,
    )
    noise = _noise(values, host=host)._replace(sumw=2.0)
    kwargs = dict(
        class_log_evidence=np.asarray([[4.0, 5.0, 6.0]]),
        new_means=None,
        Ft_y=[np.asarray([1 + 2j, 3 - 4j], dtype=np.complex64)],
        Ft_ctf=[np.asarray([5.0, 6.0], dtype=np.float32)],
        per_class_hard_assignments=np.asarray([[3, 1, 7]], dtype=np.int32),
        per_class_stats=(stats,),
        noise_stats=(noise,),
        per_class_best_pose_rotation_ids=[np.asarray([3, 1, 7], dtype=np.int32)],
        per_class_best_pose_translations=[np.zeros((3, 2), dtype=np.float32)],
        per_class_best_pose_rotations=[np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))],
    )
    expected = k_class_results._assemble_result(**kwargs)
    actual = k_class_results._assemble_result(
        **kwargs,
        host_accumulators=True,
        host_stats_publication=True,
    )
    _same_bytes(actual, expected)
    for name in (
        "Ft_y",
        "Ft_ctf",
        "class_responsibilities",
        "class_posterior_sums",
        "class_mstep_posterior_sums",
        "class_assignments",
        "pose_assignments",
    ):
        assert isinstance(getattr(actual, name), np.ndarray)
    assert all(isinstance(field, np.ndarray) for field in actual.stats)


