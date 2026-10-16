"""K-class subset result collection, subset expansion and dtype contracts."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import matches

from relax.classification import k_class, k_class_firstiter, k_class_inputs
from relax.classification.k_class_results import make_relion_stats

pytestmark = pytest.mark.unit


def _results(**overrides):
    kwargs = dict(
        n_images=3,
        accumulate_noise=False,
        return_best_pose_details=True,
        full_group_count=None,
        pose_dtype=np.float32,
        score_dtype=np.float32,
    )
    kwargs.update(overrides)
    return k_class_firstiter._PerClassSubsetResults(**kwargs)


def test_empty_class_fills_zero_results_in_class_order():
    results = _results()
    mean = jnp.ones(4, dtype=jnp.complex64)
    results.append_empty_class(
        mean=mean,
        class_log_evidence=np.asarray([1.0, 2.0, 3.0]),
        noise_variance=None,
        class_index=0,
        n_classes=1,
        n_rot=5,
        host_accumulators=True,
    )
    assert isinstance(results.Ft_y[0], np.ndarray) and np.all(results.Ft_y[0] == 0)
    assert results.Ft_ctf[0].dtype == np.float32 and results.Ft_ctf[0].shape == (4,)
    assert matches(results.hard_assignments[0], np.zeros(3, dtype=np.int32))
    stats = results.per_class_stats[0]
    assert np.all(np.isneginf(stats.best_log_score_per_image)) and stats.rotation_posterior_sums.shape == (5,)
    assert matches(stats.log_evidence_per_image, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert results.best_pose_rotations[0].shape == (3, 3, 3) and results.best_pose_rotation_ids[0].dtype == np.int32
    assert results.per_class_noise is None

    device = _results()
    device.append_empty_class(
        mean=mean,
        class_log_evidence=np.zeros(3),
        noise_variance=None,
        class_index=0,
        n_classes=1,
        n_rot=2,
        host_accumulators=False,
    )
    assert isinstance(device.Ft_y[0], jnp.ndarray)


def test_class_results_expand_the_subset_to_the_full_image_axis():
    results = _results(return_best_pose_details=True)
    image_indices = np.asarray([0, 2], dtype=np.int64)
    stats_subset = make_relion_stats(
        log_evidence_per_image=np.asarray([0.5, 0.25], dtype=np.float32),
        best_log_score_per_image=np.asarray([-1.0, -2.0], dtype=np.float32),
        max_posterior_per_image=np.asarray([1.0, 1.0], dtype=np.float32),
        rotation_posterior_sums=np.asarray([2.0], dtype=np.float32),
    )
    best_pose = (
        np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([4, 5], dtype=np.int32),
    )
    results.append_class(
        image_indices=image_indices,
        Ft_y=jnp.ones(4, dtype=jnp.complex64),
        Ft_ctf=jnp.ones(4, dtype=jnp.float32),
        hard_full=np.asarray([7, 0, 9], dtype=np.int32),
        stats_subset=stats_subset,
        class_log_evidence=np.asarray([0.5, -9.0, 0.25]),
        noise=None,
        best_pose=best_pose,
        host_accumulators=True,
    )
    assert isinstance(results.Ft_y[0], np.ndarray) and isinstance(results.Ft_ctf[0], np.ndarray)
    assert matches(results.hard_assignments[0], [7, 0, 9])
    assert results.per_class_stats[0].best_log_score_per_image.shape == (3,)
    assert matches(results.best_pose_rotation_ids[0][image_indices], [4, 5])
    assert results.best_pose_rotations[0].shape == (3, 3, 3) and results.best_pose_translations[0].shape == (3, 2)


class _SubsetDataset:
    volume_shape = (2, 2, 2)

    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.n_units = self.n_images = int(self.indices.size)

    def subset(self, indices):
        return _SubsetDataset(self.indices[np.asarray(indices, dtype=np.int64)])


def _run_subset_pass(monkeypatch, *, pass2_kwargs=None):
    """The K=2 subset pass at given poses on three images that all go to class 0, with the sparse pass a stand-in.

    Returns the result, the owner's recorded calls (``("init", kwargs)``, ``("empty", host)``,
    ``("class", host)``, ``("assemble", host, returned)``) and the keyword arguments of every
    ``assemble_result`` call.
    """

    from relax.fine_pass import dispatch
    from relax.runtime.engine_tuning import EngineTuning
    from relax.runtime.scoring_overrides import ScoringOverrides
    from relax.types import SparsePass2Output

    calls, assembled = [], []

    class Owner(k_class_firstiter._PerClassSubsetResults):
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            super().__init__(**kwargs)

        def append_empty_class(self, **kwargs):
            calls.append(("empty", kwargs["host_accumulators"]))
            return super().append_empty_class(**kwargs)

        def append_class(self, **kwargs):
            calls.append(("class", kwargs["host_accumulators"]))
            return super().append_class(**kwargs)

        def assemble(self, *args, **kwargs):
            returned = super().assemble(*args, **kwargs)
            calls.append(("assemble", kwargs["host_accumulators"], returned))
            return returned

    real_assemble_result = k_class_firstiter.assemble_result

    def spy_assemble_result(*args, **kwargs):
        assembled.append(kwargs)
        return real_assemble_result(*args, **kwargs)

    def fake_sparse(dataset, volume, *_args, **_kwargs):
        n = int(dataset.n_units)
        stats = make_relion_stats(
            log_evidence_per_image=np.zeros(n, dtype=np.float32),
            best_log_score_per_image=np.zeros(n, dtype=np.float32),
            max_posterior_per_image=np.ones(n, dtype=np.float32),
            rotation_posterior_sums=np.zeros(1, dtype=np.float32),
        )
        return SparsePass2Output(
            jnp.zeros_like(volume), jnp.zeros_like(jnp.real(volume)), np.zeros(n, dtype=np.int32),
            np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0), np.zeros((n, 2), dtype=np.float32),
            np.zeros(n, dtype=np.int32), stats,
        )

    monkeypatch.setattr(k_class_firstiter, "_PerClassSubsetResults", Owner)
    monkeypatch.setattr(k_class_firstiter, "assemble_result", spy_assemble_result)
    monkeypatch.setattr(dispatch, "compute_pass2_stats_sparse", fake_sparse)
    n_images = 3
    result = k_class_firstiter.run_sparse_firstiter_global_winner_subset_pass2(
        _SubsetDataset(np.arange(n_images)),
        jnp.zeros((2, 8), dtype=jnp.complex64),
        jnp.ones(8, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.eye(3, dtype=np.float32)[None],
        None,
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        [[np.asarray([0], dtype=np.int32)] * n_images] * 2,
        "linear_interp",
        coarse_result=None,
        coarse_class_assignments=np.zeros(n_images, dtype=np.int32),
        n_rot_coarse=1,
        n_fine_trans=1,
        healpix_order=0,
        oversampling_order=1,
        accumulate_noise=False,
        return_best_pose_details=False,
        pass2_kwargs={"engine_tuning": EngineTuning(), "scoring_overrides": ScoringOverrides(), **(pass2_kwargs or {})},
    )
    return result, calls, assembled


def test_subset_pass_uses_the_owner(monkeypatch):
    """The subset pass collects its classes in the owner and returns the owner's assembled result: one owner, one
    empty class kept on the device, one scored class hosted, one assembly that alone calls ``assemble_result``,
    with the subset counts as the class posterior sums and the winner-take-all assembly."""

    result, calls, assembled = _run_subset_pass(monkeypatch)

    assert [call[0] for call in calls] == ["init", "class", "empty", "assemble"]
    assert calls[1] == ("class", True) and calls[2] == ("empty", False)
    assert calls[3][1] is True and calls[3][2] is result
    assert len(assembled) == 1
    np.testing.assert_array_equal(assembled[0]["class_posterior_sums_override"], np.asarray([3, 0], dtype=np.float64))
    assert assembled[0]["class_posterior_sums_override"].dtype == np.float64
    assert assembled[0]["firstiter_winner_take_all"] is True


def test_dtype_rules():
    assert k_class_firstiter._pose_dtype_from_kwargs({}) is np.float32
    assert k_class_firstiter._score_dtype_from_kwargs({}) is np.float32
    assert k_class_firstiter._pose_dtype_from_kwargs({"use_float64_projections": True}) is np.float64
    assert k_class_firstiter._score_dtype_from_kwargs({"use_float64_projections": True}) is np.float32
    assert k_class_firstiter._score_dtype_from_kwargs({"use_float64_scoring": True}) is np.float64


@pytest.mark.parametrize(
    "pass2_kwargs, pose_dtype, score_dtype",
    [({}, np.float32, np.float32), ({"use_float64_projections": True}, np.float64, np.float32),
     ({"use_float64_scoring": True}, np.float64, np.float64)],
)
def test_rules_and_assembly_are_not_repeated(monkeypatch, pass2_kwargs, pose_dtype, score_dtype):
    """One reader of the projection precision for every K-class route, and the subset pass takes its result dtypes
    from the dtype rules and assembles through the owner."""

    assert k_class.projection_float64_from_kwargs is k_class_inputs.projection_float64_from_kwargs
    assert k_class_firstiter.projection_float64_from_kwargs is k_class_inputs.projection_float64_from_kwargs
    _result, calls, assembled = _run_subset_pass(monkeypatch, pass2_kwargs=pass2_kwargs)
    owner = calls[0][1]
    assert owner["pose_dtype"] is pose_dtype and owner["score_dtype"] is score_dtype
    assert len(assembled) == 1 and assembled[0]["firstiter_winner_take_all"] is True
