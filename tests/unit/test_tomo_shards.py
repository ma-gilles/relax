"""Cutting a half into shards (relax.parallel.shards) and the shard of a subtomogram half's call
(relax.steps.tomo_shards): every particle is in one shard, heavy particles run alone, and a shard keeps the
half's coarse orientations."""

from __future__ import annotations

import numpy as np
import pytest

from relax.parallel.shards import cut_shards, heaviest_first, particle_costs
from relax.steps.half_inputs import HalfScoringData, HalfSet, TiltPrior
from relax.steps.tomo_half import TomoLocalRotations
from relax.steps.tomo_shards import TomoHalfJob

pytestmark = pytest.mark.unit


def _covered_once(shards, n):
    joined = np.concatenate(shards)
    assert np.array_equal(np.sort(joined), np.arange(n))


def test_heavy_particles_are_alone_and_the_rest_in_contiguous_blocks():
    costs = np.ones(40)
    costs[[5, 31]] = 100.0  # each above 1/(2*4) of the half's 238
    shards = cut_shards(costs, ranks=4)
    _covered_once(shards, 40)
    assert [5] in [list(s) for s in shards] and [31] in [list(s) for s in shards]
    blocks = [s for s in shards if s.size > 1]
    assert len(blocks) == 8
    for block in blocks:
        rest = np.setdiff1d(np.arange(block[0], block[-1] + 1), [5, 31])
        assert np.array_equal(block, rest)
    assert [int(s[0]) for s in shards] == sorted(int(s[0]) for s in shards)


def test_uniform_costs_give_equal_blocks():
    shards = cut_shards(np.ones(64), ranks=2)
    assert [s.size for s in shards] == [16, 16, 16, 16]


@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_fewer_particles_than_blocks(n):
    shards = cut_shards(np.arange(1.0, n + 1.0), ranks=4)
    _covered_once(shards, n)


def test_costs_and_order():
    assert np.array_equal(particle_costs(None, [3, 4]), [3.0, 4.0])
    assert np.array_equal(particle_costs([0, 10], [3, 4]), [3.0, 40.0])
    assert heaviest_first([("a", 1.0), ("b", 5.0), ("c", 1.0)]) == ["b", "a", "c"]
    with pytest.raises(ValueError):
        particle_costs([1, 2, 3], [1, 2])


class _Dataset:
    def __init__(self, units):
        self.units = np.asarray(units)
        self.n_units = self.units.size

    def subset(self, units):
        return _Dataset(self.units[units])


def test_a_shard_keeps_the_half_union_and_takes_its_particles_rows():
    n = 5
    half = HalfSet(
        index=1,
        dataset=_Dataset(np.arange(10, 15)),
        optics_group_ids=np.arange(n),
        rotation_eulers=np.arange(3 * n).reshape(n, 3).astype(float),
        translations=np.arange(3 * n).reshape(n, 3).astype(float) / 2,
        scale_corrections=np.linspace(1, 2, n),
    )
    union = np.array([3, 7, 9, 11])
    rotations = TomoLocalRotations(
        coarse_rotation_ids=union,
        unit_rotation_ids=tuple(np.array([u % 4]) for u in range(n)),
        unit_rotation_log_priors=tuple(np.array([-float(u)]) for u in range(n)),
    )
    reference = object()
    job = TomoHalfJob(
        data=HalfScoringData(
            particles=half,
            reference=reference,
            noise_variance=None,
            scale_group_ids=np.arange(n) * 2,
            tilt_prior=TiltPrior(sigma_deg=3.0, centre_deg=np.arange(n, dtype=float)),
        ),
        local_rotations=rotations,
        unit_seed_classes=np.zeros(n, dtype=np.int64),
        arguments={"symmetry": "C1", "class_log_priors": None},
    )
    units = np.array([4, 1])
    shard = job.shard(units)
    assert shard.n_units == 2 and np.array_equal(shard.data.particles.dataset.units, [14, 11])
    for name in ("optics_group_ids", "rotation_eulers", "translations", "scale_corrections"):
        assert np.array_equal(getattr(shard.data.particles, name), getattr(half, name)[units])
    assert shard.data.particles.image_corrections is None and shard.data.particles.index == 1
    assert np.array_equal(shard.data.scale_group_ids, [8, 2])
    assert np.array_equal(shard.data.tilt_prior.centre_deg, [4.0, 1.0])
    assert shard.data.reference is reference and shard.arguments is job.arguments
    assert shard.local_rotations.coarse_rotation_ids is union
    assert [int(r[0]) for r in shard.local_rotations.unit_rotation_ids] == [0, 1]
    assert [float(p[0]) for p in shard.local_rotations.unit_rotation_log_priors] == [-4.0, -1.0]
    assert np.array_equal(shard.unit_seed_classes, [0, 0])
    with pytest.raises(ValueError):
        job.shard([5])
    with pytest.raises(NotImplementedError):
        TomoHalfJob(job.data, None, None, {"class_log_priors": np.zeros(2)}).shard([0])


class _TiltDataset:
    """A half of particles ``units`` with ``images[u]`` tilt images each."""

    def __init__(self, units, images):
        self.units = np.asarray(units)
        self.n_units = self.units.size
        self.unit_image_offsets = np.concatenate([[0], np.cumsum(images)]).astype(np.int64)

    def subset(self, units):
        return _TiltDataset(self.units[units], np.diff(self.unit_image_offsets)[units])


_FAKE_GROUP = 4
_PLANS_SEEN = []


def _fake_tomo_scorer(data, *, local_rotations, unit_seed_classes, plan_use=None, **arguments):
    """Per-particle rows from the particle ids and integer-valued additive sums, so that any partition joins to
    the whole half's values exactly. Asked for its coarse plan, it gives groups of ``_FAKE_GROUP`` particles in
    reverse particle order (a frame-shared pass's order); a shard records the plan it was given."""

    from relax.scoring.tomo_coarse import CoarsePassBudget, CoarsePassPlan
    from relax.steps.score_outputs import HalfScoreResult
    from relax.types import make_noise_stats, make_relion_stats

    if plan_use is not None and plan_use.plan_only:
        n = data.particles.dataset.n_units
        return CoarsePassPlan(CoarsePassBudget(1, 1, 1), 3, 128, 1, 3, 8, _FAKE_GROUP, 1, 8, 8, np.arange(n)[::-1])
    _PLANS_SEEN.append(None if plan_use is None else plan_use.pinned)
    ids = data.particles.dataset.units.astype(np.int64)
    n_rot = local_rotations.coarse_rotation_ids.size
    one_hot = np.zeros((ids.size, 16), dtype=np.float32)
    one_hot[np.arange(ids.size), ids % 16] = ids
    return HalfScoreResult(
        ha=(ids * 3).astype(np.int32),
        Ft_y=jnp_like(one_hot.sum(axis=0).astype(np.complex64)),
        Ft_ctf=one_hot.sum(axis=0),
        em_stats=make_relion_stats(
            log_evidence_per_image=-ids.astype(np.float64),
            best_log_score_per_image=ids.astype(np.float64),
            max_posterior_per_image=1.0 / (1.0 + ids),
            rotation_posterior_sums=np.full(n_rot, float(ids.size)),
            host_arrays=True,
        ),
        noise_stats=make_noise_stats(
            wsum_sigma2_noise=np.full(3, float(ids.sum())),
            wsum_img_power=np.full(3, 2.0 * ids.size),
            wsum_sigma2_offset=float(ids.sum()),
            sumw=float(ids.size),
            host_arrays=True,
        ),
        best_pose_translations=np.stack([ids, -ids, 2 * ids], axis=1).astype(np.float32),
        significant_counts=(ids % 5 + 1).astype(np.int32),
        mstep_full_half_axis=0,
        mstep_accumulator_shape=(4, 4, 3),
    )


def jnp_like(array):
    import jax.numpy as jnp

    return jnp.asarray(array)


def _fake_rank(connection, objects):
    """A rank's shard loop (``serve_remote_halves``'s shard messages) on a pipe."""

    from relax.parallel import channel
    from relax.refinement import remote_halves

    jobs = {}
    while True:
        try:
            task = channel.receive(connection, objects, peer="rank 0")
        except channel.RankFailed:
            return
        if isinstance(task, remote_halves.ShardJob):
            jobs[task.key] = task.job
        elif isinstance(task, remote_halves.ReleaseShardJobs):
            for key in task.keys:
                jobs.pop(key)
        else:
            result = jobs[task.key].shard(task.units).score()
            channel.send(connection, remote_halves.HalfReply(result, [], [], [], 0.0), objects)


@pytest.mark.parametrize("ranks", [3, 4])
def test_a_pool_of_ranks_scores_both_halves_as_one_call_each(monkeypatch, ranks):
    import threading
    from multiprocessing import Pipe

    from relax.refinement import remote_halves
    from relax.steps import tomo_scoring
    from relax.steps.tomo_shards import DeferredTomoHalf

    monkeypatch.setattr(tomo_scoring, "score_tomo_half_in_loop", _fake_tomo_scorer)
    rng = np.random.default_rng(0)
    datasets = [_TiltDataset(np.arange(23) + 100 * k, rng.integers(1, 40, 23)) for k in range(2)]
    objects = {"dataset_half1": datasets[0], "dataset_half2": datasets[1], "local_profile_history": []}
    rotations = TomoLocalRotations(np.arange(6), tuple(np.array([u % 6]) for u in range(23)), tuple(np.zeros(1) for _ in range(23)))
    jobs = {
        k: TomoHalfJob(
            data=HalfScoringData(particles=HalfSet(index=k, dataset=datasets[k]), reference=None, noise_variance=None),
            local_rotations=rotations,
            unit_seed_classes=None,
            arguments={},
        )
        for k in range(2)
    }
    ends = {rank: Pipe() for rank in range(1, ranks)}
    threads = [threading.Thread(target=_fake_rank, args=(theirs, objects), daemon=True) for _, theirs in ends.values()]
    for thread in threads:
        thread.start()
    pool = remote_halves.ShardPool({rank: ours for rank, (ours, _) in ends.items()}, objects)
    for _ in range(2):  # the second call cuts by the first's significant counts
        done = pool.score_halves({k: DeferredTomoHalf(jobs[k], lambda result, k=k: (k, result)) for k in range(2)})
        for k in range(2):
            assert done[k][0] == k
            ours, whole = done[k][1], jobs[k].score()
            for name in ("ha", "significant_counts", "best_pose_translations"):
                np.testing.assert_array_equal(getattr(ours, name), getattr(whole, name))
            for name in ("log_evidence_per_image", "max_posterior_per_image", "rotation_posterior_sums"):
                np.testing.assert_array_equal(getattr(ours.em_stats, name), getattr(whole.em_stats, name))
            np.testing.assert_array_equal(np.asarray(ours.Ft_y), np.asarray(whole.Ft_y))
            np.testing.assert_array_equal(ours.Ft_ctf, whole.Ft_ctf)
            assert ours.noise_stats.sumw == whole.noise_stats.sumw == 23.0
    assert pool._significant_counts[1].size == 23
    # The shards scored with their half's plan (without its frame order); the whole calls above with none.
    pinned = [plan for plan in _PLANS_SEEN if plan is not None]
    assert pinned and all(plan.frame_order is None and plan.flush_size == _FAKE_GROUP for plan in pinned)
    _PLANS_SEEN.clear()
    for ours, _ in ends.values():
        ours.close()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()


def test_a_pool_reports_a_failed_rank(monkeypatch):
    from multiprocessing import Pipe

    from relax.parallel import channel
    from relax.refinement import remote_halves
    from relax.steps import tomo_scoring
    from relax.steps.tomo_shards import DeferredTomoHalf

    monkeypatch.setattr(tomo_scoring, "score_tomo_half_in_loop", _fake_tomo_scorer)
    dataset = _TiltDataset(np.arange(8), np.ones(8, dtype=int))
    objects = {"dataset_half1": dataset, "dataset_half2": None, "local_profile_history": []}
    ours, theirs = Pipe()
    theirs.close()
    pool = remote_halves.ShardPool({1: ours, 2: Pipe()[0]}, objects)
    job = TomoHalfJob(
        data=HalfScoringData(particles=HalfSet(index=0, dataset=dataset), reference=None, noise_variance=None),
        local_rotations=TomoLocalRotations(np.arange(2), tuple(np.array([0]) for _ in range(8)), tuple(np.zeros(1) for _ in range(8))),
        unit_seed_classes=None,
        arguments={},
    )
    with pytest.raises((channel.RankFailed, OSError, BrokenPipeError)):
        pool.score_halves({0: DeferredTomoHalf(job, lambda result: result)})


def test_group_shards_keep_whole_groups_and_put_heavy_groups_alone():
    from relax.parallel.shards import cut_group_shards

    costs = np.ones(30)
    costs[13] = 50.0
    groups = [np.arange(30)[::-1][i : i + 4] for i in range(0, 30, 4)]
    shards = cut_group_shards(costs, groups, ranks=2)
    _covered_once(shards, 30)
    as_sets = [set(int(u) for u in shard) for shard in shards]
    for group in groups:
        assert sum(set(int(u) for u in group) <= shard for shard in as_sets) == 1
    heavy_group = next(group for group in groups if 13 in group)
    assert any(np.array_equal(shard, heavy_group) for shard in shards)
    assert len(shards) <= 1 + 4
    with pytest.raises(ValueError):
        cut_group_shards(costs, groups[:-1], ranks=2)
