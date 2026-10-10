"""Per-particle training membership is diagnostic, indexed and explicitly stale."""

import csv
import dataclasses
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.ppca_initial_class3d import checkpoint, expectation, iteration_loop
from relax.ppca_initial_class3d.membership import FIELDS, Membership
from relax.ppca_initial_class3d.tests.test_mixture_controller import spa_data, tiny_config, tomo_data
from relax.ppca_initial_class3d.tests.test_mixture_state_validation import _valid_state

pytestmark = pytest.mark.unit


def test_scatter_preserves_unseen_and_stale_original_particle_ids():
    membership = Membership.empty(np.asarray([90, 10, 70, 30]), 2)
    membership.update([90, 10], np.asarray([[.2, .8], [.9, .1]], np.float32),
                      iteration=1, model_iteration=0)
    membership.update([70, 90], np.asarray([[.4, .6], [.7, .3]], np.float32),
                      iteration=3, model_iteration=2)
    membership.validate(n_particles=4, n_classes=2, iteration=3)
    arrays = membership.arrays(3)
    np.testing.assert_array_equal(arrays["particle_ids"], [10, 30, 70, 90])
    assert_matches(arrays["class_probabilities"], np.asarray(
        [[.9, .1], [np.nan, np.nan], [.4, .6], [.7, .3]], np.float32))
    np.testing.assert_array_equal(arrays["class_labels"], [0, -1, 1, 0])
    np.testing.assert_array_equal(arrays["last_evaluated_iteration"], [1, -1, 3, 3])
    np.testing.assert_array_equal(arrays["model_iteration"], [0, -1, 2, 2])
    np.testing.assert_array_equal(arrays["evaluated"], [True, False, True, True])
    np.testing.assert_array_equal(arrays["evaluated_in_checkpoint_iteration"], [False, False, True, True])


def test_label_is_marginal_class_not_joint_class_pose_map():
    # Joint pose masses: class 0 has one .4 pose; class 1 has two .3 poses.
    # The largest single pose belongs to class 0, but class 1 owns .6 total mass.
    joint_pose_mass = np.asarray([[.4, 0], [.3, .3]], np.float32)
    membership = Membership.empty(np.asarray([12]), 2)
    membership.update([12], joint_pose_mass.sum(axis=1)[None], iteration=1, model_iteration=0)
    assert np.unravel_index(joint_pose_mass.argmax(), joint_pose_mass.shape)[0] == 0
    np.testing.assert_array_equal(membership.arrays(1)["class_labels"], [1])


@pytest.mark.parametrize("ids", [[1, 1], [-1, 2], [1.5, 2], [[1, 2]]])
def test_empty_rejects_invalid_particle_ids(ids):
    with pytest.raises(ValueError, match="particle IDs"):
        Membership.empty(np.asarray(ids), 2)


@pytest.mark.parametrize("ids,probabilities", [
    ([1, 1], [[.5, .5], [.5, .5]]),
    ([9], [[.5, .5]]),
    ([0], [[.5, .5]]),
    ([1, 3], [[.5, .5]]),
])
def test_update_rejects_duplicate_unknown_or_missing_probability_rows(ids, probabilities):
    membership = Membership.empty(np.asarray([1, 3]), 2)
    with pytest.raises(ValueError):
        membership.update(ids, np.asarray(probabilities, np.float32), iteration=1, model_iteration=0)
    assert np.isnan(membership.class_probabilities).all()
    np.testing.assert_array_equal(membership.last_evaluated_iteration, [-1, -1])


@pytest.mark.parametrize("probabilities", [
    [[np.nan, .5]], [[np.inf, 0]], [[-.1, 1.1]], [[0, 0]], [[.4, .4]], [[1.1, 0]],
])
def test_update_rejects_malformed_probabilities(probabilities):
    membership = Membership.empty(np.asarray([1]), 2)
    with pytest.raises(ValueError, match="probabilities"):
        membership.update([1], np.asarray(probabilities, np.float32), iteration=1, model_iteration=0)


@pytest.mark.parametrize("iteration,model_iteration", [(0, -1), (2, 2), (2, 0)])
def test_update_rejects_wrong_model_timestamp(iteration, model_iteration):
    membership = Membership.empty(np.asarray([1]), 2)
    with pytest.raises(ValueError, match="pre-update model"):
        membership.update([1], np.asarray([[.5, .5]], np.float32),
                          iteration=iteration, model_iteration=model_iteration)


@pytest.mark.parametrize("iteration", [1, 2])
def test_update_rejects_repeated_or_out_of_order_particle(iteration):
    membership = Membership.empty(np.asarray([1]), 2)
    probabilities = np.asarray([[.25, .75]], np.float32)
    membership.update([1], probabilities, iteration=2, model_iteration=1)
    with pytest.raises(ValueError, match="Duplicate or out-of-order"):
        membership.update([1], probabilities, iteration=iteration, model_iteration=iteration - 1)
    assert_matches(membership.class_probabilities, probabilities)


def test_validate_rejects_wrong_count_dtype_and_unvisited_probabilities():
    membership = Membership.empty(np.arange(3), 2)
    with pytest.raises(ValueError, match="particle set"):
        membership.validate(n_particles=4, n_classes=2, iteration=0)
    with pytest.raises(TypeError, match="float32"):
        dataclasses.replace(membership, class_probabilities=membership.class_probabilities.astype(np.float64)).validate(
            n_particles=3, n_classes=2, iteration=0)
    membership.class_probabilities[1] = [.5, .5]
    with pytest.raises(ValueError, match="Unvisited"):
        membership.validate(n_particles=3, n_classes=2, iteration=0)


def _state_with_membership():
    state, config = _valid_state()
    state.membership = Membership.empty(np.arange(len(state.order)), config.n_classes)
    state.membership.update([6, 1], np.asarray([[.8, .2], [.1, .9]], np.float32),
                            iteration=2, model_iteration=1)
    return state, config


def test_checkpoint_roundtrip_and_atomic_sidecar_views(tmp_path):
    state, config = _state_with_membership()
    path = tmp_path / "checkpoint_0003.npz"
    checkpoint.save(path, state, config, {"case": "membership"})
    restored = checkpoint.load(path, config, {"case": "membership"})
    for name, values in state.membership.arrays(3).items():
        assert_matches(restored.membership.arrays(3)[name], values)
    assert_matches(restored.theta, state.theta)
    assert_matches(restored.noise, state.noise)
    assert_matches(restored.class_prior, state.class_prior)
    with np.load(tmp_path / "memberships/checkpoint_0003.npz", allow_pickle=False) as sidecar:
        metadata = json.loads(str(sidecar["metadata"]))
        assert metadata["checkpoint_iteration"] == 3
        assert "NOT all-particle inference" in metadata["scope"]
        for name, values in state.membership.arrays(3).items():
            assert_matches(sidecar[name], values)
    with (tmp_path / "memberships/checkpoint_0003.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == len(state.order)
    assert [int(row["particle_ids"]) for row in rows] == list(range(8))
    assert [int(row["class_labels"]) for row in rows] == [-1, 1, -1, -1, -1, -1, 0, -1]
    assert_matches(np.asarray([[row[f"class_probability_{k}"] for k in range(2)] for row in rows], np.float32),
                   state.membership.class_probabilities)
    assert sorted(p.name for p in tmp_path.glob("checkpoint_*.npz")) == ["checkpoint_0003.npz"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_legacy_v1_checkpoint_remains_loadable_without_fabricated_membership(tmp_path):
    state, config = _valid_state()
    assert state.membership is None
    path = tmp_path / "checkpoint_0003.npz"
    checkpoint.save(path, state, config, {"case": "legacy"})
    with np.load(path, allow_pickle=False) as saved:
        assert not any(name.startswith("membership_") for name in saved.files)
        assert "membership_schema" not in json.loads(str(saved["metadata"]))
    restored = checkpoint.load(path, config, {"case": "legacy"})
    assert restored.membership is None
    assert_matches(restored.theta, state.theta)
    assert not (tmp_path / "memberships").exists()


@pytest.mark.parametrize("corruption", ["missing-field", "missing-schema", "unknown-schema", "shape",
                                      "dtype", "seen-nan", "future", "wrong-model", "unseen-finite"])
def test_load_rejects_corrupt_new_membership_instead_of_treating_it_as_legacy(tmp_path, corruption):
    state, config = _state_with_membership()
    path = tmp_path / "checkpoint_0003.npz"
    identity = {"case": "corrupt-membership"}
    checkpoint.save(path, state, config, identity)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if corruption == "missing-field":
        del arrays[f"membership_{FIELDS[0]}"]
    elif corruption in ("missing-schema", "unknown-schema"):
        metadata = json.loads(str(arrays["metadata"]))
        if corruption == "missing-schema":
            del metadata["membership_schema"]
        else:
            metadata["membership_schema"] = "unknown"
        arrays["metadata"] = json.dumps(metadata)
    elif corruption == "shape":
        arrays["membership_class_probabilities"] = arrays["membership_class_probabilities"][:, :1]
    elif corruption == "dtype":
        arrays["membership_class_probabilities"] = arrays["membership_class_probabilities"].astype(np.float64)
    elif corruption == "seen-nan":
        arrays["membership_class_probabilities"][1] = np.nan
    elif corruption == "future":
        arrays["membership_last_evaluated_iteration"][1] = 4
        arrays["membership_model_iteration"][1] = 3
    elif corruption == "wrong-model":
        arrays["membership_model_iteration"][1] = 2
    else:
        arrays["membership_class_probabilities"][0] = [.5, .5]
    np.savez(path, **arrays)
    with pytest.raises((ValueError, TypeError), match="[Mm]embership"):
        checkpoint.load(path, config, identity)


def _synthetic_tile(ids, probabilities):
    probabilities = np.asarray(probabilities, np.float32)
    components = tuple(SimpleNamespace(
        lhs_tri=jnp.ones((1, 1), jnp.float32), residual_gradient=jnp.ones((1, 1), jnp.complex64),
        diagnostics={"rotation_mass": np.asarray([probabilities[:, k].sum()], np.float32),
                     "offset_second_sum_px2": 0.0},
    ) for k in range(probabilities.shape[1]))
    return SimpleNamespace(original_image_ids=np.asarray(ids), class_probabilities=probabilities,
                           component_mass=probabilities.sum(axis=0), statistics=components,
                           max_posterior_per_image=np.max(probabilities, axis=1), log_likelihood=0.,
                           residual_num=jnp.ones(1, jnp.float32), residual_den=jnp.ones(1, jnp.float32),
                           best_rotation_idx=np.zeros(probabilities.shape, np.int32),
                           best_translation_idx=np.zeros(probabilities.shape, np.int32))


@pytest.mark.parametrize("actual_ids", [[90, 70], [70, 70], [90, 10]],
                         ids=["reordered-correct", "duplicate-missing", "wrong-group"])
def test_expectation_preserves_membership_and_checks_exact_group_id_coverage(monkeypatch, actual_ids):
    original_ids = np.asarray([70, 10, 90, 30])
    data = SimpleNamespace(n_images=4, original_image_indices_from_local=lambda ids: original_ids[ids])
    tile = _synthetic_tile(actual_ids, [[.4, .6], [.9, .1]])
    calls = []

    def tiles(*args, **kwargs):
        calls.append(kwargs["moments"])
        yield 0, 0, tile

    monkeypatch.setattr(expectation, "mixture_tiles", tiles)
    monkeypatch.setattr(expectation, "PosteriorDiagnostics", lambda: SimpleNamespace(add=lambda _: None))
    arguments = (data, None, SimpleNamespace(n_classes=2, oversampling=0, oversampling_start=None, stages=((1, 2, 0),)),
                 [np.asarray([0, 2])], 1,
                 SimpleNamespace(rotations=np.zeros((1, 3, 3), np.float32), eulers=np.zeros((1, 3)),
                                 translations=np.zeros((1, 3), np.float32)))
    if actual_ids != [90, 70]:
        with pytest.raises(ValueError):
            expectation.expectation_groups(*arguments)
    else:
        result = expectation.expectation_groups(*arguments)[0]
        ids, probabilities, _, _ = result.membership_tiles[0]
        np.testing.assert_array_equal(ids, actual_ids)
        assert_matches(probabilities, tile.class_probabilities)
        assert not np.shares_memory(probabilities, tile.class_probabilities)
        assert not np.shares_memory(ids, tile.original_image_ids)
        assert result.count == 2
    assert calls == [True]


@pytest.mark.parametrize("tomography", [False, True], ids=["spa", "two-noise-group-et"])
def test_real_k2_checkpoint_membership_is_preupdate_and_survives_resume(tmp_path, monkeypatch, tomography):
    data = tomo_data() if tomography else spa_data(tmp_path / "inputs")
    config = dataclasses.replace(tiny_config(2), stages=((1, 2, 0), (2, 3, 0)))
    output, identity = tmp_path / "run", {"case": "membership", "tomography": tomography}
    records = {}
    original_expectation = iteration_loop.expectation_groups

    def record_expectation(dataset, state, config, groups, iteration, grid, **kwargs):
        assert state.iteration == iteration - 1
        stats = original_expectation(dataset, state, config, groups, iteration, grid, **kwargs)
        expected = Membership.empty(np.arange(dataset.n_images), config.n_classes)
        for half in stats:
            for ids, probabilities, eulers, shifts in half.membership_tiles:
                expected.update(ids, probabilities, iteration=iteration, model_iteration=state.iteration,
                                eulers_deg=eulers, translations_px=shifts)
        records[iteration] = expected.arrays(iteration)
        return stats

    monkeypatch.setattr(iteration_loop, "expectation_groups", record_expectation)
    stopped = iteration_loop.run(data, config, output, identity, 6., stop_after=1)
    assert stopped.iteration == 1
    with np.load(output / "memberships/checkpoint_0000.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["class_labels"], np.full(data.n_images, -1))
        assert np.isnan(saved["class_probabilities"]).all()
    restored = checkpoint.load(output / "checkpoint_0001.npz", config, identity)
    for name, values in records[1].items():
        assert_matches(restored.membership.arrays(1)[name], values)
    assert not (output / "assignments.npz").exists()

    def refuse_initialization(*args, **kwargs):
        pytest.fail("Resume must not initialize another model")

    monkeypatch.setattr(iteration_loop, "initialize_state", refuse_initialization)
    final = iteration_loop.run(data, config, output, identity, 6., resume=output / "checkpoint_0001.npz")
    assert final.iteration == 2
    assert sorted(records) == [1, 2]
    for iteration in (1, 2):
        with np.load(output / f"memberships/checkpoint_{iteration:04d}.npz", allow_pickle=False) as saved:
            for name, values in records[iteration].items():
                assert_matches(saved[name], values)
    with np.load(output / "assignments.npz", allow_pickle=False) as assignments:
        np.testing.assert_array_equal(assignments["particle_ids"], np.arange(data.n_images))
        assert assignments["class_probabilities"].shape == (data.n_images, 2)
        assert_matches(assignments["class_probabilities"].sum(axis=1), np.ones(data.n_images, np.float32))
    assert sorted(p.name for p in output.glob("checkpoint_*.npz")) == [
        "checkpoint_0000.npz", "checkpoint_0001.npz", "checkpoint_0002.npz"]
    if tomography:
        assert data.n_noise_groups == 2
        assert data.n_images == 4 < data.image_offsets[-1]
