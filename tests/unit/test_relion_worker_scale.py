import json
import logging
import shutil
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.diagnostics.relion_replay import _apply_replay_correction_overrides
from relax.helpers.types import NoiseStats
from relax.refinement.half_inputs import initialize_halfsets
from relax.relion.relion_normalization import (
    update_relion_norm_scale_corrections,
)
from relax.relion.relion_worker_scale import (
    RELION_SCALE_REDUCTION_ALL_GROUPS,
    RELION_SCALE_REDUCTION_OPTICS_PREFIX,
    RelionDispatchSchedule,
    RelionFollowerScaleReplay,
    RelionFollowerScaleSetup,
    _finalize_relion_follower_scale_replay_telemetry,
    _remap_relion_follower_runtime_inputs,
    _require_relion_follower_owners,
    _validate_coupled_relion_restart_state,
    load_relion_dispatch_schedule,
    load_relion_follower_scale_replay,
    make_relion_dispatch_schedule_from_chunks,
    make_relion_follower_scale_state,
    prepare_follower_topology,
    relion_class3d_follower_owners_from_schedule,
    relion_oracle_id,
    relion_oracle_manifest_sha256,
    relion_ordered_particle_sha256,
    relion_rank1_serialized_scales,
    relion_scale_reduction_mode_from_command,
    relion_scale_reduction_mode_from_optimiser_star,
    relion_worker_group_ids,
    select_relion_follower_scales,
    update_relion_follower_scales,
    validate_relion_follower_scale_replay,
    validate_relion_follower_scale_replay_application,
    validate_relion_follower_scale_start,
    verify_relion_dispatch_schedule_oracle,
)

_ORACLE_MANIFEST = "0" * 64
_PARTICLE_ORDER = "1" * 64
_ORACLE_ID = relion_oracle_id(
    manifest_sha256=_ORACLE_MANIFEST,
    particle_order_sha256=_PARTICLE_ORDER,
)
_ORACLE_KWARGS = {
    "oracle_id": _ORACLE_ID,
    "oracle_manifest_sha256": _ORACLE_MANIFEST,
    "oracle_artifact_paths": (
        "dispatch.tsv",
        "dispatch.tsv.recovar_schedule.json",
        "run_it000_data.star",
    ),
    "particle_order_sha256": _PARTICLE_ORDER,
    "particle_star_relative_path": "run_it000_data.star",
    "dispatch_log_relative_path": "dispatch.tsv",
}
_REPLAY_KWARGS = {
    "oracle_id": _ORACLE_ID,
    "schema_version": 1,
    "boundary": "numbered_pre_score",
    "source_artifact_relative_paths": ("rank1_post.tsv", "rank2_post.tsv"),
}


def test_prepared_topology_keeps_captured_particle_identity_and_iteration_coverage(tmp_path):
    schedule = RelionDispatchSchedule(
        relion_iterations=np.asarray([1, 2], dtype=np.int64),
        owner_by_sorted_position=np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
        original_particle_id_by_sorted_position=np.asarray([[2, 0, 1], [1, 2, 0]], dtype=np.int64),
        n_followers=2,
        pool_size=1,
        random_seed=9,
        **_ORACLE_KWARGS,
        source="unit capture",
    )
    groups = SimpleNamespace(
        group_ids_per_half=(np.asarray([2, 0, 1]), np.empty(0, dtype=np.int64)),
        particle_ids_per_half=(np.asarray([0, 2, 1]), np.empty(0, dtype=np.int64)),
        n_groups=3,
        n_optics_groups=1,
        source="unit particle table",
    )
    (tmp_path / "run_it000_optimiser.star").write_text(
        "# --o run --gpu --dont_combine_weights_via_disc\ndata_optimiser_general\n"
    )
    kwargs = dict(strict_replay=True, replay_path=None, oracle_dir=tmp_path,
                  random_seed=9, init_relion_iteration=0, max_iter=2)
    topology = prepare_follower_topology(None, schedule, groups, **kwargs)
    assert topology.n_followers == 2
    assert topology.replay is None
    assert topology.reduction_mode == RELION_SCALE_REDUCTION_OPTICS_PREFIX
    assert list(topology.owners_by_iteration) == [1, 2]
    np.testing.assert_array_equal(topology.owners_by_iteration[1][0], [1, 0, 0])
    np.testing.assert_array_equal(topology.owners_by_iteration[2][0], [1, 0, 1])
    assert topology.owners_by_iteration[1][1].shape == (0,)
    assert topology.owners_by_iteration[1][1].dtype == np.int64
    with pytest.raises(SystemExit, match="requested numbered iterations:.*3"):
        prepare_follower_topology(None, schedule, groups, **{**kwargs, "max_iter": 3})
    with pytest.raises(SystemExit, match="leader-serialized STAR"):
        prepare_follower_topology(None, schedule, groups, **{**kwargs, "init_relion_iteration": 1})


@pytest.mark.parametrize("requested_followers", [None, 0])
def test_prepared_topology_without_mpi_needs_no_oracle(requested_followers):
    topology = prepare_follower_topology(
        requested_followers, None, None, strict_replay=False,
        replay_path=None, oracle_dir=None, random_seed=9,
        init_relion_iteration=0, max_iter=2,
    )
    assert topology.n_followers == 0
    assert topology.replay is topology.reduction_mode is topology.owners_by_iteration is None


@pytest.mark.parametrize(
    "requested_followers, strict_replay, replay_path, message",
    [
        (-1, False, None, "non-negative"),
        (2, False, None, "strict K>1"),
        (2, True, None, "requires --relion-dispatch-schedule"),
        (None, True, None, "dynamic MPI work queue"),
        (0, True, "replay.npz", "requires strict K>1 RELION follower topology"),
    ],
)
def test_prepared_topology_rejects_unsupported_or_incomplete_mpi_state(
    requested_followers, strict_replay, replay_path, message,
):
    with pytest.raises(SystemExit, match=message):
        prepare_follower_topology(
            requested_followers, None, None, strict_replay=strict_replay,
            replay_path=replay_path, oracle_dir=None, random_seed=9,
            init_relion_iteration=0, max_iter=2,
        )


def test_follower_scale_continuation_from_leader_star_fails_closed():
    validate_relion_follower_scale_start(n_followers=2, init_relion_iteration=0)

    with np.testing.assert_raises_regex(ValueError, "leader-serialized STAR"):
        validate_relion_follower_scale_start(n_followers=2, init_relion_iteration=3)


def test_serialized_replay_preserves_live_follower_scale_between_iterations():
    state = make_relion_follower_scale_state(
        n_followers=2,
        group_counts=[1, 1],
        n_optics_groups=1,
    )
    state = type(state)(
        scales=np.asarray([[1.0, 1.3], [1.0, 0.9]]),
        group_counts=state.group_counts,
        n_optics_groups=state.n_optics_groups,
    )
    groups = np.asarray([1, 1])
    owners = np.asarray([0, 1])
    live_scales = select_relion_follower_scales(
        state,
        group_ids=groups,
        follower_owners=owners,
    ).astype(np.float32)
    half_inputs = initialize_halfsets(
        (None, None),
        previous_best_translations=[None, None],
        previous_best_rotation_eulers=[None, None],
        image_corrections=[live_scales.copy(), np.zeros(0, dtype=np.float32)],
        scale_corrections=[live_scales.copy(), np.zeros(0, dtype=np.float32)],
        group_ids=[groups, np.zeros(0, dtype=np.int64)],
        group_count=[2, 2],
    )

    applied = _apply_replay_correction_overrides(
        relion_half_inputs=half_inputs,
        replay_override={
            "image_corrections": [np.asarray([1.3, 1.3]), np.zeros(0)],
            "serialized_scale_corrections": [
                np.asarray([1.3, 1.3]),
                np.zeros(0),
            ],
        },
    )

    assert applied == ["image_corrections", "serialized_scale_corrections"]
    assert_matches(half_inputs[0].scale_corrections, live_scales)
    np.testing.assert_allclose(half_inputs[0].image_corrections, live_scales)


def test_dynamic_dispatch_chunks_cover_every_sorted_position_once():
    schedule = make_relion_dispatch_schedule_from_chunks(
        relion_iterations=[1, 2],
        chunk_iterations=[1, 1, 1, 2, 2, 2],
        chunk_first=[0, 3, 6, 0, 3, 6],
        chunk_last=[2, 5, 6, 2, 5, 6],
        chunk_ranks=[1, 2, 1, 2, 1, 2],
        n_particles=7,
        original_particle_id_by_sorted_position=np.tile(np.arange(7), (2, 1)),
        n_followers=2,
        pool_size=3,
        random_seed=2802,
        source="unit capture",
        **_ORACLE_KWARGS,
    )

    # Ranks in this fixture are 1 and 2 (zero-based owners 0 and 1); the
    # runtime owner of the same sorted chunk changes between iterations.
    assert_matches(schedule.owner_by_sorted_position[0], [0, 0, 0, 1, 1, 1, 0])
    assert_matches(schedule.owner_by_sorted_position[1], [1, 1, 1, 0, 0, 0, 1])


def test_dynamic_dispatch_rejects_overlap_and_missing_positions():
    kwargs = dict(
        relion_iterations=[1],
        chunk_iterations=[1, 1],
        chunk_first=[0, 2],
        chunk_last=[2, 4],
        chunk_ranks=[1, 2],
        n_particles=6,
        original_particle_id_by_sorted_position=np.arange(6)[None, :],
        n_followers=2,
        pool_size=3,
        random_seed=2802,
        source="bad capture",
        **_ORACLE_KWARGS,
    )
    with np.testing.assert_raises_regex(ValueError, "overlapping"):
        make_relion_dispatch_schedule_from_chunks(**kwargs)

    kwargs["chunk_first"] = [0, 3]
    kwargs["chunk_last"] = [1, 4]
    with np.testing.assert_raises_regex(ValueError, "undispatched"):
        make_relion_dispatch_schedule_from_chunks(**kwargs)


@pytest.mark.requires_relion_bind
def test_class3d_shuffle_is_always_original_seed_plus_one():
    particle_ids = np.arange(10_000, dtype=np.int64)

    from relax.relion_bind import _relion_bind_core as bind

    sorted_particle_ids = np.asarray(
        bind.vdam_randomise_particles_order(particle_ids.size, 2802 + 1),
        dtype=np.int64,
    )

    # Captured independently from all three ranks in the K=4 fixture.
    assert int(np.flatnonzero(sorted_particle_ids == 5989)[0]) == 626
    assert_matches(np.sort(sorted_particle_ids), particle_ids)


def test_captured_schedule_maps_dynamic_owners_to_recovar_image_order():
    n_particles = 10_000
    owners = (np.arange(n_particles, dtype=np.int64) // 3) % 2
    sorted_particle_ids = np.roll(np.arange(n_particles, dtype=np.int64), 17)
    schedule = RelionDispatchSchedule(
        relion_iterations=np.asarray([1, 2]),
        owner_by_sorted_position=np.stack([owners, 1 - owners]),
        original_particle_id_by_sorted_position=np.stack(
            [sorted_particle_ids, sorted_particle_ids]
        ),
        n_followers=2,
        pool_size=3,
        random_seed=2802,
        **_ORACLE_KWARGS,
        source="unit capture",
    )
    image_particle_ids = np.arange(n_particles, dtype=np.int64)

    iter1 = relion_class3d_follower_owners_from_schedule(
        schedule,
        particle_ids_by_image=image_particle_ids,
        random_seed=2802,
        relion_iteration=1,
    )
    iter2 = relion_class3d_follower_owners_from_schedule(
        schedule,
        particle_ids_by_image=image_particle_ids,
        random_seed=2802,
        relion_iteration=2,
    )

    sorted_position = int(np.flatnonzero(sorted_particle_ids == 5989)[0])
    # The persisted identity map, not a regenerated shuffle, resolves owners.
    assert iter1[5989] == owners[sorted_position]
    assert iter2[5989] == 1 - owners[sorted_position]
    assert_matches(iter2, 1 - iter1)


def test_dispatch_schedule_npz_loads_and_fails_closed_on_seed_mismatch(tmp_path):
    path = tmp_path / "dispatch.npz"
    np.savez(
        path,
        schema_version=np.int64(3),
        relion_iterations=np.asarray([1]),
        owner_by_sorted_position=np.asarray([[0, 0, 1, 1]], dtype=np.int64),
        original_particle_id_by_sorted_position=np.arange(4, dtype=np.int64)[None, :],
        n_followers=np.int64(2),
        pool_size=np.int64(2),
        random_seed=np.int64(9),
        oracle_id=np.asarray(_ORACLE_ID),
        oracle_manifest_sha256=np.asarray(_ORACLE_MANIFEST),
        oracle_artifact_paths=np.asarray(
            ["dispatch.tsv", "dispatch.tsv.recovar_schedule.json", "run_it000_data.star"]
        ),
        particle_order_sha256=np.asarray(_PARTICLE_ORDER),
        particle_star_relative_path=np.asarray("run_it000_data.star"),
        dispatch_log_relative_path=np.asarray("dispatch.tsv"),
        source=np.asarray("same oracle run"),
    )
    schedule = load_relion_dispatch_schedule(path)
    assert schedule.source == "same oracle run"
    with np.testing.assert_raises_regex(ValueError, "random_seed"):
        relion_class3d_follower_owners_from_schedule(
            schedule,
            particle_ids_by_image=np.arange(4),
            random_seed=10,
            relion_iteration=1,
        )


def test_dispatch_oracle_identity_is_portable_and_content_bound(tmp_path):
    import pandas as pd
    import starfile

    oracle = tmp_path / "oracle"
    oracle.mkdir()
    particles = pd.DataFrame(
        {
            "rlnImageName": ["2@particles.mrcs", "1@particles.mrcs"],
            "rlnOpticsGroup": [1, 1],
            "rlnRandomSubset": [1, 2],
            "rlnGroupNumber": [2, 1],
        }
    )
    starfile.write({"particles": particles}, oracle / "run_it000_data.star")
    (oracle / "run_it000_model.star").write_text("model-state\n")
    (oracle / "dispatch.tsv").write_text("2 1 1 0 0\n2 1 1 1 1\n")
    (oracle / "dispatch.tsv.recovar_schedule.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dispatch_log_schema_version": 2,
                "schedule_schema_version": 3,
                "dispatch_log_relative_path": "dispatch.tsv",
                "n_particles": 2,
                "n_followers": 1,
                "pool_size": 2,
                "random_seed": 9,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    artifacts = (
        "dispatch.tsv",
        "dispatch.tsv.recovar_schedule.json",
        "run_it000_data.star",
        "run_it000_model.star",
    )
    manifest = relion_oracle_manifest_sha256(oracle, artifacts)
    order = relion_ordered_particle_sha256(particles)
    identity = relion_oracle_id(
        manifest_sha256=manifest,
        particle_order_sha256=order,
    )
    schedule = make_relion_dispatch_schedule_from_chunks(
        relion_iterations=[1],
        chunk_iterations=[1],
        chunk_first=[0],
        chunk_last=[1],
        chunk_ranks=[1],
        n_particles=2,
        original_particle_id_by_sorted_position=np.arange(2)[None, :],
        n_followers=1,
        pool_size=2,
        random_seed=9,
        oracle_id=identity,
        oracle_manifest_sha256=manifest,
        oracle_artifact_paths=artifacts,
        particle_order_sha256=order,
        particle_star_relative_path="run_it000_data.star",
        dispatch_log_relative_path="dispatch.tsv",
        source="same-oracle test",
    )

    verify_relion_dispatch_schedule_oracle(schedule, oracle)
    relocated = tmp_path / "relocated"
    shutil.copytree(oracle, relocated)
    verify_relion_dispatch_schedule_oracle(schedule, relocated)

    (relocated / "dispatch.tsv").write_text("2 1 1 0 1\n2 1 1 1 0\n")
    with np.testing.assert_raises_regex(ValueError, "state manifest does not match"):
        verify_relion_dispatch_schedule_oracle(schedule, relocated)
    shutil.copy2(oracle / "dispatch.tsv", relocated / "dispatch.tsv")

    (relocated / "run_it000_model.star").write_text("different-model-state\n")
    with np.testing.assert_raises_regex(ValueError, "state manifest does not match"):
        verify_relion_dispatch_schedule_oracle(schedule, relocated)


def test_ordered_particle_identity_detects_row_order_and_labels():
    import pandas as pd

    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@x.mrcs", "2@x.mrcs"],
            "rlnOpticsGroup": [1, 2],
            "rlnRandomSubset": [1, 2],
            "rlnGroupNumber": [4, 7],
        }
    )
    baseline = relion_ordered_particle_sha256(particles)
    assert baseline != relion_ordered_particle_sha256(particles.iloc[::-1].reset_index(drop=True))
    changed_group = particles.copy()
    changed_group.loc[0, "rlnGroupNumber"] = 5
    assert baseline != relion_ordered_particle_sha256(changed_group)


def test_dispatch_schedule_rejects_legacy_unbound_npz(tmp_path):
    path = tmp_path / "legacy_dispatch.npz"
    np.savez(
        path,
        relion_iterations=np.asarray([1], dtype=np.int64),
        owner_by_sorted_position=np.asarray([[0]], dtype=np.int64),
        n_followers=np.int64(1),
        pool_size=np.int64(1),
        random_seed=np.int64(9),
    )
    with np.testing.assert_raises_regex(ValueError, "missing keys.*oracle"):
        load_relion_dispatch_schedule(path)


def test_dispatch_builder_writes_verified_oracle_schema(tmp_path, monkeypatch):
    import pandas as pd
    import starfile

    from scripts import build_relion_dispatch_schedule

    oracle = tmp_path / "oracle"
    oracle.mkdir()
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@x.mrcs", "2@x.mrcs"],
            "rlnOpticsGroup": [1, 1],
            "rlnGroupNumber": [1, 2],
        }
    )
    starfile.write({"particles": particles}, oracle / "run_it000_data.star")
    (oracle / "run_it000_model.star").write_text("model\n")
    dispatch_log = oracle / "dispatch.log"
    dispatch_log.write_text("2 1 1 0 1\n2 1 2 1 0\n")
    scale_state = oracle / "iter2_rank1_post.tsv"
    scale_state.write_text("rank\tscale\n1\t1.0\n")
    output = tmp_path / "dispatch.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_relion_dispatch_schedule.py",
            "--dispatch-log",
            str(dispatch_log),
            "--output",
            str(output),
            "--n-particles",
            "2",
            "--n-followers",
            "2",
            "--pool-size",
            "2",
            "--random-seed",
            "9",
            "--oracle-dir",
            str(oracle),
            "--oracle-artifact",
            "iter2_rank1_post.tsv",
        ],
    )

    build_relion_dispatch_schedule.main()

    schedule = load_relion_dispatch_schedule(output)
    verify_relion_dispatch_schedule_oracle(schedule, oracle)
    assert schedule.particle_star_relative_path == "run_it000_data.star"
    assert schedule.oracle_artifact_paths == (
        "dispatch.log",
        "dispatch.log.recovar_schedule.json",
        "iter2_rank1_post.tsv",
        "run_it000_data.star",
        "run_it000_model.star",
    )

    with np.load(output, allow_pickle=False) as stored:
        tampered_payload = {name: np.asarray(stored[name]) for name in stored.files}
    tampered_payload["owner_by_sorted_position"] = np.asarray([[1, 0]], dtype=np.int64)
    tampered = tmp_path / "dispatch_tampered.npz"
    np.savez_compressed(tampered, **tampered_payload)
    tampered_schedule = load_relion_dispatch_schedule(tampered)
    with np.testing.assert_raises_regex(ValueError, "not derived from the bound dispatch"):
        verify_relion_dispatch_schedule_oracle(tampered_schedule, oracle)


def test_dispatch_builder_accepts_v2_identity_records(tmp_path, monkeypatch):
    import pandas as pd
    import starfile

    from scripts import build_relion_dispatch_schedule

    oracle = tmp_path / "oracle_v2"
    oracle.mkdir()
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@x.mrcs", "2@x.mrcs", "3@x.mrcs"],
            "rlnOpticsGroup": [1, 1, 1],
            "rlnGroupNumber": [1, 2, 3],
        }
    )
    starfile.write({"particles": particles}, oracle / "run_it000_data.star")
    (oracle / "run_it000_model.star").write_text("model\n")
    dispatch_log = oracle / "dispatch_v2.log"
    dispatch_log.write_text(
        "2 1 1 0 2\n"
        "2 1 2 1 0\n"
        "2 1 1 2 1\n"
    )
    output = tmp_path / "dispatch_v2.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_relion_dispatch_schedule.py",
            "--dispatch-log",
            str(dispatch_log),
            "--output",
            str(output),
            "--n-particles",
            "3",
            "--n-followers",
            "2",
            "--pool-size",
            "3",
            "--random-seed",
            "9",
            "--oracle-dir",
            str(oracle),
        ],
    )

    build_relion_dispatch_schedule.main()
    schedule = load_relion_dispatch_schedule(output)
    assert_matches(schedule.owner_by_sorted_position, [[0, 1, 0]])
    assert_matches(
        schedule.original_particle_id_by_sorted_position, [[2, 0, 1]]
    )
    verify_relion_dispatch_schedule_oracle(schedule, oracle)

    with np.load(output, allow_pickle=False) as stored:
        tampered_payload = {name: np.asarray(stored[name]) for name in stored.files}
    tampered_payload["original_particle_id_by_sorted_position"] = np.asarray(
        [[0, 1, 2]], dtype=np.int64
    )
    tampered = tmp_path / "dispatch_v2_tampered_identity.npz"
    np.savez_compressed(tampered, **tampered_payload)
    tampered_schedule = load_relion_dispatch_schedule(tampered)
    with np.testing.assert_raises_regex(
        ValueError, "particle identities were not derived from the bound dispatch"
    ):
        verify_relion_dispatch_schedule_oracle(tampered_schedule, oracle)


def test_dispatch_builder_rejects_nonbijective_v2_original_ids(tmp_path, monkeypatch):
    import pandas as pd
    import starfile

    from scripts import build_relion_dispatch_schedule

    oracle = tmp_path / "oracle_bad_v2"
    oracle.mkdir()
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@x.mrcs", "2@x.mrcs"],
            "rlnOpticsGroup": [1, 1],
            "rlnGroupNumber": [1, 2],
        }
    )
    starfile.write({"particles": particles}, oracle / "run_it000_data.star")
    dispatch_log = oracle / "dispatch_bad_v2.log"
    dispatch_log.write_text("2 1 1 0 0\n2 1 2 1 0\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_relion_dispatch_schedule.py",
            "--dispatch-log",
            str(dispatch_log),
            "--output",
            str(tmp_path / "bad.npz"),
            "--n-particles",
            "2",
            "--n-followers",
            "2",
            "--pool-size",
            "2",
            "--random-seed",
            "9",
            "--oracle-dir",
            str(oracle),
        ],
    )
    with np.testing.assert_raises_regex(SystemExit, "original particle IDs are not a bijection"):
        build_relion_dispatch_schedule.main()


def test_dispatch_schedule_rejects_noninteger_schema_fields_before_cast(tmp_path):
    path = tmp_path / "dispatch_bad_dtype.npz"
    valid = {
        "schema_version": np.int64(3),
        "relion_iterations": np.asarray([1], dtype=np.int64),
        "owner_by_sorted_position": np.asarray([[0, 0]], dtype=np.int64),
        "original_particle_id_by_sorted_position": np.arange(2, dtype=np.int64)[None, :],
        "n_followers": np.int64(1),
        "pool_size": np.int64(2),
        "random_seed": np.int64(9),
        "oracle_id": np.asarray(_ORACLE_ID),
        "oracle_manifest_sha256": np.asarray(_ORACLE_MANIFEST),
        "oracle_artifact_paths": np.asarray(
            ["dispatch.tsv", "dispatch.tsv.recovar_schedule.json", "run_it000_data.star"]
        ),
        "particle_order_sha256": np.asarray(_PARTICLE_ORDER),
        "particle_star_relative_path": np.asarray("run_it000_data.star"),
        "dispatch_log_relative_path": np.asarray("dispatch.tsv"),
    }

    values = dict(valid)
    values["schema_version"] = np.int64(2)
    np.savez(path, **values)
    with np.testing.assert_raises_regex(
        ValueError, "schema v2 lacks.*rebuild schema v3"
    ):
        load_relion_dispatch_schedule(path)

    for key in ("relion_iterations", "n_followers", "pool_size", "random_seed"):
        values = dict(valid)
        values[key] = np.asarray([1.0]) if key == "relion_iterations" else np.float64(1.0)
        np.savez(path, **values)
        with np.testing.assert_raises_regex(ValueError, rf"{key} must have an integer dtype"):
            load_relion_dispatch_schedule(path)

    values = dict(valid)
    values["original_particle_id_by_sorted_position"] = np.asarray([[0, 0]])
    np.savez(path, **values)
    with np.testing.assert_raises_regex(ValueError, "must be a permutation"):
        load_relion_dispatch_schedule(path)


def test_dispatch_schedule_allows_idle_followers_but_rejects_invalid_owners(tmp_path):
    schedule = make_relion_dispatch_schedule_from_chunks(
        relion_iterations=[1],
        chunk_iterations=[1],
        chunk_first=[0],
        chunk_last=[3],
        chunk_ranks=[3],
        n_particles=4,
        original_particle_id_by_sorted_position=np.arange(4)[None, :],
        n_followers=4,
        pool_size=4,
        random_seed=9,
        source="idle follower capture",
        **_ORACLE_KWARGS,
    )
    path = tmp_path / "dispatch_idle_followers.npz"
    np.savez(
        path,
        schema_version=np.int64(3),
        relion_iterations=schedule.relion_iterations,
        owner_by_sorted_position=schedule.owner_by_sorted_position,
        original_particle_id_by_sorted_position=(
            schedule.original_particle_id_by_sorted_position
        ),
        n_followers=np.int64(schedule.n_followers),
        pool_size=np.int64(schedule.pool_size),
        random_seed=np.int64(schedule.random_seed),
        oracle_id=np.asarray(_ORACLE_ID),
        oracle_manifest_sha256=np.asarray(_ORACLE_MANIFEST),
        oracle_artifact_paths=np.asarray(
            ["dispatch.tsv", "dispatch.tsv.recovar_schedule.json", "run_it000_data.star"]
        ),
        particle_order_sha256=np.asarray(_PARTICLE_ORDER),
        particle_star_relative_path=np.asarray("run_it000_data.star"),
        dispatch_log_relative_path=np.asarray("dispatch.tsv"),
    )

    loaded = load_relion_dispatch_schedule(path)
    assert_matches(loaded.owner_by_sorted_position, [[2, 2, 2, 2]])
    assert loaded.n_followers == 4

    np.savez(
        path,
        schema_version=np.int64(3),
        relion_iterations=np.asarray([1], dtype=np.int64),
        owner_by_sorted_position=np.asarray([[-1, -1]], dtype=np.int64),
        original_particle_id_by_sorted_position=np.arange(2, dtype=np.int64)[None, :],
        n_followers=np.int64(4),
        pool_size=np.int64(2),
        random_seed=np.int64(9),
        oracle_id=np.asarray(_ORACLE_ID),
        oracle_manifest_sha256=np.asarray(_ORACLE_MANIFEST),
        oracle_artifact_paths=np.asarray(
            ["dispatch.tsv", "dispatch.tsv.recovar_schedule.json", "run_it000_data.star"]
        ),
        particle_order_sha256=np.asarray(_PARTICLE_ORDER),
        particle_star_relative_path=np.asarray("run_it000_data.star"),
        dispatch_log_relative_path=np.asarray("dispatch.tsv"),
    )
    with np.testing.assert_raises_regex(ValueError, "owners are out of bounds"):
        load_relion_dispatch_schedule(path)


def test_follower_scale_replay_loads_sparse_complete_states_and_validates_topology(tmp_path):
    path = tmp_path / "follower_scale_replay.npz"
    scales = np.asarray(
        [
            [[1.0, 1.1, 0.9], [1.0, 0.8, 1.2]],
            [[1.0, 1.2, 0.8], [1.0, 0.7, 1.3]],
        ],
        dtype=np.float64,
    )
    header = "iteration\tmpi_rank\tgroup_index\tscale_post\n"
    for rank, source_name in enumerate(("rank1_post.tsv", "rank2_post.tsv"), start=1):
        rows = [header]
        for iteration_idx, iteration in enumerate((1, 3)):
            for group in range(3):
                rows.append(
                    f"{iteration}\t{rank}\t{group}\t{scales[iteration_idx, rank - 1, group]:.17g}\n"
                )
        (tmp_path / source_name).write_text("".join(rows))
    np.savez(
        path,
        schema_version=np.int64(1),
        boundary=np.asarray("numbered_pre_score"),
        source_artifact_relative_paths=np.asarray(["rank1_post.tsv", "rank2_post.tsv"]),
        relion_iterations=np.asarray([2, 4], dtype=np.int64),
        follower_scales=scales,
        oracle_id=np.asarray(_ORACLE_ID),
        source=np.asarray("causal unit replay"),
    )

    replay = load_relion_follower_scale_replay(path)

    assert_matches(replay.relion_iterations, [2, 4])
    assert_matches(replay.follower_scales, scales)
    assert replay.source == "causal unit replay"
    validate_relion_follower_scale_replay(
        replay,
        n_followers=2,
        n_groups=3,
        schedule_iterations=[1, 2, 3, 4],
        schedule_oracle_id=_ORACLE_ID,
        schedule_artifact_paths=["rank1_post.tsv", "rank2_post.tsv"],
        oracle_dir=tmp_path,
    )
    stale_replay = RelionFollowerScaleReplay(
        relion_iterations=replay.relion_iterations,
        follower_scales=replay.follower_scales + 0.01,
        oracle_id=replay.oracle_id,
        schema_version=replay.schema_version,
        boundary=replay.boundary,
        source_artifact_relative_paths=replay.source_artifact_relative_paths,
        source="stale payload",
    )
    with np.testing.assert_raises_regex(ValueError, "not derived from the bound source"):
        validate_relion_follower_scale_replay(stale_replay, oracle_dir=tmp_path)
    with np.testing.assert_raises_regex(ValueError, "follower dimension"):
        validate_relion_follower_scale_replay(replay, n_followers=3, n_groups=3)
    with np.testing.assert_raises_regex(ValueError, "absent from the captured dispatch"):
        validate_relion_follower_scale_replay(
            replay,
            n_followers=2,
            n_groups=3,
            schedule_iterations=[1, 2, 3],
        )
    with np.testing.assert_raises_regex(ValueError, "oracle_id does not match"):
        validate_relion_follower_scale_replay(
            replay,
            n_followers=2,
            n_groups=3,
            schedule_iterations=[1, 2, 3, 4],
            schedule_oracle_id="f" * 64,
        )
    with np.testing.assert_raises_regex(ValueError, "absent from the verified oracle manifest"):
        validate_relion_follower_scale_replay(
            replay,
            n_followers=2,
            n_groups=3,
            schedule_iterations=[1, 2, 3, 4],
            schedule_oracle_id=_ORACLE_ID,
            schedule_artifact_paths=["rank1_post.tsv"],
        )


def test_follower_scale_replay_reloads_continuation_model_on_every_follower(tmp_path):
    model = tmp_path / "run_it011_model.star"
    model.write_text(
        """data_model_groups

loop_
_rlnGroupNumber #1
_rlnGroupName #2
_rlnGroupNrParticles #3
_rlnGroupScaleCorrection #4
1 group1 4 0.916122
2 group2 5 1.194794
"""
    )
    scales = np.asarray(
        [[[0.916122, 1.194794], [0.916122, 1.194794]]],
        dtype=np.float64,
    )
    replay = RelionFollowerScaleReplay(
        relion_iterations=np.asarray([12], dtype=np.int64),
        follower_scales=scales,
        oracle_id=_ORACLE_ID,
        schema_version=1,
        boundary="numbered_pre_score",
        source_artifact_relative_paths=(model.name,),
        source="RELION continuation model reload",
    )

    validate_relion_follower_scale_replay(
        replay,
        n_followers=2,
        n_groups=2,
        schedule_iterations=[12],
        schedule_oracle_id=_ORACLE_ID,
        schedule_artifact_paths=[model.name],
        numbered_iterations=[12],
        first_numbered_iteration=1,
        oracle_dir=tmp_path,
    )


def test_follower_scale_replay_rejects_first_and_outside_numbered_iterations():
    first = RelionFollowerScaleReplay(
        relion_iterations=np.asarray([1], dtype=np.int64),
        follower_scales=np.ones((1, 2, 3), dtype=np.float64),
        **_REPLAY_KWARGS,
        source="first boundary",
    )
    with np.testing.assert_raises_regex(ValueError, "cannot target the first numbered"):
        validate_relion_follower_scale_replay(
            first,
            n_followers=2,
            n_groups=3,
            schedule_iterations=[1, 2, 3, 4, 5],
            numbered_iterations=[1, 2, 3, 4],
            first_numbered_iteration=1,
        )

    post_numbered = RelionFollowerScaleReplay(
        relion_iterations=np.asarray([5], dtype=np.int64),
        follower_scales=np.ones((1, 2, 3), dtype=np.float64),
        **_REPLAY_KWARGS,
        source="post-numbered boundary",
    )
    with np.testing.assert_raises_regex(ValueError, "outside the requested numbered"):
        validate_relion_follower_scale_replay(
            post_numbered,
            n_followers=2,
            n_groups=3,
            schedule_iterations=[1, 2, 3, 4, 5],
            numbered_iterations=[1, 2, 3, 4],
            first_numbered_iteration=1,
        )


def test_follower_scale_replay_application_requires_every_row_exactly_once():
    replay = RelionFollowerScaleReplay(
        relion_iterations=np.asarray([2, 4], dtype=np.int64),
        follower_scales=np.ones((2, 2, 3), dtype=np.float64),
        **_REPLAY_KWARGS,
        source="application accounting",
    )

    requested, applied = validate_relion_follower_scale_replay_application(
        replay,
        applied_iterations=[2, 4],
    )
    assert_matches(requested, [2, 4])
    assert_matches(applied, [2, 4])

    for observed in ([], [2], [2, 2, 4], [2, 3, 4]):
        with np.testing.assert_raises_regex(RuntimeError, "not applied exactly once"):
            validate_relion_follower_scale_replay_application(
                replay,
                applied_iterations=observed,
            )


def test_follower_scale_replay_rejects_duplicate_iterations_and_nonpositive_scales(tmp_path):
    path = tmp_path / "bad_follower_scale_replay.npz"
    np.savez(
        path,
        schema_version=np.int64(1),
        boundary=np.asarray("numbered_pre_score"),
        source_artifact_relative_paths=np.asarray(["rank1_post.tsv", "rank2_post.tsv"]),
        relion_iterations=np.asarray([3, 3], dtype=np.int64),
        follower_scales=np.ones((2, 2, 3), dtype=np.float64),
        oracle_id=np.asarray(_ORACLE_ID),
    )
    with np.testing.assert_raises_regex(ValueError, "unique"):
        load_relion_follower_scale_replay(path)

    np.savez(
        path,
        schema_version=np.int64(1),
        boundary=np.asarray("numbered_pre_score"),
        source_artifact_relative_paths=np.asarray(["rank1_post.tsv", "rank2_post.tsv"]),
        relion_iterations=np.asarray([3], dtype=np.int64),
        follower_scales=np.asarray([[[1.0, 0.0], [1.0, 1.0]]]),
        oracle_id=np.asarray(_ORACLE_ID),
    )
    with np.testing.assert_raises_regex(ValueError, "strictly positive"):
        load_relion_follower_scale_replay(path)

    np.savez(
        path,
        schema_version=np.int64(1),
        boundary=np.asarray("numbered_pre_score"),
        source_artifact_relative_paths=np.asarray(["rank1_post.tsv", "rank2_post.tsv"]),
        relion_iterations=np.asarray([3], dtype=np.int64),
        follower_scales=np.ones((1, 2, 2), dtype=np.complex128),
        oracle_id=np.asarray(_ORACLE_ID),
    )
    with np.testing.assert_raises_regex(ValueError, "real numeric dtype"):
        load_relion_follower_scale_replay(path)


def test_worker_group_axis_and_owner_change_select_runtime_scale():
    state = make_relion_follower_scale_state(
        n_followers=2,
        group_counts=[1, 1, 1],
        n_optics_groups=1,
        initial_group_scales=[1, 1, 1],
    )
    state = type(state)(
        scales=np.asarray([[1.0, 1.3, 0.8], [1.0, 0.9, 1.2]]),
        group_counts=state.group_counts,
        n_optics_groups=state.n_optics_groups,
    )
    groups = np.asarray([1, 2])

    assert_matches(
        relion_worker_group_ids(groups, [0, 1], n_groups=3),
        [1, 5],
    )
    np.testing.assert_allclose(
        select_relion_follower_scales(state, group_ids=groups, follower_owners=[0, 1]),
        [1.3, 1.2],
    )
    np.testing.assert_allclose(
        select_relion_follower_scales(state, group_ids=groups, follower_owners=[1, 0]),
        [0.9, 0.8],
    )


def test_final_dispatch_uses_next_absolute_iteration_and_missing_row_fails_closed():
    numbered = [np.asarray([0, 1]), np.zeros(0, dtype=np.int64)]
    final = [np.asarray([1, 0]), np.zeros(0, dtype=np.int64)]
    schedule = {3: numbered, 4: final}

    assert _require_relion_follower_owners(
        schedule,
        relion_iteration=3,
        stage="numbered",
    ) is numbered
    assert _require_relion_follower_owners(
        schedule,
        relion_iteration=4,
        stage="final all-data",
    ) is final
    with np.testing.assert_raises_regex(RuntimeError, "refusing to reuse stale follower owners"):
        _require_relion_follower_owners(
            {3: numbered},
            relion_iteration=4,
            stage="final all-data",
        )


def test_final_dispatch_remaps_scoring_scale_norm_ratio_and_xa_aa_group_ids():
    state = make_relion_follower_scale_state(
        n_followers=2,
        group_counts=[1, 1],
        n_optics_groups=1,
    )
    state = type(state)(
        scales=np.asarray([[1.0, 1.2], [1.0, 0.8]]),
        group_counts=state.group_counts,
        n_optics_groups=state.n_optics_groups,
    )
    half_inputs = initialize_halfsets(
        (None, None),
        previous_best_translations=[None, None],
        previous_best_rotation_eulers=[None, None],
        image_corrections=[np.asarray([2.4, 2.4]), np.zeros(0)],
        scale_corrections=[np.asarray([1.2, 0.8]), np.zeros(0)],
        group_ids=[np.asarray([1, 1]), np.zeros(0, dtype=np.int64)],
        group_count=[2, 2],
    )

    stats_group_ids = _remap_relion_follower_runtime_inputs(
        state=state,
        relion_half_inputs=half_inputs,
        follower_owners_per_half=[
            np.asarray([1, 0]),
            np.zeros(0, dtype=np.int64),
        ],
        physical_group_count=2,
        dtype=np.float32,
    )

    np.testing.assert_allclose(half_inputs[0].scale_corrections, [0.8, 1.2])
    np.testing.assert_allclose(half_inputs[0].image_corrections, [1.6, 3.6])
    assert_matches(stats_group_ids[0], [3, 1])
    assert stats_group_ids[1].size == 0


def _follower_statistics(n_images):
    """Norm residuals and follower-by-group scale sums for the stand-in engine (two followers, two groups)."""
    return dict(
        wsum_norm_correction=np.ones(n_images),
        wsum_scale_correction_xa=np.ones((2, 2)),
        wsum_scale_correction_aa=np.full((2, 2), 2.0),
    )


def _follower_run(monkeypatch, *, final, follower_replay=None):
    """A two-iteration Class3D run under the strict follower-scale emulation; ``final`` converges it so the
    final all-data pass runs (RELION iteration 3)."""
    from helpers.tiny_refinement import follower_scale_replay, run_tiny_refinement

    return run_tiny_refinement(
        monkeypatch, n_classes=2, final_after_max_iter=False, converge_after=2 if final else None,
        engine_noise_fields=_follower_statistics, **follower_scale_replay(3, replay=follower_replay),
    )


def _follower_scale_replay(iteration=2):
    return RelionFollowerScaleReplay(
        relion_iterations=np.asarray([iteration], dtype=np.int64),
        follower_scales=np.full((1, 2, 2), 1.5, dtype=np.float64),
        **_REPLAY_KWARGS,
        source="tiny follower replay",
    )


def test_final_dispatch_remap_is_wired_before_final_scoring(monkeypatch):
    from helpers.tiny_refinement import CallTrace

    from relax.refinement import finalization, iteration_loop
    from relax.relion import relion_worker_scale

    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "_dispatch_relion_follower_scale_for_final_all_data", "dispatch")
    trace.wrap(relion_worker_scale, "_require_relion_follower_owners", "owners")
    trace.wrap(relion_worker_scale, "_remap_relion_follower_runtime_inputs", "remap")
    trace.wrap(finalization, "run_final_all_data", "final")
    _follower_run(monkeypatch, final=True)
    final_calls = [call for call in trace.calls_seen if call.label in ("dispatch", "final") or "dispatch" in call.inside]
    assert [call.label for call in final_calls] == ["dispatch", "owners", "remap", "final"]
    assert final_calls[1].kwargs["stage"] == "final all-data" and final_calls[1].kwargs["relion_iteration"] == 3


@pytest.mark.parametrize("final", [False, True])
def test_numbered_scale_telemetry_brackets_scoring_and_mstep_boundaries(monkeypatch, final):
    """Pre-score telemetry is recorded at dispatch, before the input source's state and scoring; the M-step
    installs the updated state on the setup, then the post-M-step scales are recorded before convergence.
    Both are copies; the result reports both trajectories through one result_outputs call."""
    from helpers.tiny_refinement import CallTrace

    from relax.parity.relion_replay_source import RelionReplaySource
    from relax.refinement import iteration_loop
    from relax.relion import relion_worker_scale

    seen = {}

    def dispatched(call):
        seen["setup"], seen["history"] = call.args[0], call.args[1]
        scales = seen["setup"].follower_scale_state.scales
        assert not np.shares_memory(seen["history"].relion_scale_follower_scales_numbered_pre_score_trajectory[-1], scales)

    def post_mstep_count(expected):
        def check(call):
            trajectory = seen["history"].relion_scale_follower_scales_numbered_post_mstep_trajectory
            assert len(trajectory) == expected(call)
        return check

    def installed(call):
        (update,) = [c for c in trace.calls("update_scales") if c.inside == ("update",)][-1:]
        assert seen["setup"].follower_scale_state is update.result

    def recorded(call):
        trajectory = seen["history"].relion_scale_follower_scales_numbered_post_mstep_trajectory
        scales = seen["setup"].follower_scale_state.scales
        np.testing.assert_array_equal(trajectory[-1], scales)
        assert not np.shares_memory(trajectory[-1], scales)

    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "_dispatch_relion_follower_scale_for_numbered_iteration", "dispatch", after=dispatched)
    trace.wrap(RelionReplaySource, "numbered_state", "replay")
    trace.wrap(iteration_loop, "_update_relion_follower_corrections", "update",
               before=post_mstep_count(lambda call: len(trace.calls("update")) - 1), after=installed)
    trace.wrap(relion_worker_scale, "update_relion_follower_scales", "update_scales")
    trace.wrap(iteration_loop, "update_iteration_convergence", "convergence",
               before=post_mstep_count(lambda call: len(trace.calls("convergence"))))
    trace.wrap(iteration_loop, "update_iteration_convergence", "convergence_recorded", before=recorded)
    trace.wrap(RelionFollowerScaleSetup, "result_outputs", "result")
    result = _follower_run(monkeypatch, final=final)
    numbered = ["dispatch", "replay", "update", "convergence"]
    assert trace.labels(*numbered, "result") == numbered * 2 + ["result"]
    for trajectory in (result.follower_scale.follower_scales_numbered_pre_score_trajectory,
                       result.follower_scale.follower_scales_numbered_post_mstep_trajectory):
        assert trajectory.shape == (2, 2, 2) and trajectory.dtype == np.float64


def test_relion_norm_scale_updates_are_not_disabled_for_k_class(monkeypatch, caplog):
    """Class3D without the follower emulation runs RELION's norm and scale updates and logs their ranges."""
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.refinement import iteration_loop
    from relax.relion import relion_normalization
    from relax.relion.relion_normalization import _format_relion_correction_range

    def statistics(n_images):
        return dict(wsum_norm_correction=np.ones(n_images), wsum_scale_correction_xa=np.ones(1),
                    wsum_scale_correction_aa=np.full(1, 2.0))

    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "numbered_norm_scale_update", "numbered")
    trace.wrap(relion_normalization, "prepare_norm_scale_update", "prepare")
    trace.wrap(relion_normalization, "update_relion_norm_scale_corrections", "update")
    trace.wrap(iteration_loop, "log_norm_scale_update", "log")
    with caplog.at_level(logging.INFO, logger="relax.refinement.iteration_loop"):
        run_tiny_refinement(monkeypatch, n_classes=2, final_after_max_iter=False, engine_noise_fields=statistics)
    assert trace.labels() == ["numbered", "prepare", "update", "log"] * 2
    for numbered in trace.calls("numbered"):
        assert numbered.kwargs["do_scale_correction"] is True and numbered.result is not None
    messages = [record.getMessage() for record in caplog.records if "RELION norm correction update" in record.getMessage()]
    for message, numbered in zip(messages, trace.calls("numbered"), strict=True):
        update = numbered.result
        for half in range(2):
            assert _format_relion_correction_range(update.image_corrections_per_half[half]) in message
            assert _format_relion_correction_range(update.scale_corrections_per_half[half]) in message


def test_sparse_follower_scale_replay_replaces_state_before_remap_and_telemetry(monkeypatch):
    from helpers.tiny_refinement import CallTrace

    from relax.helpers.iteration_history import RefinementHistory
    from relax.refinement import iteration_loop
    from relax.relion import relion_worker_scale

    replay = _follower_scale_replay(2)
    seen = {}

    def dispatch_starts(call):
        seen["setup"] = call.args[0]

    def remapped_state(call):
        seen.setdefault("remapped", []).append(np.array(call.kwargs["state"].scales, copy=True))

    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "_dispatch_relion_follower_scale_for_numbered_iteration", "dispatch",
               before=dispatch_starts)
    trace.wrap(relion_worker_scale, "_require_relion_follower_owners", "owners")
    trace.wrap(relion_worker_scale, "_remap_relion_follower_runtime_inputs", "remap", before=remapped_state)
    trace.wrap(RefinementHistory, "record_follower_scale_pre_score", "telemetry")
    _follower_run(monkeypatch, final=False, follower_replay=replay)
    in_dispatch = [call.label for call in trace.calls_seen if call.inside == ("dispatch",)]
    # Iteration 1 has nothing to remap; iteration 2 replays the state, remaps, then records it.
    assert in_dispatch == ["owners", "telemetry", "owners", "remap", "telemetry"]
    np.testing.assert_array_equal(seen["remapped"][0], replay.follower_scales[0])
    np.testing.assert_array_equal(trace.calls("telemetry")[1].args[1], replay.follower_scales[0])


def test_strict_restart_requires_coupled_perturbation_and_model_scale_state():
    _validate_coupled_relion_restart_state(
        perturb_restart_state_iterations=(11,),
        follower_replay_by_iteration={12: np.ones((2, 4))},
        follower_replay_source_artifacts=("run_it011_model.star",),
    )

    with np.testing.assert_raises_regex(ValueError, "missing numbered iterations.*12"):
        _validate_coupled_relion_restart_state(
            perturb_restart_state_iterations=(11,),
            follower_replay_by_iteration={},
            follower_replay_source_artifacts=(),
        )

    with np.testing.assert_raises_regex(ValueError, "unmatched numbered iterations.*12"):
        _validate_coupled_relion_restart_state(
            perturb_restart_state_iterations=(),
            follower_replay_by_iteration={12: np.ones((2, 4))},
            follower_replay_source_artifacts=("run_it011_model.star",),
        )


@pytest.mark.parametrize("final", [False, True])
def test_sparse_follower_scale_replay_accounting_guards_every_result_return(monkeypatch, final):
    """Each return path (no final pass, final all-data) validates the replay accounting once, against the
    iterations the controller applied, and reports what that validation returned.

    The local-search diagnostic return has no CPU run.
    """
    from helpers.tiny_refinement import CallTrace

    from relax.refinement import iteration_loop

    replay = _follower_scale_replay(2)
    seen = {}
    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "_dispatch_relion_follower_scale_for_numbered_iteration", "dispatch",
               before=lambda call: seen.setdefault("history", call.args[1]))
    trace.wrap(iteration_loop, "_finalize_relion_follower_scale_replay_telemetry", "accounting")
    result = _follower_run(monkeypatch, final=final, follower_replay=replay)
    (accounting,) = trace.calls("accounting")
    assert accounting.args[0] is replay
    assert accounting.kwargs["applied_iterations"] is seen["history"].relion_follower_scale_replay_applied_iterations
    requested, applied = accounting.result
    assert result.replay.requested_iterations is requested
    assert result.replay.applied_iterations is applied
    np.testing.assert_array_equal(applied, [2])


def test_disabled_follower_replay_completion_does_not_consume_history_or_log():
    assert _finalize_relion_follower_scale_replay_telemetry(
        None, applied_iterations=object(), logger=None,
    ) == (None, None)


def test_follower_replay_completion_returns_copies_and_logs_after_validation(caplog):
    replay = RelionFollowerScaleReplay(
        relion_iterations=np.asarray([2, 4], dtype=np.int64),
        follower_scales=np.ones((2, 2, 3), dtype=np.float64),
        **_REPLAY_KWARGS,
        source="completion accounting",
    )
    observed = np.asarray([2, 4], dtype=np.int64)
    completion_logger = logging.getLogger("relax.test.follower_completion")
    caplog.set_level(logging.INFO, logger=completion_logger.name)

    requested, applied = _finalize_relion_follower_scale_replay_telemetry(
        replay, applied_iterations=observed, logger=completion_logger,
    )

    assert_matches(requested, [2, 4])
    assert_matches(applied, [2, 4])
    assert requested.dtype == applied.dtype == np.dtype(np.int64)
    assert not np.shares_memory(requested, replay.relion_iterations)
    assert not np.shares_memory(applied, observed)
    assert caplog.messages == [
        "Diagnostic RELION follower-scale replay complete: "
        "source=completion accounting requested=[2, 4] applied=[2, 4]"
    ]


def test_follower_replay_completion_never_logs_success_for_invalid_accounting(caplog):
    replay = RelionFollowerScaleReplay(
        relion_iterations=np.asarray([2, 4], dtype=np.int64),
        follower_scales=np.ones((2, 2, 3), dtype=np.float64),
        **_REPLAY_KWARGS,
        source="incomplete accounting",
    )
    completion_logger = logging.getLogger("relax.test.follower_completion")
    caplog.set_level(logging.INFO, logger=completion_logger.name)

    for observed in ([], [2], [4, 2], [2, 2, 4], [2, 3, 4], [2.0, 4.0]):
        with np.testing.assert_raises(RuntimeError):
            _finalize_relion_follower_scale_replay_telemetry(
                replay, applied_iterations=observed, logger=completion_logger,
            )
    assert caplog.messages == []


def test_only_optics_prefix_scale_stats_are_combined_between_followers():
    state = make_relion_follower_scale_state(
        n_followers=2,
        group_counts=np.ones(4),
        n_optics_groups=1,
    )
    xa = np.asarray([[2.0, 5.0, 1.0, 1.0], [4.0, 0.0, 1.0, 1.0]])
    aa = np.asarray([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]])

    updated = update_relion_follower_scales(
        state,
        wsum_signal_product=xa,
        wsum_reference_power=aa,
        reduction_mode=RELION_SCALE_REDUCTION_OPTICS_PREFIX,
    )

    # --dont_combine_weights_via_disc (segmented MPI pack). Group 0 is reduced to raw scale 3 on both followers. Group 1 remains
    # local: raw 5 on follower 1 and the zero-AA default 1 on follower 2.
    # Follower-local normalization changes absolute values but preserves the
    # expected within-follower ratios.
    np.testing.assert_allclose(updated.scales[:, 0] / updated.scales[:, 2], [3.0, 3.0])
    np.testing.assert_allclose(updated.scales[0, 1] / updated.scales[0, 2], 5.0)
    np.testing.assert_allclose(updated.scales[1, 1] / updated.scales[1, 2], 1.0)


def test_captured_group5989_rank_states_reproduce_runtime_and_star_values():
    n_groups = 10_000
    target_group = 5989
    rank1_avg = 1.0265163330375102
    rank2_avg = 1.026738611317099
    target_raw_rank1 = 1.3489885472342609

    # Construct the smallest homogeneous background whose two independent
    # follower normalizers equal the captured values while preserving the
    # observed combined group-0 boundary.
    matrix = np.asarray([[n_groups - 1.5, 0.5], [0.5, n_groups - 1.5]])
    rhs = np.asarray(
        [n_groups * rank1_avg - target_raw_rank1, n_groups * rank2_avg - 1.0],
    )
    background_rank1, background_rank2 = np.linalg.solve(matrix, rhs)
    xa = np.vstack(
        [
            np.full(n_groups, background_rank1, dtype=np.float64),
            np.full(n_groups, background_rank2, dtype=np.float64),
        ]
    )
    aa = np.ones((2, n_groups), dtype=np.float64)
    xa[0, target_group] = target_raw_rank1
    xa[1, target_group] = 0.0
    aa[1, target_group] = 0.0
    state = make_relion_follower_scale_state(
        n_followers=2,
        group_counts=np.ones(n_groups),
        n_optics_groups=1,
    )

    updated = update_relion_follower_scales(
        state,
        wsum_signal_product=xa,
        wsum_reference_power=aa,
        reduction_mode=RELION_SCALE_REDUCTION_OPTICS_PREFIX,
    )

    np.testing.assert_allclose(
        updated.scales[0, target_group],
        1.3141423120297953,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        updated.scales[1, target_group],
        0.973957723005275,
        rtol=0.0,
        atol=2e-15,
    )
    assert_matches(relion_rank1_serialized_scales(updated)[target_group], updated.scales[0, target_group])


def test_firstiter_cc_preserves_follower_scale_state_exactly():
    state = make_relion_follower_scale_state(
        n_followers=2,
        group_counts=[1, 1],
        n_optics_groups=1,
        initial_group_scales=[0.8, 1.2],
    )

    updated = update_relion_follower_scales(
        state,
        # RELION skips scale correction during firstiter_cc, so production
        # paths need not materialize expanded follower XA/AA statistics.
        wsum_signal_product=None,
        wsum_reference_power=None,
        reduction_mode=RELION_SCALE_REDUCTION_OPTICS_PREFIX,
        relion_firstiter_cc_this_iter=True,
    )

    assert updated is state


def test_file_combined_oracle_reduces_every_physical_group():
    """Without --dont_combine_weights_via_disc RELION writes one full MlWsumModel pack.

    That message carries all nr_groups XA/AA entries, so group 1, which has
    statistics only on follower 1, reaches follower 2 as well.
    """
    state = make_relion_follower_scale_state(
        n_followers=2,
        group_counts=np.ones(4),
        n_optics_groups=1,
    )
    xa = np.asarray([[2.0, 5.0, 1.0, 1.0], [4.0, 0.0, 1.0, 1.0]])
    aa = np.asarray([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]])

    updated = update_relion_follower_scales(
        state,
        wsum_signal_product=xa,
        wsum_reference_power=aa,
        reduction_mode=RELION_SCALE_REDUCTION_ALL_GROUPS,
    )

    np.testing.assert_allclose(updated.scales[:, 0] / updated.scales[:, 2], [3.0, 3.0])
    np.testing.assert_allclose(updated.scales[:, 1] / updated.scales[:, 2], [5.0, 5.0])
    assert_matches(updated.scales[0], updated.scales[1])


def test_scale_reduction_mode_follows_the_oracle_weight_combination():
    # Commands recorded by the D6 C4 Class3D oracle (MPI combine) and the
    # data_pdb_k4_5k_128 capture 14248154 (file combine).
    mpi = (
        "--i particles.star --ref reference_init_classes_relion.star --o run --iter 5 --K 4 "
        "--firstiter_cc --sym C4 --pool 3 --dont_combine_weights_via_disc --random_seed 41001 --gpu 0 --j 4"
    )
    via_disc = (
        "--o run --i particles.star --K 4 --iter 3 --sym C1 --firstiter_cc --gpu 0:0 --j 4 --pool 1"
    )
    assert relion_scale_reduction_mode_from_command(mpi) == RELION_SCALE_REDUCTION_OPTICS_PREFIX
    assert relion_scale_reduction_mode_from_command(via_disc) == RELION_SCALE_REDUCTION_ALL_GROUPS
    with pytest.raises(ValueError, match="ALTCPU"):
        relion_scale_reduction_mode_from_command("--o run --dont_combine_weights_via_disc --j 4")


def test_scale_reduction_mode_is_read_from_the_optimiser_header(tmp_path):
    star = tmp_path / "run_it000_optimiser.star"
    star.write_text(
        "\n# RELION optimiser; version 5.0.1\n"
        "# --o run --K 4 --dont_combine_weights_via_disc --gpu 0 \n\n"
        "# version 50001\n\ndata_optimiser_general\n\n_rlnCurrentIteration 0\n"
    )
    mode, command = relion_scale_reduction_mode_from_optimiser_star(star)
    assert mode == RELION_SCALE_REDUCTION_OPTICS_PREFIX
    assert command == "--o run --K 4 --dont_combine_weights_via_disc --gpu 0"
    bare = tmp_path / "bare_optimiser.star"
    bare.write_text("data_optimiser_general\n\n_rlnCurrentIteration 0\n")
    with pytest.raises(ValueError, match="does not record"):
        relion_scale_reduction_mode_from_optimiser_star(bare)


def test_follower_scale_update_rejects_an_unknown_reduction_mode():
    state = make_relion_follower_scale_state(n_followers=2, group_counts=[1, 1], n_optics_groups=1)
    with pytest.raises(ValueError, match="reduction_mode"):
        update_relion_follower_scales(
            state,
            wsum_signal_product=np.ones((2, 2)),
            wsum_reference_power=np.ones((2, 2)),
            reduction_mode=None,
        )


def test_no_strict_topology_is_array_identical_to_legacy_global_updater():
    stats = NoiseStats(
        wsum_sigma2_noise=jnp.asarray([0.0]),
        wsum_img_power=jnp.asarray([0.0]),
        wsum_sigma2_offset=0.0,
        sumw=3.0,
        wsum_norm_correction=jnp.asarray([2.0, 4.0, 8.0]),
        wsum_scale_correction_xa=jnp.asarray([2.0, 9.0]),
        wsum_scale_correction_aa=jnp.asarray([1.0, 3.0]),
    )
    empty = NoiseStats(
        wsum_sigma2_noise=jnp.asarray([0.0]),
        wsum_img_power=jnp.asarray([0.0]),
        wsum_sigma2_offset=0.0,
        sumw=0.0,
    )
    kwargs = dict(
        noise_stats_per_half=[stats, empty],
        image_corrections_per_half=[np.asarray([1.0, 1.1, 0.9]), np.zeros(0)],
        scale_corrections_per_half=[np.asarray([1.0, 1.2, 1.2]), np.zeros(0)],
        group_ids_per_half=[np.asarray([0, 1, 1]), np.zeros(0, dtype=np.int64)],
        group_count_per_half=[2, 2],
        do_norm_correction=True,
    )

    legacy = update_relion_norm_scale_corrections(**kwargs)
    follower_state = None
    routed = update_relion_norm_scale_corrections(
        **kwargs,
        do_scale_correction=follower_state is None,
    )

    assert legacy.avg_norm_correction_per_half == routed.avg_norm_correction_per_half
    assert legacy.zero_norm_residual_counts == routed.zero_norm_residual_counts
    for field in (
        "norm_corrections_per_half",
        "group_scale_corrections_per_half",
        "image_corrections_per_half",
        "scale_corrections_per_half",
    ):
        for expected, actual in zip(getattr(legacy, field), getattr(routed, field), strict=True):
            assert_matches(np.asarray(actual), np.asarray(expected))
