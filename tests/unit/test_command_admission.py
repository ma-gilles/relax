"""Command admission against real portable manifests and optimiser STAR metadata."""

from __future__ import annotations

import ast
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import starfile
from helpers.float_compare import assert_matches

from relax.refinement import command_options
from relax.relion import relion_worker_scale

pytestmark = pytest.mark.unit
LOG = logging.getLogger(__name__)


@pytest.fixture
def dispatch_inputs(tmp_path):
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    particles = pd.DataFrame({
        "rlnImageName": ["2@particles.mrcs", "1@particles.mrcs"],
        "rlnOpticsGroup": [1, 1], "rlnRandomSubset": [1, 2],
        "rlnGroupNumber": [2, 1],
    })
    starfile.write({"particles": particles}, oracle / "run_it000_data.star")
    (oracle / "dispatch.tsv").write_text("2 1 1 0 0\n2 1 1 1 1\n")
    (oracle / "dispatch.tsv.recovar_schedule.json").write_text(json.dumps({
        "schema_version": 2, "dispatch_log_schema_version": 2,
        "schedule_schema_version": 3, "dispatch_log_relative_path": "dispatch.tsv",
        "n_particles": 2, "n_followers": 1, "pool_size": 2, "random_seed": 9,
    }))
    (oracle / "run_optimiser.star").write_text("optimiser-state\n")
    (oracle / "run_it001_sampling.star").write_text("numbered-sampling\n")
    (oracle / "run_sampling.star").write_text("final-sampling\n")
    artifacts = tuple(sorted(path.name for path in oracle.iterdir()))
    manifest = relion_worker_scale.relion_oracle_manifest_sha256(oracle, artifacts)
    order = relion_worker_scale.relion_ordered_particle_sha256(particles)
    schedule_path = tmp_path / "dispatch.npz"
    np.savez(
        schedule_path,
        schema_version=np.int64(3),
        relion_iterations=np.asarray([1], dtype=np.int64),
        owner_by_sorted_position=np.asarray([[0, 0]], dtype=np.int64),
        original_particle_id_by_sorted_position=np.asarray([[0, 1]], dtype=np.int64),
        n_followers=np.int64(1), pool_size=np.int64(2), random_seed=np.int64(9),
        oracle_id=relion_worker_scale.relion_oracle_id(
            manifest_sha256=manifest, particle_order_sha256=order,
        ),
        oracle_manifest_sha256=manifest, oracle_artifact_paths=artifacts,
        particle_order_sha256=order, particle_star_relative_path="run_it000_data.star",
        dispatch_log_relative_path="dispatch.tsv", source="command-admission test",
    )
    args = SimpleNamespace(
        relion_dispatch_schedule=schedule_path, perturb_replay_relion_dir=oracle,
        relion_init_dir=None, relion_optimiser=None, n_classes=4,
        relion_half_sets=None, data_dir=tmp_path, relion_half_sets_from_input=False,
    )
    return args, particles, oracle


def test_no_dispatch_reads_no_oracle_or_particle_metadata():
    admitted = command_options.load_verified_dispatch_schedule(
        SimpleNamespace(relion_dispatch_schedule=None), object(), strict_replay=False,
    )
    assert admitted.schedule is None
    assert admitted.oracle_dirs == []


def test_dispatch_refuses_non_replay_before_loading(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("schedule should not be read")

    monkeypatch.setattr(relion_worker_scale, "load_relion_dispatch_schedule", forbidden)
    with pytest.raises(SystemExit, match="strict K>1 RELION replay/init state only"):
        command_options.load_verified_dispatch_schedule(
            SimpleNamespace(relion_dispatch_schedule="absent.npz"), object(), strict_replay=False,
        )


@pytest.mark.parametrize("other_root", ["same", "symlink", "relocated"])
def test_verified_dispatch_keeps_particle_identity_and_root_order(dispatch_inputs, tmp_path, other_root):
    args, particles, oracle = dispatch_inputs
    if other_root == "same":
        args.relion_init_dir = oracle
    elif other_root == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(oracle, target_is_directory=True)
        args.relion_init_dir = alias
    else:
        copy = tmp_path / "relocated"
        shutil.copytree(oracle, copy)
        args.relion_init_dir = copy
    admitted = command_options.load_verified_dispatch_schedule(args, particles, strict_replay=True)
    expected_roots = [oracle.resolve()]
    if other_root == "relocated":
        expected_roots.append(args.relion_init_dir.resolve())
    assert admitted.oracle_dirs == expected_roots
    assert admitted.schedule.source == "command-admission test"
    assert admitted.schedule.particle_order_sha256 == relion_worker_scale.relion_ordered_particle_sha256(particles)
    assert_matches(admitted.schedule.original_particle_id_by_sorted_position, [[0, 1]])


@pytest.mark.parametrize("damage, message", [
    ("manifest", "state manifest does not match"),
    ("optimiser_unmanifested", "consumed RELION optimiser is not included"),
    ("optimiser_outside", "consumed RELION optimiser must belong"),
    ("numbered_sampling", "consumed RELION sampling state is not included"),
    ("final_sampling", "consumed RELION sampling state is not included"),
    ("particle_order", "authoritative RELION group/half-set particle order"),
    ("particle_labels", "authoritative RELION group/half-set particle order"),
])
def test_dispatch_refuses_unbound_consumed_inputs(dispatch_inputs, tmp_path, damage, message):
    args, particles, oracle = dispatch_inputs
    if damage == "manifest":
        (oracle / "dispatch.tsv").write_text("tampered\n")
    elif damage.startswith("optimiser"):
        path = oracle / "other_optimiser.star" if damage.endswith("unmanifested") else tmp_path / "outside.star"
        path.write_text("other optimiser\n")
        args.relion_optimiser = path
    elif damage == "numbered_sampling":
        (oracle / "run_it002_sampling.star").write_text("unmanifested\n")
    elif damage == "final_sampling":
        # This root has no final sampling in the captured manifest.
        source = oracle / "run_sampling.star"
        source.unlink()
        schedule = dict(np.load(args.relion_dispatch_schedule, allow_pickle=False))
        paths = [str(p) for p in schedule["oracle_artifact_paths"] if str(p) != source.name]
        manifest = relion_worker_scale.relion_oracle_manifest_sha256(oracle, paths)
        schedule.update(oracle_artifact_paths=np.asarray(paths), oracle_manifest_sha256=np.asarray(manifest),
                        oracle_id=np.asarray(relion_worker_scale.relion_oracle_id(
                            manifest_sha256=manifest, particle_order_sha256=str(schedule["particle_order_sha256"]),
                        )))
        np.savez(args.relion_dispatch_schedule, **schedule)
        source.write_text("unmanifested final\n")
    elif damage == "particle_order":
        particles = particles.iloc[::-1].reset_index(drop=True)
    else:
        particles = particles.copy()
        particles.loc[0, "rlnGroupNumber"] = 99
    with pytest.raises(SystemExit, match=message) as failure:
        command_options.load_verified_dispatch_schedule(args, particles, strict_replay=True)
    assert isinstance(failure.value.__cause__, ValueError)


def test_dispatch_verifies_all_roots_before_discovering_inputs(dispatch_inputs, tmp_path, monkeypatch):
    args, particles, oracle = dispatch_inputs
    copy = tmp_path / "other"
    shutil.copytree(oracle, copy)
    args.relion_init_dir = copy
    events = []
    verify = relion_worker_scale.verify_relion_dispatch_schedule_oracle
    discover = command_options.find_relion_optimiser_star

    def record_verify(schedule, root):
        events.append(("verify", root))
        verify(schedule, root)

    def record_discover(request):
        events.append(("discover", request))
        return discover(request)

    monkeypatch.setattr(relion_worker_scale, "verify_relion_dispatch_schedule_oracle", record_verify)
    monkeypatch.setattr(command_options, "find_relion_optimiser_star", record_discover)
    command_options.load_verified_dispatch_schedule(args, particles, strict_replay=True)
    assert events == [("verify", oracle.resolve()), ("verify", copy.resolve()), ("discover", args)]


@pytest.mark.parametrize("failure, wrapped", [(OSError("read failure"), True), (RuntimeError("unexpected"), False)])
def test_dispatch_preserves_exception_scope(monkeypatch, failure, wrapped):
    def fail(path):
        raise failure

    monkeypatch.setattr(relion_worker_scale, "load_relion_dispatch_schedule", fail)
    with pytest.raises(SystemExit if wrapped else RuntimeError) as caught:
        command_options.load_verified_dispatch_schedule(
            SimpleNamespace(relion_dispatch_schedule="capture.npz"), object(), strict_replay=True,
        )
    if wrapped:
        assert caught.value.__cause__ is failure
    else:
        assert caught.value is failure


@pytest.mark.parametrize("classes, override, expected, source", [
    (1, None, -1, "relion_cli_default"), (4, None, -1, "relion_cli_default"),
    (4, 0, 0, "cli_override"), (4, 37, 37, "cli_override"),
])
def test_runtime_controls_without_optimiser(classes, override, expected, source):
    controls = command_options.resolve_relion_runtime_controls(
        None, max_significants=override, target_iteration=8,
        firstiter_cc=False, n_classes=classes, log=LOG,
    )
    assert controls.do_ctf_correction is None
    assert controls.firstiter_ini_high_angstrom is None
    assert controls.max_significants_resolution == {
        "maximum_significants_argument": -1 if override is None else None,
        "active_max_significants": expected, "source": source,
        "gradient_refine": False, "do_grad": False, "target_iteration": 8,
    }


@pytest.mark.parametrize("ctf, ini_high, saved, override, gradient, expected, source", [
    (1, 40, 17, None, False, 17, "relion_optimiser_argument"),
    (0, -1, 17, 5, False, 5, "cli_override"),
    (None, None, None, None, False, -1, "relion_optimiser_argument"),
    (1, 40, -1, None, True, 400, "relion_gradient_runtime_default"),
    (1, 40, -1, -1, True, -1, "cli_override"),
])
def test_runtime_controls_read_saved_metadata_and_active_precedence(
    tmp_path, ctf, ini_high, saved, override, gradient, expected, source,
):
    path = tmp_path / "run_optimiser.star"
    text = "# relion_refine --i particles.star" + (f" --ini_high {ini_high}" if ini_high is not None else "") + "\n\ndata_optimiser_general\n"
    text += f"_rlnDoGradientRefine {int(gradient)}\n_rlnNumberOfIterations 25\n_rlnGradEmIters 1\n"
    if ctf is not None:
        text += f"_rlnDoCorrectCtf {ctf}\n"
    if saved is not None:
        text += f"_rlnMaximumSignificantPoses {saved}\n"
    path.write_text(text)
    controls = command_options.resolve_relion_runtime_controls(
        path, max_significants=override, target_iteration=2,
        firstiter_cc=False, n_classes=4, log=LOG,
    )
    assert controls.do_ctf_correction is (None if ctf is None else bool(ctf))
    if ini_high is None or ini_high <= 0:
        assert controls.firstiter_ini_high_angstrom is None
    else:
        assert_matches(controls.firstiter_ini_high_angstrom, float(ini_high))
    resolution = controls.max_significants_resolution
    assert resolution["active_max_significants"] == expected
    assert resolution["source"] == source
    assert resolution["maximum_significants_argument"] == saved
    assert resolution["target_iteration"] == 2
    assert resolution["gradient_refine"] == gradient


def test_runtime_controls_preserve_legacy_saved_cap_and_missing_gradient_failure(tmp_path):
    path = tmp_path / "run_optimiser.star"
    path.write_text("data_optimiser_general\n_maximum_significant_poses 23\n")
    controls = command_options.resolve_relion_runtime_controls(
        path, max_significants=None, target_iteration=1,
        firstiter_cc=True, n_classes=4, log=LOG,
    )
    assert controls.max_significants_resolution["active_max_significants"] == 23
    path.write_text("data_optimiser_general\n_rlnDoGradientRefine 1\n")
    with pytest.raises(ValueError, match="requires rlnNumberOfIterations"):
        command_options.resolve_relion_runtime_controls(
            path, max_significants=None, target_iteration=1,
            firstiter_cc=True, n_classes=4, log=LOG,
        )


def test_actual_command_keeps_admission_topology_and_option_installation_visible(dispatch_inputs):
    args, particles, oracle = dispatch_inputs
    args.relion_scale_followers = None
    args.relion_follower_scale_replay = None
    args.seed = 9
    args.init_relion_iteration = 0
    args.max_iter = 1
    source = Path(command_options.__file__).with_name("full_refinement.py")
    main = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main")
    first = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "strict_relion_scale_context" for t in node.targets))
    last = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "follower_topology" for t in node.targets))
    calls = []

    def prepare(followers, schedule, groups, **kwargs):
        calls.append((followers, schedule, groups, kwargs))
        return object()

    groups = object()
    scope = dict(args=args, command_options=command_options, prepare_follower_topology=prepare,
                 group_particle_source=SimpleNamespace(particles=particles, path=oracle / "run_it000_data.star"),
                 particle_groups=groups, logger=LOG)
    exec(compile(ast.Module(body=main.body[first:last + 1], type_ignores=[]), str(source), "exec"), scope)
    _, schedule, routed_groups, kwargs = calls[0]
    assert schedule is scope["dispatch"].schedule is scope["relion_dispatch_schedule"]
    assert routed_groups is groups
    assert kwargs["oracle_dir"] == oracle.resolve()
    assert kwargs["strict_replay"] is True
    assert kwargs["random_seed"] == 9
    assert kwargs["max_iter"] == 1
    args.max_significants = None
    args.firstiter_cc = False
    scope["optimiser_star"] = oracle / "run_optimiser.star"
    first = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "runtime_controls" for t in node.targets))
    last = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "relion_firstiter_ini_high_angstrom" for t in node.targets))
    exec(compile(ast.Module(body=main.body[first:last + 1], type_ignores=[]), str(source), "exec"), scope)
    assert args.max_significants == -1
    assert scope["expected_accuracy_do_ctf_correction"] is None
    assert scope["relion_firstiter_ini_high_angstrom"] is None
