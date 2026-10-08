"""``relax refine`` / ``relax class3d`` (``full_refinement.main``) run on a tiny data directory.

The tests run the command (``helpers.tiny_main``) and check what it hands the controller, what it refuses and
what it writes, instead of reading its source.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.tiny_main import controller_inputs, run_tiny_main, write_tiny_data_dir
from helpers.tiny_refinement import CallTrace

from relax.refinement import full_refinement

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("stop_requested", [False, True])
def test_a_requested_significance_dump_stop_exits_cleanly(monkeypatch, caplog, stop_requested):
    """A coarse-significance dump that completes its target stops the run with status 0 only when its stop was
    requested."""
    completed = type("SignificanceDumpComplete", (RuntimeError,), {})

    def main(command=None):
        raise completed("target written")

    monkeypatch.setattr(full_refinement, "main", main)
    monkeypatch.delenv("RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", raising=False)
    if not stop_requested:
        with pytest.raises(completed):
            full_refinement.run_from_command_line("refine")
        return
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "1")
    with caplog.at_level(logging.INFO, logger=full_refinement.logger.name), pytest.raises(SystemExit) as stop:
        full_refinement.run_from_command_line("refine")
    assert stop.value.code == 0
    assert any("pass-2/M-step work" in record.getMessage() for record in caplog.records)


def test_the_controller_gets_the_active_max_significants(monkeypatch, tmp_path):
    """Main hands the controller the runtime resolver's active value, not the saved sentinel."""
    trace = CallTrace(monkeypatch).wrap(full_refinement.command_options, "resolve_relion_runtime_controls", "controls")
    inputs = controller_inputs(monkeypatch, tmp_path, "refine", "--max_significants", "9")
    (controls,) = trace.calls("controls")
    assert controls.kwargs["max_significants"] == 9
    active = controls.result.max_significants_resolution["active_max_significants"]
    assert inputs["options"].adaptive.max_significants == active


def test_the_benchmark_ledger_records_the_max_significants_resolution(monkeypatch, tmp_path):
    trace = CallTrace(monkeypatch).wrap(full_refinement.command_options, "resolve_relion_runtime_controls", "controls")
    ledger = tmp_path / "ledger.json"
    run_tiny_main(monkeypatch, tmp_path, "refine", "--max_iter", "1", "--max_significants", "9",
                  "--benchmark_ledger_json", ledger)
    resolution = trace.calls("controls")[0].result.max_significants_resolution
    assert json.loads(ledger.read_text())["max_significants_resolution"] == json.loads(json.dumps(resolution))


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--firstiter_cc", "--apply-initial-lowpass", "--init_resolution", "30"], 30.0),
        (["--firstiter_cc", "--no-apply-initial-lowpass", "--init_resolution", "30"], None),
        (["--no-firstiter_cc", "--apply-initial-lowpass", "--init_resolution", "30"], None),
    ],
)
def test_firstiter_cc_passes_relion_cli_ini_high_to_refinement_loop(monkeypatch, tmp_path, arguments, expected):
    """RELION ``--firstiter_cc`` and ``--ini_high`` are distinct knobs.

    A fresh run has no optimiser command: the post-iteration-1 low-pass uses the start-up low-pass
    ``ini_high`` (``--apply-initial-lowpass`` at ``--init_resolution``), and only under ``--firstiter_cc``;
    ``--init_resolution`` alone is not an ``ini_high``.
    """
    parity = controller_inputs(monkeypatch, tmp_path, "refine", *arguments)["options"].parity
    assert parity.relion_firstiter_ini_high_angstrom == expected
    assert parity.emulate_relion_firstiter_cc is ("--firstiter_cc" in arguments)


@pytest.mark.parametrize(
    ("option", "field"),
    [
        ("--stop_after_local_search", "stop_after_local_search"),
        ("--stop_after_local_search_score_only", "stop_after_local_search_score_only"),
        ("--stop_after_local_search_profile", "stop_after_local_search_profile"),
    ],
)
def test_local_search_probe_switches_pass_to_refinement_loop(monkeypatch, tmp_path, option, field):
    assert getattr(controller_inputs(monkeypatch, tmp_path, "refine")["options"].local_search, field) is False
    assert getattr(controller_inputs(monkeypatch, tmp_path / "on", "refine", option)["options"].local_search, field) is True


def test_the_intermediates_dump_reaches_the_refinement_loop_as_its_observer(monkeypatch, tmp_path):
    from relax.diagnostics.observers import IntermediatesObserver

    assert controller_inputs(monkeypatch, tmp_path, "refine")["observer"] is None
    for extra, skip in (((), False), (("--save_intermediates_skip_unregularized",), True)):
        dump = tmp_path / f"dump_{skip}"
        observer = controller_inputs(
            monkeypatch, tmp_path / f"on_{skip}", "refine", "--save_intermediates_dir", str(dump), *extra,
        )["observer"]
        assert isinstance(observer, IntermediatesObserver)
        assert observer.directory == str(dump) and observer.skip_unregularized is skip


@pytest.mark.parametrize(
    ("command", "arguments", "message"),
    [
        ("refine", ["--diagnostic_single_half"], "only valid with --stop_after_local_search"),
        ("class3d", ["--n_classes", "2", "--diagnostic_single_half", "--stop_after_local_search"], "K=1-only"),
        ("refine", ["--n_classes", "2"], "use relax class3d"),
        ("refine", ["--no-relion-half-sets-from-input"], "K=1 auto-refine uses RELION's half sets"),
        ("refine", ["--perturb-replay-restart-state-iterations", "3"], "requires --perturb_replay_relion_dir"),
        ("refine", ["--perturb-replay-restart-provenance", "<DATA>/particles.star"],
         "requires --perturb-replay-restart-state-iterations"),
    ],
)
def test_the_command_refuses_unsupported_combinations(monkeypatch, tmp_path, command, arguments, message):
    with pytest.raises(SystemExit, match=message):
        controller_inputs(monkeypatch, tmp_path, command, *arguments, n_classes=2)


def test_the_live_initial_noise_diagnostic_is_refused_outside_a_relion_seeded_cold_start(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_K1_RELION_LIVE_INITIAL_NOISE", "1")
    with pytest.raises(ValueError, match="strict fresh K=1 cold start: relion_init_dir is required$"):
        controller_inputs(monkeypatch, tmp_path, "refine")
    with pytest.raises(ValueError, match="n_classes must equal 1; relion_init_dir is required; RELION half-set"):
        controller_inputs(monkeypatch, tmp_path / "k2", "class3d", "--n_classes", "2", n_classes=2)


def test_diagnostic_single_half_empties_half_two(monkeypatch, tmp_path):
    # Run files need both halves (they partition the particle table), so the probe writes none.
    inputs = controller_inputs(monkeypatch, tmp_path, "refine", "--diagnostic_single_half", "--stop_after_local_search",
                               "--write-iteration-every", "0")
    half1, half2 = inputs["experiment_datasets"]
    assert half1.n_units > 0 and half2.n_units == 0


def test_k1_half_sets_are_relions_and_class3d_uses_all_data_once(monkeypatch, tmp_path):
    """No seeded NumPy split: K=1 takes RELION's random halves of the input STAR, Class3D every particle once."""
    from relax.relion.input_particle_table import glibc_rand_sequence

    k1 = controller_inputs(monkeypatch, tmp_path / "k1", "refine")["experiment_datasets"]
    halves = glibc_rand_sequence(42, 12) % 2 + 1
    assert [half.n_units for half in k1] == [int(np.sum(halves == 1)), int(np.sum(halves == 2))]
    trace = CallTrace(monkeypatch).wrap(
        full_refinement.input_particle_table, "prepare_class3d_particle_layout", "class3d_layout",
    )
    k2 = controller_inputs(monkeypatch, tmp_path / "k2", "class3d", "--n_classes", "2", n_classes=2)
    assert len(trace.calls("class3d_layout")) == 1
    assert sum(half.n_units for half in k2["experiment_datasets"]) == 12


def test_native_group_ids_are_available_to_k_class_refinement(monkeypatch, tmp_path):
    """Class3D gets the physical group layout too: group ids per half, the group and optics-group counts."""
    trace = CallTrace(monkeypatch).wrap(
        full_refinement.input_particle_table, "prepare_particle_group_layout", "groups",
    )
    replay = controller_inputs(monkeypatch, tmp_path, "class3d", "--n_classes", "2", n_classes=2)["options"].replay
    (groups,) = trace.calls("groups")
    layout = groups.result.layout
    assert replay.init_group_ids == list(layout.group_ids_per_half)
    assert replay.init_group_count == layout.n_groups and replay.init_relion_optics_group_count == layout.n_optics_groups


def test_init_noise_from_npz_replaces_the_startup_estimate(monkeypatch, tmp_path):
    """A fresh run estimates the start-up noise from the images; --init_noise_from_npz loads an archive's."""
    from relax.helpers import iteration_history
    from relax.refinement import startup_noise

    archive = run_tiny_main(monkeypatch, tmp_path, "refine", "--max_iter", "1", output="first") / "refinement_results.npz"
    trace = CallTrace(monkeypatch).wrap(startup_noise, "prepare_startup_noise", "estimate")
    fresh = controller_inputs(monkeypatch, tmp_path / "fresh", "refine")
    assert len(trace.calls("estimate")) == 1
    assert_matches(np.asarray(fresh["init_noise_variance"]), np.asarray(trace.calls("estimate")[0].result.pixel_variance))
    loaded = controller_inputs(monkeypatch, tmp_path / "loaded", "refine", "--init_noise_from_npz", archive)
    assert len(trace.calls("estimate")) == 1
    from recovar.reconstruction import noise as recon_noise

    radial = iteration_history._load_init_noise_radial_npz(str(archive), "last")["noise_radial"]
    expected = recon_noise.make_radial_noise(radial, loaded["experiment_datasets"][0].image_shape)
    assert_matches(np.asarray(loaded["init_noise_variance"]), np.asarray(expected))


def test_the_command_writes_its_archive_maps_and_history(monkeypatch, tmp_path):
    """Main writes the archive (with the refinement and class history) and the final maps."""
    k1 = run_tiny_main(monkeypatch, tmp_path, "refine", "--max_iter", "1", output="k1")
    with np.load(k1 / "refinement_results.npz", allow_pickle=True) as archive:
        assert {"fsc_iter_000", "perturb_replay_restart_state_iterations",
                "perturb_replay_restart_provenance_path", "perturb_replay_restart_provenance_sha256"} <= set(archive)
    assert {"final_half1.mrc", "final_half2.mrc", "final_merged.mrc"} <= {path.name for path in k1.iterdir()}
    data = write_tiny_data_dir(tmp_path / "data2", n_classes=2)
    k2 = run_tiny_main(monkeypatch, tmp_path, "class3d", "--max_iter", "1", "--n_classes", "2", data=data, output="k2")
    with np.load(k2 / "refinement_results.npz", allow_pickle=True) as archive:
        assert {"class_weights", "class_assignments_iter_000"} <= set(archive)


@pytest.mark.parametrize(("command", "n_classes", "join"), [("refine", 1, 40.0), ("class3d", 2, -1.0)])
def test_run_files_record_the_runs_mask_edge_and_join_resolution(monkeypatch, tmp_path, command, n_classes, join):
    # RELION writes its live values to optimiser_general (ml_optimiser.cpp:1660-1665): width_mask_edge from
    # --maskedge, and low_resol_join_halves 40 from the Refine3D job (pipeline_jobs.cpp:4509) or the binary's
    # default -1 for Class3D (ml_optimiser.cpp:895). relax wrote 5 and 40 whatever the run did.
    data = write_tiny_data_dir(tmp_path / "data", n_classes=n_classes)
    arguments = ["--max_iter", "1", "--particle_diameter_ang", "40", "--width_mask_edge_px", "7"]
    if n_classes > 1:
        arguments += ["--n_classes", str(n_classes)]
    out = run_tiny_main(monkeypatch, tmp_path, command, *arguments, n_classes=n_classes, data=data)
    text = (out / "run_it001_optimiser.star").read_text()
    values = dict(line.split()[:2] for line in text.splitlines() if line.startswith("_rln") and len(line.split()) >= 2)
    assert int(values["_rlnWidthMaskEdge"]) == 7
    assert float(values["_rlnJoinHalvesUntilThisResolution"]) == join


def test_model_star_names_the_scale_groups_as_relion(monkeypatch, tmp_path):
    # RELION names each scale group by rlnGroupName or, without it, the micrograph name after its job
    # directory (exp_model.cpp:926-955), numbers them by first appearance in micrograph-sorted order
    # (exp_model.cpp:900-901) and writes those names to model_groups (ml_model.cpp:780, 1146); relax wrote
    # group_N. The tiny data put each particle on its own micrograph "1", "2", ...
    import starfile

    data = write_tiny_data_dir(tmp_path / "data")
    micrographs = starfile.read(data / "particles.star")["particles"]["rlnMicrographName"].astype(str)
    expected = sorted(set(micrographs), key=str.encode)
    out = run_tiny_main(monkeypatch, tmp_path, "refine", "--max_iter", "1", data=data)
    groups = starfile.read(out / "run_it001_half1_model.star")["model_groups"]
    assert groups["rlnGroupName"].astype(str).tolist() == expected
