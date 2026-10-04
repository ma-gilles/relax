"""CPU checks of the test-tier runner, receipts and pinned-output comparison; nothing is submitted."""

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts import em_tier_noise_envelope, em_tier_pinned, run_test_tier, write_test_receipt

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def test_medium_plan_covers_fast_tier_sweep_and_end_to_end():
    items = {i.name: i for i in run_test_tier.plan("medium", REPO_ROOT, "HEAD")}
    collected = run_test_tier.fast_cases(REPO_ROOT, sys.executable)
    assert len(collected) >= 10 and set(collected) <= set(items)
    assert {"cpu_fast_guard", "cpu_merge_units", "vdam_k1_50k", "e2e_k1_5k_standalone"} <= set(items)
    swept = [a for i in items.values() if i.name.startswith("unit_") for a in i.argv if a.endswith(".py")]
    assert len(swept) == len(set(swept)), "a sweep file is in two shards"
    assert run_test_tier.FAST not in swept and run_test_tier.E2E not in swept
    for isolated in run_test_tier.ISOLATE:
        shard = [i for i in items.values() if isolated in i.argv]
        assert len(shard) == 1 and [a for a in shard[0].argv if a.endswith(".py")] == [isolated]
    assert all(items[c].required for c in collected)
    assert not any(i.required for n, i in items.items() if n.startswith("unit_"))


def test_smoke_plan_covers_local_search_and_global_k1():
    items = {i.name: i for i in run_test_tier.plan("smoke", REPO_ROOT, "HEAD")}
    nodes = [a.split("::")[1] for a in items["replays"].argv if "::" in a]
    assert nodes == [run_test_tier.FAST_CASES[c] for c in run_test_tier.SMOKE_REPLAYS]
    assert {"k1_local_replay", "k1_adaptive_replay", "kclass_replay"} == set(run_test_tier.SMOKE_REPLAYS)


def test_selected_coarse_engine_reaches_gpu_items_only():
    items = [
        run_test_tier.Item("guard", ["true"], False, 1),
        run_test_tier.Item("replays", ["true"], True, 1),
    ]
    run_test_tier.select_coarse_engine(items, "auto")
    assert items[0].env == {}
    assert items[1].env["RELAX_TIER_COARSE_ENGINE"] == "auto"
    run_test_tier.select_coarse_engine(items, "gemm_dense")
    assert items[0].env == {}
    assert items[1].env["RELAX_TIER_COARSE_ENGINE"] == "gemm_dense"
    with pytest.raises(ValueError, match="unknown tier coarse engine"):
        run_test_tier.select_coarse_engine(items, "misspelled")


def test_long_dry_run_never_prepares_completion_or_builds(tmp_path, monkeypatch):
    source = {"head": "abc", "dirty": False, "diff_sha256": "empty", "untracked": {}, "src": str(tmp_path)}
    monkeypatch.setattr(run_test_tier, "freeze_source", lambda _root: source)
    monkeypatch.setattr(run_test_tier, "verify_fixtures", lambda *_args: None)
    monkeypatch.setattr(run_test_tier, "plan", lambda *_args: [run_test_tier.Item("global", ["true"], True, 1)])

    def forbidden(*_args):
        raise AssertionError("a dry-run must not prepare completion or build natives")

    monkeypatch.setattr(run_test_tier, "prepare_long", forbidden)
    monkeypatch.setattr(run_test_tier, "build_natives", forbidden)
    root = tmp_path / "dry"
    args = Namespace(tier="long", run_root=root, base="HEAD", coarse_engine="gemm_hybrid", dry_run=True)
    assert run_test_tier.cmd_submit(args) == 0
    spec = json.loads((root / "PLAN.json").read_text())
    assert spec["coarse_engine"] == "gemm_hybrid"
    assert spec["items"][0]["env"]["RELAX_TIER_COARSE_ENGINE"] == "gemm_hybrid"


def test_touched_gpu_tests_follow_imports(tmp_path):
    tests = tmp_path / "tests" / "unit"
    tests.mkdir(parents=True)
    (tmp_path / "tests" / "tiers").mkdir()
    (tests / "test_a.py").write_text(
        "import pytest\nfrom relax.local import local_em_engine\npytestmark = pytest.mark.gpu\n"
    )
    (tests / "test_b.py").write_text("import pytest\nimport relax.cuda.kernels\npytestmark = pytest.mark.gpu\n")
    (tests / "test_c.py").write_text("from relax.local import local_em_engine\n")  # no gpu marker
    run_test_tier.TIERS_DIR, saved = tmp_path / "tests" / "tiers", run_test_tier.TIERS_DIR
    try:
        (run_test_tier.TIERS_DIR / "gpu_path_map.json").write_text(json.dumps({"map": {"docs/*": ["tests/unit/x.py"]}}))
        assert run_test_tier.touched_gpu_tests(tmp_path, ["relax/local/local_em_engine.py"]) == ["tests/unit/test_a.py"]
        assert run_test_tier.touched_gpu_tests(tmp_path, ["relax/cuda/relax_kernels.cu"]) == ["tests/unit/test_b.py"]
        assert run_test_tier.touched_gpu_tests(tmp_path, ["docs/a.md"]) == ["tests/unit/x.py"]
    finally:
        run_test_tier.TIERS_DIR = saved


def test_skipped_required_item_fails(tmp_path):
    item = run_test_tier.Item(
        "case", [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", str(tmp_path / "t.py")], False, 1
    )
    (tmp_path / "t.py").write_text("import pytest\n\ndef test_x():\n    pytest.skip('missing')\n")
    result = run_test_tier.run_item(item, tmp_path / "run", tmp_path, None)
    assert result["rc"] == 0 and result["junit"]["skipped"] == 1 and result["status"] == "fail"
    item.required = False
    assert run_test_tier.run_item(item, tmp_path / "run2", tmp_path, None)["status"] == "pass"


def test_receipt_is_written_and_appended(tmp_path):
    (tmp_path / "PLAN.json").write_text(json.dumps({"coarse_engine": "gemm_hybrid",
                                                 "source": {"head": "abc", "dirty": False, "diff_sha256": "d"}}))
    log = tmp_path / "receipts.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "write_test_receipt.py"),
            "--run-root",
            str(tmp_path),
            "--tier",
            "smoke",
            "--status",
            "pass",
            "--job",
            "123",
            "--gpu-model",
            "NVIDIA A100-SXM4-80GB",
        ],
        check=True,
        env={"RELAX_TEST_RECEIPTS": str(log), "PATH": "/usr/bin:/bin"},
        capture_output=True,
    )
    receipt = json.loads((tmp_path / "RECEIPT.json").read_text())
    assert (receipt["sha"], receipt["tier"], receipt["status"], receipt["job"]) == ("abc", "smoke", "pass", "123")
    assert receipt["gpu_model"] == "NVIDIA A100-SXM4-80GB"
    assert receipt["coarse_engine"] == "gemm_hybrid"
    assert json.loads(log.read_text().splitlines()[0]) == receipt


def _completion_record(choice: str) -> dict:
    dense = choice == "gemm_dense"
    return {
        "requested": choice, "resolved": choice, "strategy": "dense" if dense else "hybrid",
        "fine_engine": None if dense else "resident", "images": 1, "classes": 1,
        "rotations": 2, "translations": 2, "hypotheses_per_image": 4,
        "coarse_candidates_per_image": 4, "fine_candidates_per_image": 4 if dense else 8,
        "evaluated_fine_candidates_total": 4,
        "selected_fine_candidates_total": 4,
        "selected_fine_candidates_known": True,
        "current_size": 32, "reconstruction_current_size": 64,
        "score_mode": "gaussian", "posterior_policy": "gaussian", "pruned": not dense,
        "precision": {"score": "float32", "projection": "float32", "mstep": "float32"},
    }


@pytest.mark.parametrize("arm", ["completion_k1", "completion_k4"])
@pytest.mark.parametrize("choice", ["gemm_hybrid", "gemm_dense"])
def test_selected_completion_requires_executed_record(tmp_path, arm, choice):
    output = tmp_path / "completion" / ("k1_100k256_recovar" if arm == "completion_k1" else "k4_100k256_recovar")
    output.mkdir(parents=True)
    np.savez(output / "refinement_results.npz", coarse_engine_trajectory=json.dumps([[_completion_record(choice)]]))
    item = run_test_tier.Item(arm, [sys.executable, "-c", "pass"], True, 1,
                              env={"RELAX_TIER_COARSE_ENGINE": choice})
    good = run_test_tier.run_item(item, tmp_path, REPO_ROOT, "test-gpu")
    assert good["status"] == "pass" and good["coarse_audit_rc"] == 0
    (output / "refinement_results.npz").unlink()
    bad = run_test_tier.run_item(item, tmp_path, REPO_ROOT, "test-gpu")
    assert bad["rc"] == 0 and bad["status"] == "fail" and bad["coarse_audit_rc"] != 0
    assert "coarse-engine execution audit failed" in (tmp_path / "items" / arm / "log.txt").read_text()


def test_auto_completion_skips_record_audit(tmp_path):
    item = run_test_tier.Item("completion_k1", [sys.executable, "-c", "pass"], True, 1,
                              env={"RELAX_TIER_COARSE_ENGINE": "auto"})
    result = run_test_tier.run_item(item, tmp_path, REPO_ROOT, "test-gpu")
    assert result["status"] == "pass" and "coarse_audit_rc" not in result


def test_completion_audit_failure_blocks_dependent_summary(tmp_path):
    items = [
        run_test_tier.Item("completion_k1", [sys.executable, "-c", "pass"], True, 1,
                           env={"RELAX_TIER_COARSE_ENGINE": "gemm_dense"}),
        run_test_tier.Item("completion_summary", [sys.executable, "-c", "pass"], False, 1,
                           after=["completion_k1"]),
    ]
    results = {r["name"]: r for r in run_test_tier.execute(items, tmp_path, REPO_ROOT, ["test-gpu"])}
    assert results["completion_k1"]["status"] == "fail"
    assert results["completion_summary"]["rc"] is None and results["completion_summary"]["status"] == "fail"


def test_selected_mode_reaches_summary_and_older_receipt_defaults_auto(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "PLAN.json").write_text(json.dumps({"tier": "long", "coarse_engine": "gemm_dense",
                                                  "source": {"head": "abc"}, "items": []}))
    monkeypatch.setattr(run_test_tier, "check_native_sources", lambda *_args: None)
    monkeypatch.setattr(run_test_tier, "check_imports", lambda *_args: ({}, None))
    monkeypatch.setattr(run_test_tier, "gpu_models", lambda *_args: {"test-gpu": "A100"})
    monkeypatch.setattr(run_test_tier, "execute", lambda *_args: [])
    monkeypatch.setattr(run_test_tier, "write_receipt", lambda *_args: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "test-gpu")
    assert run_test_tier.cmd_run(Namespace(run_root=tmp_path, tier="long")) == 0
    assert json.loads((tmp_path / "SUMMARY.json").read_text())["coarse_engine"] == "gemm_dense"
    (tmp_path / "PLAN.json").write_text(json.dumps({"source": {"head": "abc"}}))
    assert write_test_receipt.main(["--run-root", str(tmp_path), "--tier", "long", "--status", "pass",
                                    "--job", "123", "--gpu-model", "A100"]) == 0
    assert json.loads((tmp_path / "RECEIPT.json").read_text())["coarse_engine"] == "auto"


def _case(auc, shell):
    return {"summary": {"min_fsc_auc": auc, "mean_fsc_auc": auc, "min_shell_fsc_in_band": shell}}


def test_pinned_comparison_reports_until_approved():
    pinned = {"cases": {c: _case(0.9999, 0.999) for c in em_tier_pinned.TIER_CASES["smoke"]}}
    now = {"cases": {c: _case(0.9999, 0.999) for c in em_tier_pinned.TIER_CASES["smoke"]}}
    now["cases"]["k1_local_replay"] = _case(0.9990, 0.999)
    thresholds = {"approved": False, "pinned_tolerance": {"min_fsc_auc_drop": 1e-4, "min_shell_fsc_drop": 1e-3}}
    verdict = em_tier_pinned.compare("smoke", now, pinned, thresholds)["verdict"]
    assert verdict["status"] == "fail" and not verdict["enforced"] and "k1_local_replay" in verdict["failures"][0]
    thresholds["approved"] = True
    assert em_tier_pinned.compare("smoke", now, pinned, thresholds)["verdict"]["enforced"]
    del now["cases"]["kclass_replay"]
    assert any(
        "kclass_replay" in f for f in em_tier_pinned.compare("smoke", now, pinned, thresholds)["verdict"]["failures"]
    )


def test_fixture_sets_of_every_tier_are_in_the_manifest():
    from helpers.em_fixtures import _manifest

    sets = set(_manifest())
    for tier, names in run_test_tier.FIXTURE_SETS.items():
        assert set(names) <= sets, (tier, sorted(set(names) - sets))


def test_pinned_outputs_are_kept_and_compared_per_gpu_model(tmp_path, monkeypatch):
    monkeypatch.setattr(em_tier_pinned, "PINNED", tmp_path / "pinned.json")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"approved": False, "pinned_tolerance": {"min_fsc_auc_drop": 1e-5, "min_shell_fsc_drop": 1e-4}}))
    monkeypatch.setattr(em_tier_pinned, "THRESHOLDS", thresholds)
    cases = {c: _case(0.9999, 0.999) | {"pairs": {}} for c in em_tier_pinned.TIER_CASES["medium"]}
    fsc = tmp_path / "fsc.json"
    fsc.write_text(json.dumps({"cases": cases}))
    receipt = tmp_path / "RECEIPT.json"
    summary = tmp_path / "SUMMARY.json"
    items = [{"name": c, "status": "pass"} for c in em_tier_pinned.TIER_CASES["medium"]]
    summary.write_text(json.dumps({"items": items + [{"name": "unit_05", "status": "fail"}]}))
    for model in ("NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"):
        receipt.write_text(json.dumps({"status": "fail", "dirty": False, "gpu_model": model, "sha": "s", "job": "j",
                                       "summary": str(summary)}))
        assert em_tier_pinned.main(["regenerate", "--fsc", str(fsc), "--receipt", str(receipt)]) == 0
    stored = json.loads((tmp_path / "pinned.json").read_text())
    assert sorted(stored["models"]) == ["NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"]
    source = stored["models"]["NVIDIA H100 80GB HBM3"]["cases"]["k1_replay"]["source"]
    assert source["other_failed_items"] == ["unit_05"] and source["tier_status"] == "fail"
    out = tmp_path / "cmp.json"
    em_tier_pinned.main(["compare", "--tier", "smoke", "--fsc", str(fsc), "--gpu-model", "NVIDIA H100 80GB HBM3", "--output", str(out)])
    assert json.loads(out.read_text())["verdict"]["status"] == "pass"
    em_tier_pinned.main(["compare", "--tier", "smoke", "--fsc", str(fsc), "--gpu-model", "NVIDIA L40S", "--output", str(out)])
    assert json.loads(out.read_text())["verdict"]["status"] == "not_configured"
    receipt.write_text(json.dumps({"status": "pass", "dirty": False, "gpu_model": "A100,H100", "summary": str(summary)}))
    with pytest.raises(SystemExit, match="one model"):
        em_tier_pinned.main(["regenerate", "--fsc", str(fsc), "--receipt", str(receipt)])
    summary.write_text(json.dumps({"items": [dict(i, status="fail") if i["name"] == "k1_replay" else i for i in items]}))
    receipt.write_text(json.dumps({"status": "fail", "dirty": False, "gpu_model": "H100", "summary": str(summary)}))
    with pytest.raises(SystemExit, match="k1_replay"):
        em_tier_pinned.main(["regenerate", "--fsc", str(fsc), "--receipt", str(receipt)])


def test_long_plan_packs_the_arms_into_one_job_with_dependencies(tmp_path):
    items = {i.name: i for i in run_test_tier.plan("long", REPO_ROOT, "HEAD", tmp_path)}
    gpu = [n for n, i in items.items() if i.gpu]
    assert sorted(gpu) == sorted(run_test_tier.LONG_ARM_SECONDS)
    assert items["completion_summary"].after == ["completion_k1", "completion_k4"]
    assert set(items["bands"].after) >= {"completion_k1", "completion_k4", "long_kclass", "long_k1_standalone"}
    assert run_test_tier.TIER_JOB["long"]["gpus"] == 4


def test_executor_honours_dependencies_and_skips_after_failures(tmp_path):
    ok = [sys.executable, "-c", "pass"]
    bad = [sys.executable, "-c", "raise SystemExit(3)"]
    items = [
        run_test_tier.Item("setup", ok, False, 1),
        run_test_tier.Item("arm", ok, True, 5, after=["setup"]),
        run_test_tier.Item("broken", bad, True, 4),
        run_test_tier.Item("after_broken", ok, True, 3, after=["broken"]),
        run_test_tier.Item("summary", ok, False, 1, after=["arm"]),
    ]
    results = {r["name"]: r for r in run_test_tier.execute(items, tmp_path, tmp_path, ["g0", "g1"])}
    assert [results[n]["status"] for n in ("setup", "arm", "summary")] == ["pass"] * 3
    assert results["broken"]["status"] == "fail" and results["after_broken"]["rc"] is None


def test_tier_jobs_follow_the_slurm_sizing_rule(tmp_path):
    for tier, job in run_test_tier.TIER_JOB.items():
        assert job["mem_gb"] / job["gpus"] <= 180
        for queue in run_test_tier.QUEUES:
            (tmp_path / "src").mkdir(exist_ok=True)
            text = run_test_tier.write_sbatch(tmp_path, tier, tmp_path, queue, "any").read_text()
            assert f"--gres=gpu:{job['gpus']}" in text and "--exclusive" not in text
            assert f"--cpus-per-task={8 * job['gpus']}" in text
    text = run_test_tier.write_sbatch(tmp_path, "medium", tmp_path, "general", "a100").read_text()
    assert "--constraint=a100,gpu80" in text and "--partition" not in text


def test_noise_envelope_takes_the_largest_same_code_difference(tmp_path, monkeypatch, capsys):
    runs = {
        "a": {"k1_replay": {"gate.min_fsc_auc": 0.9, "gate.pmax_mean_abs": 0.1}},
        "b": {"k1_replay": {"gate.min_fsc_auc": 0.9 + 2e-9, "gate.pmax_mean_abs": 0.1}},
        "c": {"k1_replay": {"gate.min_fsc_auc": 0.9 - 5e-9, "gate.pmax_mean_abs": 0.1}},
        "cand": {"k1_replay": {"gate.min_fsc_auc": 0.9 - 1e-6, "gate.pmax_mean_abs": 0.1}},
    }
    monkeypatch.setattr(em_tier_noise_envelope, "run_metrics", lambda root: runs[root.name])
    out = tmp_path / "envelope.json"
    em_tier_noise_envelope.main(["build", "--pair", "p1:h100:a:b", "--pair", "p2:h100:a:c", "--output", str(out)])
    row = json.loads(out.read_text())["cases"]["k1_replay"]["gate.min_fsc_auc"]
    assert row["n_pairs"] == 2 and row["max_abs_diff"] == pytest.approx(5e-9)
    assert row["noise_limit"] == pytest.approx(5e-8)
    pmax = json.loads(out.read_text())["cases"]["k1_replay"]["gate.pmax_mean_abs"]
    assert pmax["max_abs_diff"] == 0.0 and pmax["noise_limit"] == em_tier_noise_envelope.NOISE_FLOOR
    em_tier_noise_envelope.main(["check", "--control", "a", "--candidate", "cand", "--envelope", str(out)])
    assert "1 metric(s) outside" in capsys.readouterr().out


def test_smoke_defers_touched_gpu_files_over_its_budget(monkeypatch):
    seconds = {"tests/unit/a.py": 30, "tests/unit/b.py": 50, "tests/unit/big.py": 1174}
    monkeypatch.setattr(run_test_tier, "_durations", lambda: seconds)
    kept, deferred = run_test_tier.smoke_touched_split(
        ["tests/unit/big.py", "tests/unit/b.py", "tests/unit/a.py", "tests/unit/new.py"], replay_seconds=200
    )
    assert kept == ["tests/unit/a.py", "tests/unit/b.py", "tests/unit/new.py"]  # 10 + 30 + 50 <= 100
    assert deferred == ["tests/unit/big.py"]


def test_smoke_always_runs_the_changes_own_gpu_test_files(monkeypatch):
    """An added or modified GPU test file runs in smoke even when the importers fill the budget."""
    seconds = {"tests/unit/a.py": 30, "tests/unit/b.py": 50, "tests/unit/own_slow.py": 400}
    monkeypatch.setattr(run_test_tier, "_durations", lambda: seconds)
    touched = ["tests/unit/a.py", "tests/unit/b.py", "tests/unit/new.py", "tests/unit/own_slow.py", "tests/unit/z.py"]
    changed = ["relax/helpers/projection.py", "tests/unit/new.py", "tests/unit/own_slow.py"]
    kept, deferred = run_test_tier.smoke_touched_split(touched, replay_seconds=200, changed=changed)
    assert set(kept) >= {"tests/unit/new.py", "tests/unit/own_slow.py"}
    # The own files used the room (100 s): every importer is deferred to medium.
    assert deferred == ["tests/unit/a.py", "tests/unit/b.py", "tests/unit/z.py"]


def test_gpu_files_include_skipif_gated_resident_tests():
    files = set(run_test_tier.gpu_test_files(REPO_ROOT))
    assert {"tests/unit/test_resident_pass2_driver.py", "tests/unit/test_resident_local_pass2.py"} <= files
    selected = run_test_tier.touched_gpu_tests(REPO_ROOT, ["relax/sparse_pass2/resident_pass2.py"])
    assert "tests/unit/test_resident_pass2_driver.py" in selected
    assert "tests/unit/initial_model/test_audit_vdam_repeat_panel.py" not in files


# Test files whose CUDA tests are gated by a skipif, a runtime backend check or a GPU fixture
# rather than the gpu marker. They must still run in the GPU tiers.
SKIPIF_GATED_GPU_FILES = (
    "tests/unit/test_em_stage_glue_programs.py",
    "tests/unit/test_local_backprojection_relion_f32.py",
    "tests/unit/test_normalized_cc_replay.py",
    "tests/unit/test_resident_local_pass2.py",
    "tests/unit/test_resident_pass2_driver.py",
)


def test_every_cuda_test_file_runs_in_the_gpu_tiers():
    gpu_files = set(run_test_tier.gpu_test_files(REPO_ROOT))
    assert set(SKIPIF_GATED_GPU_FILES) <= gpu_files
    # The medium sweep runs every test file, whatever its marker, with the GPU flags and the
    # opt-ins that would otherwise skip a CUDA test.
    items = [i for i in run_test_tier.plan("medium", REPO_ROOT, "HEAD") if i.name.startswith("unit_")]
    swept = {a for i in items for a in i.argv if a.endswith(".py")}
    assert gpu_files <= swept
    for item in items:
        assert "--run-gpu" in item.argv
        assert item.env["RELAX_RUN_CUDA_XHALF_TEST"] == "1"
        assert Path(item.env["RELAX_P4J_STAR_FIXTURE"]).is_file()


def test_a_new_case_is_pinned_alone_and_reported_until_then(tmp_path, monkeypatch):
    monkeypatch.setattr(em_tier_pinned, "PINNED", tmp_path / "pinned.json")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"approved": True, "pinned_tolerance": {"min_fsc_auc_drop": 1e-5, "min_shell_fsc_drop": 1e-4}}))
    monkeypatch.setattr(em_tier_pinned, "THRESHOLDS", thresholds)
    medium = em_tier_pinned.TIER_CASES["medium"]
    new = "k1_gui60_coldstart_standalone"
    cases = {c: _case(0.9999, 0.999) | {"pairs": {}} for c in medium}
    fsc = tmp_path / "fsc.json"
    fsc.write_text(json.dumps({"cases": cases}))
    summary = tmp_path / "SUMMARY.json"
    summary.write_text(json.dumps({"items": [{"name": c, "status": "pass"} for c in medium]}))
    receipt = tmp_path / "RECEIPT.json"
    receipt.write_text(json.dumps({"status": "pass", "dirty": False, "gpu_model": "H100", "sha": "a", "job": "1",
                                   "summary": str(summary)}))
    old = [c for c in medium if c != new]
    em_tier_pinned.main(["regenerate", "--fsc", str(fsc), "--receipt", str(receipt), "--cases", *old])
    out = tmp_path / "cmp.json"
    em_tier_pinned.main(["compare", "--tier", "medium", "--fsc", str(fsc), "--gpu-model", "H100", "--output", str(out)])
    result = json.loads(out.read_text())
    assert result["cases"][new]["status"] == "not_pinned" and result["verdict"]["status"] == "pass"
    receipt.write_text(json.dumps({"status": "pass", "dirty": False, "gpu_model": "H100", "sha": "b", "job": "2",
                                   "summary": str(summary)}))
    em_tier_pinned.main(["regenerate", "--fsc", str(fsc), "--receipt", str(receipt), "--cases", new])
    stored = json.loads((tmp_path / "pinned.json").read_text())["models"]["H100"]["cases"]
    assert stored[new]["source"]["sha"] == "b" and stored["k1_replay"]["source"]["sha"] == "a"


def test_a_case_is_pinned_from_a_named_repeat_item_with_a_note(tmp_path, monkeypatch):
    monkeypatch.setattr(em_tier_pinned, "PINNED", tmp_path / "pinned.json")
    case = "k1_os1_coldstart_standalone"
    fsc = tmp_path / "fsc.json"
    fsc.write_text(json.dumps({"cases": {case: _case(0.9998, 0.999) | {"pairs": {}}}}))
    summary = tmp_path / "SUMMARY.json"
    summary.write_text(json.dumps({"items": [{"name": "os1_rep3", "status": "pass"}, {"name": "os1_rep1", "status": "fail"}]}))
    receipt = tmp_path / "RECEIPT.json"
    receipt.write_text(json.dumps({"status": "fail", "dirty": False, "gpu_model": "H100", "sha": "c", "job": "3",
                                   "summary": str(summary)}))
    note = json.dumps({"pin_mode": "lowest of 6 same-code repeats"})
    em_tier_pinned.main(["regenerate", "--fsc", str(fsc), "--receipt", str(receipt), "--cases", case,
                         "--item", "os1_rep3", "--note", note])
    source = json.loads((tmp_path / "pinned.json").read_text())["models"]["H100"]["cases"][case]["source"]
    assert source["item"] == "os1_rep3" and source["pin_mode"] == "lowest of 6 same-code repeats"
    with pytest.raises(SystemExit, match="not passed"):
        em_tier_pinned.main(["regenerate", "--fsc", str(fsc), "--receipt", str(receipt), "--cases", case,
                             "--item", "os1_rep1"])


def test_stale_natives_are_refused(tmp_path, monkeypatch):
    from scripts import native_sources

    src = tmp_path / "src"
    for rel in ("relax/cuda/relax_kernels.cu", "relax/relion_bind/module.cpp", "scripts/build_test_natives.sh"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(rel)
    natives = tmp_path / "natives"
    natives.mkdir()
    assert "missing" in native_sources.check(natives, src)
    (natives / "NATIVE.json").write_text(json.dumps({"sha256": {}}))
    assert "no native_sources record" in native_sources.check(natives, src)
    assert native_sources.main(["record", str(natives), "--root", str(src)]) == 0
    assert native_sources.check(natives, src) is None
    (src / "relax/cuda/relax_kernels.cu").write_text("changed kernel")
    assert "stale natives" in native_sources.check(natives, src)
    # The tier runner refuses to start on such natives.
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "src").symlink_to(src)
    (run_root / "natives").symlink_to(natives)
    (run_root / "PLAN.json").write_text(json.dumps({"tier": "smoke", "items": []}))
    with pytest.raises(SystemExit, match="stale natives"):
        run_test_tier.main(["run", "smoke", "--run-root", str(run_root)])


def test_natives_compiled_from_a_shadowing_recovar_checkout_are_refused(tmp_path, monkeypatch):
    """A build started inside a recovar checkout compiles that checkout's kernels (python -m puts the working
    directory first on sys.path), whatever recovar commit is installed: natives_bab1082 (2026-10-03) lacked
    BackprojectIndexedRuntimeRadius. ``record`` resolves recovar as the build did, and ``check`` compares the
    compiled kernel sources with those of the recovar a run imports."""
    from scripts import native_sources

    src = tmp_path / "src"
    for rel in ("relax/cuda/relax_kernels.cu", "relax/relion_bind/module.cpp", "scripts/build_test_natives.sh"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(rel)
    checkout = tmp_path / "recovar_checkout"
    (checkout / "recovar" / "cuda").mkdir(parents=True)
    (checkout / "recovar" / "__init__.py").write_text("")
    (checkout / "recovar" / "cuda" / "cuda_backproject.cu").write_text("// an older kernel")
    shadowed, clean = tmp_path / "shadowed", tmp_path / "clean"
    for natives, cwd in ((shadowed, checkout), (clean, tmp_path)):
        natives.mkdir()
        monkeypatch.chdir(cwd)
        assert native_sources.main(["record", str(natives), "--root", str(src)]) == 0
    monkeypatch.chdir(tmp_path)
    recorded = json.loads((shadowed / "NATIVE.json").read_text())["recovar_kernels"]
    assert recorded["path"] == str((checkout / "recovar" / "cuda").resolve())
    assert "compiled the recovar kernels" in native_sources.check(shadowed, src)
    assert native_sources.check(clean, src) is None
    # Natives without the record cannot be verified and are refused too.
    record = json.loads((clean / "NATIVE.json").read_text())
    del record["recovar_kernels"]
    (clean / "NATIVE.json").write_text(json.dumps(record))
    assert "does not record which recovar kernel sources" in native_sources.check(clean, src)


def test_relax_must_be_imported_from_the_snapshot(tmp_path):
    import os

    from scripts import native_sources

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    files, reason = native_sources.check_imports(REPO_ROOT, env)
    assert reason is None and files["relax_from_snapshot"]
    assert files["relax"].startswith(str(REPO_ROOT.resolve())) and files["recovar"]
    # A snapshot elsewhere whose processes still import this checkout's relax is refused.
    files, reason = native_sources.check_imports(tmp_path, env)
    assert not files["relax_from_snapshot"] and "not from the snapshot" in reason


def test_every_fast_oracle_is_labelled_continue_or_uninterrupted():
    from scripts import em_tier_fsc

    sets = {case.relion_set for case in em_tier_fsc.CASES.values()}
    assert sets <= set(em_tier_fsc.ORACLE_RUN), sorted(sets - set(em_tier_fsc.ORACLE_RUN))
    assert set(em_tier_fsc.ORACLE_RUN.values()) <= {"uninterrupted", "continue"}


def test_local_gpus_follow_the_session_subset_within_1_to_3(monkeypatch):
    monkeypatch.delenv("RELAX_LOCAL_GPUS", raising=False)
    assert run_test_tier.local_gpus() == ("1", "2", "3")
    monkeypatch.setenv("RELAX_LOCAL_GPUS", "1,3")
    assert run_test_tier.local_gpus() == ("1", "3")
    for bad in ("0,1", "4", " , "):
        monkeypatch.setenv("RELAX_LOCAL_GPUS", bad)
        with pytest.raises(SystemExit, match="RELAX_LOCAL_GPUS"):
            run_test_tier.local_gpus()


def test_idle_local_gpu_skips_gpus_outside_the_session_subset(monkeypatch):
    listing = "1, GPU-one, 0, NVIDIA A100\n2, GPU-two, 0, NVIDIA A100\n3, GPU-three, 0, NVIDIA A100\n"

    def fake_check_output(command, text=True):
        return listing if "--query-gpu=index,uuid,memory.used,name" in command else ""

    monkeypatch.setattr(run_test_tier.subprocess, "check_output", fake_check_output)
    monkeypatch.setenv("RELAX_LOCAL_GPUS", "3")
    assert run_test_tier.idle_local_gpu() == "GPU-three"


def test_cpu_merge_units_name_existing_tests():
    """A renamed or deleted test must leave the tier's CPU merge list: pytest fails the whole item on one stale id."""
    for node in run_test_tier.CPU_MERGE_UNITS:
        path, *parts = node.split("::")
        source = (REPO_ROOT / path).read_text()
        for part in parts:
            assert f"def {part}(" in source or f"class {part}" in source, node
