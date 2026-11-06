"""The scores scripts/ppca_schedule/score_suite_run.py collects from a suite scoring work directory, and how it finds the suite's tools."""

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from scripts.ppca_schedule import score_suite_run as score

pytestmark = pytest.mark.unit

DATASET = "toy"
SCORER_STUB = """
    import json, sys
    from pathlib import Path
    options = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    scored = Path(options["--run"]) / f"suite_score_{Path(options['--checkpoint']).stem}"
    scored.mkdir(exist_ok=True)
    (scored / "summary.json").write_text(json.dumps({"mean_fsc_to_true_mean": 0.9}))
"""
FIT_STUB = """
    import json, os, sys
    from pathlib import Path
    work = Path(sys.argv[1])
    for scored in work.glob("suite_score_checkpoint_*"):
        (scored / "direct_fit.json").write_text(json.dumps({"pose_median_deg": 4.0, "frac_below_10": 0.99}))
    (work / "fit_environment.json").write_text(
        json.dumps({"PYTHONPATH": os.environ.get("PYTHONPATH", ""), "SUITE_DATASETS": os.environ["SUITE_DATASETS"]})
    )
"""


def write_script(path: Path, body: str) -> Path:
    """Write ``body`` (dedented) to ``path`` and return it."""
    path.write_text(textwrap.dedent(body).lstrip())
    return path


def make_site(tmp_path: Path, env_files: bool) -> dict:
    """A run with one checkpoint, a toy dataset and stub suite tools; the paths of all of them.

    ``env_files`` adds gpu_env.sh and cpu_env.sh (each defining PY; the CPU one also SOURCE) to the suite tools.
    """
    run = tmp_path / "run"
    run.mkdir()
    theta = np.concatenate([np.ones((4, 1)), np.full((4, 2), 2.0)], axis=1).astype(np.complex64)
    np.savez(run / "checkpoint_0010.npz", theta=theta)
    evaluation = tmp_path / "datasets" / DATASET / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "manifest.json").write_text(json.dumps({"true_rank": 2}))
    suite_tools, fit_tools = tmp_path / "suite_tools", tmp_path / "fit_tools"
    suite_tools.mkdir()
    fit_tools.mkdir()
    write_script(suite_tools / "suite_score.py", SCORER_STUB)
    write_script(fit_tools / "direct_fit.py", FIT_STUB)
    if env_files:
        (suite_tools / "gpu_env.sh").write_text(f"PY={sys.executable}\n")
        (suite_tools / "cpu_env.sh").write_text(f"PY={sys.executable}\nSOURCE=/pinned/checkout\n")
    return {"run": run, "datasets": tmp_path / "datasets", "suite_tools": suite_tools, "fit_tools": fit_tools}


def run_main(site: dict, *extra: str) -> dict:
    """Score checkpoint 10 of the toy run through ``main`` and return the JSON it writes."""
    output = site["run"] / "suite_scores.json"
    argv = [
        str(site["run"]),
        "--dataset", DATASET, "--arm", "toy_arm", "--seed", "1",
        "--datasets-root", str(site["datasets"]),
        "--suite-tools", str(site["suite_tools"]),
        "--fit-tools", str(site["fit_tools"]),
        "--checkpoints", "10",
        "-o", str(output),
        *extra,
    ]  # fmt: skip
    assert score.main(argv) == 0
    return json.loads(output.read_text())


def _checkpoint(directory, step, mean, loadings):
    theta = np.concatenate([np.asarray(mean)[:, None], np.asarray(loadings)], axis=1).astype(np.complex64)
    np.savez(directory / f"checkpoint_{step:04d}.npz", theta=theta)


def test_the_pc_power_is_the_loadings_over_the_mean_and_ignores_the_mean_sign(tmp_path):
    _checkpoint(tmp_path, 10, [1, 2j], [[3, 0], [0, 4j]])
    assert score.pc_over_mean_power(tmp_path / "checkpoint_0010.npz") == pytest.approx(25 / 5)
    (tmp_path / "negated").mkdir()
    copy = score.write_mean_negated_copy(tmp_path / "checkpoint_0010.npz", tmp_path / "negated")
    assert score.pc_over_mean_power(copy) == pytest.approx(25 / 5)


def test_a_scored_checkpoint_gives_the_summary_fields_the_direct_pose_and_the_pc_power(tmp_path):
    _checkpoint(tmp_path, 20, [1, 1], [[1], [1]])
    scored = tmp_path / "suite_score_checkpoint_0020"
    scored.mkdir()
    (scored / "summary.json").write_text(
        json.dumps({"mean_fsc_to_true_mean": 0.9, "kmeans_ari": 0.5, "pose_median_deg": 99.0, "checkpoint": "c.npz", "q": 3})
    )
    states = [{"state": 0, "fsc_direct_frame": 0.7}, {"state": 1, "fsc_direct_frame": 0.9}]
    (scored / "direct_fit.json").write_text(
        json.dumps(
            {"pose_median_deg": 4.0, "frac_below_10": 0.99, "state_fsc_mean_direct": 0.8, "rotation": [[1.0]], "states": states}
        )
    )
    row = score.collect_scores(tmp_path, 20)
    assert row == {
        "pc_over_mean_power": pytest.approx(1.0),
        "mean_fsc_to_true_mean": 0.9,
        "kmeans_ari": 0.5,
        "pose_direct_median_deg": 4.0,
        "pose_direct_within_10deg": 0.99,
        "state_fsc_direct": 0.8,
        "state_fsc_direct_min": 0.7,
    }
    assert score.per_state_fsc(tmp_path, 20) == [0.7, 0.9]


def test_a_rank_zero_checkpoint_takes_its_mean_fsc_from_the_h1_metrics(tmp_path):
    _checkpoint(tmp_path, 30, [1, 1], [[0], [0]])
    scored = tmp_path / "suite_score_checkpoint_0030"
    scored.mkdir()
    (scored / "direct_fit.json").write_text(json.dumps({"pose_median_deg": 4.5, "frac_below_10": 0.9}))
    (scored / "h1_metrics.json").write_text(json.dumps({"mean_fsc_raw_1_15_box64": 0.97, "first_shell_below_0p5": 40}))
    row = score.collect_scores(tmp_path, 30)
    assert row["mean_fsc_to_true_mean"] == 0.97 and row["first_shell_below_0p5"] == 40
    assert "state_fsc_direct" not in row and row["pc_over_mean_power"] == 0


def test_the_suites_own_environment_files_give_the_interpreters_and_the_fits_import_path(tmp_path):
    """Without --scorer-python or --fit-python, PY comes from gpu_env.sh and cpu_env.sh, and SOURCE is the fit's PYTHONPATH."""
    site = make_site(tmp_path, env_files=True)
    scores = run_main(site)
    assert scores["checkpoints"]["10"]["mean_fsc_to_true_mean"] == 0.9
    assert scores["checkpoints"]["10"]["pose_direct_median_deg"] == 4.0
    work = site["run"] / "suite_work" / f"{DATASET}__toy_arm_s1"
    seen = json.loads((work / "fit_environment.json").read_text())
    assert seen == {"PYTHONPATH": "/pinned/checkout", "SUITE_DATASETS": str(site["datasets"])}
    with np.load(work / "checkpoint_0010.npz") as copy:  # the scorer sees the checkpoint with its mean negated
        np.testing.assert_array_equal(copy["theta"][:, 0], -np.ones(4))
        np.testing.assert_array_equal(copy["theta"][:, 1:], np.full((4, 2), 2.0))


def test_explicit_interpreters_work_without_any_environment_file_and_keep_the_callers_import_path(
    tmp_path, monkeypatch
):
    site = make_site(tmp_path, env_files=False)
    monkeypatch.setenv("PYTHONPATH", "/callers/path")
    run_main(site, "--scorer-python", sys.executable, "--fit-python", sys.executable)
    work = site["run"] / "suite_work" / f"{DATASET}__toy_arm_s1"
    assert json.loads((work / "fit_environment.json").read_text())["PYTHONPATH"] == "/callers/path"


def test_a_missing_tool_is_reported_with_every_other_missing_one_before_anything_is_scored(tmp_path):
    site = make_site(tmp_path, env_files=False)
    (site["fit_tools"] / "direct_fit.py").unlink()
    with pytest.raises(FileNotFoundError) as error:
        run_main(site, "--gpu-env", str(tmp_path / "no_gpu_env.sh"), "--scorer-python", str(tmp_path / "no_python"))
    message = str(error.value)
    assert "direct_fit.py" in message and "no_gpu_env.sh" in message and "no_python" in message
    assert not (site["run"] / "suite_work").exists()


def test_no_interpreter_anywhere_fails_the_scorer_and_the_log_names_the_option(tmp_path):
    site = make_site(tmp_path, env_files=False)
    with pytest.raises(RuntimeError, match="suite scorer failed"):
        run_main(site)
    log = site["run"] / "suite_work" / f"{DATASET}__toy_arm_s1" / "checkpoint_0010.score.log"
    assert "--scorer-python" in log.read_text()


def test_a_mean_only_run_is_scored_without_its_loading_columns(tmp_path):
    """--mean-only hands the scorer a model of the mean alone (the mean still negated), and the PC power reads 0."""
    site = make_site(tmp_path, env_files=True)
    scores = run_main(site, "--mean-only")
    work = site["run"] / "suite_work" / f"{DATASET}__toy_arm_s1"
    with np.load(work / "checkpoint_0010.npz") as copy:
        assert copy["theta"].shape == (4, 1)
        np.testing.assert_array_equal(copy["theta"][:, 0], -np.ones(4))
    assert scores["checkpoints"]["10"]["pc_over_mean_power"] == 0
