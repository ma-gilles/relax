"""Actual K=2, q=1 CLI smoke with RECOVAR SPA and native cryo-ET simulations.

Run inside a GPU Slurm allocation, from this checkout::

    python -m relax.ppca_initial_class3d.tests.validate_k2_cli --output /unique/run/root

This checks execution, outputs and restart, not scientific recovery. The two
planted volumes are homogeneous within each class: nonzero latent recovery is
not tested. Simulator truth stays under evaluation/ and is never a CLI input.
All ET continuation branches start from one copied checkpoint. Repeated
uninterrupted continuations measure same-path GPU variation; they do not widen
the comparison band or turn an unsuccessful restart comparison into a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODULE = "relax.ppca_initial_class3d.tests.validate_k2_cli"
BOX, COUNT, VOXEL, SEED = 32, 16, 4.0, 1729
DIAMETER = 0.75 * BOX * VOXEL
ITERATIONS, FORK_ITERATION, STOP_ITERATION = 6, 2, 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def prepare_fixtures(output):
    """Use the pinned RECOVAR simulators, in a separate CPU process."""
    import pandas as pd
    import recovar
    from recovar.data_io.starfile import read_star, write_star_blocks
    from recovar.simulation import relion_tomo, simulator

    from scripts.prepare_vdam_ppca_fixture import prepare

    output.mkdir(parents=True, exist_ok=False)
    (output / "SAFE_TO_DELETE").touch()
    sources = output / "evaluation" / "source_volumes"
    sources.mkdir(parents=True)
    axis = (np.arange(BOX) - BOX / 2) * VOXEL
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")

    def blob(x, y, z, width):
        return np.exp(-((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) / width)

    volumes = (
        blob(-14, 0, 0, 110) + 0.8 * blob(10, 15, 0, 65) + 0.5 * blob(4, -10, 18, 50),
        blob(-10, -10, 0, 100) + 0.8 * blob(14, 0, 10, 60)
        + 0.65 * blob(0, 18, -12, 55) + 0.45 * blob(6, -14, -20, 45),
    )
    names = [f"vol{k:04d}.mrc" for k in range(2)]
    for name, volume in zip(names, volumes, strict=True):
        with mrcfile.new(sources / name) as mrc:
            mrc.set_data(np.asarray(volume, np.float32))
            mrc.voxel_size = VOXEL
    spa = output / "spa"
    prepare(spa, sources, [COUNT // 2, COUNT // 2], seed=SEED, box=BOX,
            noise_level=0.0005, maps=names, atomic_solvent_correction=False)

    et = output / "et"
    training, evaluation = et / "training", et / "evaluation"
    evaluation.mkdir(parents=True)
    generated = relion_tomo.generate_relion5_tomo_dataset(
        str(training), str(sources / "vol"), VOXEL, n_particles=COUNT,
        grid_size=BOX, n_tomograms=2, max_tilt=30.0, tilt_step=10.0,
        tomogram_size=(512, 512, 128), hidden_tilt_fraction=0.2,
        snr=1.0, contrast_std=0.0, origin_std_angstrom=0.0,
        volume_distribution=np.asarray([0.5, 0.5]), seed=SEED,
        atomic_solvent_correction=False, premultiplied_ctf=False,
    )
    particles, optics = read_star(generated["particles"])
    require(len(particles) == COUNT and optics is not None, "Invalid generated native particle STAR")
    angles = ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]
    origins = [f"_rlnOrigin{axis}Angst" for axis in "XYZ"]
    truth = generated["simulation_info"]
    require(np.array_equal(np.unique(truth["particle_volume"]), [0, 1]), "ET seed did not sample both classes")
    np.savez(evaluation / "truth.npz", labels=truth["particle_volume"],
             particle_names=np.asarray(truth["particle_names"]),
             eulers_deg=particles[angles].to_numpy(dtype=np.float64),
             origins_angstrom=truth["particle_origins_angstrom"],
             flat_rows_particle=truth["flat_rows_particle"],
             flat_rows_bg_mean=truth["flat_rows_bg_mean"], flat_rows_bg_std=truth["flat_rows_bg_std"])
    # Move every simulator file carrying true particle poses out of training.
    for source, target in ((Path(generated["particles"]), evaluation / "particles_truth.star"),
                           (Path(generated["particles_2d"]), evaluation / "particles_2d_truth.star"),
                           (training / "simulation_info.pkl", evaluation / "simulation_info.pkl")):
        shutil.move(source, target)
    particles.loc[:, angles + origins] = 0.0
    write_star_blocks(str(training / "particles.star"), {
        "data_general": pd.DataFrame({"_rlnTomoSubTomosAre2DStacks": [1]}),
        "data_optics": optics, "data_particles": particles,
    })
    neutral, _ = read_star(str(training / "particles.star"))
    require(np.all(neutral[angles + origins].to_numpy(dtype=float) == 0), "Nonneutral ET training poses")
    dist = importlib.metadata.distribution("recovar")
    direct_url = dist.read_text("direct_url.json")
    record = {
        "recovar_version": dist.version, "recovar_package": str(Path(recovar.__file__).resolve()),
        "direct_url": json.loads(direct_url) if direct_url else None,
        "simulator_sources": {str(Path(m.__file__).resolve()): digest(m.__file__)
                              for m in (simulator, relion_tomo)},
        "spa_api": "scripts.prepare_vdam_ppca_fixture.prepare -> recovar.simulation.simulator.simulate_data",
        "et_api": "recovar.simulation.relion_tomo.generate_relion5_tomo_dataset",
        "box": BOX, "particles_per_case": COUNT, "voxel_size": VOXEL, "seed": SEED,
        "truth_boundary": "Only spa/training/manifest.json and et/training/optimisation_set.star enter training",
        "limits": ["Two homogeneous classes; no planted within-class latent variation",
                   "Six coarse iterations establish wiring, not class/pose/map recovery",
                   "ET simulator normalizes each tilt background; original normalization is saved with truth"],
        "source_volume_sha256": {name: digest(sources / name) for name in names},
    }
    write_json(output / "fixtures.json", record)


def run_cli(arguments, log):
    command = [sys.executable, "-m", "relax.command_line", "ppca_initial_model", *arguments]
    print("Running: " + " ".join(command), flush=True)
    with open(log, "w") as stream:
        subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
    return command


def compare(actual, desired, context):
    # The repository's standard float32 comparison band, without a local tolerance.
    sys.path.insert(0, str(ROOT / "tests"))
    from helpers.float_compare import assert_trees_match

    assert_trees_match(actual, desired, err_msg=context)


def fork_checkpoint_run(source, destination, iteration):
    """Copy one existing trajectory prefix, never initialize a second model."""
    source, destination = Path(source), Path(destination)
    boundary = source / f"checkpoint_{iteration:04d}.npz"
    require(boundary.is_file(), f"Missing shared continuation checkpoint: {boundary}")
    with np.load(boundary, allow_pickle=False) as checkpoint:
        require(json.loads(str(checkpoint["metadata"]))["iteration"] == iteration,
                "Checkpoint filename and stored iteration disagree")
    records = [json.loads(line) for line in (source / "iterations.jsonl").read_text().splitlines()]
    prefix = [record for record in records if record["iteration"] <= iteration]
    require([record["iteration"] for record in prefix] == list(range(1, iteration + 1)),
            "Shared checkpoint has an incomplete or duplicated iteration history")
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.glob("checkpoint_*.npz"):
        if int(path.stem.split("_")[-1]) <= iteration:
            shutil.copy2(path, destination / path.name)
    (destination / "iterations.jsonl").write_text("".join(json.dumps(record) + "\n" for record in prefix))
    copied = destination / boundary.name
    # This checks copied-file provenance, not bitwise numerical parity of two runs.
    require(digest(copied) == digest(boundary), "Continuation checkpoint copy changed")
    return copied


def comparison_details(actual, desired):
    """Report the existing comparison rule; diagnostics never change its band."""
    try:
        compare(actual, desired, "same-checkpoint continuation")
    except AssertionError as error:
        result = {"passed": False, "reason": str(error)}
    else:
        result = {"passed": True}
    if isinstance(actual, dict) or isinstance(desired, dict):
        return result
    a, b = np.asarray(actual), np.asarray(desired)
    if a.shape != b.shape or not a.size:
        return result
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        # Higher precision here only measures differences in saved float32 output.
        dtype = np.complex128 if a.dtype.kind == "c" or b.dtype.kind == "c" else np.float64
        a, b = a.astype(dtype), b.astype(dtype)
        finite = np.isfinite(a) & np.isfinite(b)
        if np.any(finite):
            result.update(max_abs=float(np.max(np.abs(a[finite] - b[finite]))),
                          scale=float(max(np.max(np.abs(a[finite])), np.max(np.abs(b[finite])))))
    else:
        result["different_entries"] = int(np.count_nonzero(a != b))
    return result


def compare_run_artifacts(actual, desired, iteration):
    """Retain every discrepancy, including metadata and assignment differences."""
    results = {}
    for name in (f"checkpoint_{iteration:04d}.npz", "assignments.npz"):
        fields = {}
        with np.load(Path(actual) / name, allow_pickle=False) as a, np.load(
                Path(desired) / name, allow_pickle=False) as b:
            require(set(a.files) == set(b.files), f"Different output fields in {name}")
            for key in a.files:
                left, right = a[key], b[key]
                if key == "metadata":
                    left, right = json.loads(str(left)), json.loads(str(right))
                fields[key] = comparison_details(left, right)
        results[name] = fields
    return {"actual": str(actual), "desired": str(desired), "fields": results,
            "passed": all(field["passed"] for fields in results.values() for field in fields.values())}


def check_run(output, dimensions, expected_identity, *, iterations=ITERATIONS):
    run = json.loads((output / "run.json").read_text())
    require(run["command"] == "ppca_initial_model", "Wrong CLI owner")
    require(run["config"]["n_classes"] == 2 and run["config"]["q"] == 1, "Wrong K/q")
    require(all(run["identity"].get(k) == v for k, v in expected_identity.items()), "Input identity mismatch")
    require(bool(run["identity"]["source"]["files"]), "Missing source identity")
    require("cuda_sha256" in run["identity"]["source"]["native"], "CLI did not record its GPU native library")
    with np.load(output / f"checkpoint_{iterations:04d}.npz", allow_pickle=False) as checkpoint:
        meta = json.loads(str(checkpoint["metadata"]))
        require(meta["schema"] == "relax-ppca-initial-class3d-v1" and meta["iteration"] == iterations,
                "Wrong checkpoint schema or iteration")
        require(meta["identity"] == run["identity"] and meta["config"] == run["config"],
                "Run/checkpoint identity mismatch")
        for key in ("theta", "noise", "class_prior", "first", "second", "direction_prior"):
            require(np.all(np.isfinite(checkpoint[key])), f"Nonfinite checkpoint {key}")
        require(checkpoint["theta"].shape == (2, BOX * BOX * (BOX // 2 + 1), 2), "Wrong model shape")
        require(checkpoint["theta"].dtype == np.complex64, "Production model is not complex64")
        require(np.all(checkpoint["noise"] > 0), "Nonpositive noise")
        if dimensions == 3:
            require(checkpoint["noise"].ndim == 2 and checkpoint["noise"].shape[0] == 2,
                    "Native ET simulation must exercise TWO noise groups")
        require(np.all(checkpoint["class_prior"] > 0), "Nonpositive class prior")
        compare(checkpoint["class_prior"].sum(), np.float32(1), "Class prior sum")
        with np.load(output / "models.npz", allow_pickle=False) as model:
            for key in ("theta", "noise", "class_prior"):
                compare(model[key], checkpoint[key], f"Exported {key}")
    with np.load(output / "assignments.npz", allow_pickle=False) as a:
        require(np.array_equal(a["particle_ids"], np.arange(COUNT)), "Physical-particle IDs are not complete")
        require(a["class_probabilities"].shape == (COUNT, 2), "Wrong responsibility shape")
        require(a["class_conditional_z"].shape == (COUNT, 2, 1), "Wrong conditional latent shape")
        require(a["z"].shape == (COUNT, 1), "Wrong selected latent shape")
        require(a["translations_px"].shape == (COUNT, dimensions), "Wrong translation dimensionality")
        require(a["rotations"].shape == (COUNT, 3, 3), "Wrong rotation shape")
        for name in a.files:
            require(np.all(np.isfinite(a[name])), f"Nonfinite assignment {name}")
        require(np.all((a["class_probabilities"] >= 0) & (a["class_probabilities"] <= 1)),
                "Responsibilities outside [0,1]")
        compare(a["class_probabilities"].sum(axis=1), np.ones(COUNT, np.float32), "Responsibility sums")
        require(np.array_equal(a["class_labels"], a["class_probabilities"].argmax(axis=1)), "Invalid marginal MAP labels")
        compare(a["z"], a["class_conditional_z"][np.arange(COUNT), a["class_labels"]], "Selected latent")
        compare(a["translations_angstrom"], a["translations_px"] * np.float32(VOXEL), "Translation units")
    for k in range(2):
        for channel in ("mean", "loading000"):
            for name in (f"class{k:03d}_model_{channel}.mrc", f"class{k:03d}_{channel}_gridding_corrected.mrc"):
                with mrcfile.open(output / name) as mrc:
                    require(mrc.data.shape == (BOX, BOX, BOX) and np.all(np.isfinite(mrc.data)), f"Invalid map {name}")
                    compare(np.float32(mrc.voxel_size.x), np.float32(VOXEL), f"Voxel size {name}")
    records = [json.loads(line) for line in (output / "iterations.jsonl").read_text().splitlines()]
    require([r["iteration"] for r in records] == list(range(1, iterations + 1)),
            "Incomplete or duplicated iteration history")
    require(all(np.isfinite(r["log_likelihood"]) for r in records), "Nonfinite iteration likelihood")
    classes = json.loads((output / "classes.json").read_text())
    require(classes["particle_count"] == COUNT and sum(classes["hard_counts"]) == COUNT, "Class count mismatch")
    return {"particles": COUNT, "translation_dimensions": dimensions, "hard_counts": classes["hard_counts"],
            "class_prior": classes["class_prior"], "output": str(output),
            "iterations": iterations, "noise_groups": 2 if dimensions == 3 else 1}


def validate(output):
    require(bool(os.environ.get("SLURM_JOB_ID")), "This actual CLI smoke requires an existing Slurm allocation")
    require(not output.exists(), "Use a new output directory")
    # The generation subprocess releases all simulator resources before GPU training.
    subprocess.run([sys.executable, "-m", MODULE, "--output", str(output), "--prepare-only"],
                   cwd=ROOT, env=dict(os.environ, JAX_PLATFORMS="cpu"), check=True)
    probe = subprocess.run([sys.executable, "-c",
                            "import json,jax; d=jax.devices(); assert any(x.platform=='gpu' for x in d); "
                            "print(json.dumps([str(x) for x in d]))"], cwd=ROOT, check=True, text=True, capture_output=True)
    common = ["--K", "2", "--q", "1", "--iterations", str(ITERATIONS), "--seed", "11", "--no-auto-sampling",
              "--stages", "[[1,4,0]]", "--shift-range", "1", "--shift-step", "1", "--image-batch-size", "8",
              "--rotation-block-size", "16", "--ppca-gemm-precision", "fp32", "--ppca-pass2-mass-floor", "0"]
    commands, results = [], {}
    manifest = output / "spa/training/manifest.json"
    spa_run = output / "spa/run"
    commands.append(run_cli([str(manifest), "-o", str(spa_run), *common], output / "spa/cli.log"))
    results["spa"] = check_run(spa_run, 2, {"manifest_sha256": digest(manifest)})
    ios = output / "et/training/optimisation_set.star"
    et_input = ["--ios", str(ios), "--particle-diameter", str(DIAMETER)]
    prefix, resumed = output / "et/prefix", output / "et/resumed"
    direct_runs = [output / f"et/{name}" for name in ("run", "direct_control1", "direct_control2")]
    identity = {"ios_sha256": digest(ios), "particles_sha256": digest(ios.parent / "particles.star"),
                "tomograms_sha256": digest(ios.parent / "tomograms.star")}
    commands.append(run_cli([*et_input, "-o", str(prefix), *common, "--stop-after", str(FORK_ITERATION)],
                            output / "et/prefix.log"))
    shared = prefix / f"checkpoint_{FORK_ITERATION:04d}.npz"
    require(shared.is_file(), "Shared prefix did not save its boundary checkpoint")
    require(not (prefix / "assignments.npz").exists(), "Prefix prematurely exported assignments")
    branches = {path: fork_checkpoint_run(prefix, path, FORK_ITERATION) for path in [*direct_runs, resumed]}
    for index, path in enumerate(direct_runs):
        commands.append(run_cli([*et_input, "-o", str(path), *common, "--resume", str(branches[path])],
                                output / f"et/direct{index}.log"))
        results[f"et_direct{index}"] = check_run(path, 3, identity)
    commands.append(run_cli([*et_input, "-o", str(resumed), *common, "--resume", str(branches[resumed]),
                             "--stop-after", str(STOP_ITERATION)], output / "et/stop.log"))
    stopped = resumed / f"checkpoint_{STOP_ITERATION:04d}.npz"
    require(stopped.is_file(), "Stopped continuation did not save its checkpoint")
    require(not (resumed / "assignments.npz").exists(), "Stopped run prematurely exported assignments")
    commands.append(run_cli([*et_input, "-o", str(resumed), *common,
                             "--resume", str(stopped)], output / "et/resume.log"))
    results["et_resumed"] = check_run(resumed, 3, identity)
    for copied in branches.values():
        require(digest(copied) == digest(shared), "A continuation changed its shared input checkpoint")
    restart = [compare_run_artifacts(resumed, path, ITERATIONS) for path in direct_runs]
    controls = [compare_run_artifacts(right, left, ITERATIONS)
                for i, left in enumerate(direct_runs) for right in direct_runs[i + 1:]]
    accepted = all(comparison["passed"] for comparison in restart + controls)
    report = {"status": "passed" if accepted else "failed",
              "execution_and_output_checks": "passed",
              "qualification": "CLI/output/restart wiring only; not scientific recovery",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "devices": json.loads(probe.stdout),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "script_sha256": digest(__file__), "commands": commands, "cases": results,
              "restart": {"shared_checkpoint": str(shared), "shared_checkpoint_sha256": digest(shared),
                          "fork_iteration": FORK_ITERATION, "stop_iteration": STOP_ITERATION,
                          "final_iteration": ITERATIONS, "comparisons": restart,
                          "same_path_controls": controls,
                          "comparison_policy": "Unchanged repository float32 band; controls are not a wider acceptance band"},
              "fixtures": json.loads((output / "fixtures.json").read_text())}
    write_json(output / "validation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    require(accepted, "Same-checkpoint numerical comparison failed; see validation.json (no tolerances relaxed)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Only generate fixtures (used by the CPU child)")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare_only:
        prepare_fixtures(output)
    else:
        validate(output)


if __name__ == "__main__":
    main()
