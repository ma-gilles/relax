"""Learn one ab-initio q=2 PPCA model from particles and known CTFs."""

import argparse
import dataclasses
import json
import os

os.environ.setdefault("RECOVAR_EM_XLA_DEFAULTS", "1")
import subprocess
from pathlib import Path

import numpy as np

from relax.ppca_initial_model.checkpoint import file_hash
from relax.ppca_initial_model.config import Config


def add_args(parser):
    parser.add_argument("manifest", help="Training-only fixture manifest")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--stages", help="JSON list of [first iteration, Fourier radius, HEALPix order]")
    parser.add_argument("--oversampling", type=int, default=1)
    parser.add_argument("--shift-range", type=float, default=6)
    parser.add_argument("--shift-step", type=float, default=2)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--rotation-block-size", type=int, default=128)
    parser.add_argument("--fine-image-tile-size", type=int, default=1,
                        help="Batch identical full-support fine rows; 1 keeps the reference path")
    parser.add_argument("--stream-full-fine-rows", action="store_true",
                        help="Stream rows of the shared fine grid with exact per-image support masks on the device")
    parser.add_argument("--fine-devices", type=int, default=1,
                        help="Local GPUs for the streamed pass; image tiles are split across them")
    parser.add_argument("--stochastic-batch-size", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int, help="Checkpoint stop without changing the scientific schedule")
    parser.add_argument("--stop-file", help="Stop between iterations after an atomic checkpoint")


def load_training(path):
    """Load the training fixture with its unit-contrast image sign (algorithm §15)."""
    from recovar.data_io.cryoem_dataset import load_dataset

    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest["schema"] != "recovar-ppca-training-v1":
        raise ValueError("Unsupported training manifest")
    allowed = {
        "schema",
        "n_images",
        "box",
        "voxel_size",
        "particle_diameter_ang",
        "shift_range_px",
        "particles",
        "ctf",
        "neutral_poses",
        "particle_ids",
        "contrast",
        "image_multiplier",
        "files",
    }
    if set(manifest) - allowed:
        raise ValueError("Unexpected training metadata; truth must stay outside the trainer")
    inputs = [manifest[key] for key in ("particles", "ctf", "neutral_poses", "particle_ids")]
    if len(set(inputs)) != 4 or set(inputs) != set(manifest["files"]):
        raise ValueError("Every training input must have exactly one recorded identity")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or file_hash(path.parent / name) != digest:
            raise ValueError("Training input identity mismatch")
    if manifest["contrast"] != 1 or manifest["image_multiplier"] != 1:
        raise ValueError("First version requires unit contrast and unscaled particles")
    data = load_dataset(
        str(path.parent / manifest["particles"]),
        poses_file=str(path.parent / manifest["neutral_poses"]),
        ctf_file=str(path.parent / manifest["ctf"]),
        uninvert_data=False,
        dtype=np.complex64,
    )
    if not np.allclose(np.asarray(data.rotation_matrices), np.eye(3), atol=0, rtol=0) or np.any(
        np.asarray(data.translations)
    ):
        raise ValueError("Training poses must be neutral; supplied poses are not accepted")
    ids = np.load(path.parent / manifest["particle_ids"], allow_pickle=False)
    if not np.array_equal(ids, np.arange(data.n_images)):
        raise ValueError("Expected contiguous stable original particle IDs")
    if data.grid_size != manifest["box"] or data.n_images != manifest["n_images"]:
        raise ValueError("Training geometry mismatch")
    if not np.isclose(data.voxel_size, manifest["voxel_size"], rtol=1e-6, atol=0):
        raise ValueError("Training pixel size mismatch")
    return data, manifest, {"manifest_sha256": file_hash(path)}


def source_identity():
    import jax
    import recovar

    from relax.relion_bind import _relion_bind_core as binding

    native = {"relion_binding_sha256": file_hash(binding.__file__)}
    if jax.default_backend() == "gpu":
        from recovar import cuda_backproject

        if not cuda_backproject.cuda_available():
            raise RuntimeError("GPU execution requires the explicitly built custom CUDA library")
        native["cuda_sha256"] = file_hash(cuda_backproject._loaded_lib_path)
    repo = Path(__file__).resolve().parents[2]
    # Includes untracked implementation files; a clean HEAD alone is insufficient.
    files = sorted((repo / "relax/ppca_initial_model").glob("*.py"))
    files += sorted((repo / "relax/ppca_refinement").glob("*.py"))
    files += [Path(__file__), repo / "relax/relion/relion_project.py", repo / "pixi.lock"]
    recovar_stats = Path(recovar.__file__).resolve().parent / "ppca/pose_accumulators.py"
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "files": {str(p.relative_to(repo)): file_hash(p) for p in files},
        "recovar_stats": {"path": str(recovar_stats), "sha256": file_hash(recovar_stats)},
        "native": native,
    }


def main(args=None):
    if not isinstance(args, argparse.Namespace):
        parser = argparse.ArgumentParser(description=__doc__)
        add_args(parser)
        args = parser.parse_args(args)
    import os

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Particle refinement runs require Slurm; use unit tests for local numerical checks")
    from relax.ppca_initial_model.iteration_loop import run

    config = Config(
        q=args.q,
        seed=args.seed,
        iterations=args.iterations,
        oversampling=args.oversampling,
        shift_range=args.shift_range,
        shift_step=args.shift_step,
        image_batch_size=args.image_batch_size,
        rotation_block_size=args.rotation_block_size,
        fine_image_tile_size=args.fine_image_tile_size,
        stream_full_fine_rows=args.stream_full_fine_rows,
        fine_devices=args.fine_devices,
        stochastic_batch_size=args.stochastic_batch_size,
        stages=tuple(tuple(stage) for stage in json.loads(args.stages)) if args.stages else Config().stages,
    )
    data, manifest, identity = load_training(args.manifest)
    identity["source"] = source_identity()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and list(output.glob("checkpoint_*.npz")):
        raise ValueError("Output already contains checkpoints; use --resume or a new directory")
    if args.resume:
        from relax.ppca_initial_model.checkpoint import load

        # Reject a mismatched checkpoint before changing existing run metadata.
        load(args.resume, config, identity)
    (output / "run.json").write_text(
        json.dumps({"config": dataclasses.asdict(config), "identity": identity}, indent=2) + "\n"
    )
    run(
        data,
        config,
        output,
        identity,
        manifest["particle_diameter_ang"],
        resume=args.resume,
        stop_after=args.stop_after,
        stop_file=args.stop_file,
    )


if __name__ == "__main__":
    main()
