"""Learn one ab-initio PPCA model from particles and known CTFs."""

import argparse
import dataclasses
import json
import logging
import os

os.environ.setdefault("RECOVAR_EM_XLA_DEFAULTS", "1")
import subprocess
from pathlib import Path

import numpy as np

from relax.ppca_initial_model.checkpoint import file_hash
from relax.ppca_initial_model.config import Config
from relax.runtime.host_memory import available_memory_bytes

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("manifest", nargs="?", help="Training-only fixture manifest (single particles)")
    parser.add_argument("--ios", help="RELION 5 optimisation set of subtomogram particles (2D stacks) instead")
    parser.add_argument("--particle-diameter", type=float, help="Particle diameter in Angstrom (with --ios)")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--stages", help="JSON list of [first iteration, Fourier radius, HEALPix order]")
    parser.add_argument(
        "--oversampling",
        type=int,
        default=0,
        choices=(0, 1),
        help="Adaptive oversampling: 0 scores each stage's full pose grid; 1 runs the last stage with RELION's two "
        "passes, the children of each image's significant coarse samples, and the earlier stages dense "
        "(vdam_ppca_algorithm.md section 14)",
    )
    parser.add_argument(
        "--maxsig",
        type=int,
        default=100,
        help="Significant coarse samples kept per image at most with --oversampling 1 (RELION's --maxsig; 100 for "
        "gradient runs with one class). Each update records how many images it stopped short of the 0.999 mass",
    )
    parser.add_argument("--shift-range", type=float, default=6)
    parser.add_argument("--shift-step", type=float, default=2)
    parser.add_argument("--image-batch-size", type=int, default=150)
    parser.add_argument("--rotation-block-size", type=int, default=512)
    parser.add_argument("--fine-image-tile-size", type=int, default=1,
                        help="Batch identical full-support fine rows; 1 keeps the reference path")
    parser.add_argument("--stream-full-fine-rows", action="store_true",
                        help="Stream rows of the shared fine grid with exact per-image support masks on the device")
    parser.add_argument(
        "--stream-coarse-recompute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="GPU stream over the full pose grid (default); --no-stream-coarse-recompute selects the host-mask "
        "dense reference engine (q <= 2)",
    )
    parser.add_argument("--fine-devices", type=int, default=1,
                        help="Local GPUs for the streamed pass; image tiles are split across them")
    parser.add_argument("--stochastic-batch-size", type=int)
    parser.add_argument(
        "--stochastic-all-iterations",
        action="store_true",
        help="Keep the fixed stochastic batch on the final update as well",
    )
    parser.add_argument(
        "--balanced-stochastic-halves",
        action="store_true",
        help="Draw equal-sized even/odd pseudo-halves from one shuffled batch order",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--skip-final-embeddings",
        action="store_true",
        help="Skip the separate all-particle embedding E-step after the final update",
    )
    parser.add_argument("--optimizer", choices=("vdam", "momentum_sgd"), default="vdam")
    parser.add_argument(
        "--ppca-gemm-precision",
        choices=("auto", "fp32", "tf32"),
        default="auto",
        help="streamed-engine GEMMs: auto (default; tf32 on sm_80+ GPUs, else fp32), tf32 (TF32 tensor cores, "
        "float32 accumulation) or fp32 (exact float32)",
    )
    parser.add_argument(
        "--ppca-preread-images",
        choices=("auto", "on", "off"),
        default="auto",
        help="particle images in host memory: auto (default; read the stack once when it takes at most a "
        "quarter of the job's memory), on, or off (read each tile from disk)",
    )
    parser.add_argument(
        "--ppca-pass2-mass-floor",
        type=float,
        default=1e-10,
        help="pass 2 skips pose rows on which every image of a tile has less posterior mass (default 1e-10, "
        "applied when at most half the rows remain; 0: none skipped)",
    )
    parser.add_argument("--sgd-learning-rate", type=float, default=0.4)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int, help="Checkpoint stop without changing the scientific schedule")
    parser.add_argument("--stop-file", help="Stop between iterations after an atomic checkpoint")


# Largest share of the job's memory that "auto" lets one host copy of the particle stack take.
PREREAD_MEMORY_FRACTION = 0.25


def resolve_preread_images(setting, stack_bytes, memory_bytes):
    """Whether to read the particle stack into host memory, and the record of that decision."""
    if setting not in ("auto", "on", "off"):
        raise ValueError("Image preread must be auto, on or off")
    if setting == "auto":
        preread = stack_bytes <= PREREAD_MEMORY_FRACTION * memory_bytes
    else:
        preread = setting == "on"
    return preread, {
        "setting": setting,
        "preread": preread,
        "stack_gb": round(stack_bytes / 1e9, 3),
        "memory_gb": round(memory_bytes / 1e9, 3),
        "memory_fraction": PREREAD_MEMORY_FRACTION,
    }


def training_image_reading(path, preread_images):
    """:func:`resolve_preread_images` for a training manifest's particle stack on this host."""
    path = Path(path).resolve()
    particles = path.parent / json.loads(path.read_text())["particles"]
    return resolve_preread_images(preread_images, particles.stat().st_size, available_memory_bytes())


def load_training(path, preread_images="auto"):
    """Load the training fixture with its unit-contrast image sign (algorithm §15).

    ``preread_images`` (``Config.preread_images``) decides whether the particle stack is read into
    host memory once (:func:`training_image_reading`); the decision is logged.
    """
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
    particles = path.parent / manifest["particles"]
    preread, reading = training_image_reading(path, preread_images)
    logger.info(
        "PPCA particle images: %s (setting %s; stack %.2f GB, memory %.1f GB, preread limit %.0f%%)",
        "read into host memory once" if preread else "read from disk per tile",
        reading["setting"],
        reading["stack_gb"],
        reading["memory_gb"],
        100 * PREREAD_MEMORY_FRACTION,
    )
    data = load_dataset(
        str(particles),
        poses_file=str(path.parent / manifest["neutral_poses"]),
        ctf_file=str(path.parent / manifest["ctf"]),
        uninvert_data=False,
        dtype=np.complex64,
        lazy=not preread,
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


def refuse_unsupported_tilt_optics(particles_star):
    """Refuse the optics features of subtomogram particles that relax does not apply per tilt.

    relion_refine applies CTF premultiplication, odd and even aberrations and anisotropic
    magnification to every tilt image; the per-tilt STAR relax reads carries none of them, so a
    particle STAR that uses one fails instead of being trained on silently wrong images
    (:data:`relax.relion.metadata.TOMO_PPCA_OPTICS_FEATURES`; subtomogram Refine3D/Class3D
    have their own set, ``TOMO_OPTICS_FEATURES``).
    """
    from recovar.data_io.starfile import read_star

    from relax.relion.metadata import TOMO_PPCA_OPTICS_FEATURES, refuse_unsupported_optics

    _, optics = read_star(str(particles_star))
    refuse_unsupported_optics(optics, source=str(particles_star), supported=TOMO_PPCA_OPTICS_FEATURES)


def load_tilt_training(ios, output):
    """Subtomogram particles of a RELION 5 optimisation set (algorithm section 16).

    Only the tilt-series geometry (each image's ``Aproj``), the images and their CTF and dose
    enter training; the particles' input angles and offsets are not read.
    """
    from relax.ppca_initial_model.tomo import tilt_particles_from_tomo_dataset
    from relax.refinement.tomo_half import load_tomo_dataset
    from relax.relion.tomo_input import read_optimisation_set

    particles_star, tomograms_star = read_optimisation_set(ios)
    refuse_unsupported_tilt_optics(particles_star)
    tomo = load_tomo_dataset(
        particles_star,
        tomograms_star,
        Path(output) / "particles_2d.star",
        datadir=str(Path(particles_star).resolve().parent),
        lazy=False,
    )
    identity = {
        "ios_sha256": file_hash(ios),
        "particles_sha256": file_hash(particles_star),
        "tomograms_sha256": file_hash(tomograms_star),
        "tilt_series_sha256": tilt_series_hash(tomograms_star),
    }
    return tilt_particles_from_tomo_dataset(tomo), identity


def tilt_series_hash(tomograms_star):
    """One sha256 of the tilt-series STAR files a tomograms STAR names (their per-tilt geometry, CTF and dose).

    The particles and tomograms STAR files do not identify the tilt geometry: an edited tilt series leaves them
    unchanged, so the checkpoint identity also carries this hash and a run does not resume across such an edit.
    """
    import hashlib

    from recovar.data_io.starfile import read_star, star_column

    table, _ = read_star(str(tomograms_star))
    root = Path(tomograms_star).parent
    digest = hashlib.sha256()
    for name in sorted({str(f) for f in star_column(table, "rlnTomoTiltSeriesStarFile", required=True)}):
        digest.update(f"{name}\0{file_hash(root / name)}\0".encode())
    return digest.hexdigest()


def source_identity():
    import jax
    import recovar

    native = {}
    if jax.default_backend() == "gpu":
        from recovar import cuda_backproject

        if not cuda_backproject.cuda_available():
            raise RuntimeError("GPU execution requires the explicitly built custom CUDA library")
        native["cuda_sha256"] = file_hash(cuda_backproject._loaded_lib_path)
    repo = Path(__file__).resolve().parents[2]
    # Includes untracked implementation files; a clean HEAD alone is insufficient.
    files = sorted((repo / "relax/ppca_initial_model").glob("*.py"))
    files += sorted((repo / "relax/ppca_refinement").glob("*.py"))
    files += [Path(__file__), repo / "relax/relion/project.py"]
    # A plain file snapshot or an installed package has no lock file or git checkout; the file
    # hashes identify the source on their own.
    files += [path for path in (repo / "pixi.lock",) if path.is_file()]
    recovar_stats = Path(recovar.__file__).resolve().parent / "ppca/pose_accumulators.py"
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {
        "head": head,
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
    import sys

    # The run's INFO lines (tile plans, compiled tile sizes, resumes) reach the job log, as in Refine3D.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
    from relax.runtime.compilation_cache import activate_recovar_compilation_cache

    activate_recovar_compilation_cache()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Particle refinement runs require Slurm; use unit tests for local numerical checks")
    from relax.ppca_initial_model.iteration_loop import run

    config = Config(
        q=args.q,
        seed=args.seed,
        iterations=args.iterations,
        oversampling=args.oversampling,
        max_significant=args.maxsig,
        shift_range=args.shift_range,
        shift_step=args.shift_step,
        image_batch_size=args.image_batch_size,
        rotation_block_size=args.rotation_block_size,
        fine_image_tile_size=args.fine_image_tile_size,
        stream_full_fine_rows=args.stream_full_fine_rows,
        stream_coarse_recompute=args.stream_coarse_recompute,
        fine_devices=args.fine_devices,
        stochastic_batch_size=args.stochastic_batch_size,
        stochastic_all_iterations=args.stochastic_all_iterations,
        balanced_stochastic_halves=args.balanced_stochastic_halves,
        checkpoint_interval=args.checkpoint_interval,
        skip_final_embeddings=args.skip_final_embeddings,
        optimizer=args.optimizer,
        sgd_learning_rate=args.sgd_learning_rate,
        gemm_precision=args.ppca_gemm_precision,
        preread_images=args.ppca_preread_images,
        pass2_mass_floor=args.ppca_pass2_mass_floor,
        stages=tuple(tuple(stage) for stage in json.loads(args.stages)) if args.stages else Config().stages,
    )
    if (args.ios is None) == (args.manifest is None):
        raise ValueError("Give either a training manifest or --ios")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.ios:
        if args.particle_diameter is None:
            raise ValueError("--ios needs --particle-diameter")
        data, identity = load_tilt_training(args.ios, output)
        diameter = args.particle_diameter
    else:
        data, manifest, identity = load_training(args.manifest, config.preread_images)
        diameter = manifest["particle_diameter_ang"]
    identity["source"] = source_identity()
    if not args.resume and list(output.glob("checkpoint_*.npz")):
        raise ValueError("Output already contains checkpoints; use --resume or a new directory")
    if args.resume:
        from relax.ppca_initial_model.checkpoint import load

        # Reject a mismatched checkpoint before changing existing run metadata.
        load(args.resume, config, identity)
    (output / "run.json").write_text(
        json.dumps(
            {
                "config": dataclasses.asdict(config),
                "identity": identity,
                # Subtomogram stacks are always read into memory (load_tilt_training).
                "image_reading": None if args.ios else training_image_reading(args.manifest, config.preread_images)[1],
            },
            indent=2,
        )
        + "\n"
    )
    run(
        data,
        config,
        output,
        identity,
        diameter,
        resume=args.resume,
        stop_after=args.stop_after,
        stop_file=args.stop_file,
    )


if __name__ == "__main__":
    main()
