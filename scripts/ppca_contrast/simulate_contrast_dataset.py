#!/usr/bin/env python3
"""Simulate a PPCA test-suite dataset again with a per-particle contrast spread.

A contrast-spread twin of a suite dataset: the same states, labels, rotations, translations, CTF, noise spectrum and
noise realisation (same simulator seed), with each particle's noiseless image multiplied by its own contrast
``c_i ~ N(1, sd^2)`` (redrawn where ``c_i < --min-contrast``; or log-normal with ``--distribution lognormal``) before the noise is added, as the simulator's
``per_image_contrast`` does. With ``--contrast-sd 0`` it writes the paired control: the same images with unit contrast
simulated by the same code. The evaluation files of the base dataset are hard-linked, and ``contrast.npy`` holds the
true contrasts next to them.

Run (from the repository root, in the pinned environment; the base is a dataset folder with training/ and evaluation/):
    python scripts/ppca_contrast/simulate_contrast_dataset.py \
        --base <suite>/datasets/C__realistic__noise1__b128crop64 --output <root>/datasets/C_c30 --contrast-sd 0.3
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

CONSISTENCY_IMAGES = 256


def draw_contrast(n_images: int, sd: float, minimum: float, seed: int, distribution: str = "normal") -> np.ndarray:
    """``n_images`` contrasts of mean 1 and standard deviation ``sd``; float32.

    ``normal``: N(1, sd^2), each redrawn until it is at least ``minimum``. ``lognormal``: exp(s z - s^2 / 2) with
    ``s^2 = ln(1 + sd^2)`` (mean 1, skewed, always positive).
    """
    rng = np.random.default_rng(seed)
    if distribution == "lognormal":
        scale = np.sqrt(np.log1p(sd**2))
        return np.exp(scale * rng.standard_normal(n_images) - scale**2 / 2).astype(np.float32)
    if distribution != "normal":
        raise ValueError(f"unknown contrast distribution {distribution!r}")
    contrast = 1.0 + sd * rng.standard_normal(n_images)
    while True:
        low = contrast < minimum
        if not low.any():
            return contrast.astype(np.float32)
        contrast[low] = 1.0 + sd * rng.standard_normal(int(low.sum()))


def file_sha256(path: Path) -> str:
    """The sha256 hex digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def link_evaluation(base: Path, output: Path, dataset_id: str, contrast: np.ndarray, spread: dict) -> None:
    """Hard-link the base evaluation files into ``output/evaluation``, write ``contrast.npy`` and the manifest."""
    target = output / "evaluation"
    target.mkdir(parents=True)
    for source in sorted((base / "evaluation").iterdir()):
        if source.name != "manifest.json":
            os.link(source, target / source.name)
    np.save(target / "contrast.npy", contrast)
    manifest = json.loads((base / "evaluation" / "manifest.json").read_text())
    manifest.update(dataset_id=dataset_id, contrast_spread=spread, contrast_base_dataset=base.name)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def simulate(base: Path, output: Path, contrast: np.ndarray, spread: dict, batch: int, noise_scale: float) -> dict:
    """Write ``output/training`` for the first ``len(contrast)`` images of the base dataset's truth and return the
    consistency report."""
    import mrcfile
    from recovar import core
    from recovar.data_io.cryoem_dataset import CryoEMDataset, ImageMetadata
    from recovar.simulation import simulator
    from recovar.utils import helpers

    truth = np.load(base / "evaluation" / "truth.npz")
    evaluation = json.loads((base / "evaluation" / "manifest.json").read_text())
    box, apix = int(evaluation["box"]), float(evaluation["voxel_size"])
    n_images = len(contrast)
    labels, ctf = truth["labels"][:n_images], truth["ctf"][:n_images]
    variance = np.asarray(evaluation["variant_receipt"]["injected_spectrum"], np.float32) * np.float32(noise_scale)
    seed = int(evaluation["seed"])
    data = CryoEMDataset(
        None, apix, ImageMetadata(truth["rotations"][:n_images], truth["translations"][:n_images], ctf),
        ctf_evaluator=core.CTFEvaluator(), grid_size=box,
    )
    volumes = truth["volumes_fourier"]
    training = output / "training"
    training.mkdir(parents=True)
    ones = np.ones(n_images, np.float32)
    with mrcfile.new_mmap(training / "particles.mrcs", shape=(n_images, box, box), mrc_mode=2, overwrite=False) as stack:
        stack.set_image_stack()
        stack.voxel_size = apix
        simulator.simulate_data(
            data, volumes, variance, batch, labels, contrast, ones, seed=seed, mrc_file=stack,
            noise_rng_batch_size=64, noise_transform_batch_size=64,
        )
        stack.flush()
    simulator.save_ctf_params(str(training), box, ctf, apix)
    helpers.pickle_dump(
        (np.broadcast_to(np.eye(3, dtype=np.float32), (n_images, 3, 3)).copy(), np.zeros((n_images, 2), np.float32)),
        str(training / "neutral_poses.pkl"),
    )
    np.save(training / "particle_ids.npy", np.arange(n_images, dtype=np.int64))
    manifest = json.loads((base / "training" / "manifest.json").read_text())
    manifest.update(n_images=n_images, contrast=1.0, contrast_spread=spread)
    manifest["files"] = {name: file_sha256(training / name) for name in sorted(manifest["files"])}
    (training / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return consistency(base, training / "particles.mrcs", contrast)


def consistency(base: Path, stack_path: Path, contrast: np.ndarray) -> dict:
    """Compare the first images with the base probe: the noise-free signal against ``contrast`` times the probe signal
    is not available, so report the power of the simulated images over the base's, split by contrast."""
    import mrcfile

    probe = np.load(base / "evaluation" / "noise_probe.npz")
    n = min(len(probe["signal"]), CONSISTENCY_IMAGES * 4)
    with mrcfile.mmap(stack_path, mode="r") as stack:
        images = np.asarray(stack.data[:n], np.float64)
    with mrcfile.mmap(base / "training" / "particles.mrcs", mode="r") as stack:
        reference = np.asarray(stack.data[:n], np.float64)
    return {
        "images_compared": n,
        "mean_power_simulated": float(np.mean(images**2)),
        "mean_power_base": float(np.mean(reference**2)),
        "mean_contrast_first_images": float(np.mean(contrast[:n])),
        "signal_power_base_probe": float(np.mean(probe["signal"][:n].astype(np.float64) ** 2)),
    }


def main() -> None:
    """Parse the options, draw the contrasts, simulate and print the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="suite dataset folder (training/ and evaluation/)")
    parser.add_argument("--output", required=True, type=Path, help="new dataset folder (must not exist)")
    parser.add_argument("--contrast-sd", required=True, type=float, help="sd of the contrast about 1; 0 = unit contrast")
    parser.add_argument("--distribution", choices=("normal", "lognormal"), default="normal")
    parser.add_argument("--min-contrast", type=float, default=0.2, help="normal: contrasts below this are redrawn")
    parser.add_argument("--contrast-seed", type=int, default=9000, help="seed of the contrast stream")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="multiplier on the base noise variance")
    parser.add_argument("--n-images", type=int, default=None, help="simulate only the first N images (smoke tests)")
    parser.add_argument("--batch", type=int, default=256, help="images per simulation batch")
    args = parser.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} exists")
    n_all = len(np.load(args.base / "evaluation" / "truth.npz")["labels"])
    n_images = n_all if args.n_images is None else min(int(args.n_images), n_all)
    contrast = draw_contrast(n_images, args.contrast_sd, args.min_contrast, args.contrast_seed, args.distribution)
    spread = {
        "distribution": args.distribution, "sd": args.contrast_sd, "minimum": args.min_contrast, "seed": args.contrast_seed,
        "realised_mean": float(contrast.mean()), "realised_sd": float(contrast.std()),
        "noise_scale": args.noise_scale, "file": "evaluation/contrast.npy",
    }
    args.output.mkdir(parents=True)
    link_evaluation(args.base, args.output, args.output.name, contrast, spread)
    report = simulate(args.base, args.output, contrast, spread, args.batch, args.noise_scale)
    print(json.dumps({"dataset": str(args.output), "spread": spread, "consistency": report}, indent=2))


if __name__ == "__main__":
    main()
