#!/usr/bin/env python
"""Prepare and register an ab-initio start map for the pose-prior benchmark.

``prepare-particles`` writes the particles without their pose columns (rot, tilt, psi, origins) and Fourier-cropped to a
smaller box (the same physical box at a coarser pixel size), the input of ``relax initial_model``: ab initio needs only
the low frequencies, and a small box is far cheaper. ``upsample`` Fourier-pads the resulting map back to the grid of a
reference map. ``register`` puts that map in the ground-truth map's frame: the map has an arbitrary orientation and
handedness, so it is rigidly registered to the ground truth in both hands
(``relax.diagnostics.gt_registration.fit_rigid_both_hands``) and the hand with the larger full-spectrum FSC area is kept.
A refinement started from the registered map returns poses in the simulator's frame, up to the fit's residual
(``pose_frame_offset`` of ``evaluate_run.py`` measures it).

Run::

    python scripts/pose_prior_benchmark/initial_model.py prepare-particles --star DATASET/particles.star \\
        --out-dir PARTICLES --box 64
    python scripts/pose_prior_benchmark/initial_model.py upsample --map run_it200_class001.mrc \\
        --like DATASET/reference_gt_relion.mrc --out UPSAMPLED.mrc
    python scripts/pose_prior_benchmark/initial_model.py register --map UPSAMPLED.mrc \\
        --reference DATASET/reference_gt_relion.mrc --out-dir OUT

``PARTICLES`` receives ``particles.star`` and ``particles.<box>.mrcs``. ``OUT`` receives ``initial_model_registered.mrc``
and ``registration.json`` (hand, rotation, translation, and each hand's FSC area and resolution against the ground truth).
The maps ``upsample`` and ``register`` write carry relax's RELION-convention header label, which ``relax refine
--init_volume`` requires of a start map.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import starfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fsc_metrics import shell_fsc  # noqa: E402
from scripts.pose_prior_benchmark.evaluate_run import GT_FSC_THRESHOLDS, fsc_report, jsonable  # noqa: E402

POSE_COLUMNS = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginXAngst", "rlnOriginYAngst")
REGISTERED_NAME = "initial_model_registered.mrc"
HANDS = ("proper", "mirror")
CROP_CHUNK = 1000  # particles cropped per FFT batch
VOXEL_TOLERANCE_A = 1e-4


def crop_images(images: np.ndarray, box: int) -> np.ndarray:
    """``images`` (n, N, N) cropped in Fourier space to (n, box, box): the low-passed images on a coarser grid.
    Pixel values are kept (a constant image keeps its value). White-noise variance scales by about (box / N)^2: the
    Nyquist row and column of the cropped band have no conjugate partner and keep half their energy."""
    n_in = images.shape[-1]
    keep = np.r_[0 : box // 2, n_in - box // 2 : n_in]
    spectrum = np.fft.fft2(images)[:, keep][:, :, keep]
    return (np.fft.ifft2(spectrum).real * (box / n_in) ** 2).astype(np.float32)


def prepare_particles(star_path: Path, out_dir: Path, box: int) -> int:
    """Write the particles of ``star_path`` to ``out_dir`` without their pose columns and Fourier-cropped to ``box``
    pixels per side, as ``particles.star`` and ``particles.<box>.mrcs``; returns the number of particles.

    The stack is the single one ``rlnImageName`` names, beside the STAR. The optics' pixel size grows by the crop
    factor, so the physical box is unchanged; ``box`` equal to the stack's box only drops the poses.
    """
    blocks = starfile.read(star_path, always_dict=True)
    optics, particles = blocks["optics"], blocks["particles"]
    names = particles["rlnImageName"].str.split("@", n=1, expand=True)
    stacks = names[1].unique()
    if len(stacks) != 1:
        raise ValueError(f"the particles name {len(stacks)} stacks, expected one")
    slices = names[0].astype(int).to_numpy() - 1
    out_stack = f"particles.{box}.mrcs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with mrcfile.mmap(star_path.parent / stacks[0], permissive=True, mode="r") as src:
        box_in = src.data.shape[-1]
        if box % 2 or not 0 < box <= box_in:
            raise ValueError(f"the box must be even and at most the stack's {box_in} pixels, got {box}")
        if not (optics["rlnImageSize"] == box_in).all():
            raise ValueError(f"the optics' image size differs from the stack's {box_in} pixels")
        shape = (len(particles), box, box)
        with mrcfile.new_mmap(out_dir / out_stack, shape=shape, mrc_mode=2, overwrite=True) as dst:
            for start in range(0, len(particles), CROP_CHUNK):
                chunk = np.asarray(src.data[slices[start : start + CROP_CHUNK]], dtype=np.float32)
                dst.data[start : start + len(chunk)] = crop_images(chunk, box)
            dst.voxel_size = float(optics["rlnImagePixelSize"].iloc[0]) * box_in / box
    optics = optics.assign(rlnImagePixelSize=optics["rlnImagePixelSize"] * box_in / box, rlnImageSize=box)
    particles = particles.drop(columns=[c for c in POSE_COLUMNS if c in particles.columns])
    particles["rlnImageName"] = [f"{i + 1}@{out_stack}" for i in range(len(particles))]
    starfile.write({"optics": optics, "particles": particles}, out_dir / "particles.star", overwrite=True)
    return len(particles)


def resample_volume(volume: np.ndarray, box: int) -> np.ndarray:
    """``volume`` (N, N, N) zero-padded (or cropped) in Fourier space to ``box`` voxels per side at the same physical
    size; the density values (the mean) are kept."""
    n = volume.shape[0]
    spectrum = np.fft.fftshift(np.fft.fftn(volume))
    out = np.zeros((box,) * 3, dtype=complex)
    m = min(n, box)
    source, target = n // 2 - m // 2, box // 2 - m // 2
    out[(slice(target, target + m),) * 3] = spectrum[(slice(source, source + m),) * 3]
    return (np.fft.ifftn(np.fft.ifftshift(out)).real * (box / n) ** 3).astype(np.float32)


def read_map(path: Path) -> tuple[np.ndarray, float]:
    """A cubic MRC map as float64 and its voxel size (A)."""
    with mrcfile.open(path, permissive=True) as handle:
        return np.asarray(handle.data, dtype=np.float64).copy(), float(handle.voxel_size.x)


def write_map(path: Path, volume: np.ndarray, voxel: float) -> None:
    """Write ``volume`` (RELION sign and axis convention) as a float32 MRC with voxel size ``voxel`` (A), labeled as
    such: ``relax refine`` refuses a start map without the label."""
    from relax.io.map_io import RELAX_MAP_LABEL

    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(volume.astype(np.float32))
        handle.voxel_size = voxel
        handle.header.label[0] = RELAX_MAP_LABEL.encode("ascii")
        handle.header.nlabl = 1


def upsample(moving: np.ndarray, voxel: float, like: np.ndarray, like_voxel: float) -> np.ndarray:
    """``moving`` (voxel size ``voxel`` A) resampled to the grid of ``like``; raises ``ValueError`` when the two maps
    do not cover the same physical box."""
    box = like.shape[0]
    if abs(voxel * moving.shape[0] - like_voxel * box) > VOXEL_TOLERANCE_A * box:
        raise ValueError(
            f"the map spans {voxel * moving.shape[0]} A, the reference {like_voxel * box} A: a resampling would rescale it"
        )
    return resample_volume(moving, box)


def register(moving: np.ndarray, reference: np.ndarray, voxel: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Rigidly register ``moving`` to ``reference`` in both hands; returns the registered map of the better hand
    (larger FSC area to the reference) and the record of both fits."""
    from relax.diagnostics.gt_metrics import relion_alignment_rotations
    from relax.diagnostics.gt_registration import apply_rigid_volume_transform, fit_rigid_both_hands

    fits = fit_rigid_both_hands(moving, reference, relion_alignment_rotations(2))
    missing = [hand for hand in HANDS if hand not in fits]
    if missing:
        raise ValueError(f"the registration failed in hand(s) {missing}")
    box = reference.shape[0]
    maps, record = {}, {}
    for hand, fit in fits.items():
        maps[hand] = apply_rigid_volume_transform(
            moving, fit.rotation_matrix, fit.translation_voxels, mirror_x=fit.mirror_x
        )
        curve = shell_fsc(maps[hand], reference)
        record[hand] = {
            "rotation_matrix": fit.rotation_matrix.tolist(),
            "translation_voxels": fit.translation_voxels.tolist(),
            "mirror_x": fit.mirror_x,
            "fit_lowpass_correlation": fit.lowpass_score,
            "fine_correlation": fit.fine_score,
            "fine_optimizer_success": fit.fine_optimizer_success,
            "gt_fsc": fsc_report(curve, GT_FSC_THRESHOLDS, voxel, box),
        }
    best = max(HANDS, key=lambda hand: record[hand]["gt_fsc"]["auc"])
    other = HANDS[1 - HANDS.index(best)]
    summary = {
        "hand": best,
        "fsc_auc_margin_over_other_hand": record[best]["gt_fsc"]["auc"] - record[other]["gt_fsc"]["auc"],
        "voxel_size_A": voxel,
        "box": box,
        "hands": record,
    }
    return maps[best], summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments of the subcommands."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-particles", help="the particles without poses, Fourier-cropped to a box")
    prepare.add_argument("--star", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--box", type=int, required=True, help="pixels per side of the cropped particles (even)")
    resample = commands.add_parser("upsample", help="Fourier-pad a map to the grid of a reference map")
    resample.add_argument("--map", type=Path, required=True)
    resample.add_argument("--like", type=Path, required=True, help="a map with the target box and voxel size")
    resample.add_argument("--out", type=Path, required=True)
    registration = commands.add_parser("register", help="register an InitialModel map to the ground truth")
    registration.add_argument("--map", type=Path, required=True, help="the InitialModel map (RELION convention)")
    registration.add_argument("--reference", type=Path, required=True, help="the ground-truth map (RELION convention)")
    registration.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the subcommand."""
    args = parse_args(argv)
    if args.command == "prepare-particles":
        count = prepare_particles(args.star, args.out_dir, args.box)
        print(f"wrote {count} particles without poses at box {args.box} to {args.out_dir}")
        return
    if args.command == "upsample":
        moving, voxel = read_map(args.map)
        like, like_voxel = read_map(args.like)
        write_map(args.out, upsample(moving, voxel, like, like_voxel), like_voxel)
        print(f"wrote {args.out}: box {like.shape[0]} at {like_voxel} A (from box {moving.shape[0]} at {voxel} A)")
        return
    moving, voxel = read_map(args.map)
    reference, reference_voxel = read_map(args.reference)
    if moving.shape != reference.shape or abs(voxel - reference_voxel) > VOXEL_TOLERANCE_A:
        raise ValueError(
            f"the map is {moving.shape} at {voxel} A, the reference {reference.shape} at {reference_voxel} A"
        )
    registered, summary = register(moving, reference, voxel)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_map(args.out_dir / REGISTERED_NAME, registered, voxel)
    (args.out_dir / "registration.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    chosen = summary["hands"][summary["hand"]]["gt_fsc"]
    print(
        f"hand {summary['hand']}, FSC area {chosen['auc']:.3f} "
        f"(margin {summary['fsc_auc_margin_over_other_hand']:.3f} over the other hand), "
        f"0.5 sustained crossing (A) {chosen['at_0.5']['sustained_A']}"
    )


if __name__ == "__main__":
    main()
