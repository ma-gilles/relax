#!/usr/bin/env python
"""Derive an optics-feature fixture from a curated synthetic fixture.

Takes the first ``--n-images`` particles of a curated fixture (particles.star, its
stack and its reference/GT maps) and writes them with one RELION optics feature
applied, so relax and RELION can be compared on data that exercises it:

``--premultiply-ctf``
    Each image is multiplied in Fourier space by its own RELION CTF
    (``CTF::getFftwImage`` through relax's RELION binding, the arithmetic
    ``relion_refine`` evaluates for the particle) and the optics group is marked
    ``rlnCtfDataAreCtfPremultiplied 1``. A premultiplied image of real data is
    ``CTF * (CTF * projection + noise)``: exactly what this produces from the
    source image.

The base fixture's ground truth, initial reference and masks stay valid because
the particles, their poses and the maps are unchanged. The output directory gets a
README.md and GENERATION.json (command, source SHA, input and output sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import starfile

COPIED_MAPS = (
    "reference_gt.mrc",
    "reference_gt_class001.mrc",
    "reference_gt_relion.mrc",
    "reference_init_class001.mrc",
    "reference_init_class001_relion.mrc",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _column(frame, name, optics, default=None):
    """A per-particle value from the particle table, else its optics group's, else ``default``."""

    if name in frame.columns:
        return np.asarray(frame[name], dtype=np.float64)
    groups = np.asarray(frame["rlnOpticsGroup"], dtype=np.int64)
    by_group = {}
    for _, row in optics.iterrows():
        group = int(row["rlnOpticsGroup"])
        if name in optics.columns:
            by_group[group] = float(row[name])
        elif default is not None:
            by_group[group] = float(default)
        else:
            raise KeyError(f"optics group {group} has no {name}")
    return np.asarray([by_group[int(g)] for g in groups], dtype=np.float64)


def relion_ctf_images(particles, optics, box: int, *, n_threads: int) -> np.ndarray:
    """RELION's ``Fctf`` of every particle on the FFTW half grid ``(N, box, box // 2 + 1)``.

    The same ``CTF::setValues`` + ``getFftwImage`` the binding evaluates for
    ``relion_refine``'s CTF operands (no padding, not absolute, with damping).
    """

    from relax.relion_bind import _relion_bind_core as relion_bind

    params = np.stack(
        [
            _column(particles, "rlnDefocusU", optics),
            _column(particles, "rlnDefocusV", optics),
            _column(particles, "rlnDefocusAngle", optics),
            _column(particles, "rlnVoltage", optics),
            _column(particles, "rlnSphericalAberration", optics),
            _column(particles, "rlnAmplitudeContrast", optics),
            _column(particles, "rlnCtfBfactor", optics, 0.0),
            _column(particles, "rlnImagePixelSize", optics),
            _column(particles, "rlnPhaseShift", optics, 0.0),
            _column(particles, "rlnCtfScalefactor", optics, 1.0),
        ],
        axis=1,
    )
    return np.asarray(
        relion_bind.get_ctf_images_batch(np.ascontiguousarray(params), box, box, False, False, True, n_threads),
        dtype=np.float64,
    )


def premultiply(images: np.ndarray, ctf_fftw_half: np.ndarray) -> np.ndarray:
    """``irfft2(rfft2(image) * Fctf)`` in float64, returned as float32.

    ``Fctf`` is real and point-symmetric, so the product does not depend on where
    the transform puts the image origin.
    """

    box = images.shape[-1]
    spectrum = np.fft.rfft2(np.asarray(images, dtype=np.float64), axes=(-2, -1))
    return np.fft.irfft2(spectrum * ctf_fftw_half, s=(box, box), axes=(-2, -1)).astype(np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path, required=True, help="curated fixture with particles.star")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-images", type=int, required=True)
    parser.add_argument("--premultiply-ctf", action="store_true")
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)

    if not args.premultiply_ctf:
        parser.error("choose an optics feature (--premultiply-ctf)")
    base = args.base_dir.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)

    tables = starfile.read(base / "particles.star", always_dict=True)
    optics = tables["optics"].copy()
    particles = tables["particles"].iloc[: int(args.n_images)].copy()
    if len(particles) != int(args.n_images):
        raise SystemExit(f"{base} has only {len(particles)} particles")
    if optics["rlnOpticsGroup"].nunique() != 1:
        raise SystemExit("the base fixture must have one optics group")

    locations = [str(name).split("@") for name in particles["rlnImageName"]]
    stacks = {stack for _, stack in locations}
    if len(stacks) != 1:
        raise SystemExit(f"the base particles must come from one stack, got {sorted(stacks)}")
    (stack_name,) = stacks
    slices = np.asarray([int(index) - 1 for index, _ in locations], dtype=np.int64)
    box = int(optics["rlnImageSize"].iloc[0])
    out_stack_name = f"particles.{box}.mrcs"

    with mrcfile.mmap(base / stack_name, mode="r", permissive=True) as source:
        data = source.data
        voxel_size = source.voxel_size.copy()
        if data.shape[1:] != (box, box):
            raise SystemExit(f"stack images are {data.shape[1:]}, the optics table says {box}")
        ctf = relion_ctf_images(particles, optics, box, n_threads=int(args.threads))
        with mrcfile.new_mmap(
            out / out_stack_name, shape=(len(slices), box, box), mrc_mode=2, overwrite=False
        ) as target:
            for start in range(0, len(slices), int(args.batch)):
                stop = min(start + int(args.batch), len(slices))
                target.data[start:stop] = premultiply(np.asarray(data[slices[start:stop]]), ctf[start:stop])
            target.voxel_size = voxel_size

    optics["rlnCtfDataAreCtfPremultiplied"] = 1
    particles["rlnImageName"] = [f"{row + 1}@{out_stack_name}" for row in range(len(particles))]
    starfile.write({"optics": optics, "particles": particles}, out / "particles.star", overwrite=True)

    for name in COPIED_MAPS:
        if (base / name).is_file():
            shutil.copy2(base / name, out / name)
    for name in ("reference_init_classes_relion.star",):
        if (base / name).is_file():
            text = (base / name).read_text().replace(str(base), str(out))
            (out / name).write_text(text)

    here = Path(__file__).resolve()
    repo = here.parents[1]
    record = {
        "command": [sys.executable, str(here), *(argv if argv is not None else sys.argv[1:])],
        "relax_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "relax_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
            )
        ),
        "base_dir": str(base),
        "base_particles_star_sha256": _sha256(base / "particles.star"),
        "n_images": int(args.n_images),
        "features": {"ctf_premultiplied": bool(args.premultiply_ctf)},
        "ctf": "relax.relion_bind get_ctf_images_batch (CTF::setValues + getFftwImage, damping on, no padding)",
        "outputs_sha256": {path.name: _sha256(path) for path in sorted(out.iterdir()) if path.is_file()},
    }
    (out / "GENERATION.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
