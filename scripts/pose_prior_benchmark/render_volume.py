#!/usr/bin/env python
"""Render maps of the pose-prior benchmark in ChimeraX from the dataset's preferred viewing direction and from the side.

Viewing directions that are not uniform leave a missing cone in Fourier space around the preferred axis: a map is
sharp in the plane perpendicular to it and smeared along it. The ``front`` view looks down the preferred axis (the
eigenvector of the largest eigenvalue of the true viewing directions' second-moment matrix, from the simulator's
pose distribution; +z when the distribution is uniform), where the image plane holds the well-sampled directions.
The ``side`` view is the front view turned 90 degrees about the image's vertical axis, so the structure stays
upright and the preferred axis runs across the image, where the smearing shows.

The molecule's density has the sign of the ground-truth map's larger extreme (negative in the simulator's maps: the
GT map runs from -0.07 to +0.02). A negative molecule is drawn from a copy of the map with its sign turned (ChimeraX
closes the surface of a negative level around the whole box instead). Every map is drawn at the level that encloses
as many voxels as the ground-truth map does at ``GT_LEVEL_FRACTION`` of its extreme, so maps of different scale are
comparable, with one field of view (a distant camera of narrow angle, nearly orthographic).

ChimeraX draws offscreen only through OSMesa, which the cluster's build lacks; the script runs it under ``xvfb-run``
with software OpenGL (about 15 s of start-up, then a few seconds per image). A Python driver runs the commands one
by one and records each one that fails, since ChimeraX's own exit status and output say nothing about it.

Run (CPU node, ChimeraX on PATH or ``--chimerax``)::

    python scripts/pose_prior_benchmark/render_volume.py --dataset-dir DATASET \\
        --job learned/refine/final_merged.mrc render/learned --job dataset/reference_gt_relion.mrc render/truth

Each job writes ``<prefix>_front.png`` and ``<prefix>_side.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import mrcfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VIEWS = ("front", "side")
RENDER_NAME = "render"  # a variant's images are <evaluation dir>/render_<view>.png
TRUTH_RENDER_NAME = "render_truth"  # the ground truth's are <dataset dir>/render_truth_<view>.png
GT_LEVEL_FRACTION = 0.1  # of the ground-truth map's extreme: the surface whose enclosed volume every map matches
HEALPIX_ORDER = 4  # of the true prior whose second moment gives the preferred axis
AXIS_TOLERANCE = 0.01  # a top eigenvalue this far above 1/3 is a preferred axis; below it the views use +z
IMAGE_PIXELS = 800
SUPERSAMPLE = 3
FIELD_FRACTION = 0.75  # width of the field of view at the box centre over the box edge
FIELD_OF_VIEW_DEG = 6.0  # of the camera: narrow, so the perspective is nearly orthographic
DUST_SIZE_A = 12.0  # surface blobs smaller than this are hidden
SURFACE_COLOR = "#7fa6cf"


class RenderJob(NamedTuple):
    """One map to draw: its file, the prefix of its images, the isosurface level of the map times ``sign`` and the
    ``sign`` (-1 draws a copy of the map with its sign turned) of the molecule's density."""

    volume_path: Path
    prefix: Path
    level: float
    sign: float


class MapGeometry(NamedTuple):
    """A map's box and voxel size as the ChimeraX scene sees it (origin at 0)."""

    box: int
    voxel_size: float

    @property
    def center(self) -> np.ndarray:
        """The rotation centre of the box in scene coordinates (A)."""
        return np.full(3, self.voxel_size * self.box / 2.0)


def preferred_axis(poses: dict[str, Any], order: int = HEALPIX_ORDER) -> np.ndarray:
    """The preferred viewing axis of a pose-distribution record: a unit vector in the map frame.

    It is the eigenvector of the largest eigenvalue of the second-moment matrix of the true viewing directions
    (RELION's rot and tilt are the azimuth and polar angle of the viewing direction), signed so that its z, else y,
    else x component is positive. A distribution without a preferred axis, the top eigenvalue within
    ``AXIS_TOLERANCE`` of 1/3, gives +z.
    """
    from relax.sampling.healpix import n_pixels, pixel_rot_tilt
    from scripts.pose_prior_benchmark.true_prior import population_prior

    prior = population_prior(poses, order)
    rot, tilt = pixel_rot_tilt(order, np.arange(n_pixels(order)))
    rot, tilt = np.radians(np.asarray(rot)), np.radians(np.asarray(tilt))
    directions = np.stack([np.sin(tilt) * np.cos(rot), np.sin(tilt) * np.sin(rot), np.cos(tilt)], axis=1)
    second_moment = (prior[:, None, None] * directions[:, :, None] * directions[:, None, :]).sum(axis=0)
    values, vectors = np.linalg.eigh(second_moment)
    if values[-1] - 1.0 / 3.0 < AXIS_TOLERANCE:
        return np.array([0.0, 0.0, 1.0])
    axis = vectors[:, -1]
    leading = next(component for component in (axis[2], axis[1], axis[0]) if abs(component) > 1e-9)
    return axis if leading > 0 else -axis


def view_directions(axis: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The camera direction (towards the viewer) and the up vector of each view, for the preferred ``axis``.

    ``front`` looks down the axis with y up (the y axis made perpendicular to it; x when the axis is along y).
    ``side`` keeps that up vector and looks along the front view's right-hand direction: the front view turned 90
    degrees about the vertical.
    """
    axis = axis / np.linalg.norm(axis)
    up = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    up = up - axis * (axis @ up)
    up = up / np.linalg.norm(up)
    return {"front": (axis, up), "side": (np.cross(up, axis), up)}


def view_matrix(direction: np.ndarray, up: np.ndarray, center: np.ndarray) -> list[float]:
    """ChimeraX's 3 x 4 model matrix (row-major) that turns the map about ``center`` so ``direction`` points at the
    viewer (+z) and ``up`` points up (+y)."""
    direction = direction / np.linalg.norm(direction)
    up = up - direction * (direction @ up)
    up = up / np.linalg.norm(up)
    rotation = np.stack([np.cross(up, direction), up, direction])  # rows: screen x, y, z in map coordinates
    translation = center - rotation @ center
    return [float(x) for x in np.concatenate([rotation, translation[:, None]], axis=1).ravel()]


def molecule_sign(volume: np.ndarray) -> float:
    """-1 when the larger extreme of ``volume`` is its minimum (the molecule is negative density), else +1."""
    return -1.0 if abs(volume.min()) > abs(volume.max()) else 1.0


def enclosed_voxels(volume: np.ndarray, fraction: float = GT_LEVEL_FRACTION) -> int:
    """The number of voxels of ``volume`` above ``fraction`` of its maximum."""
    return int(np.count_nonzero(volume > fraction * volume.max()))


def matched_level(volume: np.ndarray, n_voxels: int) -> float:
    """The level above which ``volume`` has ``n_voxels`` voxels (midway to the next lower value)."""
    ordered = np.sort(volume.ravel())
    return float(0.5 * (ordered[-n_voxels - 1] + ordered[-n_voxels]))


def read_map(path: Path) -> tuple[np.ndarray, MapGeometry]:
    """A map's voxels (float64) and its geometry."""
    with mrcfile.open(path, permissive=True) as handle:
        volume = np.asarray(handle.data, dtype=np.float64)
        voxel = float(handle.voxel_size.x)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"{path}: expected a cubic map, got shape {volume.shape}")
    return volume, MapGeometry(volume.shape[0], voxel)


def chimerax_commands(
    jobs: list[RenderJob], geometry: MapGeometry, matrices: dict[str, list[float]], image_pixels: int
) -> list[str]:
    """The ChimeraX commands that render every job in every view.

    ``matrices`` holds each view's model matrix; the camera sits on the z axis through the box centre, far enough that
    its field of view is ``FIELD_FRACTION`` of the box at the centre.
    """
    center = geometry.center
    width = FIELD_FRACTION * geometry.box * geometry.voxel_size
    distance = 0.5 * width / np.tan(np.radians(FIELD_OF_VIEW_DEG) / 2.0)
    camera = f"1,0,0,{center[0]:.4f},0,1,0,{center[1]:.4f},0,0,1,{center[2] + distance:.4f}"
    commands = [
        "set bgColor white",
        "lighting soft",
        "graphics silhouettes true width 1.5",
        f"camera mono fieldOfView {FIELD_OF_VIEW_DEG}",
    ]
    for volume_path, prefix, level, sign in jobs:
        shown = "#1" if sign > 0 else "#2"
        commands += ["close", f"open {volume_path} format mrc"]
        if sign < 0:
            commands += ["volume scale #1 factor -1 modelId 2", "hide #1 models"]
        commands += [
            f"volume {shown} voxelSize {geometry.voxel_size} originIndex 0,0,0 level {level:.8g} style surface color {SURFACE_COLOR}",
            f"surface dust {shown} size {DUST_SIZE_A}",
        ]
        for view in VIEWS:
            numbers = ",".join(f"{x:.6f}" for x in matrices[view])
            commands += [
                f"view matrix models {shown},{numbers}",
                f"view matrix camera {camera}",
                f"save {prefix}_{view}.png width {image_pixels} height {image_pixels} supersample {SUPERSAMPLE}",
            ]
    return commands


def driver_script(commands: list[str], log_path: Path) -> str:
    """A ChimeraX Python script that runs ``commands`` in order and appends ``OK`` or ``FAIL`` and the error of each
    to ``log_path``."""
    return (
        "from chimerax.core.commands import run\n"
        f"commands = {commands!r}\n"
        f"with open({str(log_path)!r}, 'a') as log:\n"
        "    for command in commands:\n"
        "        try:\n"
        "            run(session, command)\n"
        "            log.write('OK   ' + command + '\\n')\n"
        "        except BaseException as error:\n"
        "            log.write('FAIL ' + command + ' :: ' + repr(error) + '\\n')\n"
        "        log.flush()\n"
    )


def chimerax_environment(chimerax: str) -> dict[str, str]:
    """The current environment plus the library directory that sits beside the ChimeraX executable (the layout of an
    install, ``usr/bin/chimerax`` and ``usr/libexec/UCSF-ChimeraX/lib``, which the cluster's module puts on
    ``LD_LIBRARY_PATH``), when it exists."""
    environment = dict(os.environ)
    library = Path(shutil.which(chimerax) or chimerax).absolute().parents[1] / "libexec" / "UCSF-ChimeraX" / "lib"
    if library.is_dir():
        environment["LD_LIBRARY_PATH"] = f"{library}:{environment['LD_LIBRARY_PATH']}" if environment.get("LD_LIBRARY_PATH") else str(library)
    return environment


def render_maps(
    dataset_dir: Path, jobs: list[tuple[Path, Path]], chimerax: str, xvfb_run: str, image_pixels: int = IMAGE_PIXELS
) -> list[Path]:
    """Render each ``(map, output prefix)`` of ``jobs`` in the front and side views; returns the PNGs written.

    The preferred axis comes from ``dataset_dir/pose_distribution.json``, the sign of the molecule's density and the
    isosurface level of every map from the ground-truth map ``dataset_dir/reference_gt_relion.mrc`` (see the module
    docstring). All maps must share its box
    and voxel size. Raises when a ChimeraX command fails or an image is missing.
    """
    poses = json.loads((dataset_dir / "pose_distribution.json").read_text())
    reference, geometry = read_map(dataset_dir / "reference_gt_relion.mrc")
    sign = molecule_sign(reference)
    n_voxels = enclosed_voxels(sign * reference)
    directions = view_directions(preferred_axis(poses))
    matrices = {view: view_matrix(*directions[view], geometry.center) for view in VIEWS}
    planned = []
    for volume_path, prefix in jobs:
        volume, other = read_map(volume_path)
        if other != geometry:
            raise ValueError(f"{volume_path} has box {other.box} and voxel size {other.voxel_size}, expected {geometry}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        planned.append(RenderJob(volume_path.resolve(), prefix.resolve(), matched_level(sign * volume, n_voxels), sign))
    with tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / "commands.log"
        script = Path(scratch) / "render.py"
        script.write_text(driver_script(chimerax_commands(planned, geometry, matrices, image_pixels), log))
        command = [xvfb_run, "-a", "-s", "-screen 0 1280x1024x24", chimerax, "--exit", str(script)]
        done = subprocess.run(command, capture_output=True, text=True, env=chimerax_environment(chimerax))
        record = log.read_text() if log.is_file() else ""
    written = [Path(f"{job.prefix}_{view}.png") for job in planned for view in VIEWS]
    failed = [line for line in record.splitlines() if line.startswith("FAIL")]
    missing = [str(path) for path in written if not path.is_file()]
    if done.returncode != 0 or failed or missing:
        raise RuntimeError(f"ChimeraX exit {done.returncode}; failed commands {failed}; missing images {missing}\n{done.stderr[-1500:]}")
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-dir", type=Path, required=True, help="the dataset (pose_distribution.json and the ground-truth map)")
    parser.add_argument(
        "--job", nargs=2, action="append", required=True, metavar=("MAP", "PREFIX"),
        help="a map to render and the prefix of its images; repeatable; a relative MAP is read as it is",
    )
    parser.add_argument("--chimerax", default=shutil.which("chimerax") or "chimerax", help="the ChimeraX executable")
    parser.add_argument("--xvfb-run", default=shutil.which("xvfb-run") or "xvfb-run")
    parser.add_argument("--image-pixels", type=int, default=IMAGE_PIXELS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    jobs = [(Path(volume), Path(prefix)) for volume, prefix in args.job]
    for path in render_maps(args.dataset_dir, jobs, args.chimerax, args.xvfb_run, args.image_pixels):
        print(path)


if __name__ == "__main__":
    main()
