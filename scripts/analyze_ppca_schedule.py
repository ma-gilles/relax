"""Shadow evaluation of adaptive-schedule criteria on a `relax ppca_initial_model` run's iterations.jsonl.

For each iteration of the run's last trajectory this prints what the run used (radius, HEALPix order, batch, step,
fudge, pose entropy) next to what three candidate criteria would ask for at that iteration. It changes no run:

  rel_order  RELION's resolution-based angular step, 360 / ceil(pi D / d), the rule auto-refine applies
             (ml_optimiser.cpp updateAngularSampling at RELION 5.0.1 f2c1a38), as the smallest HEALPix order whose
             step is at most that;
  het_order  the same with the heterogeneity-aware step d / (kappa R) of docs/development/ppca_optimizer_axes.md
             section 3 (kappa about 3.5; R the particle radius), the step below which PCs stop absorbing pose error;
  edge_mean / edge_load
             the last shell before the first whose mean (loading) gate is below --tau, RELION's data_vs_prior scan
             applied to the gate. Both pseudo-halves share one model, so a gate near 1 does not show that a shell is
             reliable; the columns exist to show whether the gate separates shells at all.

Run (CPU, seconds):
  python scripts/analyze_ppca_schedule.py RUN_DIR [--iterations 1 100 399 400 600] [--kappa 3.5] [--tau 0.9]
      [--csv OUT.csv]
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relax.sampling.healpix import angular_sampling_deg  # noqa: E402

MAX_ORDER = 8


@dataclass(frozen=True)
class Geometry:
    """Box and pixel size of the run's images and the particle diameter, in pixels and Angstrom."""

    box: int
    voxel_size: float
    diameter_ang: float


@dataclass(frozen=True)
class ScheduleRow:
    """One iteration: what the run used and what each criterion asks for."""

    iteration: int
    radius: int
    resolution_ang: float
    healpix_order: int
    rel_order: int
    het_order: int
    batch: int
    step: float
    fudge: float
    pose_entropy: float
    edge_mean: int
    edge_load: int


def last_trajectory(rows: list[dict]) -> list[dict]:
    """The records of the last run in an appended iterations.jsonl: iteration 1 starts a run and a repeated
    iteration (a resume) supersedes the earlier lines from there on."""
    run: list[dict] = []
    for row in rows:
        k = row["iteration"]
        run = [row] if k == 1 or not run else [r for r in run if r["iteration"] < k] + [row]
    return run


def resolution_ang(radius: int, geometry: Geometry) -> float:
    """Resolution in Angstrom of Fourier shell ``radius`` of the box."""
    return geometry.box * geometry.voxel_size / radius


def relion_step_deg(resolution: float, diameter: float) -> float:
    """RELION's resolution-based angular step: ``360 / ceil(pi * diameter / resolution)`` degrees."""
    return 360.0 / math.ceil(math.pi * diameter / resolution)


def heterogeneity_step_deg(resolution: float, diameter: float, kappa: float) -> float:
    """The step ``d / (kappa R)`` in degrees, ``R = diameter / 2``, below which PCs do not absorb pose error."""
    return math.degrees(resolution / (kappa * diameter / 2))


def order_for_step(step_deg: float) -> int:
    """The smallest HEALPix order whose angular step is at most ``step_deg``."""
    for order in range(MAX_ORDER + 1):
        if angular_sampling_deg(order) <= step_deg:
            return order
    raise ValueError(f"No HEALPix order up to {MAX_ORDER} has a step of at most {step_deg} degrees")


def gate_edge(gate: np.ndarray, radius: int, tau: float) -> int:
    """The last shell of ``1..radius`` before the first one whose gate is below ``tau``; ``radius`` if none is.

    RELION's ``updateCurrentResolution`` scan: the first shell below the threshold, minus one.
    """
    for shell in range(1, radius + 1):
        if gate[shell] < tau:
            return shell - 1
    return radius


def schedule_row(record: dict, geometry: Geometry, *, kappa: float, tau: float) -> ScheduleRow:
    """Evaluate the criteria on one iteration record at the run's own radius."""
    radius = int(record["radius"])
    d = resolution_ang(radius, geometry)
    gates = np.asarray(record["gates"], np.float64)
    entropies = [p["pose_entropy_mean"] for p in record["posterior"] if p.get("pose_entropy_mean") is not None]
    return ScheduleRow(
        iteration=int(record["iteration"]),
        radius=radius,
        resolution_ang=d,
        healpix_order=int(record["healpix_order"]),
        rel_order=order_for_step(relion_step_deg(d, geometry.diameter_ang)),
        het_order=order_for_step(heterogeneity_step_deg(d, geometry.diameter_ang, kappa)),
        batch=int(sum(record["half_counts"])),
        step=float(record["step"]),
        fudge=float(record["fudge"]),
        pose_entropy=float(np.mean(entropies)) if entropies else float("nan"),
        edge_mean=gate_edge(gates[:, 0], radius, tau),
        edge_load=gate_edge(gates[:, 1], radius, tau),
    )


def read_geometry(run: Path) -> Geometry:
    """Box, pixel size and particle diameter of a run, from its run.json and the particle STAR it names."""
    from recovar.data_io.starfile import read_star

    meta = json.loads((run / "run.json").read_text())
    _, optics = read_star(meta["input"]["particles"])
    return Geometry(
        box=int(optics["_rlnImageSize"].iloc[0]),
        voxel_size=float(optics["_rlnImagePixelSize"].iloc[0]),
        diameter_ang=float(meta["input"]["particle_diameter_ang"]),
    )


def format_rows(rows: list[ScheduleRow]) -> str:
    """The rows as a fixed-width table."""
    header = "iter radius  d(A) hp  rel het  batch   step  fudge  H_pose edge_mean edge_load"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.iteration:4d} {r.radius:6d} {r.resolution_ang:5.1f} {r.healpix_order:2d} {r.rel_order:4d} "
            f"{r.het_order:3d} {r.batch:6d} {r.step:6.3f} {r.fudge:6.3f} {r.pose_entropy:7.2f} {r.edge_mean:9d} "
            f"{r.edge_load:9d}"
        )
    return "\n".join(lines)


def main() -> None:
    """Print (and optionally write as CSV) the schedule rows of a run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory holding iterations.jsonl and run.json")
    parser.add_argument("--iterations", type=int, nargs="+", help="iterations to print (default: stage changes)")
    parser.add_argument("--kappa", type=float, default=3.5, help="heterogeneity-aware angular constant")
    parser.add_argument("--tau", type=float, default=0.9, help="gate threshold of the edge columns")
    parser.add_argument("--csv", type=Path, help="write every iteration to this CSV file")
    args = parser.parse_args()

    geometry = read_geometry(args.run)
    records = last_trajectory([json.loads(line) for line in open(args.run / "iterations.jsonl")])
    rows = [schedule_row(r, geometry, kappa=args.kappa, tau=args.tau) for r in records]
    if args.csv:
        with open(args.csv, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(ScheduleRow.__dataclass_fields__)
            writer.writerows(tuple(vars(r).values()) for r in rows)
    if args.iterations:
        shown = [r for r in rows if r.iteration in set(args.iterations)]
    else:
        # The first iteration, the last one before and the first after each radius or order change, and the last.
        changes = [
            i
            for i in range(1, len(rows))
            if (rows[i].radius, rows[i].healpix_order) != (rows[i - 1].radius, rows[i - 1].healpix_order)
        ]
        keep = {0, len(rows) - 1} | set(changes) | {i - 1 for i in changes}
        shown = [rows[i] for i in sorted(keep)]
    print(
        f"box {geometry.box}, voxel {geometry.voxel_size} A, diameter {geometry.diameter_ang} A, "
        f"kappa {args.kappa}, tau {args.tau}, {len(rows)} iterations"
    )
    print(format_rows(shown))


if __name__ == "__main__":
    main()
