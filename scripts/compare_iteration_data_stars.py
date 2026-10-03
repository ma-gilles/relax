#!/usr/bin/env python
"""Per-iteration particle agreement between two RELION-format run directories.

For every iteration whose ``run_itNNN_data.star`` exists in both directories (a RELION run, a relax
run, or one of each), particles are matched by ``rlnImageName`` (``rlnTomoParticleName`` for
subtomograms) and the script prints the fraction with the same class, the same pose (Euler angles
within ``--angle-tol`` degrees and origins within ``--origin-tol`` Angstrom), and the same
``rlnNrOfSignificantSamples``. It also prints each side's mean significant count and mean
``rlnMaxValueProbDistribution``, and the median angular distance between the two poses.

Use it to find the first iteration where two runs part (a bisect over commits, or relax against the
RELION run it replays). It reads metadata only, so it runs on CPU in seconds:

    python scripts/compare_iteration_data_stars.py RELION_DIR RELAX_DIR [--iters 1 2 3]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import starfile
from scipy.spatial.transform import Rotation

_ANGLES = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
_ORIGINS = ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")


def read_particles(path: Path):
    """The particles table of a data.star, indexed by particle name."""

    table = starfile.read(path, always_dict=True)["particles"]
    key = "rlnTomoParticleName" if "rlnTomoParticleName" in table else "rlnImageName"
    return table.set_index(key)


def _iterations(directory: Path) -> set[int]:
    pattern = re.compile(r"run_it(\d{3})_data\.star$")
    return {int(m.group(1)) for p in directory.glob("run_it*_data.star") if (m := pattern.search(p.name))}


def _angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def compare_iteration(lhs, rhs, *, angle_tol: float, origin_tol: float) -> dict:
    names = lhs.index.intersection(rhs.index)
    a, b = lhs.loc[names], rhs.loc[names]
    out = {"n": int(len(names))}
    if "rlnClassNumber" in a and "rlnClassNumber" in b:
        out["class"] = float(np.mean(a["rlnClassNumber"].to_numpy() == b["rlnClassNumber"].to_numpy()))
    same_pose = np.ones(len(names), dtype=bool)
    for col in _ANGLES:
        same_pose &= _angle_diff_deg(a[col].to_numpy(float), b[col].to_numpy(float)) <= angle_tol
    for col in _ORIGINS:
        if col in a and col in b:
            same_pose &= np.abs(a[col].to_numpy(float) - b[col].to_numpy(float)) <= origin_tol
    out["pose"] = float(same_pose.mean())
    eulers = [Rotation.from_euler("ZYZ", t[list(_ANGLES)].to_numpy(float), degrees=True) for t in (a, b)]
    out["median_angle_deg"] = float(np.median(np.degrees((eulers[0].inv() * eulers[1]).magnitude())))
    for col, label in (("rlnNrOfSignificantSamples", "nsig"), ("rlnMaxValueProbDistribution", "pmax")):
        if col in a and col in b:
            lv, rv = a[col].to_numpy(float), b[col].to_numpy(float)
            if label == "nsig":
                out["nsig_equal"] = float(np.mean(lv == rv))
            out[f"{label}_mean"] = (float(lv.mean()), float(rv.mean()))
    return out


def _format(it: int, r: dict) -> str:
    parts = [f"it{it:03d} n={r['n']}"]
    for key in ("class", "pose", "nsig_equal"):
        if key in r:
            parts.append(f"{key.replace('_equal', '')} {r[key]:.4f}")
    parts.append(f"median angle {r['median_angle_deg']:.3f} deg")
    for key in ("nsig_mean", "pmax_mean"):
        if key in r:
            parts.append(f"{key.split('_')[0]} {r[key][0]:.4f}/{r[key][1]:.4f}")
    return " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lhs", type=Path, help="first run directory (e.g. the RELION reference)")
    parser.add_argument("rhs", type=Path, help="second run directory")
    parser.add_argument("--iters", type=int, nargs="*", help="iterations to compare (default: all common ones)")
    parser.add_argument("--angle-tol", type=float, default=0.01, help="degrees per Euler angle for the same pose")
    parser.add_argument("--origin-tol", type=float, default=0.01, help="Angstrom per origin component")
    args = parser.parse_args(argv)
    common = sorted(_iterations(args.lhs) & _iterations(args.rhs))
    for it in args.iters or common:
        if it not in common:
            print(f"it{it:03d}: missing in one run")
            continue
        r = compare_iteration(
            read_particles(args.lhs / f"run_it{it:03d}_data.star"),
            read_particles(args.rhs / f"run_it{it:03d}_data.star"),
            angle_tol=args.angle_tol,
            origin_tol=args.origin_tol,
        )
        print(_format(it, r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
