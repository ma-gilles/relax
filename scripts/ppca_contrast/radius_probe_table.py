#!/usr/bin/env python3
"""Tabulate the contrast estimates of ``infer_for_suite.py`` runs at several Fourier radii against the simulated contrast.

``infer_for_suite.py`` re-runs a checkpoint's own E-step on the evaluation images; run it once per radius into
``PROBE/<arm>_c<checkpoint>_r<radius>/suite_score_checkpoint_<checkpoint>/inference.npz``. For each such directory this
prints the mean and spread of the estimated contrast, the regression ``estimate = a + b * truth`` and the correlation,
so the dependence of the estimate's mean (and its slope) on the radius is visible.

Run:
    python -I scripts/ppca_contrast/radius_probe_table.py PROBE --truth DATASET/evaluation/contrast.npy
"""

import argparse
import re
from pathlib import Path

import numpy as np


def probe_row(inference: Path, truth: np.ndarray) -> dict[str, float]:
    """Mean, spread, regression on the true contrast and correlation of one ``inference.npz``'s contrast estimate."""
    with np.load(inference, allow_pickle=False) as saved:
        ids, estimate = saved["particle_ids"], saved["contrast"].astype(np.float64)
    true = truth[ids]
    slope, intercept = np.polyfit(true, estimate, 1)
    return {
        "n": float(len(ids)),
        "mean": float(estimate.mean()),
        "sd": float(estimate.std()),
        "intercept": float(intercept),
        "slope": float(slope),
        "corr": float(np.corrcoef(true, estimate)[0, 1]),
    }


def main() -> None:
    """Print one table of rows ordered by arm, checkpoint and radius."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe", type=Path, help="directory of <arm>_c<checkpoint>_r<radius> work directories")
    parser.add_argument("--truth", type=Path, required=True, help="the dataset's evaluation/contrast.npy")
    args = parser.parse_args()
    truth = np.load(args.truth)
    print(f"true contrast: mean {truth.mean():.3f}, sd {truth.std():.3f}")
    print("arm checkpoint radius |  n    mean   sd   | estimate = a + b * truth | corr")
    pattern = re.compile(r"(?P<arm>.+)_c(?P<ckpt>\d+)_r(?P<radius>\d+)$")
    entries = []
    for work in args.probe.iterdir():
        match = pattern.match(work.name)
        if not match:
            continue
        for inference in work.glob("suite_score_checkpoint_*/inference.npz"):
            entries.append((match["arm"], int(match["ckpt"]), int(match["radius"]), inference))
    for arm, ckpt, radius, inference in sorted(entries):
        r = probe_row(inference, truth)
        print(
            f"{arm} {ckpt:5d} {radius:3d} | {int(r['n']):5d} {r['mean']:.3f} {r['sd']:.3f} | "
            f"{r['intercept']:+.3f} + {r['slope']:.3f} | {r['corr']:.3f}"
        )


if __name__ == "__main__":
    main()
