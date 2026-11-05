#!/usr/bin/env python
"""Evaluate one pose-prior benchmark run: map resolution and direction-prior accuracy.

Resolution (unmasked FSC; resolution = box * voxel / first shell below the threshold, as the scorecards do):
  * half-map FSC of ``final_half1_unfil.mrc`` and ``final_half2_unfil.mrc`` at 0.143;
  * FSC of the final merged map, and of each half map, to the ground-truth map at 0.5 and 0.143;
  * the resolution the refinement reported;
  * per numbered iteration (``fsc_by_iteration.csv``): each unfiltered half map's FSC to the ground truth and the two
    half maps' FSC to each other and the Fourier radius the iteration ran with, so runs that stop at different
    iterations or orders compare at equal iteration.

Conical FSC (``scripts/conical_fsc.py``, which cites its sources): the FSC inside cones of half-angle
``--cone-half-angle-deg`` along ``--n-cone-axes`` axes spread over the hemisphere, for the two half maps (as refined) and
for the final merged map against the truth (aligned). ``metrics.json`` ``conical_fsc.{half_maps,gt_merged_aligned}``
holds each pair's cFAR (the smallest over the largest area under the cone curves: 1 for an isotropic map),
the axes of the smallest and largest area and the crossing resolutions of the best and worst cone;
``conical_fsc.npz`` holds every cone's curve and voxel counts and ``conical_fsc.png`` the mean +- SD curves and the
heatmap of the areas over the cone axis. A cone axis is a direction in Fourier space in the frame of the maps (RELION's),
not a viewing direction.

Direction-prior accuracy: ``run_itNNN_half{1,2}_model.star`` holds each half set's learned direction prior
(``model_pdf_orient_class_1``, one probability per HEALPix direction at that iteration's order). Its target is
the population distribution of the true viewing directions at the same order, computed in closed form from the
simulator's pose distribution (no sampling noise). Per iteration and half: total variation, Jensen-Shannon
divergence, Hellinger distance, KL divergence and the population mass the prior sets to zero. Two references
sit beside them: the uniform prior, and the histogram of the half's true viewing directions (the plug-in prior
that perfect poses would give), which is how close any estimate from that many particles can come. A run scored
with ``relax refine --fixed-direction-prior`` still estimates and writes the learned prior, so it is evaluated
the same way; ``run.direction_prior_used`` in ``metrics.json`` says which prior scored the run.

Frame offset: a run started from a map that sits a little off the ground truth's frame returns maps and poses in that
frame, which adds the offset to every pose error and takes from the map's FSC to the truth. Both are therefore reported
twice, as refined and aligned:
  * ``pose_frame_offset`` in ``metrics.json`` is the average pose offset: the rotation (matrix, angle in degrees, and the
    share of particles within 15 degrees of it) that best maps the true viewing directions onto the estimated ones.
    ``pose_direction_error_aligned_deg`` is the direction error after turning the estimated directions back by it;
    ``pose_direction_error_deg`` is the error as refined.
  * ``resolution.alignment`` is the rotation and shift that bring the final merged map onto the ground truth
    (``align_to_truth.py``, fitted on both maps low-passed at ``ALIGN_RESOLUTION_A``); ``gt_merged_aligned`` and
    ``gt_half{1,2}_aligned`` are the FSC to the truth of the final maps after that transform, ``gt_merged`` and
    ``gt_half{1,2}`` as refined. The half-map FSC does not depend on the frame. ``interpolation_round_trip`` is the FSC
    to the truth of the truth transformed and transformed back, which is what the interpolation alone costs an aligned
    map. The per-iteration FSC (``fsc_by_iteration.csv``) is as refined.

Run::

    python scripts/pose_prior_benchmark/evaluate_run.py --dataset-dir DATASET --run-dir REFINE --out-dir OUT

``OUT`` receives ``metrics.json``, ``prior_by_iteration.csv``, ``fsc_by_iteration.csv``, ``fsc_curves.csv``,
``pose_errors.npz`` (every particle's final direction error, degrees, one array per half set as ``half1``, ``half2``
and aligned as ``aligned_half1``, ``aligned_half2``), ``conical_fsc.npz`` (``axes``, and per pair ``<pair>_curves`` and
``<pair>_voxels``, one row per cone), ``conical_fsc.png`` and ``summary.png`` (the FSC curves and every learned-prior
metric per iteration). The orientation-distribution plots are made by the Snakefile
with ``angdist``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from relax.sampling.orientation_priors import infer_direction_prior_healpix_order  # noqa: E402
from relax.star.blocks import read_star_blocks  # noqa: E402
from scripts.conical_fsc import (  # noqa: E402
    FSC_THRESHOLD,
    HALF_ANGLE_DEG,
    N_AXES,
    ConicalFsc,
    cone_axes,
    conical_fsc,
    conical_fsc_summary,
)
from scripts.fsc_metrics import centered_corr, first_shell_below, normalized_fsc_auc, shell_fsc  # noqa: E402
from scripts.masked_fsc import first_sustained_below  # noqa: E402
from scripts.pose_prior_benchmark.align_to_truth import (  # noqa: E402
    apply_fit,
    fit_rigid_to_reference,
    inverse_fit,
    rotation_angle_deg,
)
from scripts.pose_prior_benchmark.conical_fsc_plot import plot_conical_fsc  # noqa: E402
from scripts.pose_prior_benchmark.true_prior import population_prior  # noqa: E402

HALF_FSC_THRESHOLD = 0.143
GT_FSC_THRESHOLDS = (0.5, 0.143)
SUSTAINED_SHELLS = 3  # the scorecards' crossing rule: this many consecutive shells below the threshold
FRAME_FIT_TRIMS_DEG = (60.0, 30.0, 15.0)  # frame_offset keeps the particles within each of these of the previous fit
ALIGN_RESOLUTION_A = 10.0  # the maps are low-passed at this resolution to fit their rigid alignment to the truth
PRIOR_SUM_TOLERANCE = 1e-3
PRIOR_BLOCK = "model_pdf_orient_class_1"
GT_MAP_NAME = "reference_gt_relion.mrc"
RUN_MAP_NAMES = {"merged": "final_merged.mrc", "half1": "final_half1_unfil.mrc", "half2": "final_half2_unfil.mrc"}
HALVES = (1, 2)
PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e3e2de",
    "half1": "#2a78d6",
    "half2": "#eb6834",
    "reference": "#8a8984",
    "order": "#c4c3be",
}


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------


def image_key(name: str) -> tuple[int, str]:
    """Return ``(stack index, stack file name)`` of an ``index@stack.mrcs`` image name."""
    index, _, stack = name.partition("@")
    return int(index), Path(stack).name


def read_particles(star_path: Path) -> dict[str, Any]:
    """Read image keys, optional half-set labels and Euler angles (degrees) of a particle STAR file."""
    table = read_star_blocks(star_path)["particles"]
    out: dict[str, Any] = {"image_key": [image_key(name) for name in table["rlnImageName"]]}
    for label, column in (("rot", "rlnAngleRot"), ("tilt", "rlnAngleTilt")):
        out[label] = np.asarray(table[column], dtype=np.float64)
    if "rlnRandomSubset" in table:
        out["half"] = np.asarray(table["rlnRandomSubset"], dtype=np.int64)
    return out


def read_dataset_geometry(dataset_dir: Path) -> tuple[float, int]:
    """Return the simulated particles' voxel size (A) and box size (pixels)."""
    optics = read_star_blocks(dataset_dir / "particles.star")["optics"]
    return float(optics["rlnImagePixelSize"][0]), int(optics["rlnImageSize"][0])


def list_iterations(run_dir: Path) -> list[int]:
    """Return the numbered iterations whose two half-set model STAR files exist, ascending."""
    names = (p.name for p in run_dir.glob("run_it*_half1_model.star"))
    found = {int(m.group(1)) for m in (re.fullmatch(r"run_it(\d+)_half1_model\.star", n) for n in names) if m}
    iterations = sorted(i for i in found if (run_dir / f"run_it{i:03d}_half2_model.star").is_file())
    if not iterations:
        raise FileNotFoundError(f"no run_itNNN_half{{1,2}}_model.star pair in {run_dir}")
    return iterations


def read_direction_prior(model_star: Path) -> np.ndarray | None:
    """Return a half set's learned direction prior, or None when its model STAR has no prior block."""
    block = read_star_blocks(model_star).get(PRIOR_BLOCK)
    if not block or not block.get("rlnOrientationDistribution"):
        return None
    prior = np.asarray(block["rlnOrientationDistribution"], dtype=np.float64)
    if not np.all(np.isfinite(prior)) or np.any(prior < 0) or abs(prior.sum() - 1.0) > PRIOR_SUM_TOLERANCE:
        raise ValueError(f"{model_star}: direction prior is not a probability vector (sum {prior.sum():.6f})")
    return prior


def true_directions_per_half(run_dir: Path, iteration: int, dataset_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    """Split the true and the estimated (rot, tilt) of the particles by the run's half sets.

    The run's ``run_itNNN_data.star`` gives each particle's half set and estimated angles; the dataset's
    ``particles.star`` gives the true angles. Particles are matched by image name.
    """
    truth = read_particles(dataset_dir / "particles.star")
    row_of = {key: i for i, key in enumerate(truth["image_key"])}
    if len(row_of) != len(truth["image_key"]):
        raise ValueError("particles.star has duplicate image names")
    run = read_particles(run_dir / f"run_it{iteration:03d}_data.star")
    if "half" not in run:
        raise ValueError(f"run_it{iteration:03d}_data.star has no rlnRandomSubset column")
    try:
        rows = np.asarray([row_of[key] for key in run["image_key"]])
    except KeyError as exc:
        raise ValueError(f"run particle {exc.args[0]} is not in the dataset's particles.star") from exc
    if rows.size != len(row_of):
        raise ValueError(f"the run holds {rows.size} particles, the dataset {len(row_of)}")
    halves = {}
    for half in HALVES:
        keep = run["half"] == half
        halves[half] = {
            "true_rot": truth["rot"][rows[keep]],
            "true_tilt": truth["tilt"][rows[keep]],
            "est_rot": run["rot"][keep],
            "est_tilt": run["tilt"][keep],
        }
    return halves


def read_pose_distribution(dataset_dir: Path) -> dict[str, Any]:
    """Read the simulator's pose-distribution record (``simulate_dataset.py``)."""
    record = json.loads((dataset_dir / "pose_distribution.json").read_text())
    if record["poses"] not in ("uniform", "kent"):
        raise ValueError(f"unknown pose distribution {record['poses']!r}")
    return record


# ---------------------------------------------------------------------------------------------------------------
# Direction-prior accuracy
# ---------------------------------------------------------------------------------------------------------------


def direction_histogram(rot_deg: np.ndarray, tilt_deg: np.ndarray, order: int) -> np.ndarray:
    """Count viewing directions per RELION HEALPix (NEST) direction at ``order``; ``(rot, tilt)`` are degrees."""
    nside = 1 << int(order)
    pixels = hp.ang2pix(nside, np.deg2rad(tilt_deg), np.deg2rad(rot_deg), nest=True)
    return np.bincount(pixels, minlength=hp.nside2npix(nside)).astype(np.float64)


def kl_bits(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence ``sum p log2(p / q)`` over the support of ``p``; ``q`` must be positive there."""
    support = p > 0
    return float(np.sum(p[support] * np.log2(p[support] / q[support])))


def distribution_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Distances from a probability vector ``estimate`` to the ``truth`` distribution (both normalized here).

    ``kl_bits`` is KL(truth || estimate), infinite when the estimate is zero where the truth has mass;
    ``truth_mass_on_zero_prior`` is that mass.
    """
    p = truth / truth.sum()
    q = estimate / estimate.sum()
    mid = 0.5 * (p + q)
    lost = float(p[q == 0].sum())
    return {
        "tv": float(0.5 * np.abs(p - q).sum()),
        "js_bits": 0.5 * kl_bits(p, mid) + 0.5 * kl_bits(q, mid),
        "hellinger": float(math.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))),
        "kl_bits": math.inf if lost > 0 else kl_bits(p, q),
        "truth_mass_on_zero_prior": lost,
    }


def js_skill(js_bits: float, js_bits_uniform: float) -> float:
    """1 for a perfect prior and 0 for a uniform one; NaN when the truth is uniform (nothing to gain)."""
    return 1.0 - js_bits / js_bits_uniform if js_bits_uniform > 0 else math.nan


def prior_rows(
    run_dir: Path, iterations: list[int], halves: dict[int, dict[str, np.ndarray]], poses: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per iteration and half set: the learned prior's distances to the population of true directions.

    ``*_uniform`` is the same metric for a uniform prior and ``*_oracle`` for the histogram of the half's true
    directions (the prior perfect poses would give); ``js_skill`` is ``1 - JS / JS of the uniform prior`` and
    ``js_bits_vs_true_histogram`` is the learned prior's Jensen-Shannon distance to that histogram.
    """
    populations: dict[int, np.ndarray] = {}

    def population_at(order: int) -> np.ndarray:
        if order not in populations:
            populations[order] = population_prior(poses, order)
        return populations[order]

    rows = []
    for iteration in iterations:
        for half in HALVES:
            row: dict[str, Any] = {"iteration": iteration, "half": half}
            prior = read_direction_prior(run_dir / f"run_it{iteration:03d}_half{half}_model.star")
            if prior is None:
                rows.append({**row, "prior_available": False})
                continue
            order = infer_direction_prior_healpix_order(prior)
            population = population_at(order)
            histogram = direction_histogram(halves[half]["true_rot"], halves[half]["true_tilt"], order)
            uniform = distribution_metrics(population, np.full(prior.size, 1.0 / prior.size))
            oracle = distribution_metrics(population, histogram)
            metrics = distribution_metrics(population, prior)
            rows.append(
                {
                    **row,
                    "prior_available": True,
                    "healpix_order": order,
                    "n_directions": prior.size,
                    "n_particles": int(histogram.sum()),
                    **metrics,
                    "zero_prior_fraction": float(np.mean(prior == 0)),
                    "tv_uniform": uniform["tv"],
                    "hellinger_uniform": uniform["hellinger"],
                    "kl_bits_uniform": uniform["kl_bits"],
                    "js_bits_uniform": uniform["js_bits"],
                    "js_skill": js_skill(metrics["js_bits"], uniform["js_bits"]),
                    "tv_oracle": oracle["tv"],
                    "hellinger_oracle": oracle["hellinger"],
                    "kl_bits_oracle": oracle["kl_bits"],
                    "js_bits_oracle": oracle["js_bits"],
                    "js_skill_oracle": js_skill(oracle["js_bits"], uniform["js_bits"]),
                    "truth_mass_on_zero_prior_oracle": oracle["truth_mass_on_zero_prior"],
                    "zero_prior_fraction_oracle": float(np.mean(histogram == 0)),
                    "js_bits_vs_true_histogram": distribution_metrics(histogram, prior)["js_bits"],
                }
            )
    return rows


def truth_nonuniformity(poses: dict[str, Any], halves: dict[int, dict[str, np.ndarray]], order: int) -> dict[str, Any]:
    """How far the simulated viewing directions are from uniform, at ``order``."""
    p = population_prior(poses, order)
    cumulative = np.cumsum(np.sort(p)[::-1])
    histogram = sum(direction_histogram(halves[h]["true_rot"], halves[h]["true_tilt"], order) for h in HALVES)
    return {
        "poses": poses["poses"],
        "healpix_order": order,
        "kl_from_uniform_bits": kl_bits(p, np.full(p.size, 1.0 / p.size)),
        "sphere_fraction_with_half_of_mass": float((np.searchsorted(cumulative, 0.5) + 1) / p.size),
        "directions_without_a_particle": float(np.mean(histogram == 0)),
    }


def viewing_directions(rot: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    """Unit viewing directions (n, 3) of RELION's (rot, tilt) in degrees."""
    r, t = np.deg2rad(rot), np.deg2rad(tilt)
    return np.stack([np.sin(t) * np.cos(r), np.sin(t) * np.sin(r), np.cos(t)], axis=-1)


def direction_error_deg(
    true_rot: np.ndarray, true_tilt: np.ndarray, est_rot: np.ndarray, est_tilt: np.ndarray
) -> np.ndarray:
    """Angle (degrees) between true and estimated viewing directions."""
    cosine = np.sum(viewing_directions(true_rot, true_tilt) * viewing_directions(est_rot, est_tilt), axis=-1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def aligned_direction_error_deg(
    true_rot: np.ndarray, true_tilt: np.ndarray, est_rot: np.ndarray, est_tilt: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """Angle (degrees) between the true viewing directions and the estimated ones turned back by the frame offset
    ``rotation`` (the ``rotation_matrix`` of ``frame_offset``, ``est = rotation true``)."""
    cosine = np.sum(
        viewing_directions(true_rot, true_tilt) * (viewing_directions(est_rot, est_tilt) @ rotation), axis=-1
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def frame_offset(true_directions: np.ndarray, est_directions: np.ndarray) -> dict[str, Any]:
    """The rotation Q with ``est = Q true`` that fits the viewing directions: its matrix and angle in degrees.

    A refinement started from a map in a rotated frame returns poses in that frame, so its pose errors include the
    offset. Q is a Kabsch fit trimmed to the particles within 60, 30 and 15 degrees of the previous fit;
    ``inlier_fraction`` is the share of particles within 15 degrees of the final fit.
    """
    q = np.eye(3)
    for trim in FRAME_FIT_TRIMS_DEG:
        angle = np.degrees(np.arccos(np.clip(np.sum(est_directions * (true_directions @ q.T), axis=-1), -1.0, 1.0)))
        keep = angle < trim
        u, _, vt = np.linalg.svd(true_directions[keep].T @ est_directions[keep])
        q = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    residual = np.degrees(np.arccos(np.clip(np.sum(est_directions * (true_directions @ q.T), axis=-1), -1.0, 1.0)))
    return {
        "angle_deg": rotation_angle_deg(q),
        "rotation_matrix": q.tolist(),
        "inlier_fraction": float(np.mean(residual < FRAME_FIT_TRIMS_DEG[-1])),
    }


# ---------------------------------------------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------------------------------------------


def load_maps(dataset_dir: Path, run_dir: Path) -> dict[str, np.ndarray]:
    """Load the ground-truth map and the run's final maps in one frame (RELION-convention files)."""
    from recovar.utils.helpers import load_relion_volume

    from relax.io.map_io import is_relax_map, require_relion_convention_reference

    gt_path = dataset_dir / GT_MAP_NAME
    require_relion_convention_reference(gt_path, option="ground-truth map")
    maps = {"gt": np.asarray(load_relion_volume(str(gt_path)), dtype=np.float64)}
    for name, file in RUN_MAP_NAMES.items():
        path = run_dir / file
        if not is_relax_map(path):
            raise ValueError(f"{path} has no relax map label")
        maps[name] = np.asarray(load_relion_volume(str(path)), dtype=np.float64)
    shapes = {m.shape for m in maps.values()}
    if len(shapes) != 1:
        raise ValueError(f"map shapes differ: {sorted(shapes)}")
    return maps


def load_iteration_half_maps(run_dir: Path, iteration: int) -> tuple[np.ndarray, np.ndarray]:
    """The two unfiltered half maps ``run_itNNN_half{1,2}_class001_unfil.mrc`` of one numbered iteration."""
    from recovar.utils.helpers import load_relion_volume

    from relax.io.map_io import is_relax_map

    halves = []
    for half in HALVES:
        path = run_dir / f"run_it{iteration:03d}_half{half}_class001_unfil.mrc"
        if not is_relax_map(path):
            raise ValueError(f"{path} has no relax map label")
        halves.append(np.asarray(load_relion_volume(str(path)), dtype=np.float64))
    return halves[0], halves[1]


def fsc_by_iteration(
    gt: np.ndarray,
    half_maps: dict[int, tuple[np.ndarray, np.ndarray]],
    orders: dict[int, int],
    radii: dict[int, int],
    voxel: float,
    box: int,
) -> list[dict[str, Any]]:
    """Per numbered iteration: each unfiltered half map's FSC to the ground truth ``gt`` and to the other half.

    Columns: the iteration, its HEALPix order and its Fourier radius (``radii``, pixels); the area under the half
    maps' FSC to the truth (each half, and their mean ``gt_auc``); the resolution (A) where each half's FSC to the
    truth first drops below 0.5; the area under the two halves' FSC to each other (``half_map_auc``) and the
    resolution where it drops below 0.143 (empty: never).
    """
    rows = []
    for iteration, (half1, half2) in sorted(half_maps.items()):
        gt_curves = [shell_fsc(half, gt) for half in (half1, half2)]
        half_curve = shell_fsc(half1, half2)
        aucs = [normalized_fsc_auc(curve) for curve in gt_curves]
        rows.append(
            {
                "iteration": iteration,
                "healpix_order": orders[iteration],
                "fourier_radius": radii[iteration],
                "gt_auc_half1": aucs[0],
                "gt_auc_half2": aucs[1],
                "gt_auc": float(np.mean(aucs)),
                "gt_resolution_0.5_half1_A": crossing_summary(gt_curves[0], 0.5, voxel, box)["first_A"],
                "gt_resolution_0.5_half2_A": crossing_summary(gt_curves[1], 0.5, voxel, box)["first_A"],
                "half_map_auc": normalized_fsc_auc(half_curve),
                "half_map_resolution_0.143_A": crossing_summary(half_curve, HALF_FSC_THRESHOLD, voxel, box)["first_A"],
            }
        )
    return rows


def crossing_summary(curve: np.ndarray, threshold: float, voxel: float, box: int) -> dict[str, Any]:
    """Resolution (A) of the first and the sustained FSC crossing below ``threshold``; None: never crosses."""

    def resolution(shell: int | None) -> float | None:
        return None if shell is None else box * voxel / shell

    first = first_shell_below(curve, threshold)
    sustained = first_sustained_below(curve, threshold, SUSTAINED_SHELLS)
    return {
        "first_shell": first,
        "first_A": resolution(first),
        "sustained_shell": sustained,
        "sustained_A": resolution(sustained),
    }


def fsc_report(curve: np.ndarray, thresholds: tuple[float, ...], voxel: float, box: int) -> dict[str, Any]:
    """Crossings at each threshold and the full-band area under one FSC curve."""
    report: dict[str, Any] = {f"at_{t}": crossing_summary(curve, t, voxel, box) for t in thresholds}
    report["auc"] = normalized_fsc_auc(curve)
    return report


def align_run_maps(
    maps: dict[str, np.ndarray], voxel: float, box: int
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    """The run's final maps in the ground truth's frame, the record of the alignment and its interpolation floor.

    The transform is fitted on the merged map (``align_to_truth``, band ``ALIGN_RESOLUTION_A``) and applied to it and
    to both half maps. The floor is the FSC to the truth of the truth transformed and transformed back.
    """
    shell = round(box * voxel / ALIGN_RESOLUTION_A)
    fit = fit_rigid_to_reference(maps["merged"], maps["gt"], shell)
    aligned = {name: apply_fit(maps[name], fit) for name in RUN_MAP_NAMES}
    round_trip = shell_fsc(apply_fit(apply_fit(maps["gt"], fit), inverse_fit(fit)), maps["gt"])
    record = {
        "band_resolution_A": ALIGN_RESOLUTION_A,
        "band_shell": shell,
        "angle_deg": rotation_angle_deg(fit.rotation),
        "rotation_matrix": fit.rotation.tolist(),
        "translation_voxels": fit.translation_voxels.tolist(),
        "shift_A": float(np.linalg.norm(fit.translation_voxels) * voxel),
        "scale": fit.scale,
        "iterations": fit.iterations,
        "last_rotation_step_deg": fit.last_rotation_step_deg,
        "last_shift_step_voxels": fit.last_shift_step_voxels,
        "interpolation_round_trip_fsc_auc": normalized_fsc_auc(round_trip),
    }
    return aligned, record, round_trip


def resolution_metrics(
    maps: dict[str, np.ndarray], aligned: dict[str, np.ndarray], voxel: float, box: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """FSC curves and resolution summaries: half maps against each other and the run's maps against truth, as
    refined (``maps``) and in the truth's frame (``aligned``, the keys of ``RUN_MAP_NAMES``)."""
    if maps["gt"].shape[0] != box:
        raise ValueError(f"maps have box {maps['gt'].shape[0]}, the particles {box}")
    corr = centered_corr(maps["merged"], maps["gt"])
    if not corr > 0:
        raise ValueError(f"final map and ground truth correlate at {corr:.3f}: the map convention or frame differs")
    curves = {
        "half_maps": shell_fsc(maps["half1"], maps["half2"]),
        "gt_merged": shell_fsc(maps["merged"], maps["gt"]),
        "gt_half1": shell_fsc(maps["half1"], maps["gt"]),
        "gt_half2": shell_fsc(maps["half2"], maps["gt"]),
        **{f"gt_{name}_aligned": shell_fsc(aligned[name], maps["gt"]) for name in RUN_MAP_NAMES},
    }
    report = {
        "voxel_size_A": voxel,
        "box": box,
        "nyquist_A": 2 * voxel,
        "merged_gt_correlation": corr,
        "half_maps": fsc_report(curves["half_maps"], (HALF_FSC_THRESHOLD,), voxel, box),
        **{name: fsc_report(curves[name], GT_FSC_THRESHOLDS, voxel, box) for name in curves if name != "half_maps"},
    }
    return report, curves


def conical_fsc_metrics(
    maps: dict[str, np.ndarray],
    aligned: dict[str, np.ndarray],
    voxel: float,
    box: int,
    half_angle_deg: float = HALF_ANGLE_DEG,
    n_axes: int = N_AXES,
) -> tuple[dict[str, dict[str, Any]], dict[str, ConicalFsc]]:
    """The cFSC of the half maps against each other (as refined) and of the final merged map against the truth
    (``aligned``, in the truth's frame), with each pair's summary at the 0.143 threshold of cFAR.

    The keys are ``half_maps`` and ``gt_merged_aligned``, as in ``resolution_metrics``.
    """
    axes = cone_axes(n_axes)
    pairs = {
        "half_maps": (maps["half1"], maps["half2"]),
        "gt_merged_aligned": (aligned["merged"], maps["gt"]),
    }
    cfscs = {name: conical_fsc(a, b, axes, half_angle_deg) for name, (a, b) in pairs.items()}
    return {name: conical_fsc_summary(cfsc, voxel, box, FSC_THRESHOLD) for name, cfsc in cfscs.items()}, cfscs


def conical_figure_pairs(result: dict[str, Any]) -> dict[str, tuple]:
    """``plot_conical_fsc``'s input from an ``evaluate`` result: per pair its cFSC, global FSC, summary and the
    thresholds drawn."""
    titles = {
        "half_maps": ("half maps", (HALF_FSC_THRESHOLD,)),
        "gt_merged_aligned": ("final map vs ground truth, aligned", GT_FSC_THRESHOLDS),
    }
    return {
        title: (result["conical_fsc"][name], result["curves"][name], result["metrics"]["conical_fsc"][name], levels)
        for name, (title, levels) in titles.items()
    }


def require_unseeded_run(run_dir: Path) -> str:
    """Return the pose source the run recorded; refuse a run that started from the input STAR's poses.

    The simulated ``particles.star`` holds the true angles and origins, so a run seeded from them is not a
    blind refinement (``--initial-pose-source none`` ignores them).
    """
    with np.load(run_dir / "refinement_results.npz", allow_pickle=False) as archive:
        resolved = str(archive["initial_pose_source_resolved"])
    if resolved != "none":
        raise ValueError(
            f"{run_dir} started from poses of source {resolved!r}; the benchmark needs --initial-pose-source none"
        )
    return resolved


def read_direction_prior_used(run_dir: Path) -> dict[str, Any]:
    """Which direction prior scored the run: ``fixed`` with the file relax recorded, else the ``learned`` one."""
    record = run_dir / "fixed_direction_prior.json"
    if not record.is_file():
        return {"source": "learned"}
    return {"source": "fixed", **json.loads(record.read_text())}


def read_fourier_radius(run_dir: Path, iteration: int) -> int:
    """The Fourier radius (pixels) the iteration ran with: half its ``rlnCurrentImageSize``, the same in both halves."""
    sizes = {
        int(
            read_star_blocks(run_dir / f"run_it{iteration:03d}_half{half}_model.star")["model_general"][
                "rlnCurrentImageSize"
            ]
        )
        for half in HALVES
    }
    if len(sizes) != 1:
        raise ValueError(f"iteration {iteration}: the half sets ran with different image sizes {sorted(sizes)}")
    return sizes.pop() // 2


def reported_resolution(run_dir: Path, iteration: int) -> dict[str, float]:
    """The resolution (A) the refinement reported: its final all-data pass and the last numbered iteration."""
    with np.load(run_dir / "refinement_results.npz", allow_pickle=False) as archive:
        reported = {"final_all_data_A": float(archive["convergence_current_resolution"])}
    for half in HALVES:
        general = read_star_blocks(run_dir / f"run_it{iteration:03d}_half{half}_model.star")["model_general"]
        reported[f"last_iteration_half{half}_A"] = float(general["rlnCurrentResolution"])
    return reported


# ---------------------------------------------------------------------------------------------------------------
# Evaluation and outputs
# ---------------------------------------------------------------------------------------------------------------


def evaluate(
    dataset_dir: Path, run_dir: Path, cone_half_angle_deg: float = HALF_ANGLE_DEG, n_cone_axes: int = N_AXES
) -> dict[str, Any]:
    """Evaluate one run; returns the metrics plus the arrays the figure and the CSV files need."""
    pose_source = require_unseeded_run(run_dir)
    voxel, box = read_dataset_geometry(dataset_dir)
    poses = read_pose_distribution(dataset_dir)
    iterations = list_iterations(run_dir)
    final = iterations[-1]
    halves = true_directions_per_half(run_dir, final, dataset_dir)
    rows = prior_rows(run_dir, iterations, halves, poses)
    final_rows = [r for r in rows if r["iteration"] == final]
    if not all(r["prior_available"] for r in final_rows):
        raise ValueError(f"iteration {final} has no direction prior in a half-set model STAR")
    order = final_rows[0]["healpix_order"]
    maps = load_maps(dataset_dir, run_dir)
    aligned_maps, alignment, round_trip = align_run_maps(maps, voxel, box)
    resolution, curves = resolution_metrics(maps, aligned_maps, voxel, box)
    curves["interpolation_round_trip"] = round_trip
    conical, cfscs = conical_fsc_metrics(maps, aligned_maps, voxel, box, cone_half_angle_deg, n_cone_axes)
    orders = {r["iteration"]: r["healpix_order"] for r in rows if r["half"] == 1 and r["prior_available"]}
    iteration_rows = fsc_by_iteration(
        maps["gt"],
        {i: load_iteration_half_maps(run_dir, i) for i in iterations},
        orders,
        {i: read_fourier_radius(run_dir, i) for i in iterations},
        voxel,
        box,
    )
    errors = {
        half: direction_error_deg(h["true_rot"], h["true_tilt"], h["est_rot"], h["est_tilt"])
        for half, h in halves.items()
    }
    offset = frame_offset(
        np.concatenate([viewing_directions(h["true_rot"], h["true_tilt"]) for h in halves.values()]),
        np.concatenate([viewing_directions(h["est_rot"], h["est_tilt"]) for h in halves.values()]),
    )
    aligned_errors = {
        half: aligned_direction_error_deg(
            h["true_rot"], h["true_tilt"], h["est_rot"], h["est_tilt"], np.asarray(offset["rotation_matrix"])
        )
        for half, h in halves.items()
    }
    metrics = {
        "run": {
            "n_iterations": len(iterations),
            "final_iteration": final,
            "final_healpix_order": order,
            "initial_pose_source": pose_source,
            "direction_prior_used": read_direction_prior_used(run_dir),
        },
        "truth": truth_nonuniformity(poses, halves, order),
        "resolution": {**resolution, "alignment": alignment, "reported": reported_resolution(run_dir, final)},
        "conical_fsc": conical,
        "prior_final": {
            f"half{r['half']}": {k: v for k, v in r.items() if k not in ("iteration", "half", "prior_available")}
            for r in final_rows
        },
        **{
            key: {
                f"half{h}": {
                    "median": float(np.median(e)),
                    "p90": float(np.percentile(e, 90)),
                    "fraction_over_90": float(np.mean(e > 90)),
                }
                for h, e in per_half.items()
            }
            for key, per_half in (
                ("pose_direction_error_deg", errors),
                ("pose_direction_error_aligned_deg", aligned_errors),
            )
        },
        "pose_frame_offset": offset,
    }
    return {
        "metrics": metrics,
        "rows": rows,
        "iteration_rows": iteration_rows,
        "curves": curves,
        "conical_fsc": cfscs,
        "pose_errors": errors,
        "aligned_pose_errors": aligned_errors,
        "voxel": voxel,
        "box": box,
    }


def jsonable(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats (None) for ``json.dump``."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_tables(result: dict[str, Any], out_dir: Path) -> None:
    """Write ``metrics.json``, ``prior_by_iteration.csv``, ``fsc_by_iteration.csv``, ``fsc_curves.csv``,
    ``pose_errors.npz`` and ``conical_fsc.npz``."""
    (out_dir / "metrics.json").write_text(json.dumps(jsonable(result["metrics"]), indent=2) + "\n")
    np.savez_compressed(
        out_dir / "pose_errors.npz",
        **{f"half{half}": e.astype(np.float32) for half, e in result["pose_errors"].items()},
        **{f"aligned_half{half}": e.astype(np.float32) for half, e in result["aligned_pose_errors"].items()},
    )
    cfscs = result["conical_fsc"]
    np.savez_compressed(
        out_dir / "conical_fsc.npz",
        axes=next(iter(cfscs.values())).axes,
        **{f"{name}_curves": cfsc.curves for name, cfsc in cfscs.items()},
        **{f"{name}_voxels": cfsc.voxels for name, cfsc in cfscs.items()},
    )
    rows = result["rows"]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(out_dir / "prior_by_iteration.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    iteration_rows = result["iteration_rows"]
    with open(out_dir / "fsc_by_iteration.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(iteration_rows[0]), restval="")
        writer.writeheader()
        writer.writerows({k: "" if v is None else v for k, v in row.items()} for row in iteration_rows)
    curves = result["curves"]
    with open(out_dir / "fsc_curves.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shell", "resolution_A", *curves])
        for shell in range(len(curves["half_maps"])):
            resolution = result["box"] * result["voxel"] / shell if shell else math.inf
            writer.writerow([shell, resolution, *(curves[name][shell] for name in curves)])


def plot_iteration_metric(
    ax,
    rows: list[dict[str, Any]],
    key: str,
    title: str,
    *,
    uniform_key: str | None = None,
    oracle_key: str | None = None,
    log: bool = False,
) -> None:
    """Draw one learned-prior metric against the iteration for both halves, with its two references.

    The references are the uniform prior (dashed) and the histogram of half 1's true directions (dotted). An
    infinite value (KL where the prior is zero but the truth is not) is a triangle on the top edge.
    """
    from matplotlib.ticker import MaxNLocator

    c = PALETTE
    ax.grid(color=c["grid"], linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    drawn = 0

    def draw(selected: list[dict[str, Any]], column: str, color: str, **style) -> np.ndarray:
        nonlocal drawn
        x = np.array([r["iteration"] for r in selected])
        y = np.array([r[column] for r in selected], dtype=float)
        ax.plot(x, np.where(np.isfinite(y), y, np.nan), color=color, **style)
        infinite = np.isposinf(y)
        if infinite.any():
            # One row of triangles per series, so a reference's markers do not hide the learned prior's.
            ax.plot(
                x[infinite],
                np.full(int(infinite.sum()), 1.0 - 0.045 * drawn),
                "^",
                color=color,
                markersize=6,
                clip_on=False,
                transform=ax.get_xaxis_transform(),
            )
            ax.margins(y=0.35)
        drawn += 1
        return y

    half1 = [r for r in rows if r["half"] == 1]
    values = [
        draw([r for r in rows if r["half"] == h], key, c[f"half{h}"], linewidth=2, marker="o", markersize=5)
        for h in HALVES
    ]
    for reference, linestyle in ((uniform_key, "--"), (oracle_key, ":")):
        if reference is not None:
            values.append(draw(half1, reference, c["reference"], linewidth=1.5, linestyle=linestyle))
    if all(np.all(np.isnan(v)) for v in values):
        ax.text(
            0.5,
            0.5,
            "undefined\n(the uniform prior is the truth)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=c["ink2"],
        )
    for previous, current in zip(half1, half1[1:]):
        if current["healpix_order"] != previous["healpix_order"]:
            ax.axvline(current["iteration"], color=c["order"], linewidth=1)
            ax.text(
                current["iteration"],
                1.045,
                f"{current['healpix_order']}",
                color=c["ink2"],
                fontsize=7,
                ha="left",
                va="bottom",
                transform=ax.get_xaxis_transform(),
            )
    positive = np.concatenate(values)
    ax.set_xlim(min(r["iteration"] for r in half1) - 0.5, max(r["iteration"] for r in half1) + 0.5)
    if log and np.any(np.isfinite(positive) & (positive > 0)):
        ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title, pad=20)


def plot_summary(result: dict[str, Any], label: str, path: Path) -> None:
    """Draw the FSC curves and every learned-prior metric against the iteration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    c = PALETTE
    plt.rcParams.update(
        {
            "figure.facecolor": c["surface"],
            "axes.facecolor": c["surface"],
            "savefig.facecolor": c["surface"],
            "text.color": c["ink"],
            "axes.labelcolor": c["ink2"],
            "xtick.color": c["ink2"],
            "ytick.color": c["ink2"],
            "axes.edgecolor": c["grid"],
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.6), gridspec_kw={"hspace": 0.42, "wspace": 0.3})

    ax = axes[0, 0]
    ax.grid(color=c["grid"], linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    curves, box, voxel = result["curves"], result["box"], result["voxel"]
    frequency = np.arange(len(curves["half_maps"])) / (box * voxel)
    ax.plot(frequency, curves["half_maps"], color=c["half1"], linewidth=2, label="half maps (gold standard)")
    ax.plot(
        frequency,
        curves["gt_merged"],
        color=c["half2"],
        linewidth=1.5,
        linestyle="--",
        label="final map vs ground truth, as refined",
    )
    ax.plot(
        frequency, curves["gt_merged_aligned"], color=c["half2"], linewidth=2, label="... aligned to the ground truth"
    )
    for level in (0.143, 0.5):
        ax.axhline(level, color=c["reference"], linewidth=1, linestyle=":")
        ax.text(frequency[-1], level + 0.015, f"{level}", color=c["ink2"], ha="right", va="bottom", fontsize=8)
    ax.set(xlabel="spatial frequency (1/A)", ylabel="FSC", ylim=(-0.05, 1.02))
    ax.set_title("Resolution (unmasked FSC)", pad=20)
    ax.legend(frameon=False, loc="lower left")

    rows = [r for r in result["rows"] if r["prior_available"]]
    panels = (
        (axes[0, 1], "js_bits", "Jensen-Shannon divergence (bits)", "js_bits_uniform", "js_bits_oracle", True),
        (axes[0, 2], "js_skill", "JS skill (1 = perfect, 0 = uniform prior)", None, "js_skill_oracle", False),
        (axes[0, 3], "kl_bits", "KL(true || learned) (bits)", "kl_bits_uniform", "kl_bits_oracle", True),
        (axes[1, 0], "tv", "Total variation", "tv_uniform", "tv_oracle", False),
        (axes[1, 1], "hellinger", "Hellinger distance", "hellinger_uniform", "hellinger_oracle", False),
        (
            axes[1, 2],
            "truth_mass_on_zero_prior",
            "True mass on zero-prior directions",
            None,
            "truth_mass_on_zero_prior_oracle",
            False,
        ),
        (
            axes[1, 3],
            "zero_prior_fraction",
            "Directions with zero prior (fraction)",
            None,
            "zero_prior_fraction_oracle",
            False,
        ),
    )
    for ax, key, title, uniform_key, oracle_key, log in panels:
        plot_iteration_metric(ax, rows, key, title, uniform_key=uniform_key, oracle_key=oracle_key, log=log)
    axes[0, 2].axhline(0, color=c["reference"], linewidth=1.5, linestyle="--")
    axes[0, 2].set_ylim(-0.05, 1.05)
    for ax in axes[1]:
        ax.set_xlabel("iteration")

    handles = [
        Line2D([], [], color=c["half1"], linewidth=2, marker="o", label="learned prior, half 1"),
        Line2D([], [], color=c["half2"], linewidth=2, marker="o", label="learned prior, half 2"),
        Line2D([], [], color=c["reference"], linewidth=1.5, linestyle="--", label="uniform prior"),
        Line2D(
            [],
            [],
            color=c["reference"],
            linewidth=1.5,
            linestyle=":",
            label="histogram of the true directions (half 1)",
        ),
        Line2D([], [], color=c["ink2"], linestyle="", marker="^", label="infinite (true mass on a zero prior)"),
        Line2D([], [], color=c["order"], linewidth=1.5, label="HEALPix order increases (number: new order)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(label, fontsize=12, x=0.02, ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-dir", type=Path, required=True, help="simulated dataset (simulate_dataset.py)")
    parser.add_argument("--run-dir", type=Path, required=True, help="relax refine output directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default=None, help="figure title (default: the run directory)")
    parser.add_argument(
        "--cone-half-angle-deg", type=float, default=HALF_ANGLE_DEG, help="half-angle of the conical FSC's cones"
    )
    parser.add_argument("--n-cone-axes", type=int, default=N_AXES, help="number of cone axes over the hemisphere")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(args.dataset_dir, args.run_dir, args.cone_half_angle_deg, args.n_cone_axes)
    write_tables(result, args.out_dir)
    label = args.label or str(args.run_dir)
    plot_summary(result, label, args.out_dir / "summary.png")
    plot_conical_fsc(
        conical_figure_pairs(result), result["voxel"], result["box"], label, PALETTE, args.out_dir / "conical_fsc.png"
    )


if __name__ == "__main__":
    main()
