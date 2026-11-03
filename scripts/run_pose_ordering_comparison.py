#!/usr/bin/env python
"""How each neighborhood ordering behaves on a fixture's fitted and true poses. No reconstruction.

    python -m scripts.run_pose_ordering_comparison --root data/pose_neighborhoods --out <dir> <dataset>

For the declared targets, every ordering's same-state fraction among the kept particles, the perfect ordering's
(all of the state first; a state's particles tie in truth), and the overlap between orderings. The 3D density fit is
the translation-only EM of a density on a grid, each particle saying its translation lies on its line; a particle's
position is its posterior mean. Writes ``orderings_<dataset>.json`` and the arrays the motion plot needs.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

STEP, MIN_WIDTH, KEPT = 1.5, 0.75, (50, 100, 300, 1000)
EM_ITERATIONS = 25
DEPTH_NEIGHBOURS, DEPTH_ROUNDS = (10, 30, 100), 10
TARGETS = {"caseA_5k": ((13, 1002), (1, 1)), "caseB_5k": ((13, 1002), (1, 1)), "caseC_5k": ((13, 1002), (1, 1)),
           "caseD_5k": ((40, 1002), (37, 1000))}
TARGETS_10K = ((13, 1023), (1, 1037))


def lines_point(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    projection = np.eye(3) - normals[:, :, None] * normals[:, None, :]
    return np.linalg.lstsq(projection.sum(axis=0), np.einsum("nij,nj->i", projection, points), rcond=1e-6)[0]


def density_positions(points: np.ndarray, normals: np.ndarray, width: float) -> tuple[np.ndarray, dict]:
    """Each particle's posterior mean 3D position under the fitted density, angstroms."""
    centre = lines_point(points, normals)
    relative = points - centre
    in_plane = relative - np.sum(relative * normals, axis=1, keepdims=True) * normals
    # The grid holds the bulk: at high noise a few misaligned particles lie far off and would make it enormous.
    radius = float(np.quantile(np.linalg.norm(in_plane, axis=1), 0.95)) + 3.0 * width + STEP
    axis = np.arange(-radius, radius + STEP, STEP, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[np.sum(grid**2, axis=1) <= radius**2]
    likelihood = np.empty((len(points), len(grid)), dtype=np.float32)
    single, directions = relative.astype(np.float32), normals.astype(np.float32)
    for start in range(0, len(points), 500):
        difference = grid[None] - single[start : start + 500, None]
        squared = np.sum(difference**2, axis=2) - np.sum(difference * directions[start : start + 500, None], axis=2) ** 2
        likelihood[start : start + 500] = np.exp(-squared / (2.0 * width**2))
    density = np.full(len(grid), 1.0 / len(grid), dtype=np.float32)
    for _ in range(EM_ITERATIONS):
        posterior = likelihood * density
        posterior /= posterior.sum(axis=1, keepdims=True) + 1e-30
        density = posterior.mean(axis=0)
    posterior = likelihood * density
    posterior /= posterior.sum(axis=1, keepdims=True) + 1e-30
    positions = np.asarray(posterior @ grid, dtype=np.float64)
    # A line that misses the grid has no posterior: it stays at its nearest point to the centre, so it stays far away.
    outside = posterior.sum(axis=1) < 0.5
    positions[outside] = in_plane[outside]
    return positions + centre, {"grid_points": len(grid), "radius_angstrom": radius, "width_angstrom": width, "lines_off_grid": int(outside.sum())}


def neighbour_depth_positions(features, covariance: np.ndarray, neighbours: int, rounds: int = DEPTH_ROUNDS) -> np.ndarray:
    """Every particle starts at depth 0 on its line. Each round it moves to the depth nearest the mean position of its
    K nearest particles in position and rotation, the rotation vector scaled to angstroms."""
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    relative = features.shape_factors @ np.linalg.inv(features.shape_factors[0])
    rotation = Rotation.from_matrix(relative).as_rotvec() * np.sqrt(np.trace(covariance))
    normals = features.normals
    origin = features.points - np.mean(features.points, axis=0)
    origin = origin - np.sum(origin * normals, axis=1, keepdims=True) * normals
    positions = origin.copy()
    for _ in range(rounds):
        coordinates = np.hstack([positions, rotation])
        nearest = cKDTree(coordinates).query(coordinates, k=neighbours + 1, workers=-1)[1][:, 1:]
        mean = positions[nearest].mean(axis=1)
        positions = origin + np.sum((mean - origin) * normals, axis=1, keepdims=True) * normals
    return positions


def rotation_tied_shift(features, motion) -> np.ndarray:
    """The rank 1 motion's translation as its rotation part predicts it, ``[N, 3]``."""
    shape = motion.components[:9, 0]
    score = (features.shape_factors.reshape(len(features.points), -1) - motion.mean[:9]) @ shape / (shape @ shape)
    return np.outer(score, motion.components[9:, 0])


def orderings(features, pc1_scores, bound: float, target: int, densities: dict) -> dict:
    from relax.multibody.pose_neighborhoods import compatibility_distance, low_rank_distance, rotation_distance

    out = {
        "bounded": compatibility_distance(features["features"], target, bound),
        "rotation only": rotation_distance(features["features"], target),
        "rank 1": low_rank_distance(features["motions"][0], target),
        "rank 2": low_rank_distance(features["motions"][1], target),
    }
    if pc1_scores is not None:
        out["RELION PC1"] = np.abs(pc1_scores - pc1_scores[target])
    for name, positions in densities.items():
        out[name] = np.linalg.norm(positions - positions[target], axis=1)
    return out


def score_target(scores: dict, labels: np.ndarray, target: int, kept: tuple) -> dict:
    same = labels == labels[target]
    table = {}
    for name, s in scores.items():
        hits = np.cumsum(same[np.argsort(s, kind="stable")])
        table[name] = [float(hits[n - 1] / n) for n in kept]
    table["perfect"] = [min(1.0, float(same.sum()) / n) for n in kept]
    return table


def curve_sizes(n: int) -> np.ndarray:
    return np.unique(np.geomspace(20, n, 30).round().astype(np.int64))


def overlap(scores: dict, kept: int = 300) -> dict:
    sets = {name: set(np.argsort(s, kind="stable")[:kept].tolist()) for name, s in scores.items()}
    return {f"{a} | {b}": round(len(sets[a] & sets[b]) / len(sets[a] | sets[b]), 3) for a in sets for b in sets if a < b}


def true_camera_features(run, data: Path, record: dict, body: int, states: np.ndarray, names: np.ndarray):
    """The body's simulated motion seen through the simulated cameras, so no consensus error enters."""
    from recovar import utils

    from relax.multibody.pose_neighborhoods import mask_moments, pose_features
    from scripts.multibody_adaptive_subsets import true_body_motions

    rows = np.asarray([int(n.split("@")[0]) - 1 for n in names])
    cameras = np.asarray(utils.pickle_load(data / "poses.pkl")[0])[rows]
    rotations, translations, pivot = true_body_motions(record, len(run.bodies))
    motion = rotations[states, body]
    centre = run.bodies[body].com * run.pixel_size
    moved = np.einsum("nij,j->ni", motion, centre - pivot) + pivot + translations[states, body] - centre
    normals = np.swapaxes(cameras, 1, 2)[:, :, 2]
    lifts = moved - np.sum(moved * normals, axis=1, keepdims=True) * normals
    return pose_features(motion, lifts, normals, mask_moments(run.bodies[body].mask, run.bodies[body].com, run.pixel_size))


def body_block(features, covariance, relax_motion) -> dict:
    fit, summary = relax_motion["fit"], relax_motion["summary"]
    motions = [fit(features, 1), fit(features, 2)]
    noise = summary(features, motions[0], covariance)
    width = max(noise["translation_noise_angstrom_per_direction"], MIN_WIDTH)
    density, info = density_positions(features.points, features.normals, width)
    tied = density_positions(features.points - rotation_tied_shift(features, motions[0]), features.normals, width)[0]
    positions = {"density": density, "density without tied shift": tied}
    for k in DEPTH_NEIGHBOURS:
        positions[f"neighbour depth K{k}"] = neighbour_depth_positions(features, covariance, k)
    return dict(features=features, motions=motions, noise=noise, info=info, densities=positions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("dataset")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from relax.commands.multibody import run_options
    from relax.multibody import subsets
    from relax.multibody.pose_neighborhoods import fit_low_rank_motion, mask_moments, motion_summary
    from relax.multibody.refine import MultibodyCommand, start_from_consensus
    from scripts.multibody_adaptive_subsets import oracle_result, state_rows

    fit = dict(fit=fit_low_rank_motion, summary=motion_summary)
    data = args.root / args.dataset
    record = json.loads((data / "body_motion.json").read_text())
    consensus = sorted(glob.glob(str(data / "relion_consensus/run_it*_optimiser.star")))[-1]
    command = MultibodyCommand(body_star=data / "bodies.star", healpix_order=4, auto_local_healpix_order=4, offset_range_px=3.0,
                               offset_step_px=1.5, oversampling=1, reconstruct_subtracted_bodies=True, max_iter=0)
    run, _ = start_from_consensus(consensus, command, run_options(), output=args.out / f"unused_{args.dataset}")
    result = subsets.read_multibody_result(data / "relion_multibody", run)
    truth = oracle_result(run, result, data)
    names = np.asarray(run.checkpoint.particles["rlnImageName"])
    states = state_rows(data, names)
    table = record["states"]
    body2 = np.asarray([(s["body2_twist"], s["body2_shift"], s["deformation"]) for s in table])
    level = np.asarray([s["level"] for s in table])[states]
    labels = np.unique(body2, axis=0, return_inverse=True)[1].ravel()[states]
    body1 = np.unique(np.asarray([s["body1_twist"] for s in table]), return_inverse=True)[1].ravel()[states]
    pc1 = subsets.read_flex_projections(data / "relion_flex/analyse_projections_along_eigenvectors_all_particles.txt", names)[:, 0]
    declared = TARGETS.get(args.dataset, TARGETS_10K)
    report = dict(dataset=args.dataset, kept=list(KEPT), n=len(names), bodies={})
    plots = {}
    jobs = [(1, declared)]
    if args.dataset == "caseB_5k":
        jobs.append((0, (declared[0],)))
    for body, targets in jobs:
        covariance = mask_moments(run.bodies[body].mask, run.bodies[body].com, run.pixel_size).covariance
        for source, features in (("fitted", subsets.body_pose_features(run, result, body)),
                                 ("true body pose, fitted consensus", subsets.body_pose_features(run, truth, body)),
                                 ("true body pose, true camera", true_camera_features(run, data, record, body, states, names))):
            block = body_block(features, covariance, fit)
            entry = dict(noise=block["noise"], density=block["info"], targets={})
            for state, image in targets:
                row = subsets.target_row(names, f"{image}@particles.128.mrcs")
                pc = pc1 if source == "fitted" else None
                size = int(np.sum(labels == labels[row]))
                kept = tuple(sorted({*KEPT, size}))
                scores = orderings(block, pc, float(record["depth_bounds_angstrom"][body]), row, block["densities"])
                nearby = np.flatnonzero(states == states[row])[:20]
                sizes = curve_sizes(len(names))
                repeats, curves = [], []
                for r in nearby:
                    others = orderings(block, pc, float(record["depth_bounds_angstrom"][body]), int(r), block["densities"])
                    repeats.append(score_target(others, labels, int(r), kept))
                    curves.append(score_target(others, labels, int(r), tuple(sizes.tolist())))
                target_entry = dict(
                    state=state, image=image, state_size=size, kept=list(kept),
                    same_state=score_target(scores, labels, row, kept),
                    same_state_mean_first_20={k: np.mean([r[k] for r in repeats], axis=0).round(3).tolist() for k in repeats[0]},
                    overlap_at_300=overlap(scores),
                    curve_sizes=sizes.tolist(),
                    curves_mean_first_20={k: np.mean([c[k] for c in curves], axis=0).round(3).tolist() for k in curves[0]},
                )
                if body == 0:
                    target_entry["same_body1_state"] = score_target(scores, body1, row, kept)
                entry["targets"][str(state)] = target_entry
            report["bodies"].setdefault(str(body + 1), {})[source] = entry
            if body == 1:
                from scipy.spatial.transform import Rotation

                relative = block["features"].shape_factors @ np.linalg.inv(block["features"].shape_factors[0])
                source_key = source.replace(" ", "_").replace(",", "")
                plots[f"{source_key}_rotation_vectors_deg"] = Rotation.from_matrix(relative).as_rotvec(degrees=True)
                plots[f"{source_key}_rank1_scores"] = block["motions"][0].scores[:, 0]
                plots[f"{source_key}_density_positions"] = block["densities"]["density"]
                plots[f"{source_key}_points"], plots[f"{source_key}_normals"] = block["features"].points, block["features"].normals
        plots["level"], plots["body1_twist"], plots["pc1"] = level, np.asarray([s["body1_twist"] for s in table])[states], pc1
    (args.out / f"orderings_{args.dataset}.json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(args.out / f"orderings_{args.dataset}_arrays.npz", **plots)


if __name__ == "__main__":
    main()
