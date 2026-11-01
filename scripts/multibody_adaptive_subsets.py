#!/usr/bin/env python
"""Fixed-pose adaptive subset reconstruction of one body from a finished multi-body run.

    python -m scripts.multibody_adaptive_subsets --consensus <consensus run_itNNN_optimiser.star> \\
        --bodies bodies.star --multibody <MultiBody dir> --pca <flex>_projections_along_eigenvectors_all_particles.txt \\
        --components 0 --target 1.5 --target-body 2 --out <dir>

The multi-body directory is RELION's or relax's (``run_data.star`` and the last ``run_itNNN_half<h>_body<BBB>.mrc``);
``--pca`` is ``relion_flex_analyse --PCA_orient --write_pca_projections`` of that run. ``--components`` are 0-based
columns of its scores; ``--target`` the target point in them (``--target-quantile q`` takes the q-quantile of a
single component instead). ``--oracle-coordinates`` replaces the embedding by the simulated state's twist angle
(``body_motion.json``), a labelled diagnostic. See :mod:`relax.multibody.subsets`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np


def oracle_result(run, result, data_dir: Path):
    """``result`` with every body's TRUE pose (a diagnostic): the residual that composes, with the run's consensus
    pose, to the simulated particle pose times the body's true motion, and the offset that puts the body's centre
    where the simulation put it (plane-cut benchmark: body 2 twists about ``twist_axis`` through the pivot)."""
    from dataclasses import replace

    from recovar import utils
    from recovar.utils.helpers import R_to_relion
    from scipy.spatial.transform import Rotation

    from relax.multibody import geometry
    from relax.multibody.bodies import A_ROT90
    from relax.multibody.subtraction import BodyPoses

    record = json.loads((data_dir / "body_motion.json").read_text())
    sim = utils.pickle_load(data_dir / "simulation_info.pkl")
    names = np.asarray(run.checkpoint.particles["rlnImageName"])
    rows = np.asarray([int(n.split("@")[0]) - 1 for n in names])
    true_rot = np.asarray(utils.pickle_load(data_dir / "poses.pkl")[0])[rows]  # RELION matrices
    particles = run.checkpoint.particles
    consensus = geometry.consensus_matrices(particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(float))
    cons_off = particles[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(float)
    angles = np.asarray(record["state_angles_deg"], float)[np.asarray(sim["image_assignment"])[rows]]
    axis, pivot = np.asarray(record["twist_axis"]), np.asarray(record["pivot_angstrom_centered"])
    poses = []
    for b, body in enumerate(run.bodies):
        M = np.tile(np.eye(3), (len(rows), 1, 1)) if b == 0 else Rotation.from_rotvec(np.deg2rad(angles)[:, None] * axis).as_matrix()
        target = np.einsum("nij,njk->nik", true_rot, M)
        O = body.orientation
        residual = np.einsum("ij,jk,nlk,nlm,mo->nio", A_ROT90.T, O, consensus, target, O.T)
        com = body.com * run.pixel_size
        s_true = np.einsum("nij,nj->ni", true_rot, np.einsum("nij,nj->ni", M, (com - pivot)[None]) + pivot)[:, :2]
        origin = np.einsum("nij,j->ni", consensus, com)[:, :2] - cons_off - s_true
        poses.append(BodyPoses(eulers_deg=np.asarray(R_to_relion(residual, degrees=True)), offsets_px=origin / run.pixel_size))
    return replace(result, poses=poses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--consensus", required=True)
    parser.add_argument("--bodies", required=True, type=Path)
    parser.add_argument("--multibody", required=True, type=Path)
    parser.add_argument("--pca", type=Path, default=None)
    parser.add_argument("--oracle-coordinates", type=Path, default=None, help="dataset dir: use the true twist (diagnostic)")
    parser.add_argument("--components", default="0", help="comma-separated 0-based PCA columns")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target", default=None, help="comma-separated coordinates of the target point")
    target.add_argument("--target-quantile", type=float, default=None)
    parser.add_argument("--target-body", type=int, required=True, help="1-based body")
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument("--n-min-per-half", type=int, default=50)
    parser.add_argument("--metric", default="locshellmost_likely")
    parser.add_argument("--maskrad-fraction", type=float, default=20.0, help="RECOVAR's: local windows of box / fraction")
    parser.add_argument("--oracle-poses", type=Path, default=None,
                        help="dataset dir: reconstruct with the TRUE body poses (diagnostic; plane-cut benchmark)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check-direct", action="store_true",
                        help="also compare three cumulative candidates of half 1 with direct reconstructions")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from relax.commands.multibody import run_options
    from relax.multibody import subsets
    from relax.multibody.refine import MultibodyCommand, start_from_consensus

    command = MultibodyCommand(
        body_star=args.bodies, healpix_order=4, auto_local_healpix_order=4, offset_range_px=3.0, offset_step_px=1.5,
        oversampling=1, reconstruct_subtracted_bodies=True, max_iter=0,
    )
    run, _ = start_from_consensus(args.consensus, command, run_options(), output=args.out / "unused_run")
    result = subsets.read_multibody_result(args.multibody, run)
    if args.oracle_poses is not None:
        result = oracle_result(run, result, args.oracle_poses)
    names = np.asarray(run.checkpoint.particles["rlnImageName"])
    components = [int(c) for c in args.components.split(",")]
    if args.oracle_coordinates is not None:
        from recovar import utils

        record = json.loads((args.oracle_coordinates / "body_motion.json").read_text())
        assignment = np.asarray(utils.pickle_load(args.oracle_coordinates / "simulation_info.pkl")["image_assignment"])
        rows = np.asarray([int(name.split("@")[0]) - 1 for name in names])
        scores = np.asarray(record["state_angles_deg"], dtype=np.float64)[assignment[rows]][:, None]
        components, source = [0], "oracle: true body-2 twist (deg) of each particle's simulated state"
    else:
        scores = subsets.read_flex_projections(args.pca, names)
        source = str(args.pca)
    if args.target is not None:
        z_star = np.asarray([float(v) for v in args.target.split(",")])
    else:
        z_star = np.quantile(scores[:, components], args.target_quantile, axis=0)
    distances = subsets.distances_to(scores, components, z_star)
    thresholds = subsets.nested_thresholds(
        distances, n_candidates=args.n_candidates, n_min_per_half=args.n_min_per_half,
        halves=[run.halves[h].rows for h in range(2)],
    )
    record = subsets.adaptive_subsets(
        run, result, target=args.target_body - 1, distances=distances, thresholds=thresholds, output=args.out,
        metric=args.metric, maskrad_fraction=args.maskrad_fraction,
    )
    record.update({"coordinates": source, "components": components, "target_point": z_star.tolist()})
    if args.check_direct:
        dataset = subsets.fixed_pose_dataset(run, result, args.target_body - 1)
        record["direct_check_half1"] = subsets.check_cumulative_against_direct(
            dataset, distances[np.asarray(run.halves[0].rows)], thresholds
        )
    (args.out / "subset_subsets.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
