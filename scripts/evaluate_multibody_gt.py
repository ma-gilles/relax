#!/usr/bin/env python
"""Score a multi-body refinement (relax's or RELION's) against the synthetic truth of
``scripts/prepare_multibody_benchmark.py``.

    python -m scripts.evaluate_multibody_gt --data <dataset dir> --run <MultiBody dir> [--iteration N] --json out.json

Per body: the FSC of the final map (``run_body<BBB>.mrc``; with ``--iteration``, each half map) against that body's
ground-truth map (the body's atoms alone at the unmoved pose, through the simulator's own operator), unmasked and in
the body mask, with its normalized area and the 0.5 / 0.143 crossings; and per particle the error of the recovered
body motion. The motion is compared in the particle frame, ``Q = A_consensus^T A_body``, against the true rigid motion
of the simulated state (identity for body 1, the twist for body 2), with no registration: the atom frame of the
simulation is RELION's map frame. Both the angle between ``Q`` and the truth and the recovered twist angle about the
true axis are reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import starfile
from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.simulation import synthetic_dataset
from recovar.utils.helpers import load_relion_volume
from scipy.spatial.transform import Rotation

from relax.multibody.bodies import read_bodies
from relax.multibody.geometry import body_matrices, consensus_matrices
from scripts.fsc_metrics import first_shell_below, normalized_fsc_auc, shell_fsc


def _gt_body_maps(data_dir: Path) -> list[np.ndarray]:
    sim_info = utils.pickle_load(data_dir / "simulation_info.pkl")
    # The spliceosome datasets' body maps sit at the mean pose (prepare_spliceosome_multibody.py).
    folder = "bodies_mean" if (data_dir / "bodies_mean").is_dir() else "bodies"
    volumes = synthetic_dataset.load_ground_truth_volumes(sim_info, str(data_dir / folder / "vol"))
    shape = (int(round(volumes.shape[-1] ** (1 / 3))),) * 3
    return [np.asarray(ftu.get_idft3(jnp.asarray(v).reshape(shape)).real) for v in volumes]


def _fsc_summary(fsc: np.ndarray, *, box: int, pixel: float) -> dict:
    out = {"auc": float(normalized_fsc_auc(fsc)), "fsc": fsc.tolist()}
    for threshold in (0.5, 0.143):
        shell = first_shell_below(fsc[1:], threshold)
        out[f"resolution_{threshold}"] = None if shell is None else float(box * pixel / (shell + 1))
    return out


def _true_motions(data_dir: Path, n_particles: int, image_rows: np.ndarray, n_bodies: int):
    """Each body's true rotation per particle, its true angle about the motion axis (None for a static body) and
    the axis. The plane-cut benchmark twists body 2 about ``twist_axis``; the spliceosome datasets rotate the moving
    bodies about x through the hinge, measured from the mean angles (the consensus pose)."""
    record = json.loads((data_dir / "body_motion.json").read_text())
    sim_info = utils.pickle_load(data_dir / "simulation_info.pkl")
    states = np.asarray(sim_info["image_assignment"])[image_rows]
    angles = np.asarray(record["state_angles_deg"], dtype=np.float64)[states]
    if "twist_axis" in record:
        axis = np.asarray(record["twist_axis"])
        per_body = [None, angles]
    else:
        axis = np.asarray(record["axis"], dtype=np.float64)
        moving = angles - angles.mean(axis=0)
        per_body = [None, moving[:, 0]] if n_bodies == 2 else [None, moving[:, 0], moving[:, 1]]
    truths = [
        np.tile(np.eye(3), (n_particles, 1, 1)) if a is None
        else Rotation.from_rotvec(np.deg2rad(a)[:, None] * axis[None, :]).as_matrix()
        for a in per_body
    ]
    return truths, per_body, axis


def evaluate(data_dir: Path, run_dir: Path, iteration: int | None) -> dict:
    stem = run_dir / ("run" if iteration is None else f"run_it{iteration:03d}")
    data = starfile.read(f"{stem}_data.star", always_dict=True)
    particles = data["particles"]
    pixel = float(data["optics"]["rlnImagePixelSize"].iloc[0])
    bodies = read_bodies(data_dir / "bodies.star", pixel_size=pixel)
    image_rows = np.asarray([int(name.split("@")[0]) - 1 for name in particles["rlnImageName"]])
    consensus = consensus_matrices(particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(float))
    truths, true_angles, axis = _true_motions(data_dir, len(particles), image_rows, len(bodies))
    gt = _gt_body_maps(data_dir)
    box = gt[0].shape[0]
    report = {"run": str(stem), "bodies": []}
    for b, body in enumerate(bodies):
        table = data[f"images_body_{b + 1}"]
        residual = table[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(float)
        relative = np.einsum("nji,njk->nik", consensus, body_matrices(consensus, body, residual))
        error = Rotation.from_matrix(np.einsum("nji,njk->nik", truths[b], relative)).magnitude()
        recovered = Rotation.from_matrix(relative).as_rotvec() @ axis
        mask = np.transpose(body.mask, (2, 1, 0))  # the recovar frame of the RELION-frame mask
        maps = []
        names = ["_body"] if iteration is None else ["_half1_body", "_half2_body"]
        for name in names:
            volume = load_relion_volume(f"{stem}{name}{b + 1:03d}.mrc")
            maps.append({
                "map": f"{stem}{name}{b + 1:03d}.mrc",
                "unmasked": _fsc_summary(shell_fsc(volume, gt[b]), box=box, pixel=pixel),
                "masked": _fsc_summary(shell_fsc(volume * mask, gt[b] * mask), box=box, pixel=pixel),
            })
        report["bodies"].append({
            "body": b + 1,
            "motion_error_deg": {
                "median": float(np.rad2deg(np.median(error))), "mean": float(np.rad2deg(np.mean(error))),
                "p90": float(np.rad2deg(np.percentile(error, 90))),
            },
            "recovered_twist_vs_true_corr": float(np.corrcoef(np.rad2deg(recovered), true_angles[b])[0, 1])
            if true_angles[b] is not None and np.std(true_angles[b]) > 0 else None,
            "recovered_twist_slope": float(np.polyfit(true_angles[b], np.rad2deg(recovered), 1)[0])
            if true_angles[b] is not None and np.std(true_angles[b]) > 0 else None,
            "offset_rms_angstrom": float(np.sqrt(np.mean(table[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(float) ** 2))),
            # Search-boundary occupancy: residual rotations beyond 90% of the body prior's 3-sigma cut.
            "fraction_near_prior_boundary": float(np.mean(
                np.rad2deg(Rotation.from_matrix(relative).magnitude()) > 0.9 * 3.0 * body.sigma_tilt_deg
            )) if body.sigma_tilt_deg > 0 else None,
            "maps": maps,
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(args.data, args.run, args.iteration)
    for body in report["bodies"]:
        maps = " ".join(
            f"[{Path(m['map']).name}: auc {m['unmasked']['auc']:.4f} masked {m['masked']['auc']:.4f} "
            f"res0.143 {m['masked']['resolution_0.143']}]" for m in body["maps"]
        )
        print(
            f"body {body['body']}: motion error median {body['motion_error_deg']['median']:.3f} deg, "
            f"twist corr {body['recovered_twist_vs_true_corr']}, slope {body['recovered_twist_slope']}; {maps}"
        )
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
