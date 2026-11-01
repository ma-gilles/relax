#!/usr/bin/env python
"""Generate a multi-body dataset from RECOVAR's 5nrl spliceosome trajectory (RELION ``--multibody_masks``).

The particle is the 5nrl spliceosome split into RECOVAR's three chain groups
(``recovar.simulation.trajectory_generation.prepare_5nrl_subcomplexes``): ``Ab`` (static), ``B`` and ``Db``. Its
trajectory (``generate_conformation_2D``) rotates ``B`` by ``-10 + t_right`` and ``Db`` by ``-10 + t_top`` degrees
about the x axis through RECOVAR's hinge atom. ``--motion symmetric`` is RECOVAR's ``path_symmetric``
(``t_right = t_top``): ``B`` and ``Db`` move as one rigid unit, so the dataset has two bodies, ``Ab`` and ``B+Db``.
``--motion independent`` draws ``t_right`` and ``t_top`` independently: three bodies. Each angle takes
``--n-angles`` values evenly spaced in [0, ``--max-angle-deg``], every state (one volume of RECOVAR's simulator)
with equal weight, a discretised continuous distribution.

The domains touch (closest atoms 0.1-0.3 A apart), so a body's mask is the voxels within ``--mask-atom-radius`` of
its atoms in any state that are nearer to them than to any other body's atoms, with a raised-cosine edge of
``--mask-soft-px`` toward the solvent only and a hard cut at the interfaces: the masks are disjoint, and each holds at
least ``--min-mask-coverage`` of its body's signal energy.

Written beside ``particles.star`` / ``particles.<D>.mrcs`` / ``poses.pkl`` / ``ctf.pkl``: ``states/vol%04d.mrc``,
``bodies/vol%04d.mrc`` (each body alone at ``t = 0``), ``bodies_mean/vol%04d.mrc`` (at the mean angles, the
ground truth of the body maps), ``body<b>_mask_relion.mrc``, ``bodies.star``,
``reference_init_relion.mrc`` and ``body_motion.json`` (each state's angles, the hinge, the axis, the bodies'
chain groups).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mrcfile
import numpy as np
from recovar import utils
from recovar.output.output import mkdir_safe
from recovar.simulation import simulator, solvent_contrast
from recovar.simulation.trajectory_generation import (
    CHAIN_GROUPS_5NRL,
    generate_conformation_2D,
    prepare_5nrl_subcomplexes,
)
from scipy import ndimage

from scripts.fixture_bfactor import resolve_total_bfactor
from scripts.prepare_multibody_benchmark import real_volume
from scripts.prepare_pdb_k1_relion_sanity_benchmark import _write_references

logger = logging.getLogger(__name__)


def state_angles(motion: str, n_angles: int, max_angle_deg: float) -> np.ndarray:
    """``[n_states, 2]`` ``(t_right, t_top)`` in degrees."""
    t = np.linspace(0.0, max_angle_deg, n_angles)
    if motion == "symmetric":
        return np.stack([t, t], axis=1)
    return np.stack(np.meshgrid(t, t, indexing="ij"), axis=-1).reshape(-1, 2)


def body_groups(motion: str) -> list[list[int]]:
    """The chain groups (indices into ``[Ab, B, Db]``) of each body."""
    return [[0], [1, 2]] if motion == "symmetric" else [[0], [1], [2]]


def partitioned_masks(body_coords, *, voxel_size, grid_size, radius_angstrom, soft_px, overlap=False):
    """Disjoint soft masks: each voxel within ``radius_angstrom`` of some atom goes to the nearest body; the edge is
    soft toward the solvent and hard between bodies. With ``overlap`` each body keeps its own shell and soft edge
    wherever other bodies are: masks that enclose every body and overlap where the bodies touch."""
    distances = []
    for coords in body_coords:
        occupied = np.zeros((grid_size,) * 3, dtype=bool)
        idx = np.rint(np.asarray(coords) / voxel_size).astype(np.int64) + grid_size // 2
        idx = idx[np.all((idx >= 0) & (idx < grid_size), axis=1)]
        occupied[tuple(idx.T)] = True
        distances.append(ndimage.distance_transform_edt(~occupied) * voxel_size)
    distances = np.stack(distances)
    def soft(distance):
        inside = distance - radius_angstrom  # Angstrom beyond the atom shell (<= 0 inside)
        edge = np.clip(inside / (soft_px * voxel_size), 0.0, 1.0)
        return np.where(inside <= 0.0, 1.0, np.where(edge < 1.0, 0.5 + 0.5 * np.cos(np.pi * edge), 0.0))

    if overlap:
        return [soft(d).astype(np.float32) for d in distances]
    nearest = np.argmin(distances, axis=0)
    weight = soft(distances.min(axis=0))
    return [np.where(nearest == b, weight, 0.0).astype(np.float32) for b in range(len(body_coords))]


def write_mean_pose_body_maps(out: Path, motion: str, angles: np.ndarray, *, voxel_size: float, grid_size: int):
    """Each body alone at the dataset's mean angles (``bodies_mean/vol%04d.mrc``): the ground truth of the body maps,
    which a multi-body refinement reconstructs at the consensus (mean) pose."""
    group_coords, group_elements, hinge = prepare_5nrl_subcomplexes()
    mean_right, mean_top = np.asarray(angles, dtype=np.float64).mean(axis=0)
    posed = generate_conformation_2D(group_coords, mean_right, mean_top, hinge)
    mkdir_safe(str(out / "bodies_mean"))
    for b, members in enumerate(body_groups(motion)):
        vol = real_volume(
            np.concatenate([posed[g] for g in members]), np.concatenate([group_elements[g] for g in members]),
            voxel_size=voxel_size, grid_size=grid_size,
        )
        utils.write_mrc(str(out / "bodies_mean" / f"vol{b:04d}.mrc"), vol, voxel_size=voxel_size)


def prepare(args):
    out = Path(args.output_dir)
    mkdir_safe(str(out))
    np.random.seed(args.seed)
    voxel_size = 4.25 * 128 / args.grid_size if args.voxel_size is None else args.voxel_size
    D = args.grid_size
    group_coords, group_elements, hinge = prepare_5nrl_subcomplexes()
    angles = state_angles(args.motion, args.n_angles, args.max_angle_deg)
    groups = body_groups(args.motion)
    mkdir_safe(str(out / "states"))
    mkdir_safe(str(out / "bodies"))
    all_elements = np.concatenate(group_elements)
    body_atoms_all_states = [[] for _ in groups]
    for idx, (t_right, t_top) in enumerate(angles):
        moved = generate_conformation_2D(group_coords, t_right, t_top, hinge)
        vol = real_volume(np.concatenate(moved), all_elements, voxel_size=voxel_size, grid_size=D)
        utils.write_mrc(str(out / "states" / f"vol{idx:04d}.mrc"), vol, voxel_size=voxel_size)
        for b, members in enumerate(groups):
            body_atoms_all_states[b].append(np.concatenate([moved[g] for g in members]))
        logger.info("state %d t_right %.2f t_top %.2f", idx, t_right, t_top)
    neutral = generate_conformation_2D(group_coords, 0.0, 0.0, hinge)
    body_maps = []
    for b, members in enumerate(groups):
        vol = real_volume(
            np.concatenate([neutral[g] for g in members]), np.concatenate([group_elements[g] for g in members]),
            voxel_size=voxel_size, grid_size=D,
        )
        utils.write_mrc(str(out / "bodies" / f"vol{b:04d}.mrc"), vol, voxel_size=voxel_size)
        body_maps.append(vol)
    masks = partitioned_masks(
        [np.concatenate(c) for c in body_atoms_all_states], voxel_size=voxel_size, grid_size=D,
        radius_angstrom=args.mask_atom_radius, soft_px=args.mask_soft_px, overlap=args.overlapping_masks,
    )
    coverage = [float((v * v * m).sum() / (v * v).sum()) for v, m in zip(body_maps, masks)]
    logger.info("fraction of each body's signal energy inside its mask (t = 0): %s", coverage)
    if min(coverage) < args.min_mask_coverage:
        raise RuntimeError(f"body masks hold {coverage} of their bodies' energy, below {args.min_mask_coverage}")
    rows = []
    for b, mask in enumerate(masks, start=1):
        with mrcfile.new(str(out / f"body{b}_mask_relion.mrc"), overwrite=True) as mrc:
            mrc.set_data(np.ascontiguousarray(np.transpose(mask, (2, 1, 0)), dtype=np.float32))
            mrc.voxel_size = voxel_size
        relative_to = 2 if b == 1 else 1  # the moving bodies rotate about the static Ab; Ab about the first of them
        rows.append(f"body{b}_mask_relion.mrc {relative_to} {args.body_sigma_angles:g} {args.body_sigma_offset:g}")
    (out / "bodies.star").write_text(
        "\ndata_\n\nloop_\n_rlnBodyMaskName #1\n_rlnBodyRotateRelativeTo #2\n_rlnBodySigmaAngles #3\n"
        "_rlnBodySigmaOffsetAngst #4\n" + "\n".join(rows) + "\n"
    )
    atomic_volume_kwargs, bfactor_record = resolve_total_bfactor(
        solvent_contrast.kwargs_from_cli_args(args), baked_in=0.0, source="5nrl atoms (B 0)"
    )
    simulator.generate_synthetic_dataset(
        str(out), voxel_size, str(out / "states" / "vol"), args.n_images, grid_size=D,
        volume_distribution=np.full(len(angles), 1.0 / len(angles)), dataset_params_option="uniform",
        noise_level=args.noise_level, noise_model="white", put_extra_particles=False, percent_outliers=0.0,
        volume_radius=args.volume_radius, trailing_zero_format_in_vol_name=True, noise_scale_std=0.0,
        contrast_std=0.0, disc_type="cubic", n_tilts=-1, image_dtype=np.float32, relion_normalize=True,
        **atomic_volume_kwargs,
    )
    _write_references(out, D, args.init_resolution_ang)
    names = ["Ab", "B", "Db"]
    record = {
        "source": "scripts/prepare_spliceosome_multibody.py",
        "pdb": "recovar assets/5nrl_atoms.npz",
        "trajectory": "recovar.simulation.trajectory_generation.generate_conformation_2D: B by -10 + t_right and Db "
                      "by -10 + t_top degrees about x through the hinge (centered 5nrl coordinates)",
        "motion": args.motion,
        "bodies": [{"groups": [names[g] for g in members], "chains": sum((CHAIN_GROUPS_5NRL[names[g]] for g in members), [])}
                   for members in groups],
        "hinge_angstrom_centered": np.asarray(hinge).tolist(), "axis": [1.0, 0.0, 0.0],
        "state_angles_deg": angles.tolist(),
        "energy_fraction_inside_masks": coverage,
        "grid_size": D, "voxel_size": voxel_size, "n_images": args.n_images, "noise_level": args.noise_level,
        "seed": args.seed, "bfactor": bfactor_record, "mask_atom_radius": args.mask_atom_radius,
        "mask_soft_px": args.mask_soft_px, "overlapping_masks": args.overlapping_masks, "body_sigma_angles": args.body_sigma_angles,
        "body_sigma_offset_angstrom": args.body_sigma_offset, "init_resolution_ang": args.init_resolution_ang,
    }
    (out / "body_motion.json").write_text(json.dumps(record, indent=2))
    write_mean_pose_body_maps(out, args.motion, angles, voxel_size=voxel_size, grid_size=D)
    logger.info("Spliceosome multi-body dataset ready at %s", out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--motion", choices=("symmetric", "independent"), default="independent")
    parser.add_argument("--n-images", type=int, default=20000)
    parser.add_argument("--n-angles", type=int, default=9)
    parser.add_argument("--max-angle-deg", type=float, default=40.0)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--mask-atom-radius", type=float, default=8.0)
    parser.add_argument("--mask-soft-px", type=int, default=2)
    parser.add_argument("--min-mask-coverage", type=float, default=0.99)
    parser.add_argument("--overlapping-masks", action="store_true",
                        help="each body's own soft mask, overlapping where bodies touch (needs RELION's overlap rules)")
    parser.add_argument("--body-sigma-angles", type=float, default=10.0)
    parser.add_argument("--body-sigma-offset", type=float, default=4.0)
    parser.add_argument("--init-resolution-ang", type=float, default=30.0)
    parser.add_argument("--volume-radius", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mean-pose-maps-only", action="store_true",
                        help="only (re)write bodies_mean/ of an existing dataset from its body_motion.json")
    solvent_contrast.add_cli_arguments(parser, enabled_by_default=True)
    args = parser.parse_args()
    if args.mean_pose_maps_only:
        record = json.loads((Path(args.output_dir) / "body_motion.json").read_text())
        write_mean_pose_body_maps(Path(args.output_dir), record["motion"], np.asarray(record["state_angles_deg"]),
                                  voxel_size=record["voxel_size"], grid_size=record["grid_size"])
        return
    prepare(args)


if __name__ == "__main__":
    main()
