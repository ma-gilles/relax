#!/usr/bin/env python
"""Two-body fixtures for the pose-neighborhood study, built on ``prepare_multibody_benchmark``'s plane-cut 5nrl bodies.

    python -m scripts.prepare_pose_neighborhood_benchmark --output-dir <dir> --case B --n-images 5000 --noise-level 0.1

Every case moves body 1 by its own independent twist (nuisance motion). Body 2:

- ``A``: twists, internally rigid;
- ``B``: twists, and a lobe hinges in proportion (deformation correlated with its rotation);
- ``C``: translates, no rotation, and the lobe hinges in proportion (correlated with its translation);
- ``D``: twists, and the lobe hinges independently of every rigid motion.

Written beside the usual simulator output: ``states/`` (both bodies), ``component_body1/`` and ``component_body2/``
(each body alone, moved, per state, for exact nuisance predictions), ``body2_states/`` (body 2 alone at each state's
deformation, unmoved: the reconstruction truth), ``bodies/`` (each body neutral), masks and ``bodies.star``.
``body_motion.json`` has each state's motion of each body about the pivot, and ``depth_bounds_angstrom``: the bound
``1.25 max(T_b, pixel)`` from the largest native-pivot displacement ``T_b``, fixed here before any result exists.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import mrcfile
import numpy as np
from recovar import utils
from recovar.output.output import mkdir_safe
from recovar.simulation import simulator, solvent_contrast

from relax.multibody.bodies import read_bodies
from scripts.fixture_bfactor import resolve_total_bfactor
from scripts.prepare_multibody_benchmark import (
    _write_references,
    atom_binary,
    body2_lobe,
    body2_rotation,
    body_atoms,
    deform_body2,
    real_volume,
    rigid_motion,
    soft_mask,
)

logger = logging.getLogger(__name__)
LEVELS = np.linspace(-1.0, 1.0, 9)
NUISANCE_LEVELS = np.array([-1.0, 0.0, 1.0])
DEFORMATION_LEVELS = np.array([-1.0, 0.0, 1.0])


def state_table(case: str, *, max_twist: float, max_shift: float, max_nuisance: float, max_deformation: float) -> list[dict]:
    """One row per state: body 1's twist, body 2's twist and shift (degrees, angstroms) and the lobe's hinge."""
    states = []
    if case == "D":
        grid = itertools.product(LEVELS, DEFORMATION_LEVELS, NUISANCE_LEVELS)
    else:
        grid = ((level, level, nuisance) for level, nuisance in itertools.product(LEVELS, NUISANCE_LEVELS))
    for level, deformation, nuisance in grid:
        states.append({
            "body1_twist": float(nuisance * max_nuisance),
            "body2_twist": 0.0 if case == "C" else float(level * max_twist),
            "body2_shift": float(level * max_shift) if case == "C" else 0.0,
            "deformation": 0.0 if case == "A" else float(deformation * max_deformation),
            "level": float(level),
        })
    return states


def shift_axis(axis: np.ndarray) -> np.ndarray:
    """A unit direction normal to the twist axis, along which case C's body 2 translates."""
    other = np.cross(axis, [1.0, 0.0, 0.0])
    return other / np.linalg.norm(other)


def write_volume(path: Path, coords, elements, voxel_size: float, grid_size: int) -> None:
    utils.write_mrc(str(path), real_volume(coords, elements, voxel_size=voxel_size, grid_size=grid_size), voxel_size=voxel_size)


def prepare(args) -> None:
    out = Path(args.output_dir)
    mkdir_safe(str(out))
    np.random.seed(args.seed)
    D, voxel = args.grid_size, 4.25 * 128 / args.grid_size
    (b1_xyz, b1_el), (b2_xyz, b2_el), axis, pivot, _ = body_atoms(args.cut_angstrom, args.gap_angstrom)
    lobe, base, hinge_axis = body2_lobe(b2_xyz, axis, pivot, args.lobe_radius)
    direction = shift_axis(axis)
    states = state_table(args.case, max_twist=args.max_twist, max_shift=args.max_shift, max_nuisance=args.max_nuisance,
                         max_deformation=args.max_deformation)
    for folder in ("states", "bodies", "component_body1", "component_body2", "body2_states"):
        mkdir_safe(str(out / folder))
    binary = dict(voxel_size=voxel, grid_size=D, radius_angstrom=args.mask_atom_radius)
    union = [np.zeros((D,) * 3, bool), np.zeros((D,) * 3, bool)]
    rotations, translations = [], []
    for idx, s in enumerate(states):
        rot1, rot2 = body2_rotation(s["body1_twist"], axis), body2_rotation(s["body2_twist"], axis)
        shift = s["body2_shift"] * direction
        deformed = deform_body2(b2_xyz, lobe, base, hinge_axis, s["deformation"])
        moved1, moved2 = rigid_motion(b1_xyz, pivot, rot1), rigid_motion(deformed, pivot, rot2) + shift
        name = f"vol{idx:04d}.mrc"
        write_volume(out / "states" / name, np.concatenate([moved1, moved2]), np.concatenate([b1_el, b2_el]), voxel, D)
        write_volume(out / "component_body1" / name, moved1, b1_el, voxel, D)
        write_volume(out / "component_body2" / name, moved2, b2_el, voxel, D)
        write_volume(out / "body2_states" / name, deformed, b2_el, voxel, D)
        union[0] |= atom_binary(moved1, **binary)
        union[1] |= atom_binary(moved2, **binary)
        rotations.append([rot1.as_matrix(), rot2.as_matrix()])
        translations.append([np.zeros(3), shift])
        logger.info("state %d of %d: %s", idx + 1, len(states), s)
    body1, body2 = (real_volume(c, e, voxel_size=voxel, grid_size=D) for c, e in ((b1_xyz, b1_el), (b2_xyz, b2_el)))
    utils.write_mrc(str(out / "bodies" / "vol0000.mrc"), body1, voxel_size=voxel)
    utils.write_mrc(str(out / "bodies" / "vol0001.mrc"), body2, voxel_size=voxel)
    masks = [soft_mask(u, extend_px=args.mask_extend_px, soft_px=args.mask_soft_px) for u in union]
    if float(np.max(masks[0] * masks[1])) > 0:
        raise RuntimeError("the union masks of the two bodies overlap over this motion")
    star = []
    for b, mask in enumerate(masks, start=1):
        with mrcfile.new(str(out / f"body{b}_mask_relion.mrc"), overwrite=True) as mrc:
            mrc.set_data(np.ascontiguousarray(np.transpose(mask, (2, 1, 0)), dtype=np.float32))
            mrc.voxel_size = voxel
        star.append(f"body{b}_mask_relion.mrc {3 - b} {args.body_sigma_angles:g} {args.body_sigma_offset:g}")
    (out / "bodies.star").write_text(
        "\ndata_\n\nloop_\n_rlnBodyMaskName #1\n_rlnBodyRotateRelativeTo #2\n_rlnBodySigmaAngles #3\n"
        "_rlnBodySigmaOffsetAngst #4\n" + "\n".join(star) + "\n"
    )
    bodies = read_bodies(out / "bodies.star", pixel_size=voxel)
    rotations, translations = np.asarray(rotations), np.asarray(translations)
    displacement = []
    for b, body in enumerate(bodies):
        centre = body.com * voxel - pivot
        moved = np.einsum("sij,j->si", rotations[:, b], centre) + pivot + translations[:, b]
        displacement.append(float(np.max(np.linalg.norm(moved - body.com * voxel, axis=1))))
    atomic, bfactor = resolve_total_bfactor(solvent_contrast.kwargs_from_cli_args(args), baked_in=0.0, source="5nrl body atoms (B 0)")
    simulator.generate_synthetic_dataset(
        str(out), voxel, str(out / "states" / "vol"), args.n_images, grid_size=D,
        volume_distribution=np.full(len(states), 1.0 / len(states)), dataset_params_option="uniform",
        noise_level=args.noise_level, noise_model="white", put_extra_particles=False, percent_outliers=0.0,
        volume_radius=args.volume_radius, trailing_zero_format_in_vol_name=True, noise_scale_std=0.0,
        contrast_std=0.0, disc_type="cubic", n_tilts=-1, image_dtype=np.float32, relion_normalize=True, **atomic,
    )
    _write_references(out, D, args.init_resolution_ang)
    record = {
        "source": "scripts/prepare_pose_neighborhood_benchmark.py", "case": args.case, "states": states,
        "state_body_rotations": rotations.tolist(), "state_body_translations": translations.tolist(),
        "pivot_angstrom_centered": pivot.tolist(), "twist_axis": axis.tolist(), "shift_axis": direction.tolist(),
        "max_pivot_displacement_angstrom": displacement,
        "depth_bounds_angstrom": [1.25 * max(t, voxel) for t in displacement],
        "lobe_atoms": int(lobe.sum()), "voxel_size": voxel, "grid_size": D, "n_images": args.n_images,
        "noise_level": args.noise_level, "seed": args.seed, "bfactor": bfactor,
        "max_twist_deg": args.max_twist, "max_shift_angstrom": args.max_shift, "max_nuisance_twist_deg": args.max_nuisance,
        "max_deformation_deg": args.max_deformation, "mask_atom_radius": args.mask_atom_radius,
        "mask_soft_px": args.mask_soft_px, "body_sigma_angles": args.body_sigma_angles,
        "body_sigma_offset_angstrom": args.body_sigma_offset, "volume_radius": args.volume_radius,
    }
    (out / "body_motion.json").write_text(json.dumps(record, indent=1))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--max-twist", type=float, default=15.0)
    parser.add_argument("--max-shift", type=float, default=8.0)
    parser.add_argument("--max-nuisance", type=float, default=6.0)
    parser.add_argument("--max-deformation", type=float, default=15.0)
    parser.add_argument("--cut-angstrom", type=float, default=40.0)
    parser.add_argument("--gap-angstrom", type=float, default=34.0)
    parser.add_argument("--lobe-radius", type=float, default=35.0)
    parser.add_argument("--mask-atom-radius", type=float, default=8.0)
    parser.add_argument("--mask-extend-px", type=int, default=0)
    parser.add_argument("--mask-soft-px", type=int, default=2)
    parser.add_argument("--body-sigma-angles", type=float, default=10.0)
    parser.add_argument("--body-sigma-offset", type=float, default=4.0)
    parser.add_argument("--init-resolution-ang", type=float, default=30.0)
    parser.add_argument("--volume-radius", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=17)
    solvent_contrast.add_cli_arguments(parser, enabled_by_default=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
