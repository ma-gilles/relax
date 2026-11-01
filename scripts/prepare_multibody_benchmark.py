#!/usr/bin/env python
"""Generate a two-body synthetic dataset for multi-body refinement (RELION ``--multibody_masks``).

The particle is PDB 5nrl (``recovar.simulation.trajectory_generation.prepare_5nrl_subcomplexes``, centered),
cut by a plane normal to its principal axis ``u`` at ``--cut-angstrom`` from the centroid: body 1 is the
atoms below ``cut - gap / 2``, body 2 the atoms above ``cut + gap / 2``; the slab between is removed, so the
two soft body masks (voxels within ``--mask-atom-radius`` of an atom, then a raised-cosine edge) do not
overlap. Body 2 twists rigidly about the axis ``u`` through the pivot ``centroid + cut u`` by one of
``--n-states`` angles evenly spaced in [-max, +max] degrees; every body-2 atom moves normal to ``u``, so it
stays on its side of the cut. ``--max-angle-deg 0`` gives the no-motion control.

``--deformation`` adds a small internal change of body 2 (adaptive-subset experiments): a lobe, the body-2 atoms
within ``--lobe-radius`` of its atom farthest from the cut, hinges about an axis normal to ``u`` through the lobe's
base by ``delta`` degrees. ``correlated`` ties ``delta`` to the twist (``delta = max_def * angle / max``);
``independent`` crosses every twist with ``--n-deformation-states`` levels in [-max_def, +max_def], so the
deformation carries no information about the twist. Each state is one volume of
recovar's heterogeneous simulator, so ``simulation_info.pkl``'s ``image_assignment`` is each particle's state
and ``body_motion.json`` gives every state's true body-2 rotation and pivot.

Written beside the usual ``particles.star`` / ``particles.<D>.mrcs`` / ``poses.pkl`` / ``ctf.pkl``:

- ``states/vol%04d.mrc``: the projected state volumes (recovar frame, before the solvent/B preset);
- ``bodies/vol0000.mrc``, ``bodies/vol0001.mrc``: body 1 and body 2 alone at the neutral pose, for ground
  truth through ``synthetic_dataset.load_ground_truth_volumes(sim_info, <dir>/bodies/vol)``;
- with ``--deformation``, ``body2_states/vol%04d.mrc``: body 2 alone, untwisted, at each state's deformation
  (``body_motion.json``'s ``state_deformation_deg``);
- ``body1_mask_relion.mrc``, ``body2_mask_relion.mrc`` (RELION frame) and ``bodies.star``;
- ``reference_init_relion.mrc`` / ``reference_gt_relion.mrc`` for the consensus refinement.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import jax.numpy as jnp
import mrcfile
import numpy as np
from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.output.output import mkdir_safe
from recovar.simulation import simulate_scattering_potential as ssp
from recovar.simulation import simulator, solvent_contrast
from recovar.simulation.pdb_utils import AtomGroup
from recovar.simulation.trajectory_generation import prepare_5nrl_subcomplexes, rigid_motion
from scipy import ndimage
from scipy.spatial.transform import Rotation

from scripts.fixture_bfactor import resolve_total_bfactor
from scripts.prepare_pdb_k1_relion_sanity_benchmark import _write_references

logger = logging.getLogger(__name__)


def body_atoms(cut_angstrom: float, gap_angstrom: float):
    """Body 1 and body 2 coordinates and elements, the twist axis ``u`` and the pivot (centered 5nrl frame)."""
    coords, elements, _ = prepare_5nrl_subcomplexes()
    xyz, el = np.concatenate(coords), np.concatenate(elements)
    centroid = xyz.mean(axis=0)
    u = np.linalg.svd(xyz - centroid, full_matrices=False)[2][0]
    s = (xyz - centroid) @ u
    below, above = s < cut_angstrom - gap_angstrom / 2, s > cut_angstrom + gap_angstrom / 2
    pivot = centroid + cut_angstrom * u
    return (xyz[below], el[below]), (xyz[above], el[above]), u, pivot, int((~below & ~above).sum())


def body2_rotation(angle_deg: float, axis) -> Rotation:
    return Rotation.from_rotvec(np.deg2rad(angle_deg) * np.asarray(axis, dtype=np.float64))


def body2_lobe(b2_xyz, axis, pivot, radius_angstrom: float):
    """The lobe of body 2 that deforms: its atoms within ``radius_angstrom`` of the body-2 atom farthest from the
    cut, and the hinge (a point at the lobe's base and an axis normal to ``u``)."""
    s = (b2_xyz - pivot) @ axis
    tip = b2_xyz[np.argmax(s)]
    lobe = np.linalg.norm(b2_xyz - tip, axis=1) <= radius_angstrom
    base = tip - radius_angstrom * axis
    hinge_axis = np.cross(axis, tip - pivot - ((tip - pivot) @ axis) * axis)
    if np.linalg.norm(hinge_axis) < 1e-6:
        hinge_axis = np.cross(axis, [1.0, 0.0, 0.0])
    return lobe, base, hinge_axis / np.linalg.norm(hinge_axis)


def deform_body2(b2_xyz, lobe, base, hinge_axis, delta_deg: float):
    out = np.array(b2_xyz, copy=True)
    out[lobe] = rigid_motion(b2_xyz[lobe], base, body2_rotation(delta_deg, hinge_axis))
    return out


def state_table(args):
    """Each state's (twist, deformation) in degrees."""
    angles = (
        np.zeros(1) if args.max_angle_deg == 0 else np.linspace(-args.max_angle_deg, args.max_angle_deg, args.n_states)
    )
    if args.deformation == "none":
        return angles, np.zeros_like(angles)
    if args.deformation == "correlated":
        scale = args.max_deformation_deg / args.max_angle_deg if args.max_angle_deg else 0.0
        return angles, scale * angles
    levels = np.linspace(-args.max_deformation_deg, args.max_deformation_deg, args.n_deformation_states)
    return np.repeat(angles, levels.size), np.tile(levels, angles.size)


def atom_binary(coords, *, voxel_size, grid_size, radius_angstrom):
    """Voxels within ``radius_angstrom`` of an atom, on the recovar volume grid (origin at index D // 2)."""
    occupied = np.zeros((grid_size,) * 3, dtype=bool)
    # An atom at (x, y, z) lands on volume index (x, y, z) / voxel_size + D // 2 (generate_volume_from_atoms).
    idx = np.rint(np.asarray(coords) / voxel_size).astype(np.int64) + grid_size // 2
    idx = idx[np.all((idx >= 0) & (idx < grid_size), axis=1)]
    occupied[tuple(idx.T)] = True
    return ndimage.distance_transform_edt(~occupied) * voxel_size <= radius_angstrom


def real_volume(coords, elements, *, voxel_size, grid_size):
    ag = AtomGroup()
    ag.setCoords(np.asarray(coords, dtype=np.float64))
    ag.setElements(np.asarray(elements))
    ft = ssp.generate_molecule_spectrum_from_pdb_id(
        ag, voxel_size=voxel_size, grid_size=grid_size, force_symmetry=True, from_atom_group=True,
        do_center_atoms=False, finufft_nthreads=1,
    )
    shape = (grid_size,) * 3
    return np.asarray(ftu.get_idft3(jnp.asarray(ft).reshape(shape)).real, dtype=np.float32)


def soft_mask(binary, *, extend_px: int, soft_px: int):
    """RELION ``relion_mask_create`` style: extend a binary mask, then a raised-cosine edge of ``soft_px``."""
    dist = ndimage.distance_transform_edt(~binary)
    mask = np.zeros(binary.shape, dtype=np.float32)
    mask[dist <= extend_px] = 1.0
    edge = (dist > extend_px) & (dist <= extend_px + soft_px)
    mask[edge] = 0.5 + 0.5 * np.cos(np.pi * (dist[edge] - extend_px) / soft_px)
    return mask


def prepare(args):
    out = Path(args.output_dir)
    mkdir_safe(str(out))
    np.random.seed(args.seed)
    voxel_size = 4.25 * 128 / args.grid_size if args.voxel_size is None else args.voxel_size
    D = args.grid_size

    (b1_xyz, b1_el), (b2_xyz, b2_el), axis, pivot, n_cut = body_atoms(args.cut_angstrom, args.gap_angstrom)
    logger.info("body 1: %d atoms, body 2: %d atoms, %d removed in the gap", len(b1_xyz), len(b2_xyz), n_cut)
    angles, deformations = state_table(args)
    lobe, base, hinge_axis = body2_lobe(b2_xyz, axis, pivot, args.lobe_radius)
    if args.deformation != "none":
        logger.info("deforming lobe: %d body-2 atoms, hinge axis %s", int(lobe.sum()), hinge_axis)
    mkdir_safe(str(out / "states"))
    mkdir_safe(str(out / "bodies"))
    binary = dict(voxel_size=voxel_size, grid_size=D, radius_angstrom=args.mask_atom_radius)
    union = [atom_binary(b1_xyz, **binary), np.zeros((D,) * 3, bool)]
    for idx, (angle, delta) in enumerate(zip(angles, deformations)):
        moved = rigid_motion(deform_body2(b2_xyz, lobe, base, hinge_axis, delta), pivot, body2_rotation(angle, axis))
        vol = real_volume(np.concatenate([b1_xyz, moved]), np.concatenate([b1_el, b2_el]), voxel_size=voxel_size, grid_size=D)
        utils.write_mrc(str(out / "states" / f"vol{idx:04d}.mrc"), vol, voxel_size=voxel_size)
        union[1] |= atom_binary(moved, **binary)
        logger.info("state %d angle %.3f deg deformation %.3f deg", idx, angle, delta)
    if args.deformation != "none":
        mkdir_safe(str(out / "body2_states"))
        for idx, delta in enumerate(deformations):
            vol = real_volume(deform_body2(b2_xyz, lobe, base, hinge_axis, delta), b2_el, voxel_size=voxel_size, grid_size=D)
            utils.write_mrc(str(out / "body2_states" / f"vol{idx:04d}.mrc"), vol, voxel_size=voxel_size)
    body1 = real_volume(b1_xyz, b1_el, voxel_size=voxel_size, grid_size=D)
    body2_neutral = real_volume(b2_xyz, b2_el, voxel_size=voxel_size, grid_size=D)
    utils.write_mrc(str(out / "bodies" / "vol0000.mrc"), body1, voxel_size=voxel_size)
    utils.write_mrc(str(out / "bodies" / "vol0001.mrc"), body2_neutral, voxel_size=voxel_size)
    masks = [soft_mask(u, extend_px=args.mask_extend_px, soft_px=args.mask_soft_px) for u in union]
    # The simulated maps carry low positive ringing across the whole box (about 12% of their positive mass, none
    # above 5% of the peak), so mask coverage is the fraction of each body's signal energy inside its mask.
    inside = [float((v * v * m).sum() / (v * v).sum()) for v, m in zip((body1, body2_neutral), masks)]
    logger.info("fraction of each body's signal energy inside its soft mask: %s", inside)
    if min(inside) < args.min_mask_coverage:
        raise RuntimeError(f"body masks do not cover their body volumes ({inside}): frame mismatch")
    overlap = float(np.max(masks[0] * masks[1]))
    if overlap > 0:
        raise RuntimeError(f"body masks overlap (max product {overlap}); increase --gap-angstrom")
    star_rows = []
    for b, mask in enumerate(masks, start=1):
        # RELION's frame is the axis-reversed recovar frame; write_relion_mrc would also negate (a density
        # convention), which a [0, 1] mask must not take.
        with mrcfile.new(str(out / f"body{b}_mask_relion.mrc"), overwrite=True) as mrc:
            mrc.set_data(np.ascontiguousarray(np.transpose(mask, (2, 1, 0)), dtype=np.float32))
            mrc.voxel_size = voxel_size
        star_rows.append(f"body{b}_mask_relion.mrc {3 - b} {args.body_sigma_angles:g} {args.body_sigma_offset:g}")
    (out / "bodies.star").write_text(
        "\ndata_\n\nloop_\n_rlnBodyMaskName #1\n_rlnBodyRotateRelativeTo #2\n_rlnBodySigmaAngles #3\n"
        "_rlnBodySigmaOffsetAngst #4\n" + "\n".join(star_rows) + "\n"
    )

    atomic_volume_kwargs, bfactor_record = resolve_total_bfactor(
        solvent_contrast.kwargs_from_cli_args(args), baked_in=0.0, source="5nrl body atoms (B 0)"
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
    record = {
        "source": "scripts/prepare_multibody_benchmark.py",
        "pdb": "recovar assets/5nrl_atoms.npz",
        "body1_chain_groups": ["Ab", "Db"], "body2_chain_groups": ["B"],
        "cut_angstrom": args.cut_angstrom, "gap_angstrom": args.gap_angstrom, "atoms_removed": n_cut,
        "twist_axis": axis.tolist(), "pivot_angstrom_centered": pivot.tolist(),
        "rotation": "scipy Rotation.from_rotvec(deg2rad(angle) * twist_axis) applied to body-2 atoms about the "
        "pivot (5nrl coordinates centered by prepare_5nrl_subcomplexes, before simulation)",
        "energy_fraction_inside_masks": inside,
        "state_angles_deg": angles.tolist(),
        "deformation": args.deformation, "state_deformation_deg": deformations.tolist(),
        "lobe_radius_angstrom": args.lobe_radius, "lobe_atoms": int(lobe.sum()),
        "lobe_hinge_point_angstrom_centered": base.tolist(), "lobe_hinge_axis": hinge_axis.tolist(),
        "grid_size": D, "voxel_size": voxel_size, "n_images": args.n_images, "noise_level": args.noise_level,
        "seed": args.seed, "bfactor": bfactor_record, "mask_atom_radius": args.mask_atom_radius,
        "mask_extend_px": args.mask_extend_px, "mask_soft_px": args.mask_soft_px,
        "body_sigma_angles": args.body_sigma_angles, "body_sigma_offset_angstrom": args.body_sigma_offset,
        "init_resolution_ang": args.init_resolution_ang, "volume_radius": args.volume_radius,
    }
    (out / "body_motion.json").write_text(json.dumps(record, indent=2))
    logger.info("Multi-body benchmark ready at %s", out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--n-states", type=int, default=17)
    parser.add_argument("--max-angle-deg", type=float, default=8.0)
    parser.add_argument("--cut-angstrom", type=float, default=40.0)
    parser.add_argument("--gap-angstrom", type=float, default=34.0)
    parser.add_argument("--mask-atom-radius", type=float, default=3.0)
    parser.add_argument("--mask-extend-px", type=int, default=0)
    parser.add_argument("--mask-soft-px", type=int, default=3)
    parser.add_argument("--min-mask-coverage", type=float, default=0.75,
                        help="refuse masks holding less than this fraction of each body's signal energy")
    parser.add_argument("--body-sigma-angles", type=float, default=10.0)
    parser.add_argument("--body-sigma-offset", type=float, default=4.0)
    parser.add_argument("--init-resolution-ang", type=float, default=30.0)
    parser.add_argument("--volume-radius", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--deformation", choices=("none", "correlated", "independent"), default="none")
    parser.add_argument("--max-deformation-deg", type=float, default=15.0)
    parser.add_argument("--n-deformation-states", type=int, default=3)
    parser.add_argument("--lobe-radius", type=float, default=35.0)
    solvent_contrast.add_cli_arguments(parser, enabled_by_default=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
