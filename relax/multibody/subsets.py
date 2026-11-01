"""Fixed-pose adaptive subset reconstruction of one body from a finished multi-body refinement.

A multi-body run (RELION's or relax's) gives every particle its body poses and, through ``relion_flex_analyse
--PCA_orient --write_pca_projections``, coordinates in the multi-body motion embedding. For a target point ``z*``
and a target body ``b`` the particles within distance ``h`` of ``z*`` form nested subsets ``S_h``; each subset's
body map is reconstructed at the particles' existing poses of body ``b`` from their images with the other bodies
subtracted, and RECOVAR chooses among the bandwidths by comparing each half's candidates with the other half's
narrowest-subset map (``recovar.heterogeneity.heterogeneity_volume.make_volumes_kernel_estimate_local`` with the
``"square"`` kernel, the hard-subset kernel of ``even_less_naive_heterogeneity_scheme_relion_style``).

Nothing is realigned: the residual images ``D_ib`` (:func:`relax.multibody.subtraction.subtracted_body_images`, with
each half's own body maps), the target body's composed poses and offsets, the norm corrections and the noise are
fixed for every bandwidth, so each particle's normal-equation term does not depend on ``h`` and RECOVAR's
disjoint-bin accumulation with cumulative sums gives every candidate. A candidate is a new reconstruction from the
selected images, in the target body's frame (centred on its centre of mass); it is a point-pose estimate, not
RELION's soft-posterior body map, so the all-particle candidate differs from ``run_body<BBB>.mrc``.

Distances are Euclidean in the chosen, unwhitened PCA components; a subset is ``{i : d_i <= h}`` (``np.digitize``
with ``right=True``), with the same thresholds in both halves.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import starfile
from recovar.core import fourier_transform_utils as ftu
from recovar.utils.helpers import load_relion_volume

from relax.multibody import geometry
from relax.multibody.pose_neighborhoods import PoseFeatures, body_features
from relax.multibody.subtraction import BodyPoses, dataset_with_images, subtracted_body_images

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultibodyResult:
    """A finished multi-body run, in the order of the consensus particle table: each body's residual poses, each
    body's half maps of the last numbered iteration (the references of the final pass; recovar Fourier, stored
    frame) and each half's radial noise spectrum (relax units, RELION's times ``box**4``)."""

    directory: Path
    iteration: int
    poses: list  # BodyPoses per body, every particle
    body_maps: list  # [body][half]
    noise_radial: list  # per half


def _last_numbered_iteration(directory: Path) -> int:
    numbers = [int(m.group(1)) for p in directory.glob("run_it*_half1_body001.mrc") if (m := re.search(r"run_it(\d+)_", p.name))]
    if not numbers:
        raise FileNotFoundError(f"{directory}: no run_itNNN_half1_body001.mrc")
    return max(numbers)


def _noise(directory: Path, iteration: int, box_size: int) -> list:
    frame = float(box_size) ** 4
    relax_model = directory / f"run_it{iteration:03d}_model.star"
    if relax_model.exists():
        blocks = starfile.read(relax_model, always_dict=True)
        return [np.asarray(blocks[f"model_noise_half{h + 1}"]["rlnSigma2Noise"], dtype=np.float64) * frame for h in range(2)]
    noise = []
    for h in range(2):
        blocks = starfile.read(directory / f"run_it{iteration:03d}_half{h + 1}_model.star", always_dict=True)
        noise.append(np.asarray(blocks["model_optics_group_1"]["rlnSigma2Noise"], dtype=np.float64) * frame)
    return noise


def read_multibody_result(directory, run) -> MultibodyResult:
    """The final poses (``run_data.star``) and the last numbered iteration's half maps and noise of the multi-body
    run in ``directory``, joined to ``run``'s consensus particle table by image name."""

    directory = Path(directory)
    data = starfile.read(directory / "run_data.star", always_dict=True)
    names = np.asarray(run.checkpoint.particles["rlnImageName"])
    order = _join_order(np.asarray(data["particles"]["rlnImageName"]), names, what=str(directory / "run_data.star"))
    poses = []
    for b in range(len(run.bodies)):
        table = data[f"images_body_{b + 1}"]
        poses.append(BodyPoses(
            eulers_deg=table[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(np.float64)[order],
            offsets_px=table[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(np.float64)[order] / run.pixel_size,
        ))
    iteration = _last_numbered_iteration(directory)

    def latest(b, h):
        # A body fixed before the end keeps its maps; outputs that skipped a fixed body's files hold them earlier.
        for it in range(iteration, -1, -1):
            path = directory / f"run_it{it:03d}_half{h + 1}_body{b + 1:03d}.mrc"
            if path.exists():
                return path
        raise FileNotFoundError(f"{directory}: no half-{h + 1} map of body {b + 1}")

    body_maps = [
        [
            np.asarray(ftu.get_dft3(jnp.asarray(load_relion_volume(str(latest(b, h))), dtype=jnp.float32)).reshape(-1))
            for h in range(2)
        ]
        for b in range(len(run.bodies))
    ]
    return MultibodyResult(
        directory=directory, iteration=iteration, poses=poses, body_maps=body_maps,
        noise_radial=_noise(directory, iteration, run.box_size),
    )


def _join_order(source_names: np.ndarray, target_names: np.ndarray, *, what: str) -> np.ndarray:
    """Row of ``source_names`` for every name of ``target_names``; names must be unique and match one to one."""

    if len(set(source_names)) != len(source_names):
        raise ValueError(f"{what}: duplicate image names")
    index = {name: i for i, name in enumerate(source_names)}
    missing = [n for n in target_names if n not in index]
    extra = len(source_names) - (len(target_names) - len(missing))
    if missing or extra:
        raise ValueError(f"{what}: {len(missing)} particles missing, {extra} extra (first missing: {missing[:3]})")
    return np.asarray([index[n] for n in target_names], dtype=np.int64)


def read_flex_projections(path, target_names) -> np.ndarray:
    """``relion_flex_analyse --write_pca_projections`` scores (``<o>_projections_along_eigenvectors_all_particles.txt``:
    an image name, then one column per component) for ``target_names``, ``[N, n_components]``."""

    names, rows = [], []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields:
            names.append(fields[0])
            rows.append([float(v) for v in fields[1:]])
    order = _join_order(np.asarray(names), np.asarray(target_names), what=str(path))
    return np.asarray(rows, dtype=np.float64)[order]


def distances_to(scores: np.ndarray, components, target) -> np.ndarray:
    """Euclidean distance (not squared) of every particle's chosen components to ``target``."""

    z = np.asarray(scores, dtype=np.float64)[:, list(components)]
    return np.linalg.norm(z - np.asarray(target, dtype=np.float64)[None, :], axis=1)


def body_pose_features(run, result: MultibodyResult, target: int) -> PoseFeatures:
    """The pose comparison terms of body ``target``, from that body's poses alone."""

    particles = run.checkpoint.particles
    consensus = geometry.consensus_matrices(particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(np.float64))
    return body_features(
        consensus, geometry.body_matrices(consensus, run.bodies[target], result.poses[target].eulers_deg), run.bodies[target],
        particles[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(np.float64) / run.pixel_size,
        result.poses[target].offsets_px, run.pixel_size,
    )


def target_row(names: np.ndarray, image_name: str) -> int:
    """The row of the one particle called ``image_name``."""

    rows = np.flatnonzero(np.asarray(names) == image_name)
    if rows.size != 1:
        raise ValueError(f"{image_name}: {rows.size} particles have this name")
    return int(rows[0])


def nested_thresholds(distances: np.ndarray, *, n_candidates: int, n_min_per_half: int, halves) -> np.ndarray:
    """``n_candidates`` increasing distance thresholds, common to both halves: from the smallest one keeping
    ``n_min_per_half`` particles in each half to the largest distance (the all-particle endpoint), evenly spaced
    in the number of particles kept."""

    d = np.asarray(distances, dtype=np.float64)
    smallest = max(np.sort(d[rows])[min(n_min_per_half, rows.size) - 1] for rows in halves)
    pool = np.sort(d[d >= smallest])
    distinct = np.unique(pool)
    if distinct.size <= n_candidates:  # few distinct distances (a discrete coordinate): every one is a threshold
        thresholds = distinct
    else:
        counts = np.linspace(0, pool.size - 1, n_candidates).round().astype(np.int64)
        thresholds = np.unique(pool[counts])
    thresholds[-1] = d.max()
    return thresholds


def target_body_images(run, result: MultibodyResult, target: int, half: int) -> np.ndarray:
    """Half ``half``'s images of body ``target`` with the other bodies subtracted at their final poses, with the
    half's own body maps (:func:`relax.multibody.subtraction.subtracted_body_images` at the full box)."""

    from relax.multibody.refine import half_projectors, target_projector_sets

    particles = run.halves[half]
    _, body_projectors = half_projectors(result.body_maps, run, half, run.box_size)
    projectors = target_projector_sets(result.body_maps, run, half, run.box_size, body_projectors)[target]
    poses = [BodyPoses(p.eulers_deg[particles.rows], p.offsets_px[particles.rows]) for p in result.poses]
    return subtracted_body_images(
        run.dataset.subset(particles.rows), np.arange(particles.rows.size), target, run.bodies, projectors, poses,
        particles.frames, current_size=run.box_size, padding_factor=run.settings.projection_padding_factor,
    )


def fixed_pose_dataset(run, result: MultibodyResult, target: int):
    """A RECOVAR dataset of body ``target``: every particle's subtracted image, its composed body pose (RELION's
    matrix, which is RECOVAR's) and refined body offset (RELION's origin, pixels), the consensus halves and the
    mean of the two halves' noise spectra. Images are norm-corrected and not masked (RELION reconstructs from the
    unmasked image), in RELION's image transform."""

    images = np.zeros((run.dataset.n_images, *run.dataset.image_shape), dtype=np.float32)
    for half in range(2):
        particles = run.halves[half]
        # RELION's preprocessing multiplies each image by its norm factor; RECOVAR's reconstruction reads the
        # images through the plain (unmasked, unshifted) RELION transform, so the factor is applied here.
        images[particles.rows] = target_body_images(run, result, target, half) * particles.frames.norm_factors[:, None, None]
    dataset = dataset_with_images(
        run.dataset, images, mask_diameter_angstrom=4.0 * run.box_size * run.pixel_size
    )
    dataset.image_source.backend.set_relion_fourier_backend("jax_gpu")
    dataset._metadata = copy.copy(dataset._metadata)  # the poses below are this dataset's, not the run's
    particles = run.checkpoint.particles
    consensus = geometry.consensus_matrices(
        particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(np.float64)
    )
    dataset.rotation_matrices = geometry.body_matrices(consensus, run.bodies[target], result.poses[target].eulers_deg)
    dataset.translations = result.poses[target].offsets_px
    dataset.halfset_indices = [np.asarray(run.halves[h].rows, dtype=np.int64) for h in range(2)]
    dataset.set_noise(0.5 * (result.noise_radial[0] + result.noise_radial[1]))
    return dataset


def adaptive_subsets(run, result: MultibodyResult, *, target: int, distances: np.ndarray, thresholds: np.ndarray,
                     output, prefix: str = "subset", metric: str = "locshellmost_likely", b_factor: float = 0.0,
                     maskrad_fraction: float = 20.0, save_all_estimates: bool = True) -> dict:
    """Every hard-subset candidate of body ``target`` and RECOVAR's selected map, written under ``output``
    (``recovar.output.output_paths.VolumeOutputPaths(output, prefix, 0)``); returns the run record, also written
    to ``<output>/<prefix>_subsets.json``.

    The local-selection settings are those of RECOVAR's external-embedding wrapper
    (``recovar.output.output.compute_and_save_reweighted``): ``locres_sampling = locres_maskrad = box *
    voxel / maskrad_fraction``, no mask edge, the half-spectrum reconstruction path."""

    from recovar.heterogeneity.heterogeneity_volume import make_volumes_kernel_estimate_local
    from recovar.output.output_paths import VolumeOutputPaths

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    locres = run.box_size * run.pixel_size / maskrad_fraction
    dataset = fixed_pose_dataset(run, result, target)
    halves = [np.asarray(run.halves[h].rows) for h in range(2)]
    per_half = [np.asarray(distances, dtype=np.float64)[rows] for rows in halves]
    counts = [[int(np.sum(d <= t)) for t in thresholds] for d in per_half]
    names = np.asarray(run.checkpoint.particles["rlnImageName"])
    record = {
        "multibody_run": str(result.directory), "maps_iteration": result.iteration, "target_body": target + 1,
        "thresholds": [float(t) for t in thresholds], "counts_per_half": counts, "kernel": "square",
        "membership": "d <= threshold (np.digitize right=True), common thresholds in both halves",
        "metric": metric, "b_factor": b_factor, "locres_sampling_and_maskrad": locres,
        "frame": "body frame: centred on the body's centre of mass (relax.multibody.subtraction.stored_body_map "
                 "moves a map to the stored frame)",
        "noise": "mean of the two halves' radial noise spectra of the maps' iteration",
    }
    half = np.zeros(names.size, dtype=np.int8)
    half[halves[1]] = 1
    np.savez_compressed(output / f"{prefix}_membership.npz", image_names=names, distances=np.asarray(distances), half=half)
    paths = VolumeOutputPaths(str(output), prefix, 0)
    make_volumes_kernel_estimate_local(
        per_half, dataset, paths, ndim=-1, bins=np.asarray(thresholds, dtype=np.float64), B_factor=b_factor,
        metric_used=metric, locres_sampling=locres, locres_maskrad=locres, locres_edgwidth=0,
        upsampling_for_ests=1, use_mask_ests=False, grid_correct_ests=False, save_all_estimates=save_all_estimates,
        heterogeneity_kernel="square", use_fast_rfft=True,
    )
    (output / f"{prefix}_subsets.json").write_text(json.dumps(record, indent=1))
    return record


def check_cumulative_against_direct(dataset, distances_half: np.ndarray, thresholds, *, half: int = 0, indices=None) -> list:
    """The cumulative-bin candidates of one half against reconstructions of the same hard subsets, each from its
    own single bin (``{i : d_i <= h}``): the relative difference per checked threshold, with the subset size. The
    two differ only in the order the per-bin sums are added."""

    from recovar.heterogeneity.adaptive_kernel_discretization import even_less_naive_heterogeneity_scheme_relion_style

    half_dataset = dataset.get_halfset(half)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    settings = dict(tau=None, grid_correct=False, use_spherical_mask=False, heterogeneity_kernel="square",
                    upsampling_factor=1, return_real_space=True, use_fast_rfft=True)
    cumulative = np.asarray(even_less_naive_heterogeneity_scheme_relion_style(
        half_dataset, None, distances_half, thresholds, **settings
    )).reshape(thresholds.size, -1)
    indices = [0, thresholds.size // 2, thresholds.size - 1] if indices is None else indices
    out = []
    for k in indices:
        direct = np.asarray(even_less_naive_heterogeneity_scheme_relion_style(
            half_dataset, None, distances_half, thresholds[k:k + 1], **settings
        )).reshape(-1)
        out.append({
            "candidate": int(k), "threshold": float(thresholds[k]), "n": int(np.sum(distances_half <= thresholds[k])),
            "relative_difference": float(np.linalg.norm(cumulative[k] - direct) / np.linalg.norm(direct)),
        })
    return out
