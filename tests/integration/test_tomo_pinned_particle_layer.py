"""S4.1: relax's subtomogram particle layer against RELION's cryo-ET Refine3D, pinned (CPU).

The fixture is a simulated RELION 5 tomography dataset: two optics groups, 41 tilts,
normalised tilt stacks and non-zero ground-truth 3D offsets. It comes with a RELION 5.0.1
f2c1a3 Refine3D oracle (em_fixtures/cryoet_s1_offsets_20260925, README.md there).

The test pins RELION's last split-half iteration, it019. Its E-step used the it018 half
maps, noise and group scales, and RELION's it019 poses and 3D offsets are its answer.
relax's particle layer (``relax.refinement.tomo_particles``) builds every tilt image from
a particle hypothesis:

- the image matrix is ``Aproj_i R`` and the image shift ``Aproj_i[:2] t``;
- the image diff2 values are summed per particle.

Each image is then scored or reconstructed with RELION's own operators: the relion_bind
masks, shifts and exact tomo CTF with dose, and a numpy port of Projector::project.

- E-step: at a local grid around RELION's pose, the rotation argmax is RELION's, and the
  offset argmax is RELION's with the + shift sign (the − sign loses it). Where relax's
  offset disagrees, it points toward the ground truth, because RELION moves an offset by at
  most about 0.3 Å per iteration.
- M-step noise: residuals at RELION's poses are averaged over each particle's images and
  normalised by the particles' weight per optics group. That reproduces RELION's it019
  sigma2 per group and half (the GPU path, acc_ml_optimiser_impl.h:3490-3491).

Pinned full-data runs of the same computation are in
em_work/cryoet_s1_20260923/s4_offsets_fixture_20260925 (jobs 14410210, 14410538).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mrcfile
import numpy as np
import pytest
import starfile
from helpers.em_fixtures import fixture_root, require_fixture_sets
from helpers.relion_projector_reference import make_projector, project

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

DATA = fixture_root("cryoet_s1_offsets_data")
RELION = fixture_root("cryoet_s1_offsets_relion")
IT = 19  # RELION's last split-half iteration; its E-step used the it018 model
ORIGIN = ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
ANGLES = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]


def _euler(rot, tilt, psi):
    """RELION Euler_angles2matrix (src/euler.cpp), vectorised."""
    a, b, g = np.deg2rad(rot), np.deg2rad(tilt), np.deg2rad(psi)
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return np.stack(
        [
            np.stack([cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb], -1),
            np.stack([-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb], -1),
            np.stack([sc, ss, cb], -1),
        ],
        -2,
    )


@pytest.fixture(scope="module")
def pinned():
    """RELION's it019 answer, its it018 model and the images as relax's particle layer sees them."""
    require_fixture_sets("cryoet_s1_offsets_data", "cryoet_s1_offsets_relion")
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.data_io.starfile import read_star, star_column

    from relax.refinement import tomo_particles as tp
    from relax.relion import relion_ctf, tomo_input
    from relax.relion_bind import _relion_bind_core as bind

    project_dir = DATA
    flat = project_dir / "particles_2d.star"
    rows, optics = read_star(str(flat))
    index = tomo_input.tomo_particle_index(rows)
    image_particle = index.image_particle()
    n = int(np.asarray(star_column(optics, "rlnImageSize"))[0])
    pix = float(np.asarray(star_column(optics, "rlnImagePixelSize"))[0])
    names = index.particle_names
    gt = starfile.read(project_dir / "particles.star")["particles"].set_index("rlnTomoParticleName").loc[names]
    data = starfile.read(RELION / f"run_it{IT:03d}_data.star")["particles"].set_index("rlnTomoParticleName").loc[names]
    prev = (
        starfile.read(RELION / f"run_it{IT - 1:03d}_data.star")["particles"].set_index("rlnTomoParticleName").loc[names]
    )
    assert np.array_equal(data["rlnRandomSubset"].to_numpy(), index.half_set)
    image_matrices = _euler(*(np.asarray(star_column(rows, k), dtype=float) for k in ANGLES))
    aproj = tp.tilt_projection_matrices(image_matrices, _euler(*(gt[k].to_numpy() for k in ANGLES)), image_particle)
    models = {h: starfile.read(RELION / f"run_it{IT - 1:03d}_half{h}_model.star") for h in (1, 2)}
    cs = int(models[1]["model_general"]["rlnCurrentImageSize"])
    refs = {
        h: mrcfile.open(RELION / f"run_it{IT - 1:03d}_half{h}_class001.mrc").data.astype(np.float64) for h in (1, 2)
    }
    optimiser = (RELION / f"run_it{IT:03d}_optimiser.star").read_text()
    rows_i = np.arange(n)
    y = np.where(rows_i < n // 2 + 1, rows_i, rows_i - n)
    shell = np.rint(np.sqrt(y[:, None] ** 2 + np.arange(n // 2 + 1)[None, :] ** 2)).astype(int)
    return SimpleNamespace(
        tp=tp,
        bind=bind,
        index=index,
        image_particle=image_particle,
        n=n,
        pix=pix,
        cs=cs,
        aproj=aproj,
        gt_offset=gt[ORIGIN].to_numpy(dtype=float),
        pose=data[ANGLES].to_numpy(dtype=float),
        offset=data[ORIGIN].to_numpy(dtype=float),
        prev_offset=prev[ORIGIN].to_numpy(dtype=float),
        group_number=data["rlnGroupNumber"].to_numpy(),
        image_group=np.asarray(star_column(rows, "rlnOpticsGroup"), dtype=int),
        models=models,
        new_models={h: starfile.read(RELION / f"run_it{IT:03d}_half{h}_model.star") for h in (1, 2)},
        projectors={h: make_projector(bind, refs[h], n, 2, cs) for h in (1, 2)},
        mask_radius=float(optimiser.split("_rlnParticleDiameter")[1].split()[0]) / (2 * pix),
        mask_edge=float(optimiser.split("_rlnWidthMaskEdge")[1].split()[0]),
        shell=shell,
        dataset=load_dataset(str(flat), datadir=str(project_dir.resolve()), lazy=True, dtype=np.complex128),
        ctf=lambda imgs: (
            -np.fft.ifftshift(
                relion_ctf._relion_exact_ctf_half_from_source_star_host(
                    SimpleNamespace(particles_file=str(flat)), imgs, (n, n)
                ).reshape(-1, n, n // 2 + 1),
                axes=1,
            )
        ),
    )


def _masked_images(p, imgs):
    """The E-step's images: --zero_mask soft circle, then RELION's FFT scaling (1 / box^2)."""
    real = np.asarray(p.dataset.image_source.host_images(imgs), dtype=np.float64).reshape(-1, p.n, p.n)
    real = np.stack([p.bind.soft_mask_outside_map_2d(r, p.mask_radius, p.mask_edge) for r in real])
    return np.fft.rfft2(np.fft.ifftshift(real, axes=(-2, -1))) / p.n**2


def _scale(p, h, particle):
    groups = p.models[h]["model_groups"].set_index("rlnGroupNumber")["rlnGroupScaleCorrection"]
    return float(groups.loc[p.group_number[particle]])


def _local_posterior(p, particle, shift_sign, rotation_deltas, offset_deltas):
    """Posterior over a local hypothesis grid around RELION's pose, as relax's particle layer scores it."""
    h = int(p.index.half_set[particle])
    imgs = np.arange(p.index.image_offsets[particle], p.index.image_offsets[particle + 1])
    fimg = _masked_images(p, imgs)
    sigma2 = p.models[h][f"model_optics_group_{int(p.image_group[imgs[0]])}"]["rlnSigma2Noise"].to_numpy()
    inside = (p.shell < p.cs // 2) & (p.shell > 0)
    inv_sigma2 = np.where(inside, 1.0 / sigma2[np.minimum(p.shell, sigma2.size - 1)], 0.0)
    ctf = p.ctf(imgs) * _scale(p, h, particle)
    rotations = _euler(*(p.pose[particle][None, :] + rotation_deltas).T)
    image_rotations = p.tp.tilt_image_rotations(rotations, p.aproj[imgs])
    model = project(p.projectors[h], image_rotations.reshape(-1, 3, 3), p.n).reshape(len(imgs), len(rotations), p.n, -1)
    model = model * ctf[:, None]
    trial = p.offset[particle][None, :] + offset_deltas
    shifts = p.tp.tilt_image_shifts(trial, np.zeros((1, 3)), p.aproj[imgs], np.zeros(len(imgs), dtype=int)) / p.pix
    diff2 = np.empty((len(imgs), len(rotations), len(trial)))
    for i in range(len(imgs)):
        for t in range(len(trial)):
            x = p.bind.shift_image_in_fourier_transform_2d(fimg[i], p.n, p.n, *(shift_sign * shifts[i, t]))
            diff2[i, :, t] = 0.5 * np.sum(np.abs(x[None] - model[i]) ** 2 * inv_sigma2[None], axis=(-2, -1))
    score = p.tp.particle_scores(diff2.reshape(len(imgs), -1), np.zeros(len(imgs), dtype=int), 1)[0]
    sigma_offset = float(p.models[h]["model_general"]["rlnSigmaOffsetsAngst"])
    log_prior = -np.sum((trial - p.prev_offset[particle]) ** 2, axis=1) / (2 * sigma_offset**2)
    log_post = -score.reshape(len(rotations), len(trial)) + log_prior[None, :]
    post = np.exp(log_post - log_post.max())
    return post / post.sum()


def _particles(p, per_half_and_group):
    return np.concatenate(
        [
            np.flatnonzero((p.index.half_set == h) & (p.index.optics_group == g))[:per_half_and_group]
            for h in (1, 2)
            for g in (1, 2)
        ]
    )


def test_estep_keeps_relions_pose_with_the_plus_shift_sign(pinned):
    p = pinned
    step = np.array([-1.0, 0.0, 1.0])
    grid = np.stack(np.meshgrid(step, step, step, indexing="ij"), -1).reshape(-1, 3)
    center = int(np.flatnonzero(np.all(grid == 0, axis=1))[0])
    rotation_deltas, offset_deltas = grid * 1.5, grid * 1.5  # degrees, Angstrom
    rotation_kept, offset_kept, toward_gt, minus_offset_kept = [], [], [], []
    for particle in _particles(p, 4):
        post = _local_posterior(p, particle, +1.0, rotation_deltas, offset_deltas)
        best_rotation, best_offset = np.unravel_index(int(np.argmax(post)), post.shape)
        rotation_kept.append(best_rotation == center)
        offset_kept.append(best_offset == center)
        if best_offset != center:
            moved = p.offset[particle] + offset_deltas[best_offset]
            toward_gt.append(
                np.linalg.norm(moved - p.gt_offset[particle])
                < np.linalg.norm(p.offset[particle] - p.gt_offset[particle])
            )
        minus = _local_posterior(p, particle, -1.0, rotation_deltas[center : center + 1], offset_deltas)
        minus_offset_kept.append(int(np.argmax(minus[0])) == center)
    summary = {
        "particles": len(rotation_kept),
        "rotation_kept": float(np.mean(rotation_kept)),
        "offset_kept_plus": float(np.mean(offset_kept)),
        "offset_moved_toward_gt": f"{sum(toward_gt)}/{len(toward_gt)}",
        "offset_kept_minus": float(np.mean(minus_offset_kept)),
    }
    logger.info("S4.1 pinned E-step: %s", summary)
    # Measured here (16 particles): rotation 1.0, offset 0.875 (+) and 0.0 (−), 2/2 moved toward the GT.
    # Full-data runs (64 particles, 14410210): rotation 0.98, offset 0.83 (+) and 0.00 (−).
    assert summary["rotation_kept"] >= 0.9, summary
    assert summary["offset_kept_plus"] >= 0.6, summary
    assert all(toward_gt), summary
    assert summary["offset_kept_minus"] <= 0.1, summary


def test_mstep_noise_per_optics_group_is_relions(pinned):
    p = pinned
    tp = p.tp
    particles = _particles(p, 40)
    n_half = p.n // 2 + 1
    inside = p.shell < p.cs // 2
    npix = np.bincount(p.shell.ravel(), minlength=n_half)[:n_half]
    wsum = {(h, g): np.zeros(n_half) for h in (1, 2) for g in (1, 2)}
    sumw = {(h, g): 0.0 for h in (1, 2) for g in (1, 2)}
    imgs = np.concatenate([np.arange(p.index.image_offsets[q], p.index.image_offsets[q + 1]) for q in particles])
    local_particle = np.repeat(np.arange(particles.size), np.diff(p.index.image_offsets)[particles])
    noise_scale = tp.image_noise_scale(local_particle, particles.size)
    weight = tp.image_weights(np.ones(particles.size), local_particle)
    rotations = _euler(*p.pose[particles].T)
    for start in range(0, imgs.size, 512):
        block = np.arange(start, min(start + 512, imgs.size))
        idx = imgs[block]
        particle = particles[local_particle[block]]
        fimg = _masked_images(p, idx)
        shift = (
            np.einsum("iab,ib->ia", p.aproj[idx][:, :2, :], p.offset[particle]) / p.pix
        )  # tilt_image_shifts at one pose
        fimg = np.stack([p.bind.shift_image_in_fourier_transform_2d(f, p.n, p.n, *s) for f, s in zip(fimg, shift)])
        halves = p.index.half_set[particle]
        scale = np.array([_scale(p, int(h), int(q)) for h, q in zip(halves, particle)])
        ctf = p.ctf(idx) * scale[:, None, None]
        image_rotations = np.einsum("iab,ibc->iac", p.aproj[idx], rotations[local_particle[block]])
        for h in (1, 2):
            rows = np.flatnonzero(halves == h)
            if not rows.size:
                continue
            model = project(p.projectors[h], image_rotations[rows], p.n) * ctf[rows]
            resid = np.abs(fimg[rows] - model) ** 2 * inside
            for j, r in enumerate(rows):
                k = block[r]
                g = int(p.image_group[idx[r]])
                shell_sums = np.bincount(p.shell.ravel(), resid[j].ravel(), minlength=n_half)[:n_half]
                wsum[(h, g)] += noise_scale[k] * weight[k] * shell_sums
    for q in particles:
        sumw[(int(p.index.half_set[q]), int(p.index.optics_group[q]))] += 1.0
    table = {}
    for (h, g), sums in wsum.items():
        # RELION's update: wsum_sigma2_noise / (2 sumw Npix_per_shell), in RELION's FFT units.
        estimate = sums / (2.0 * sumw[(h, g)] * np.maximum(npix, 1))
        relion = p.new_models[h][f"model_optics_group_{g}"]["rlnSigma2Noise"].to_numpy()[:n_half]
        ratio = estimate[1 : p.cs // 2] / relion[1 : p.cs // 2]
        table[f"half{h}_group{g}"] = (float(np.median(ratio)), float(ratio.min()), float(ratio.max()))
    logger.info("S4.1 pinned M-step noise ratio relax/RELION (median, min, max): %s", table)
    # Measured here (40 particles per half and group): medians 0.9991-1.0005, extremes 0.981 and 1.047
    # (one shell); all particles (14410538): median 1.00001, range 0.9992-1.0148.
    for key, (median, low, high) in table.items():
        assert abs(median - 1.0) <= 0.02 and 0.9 <= low and high <= 1.1, (key, table[key])
