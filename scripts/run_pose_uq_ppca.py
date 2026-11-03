#!/usr/bin/env python
"""Per-body pose uncertainty and projected PPCA embeddings of a finished multi-body run.

    python -m scripts.run_pose_uq_ppca <audit|synthetic|fixture> --root <pose_neighborhoods> --out <dir> <dataset>

"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def load_run(data: Path, out: Path):
    from relax.commands.multibody import run_options
    from relax.multibody import subsets
    from relax.multibody.refine import MultibodyCommand, start_from_consensus

    consensus = sorted(glob.glob(str(data / "relion_consensus/run_it*_optimiser.star")))[-1]
    command = MultibodyCommand(body_star=data / "bodies.star", healpix_order=4, auto_local_healpix_order=4, offset_range_px=3.0,
                               offset_step_px=1.5, oversampling=1, reconstruct_subtracted_bodies=True, max_iter=0)
    run, _ = start_from_consensus(consensus, command, run_options(), output=out / f"unused_{data.name}")
    return run, subsets.read_multibody_result(data / "relion_multibody", run)


def final_offset_range_angstrom(data: Path, result) -> float:
    import starfile

    table = starfile.read(data / "relion_multibody" / f"run_it{result.iteration:03d}_sampling.star", always_dict=True)
    return float(table["sampling_general"]["rlnOffsetRange"])


def body_operands(run, result, target: int, half: int, rows, *, offset_range_angstrom: float, images=None, current_size=None):
    """:class:`relax.multibody.pose_uq.BodyOperands` of ``rows`` (positions in the half) through RELAX's own subtraction
    and native preprocessing. ``images`` ``[half n, D, D]`` replaces the subtracted images of the whole half."""
    import jax.numpy as jnp
    from recovar.core.configs import ForwardModelConfig
    from recovar.reconstruction import noise as noise_utils

    from relax.fine_pass.bucket_io import prepare_unshifted_bucket_operands
    from relax.io.batch_fetch import fetch_indexed_batch
    from relax.fourier.fourier_window import make_fourier_window_spec
    from relax.fourier.preprocessing import relion_half_translation_lattice
    from relax.multibody import geometry, subsets
    from relax.multibody.pose_uq import BodyOperands
    from relax.multibody.refine import half_projectors
    from relax.multibody.subtraction import dataset_with_images
    from relax.refinement.noise_updates import noise_model_from_shells

    particles = run.halves[half]
    rows = np.asarray(rows)
    if images is None:
        images = subsets.target_body_images(run, result, target, half)
    current_size = run.box_size if current_size is None else current_size
    full = np.zeros((run.dataset.n_images, *run.dataset.image_shape), dtype=np.float32)
    table_rows = particles.rows[rows]
    full[table_rows] = images[rows]
    radius_px = geometry.body_mask_radius_px(run.bodies[target], offset_range_angstrom, run.dataset.voxel_size)
    dataset = dataset_with_images(run.dataset, full, mask_diameter_angstrom=2.0 * radius_px * float(run.dataset.voxel_size)).subset(table_rows)
    shape = tuple(dataset.image_shape)
    n_half = shape[0] * (shape[1] // 2 + 1)
    window = np.asarray(make_fourier_window_spec(shape, current_size, n_half, window_at_box=True).score_indices_np)
    noise = noise_utils.to_batched_half_pixel_noise(
        jnp.asarray(noise_model_from_shells(result.noise_radial, shape).variance_per_half[half]), shape
    ).squeeze()
    poses = result.poses[target]
    offsets = poses.offsets_px if result.alignment_offsets_px is None else result.alignment_offsets_px[target]
    n = rows.size
    config = ForwardModelConfig.from_dataset(dataset, disc_type="linear_interp", process_fn=dataset.process_images)
    batch, ctf_params, _ = fetch_indexed_batch(dataset, np.arange(n))
    unshifted = prepare_unshifted_bucket_operands(
        dataset, jnp.asarray(batch), ctf_params, np.arange(n), noise_variance_half=noise, config=config, finite_check_mode="off",
        score_with_masked_images=True, image_corrections=particles.image_corrections[rows],
        scale_corrections=particles.scale_corrections[rows], image_pre_shifts=None, use_float64_scoring=False,
        return_direct_scoring_io=True, score_only=False, score_mode="gaussian", window_indices=jnp.asarray(window),
        relion_exact_bpref_operands=True, image_translations=np.asarray(offsets[table_rows], dtype=np.float64),
    )
    fft = float(np.prod(shape))
    Y = jnp.asarray(unshifted.processed_score_half_raw, dtype=jnp.complex128)[:, window] / fft
    # relax's exact CTF rows carry the opposite of RELION's sign, and relax's maps too; the prediction Y is compared
    # with is RELION's: its CTF times the projector's data.
    ctf = -jnp.asarray(unshifted.ctf_half_rfloat, dtype=jnp.float64)[:, window] * jnp.asarray(particles.scale_corrections[rows])[:, None]
    from relax.fourier.half_spectrum import make_scoring_half_image_weights, make_shell_indices_half

    inverse = fft**2 / jnp.asarray(np.asarray(noise).reshape(1, -1), dtype=jnp.float64)
    inverse = jnp.where(jnp.asarray(make_shell_indices_half(shape)) == 0, 0.0, inverse)
    inverse = (inverse * jnp.asarray(make_scoring_half_image_weights(shape, relion_half_sum=True), dtype=jnp.float64))[:, window]
    _, projectors = half_projectors(result.body_maps, run, half, current_size)
    projector = projectors[target]
    volume = jnp.asarray(projector.data, dtype=jnp.complex128)
    volume = volume[0] if volume.ndim == 4 else volume
    lattice = jnp.asarray(relion_half_translation_lattice(shape), dtype=jnp.float64)[window]
    consensus = particles.frames.consensus[rows]
    cameras = geometry.body_matrices(consensus, run.bodies[target], poses.eulers_deg[table_rows])
    ops = BodyOperands(
        volume=volume, r_max=int(projector.r_max), padding_factor=int(run.settings.projection_padding_factor),
        image_shape=shape, window=window, lattice=lattice, pixel_size=float(run.pixel_size), image=Y, ctf=ctf,
        weight=inverse * jnp.ones((n, 1)), cameras=cameras, consensus=consensus,
    )
    return ops, unshifted, projector, noise


def subset(ops, rows):
    import jax.numpy as jnp

    rows = np.asarray(rows)
    index = jnp.asarray(rows)
    return replace(ops, image=ops.image[index], ctf=ops.ctf[index], weight=ops.weight[index], cameras=ops.cameras[rows], consensus=ops.consensus[rows])


def polish_all(ops, batch: int = 128, **kwargs):
    """:func:`relax.multibody.pose_uq.polish` in batches; returns the fields as a dict of arrays."""
    from dataclasses import asdict

    from relax.multibody import pose_uq

    n = len(ops.cameras)
    parts = [asdict(pose_uq.polish(subset(ops, np.arange(s, min(s + batch, n))), **kwargs)) for s in range(0, n, batch)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def true_pose(run, result, data: Path, target: int):
    """Every particle's true camera and native target offset (the simulation's, independent of consensus error)."""
    from relax.multibody import geometry
    from scripts.multibody_adaptive_subsets import oracle_result

    truth = oracle_result(run, result, data)
    particles = run.checkpoint.particles
    consensus = geometry.consensus_matrices(particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(float))
    return geometry.body_matrices(consensus, run.bodies[target], truth.poses[target].eulers_deg), truth.alignment_offsets_px[target], truth


def pose_errors(cameras, offsets_px, true_cameras, true_offsets_px, consensus, pixel_size):
    """Local-parameter error ``[n, 5]`` of a pose from the truth: molecular-frame rotation vector, physical shift."""
    from relax.multibody.pose_uq import log_so3

    vectors = log_so3(cameras @ np.swapaxes(true_cameras, 1, 2))
    omega = np.einsum("nji,nj->ni", consensus, vectors)
    return np.concatenate([omega, pixel_size * (np.asarray(true_offsets_px) - np.asarray(offsets_px))], axis=1)


def remove_gauge(error, consensus):
    """The map frame of a run differs from the simulation's by one rigid motion: a constant rotation and a constant 3D
    translation, seen through each camera. Removed robustly (truth-side evaluation only). Returns the error and gauge."""
    omega = np.median(error[:, :3], axis=0)
    P = consensus[:, :2, :]
    design = P.reshape(-1, 3)
    target = error[:, 3:].reshape(-1)
    keep = np.ones(target.size, dtype=bool)
    for _ in range(5):
        translation = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
        residual = target - design @ translation
        keep = np.abs(residual) < 3.0 * 1.4826 * np.median(np.abs(residual))
    out = error.copy()
    out[:, :3] -= omega
    out[:, 3:] -= np.einsum("nij,j->ni", P, translation)
    return out, dict(omega_deg=np.rad2deg(omega).tolist(), translation_angstrom=translation.tolist())


CHI2_5 = {0.5: 4.3515, 0.9: 9.2364, 0.95: 11.0705, 0.99: 15.0863}


def calibration(error, covariance) -> dict:
    """Coverage of the 5D error ellipsoids and coordinate z scores against a chi-square with five degrees of freedom."""
    solved = np.linalg.solve(covariance, error[:, :, None])[:, :, 0]
    chi2 = np.sum(error * solved, axis=1)
    standard = error / np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    return dict(
        n=len(error), mean_chi2_over_5=float(chi2.mean() / 5.0), median_chi2_over_median=float(np.median(chi2) / CHI2_5[0.5]),
        ellipsoid_coverage={str(p): float(np.mean(chi2 < v)) for p, v in CHI2_5.items()},
        z_sd=[float(v) for v in standard.std(axis=0)], z_robust_sd=[float(v) for v in 1.4826 * np.median(np.abs(standard - np.median(standard, axis=0)), axis=0)],
        z_mean=[float(v) for v in standard.mean(axis=0)],
    )


def audit(args) -> None:
    """Figure group 1 arrays: native score agreement, finite-difference convergence, sections against quadratics."""
    import jax.numpy as jnp

    from relax.cuda import fine as cuda_fine
    from relax.fine_pass.scoring import relion_cuda_fine_full_to_compact_lookup, relion_native_score_corr_img, to_relion_native_fine_units
    from relax.multibody import pose_uq
    from relax.projection.projection import project_relion_projector_half_spectrum_centered_rows_at_indices as project

    data = args.root / args.dataset
    run, result = load_run(data, args.out)
    offset_range = final_offset_range_angstrom(data, result)
    n = args.particles
    ops, unshifted, projector, noise = body_operands(run, result, args.body, 0, np.arange(n), offset_range_angstrom=offset_range)
    report = dict(dataset=args.dataset, body=args.body + 1, particles=n, box=run.box_size, pixel_size=run.pixel_size, iteration=result.iteration,
                  offset_range_angstrom=offset_range, padding=run.settings.projection_padding_factor, window_pixels=int(ops.window.size), r_max=ops.r_max,
                  map_sources=[[str(p) for p in pair] for pair in result.map_sources])
    continuous = np.asarray(pose_uq.scores(ops, np.zeros((n, 1, 5))))[:, 0]
    shape, fft, window = ops.image_shape, int(np.prod(ops.image_shape)), jnp.asarray(ops.window)
    corr = relion_native_score_corr_img(jnp.asarray(np.asarray(noise).reshape(1, -1)), unshifted.ctf_half_rfloat, shape,
                                        jnp.asarray(run.halves[0].scale_corrections[:n], dtype=jnp.float32)[:, None], zero_dc=True)[:, window]
    corr = corr * jnp.asarray(ops.weight[0] > 0, dtype=corr.dtype)
    y = to_relion_native_fine_units(jnp.asarray(unshifted.sparse_score_input_half)[:, window], fft)
    data32 = jnp.asarray(projector.data, dtype=jnp.complex64)
    data32 = data32[0] if data32.ndim == 4 else data32
    ref = project(data32, jnp.asarray(ops.cameras), shape, ops.r_max, ops.padding_factor, None, window).astype(jnp.complex64)
    lookup = jnp.asarray(relion_cuda_fine_full_to_compact_lookup(shape, run.box_size, ops.window))
    native = np.asarray(cuda_fine.relion_fine_diff2_rectangular_f32(-ref[:, None], y[:, None], corr.astype(jnp.float32), lookup))[:, 0, 0]
    report["native_vs_continuous_max_relative"] = float(np.max(np.abs(native / continuous - 1.0)))
    report["per_pixel_mean_score"] = float(np.mean(continuous) / float(np.sum(ops.weight[0] > 0)))

    # Moving the model and moving the image: same score, by construction of |phase| = 1, checked in the code path.
    rng = np.random.default_rng(1)
    shifts = np.zeros((n, 5))
    shifts[:, 3:] = rng.normal(0.0, 1.0, (n, 2))
    model_moved = np.asarray(pose_uq.scores(ops, shifts[:, None]))[:, 0]
    image_moved = np.asarray(pose_uq.scores(pose_uq.recentred(ops, shifts), np.zeros((n, 1, 5))))[:, 0]
    report["model_vs_image_shift_max_relative"] = float(np.max(np.abs(model_moved / image_moved - 1.0)))
    # Native translation: the phase of relax's own image_translations.
    moved = replace(result, poses=[replace(p, offsets_px=p.offsets_px.copy()) for p in result.poses])
    rows = run.halves[0].rows[:n]
    moved.poses[args.body].offsets_px[rows] -= shifts[:, 3:] / run.pixel_size
    native_moved = np.asarray(pose_uq.scores(body_operands(run, moved, args.body, 0, np.arange(n), offset_range_angstrom=offset_range, images=_subtracted(run, result, args.body))[0], np.zeros((n, 1, 5))))[:, 0]
    report["native_translation_vs_local_max_relative"] = float(np.max(np.abs(native_moved / model_moved - 1.0)))

    fisher = pose_uq.expected_fisher(ops)
    whiten = pose_uq.whitening(fisher)
    steps = np.array([0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0])
    gradients, hessians = [], []
    for step in steps:
        _, g, h = pose_uq.local_derivatives(ops, whiten, float(step))
        gradients.append(g)
        hessians.append(h)
    gradients, hessians = np.stack(gradients), np.stack(hessians)
    # Directional finite differences along independent random whitened directions, for the record of step dependence.
    directions = rng.standard_normal((n, 5))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    theta = np.einsum("nij,nj->ni", whiten, directions)
    grid = np.array([-3, -2, -1.5, -1, -.5, -.25, 0, .25, .5, 1, 1.5, 2, 3], dtype=np.float64)
    fine = np.linspace(-0.12, 0.12, 241)  # a sweep of about one interpolation cell at the outer radius
    sections = np.asarray(pose_uq.scores(ops, grid[None, :, None] * theta[:, None]))
    ripple = np.asarray(pose_uq.scores(ops, fine[None, :, None] * theta[:, None]))
    local = pose_uq.polish(ops)
    report["polish"] = dict(stored_minus_analysis_median=float(np.median(local.score_stored - local.score_analysis)),
                            status_counts=np.bincount(local.status).tolist())
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"audit_{args.dataset}_body{args.body + 1}.npz", native=native, continuous=continuous, steps=steps, gradients=gradients,
                        hessians=hessians, grid=grid, sections=sections, fine=fine, ripple=ripple, directions=directions, fisher=fisher, whiten=whiten,
                        whitened_eigenvalues=local.whitened_eigenvalues, fit_ratio=local.fit_ratio)
    (args.out / f"audit_{args.dataset}_body{args.body + 1}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


def _subtracted(run, result, target):
    from relax.multibody import subsets

    return subsets.target_body_images(run, result, target, 0)


def synthetic(args) -> None:
    """Stage C, independent noise realizations of the same poses, Gaussian noise at the weights the score assumes.

    ``--clean model``: the image is the score's own prediction at the stored pose, so map, nuisance, noise and
    interpolation are all exact (cameras and CTFs the fixture's, scales varied, pixel size nonunit). ``--clean
    component``: the image is the simulator's noiseless body (true state volume, true pose, its own interpolation)
    through the same preparation, scored against the run's half map, with errors against the true pose: what map error
    and body deformation add when the noise is still exactly the assumed one."""
    import jax
    import jax.numpy as jnp

    from relax.multibody import pose_uq
    from relax.multibody.subtraction import relion_translate_2d
    from relax.multibody import geometry
    from scripts.multibody_adaptive_subsets import simulated_body_images

    data = args.root / args.dataset
    run, result = load_run(data, args.out)
    offset_range = final_offset_range_angstrom(data, result)
    n = args.particles
    images = None
    if args.clean == "component":
        components, _ = simulated_body_images(run, data)
        particles = run.checkpoint.particles
        consensus = geometry.consensus_matrices(particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(float))
        origins = particles[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(float) / run.pixel_size
        integer = geometry.integer_image_shifts(origins, geometry.projected_centers(consensus, run.bodies[args.body])).astype(np.int64)
        rows = run.halves[0].rows
        images = np.stack([relion_translate_2d(components[args.body][r], integer[r]) for r in rows]).astype(np.float32)
    ops, *_ = body_operands(run, result, args.body, 0, np.arange(n), offset_range_angstrom=offset_range, images=images)
    rng = np.random.default_rng(args.seed)
    scale = np.ones(n)
    if args.clean == "model":
        scale = rng.uniform(0.8, 1.2, n)
        ops = replace(ops, ctf=ops.ctf * jnp.asarray(scale)[:, None])
        clean = pose_uq.predictions(ops)
    else:
        clean = ops.image
    results = []
    for r in range(args.realizations):
        key = jax.random.PRNGKey(args.seed + 1000 + r)
        a, b = jax.random.normal(key, (2, *clean.shape), dtype=jnp.float64)
        noise = (a + 1j * b) / jnp.sqrt(jnp.where(ops.weight > 0, ops.weight, 1.0)) * (ops.weight > 0)
        results.append(polish_all(replace(ops, image=clean + noise), batch=args.batch))
        print("realization", r, flush=True)
    stack = {k: np.stack([x[k] for x in results]) for k in ("correction", "hessian", "fisher", "status", "whitened_eigenvalues", "last_step", "fit_ratio", "cameras")}
    if args.clean == "component":
        true_cameras, true_offsets, _ = true_pose(run, result, data, args.body)
        table = run.halves[0].rows[:n]
        offsets = result.poses[args.body].offsets_px[table]
        errors = []
        for k in range(args.realizations):
            analysis = offsets - stack["correction"][k][:, 3:] / run.pixel_size
            errors.append(pose_errors(stack["cameras"][k], analysis, true_cameras[table], true_offsets[table], ops.consensus, run.pixel_size))
        flat, gauge = remove_gauge(np.concatenate(errors), np.tile(ops.consensus, (args.realizations, 1, 1)))
        stack["error"] = flat.reshape(args.realizations, n, 5)
    else:
        stack["error"], gauge = stack["correction"], None
    name = f"synthetic_{args.dataset}_body{args.body + 1}_{args.clean}"
    np.savez_compressed(args.out / f"{name}.npz", scale=scale, **stack)
    error = stack["error"].reshape(-1, 5)
    keep = stack["status"].reshape(-1) == 0
    hessian = stack["hessian"].reshape(-1, 5, 5)
    fisher = stack["fisher"].reshape(-1, 5, 5)
    report = dict(dataset=args.dataset, body=args.body + 1, clean=args.clean, particles=n, realizations=args.realizations, gauge=gauge,
                  qualified_fraction=float(keep.mean()), status_counts=np.bincount(stack["status"].reshape(-1)).tolist(),
                  observed_hessian=calibration(error[keep], np.linalg.inv(hessian[keep])), expected_fisher=calibration(error[keep], np.linalg.inv(fisher[keep])),
                  all_expected_fisher=calibration(error, np.linalg.inv(fisher)))
    # Spread about each particle's own mean over realizations against the prediction: the noise part alone; the
    # mean offset per particle is the model (map, deformation) part.
    centred = stack["error"] - stack["error"].mean(axis=0)
    predicted = np.mean(np.linalg.inv(fisher).reshape(args.realizations, n, 5, 5), axis=0)
    spread = np.einsum("rni,rnj->nij", centred, centred) / (args.realizations - 1)
    report["within_particle_over_predicted_variance_median"] = np.median(np.diagonal(spread, axis1=1, axis2=2) / np.diagonal(predicted, axis1=1, axis2=2), axis=0).tolist()
    bias = stack["error"].mean(axis=0)
    report["particle_mean_error_over_predicted_sd_rms"] = np.sqrt(np.mean(bias**2 / np.diagonal(predicted, axis1=1, axis2=2), axis=0)).tolist()
    (args.out / f"{name}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


def noise_check(args) -> None:
    """Is the score's noise model the fixture's noise? The simulated noise alone (raw image minus every noiseless
    body), through the same preparation and mask: per-shell ``mean w |N|^2 / 2`` (one when the weights are right), and
    the correlation of neighbouring Fourier pixels the diagonal model sets to zero."""
    import jax.numpy as jnp

    from relax.multibody import geometry
    from relax.multibody.subtraction import relion_translate_2d
    from scripts.multibody_adaptive_subsets import simulated_body_images

    data = args.root / args.dataset
    run, result = load_run(data, args.out)
    offset_range = final_offset_range_angstrom(data, result)
    components, _ = simulated_body_images(run, data)
    particles = run.checkpoint.particles
    consensus = geometry.consensus_matrices(particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(float))
    origins = particles[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(float) / run.pixel_size
    integer = geometry.integer_image_shifts(origins, geometry.projected_centers(consensus, run.bodies[args.body])).astype(np.int64)
    raw = np.asarray(run.dataset.image_source.backend.source.get(np.arange(run.dataset.n_images)))
    rows = run.halves[0].rows
    n = args.particles
    noise = np.stack([relion_translate_2d(raw[r] - sum(c[r] for c in components), integer[r]) for r in rows]).astype(np.float32)
    report = {}
    # The widest RELION mask the box holds, against the body's own mask.
    widest = (run.box_size / 2 - run.width_mask_edge - 1) * run.pixel_size - run.bodies[args.body].max_radius_angstrom
    for label, offset in (("body mask", offset_range), ("box mask", widest)):
        ops, *_ = body_operands(run, result, args.body, 0, np.arange(n), offset_range_angstrom=offset, images=noise)
        w = np.asarray(ops.weight)
        power = np.asarray(jnp.abs(ops.image) ** 2) * w / 2.0
        lattice = np.asarray(ops.lattice)
        radius = np.sqrt(np.sum(lattice**2, axis=1)) * run.box_size
        shells = np.minimum(radius.astype(int), run.box_size // 2)
        valid = w[0] > 0
        per_shell = [float(power[:, (shells == k) & valid].mean()) if np.any((shells == k) & valid) else None for k in range(run.box_size // 2 + 1)]
        report[label] = dict(mean=float(power[:, valid].mean()), per_shell=per_shell)
    (args.out / f"noise_{args.dataset}_body{args.body + 1}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: dict(mean=v["mean"], shells_every_8=v["per_shell"][::8]) for k, v in report.items()}, indent=1), flush=True)


def sandwich(args) -> None:
    """The sandwich covariance ``H^-1 G H^-1`` with ``G`` the score-gradient covariance of white real-space noise
    through the native preparation (mask included), by Monte Carlo, at the analysis poses of an earlier ``fixture``
    run, and its calibration against the same truth errors. ``--noise-variance`` is the raw per-pixel variance; the
    default reads it from the fixture's own simulated noise (a real data set would take it from RELION's spectrum
    divided by the consensus mask's mean square, the ratio the noise check measures)."""
    import jax
    import jax.numpy as jnp

    from relax.multibody import geometry, pose_uq
    from relax.multibody.subtraction import relion_translate_2d
    from scripts.multibody_adaptive_subsets import simulated_body_images

    data = args.root / args.dataset
    run, result = load_run(data, args.out)
    offset_range = final_offset_range_angstrom(data, result)
    merged = dict(np.load(args.input / f"uq_{args.dataset}_body{args.body + 1}_estimated.npz"))
    first = merged["half"] == 0
    n = int(first.sum()) if args.particles is None else min(args.particles, int(first.sum()))
    components, _ = simulated_body_images(run, data)
    raw = np.asarray(run.dataset.image_source.backend.source.get(run.halves[0].rows[:n]))
    variance = float(np.var(raw - sum(c[run.halves[0].rows[:n]] for c in components)))
    ops, *_ = body_operands(run, result, args.body, 0, np.arange(n), offset_range_angstrom=offset_range)
    ops = pose_uq.recentred(ops, merged["correction"][first][:n])
    J = pose_uq.prediction_jacobian(ops)
    rng = np.random.default_rng(args.seed)
    draws = []
    for k in range(args.realizations):
        white = (rng.standard_normal((run.halves[0].rows.size, run.box_size, run.box_size)) * np.sqrt(variance)).astype(np.float32)
        noise_ops, *_ = body_operands(run, result, args.body, 0, np.arange(n), offset_range_angstrom=offset_range, images=white)
        # The fixed-frame prediction and the analysis-pose translation: the noise image needs the same phase.
        noise = noise_ops.image * pose_uq.translation_phases(-jnp.asarray(merged["correction"][first][:n, 3:]) / ops.pixel_size, ops.lattice)
        draws.append(pose_uq.score_gradient_of_noise(J, ops.weight, noise))
    draws = np.stack(draws)
    G = np.einsum("kna,knb->nab", draws, draws) / len(draws)
    H, F = merged["hessian"][first][:n], merged["fisher"][first][:n]
    Hinv = np.linalg.inv(H)
    sandwich_cov = Hinv @ G @ Hinv
    error, gauge = remove_gauge(merged["error_analysis"], merged["consensus"])
    error = error[first][:n]
    keep = merged["status"][first][:n] == 0
    report = dict(dataset=args.dataset, body=args.body + 1, n=n, draws=args.realizations, raw_noise_variance=variance,
                  G_over_F_trace_median=float(np.median(np.trace(G, axis1=1, axis2=2) / np.trace(F, axis1=1, axis2=2))),
                  G_over_F_eigen_median=np.median(np.linalg.eigvalsh(np.linalg.solve(np.linalg.cholesky(F), np.swapaxes(np.linalg.solve(np.linalg.cholesky(F), G), 1, 2))), axis=0).tolist(),
                  raw_hessian=calibration(error[keep], Hinv[keep]), sandwich=calibration(error[keep], sandwich_cov[keep]))
    np.savez_compressed(args.out / f"sandwich_{args.dataset}_body{args.body + 1}.npz", G=G, sandwich=sandwich_cov, error=error, keep=keep)
    (args.out / f"sandwich_{args.dataset}_body{args.body + 1}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


def fixture_uq(args) -> None:
    """Stage C second arm and stage D input: the local analysis of a fixture's particles, against the simulated truth.

    ``--nuisance estimated`` subtracts the half maps at fitted poses (what a real data set allows), ``exact`` subtracts
    the noiseless simulated other body. Both keep each image's original noise and the run's estimated noise spectrum."""
    from relax.multibody import geometry, subsets
    from relax.multibody.pose_uq import covariance_from_hessian
    from scripts.multibody_adaptive_subsets import exact_nuisance_images, require_unit_true_contrast, simulated_body_images

    data = args.root / args.dataset
    run, result = load_run(data, args.out)
    offset_range = final_offset_range_angstrom(data, result)
    body = run.bodies[args.body]
    true_cameras, true_offsets, _ = true_pose(run, result, data, args.body)
    exact = None
    if args.nuisance == "exact":
        require_unit_true_contrast(data)
        components, _ = simulated_body_images(run, data)
        exact = exact_nuisance_images(run, data, args.body, components)
    out = {}
    for half in range(2):
        particles = run.halves[half]
        count = particles.rows.size if args.particles is None else min(args.particles, particles.rows.size)
        images = None if exact is None else exact[particles.rows]
        started = time.time()
        ops, *_ = body_operands(run, result, args.body, half, np.arange(count), offset_range_angstrom=offset_range, images=images)
        prepared = time.time()
        local = polish_all(ops, batch=args.batch)
        finished = time.time()
        table = particles.rows[:count]
        offsets = result.poses[args.body].offsets_px[table]
        analysis = offsets - local["correction"][:, 3:] / run.pixel_size
        local["table_rows"], local["half"] = table, np.full(count, half)
        local["stored_cameras"] = ops.cameras
        local["consensus"] = ops.consensus
        local["stored_offsets_px"], local["analysis_offsets_px"] = offsets, analysis
        local["error_stored"] = pose_errors(ops.cameras, offsets, true_cameras[table], true_offsets[table], ops.consensus, run.pixel_size)
        local["error_analysis"] = pose_errors(local["cameras"], analysis, true_cameras[table], true_offsets[table], ops.consensus, run.pixel_size)
        local["signal_power"] = np.asarray(_signal_power(ops))
        local["seconds"] = np.full(count, (finished - prepared) / count)
        out.setdefault("prepare_seconds", []).append(prepared - started)
        out.setdefault("polish_seconds", []).append(finished - prepared)
        out.setdefault("parts", []).append(local)
    merged = {k: np.concatenate([p[k] for p in out["parts"]]) for k in out["parts"][0]}
    name = f"uq_{args.dataset}_body{args.body + 1}_{args.nuisance}"
    np.savez_compressed(args.out / f"{name}.npz", **merged)
    report = fixture_report(merged, args)
    report.update(prepare_seconds=out["prepare_seconds"], polish_seconds=out["polish_seconds"])
    (args.out / f"{name}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


def _signal_power(ops):
    import jax.numpy as jnp

    from relax.multibody import pose_uq

    clean = pose_uq.predictions(ops)
    return jnp.sum(ops.weight * jnp.abs(clean) ** 2, axis=1)


def fixture_report(merged: dict, args) -> dict:
    consensus = merged["consensus"]
    report = dict(dataset=args.dataset, body=args.body + 1, nuisance=args.nuisance, n=len(consensus),
                  status_counts=np.bincount(merged["status"]).tolist(), qualified=float(np.mean(merged["status"] == 0)),
                  median_score_drop=float(np.median(merged["score_stored"] - merged["score_analysis"])),
                  median_correction_deg=float(np.rad2deg(np.median(np.linalg.norm(merged["correction"][:, :3], axis=1)))),
                  median_correction_angstrom=float(np.median(np.linalg.norm(merged["correction"][:, 3:], axis=1))))
    keep = merged["status"] == 0
    hessian, fisher = merged["hessian"], merged["fisher"]
    predicted = np.linalg.inv(fisher)
    report["predicted_sd_median"] = dict(rotation_deg=np.rad2deg(np.median(np.sqrt(np.diagonal(predicted, axis1=1, axis2=2)[:, :3]), axis=0)).tolist(),
                                         shift_angstrom=np.median(np.sqrt(np.diagonal(predicted, axis1=1, axis2=2)[:, 3:]), axis=0).tolist())
    for name in ("error_stored", "error_analysis"):
        error, gauge = remove_gauge(merged[name], consensus)
        gross = np.linalg.norm(error[:, :3], axis=1) > np.deg2rad(5.0)
        robust = 1.4826 * np.median(np.abs(error), axis=0)
        report[name] = dict(gauge=gauge, gross_over_5deg=float(gross.mean()), robust_sd_rotation_deg=np.rad2deg(robust[:3]).tolist(), robust_sd_shift_angstrom=robust[3:].tolist(),
                            observed_hessian_qualified=calibration(error[keep], np.linalg.inv(hessian[keep])),
                            expected_fisher_all=calibration(error, predicted), expected_fisher_not_gross=calibration(error[~gross], predicted[~gross]))
    return report


def pose_table(run, merged: dict, target: int, *, use_stored: bool = False) -> dict:
    """The observation table of a body in particle-table order: ``y``, ``P``, ``S``, ``R`` and the records around them."""
    from relax.multibody import geometry
    from relax.multibody import pose_uq as uq
    from relax.multibody.pose_neighborhoods import mask_moments

    order = np.argsort(merged["table_rows"])
    sorted_fields = {k: v[order] for k, v in merged.items()}
    body = run.bodies[target]
    moments = mask_moments(body.mask, body.com, run.pixel_size)
    root = uq.covariance_root(moments.covariance)
    consensus = sorted_fields["consensus"]
    cameras = sorted_fields["stored_cameras" if use_stored else "cameras"]
    offsets = sorted_fields["stored_offsets_px" if use_stored else "analysis_offsets_px"]
    particles = run.checkpoint.particles
    origins = particles[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(float)[sorted_fields["table_rows"]] / run.pixel_size
    projected = geometry.projected_centers(consensus, body)
    fractional = origins - projected - geometry.integer_image_shifts(origins, projected)
    shifts = run.pixel_size * (fractional - offsets)
    rotations = np.swapaxes(consensus, 1, 2) @ cameras
    P = uq.camera_projection(consensus)
    y = uq.observed_features(rotations, shifts, P, root, moments.centroid)
    hessian = sorted_fields["hessian"]
    usable = sorted_fields["status"] == 0
    S = np.full_like(hessian, np.nan)
    S[usable] = uq.covariance_from_hessian(hessian[usable])
    R = np.full((len(y), 11, 11), np.nan)
    R[usable] = uq.propagate(uq.feature_jacobian(rotations[usable], P[usable], root, moments.centroid), S[usable])
    return dict(fields=sorted_fields, y=y, P=P, S=S, R=R, usable=usable, rotations=rotations, shifts=shifts, root=root, centroid=moments.centroid,
                covariance=moments.covariance)


RANKS = (2, 4, 6, 10)
FLOORS = (1e-4, 1e-3, 1e-2, 1e-1)


def held_out(fit, y, P, R) -> dict:
    from relax.multibody import pose_ppca as pp

    report = dict(log_likelihood_per_particle=pp.log_likelihood(y, P, R, fit) / len(y))
    shifts = [(f"body{b + 1}_shift_{axis}", 11 * b + 9 + k) for b in range(fit.bodies) for k, axis in enumerate("xy")]
    for name, coordinate in shifts:
        mean, variance = pp.withheld_prediction(y, P, R, fit, coordinate)
        residual = y[:, coordinate] - mean
        baseline = y[:, coordinate] - (pp.operators(P, fit.bodies) @ fit.mean)[:, coordinate]
        report[name] = dict(rmse=float(np.sqrt(np.mean(residual**2))), robust_rmse=float(1.4826 * np.median(np.abs(residual))),
                            mean_only_robust_rmse=float(1.4826 * np.median(np.abs(baseline))),
                            z_sd=float(np.std(residual / np.sqrt(variance))), z_robust_sd=float(1.4826 * np.median(np.abs(residual / np.sqrt(variance)))))
    return report


def fit_body(table: dict, seed: int) -> dict:
    """Ranks and floors on the fit split, floor chosen by held-out likelihood alone: truth plays no part."""
    from relax.multibody import pose_ppca as pp

    usable = np.flatnonzero(table["usable"])
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(usable)
    validation, training = np.sort(permuted[: len(usable) // 5]), np.sort(permuted[len(usable) // 5 :])
    y, P, R = table["y"], table["P"], table["R"]
    sweep, fits = {}, {}
    for rank in RANKS:
        for floor in FLOORS:
            started = time.time()
            fit = pp.fit_projected_ppca(y[training], P[training], R[training], rank, floor=floor, tolerance=1e-6)
            fits[(rank, floor)] = fit
            sweep[f"q{rank}_floor{floor:g}"] = dict(
                rank=rank, floor=floor, tau2=fit.tau2, at_floor=fit.tau2_at_floor, effective_rank=fit.effective_rank, iterations=len(fit.log_likelihood),
                converged=fit.converged, seconds=time.time() - started, amplitudes=fit.amplitudes.tolist(),
                train_per_particle=fit.log_likelihood[-1] / len(training), validation=held_out(fit, y[validation], P[validation], R[validation]),
            )
            print(f"q={rank} floor={floor:g} tau2={fit.tau2:.4g} it={len(fit.log_likelihood)} val={sweep[f'q{rank}_floor{floor:g}']['validation']['log_likelihood_per_particle']:.3f}", flush=True)
    # The validation likelihood rises without limit as the floor falls: fitted rotation features lie on SO(3), so the
    # six normal directions R leaves empty are explained by tau alone. The floor is therefore chosen by the withheld
    # shift prediction on the validation split (truth free), the depth-relevant quantity; ties go to the larger floor.
    def shift_error(rank, floor):
        validation = sweep[f"q{rank}_floor{floor:g}"]["validation"]
        return float(np.mean([v["robust_rmse"] for k, v in validation.items() if "shift" in k]))

    best = {rank: min(sorted(FLOORS, reverse=True), key=lambda f: round(shift_error(rank, f), 4)) for rank in RANKS}
    return dict(sweep=sweep, fits=fits, best_floor=best, training=training, validation=validation)


def zero_depth_pca(table: dict, rank: int) -> np.ndarray:
    """Ordinary PCA of the matrix features with depth filled by zero, the baseline: principal coordinates ``[N, rank]``."""
    lifted = np.concatenate([table["y"][:, :9], np.einsum("nji,nj->ni", table["P"], table["y"][:, 9:])], axis=1)
    centred = lifted - lifted.mean(axis=0)
    _, values, right = np.linalg.svd(centred[table["usable"]], full_matrices=False)
    return centred @ right[:rank].T


def fixture_labels(run, data: Path) -> dict:
    from scripts.multibody_adaptive_subsets import state_rows

    record = json.loads((data / "body_motion.json").read_text())
    names = np.asarray(run.checkpoint.particles["rlnImageName"])
    states = state_rows(data, names)
    table = record["states"]
    body2 = np.asarray([(s["body2_twist"], s["body2_shift"], s["deformation"]) for s in table])
    return dict(names=names, states=states, record=record, level=np.asarray([s["level"] for s in table])[states], labels=np.unique(body2, axis=0, return_inverse=True)[1].ravel()[states],
                body1=np.unique(np.asarray([s["body1_twist"] for s in table]), return_inverse=True)[1].ravel()[states])


def target_curves(scores_of, labels, states, names, dataset: str) -> dict:
    """The fixture's target curves of an ordering, as ``run_pose_ordering_comparison`` scores them."""
    from relax.multibody import subsets
    from scripts.run_pose_ordering_comparison import TARGETS, TARGETS_10K, curve_sizes, score_target

    report = {}
    for state, image in TARGETS.get(dataset, TARGETS_10K):
        row = subsets.target_row(names, f"{image}@particles.128.mrcs")
        nearby = np.flatnonzero(states == states[row])[:20]
        sizes = curve_sizes(len(names))
        curves = [score_target(scores_of(int(r)), labels, int(r), tuple(sizes.tolist())) for r in nearby]
        report[str(state)] = dict(curve_sizes=sizes.tolist(), curves_mean_first_20={k: np.mean([c[k] for c in curves], axis=0).round(4).tolist() for k in curves[0]})
    return report


def joint_table(tables: list) -> dict:
    """All bodies at once: features concatenated, ``R_i`` block diagonal (independent body errors given the consensus)."""
    n, bodies = len(tables[0]["y"]), len(tables)
    R = np.zeros((n, 11 * bodies, 11 * bodies))
    for b, t in enumerate(tables):
        R[:, 11 * b : 11 * (b + 1), 11 * b : 11 * (b + 1)] = t["R"]
    return dict(y=np.concatenate([t["y"] for t in tables], axis=1), P=tables[0]["P"], R=R, usable=np.all([t["usable"] for t in tables], axis=0))


def true_observations(run, data: Path, labels: dict, target: int, root: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(y, P, Q)`` of a body from the simulated motion and cameras: no consensus error, no pose error."""
    from recovar import utils

    from relax.multibody import pose_uq as uq
    from relax.multibody.pose_neighborhoods import mask_moments
    from scripts.run_pose_ordering_comparison import true_camera_features

    features = true_camera_features(run, data, labels["record"], target, labels["states"], labels["names"])
    moments = mask_moments(run.bodies[target].mask, run.bodies[target].com, run.pixel_size)
    values, vectors = np.linalg.eigh(moments.covariance)
    Q = features.shape_factors @ ((vectors / np.sqrt(values)) @ vectors.T)
    rows = np.asarray([int(n.split("@")[0]) - 1 for n in labels["names"]])
    P = uq.camera_projection(np.asarray(utils.pickle_load(data / "poses.pkl")[0])[rows])
    return np.concatenate([(Q @ root).reshape(-1, 9), np.einsum("nij,nj->ni", P, features.points)], axis=1), P, Q


def explained(coordinates: np.ndarray, truth: np.ndarray, usable: np.ndarray) -> float:
    """R^2 of a linear prediction of a true scalar from coordinates, on the usable particles (evaluation only)."""
    x = coordinates[usable] - coordinates[usable].mean(axis=0)
    t = truth[usable] - truth[usable].mean()
    if np.var(t) == 0.0:
        return float("nan")
    return float(1.0 - np.sum((t - x @ np.linalg.lstsq(x, t, rcond=None)[0]) ** 2) / np.sum(t**2))


def cross_body_tests(tables: list, fits: list, seed: int) -> dict:
    """Do the two bodies' errors correlate? Truth free: canonical correlations of the per-body PPCA residuals' first
    five whitened coordinates, against a permutation of body 2. With truth: the 5 x 5 correlation of the pose errors."""
    from relax.multibody import pose_ppca as pp

    keep = tables[0]["usable"] & tables[1]["usable"]
    parts = []
    for table, fit in zip(tables, fits):
        posterior = pp.latent_posterior(table["y"][keep], table["P"][keep], table["R"][keep], fit)
        A = pp.operators(table["P"][keep])
        parts.append(table["y"][keep] - np.einsum("nij,nj->ni", A, posterior.structured_mean))

    def canonical(a, b):
        a, b = a - a.mean(axis=0), b - b.mean(axis=0)
        ua = np.linalg.svd(a, full_matrices=False)[0][:, :6]
        ub = np.linalg.svd(b, full_matrices=False)[0][:, :6]
        return np.linalg.svd(ua.T @ ub, compute_uv=False)

    # Rotation-feature residuals: they read the same molecular frame in both bodies, so a shared consensus error shows.
    observed = canonical(parts[0][:, :9], parts[1][:, :9])
    null = np.stack([canonical(parts[0][:, :9], parts[1][np.random.default_rng(seed + k).permutation(len(parts[1]))][:, :9]) for k in range(20)])
    errors = []
    for table in tables:
        error, _ = remove_gauge(table["fields"]["error_analysis"], table["fields"]["consensus"])
        errors.append(error[keep])
    gross = (np.linalg.norm(errors[0][:, :3], axis=1) < np.deg2rad(5.0)) & (np.linalg.norm(errors[1][:, :3], axis=1) < np.deg2rad(5.0))
    matrix = np.corrcoef(errors[0][gross].T, errors[1][gross].T)[:5, 5:]
    # Cross-covariance relative to the diagonal R the block model assumes zero.
    return dict(residual_canonical_correlations=observed.tolist(), permuted_mean=null.mean(axis=0).tolist(), permuted_max=null.max(axis=0).tolist(),
                pose_error_correlation_image_level=matrix.tolist(), n=int(keep.sum()))


def molecular_error(table: dict, true_rotations: np.ndarray) -> np.ndarray:
    """``[n_usable, 3]`` molecular-frame rotation error (radians) of Q = A^T B against the true motion, after a constant
    rotation on both sides and a median (truth-side gauge)."""
    from relax.multibody.pose_uq import log_so3

    usable = table["usable"]
    fitted, truth = table["rotations"][usable], true_rotations[usable]
    left, right = np.eye(3), np.eye(3)
    for _ in range(20):
        u, _, vt = np.linalg.svd(np.sum(fitted @ right.T @ np.swapaxes(truth, 1, 2), axis=0))
        left = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        u, _, vt = np.linalg.svd(np.sum(np.swapaxes(truth, 1, 2) @ left.T @ fitted, axis=0))
        right = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    error = log_so3(fitted @ np.swapaxes(left @ truth @ right, 1, 2))
    return error - np.median(error, axis=0)


def twist(args) -> None:
    """Is the molecular-frame error the consensus following the body motions? Per axis, its correlation with each true
    state variable and the error left after a linear fit on all of them (with intercept)."""
    data = args.root / args.dataset
    run, _ = load_run(data, args.out)
    labels = fixture_labels(run, data)
    states = labels["record"]["states"]
    variables = {k: np.asarray([s[k] for s in states], dtype=float)[labels["states"]] for k in ("body1_twist", "body2_twist", "body2_shift", "deformation", "level")}
    report = {}
    arrays = {}
    for b in range(len(run.bodies)):
        merged = dict(np.load(args.input / f"uq_{args.dataset}_body{b + 1}_estimated.npz"))
        table = pose_table(run, merged, b)
        _, _, true_Q = true_observations(run, data, labels, b, table["root"])
        error = np.rad2deg(molecular_error(table, true_Q))
        usable = table["usable"]
        values = {k: v[usable] for k, v in variables.items() if np.var(v) > 0}
        design = np.column_stack([np.ones(int(usable.sum()))] + list(values.values()))
        fitted = design @ np.linalg.lstsq(design, error, rcond=None)[0]
        residual = error - fitted
        robust = lambda x: (1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)).round(3).tolist()
        report[str(b + 1)] = dict(
            n=int(usable.sum()), variables=list(values), correlation={k: [float(np.corrcoef(error[:, a], v)[0, 1]) for a in range(3)] for k, v in values.items()},
            r2_per_axis=[float(1 - residual[:, a].var() / error[:, a].var()) for a in range(3)], robust_error_deg=robust(error), robust_residual_deg=robust(residual),
            sd_error_deg=error.std(axis=0).round(3).tolist(), sd_residual_deg=residual.std(axis=0).round(3).tolist(),
            mean_error_by_body1_twist={str(t): error[values["body1_twist"] == t].mean(axis=0).round(3).tolist() for t in np.unique(values.get("body1_twist", [0.0]))} if "body1_twist" in values else None,
        )
        arrays[f"body{b + 1}_error_deg"], arrays[f"body{b + 1}_usable"] = error, usable
    arrays.update({k: v for k, v in variables.items()})
    np.savez_compressed(args.out / f"twist_{args.dataset}.npz", **arrays)
    (args.out / f"twist_{args.dataset}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


def consensus_gap(table: dict, true_rotations: np.ndarray) -> dict:
    """The uncertainty S covers the camera-level pose. Against the molecular-frame rotation Q = A^T B it can only cover
    the part free of consensus error: the 3D rotation error against S's block, before and after a median gauge."""
    from relax.multibody.pose_uq import log_so3

    usable = table["usable"]
    fitted, truth = table["rotations"][usable], true_rotations[usable]
    # The run's map frame and body orientation differ from the simulation's by constant rotations on both sides,
    # Q_fit ~ C_l Q_true C_r; a median of log(Q_fit Q_true^T) removes only a left constant (truth-side gauge).
    left, right = np.eye(3), np.eye(3)
    for _ in range(20):
        u, _, vt = np.linalg.svd(np.sum(fitted @ right.T @ np.swapaxes(truth, 1, 2), axis=0))
        left = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        u, _, vt = np.linalg.svd(np.sum(np.swapaxes(truth, 1, 2) @ left.T @ fitted, axis=0))
        right = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    error = log_so3(fitted @ np.swapaxes(left @ truth @ right, 1, 2))
    error = error - np.median(error, axis=0)
    S = table["S"][usable][:, :3, :3]
    chi2 = np.sum(error * np.linalg.solve(S, error[:, :, None])[:, :, 0], axis=1)
    camera = remove_gauge(table["fields"]["error_analysis"], table["fields"]["consensus"])[0][usable, :3]
    camera_chi2 = np.sum(camera * np.linalg.solve(S, camera[:, :, None])[:, :, 0], axis=1)
    return dict(molecular_frame_median_chi2_over_2p37=float(np.median(chi2) / 2.366), molecular_frame_robust_error_deg=np.rad2deg(1.4826 * np.median(np.abs(error), axis=0)).tolist(),
                molecular_frame_coverage_90=float(np.mean(chi2 < 6.2514)), camera_median_chi2_over_2p37=float(np.median(camera_chi2) / 2.366),
                camera_robust_error_deg=np.rad2deg(1.4826 * np.median(np.abs(camera), axis=0)).tolist(), camera_coverage_90=float(np.mean(camera_chi2 < 6.2514)),
                predicted_sd_deg=np.rad2deg(np.sqrt(np.median(np.diagonal(S, axis1=1, axis2=2), axis=0))).tolist())


def gap(args) -> None:
    """The consensus-error measurement alone, from the fixture tables: no PPCA."""
    data = args.root / args.dataset
    run, _ = load_run(data, args.out)
    labels = fixture_labels(run, data)
    report = {}
    for b in range(len(run.bodies)):
        merged = dict(np.load(args.input / f"uq_{args.dataset}_body{b + 1}_estimated.npz"))
        table = pose_table(run, merged, b)
        _, _, true_Q = true_observations(run, data, labels, b, table["root"])
        report[str(b + 1)] = consensus_gap(table, true_Q)
    (args.out / f"gap_{args.dataset}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


def embed(args) -> None:
    """Stage D: per-body controls and the joint fit, MAP coordinates, neighborhood curves, and the assumption tests."""
    from relax.multibody import pose_ppca as pp
    from relax.multibody import pose_uq as uq

    data = args.root / args.dataset
    run, _ = load_run(data, args.out)
    merged = [dict(np.load(args.input / f"uq_{args.dataset}_body{b + 1}_estimated.npz")) for b in range(len(run.bodies))]
    tables = [pose_table(run, m, b) for b, m in enumerate(merged)]
    labels = fixture_labels(run, data)
    joint = joint_table(tables)
    truth_values = dict(body2_level=labels["level"], body1_twist=np.asarray([s["body1_twist"] for s in labels["record"]["states"]])[labels["states"]])
    report = dict(dataset=args.dataset, n=len(joint["y"]), usable_per_body=[int(t["usable"].sum()) for t in tables], usable_joint=int(joint["usable"].sum()),
                  floors=list(FLOORS), ranks=list(RANKS), per_body={}, curves={}, explained={})
    arrays = dict(table_rows=tables[0]["fields"]["table_rows"], half=tables[0]["fields"]["half"], image_names=labels["names"], P=joint["P"], y=joint["y"], R=joint["R"],
                  level=labels["level"], states=labels["states"], labels=labels["labels"], body1=labels["body1"], body1_twist=truth_values["body1_twist"])
    per_body_fits = []
    orderings = {0: {}, 1: {}}
    for b, table in enumerate(tables):
        started = time.time()
        fitted = fit_body(table, args.seed)
        best = fitted["fits"][(10, fitted["best_floor"][10])]
        per_body_fits.append(best)
        report["per_body"][str(b + 1)] = dict(sweep=fitted["sweep"], best_floor={str(k): v for k, v in fitted["best_floor"].items()}, seconds=time.time() - started, consensus_gap=None)
        usable = table["usable"]
        posterior = pp.latent_posterior(table["y"][usable], table["P"][usable], table["R"][usable], best)
        principal = np.full((len(usable), 10), np.nan)
        principal[usable] = posterior.principal(best)[0]
        orderings[b]["per-body PPCA q10"] = principal
        report["explained"][f"per_body_{b + 1}"] = {k: explained(principal, v, usable) for k, v in truth_values.items()}
        arrays.update({f"body{b + 1}_{k}": v for k, v in dict(
            status=table["fields"]["status"], S=table["S"], stored_cameras=table["fields"]["stored_cameras"], analysis_cameras=table["fields"]["cameras"], stored_offsets_px=table["fields"]["stored_offsets_px"],
            analysis_offsets_px=table["fields"]["analysis_offsets_px"], W=best.W, mean=best.mean, tau2=best.tau2, principal=principal, latent_covariance=posterior.covariance,
            signal_power=table["fields"]["signal_power"], pca=zero_depth_pca(table, 10)).items()})
        orderings[b]["zero-depth PCA 10"] = arrays[f"body{b + 1}_pca"]
        true_y, true_P, true_Q = true_observations(run, data, labels, b, table["root"])
        report["per_body"][str(b + 1)]["consensus_gap"] = consensus_gap(table, true_Q)
        table["true_y"], table["true_P"] = true_y, true_P
        # Finite-angle check of the propagated covariance on a sample.
        sample = np.flatnonzero(usable)[:: max(1, int(usable.sum()) // 200)]
        bias, spread = uq.finite_angle_moments(table["rotations"][sample], table["shifts"][sample], table["P"][sample], table["root"], table["centroid"], table["S"][sample], seed=args.seed)
        R = table["R"][sample]
        left = np.linalg.svd(uq.feature_jacobian(table["rotations"][sample], table["P"][sample], table["root"], table["centroid"]), full_matrices=True)[0]
        tangent, normal = left[:, :, :5], left[:, :, 5:]
        report["per_body"][str(b + 1)]["finite_angle"] = dict(
            tangent_variance_ratio_median=float(np.median(np.trace(np.swapaxes(tangent, 1, 2) @ spread @ tangent, axis1=1, axis2=2) / np.trace(np.swapaxes(tangent, 1, 2) @ R @ tangent, axis1=1, axis2=2))),
            normal_variance_median=float(np.median(np.trace(np.swapaxes(normal, 1, 2) @ spread @ normal, axis1=1, axis2=2))), tangent_variance_median=float(np.median(np.trace(R, axis1=1, axis2=2))),
            bias_norm_median=float(np.median(np.linalg.norm(bias, axis=1))), bias_normal_median=float(np.median(np.linalg.norm(np.einsum("nji,nj->ni", normal, bias), axis=1))))
        arrays.update({f"body{b + 1}_finite_{k}": v for k, v in dict(bias=bias, spread=spread, R=R, sample=sample).items()})
    report["cross_body"] = cross_body_tests(tables, per_body_fits, args.seed)
    print(json.dumps(report["cross_body"]), flush=True)

    started = time.time()
    fitted = fit_body(joint, args.seed)
    report["joint"] = dict(sweep=fitted["sweep"], best_floor={str(k): v for k, v in fitted["best_floor"].items()}, seconds=time.time() - started)
    usable = joint["usable"]
    decoded = {}
    for rank in RANKS:
        fit = fitted["fits"][(rank, fitted["best_floor"][rank])]
        posterior = pp.latent_posterior(joint["y"][usable], joint["P"][usable], joint["R"][usable], fit)
        principal = np.full((len(usable), rank), np.nan)
        z = np.full((len(usable), rank), np.nan)
        principal[usable], z[usable] = posterior.principal(fit)[0], posterior.z_map
        report["joint"].setdefault("posterior_identity_max", {})[str(rank)] = float(np.max(np.abs(np.linalg.inv(np.eye(rank) + posterior.information) - posterior.covariance)))
        for b in (0, 1):
            orderings[b][f"joint q{rank} whole latent"] = principal
            orderings[b][f"joint q{rank} decoded body {b + 1}"] = z @ fit.body_rows(b).T
            decoded[(rank, b)] = z @ fit.body_rows(b).T
            report["explained"][f"joint_q{rank}_decoded_body{b + 1}"] = {k: explained(decoded[(rank, b)], v, usable) for k, v in truth_values.items()}
        report["explained"][f"joint_q{rank}_whole"] = {k: explained(principal, v, usable) for k, v in truth_values.items()}
        amplitudes = np.linalg.norm(fit.W[:12], axis=0), np.linalg.norm(fit.W[12:], axis=0)
        report["joint"].setdefault("column_body_amplitudes", {})[str(rank)] = [a.tolist() for a in amplitudes]
        report["joint"].setdefault("latent_truth_correlation", {})[str(rank)] = {k: [float(np.corrcoef(principal[usable, c], v[usable])[0, 1]) if np.var(v) > 0 else None for c in range(rank)] for k, v in truth_values.items()}
        if rank == 10:
            arrays.update(joint_mean=fit.mean, joint_W=fit.W, joint_tau2=fit.tau2, joint_floor=fit.floor, joint_log_likelihood=fit.log_likelihood, joint_z_map=z, joint_principal=principal,
                          joint_latent_covariance=posterior.covariance, joint_principal_covariance=posterior.principal(fit)[1], joint_information=posterior.information, joint_linear=posterior.linear,
                          joint_psi=posterior.psi, joint_structured_mean=posterior.structured_mean, joint_feature_mean=posterior.feature_mean, joint_amplitudes=fit.amplitudes, joint_usable=usable)

    # The perfect-pose control: the joint model on true features.
    true_y = np.concatenate([t["true_y"] for t in tables], axis=1)
    true_P = tables[0]["true_P"]
    zero = np.zeros((len(true_y), 22, 22))
    perfect = pp.fit_projected_ppca(true_y, true_P, zero, 10, floor=1e-4, tolerance=1e-6)
    posterior = pp.latent_posterior(true_y, true_P, zero, perfect)
    perfect_z = posterior.z_map
    report["true_pose_joint"] = dict(amplitudes=perfect.amplitudes.tolist(), tau2=perfect.tau2)
    arrays["true_joint_principal"] = posterior.principal(perfect)[0]
    true_orderings = {b: {"joint q10 decoded body, true poses": perfect_z @ perfect.body_rows(b).T, "joint q10 whole latent, true poses": arrays["true_joint_principal"]} for b in (0, 1)}

    def distances(coordinates: dict, target: int) -> dict:
        return {k: np.nan_to_num(np.linalg.norm(v - v[target], axis=1), nan=np.inf) for k, v in coordinates.items()}

    state_labels = {1: labels["labels"], 0: labels["body1"]}
    for b in (0, 1):
        report["curves"][str(b + 1)] = target_curves(lambda r: distances(orderings[b], r), state_labels[b], labels["states"], labels["names"], args.dataset)
        report["curves"][f"{b + 1}_true"] = target_curves(lambda r: distances(true_orderings[b], r), state_labels[b], labels["states"], labels["names"], args.dataset)
    out = f"embed_{args.dataset}"
    np.savez_compressed(args.out / f"{out}.npz", **arrays)
    (args.out / f"{out}.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ("usable_per_body", "usable_joint", "explained", "cross_body")}, indent=1, default=float), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, function in (("audit", audit), ("synthetic", synthetic), ("fixture", fixture_uq), ("embed", embed), ("noise", noise_check), ("sandwich", sandwich), ("gap", gap), ("twist", twist)):
        a = sub.add_parser(name)
        a.set_defaults(function=function)
        a.add_argument("--root", required=True, type=Path)
        a.add_argument("--out", required=True, type=Path)
        a.add_argument("--body", type=int, default=1)
        a.add_argument("--particles", type=int, default=None if name == "fixture" else 64)
        a.add_argument("--batch", type=int, default=128)
        a.add_argument("--seed", type=int, default=20261102)
        a.add_argument("--realizations", type=int, default=16)
        a.add_argument("--clean", choices=("model", "component"), default="model")
        a.add_argument("--input", type=Path, help="embed: where the fixture stage wrote its tables; default --out")
        a.add_argument("--nuisance", choices=("estimated", "exact"), default="estimated")
        a.add_argument("dataset")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.input = args.input or args.out
    logging.basicConfig(level=logging.WARNING)
    args.function(args)


if __name__ == "__main__":
    main()
