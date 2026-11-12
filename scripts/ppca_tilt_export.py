"""Per-particle poses and latents of a subtomogram `relax ppca_initial_model --ios` run, from any checkpoint.

    python scripts/ppca_tilt_export.py RUN_DIR [--checkpoint RUN_DIR/checkpoint_NNNN.npz] [--out FILE.npz]

The run writes per-particle latents only after its last update (embeddings.npz). This pass gives them for any
checkpoint: with the checkpoint's model, noise and stage (Fourier radius, HEALPix order, shift grid), it scores every
particle on the stage's full pose grid and keeps

- the top pose: ``rotations`` (P, 3, 3), ``eulers_deg`` (P, 3, RELION rot / tilt / psi of those matrices),
  ``translations_px`` / ``translations_ang`` (P, 3, the 3D shift in the tomogram frame);
- ``pmax`` (the top pose's posterior probability) and ``llr`` (log evidence ratio per image, a junk score);
- ``z`` (P, q): the posterior latent mean, the PPCA embedding (``embeddings.npz``'s ``z`` for the last checkpoint);
- ``particle_ids`` (row of the particle STAR the run read) and ``particle_names`` (``rlnTomoParticleName``).

``--star FILE`` also writes the run's input particle STAR (all its blocks) with each particle's ``rlnAngleRot/Tilt/Psi``,
``rlnOriginX/Y/ZAngst`` (the 3D shift times the image pixel size, the convention relax and RELION read back as the
particle's offset) and ``rlnMaxValueProbDistribution`` (pmax) set from this pass.

The rotations are in the frame of the run's own map (ab initio: arbitrary up to one global rotation and hand).
Run it on a GPU node in relax's environment; it reads the run's run.json for the input and config.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="default: the run's latest checkpoint_NNNN.npz")
    parser.add_argument("--out", type=Path, help="default: RUN_DIR/export_NNNN.npz")
    parser.add_argument("--star", type=Path, help="also write the input particle STAR with these poses")
    parser.add_argument("--preread-images", default="auto", choices=("auto", "on", "off"))
    a = parser.parse_args(argv)

    import starfile
    from recovar.reconstruction.noise import make_radial_noise

    from relax.commands.ppca_initial_model import load_tilt_training
    from relax.ppca_initial_model import checkpoint
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.iteration_loop import _rotation_grid
    from relax.ppca_initial_model.tomo import free_low_shells, load_tilt_tile, tilt_tiles
    from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
    from relax.ppca_refinement.full_row_stream import (
        accumulate_full_row_tiles,
        full_row_tile_embeddings,
        prepare_full_row_stream,
    )
    from relax.ppca_refinement.full_row_tile_plan import plan_tile_images
    from relax.sampling import grids
    from relax.sampling.healpix import euler_matrix_to_angles
    from relax.star.tomo_input import read_optimisation_set

    t0 = time.monotonic()
    run = json.loads((a.run_dir / "run.json").read_text())
    fields = {f.name for f in dataclasses.fields(Config)}
    raw = {k: v for k, v in run["config"].items() if k in fields}
    raw["stages"] = tuple(tuple(s) for s in raw["stages"])
    if raw.get("shift_schedule"):
        raw["shift_schedule"] = tuple(tuple(e) for e in raw["shift_schedule"])
    config = Config(**raw)
    ckpt = a.checkpoint or sorted(a.run_dir.glob("checkpoint_*.npz"))[-1]
    with np.load(ckpt, allow_pickle=False) as arrays:
        identity = json.loads(str(arrays["metadata"]))["identity"]
    state = checkpoint.load(ckpt, config, identity)
    ios = run["input"]["ios"]
    particles = load_tilt_training(ios, a.run_dir, a.preread_images)[0]
    if config.likelihood_min_shell:  # the run's likelihood: no weight on image shells below K
        particles = free_low_shells(particles, config.likelihood_min_shell)
    radius, hp = config.stage(state.iteration)
    radius = min(radius, particles.grid_size // 2 - 1)
    shift_range, shift_step = config.shifts(state.iteration)
    print(f"export at {ckpt.name} (update {state.iteration}): radius {radius}, HEALPix {hp}, "
          f"shifts +-{shift_range} px step {shift_step}", flush=True)
    rotations = _rotation_grid(hp)
    translations = grids.get_relion_translation_grid_3d(shift_range, shift_step).astype(np.float32)
    variance = state.offset_variance if config.offset_sigma_hold_px is None else config.offset_sigma_hold_px**2
    prior = -np.sum(translations**2, axis=-1) / (2 * variance)
    prior = (prior - np.log(np.sum(np.exp(prior)))).astype(np.float32)
    rotation_prior = (
        np.asarray(state.direction_prior, np.float32)
        if state.direction_order == hp and state.direction_prior is not None and not config.uniform_direction_prior
        else np.full(len(rotations), 1 / len(rotations), np.float32)
    )
    log_rot = np.log(rotation_prior, where=rotation_prior > 0, out=np.full_like(rotation_prior, -np.inf))
    noise_rows = np.asarray(state.noise).reshape(-1, np.asarray(state.noise).shape[-1])
    keep, embedded = [], []
    for group, row in enumerate(noise_rows):
        stream = prepare_full_row_stream(
            particles, state.theta[:, 0], state.theta[:, 1:],
            noise_variance=np.asarray(make_radial_noise(row, particles.image_shape), np.float32),
            rotations=rotations, translations=translations, rotation_log_prior=log_rot, translation_log_prior=prior,
            rotation_parent=np.zeros(len(rotations), np.int32), translation_parent=np.zeros(len(translations), np.int32),
            n_coarse_rotations=1, n_coarse_translations=1,
            geometry=GeometryConfig(current_size=2 * radius, q=config.q, volume_domain="fourier_half"),
            schedule=ScheduleConfig(image_batch_size=config.image_batch_size, rotation_block_size=config.rotation_block_size),
            scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
            tile_loader=load_tilt_tile, metric_trace_only=True, gemm_precision=config.gemm_precision,
            pass2_mass_floor=config.pass2_mass_floor,
        )
        stream = stream._replace(tile_images=plan_tile_images(stream, config.image_batch_size))
        tiles = [t for t in tilt_tiles(particles, np.arange(particles.n_images), stream.tile_images)
                 if particles.particle_noise_group[t[0]] == group]
        if not tiles:
            continue

        def consume(position, part):
            d = part.diagnostics
            n = len(part.original_image_ids)
            keep.append({"ids": np.asarray(part.original_image_ids), "rot": np.asarray(d["best_rotation_idx"]),
                         "trans": np.asarray(d["best_translation_idx"]), "pmax": np.asarray(d["max_posterior_per_image"]),
                         "llr": np.asarray(d.get("log_evidence_ratio_per_image", np.full(n, np.nan)))})

        accumulate_full_row_tiles(stream, [(t, [None] * len(t)) for t in tiles], consume=consume)
        for t in tiles:
            e = full_row_tile_embeddings(stream, t, [None] * len(t))
            embedded.append((np.asarray(e.original_image_ids), np.asarray(e.embeddings)))
    ids = np.concatenate([k["ids"] for k in keep])
    rot = np.concatenate([k["rot"] for k in keep])
    trans = np.concatenate([k["trans"] for k in keep])
    if not np.array_equal(np.sort(ids), np.arange(particles.n_images)) or rot.min() < 0:
        raise ValueError("The pose pass does not cover every particle with a finite pose")
    z_ids = np.concatenate([e[0] for e in embedded])
    z_all = np.concatenate([e[1] for e in embedded])
    z = z_all[np.argsort(z_ids)][np.searchsorted(np.sort(z_ids), ids)]
    names = starfile.read(read_optimisation_set(ios)[0])["particles"]["rlnTomoParticleName"].to_numpy()
    out = a.out or a.run_dir / f"export_{state.iteration:04d}.npz"
    R = rotations[rot]
    np.savez(
        out, particle_ids=ids, particle_names=names[ids].astype(str), rotations=R, eulers_deg=euler_matrix_to_angles(R),
        translations_px=translations[trans], translations_ang=translations[trans] * particles.voxel_size, z=z,
        pmax=np.concatenate([k["pmax"] for k in keep]), llr=np.concatenate([k["llr"] for k in keep]),
        iteration=state.iteration, healpix_order=hp, radius=radius, checkpoint=str(ckpt),
    )
    if a.star is not None:
        blocks = starfile.read(read_optimisation_set(ios)[0], always_dict=True)
        table = blocks["particles"].copy()
        order = np.argsort(ids)
        eulers = euler_matrix_to_angles(R)[order]
        origins = (translations[trans] * particles.voxel_size)[order]
        for k, column in enumerate(("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")):
            table[column] = eulers[:, k]
        for k, column in enumerate(("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")):
            table[column] = origins[:, k]
        table["rlnMaxValueProbDistribution"] = np.concatenate([k["pmax"] for k in keep])[order]
        blocks["particles"] = table
        starfile.write(blocks, a.star, overwrite=True)
        print(f"wrote {a.star}", flush=True)
    print(f"wrote {out}: {len(ids)} particles, q {z.shape[1]}, {time.monotonic() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
