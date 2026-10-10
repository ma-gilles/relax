"""All-particle checkpoint inference, usable as a standalone script with frozen RELAX imports.

No training, sampling, checkpoint rewrite or model export occurs. Species names
are source-provenance annotations used only after inference, not model inputs.
"""

import argparse
import csv
import dataclasses
import json
import os
import re
import time
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_npz(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def species_from_names(names):
    matches = [re.fullmatch(r"(thg|betagal|ribosome)_\d+", str(name).rsplit("/", 1)[-1]) for name in names]
    require(all(match is not None for match in matches), "Unknown species provenance in native particle names")
    return np.asarray([match.group(1) for match in matches], dtype=str)


def collect(tiles, n_particles, n_classes, save_partial):
    """Scatter existing physical-particle posteriors; reject missing/duplicate IDs."""
    probabilities = np.full((n_particles, n_classes), np.nan, np.float32)
    visited = np.zeros(n_particles, bool)
    progress_at, save_at, started = 100, 500, time.monotonic()
    for _, _, tile in tiles:
        ids, values = np.asarray(tile.original_image_ids), np.asarray(tile.class_probabilities)
        require(ids.ndim == 1 and ids.dtype.kind in "iu" and len(ids) > 0,
                "Inference tile must contain integer physical-particle IDs")
        require(np.all((ids >= 0) & (ids < n_particles)) and len(np.unique(ids)) == len(ids),
                "Invalid or duplicate inference particle IDs")
        require(not np.any(visited[ids]), "A particle was evaluated more than once")
        require(values.shape == (len(ids), n_classes) and values.dtype == np.float32,
                "Inference probability shape/dtype mismatch")
        require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
                and np.allclose(values.sum(axis=1), 1, rtol=1e-6, atol=1e-6),
                "Inference class probabilities must be finite and normalized")
        probabilities[ids], visited[ids] = values, True
        count = int(visited.sum())
        if count >= progress_at or count == n_particles:
            print(f"Evaluated {count}/{n_particles} physical particles ({time.monotonic() - started:.0f} s)", flush=True)
            progress_at = (count // 100 + 1) * 100
        if count >= save_at:
            save_partial(probabilities, visited)
            save_at = (count // 500 + 1) * 500
    require(visited.all(), f"Incomplete inference: {int(visited.sum())}/{n_particles} particles")
    return probabilities


def write_tables(output, names, species, probabilities):
    labels = probabilities.argmax(axis=1).astype(np.int32)
    with (output / "particle_membership.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["particle_id", "particle_name", "species_provenance", "class_label",
                         *[f"class_probability_{k}" for k in range(probabilities.shape[1])]])
        for i, name in enumerate(names):
            writer.writerow([i, name, species[i], labels[i], *[format(float(p), ".9g") for p in probabilities[i]]])
    counts = np.asarray([[np.count_nonzero((labels == k) & (species == s))
                          for s in ("thg", "betagal", "ribosome")] for k in range(probabilities.shape[1])])
    with (output / "class_species_composition.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["class_label", "particles", "THG", "Beta-Gal", "Ribosome",
                         "THG_percent", "Beta-Gal_percent", "Ribosome_percent"])
        for k, row in enumerate(counts):
            percentages = row * 100.0 / row.sum() if row.sum() else np.full(3, np.nan)
            writer.writerow([k, int(row.sum()), *row, *percentages])
    return labels, counts


def evaluate(args):
    # Imports resolve from the launcher's frozen source. Reuse final-inference
    # search policy without invoking training or changing the saved state.
    from relax.commands.ppca_initial_model import refuse_unsupported_tilt_optics, tilt_series_hash
    from relax.ppca_initial_class3d import checkpoint
    from relax.ppca_initial_class3d.config import Config
    from relax.ppca_initial_class3d.expectation import mixture_tiles, pose_grid
    from relax.ppca_initial_class3d.images import zero_masked_tilt_particles
    from relax.ppca_initial_class3d.iteration_loop import candidate_source, local_search_policy
    from relax.ppca_initial_class3d.local_search import LocalGrid
    from relax.ppca_initial_class3d.tomo_input import load_tomo_dataset
    from relax.ppca_initial_model.checkpoint import file_hash
    from relax.ppca_initial_model.tomo import tilt_particles_from_tomo_dataset
    from relax.relion.tomo_input import read_optimisation_set

    require(args.expected_particles > 0, "Expected physical-particle count must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    run_dir, ios = Path(args.run_dir), Path(args.ios).resolve()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    before = file_hash(checkpoint_path)
    run = json.loads((run_dir / "run.json").read_text())
    config = Config(**run["config"])
    particles_star, tomograms_star = map(Path, read_optimisation_set(ios))
    identity = {"ios_sha256": file_hash(ios), "particles_sha256": file_hash(particles_star),
                "tomograms_sha256": file_hash(tomograms_star), "tilt_series_sha256": tilt_series_hash(tomograms_star)}
    require(all(run["identity"].get(key) == value for key, value in identity.items()),
            "Native input hashes differ from the training run")
    state = checkpoint.load(checkpoint_path, config, run["identity"])
    require(state.iteration >= 1, "Inference requires a checkpoint with a defined training stage")
    require(not config.auto_sampling or state.sampling is not None,
            "Auto-sampled inference requires the checkpoint's recorded sampling state")
    diameter_ang = run["identity"].get("particle_diameter_ang")
    if config.zero_mask:
        require(diameter_ang is not None and np.isfinite(diameter_ang) and diameter_ang > 0,
                "Masked inference requires the training particle diameter")
    refuse_unsupported_tilt_optics(particles_star)
    print(f"Checkpoint {checkpoint_path.name}: iteration {state.iteration}, K={config.n_classes}, q={config.q}", flush=True)
    tomo = load_tomo_dataset(particles_star, tomograms_star, output / "particles_2d.star",
                             datadir=str(particles_star.resolve().parent), lazy=True)
    dataset = tilt_particles_from_tomo_dataset(tomo)
    if config.zero_mask:
        dataset = zero_masked_tilt_particles(
            dataset, tomo.images.process_images_half, diameter_ang / dataset.voxel_size,
        )
    names = np.asarray(tomo.particle_names, dtype=str)
    ids = np.arange(dataset.n_images, dtype=np.int64)
    require(dataset.n_images == args.expected_particles == len(names) == len(np.unique(names)),
            "Native physical-particle count/names do not match the requested complete dataset")
    require(np.array_equal(dataset.original_image_indices_from_local(ids), ids)
            and np.array_equal(np.sort(state.order), ids), "Checkpoint/native physical-particle ordering mismatch")
    species = species_from_names(names)
    grid = pose_grid(dataset, config, state.iteration, sampling_state=state.sampling)
    # Like the controller's final inference, evaluate the saved model using
    # that update's search policy, not the next update's local-search boundary.
    last = dataclasses.replace(state, iteration=state.iteration - 1)
    local, sigma_deg, local_range = local_search_policy(config, last, grid)
    require(not local or state.membership is not None,
            "Local-search inference requires checkpoint particle pose history")
    candidates = (candidate_source(dataset, state.membership, LocalGrid.of(grid), sigma_deg=sigma_deg,
                                   offset_range_px=local_range, counts=[]) if local else None)
    common = dict(particle_ids=ids, particle_names=names, species_provenance=species,
                  checkpoint_iteration=np.int64(state.iteration), checkpoint_sha256=before)

    def partial(probabilities, visited):
        labels = np.full(len(ids), -1, np.int32)
        labels[visited] = probabilities[visited].argmax(axis=1)
        save_npz(output / "partial_memberships.npz", **common, status="PARTIAL_NOT_COMPLETE",
                 visited=visited, class_probabilities=probabilities, class_labels=labels)

    print(f"Inferring ALL {len(ids)} particles; stage {config.stage(state.iteration)}; "
          f"{len(grid.rotations)} rotations x {len(grid.translations)} shifts", flush=True)
    tiles = mixture_tiles(dataset, state, config, [ids], state.iteration, grid, moments=False,
                          diameter_ang=diameter_ang, candidates=candidates)
    probabilities = collect(tiles, len(ids), config.n_classes, partial)
    after = file_hash(checkpoint_path)
    require(before == after, "Source checkpoint changed during inference")
    labels, counts = write_tables(output, names, species, probabilities)
    save_npz(output / "memberships.npz", **common, class_probabilities=probabilities, class_labels=labels,
             class_prior=np.asarray(state.class_prior), visited=np.ones(len(ids), bool), status="COMPLETED")
    summary = dict(status="COMPLETED", inference_only=True, all_particles=True, particles=len(ids),
                   checkpoint=str(checkpoint_path), checkpoint_iteration=state.iteration,
                   checkpoint_sha256_before=before, checkpoint_sha256_after=after,
                   class_index_base=0, hard_assignment="class posterior marginalized over poses and latent coordinates",
                   n_classes=config.n_classes, q=config.q, stage=list(config.stage(state.iteration)),
                   zero_mask=bool(config.zero_mask), particle_diameter_ang=diameter_ang,
                   local_searches=local,
                   sampling=None if state.sampling is None else state.sampling.to_json(),
                   input_identity=identity, species_order=["thg", "betagal", "ribosome"],
                   species_labels="input-name provenance only; not independent molecular ground truth",
                   class_species_counts=counts.tolist(), class_counts=counts.sum(axis=1).tolist())
    temporary = output / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, output / "summary.json")  # Publish completion only after every final artifact.
    print(f"COMPLETE: all {len(ids)} physical particles; {output}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "checkpoint", "ios", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--expected-particles", required=True, type=int)
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
