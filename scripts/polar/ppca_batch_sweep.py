"""Matched 100k-particle PPCA SGD sweep at fixed particle presentations."""

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

TOTAL_VISITS = 1_800_000
BATCHES = (10, 300, 3000)
VISIT_MILESTONES = (3000, 30000, 60000, 150000, 300000, 600000,
                    900000, 1200000, 1500000, TOTAL_VISITS)
EXPECTED_LOCK = "12fa63045fb8ec121ee1c5b827b64a0c3d2bea5085bfc1187eee5bb1ef4bd347"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def envelope(state):
    result = {key: hashlib.sha256(np.ascontiguousarray(np.asarray(getattr(state, key))).tobytes()).hexdigest()
              for key in ("theta", "noise", "order")}
    result["rng_state"] = hashlib.sha256(json.dumps(state.rng_state, sort_keys=True).encode()).hexdigest()
    return result


def config_from_template(path, seed, batch):
    from relax.ppca_initial_model.config import Config

    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
    old = Config(**{**metadata["config"], "stages": tuple(tuple(s) for s in metadata["config"]["stages"])})
    if (old.optimizer != "momentum_sgd" or old.q != 2 or old.seed != seed
            or old.iterations != 6000 or old.sgd_learning_rate != 1.2
            or old.stages != ((1, 31, 3),) or old.oversampling != 0
            or old.stochastic_batch_size != 300 or not old.stochastic_all_iterations
            or not old.balanced_stochastic_halves or not old.stream_coarse_recompute
            or old.stream_full_fine_rows or old.fine_devices != 1 or old.shift_range != 6
            or old.shift_step != 2):
        raise ValueError("Template does not match the frozen HP3 noise-1 SGD protocol")
    if batch not in BATCHES or TOTAL_VISITS % batch:
        raise ValueError("Invalid batch or visit budget")
    updates = TOTAL_VISITS // batch
    return dataclasses.replace(old, stochastic_batch_size=batch, iterations=updates,
                               checkpoint_interval=updates + 1, skip_final_embeddings=True)


def verify_context():
    import jax

    source = Path(os.environ["POLAR_SOURCE_ROOT"]).resolve()
    fixture = Path(os.environ["POLAR_FIXTURE_ROOT"]).resolve()
    root = Path(os.environ["POLAR_RUN_ROOT"]).resolve()
    meta = json.loads((source / ".polar-source.json").read_text())
    if (jax.default_backend() != "gpu" or "A100" not in jax.devices()[0].device_kind
            or meta["head"] != os.environ["PPCA_EXPECTED_HEAD"]
            or sha(source / "pixi.lock") != EXPECTED_LOCK):
        raise ValueError("Wrong GPU, source or environment")
    for path in (fixture, root):
        if not path.is_relative_to(Path("/scratch/network/mg6942/relax-polar")):
            raise ValueError("Polar job data must stay on network scratch")
    receipt = json.loads((fixture / "fixture_verified.json").read_text())
    manifest = fixture / "training/manifest.json"
    if receipt["n_images"] != 100000 or receipt["identity"]["manifest_sha256"] != sha(manifest):
        raise ValueError("Wrong verified 100k fixture")
    return source, fixture, root, meta, receipt


def initialize(seed, batch, dataset, fixture_meta, identity, output, template):
    from relax.ppca_initial_model import iteration_loop

    config = config_from_template(template, seed, batch)
    state = iteration_loop.run(dataset, config, output, identity,
                               fixture_meta["particle_diameter_ang"], stop_after=0,
                               log_direction_prior=False)
    if state.theta.dtype != np.complex64 or state.noise.dtype != np.float32:
        raise ValueError("Nonproduction CP0 precision")
    return state, config


def audit_segment(path, offset, first, last, batch, visits_by_image, selection_hash):
    """Check every update and particle selection appended by one run segment."""
    row = None
    with path.open("rb") as stream:
        stream.seek(offset)
        for iteration in range(first, last + 1):
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                raise ValueError(f"Missing or incomplete log row for update {iteration}")
            row = json.loads(line)
            if (row["iteration"] != iteration or row["half_counts"] != [batch // 2] * 2
                    or row["dtype"] != "complex64" or "direction_prior" in row
                    or not np.isfinite(row["mean_power"] + row["loading_power"])):
                raise ValueError(f"Invalid log row for update {iteration}")
            halves = row["particle_ids"]
            if len(halves) != 2 or any(len(ids) != batch // 2 for ids in halves):
                raise ValueError(f"Wrong particle selection size at update {iteration}")
            ids = np.asarray(halves[0] + halves[1], dtype=np.int32)
            if (np.any(ids < 0) or np.any(ids >= len(visits_by_image))
                    or len(np.unique(ids)) != batch):
                raise ValueError(f"Invalid particle IDs at update {iteration}")
            selection_hash.update(np.asarray([iteration], np.int32).tobytes())
            selection_hash.update(ids.tobytes())
            np.add.at(visits_by_image, ids, 1)
        new_offset = stream.tell()
        if stream.read(1):
            raise ValueError("Unexpected extra update log rows")
    return row, new_offset


def main():
    if os.environ.get("PPCA_SWEEP_SELF_TEST") == "1":
        assert [TOTAL_VISITS // b for b in BATCHES] == [180000, 6000, 600]
        assert all(v % b == 0 for v in VISIT_MILESTONES for b in BATCHES)
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            log = Path(temp) / "iterations.jsonl"
            rows = [{"iteration": i, "half_counts": [2, 2], "dtype": "complex64",
                     "mean_power": 1.0, "loading_power": 2.0,
                     "particle_ids": [[0, 2], [1, 3]]} for i in (1, 2)]
            log.write_text("".join(json.dumps(row) + "\n" for row in rows))
            counts = np.zeros(4, np.int32)
            row, offset = audit_segment(log, 0, 1, 2, 4, counts, hashlib.sha256())
            assert row["iteration"] == 2 and offset == log.stat().st_size
            assert counts.tolist() == [2, 2, 2, 2]
        print("visit milestones and batch counts passed")
        return
    import jax.numpy as jnp

    from relax.commands.ppca_initial_model import load_training
    from relax.ppca_initial_model import checkpoint, iteration_loop

    source, fixture, root, source_meta, receipt = verify_context()
    mode = os.environ["PPCA_SWEEP_MODE"]
    if mode not in ("init", "train"):
        raise ValueError("Mode must be init or train")
    manifest = fixture / "training/manifest.json"
    dataset, fixture_meta, _ = load_training(manifest)
    if dataset.n_images != 100000:
        raise ValueError("Wrong training count")
    if mode == "init":
        records = {}
        for seed in (11, 12):
            template = Path(os.environ[f"PPCA_TEMPLATE_SEED{seed}"]).resolve()
            output = root / f"seed{seed}"
            identity = {"fixture_manifest_sha256": sha(manifest), "source_id": source_meta["source_id"],
                        "seed": seed, "canonical": True}
            state, config = initialize(seed, 300, dataset, fixture_meta, identity, output, template)
            cp = output / "checkpoint_0000.npz"
            records[str(seed)] = {"checkpoint": str(cp), "sha256": sha(cp), "envelope": envelope(state),
                                  "config": dataclasses.asdict(config)}
        (root / "canonical.json").write_text(json.dumps({"schema": "ppca_batch_canonical_v1",
                                                       "fixture_manifest_sha256": sha(manifest),
                                                       "source": source_meta, "seeds": records}, indent=2) + "\n")
        print(root / "canonical.json", flush=True)
        return

    seed = int(os.environ["PPCA_SEED"])
    batch = int(os.environ["PPCA_BATCH"])
    template = Path(os.environ[f"PPCA_TEMPLATE_SEED{seed}"]).resolve()
    canonical_path = Path(os.environ["PPCA_CANONICAL_CP0"]).resolve()
    canonical_record = json.loads((canonical_path.parent.parent / "canonical.json").read_text())
    if (canonical_record["fixture_manifest_sha256"] != sha(manifest)
            or canonical_record["source"]["source_id"] != source_meta["source_id"]
            or canonical_record["seeds"][str(seed)]["sha256"] != sha(canonical_path)):
        raise ValueError("Canonical source/fixture/CP0 mismatch")
    identity = {"fixture_manifest_sha256": sha(manifest), "source_id": source_meta["source_id"],
                "seed": seed, "batch": batch, "particle_visits": TOTAL_VISITS}
    state, config = initialize(seed, batch, dataset, fixture_meta, identity, root, template)
    with np.load(canonical_path, allow_pickle=False) as saved:
        cm = json.loads(str(saved["metadata"]))
        ctheta, cnoise, corder, cprior = (np.asarray(saved[k]) for k in
                                         ("theta", "noise", "order", "direction_prior"))
    if (cm["iteration"] != 0 or not np.array_equal(np.asarray(state.noise), cnoise)
            or not np.array_equal(state.order, corder)
            or not np.array_equal(np.asarray(state.direction_prior), cprior)
            or state.rng_state != cm["rng_state"]):
        raise ValueError("Independent CP0 envelope differs from canonical")
    initial_delta = float(np.linalg.norm(np.asarray(state.theta) - ctheta) / np.linalg.norm(ctheta))
    if not np.isfinite(initial_delta) or initial_delta > 1e-5:
        raise ValueError(f"Independent CP0 theta differs: {initial_delta}")
    state = dataclasses.replace(state, theta=jnp.asarray(ctheta))
    cp0 = root / "checkpoint_0000.npz"
    checkpoint.save(cp0, state, config, identity)
    initial = {"schema": "ppca_batch_sweep_init_v1", "seed": seed, "batch": batch,
               "source": source_meta, "fixture_manifest_sha256": sha(manifest),
               "canonical_cp0_sha256": sha(canonical_path), "cp0_sha256": sha(cp0),
               "cp0_envelope": envelope(state), "independent_theta_rel_l2": initial_delta,
               "config": dataclasses.asdict(config), "precision": "complex64/float32"}
    (root / "initialization.json").write_text(json.dumps(initial, indent=2) + "\n")
    started = time.monotonic()
    previous = cp0
    progress = []
    visits_by_image = np.zeros(dataset.n_images, np.int32)
    selection_hash = hashlib.sha256()
    log_path = root / "iterations.jsonl"
    log_offset = 0
    previous_iteration = 0
    for visits in VISIT_MILESTONES:
        end = visits // batch
        state = iteration_loop.run(dataset, config, root, identity,
                                   fixture_meta["particle_diameter_ang"],
                                   resume=previous, stop_after=end,
                                   log_direction_prior=False)
        if state.iteration != end:
            raise ValueError("Sweep stopped before requested visit milestone")
        previous = root / f"checkpoint_{end:04d}.npz"
        row, log_offset = audit_segment(log_path, log_offset, previous_iteration + 1, end,
                                        batch, visits_by_image, selection_hash)
        previous_iteration = end
        if int(visits_by_image.sum()) != visits:
            raise ValueError("Audited particle visits differ from milestone")
        progress.append({"particle_visits": visits, "iteration": end,
                         "checkpoint": str(previous), "sha256": sha(previous),
                         "selection_sha256": selection_hash.hexdigest(),
                         "distinct_images_seen": int(np.count_nonzero(visits_by_image)),
                         "mean_power": row["mean_power"], "loading_power": row["loading_power"],
                         "pmax": float(np.mean([p["pmax_mean"] for p in row["posterior"]])),
                         "elapsed_seconds": time.monotonic() - started})
        (root / "progress.json").write_text(json.dumps({"schema": "ppca_batch_sweep_progress_v1",
                                                  "seed": seed, "batch": batch, "source": source_meta,
                                                  "milestones": progress}, indent=2) + "\n")
        print(json.dumps(progress[-1]), flush=True)
    (root / "result.json").write_text(json.dumps({"schema": "ppca_batch_sweep_result_v1",
                                               "initialization": initial, "milestones": progress,
                                               "total_particle_visits": TOTAL_VISITS,
                                               "visits_per_image": {
                                                   "min": int(visits_by_image.min()),
                                                   "median": float(np.median(visits_by_image)),
                                                   "max": int(visits_by_image.max()),
                                                   "distinct": int(np.count_nonzero(visits_by_image))},
                                               "selection_sha256": selection_hash.hexdigest(),
                                               "wall_seconds": time.monotonic() - started}, indent=2) + "\n")


if __name__ == "__main__":
    main()
