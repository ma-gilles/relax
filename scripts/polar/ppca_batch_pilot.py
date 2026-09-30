"""Time exact-source PPCA momentum SGD at three minibatch sizes on Polar."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np


HEAD = "ef4d73b5bffed1f70accdcc662a481f13056b82e"
TOTAL_VISITS = 1_800_000
BATCHES = (10, 300, 3000)
PILOT_UPDATES = 5


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def planned_updates(batch):
    if TOTAL_VISITS % batch or batch < 2 or batch % 2:
        raise ValueError("Every batch must be even and divide the visit budget")
    return TOTAL_VISITS // batch


def main():
    if {batch: planned_updates(batch) for batch in BATCHES} != {10: 180000, 300: 6000, 3000: 600}:
        raise AssertionError("Visit budget changed")
    if os.environ.get("PILOT_SELF_TEST") == "1":
        print("visit-budget self-test passed")
        return

    import jax
    from relax.commands.ppca_initial_model import load_training
    from relax.ppca_initial_model import iteration_loop
    from relax.ppca_initial_model.config import Config

    source = Path(os.environ["POLAR_SOURCE_ROOT"])
    fixture = Path(os.environ["POLAR_FIXTURE_ROOT"])
    run_root = Path(os.environ["POLAR_RUN_ROOT"])
    template_cp = Path(os.environ["PPCA_TEMPLATE_CP0"])
    if jax.default_backend() != "gpu" or "A100" not in jax.devices()[0].device_kind:
        raise RuntimeError("Pilot requires one scheduler-assigned A100")
    if json.loads((source / ".polar-source.json").read_text())["head"] != HEAD:
        raise ValueError("Wrong frozen science source")
    for path in (fixture, run_root, template_cp):
        if not path.resolve().is_relative_to(Path("/scratch/network/mg6942/relax-polar")):
            raise ValueError("Pilot input/output escaped Polar network scratch")
    with np.load(template_cp, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        template_theta = np.asarray(saved["theta"])
        template_noise = np.asarray(saved["noise"])
        template_order = np.asarray(saved["order"])
    template = Config(**{**metadata["config"], "stages": tuple(tuple(s) for s in metadata["config"]["stages"])})
    if (template.optimizer != "momentum_sgd" or template.q != 2 or template.seed != 11
            or template.sgd_learning_rate != 1.2 or template.stages != ((1, 31, 3),)
            or template.stochastic_batch_size != 300 or template.iterations != 6000
            or not template.stochastic_all_iterations or not template.balanced_stochastic_halves):
        raise ValueError("Template is not the exact noise-1 HP3 SGD protocol")
    manifest = fixture / "training/manifest.json"
    data, fixture_meta, manifest_identity = load_training(manifest)
    if data.n_images != 20000 or sha(manifest) != os.environ["PPCA_MANIFEST_SHA256"]:
        raise ValueError("Wrong timing-only fixture")
    results = []
    for batch in BATCHES:
        config = dataclasses.replace(template, iterations=planned_updates(batch),
                                     stochastic_batch_size=batch, checkpoint_interval=1)
        output = run_root / f"batch{batch}"
        identity = {**manifest_identity, "pilot": True, "batch": batch, "source_head": HEAD}
        state = iteration_loop.run(data, config, output, identity,
                                   fixture_meta["particle_diameter_ang"], stop_after=0)
        theta = np.asarray(state.theta)
        initial_rel_l2 = float(np.linalg.norm(theta - template_theta) / np.linalg.norm(template_theta))
        if (initial_rel_l2 > 1e-5 or not np.array_equal(np.asarray(state.noise), template_noise)
                or not np.array_equal(state.order, template_order)
                or state.rng_state != metadata["rng_state"]):
            raise ValueError(f"Unmatched timing CP0 for batch {batch}: {initial_rel_l2}")
        result = iteration_loop.run(data, config, output, identity,
                                    fixture_meta["particle_diameter_ang"],
                                    resume=output / "checkpoint_0000.npz", stop_after=PILOT_UPDATES)
        if result.iteration != PILOT_UPDATES:
            raise ValueError("Pilot ended early")
        rows = [json.loads(line) for line in (output / "iterations.jsonl").read_text().splitlines()]
        if len(rows) != PILOT_UPDATES or any(r["half_counts"] != [batch // 2] * 2 for r in rows):
            raise ValueError("Pilot batch audit failed")
        results.append({"batch": batch, "planned_updates": config.iterations,
                        "particle_visits": batch * config.iterations,
                        "cp0_theta_rel_l2": initial_rel_l2,
                        "update_seconds": [r["elapsed_seconds"] for r in rows],
                        "steady_median_seconds": float(np.median([r["elapsed_seconds"] for r in rows[2:]])),
                        "posterior_pmax_last": float(np.mean([p["pmax_mean"] for p in rows[-1]["posterior"]])),
                        "last_checkpoint_sha256": sha(output / f"checkpoint_{PILOT_UPDATES:04d}.npz")})
        print(json.dumps(results[-1]), flush=True)
    (run_root / "pilot.json").write_text(json.dumps({"schema": "ppca_batch_pilot_v1",
                                               "source_head": HEAD, "fixture_manifest_sha256": sha(manifest),
                                               "template_cp0_sha256": sha(template_cp), "results": results},
                                              indent=2) + "\n")


if __name__ == "__main__":
    main()
