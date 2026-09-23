"""Slurm-only multi-iteration/resume check; no scientific recovery claim."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import os

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this particle trajectory under Slurm")
    from relax.commands.ppca_initial_model import load_training, source_identity
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.iteration_loop import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data, manifest, identity = load_training(args.manifest)
    identity["source"] = source_identity()
    root = Path(args.output)
    cfg = Config(
        iterations=2,
        stages=((1, 2, 0), (2, 3, 0)),
        oversampling=1,
        shift_range=0,
        shift_step=2,
        image_batch_size=4,
        rotation_block_size=128,
    )
    full = run(data, cfg, root / "full", identity, manifest["particle_diameter_ang"])
    run(data, cfg, root / "resumed", identity, manifest["particle_diameter_ang"], stop_after=1)
    resumed = run(
        data,
        cfg,
        root / "resumed",
        identity,
        manifest["particle_diameter_ang"],
        resume=root / "resumed/checkpoint_0001.npz",
    )
    maxima = {}
    for name, a, b in [
        ("theta", full.theta, resumed.theta),
        ("noise", full.noise, resumed.noise),
        ("first", full.moments.first, resumed.moments.first),
        ("second", full.moments.second, resumed.moments.second),
        ("initialized", full.moments.initialized, resumed.moments.initialized),
    ]:
        np.testing.assert_array_equal(a, b)
        maxima[name] = 0
    with np.load(root / "full/embeddings.npz") as a, np.load(root / "resumed/embeddings.npz") as b:
        np.testing.assert_array_equal(a["particle_ids"], b["particle_ids"])
        np.testing.assert_array_equal(a["z"], b["z"])
    summary = {
        "implementation_smoke": "passed",
        "resume_max_absolute_differences": maxima,
        "radii": [2, 3],
        "source": identity,
        "job_id": os.environ["SLURM_JOB_ID"],
        "scientific_recovery": "not measured",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
