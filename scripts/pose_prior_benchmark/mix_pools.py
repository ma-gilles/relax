#!/usr/bin/env python
"""Mix simulated pose-prior benchmark datasets ("pools") into one dataset of a given composition.

Each pool is a ``simulate_dataset.py`` output of the same specimen, grid and noise level whose particles follow one
viewing-direction distribution (a Kent cap, uniform). The mixture takes the first ``COUNT`` particles of every pool, so
a mixture of fewer particles is a subset of a larger one and every particle keeps its simulated image. It writes only
a ``particles.star`` that names the pools' stacks through symbolic links (``LABEL.<stack name>``), so no image is
copied. The micrograph name of a particle gets ``LABEL_`` in front, so particles of different pools never share a
micrograph (RELION's scale group), as the particles of one pool never do. The true viewing-direction distribution of
the mixture, the pools' distributions weighted by ``COUNT / total``, is recorded in ``pose_distribution.json``
(``true_prior.population_prior`` reads it).

Run (CPU node)::

    python scripts/pose_prior_benchmark/mix_pools.py --out-dir OUT --pool kent KENT_DIR 33000 --pool uniform UNIFORM_DIR 67000

``OUT`` receives ``particles.star``, the stack links, copies of the first pool's ``reference_gt_relion.mrc`` and
``reference_init_relion.mrc`` (every pool's must match them) and ``pose_distribution.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, NamedTuple

import mrcfile
import numpy as np
import pandas as pd
import starfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.pose_prior_benchmark.add_offsets import stack_name_of  # noqa: E402

REFERENCE_FILES = ("reference_gt_relion.mrc", "reference_init_relion.mrc")
REFERENCE_TOLERANCE = 1e-4  # relative rms difference of a pool's reference map from the first pool's
POSE_RECORD_KEYS = ("poses", "dataset_params_option")  # what a pool's pose record contributes to the mixture's
MICROGRAPH_COLUMN = "rlnMicrographName"


class Pool(NamedTuple):
    """One pool of a mixture: its label, its dataset directory and the number of its first particles to take."""

    label: str
    directory: Path
    count: int


def check_references(pools: list[Pool]) -> None:
    """Raise unless every pool's reference maps equal the first pool's (the same specimen and low-pass)."""
    for name in REFERENCE_FILES:
        with mrcfile.open(pools[0].directory / name, permissive=True) as first:
            reference = np.asarray(first.data, dtype=np.float64)
        scale = np.sqrt(np.mean(reference**2))
        for pool in pools[1:]:
            with mrcfile.open(pool.directory / name, permissive=True) as other:
                data = np.asarray(other.data, dtype=np.float64)
            if data.shape != reference.shape or np.sqrt(np.mean((data - reference) ** 2)) > REFERENCE_TOLERANCE * scale:
                raise ValueError(f"{name} of pool {pool.label} differs from that of pool {pools[0].label}")


def select_pool(pool: Pool) -> tuple[Any, Any]:
    """The optics block and the first ``pool.count`` particles of a pool, their images renamed to the stack link
    ``LABEL.<stack name>`` and their micrograph names prefixed by ``LABEL_``."""
    blocks = starfile.read(pool.directory / "particles.star", always_dict=True)
    optics, particles = blocks["optics"], blocks["particles"]
    stack = stack_name_of(particles)
    if not 0 < pool.count <= len(particles):
        raise ValueError(f"pool {pool.label} has {len(particles)} particles, {pool.count} asked")
    particles = particles.iloc[: pool.count].copy()
    link = f"{pool.label}.{stack}"
    particles["rlnImageName"] = [f"{index}@{link}" for index in range(1, pool.count + 1)]
    particles[MICROGRAPH_COLUMN] = [f"{pool.label}_{name}" for name in particles[MICROGRAPH_COLUMN]]
    return optics, particles


def link_stack(pool: Pool, out_dir: Path) -> None:
    """Link ``out_dir/LABEL.<stack name>`` to the pool's stack file."""
    stack = stack_name_of(starfile.read(pool.directory / "particles.star", always_dict=True)["particles"])
    link = out_dir / f"{pool.label}.{stack}"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink((pool.directory / stack).resolve(), link)


def pool_pose_record(pool: Pool) -> dict[str, Any]:
    """A pool's pose-distribution record reduced to what the mixture's true prior needs."""
    record = json.loads((pool.directory / "pose_distribution.json").read_text())
    return {key: record[key] for key in POSE_RECORD_KEYS if key in record}


def mix_pools(pools: list[Pool], out_dir: Path) -> dict[str, Any]:
    """Write the mixture of ``pools`` to ``out_dir``; returns its pose-distribution record.

    Refuses pools whose optics differ (the same CTF model, voxel size and box) or whose reference maps differ, and
    duplicate labels.
    """
    if not pools:
        raise ValueError("no pools to mix")
    if len({pool.label for pool in pools}) != len(pools):
        raise ValueError("pool labels must be distinct")
    check_references(pools)
    selected = [select_pool(pool) for pool in pools]
    for pool, (optics, _) in zip(pools[1:], selected[1:]):
        if not optics.equals(selected[0][0]):
            raise ValueError(f"the optics of pool {pool.label} differ from those of pool {pools[0].label}")
    total = sum(pool.count for pool in pools)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pool in pools:
        link_stack(pool, out_dir)
    particles = pd.concat([table for _, table in selected], ignore_index=True)
    starfile.write({"optics": selected[0][0], "particles": particles}, out_dir / "particles.star", overwrite=True)
    for name in REFERENCE_FILES:
        shutil.copy2(pools[0].directory / name, out_dir / name)
    record = {
        "poses": "mixture",
        "components": [{"weight": pool.count / total, "poses": pool_pose_record(pool)} for pool in pools],
        "n_images": total,
        "pools": [{"label": pool.label, "directory": str(pool.directory), "count": pool.count} for pool in pools],
    }
    (out_dir / "pose_distribution.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--pool",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "DIR", "COUNT"),
        help="a pool (simulate_dataset.py output) and the number of its first particles to take; repeatable",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    pools = [Pool(label, Path(directory), int(count)) for label, directory, count in args.pool]
    record = mix_pools(pools, args.out_dir)
    print(json.dumps({key: record[key] for key in ("n_images", "pools")}, indent=2))


if __name__ == "__main__":
    main()
