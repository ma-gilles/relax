#!/usr/bin/env python
"""How non-uniform are the viewing directions of a particle STAR file, after expanding the point-group symmetry.

A refinement with symmetry reports each particle's pose in one asymmetric unit, so the raw ``(rot, tilt)`` leave the
rest of the sphere empty and look far from uniform whatever the particles do. Each viewing direction is therefore
replaced by its ``|G|`` symmetry-equivalent directions (the RELION operators of ``relax.sampling.symmetry``) before
it is counted on the RELION HEALPix grid. Per HEALPix order the script reports the benchmark's measure
(``evaluate_run.kl_bits`` of the histogram against the uniform distribution, as ``truth_nonuniformity`` does for a
simulated dataset), the share of the sphere that holds half of the particles and the share of directions without a
particle, each beside the same number for as many uniformly distributed particles (the floor a finite sample cannot
go below), and the raw, unexpanded KL for contrast. With ``--config`` it adds the population KL of the Kent scenarios
of the benchmark's config.yaml at the same orders.

Run::

    python scripts/pose_prior_benchmark/star_nonuniformity.py --star test.star --symmetry C3 --out-dir OUT \
        --orders 1 2 3 4 5 --config scripts/pose_prior_benchmark/config.yaml --expanded-star OUT/expanded.star

``OUT`` receives ``nonuniformity.json``; ``--expanded-star`` writes the expanded directions as a STAR file for
``angdist``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from relax.sampling.symmetry import rotational_operators  # noqa: E402
from relax.star.blocks import read_star_blocks  # noqa: E402
from scripts.pose_prior_benchmark.evaluate_run import kl_bits, viewing_directions  # noqa: E402
from scripts.pose_prior_benchmark.true_prior import kent_population_prior  # noqa: E402

DEFAULT_ORDERS = (1, 2, 3, 4, 5)
DEFAULT_NULL_DRAWS = 20


def read_viewing_directions(star_path: Path) -> np.ndarray:
    """Return the ``(n, 3)`` unit viewing directions of the ``rlnAngleRot`` and ``rlnAngleTilt`` of a particle STAR file."""
    table = read_star_blocks(star_path)["particles"]
    rot = np.asarray(table["rlnAngleRot"], dtype=np.float64)
    tilt = np.asarray(table["rlnAngleTilt"], dtype=np.float64)
    return viewing_directions(rot, tilt)


def expand_directions(vectors: np.ndarray, symmetry: str) -> np.ndarray:
    """Return the ``(n * |G|, 3)`` symmetry-equivalent directions of ``(n, 3)`` unit vectors, particle by particle."""
    operators = rotational_operators(symmetry)
    return np.einsum("kij,nj->nki", operators, vectors).reshape(-1, 3)


def direction_counts(vectors: np.ndarray, order: int) -> np.ndarray:
    """Count unit vectors per RELION HEALPix (NEST) direction at ``order``."""
    nside = 1 << int(order)
    pixels = hp.vec2pix(nside, vectors[:, 0], vectors[:, 1], vectors[:, 2], nest=True)
    return np.bincount(pixels, minlength=hp.nside2npix(nside)).astype(np.float64)


def concentration(counts: np.ndarray) -> dict[str, float]:
    """The non-uniformity of a histogram: KL from uniform (bits), sphere share holding half the mass, empty share."""
    p = counts / counts.sum()
    cumulative = np.cumsum(np.sort(p)[::-1])
    return {
        "kl_from_uniform_bits": kl_bits(p, np.full(p.size, 1.0 / p.size)),
        "sphere_fraction_with_half_of_mass": float((np.searchsorted(cumulative, 0.5) + 1) / p.size),
        "directions_without_a_particle": float(np.mean(counts == 0)),
    }


def uniform_particles(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``n`` unit vectors drawn uniformly from the sphere."""
    z = rng.uniform(-1.0, 1.0, n)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, n)
    radius = np.sqrt(1.0 - z**2)
    return np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=-1)


def uniform_floor(n: int, symmetry: str, order: int, draws: int, rng: np.random.Generator) -> dict[str, float]:
    """The mean and standard deviation, over ``draws`` samples, of every ``concentration`` value for ``n`` uniform
    particles expanded by ``symmetry`` (a uniform particle's orbit is the orbit of its asymmetric-unit image)."""
    rows = [concentration(direction_counts(expand_directions(uniform_particles(n, rng), symmetry), order)) for _ in range(draws)]
    return {
        f"{stat}_{key}": float(fn([row[key] for row in rows]))
        for key in rows[0]
        for stat, fn in (("mean", np.mean), ("std", np.std))
    }


def nonuniformity_by_order(
    vectors: np.ndarray, symmetry: str, orders: tuple[int, ...], draws: int, seed: int
) -> list[dict[str, Any]]:
    """One row per HEALPix order: the expanded and the raw directions' non-uniformity and the uniform floor."""
    rng = np.random.default_rng(seed)
    expanded = expand_directions(vectors, symmetry)
    rows = []
    for order in orders:
        row: dict[str, Any] = {"healpix_order": order, "n_directions": hp.nside2npix(1 << order)}
        row.update({f"expanded_{k}": v for k, v in concentration(direction_counts(expanded, order)).items()})
        row["unexpanded_kl_from_uniform_bits"] = concentration(direction_counts(vectors, order))["kl_from_uniform_bits"]
        row.update({f"uniform_{k}": v for k, v in uniform_floor(len(vectors), symmetry, order, draws, rng).items()})
        row["expanded_kl_above_floor_bits"] = row["expanded_kl_from_uniform_bits"] - row["uniform_mean_kl_from_uniform_bits"]
        rows.append(row)
    return rows


def kent_scenario_kl(config_path: Path, orders: tuple[int, ...]) -> dict[str, dict[int, float]]:
    """Population KL from uniform (bits) per HEALPix order of each Kent scenario of the benchmark's config file."""
    scenarios = yaml.safe_load(config_path.read_text())["scenarios"]
    out: dict[str, dict[int, float]] = {}
    for name, scenario in scenarios.items():
        poses = scenario.get("poses")
        if not poses or poses["kind"] != "kent":
            continue
        out[name] = {}
        for order in orders:
            p = kent_population_prior(poses["kappa"], poses["beta"], poses["mu"], poses["mu0"], order)
            out[name][order] = kl_bits(p, np.full(p.size, 1.0 / p.size))
    return out


def write_expanded_star(path: Path, vectors: np.ndarray) -> None:
    """Write directions as a particle STAR file with ``rlnAngleRot`` and ``rlnAngleTilt`` (degrees) for ``angdist``.

    The file has a one-row optics block too: ``starfile`` returns a bare table, not blocks by name, for a single block.
    """
    rot = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    tilt = np.degrees(np.arccos(np.clip(vectors[:, 2], -1.0, 1.0)))
    lines = ["data_optics\n", "\n", "loop_\n", "_rlnOpticsGroup #1\n", "1\n", "\n"]
    lines += ["data_particles\n", "\n", "loop_\n", "_rlnAngleRot #1\n", "_rlnAngleTilt #2\n"]
    lines += [f"{r:.6f} {t:.6f}\n" for r, t in zip(rot, tilt)]
    path.write_text("".join(lines))


def main() -> None:
    """Command line: score a STAR file's symmetry-expanded viewing directions and write ``nonuniformity.json``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--star", type=Path, required=True, help="particle STAR file with rlnAngleRot and rlnAngleTilt")
    parser.add_argument("--symmetry", default="C1", help="RELION point-group label of the poses' symmetry, e.g. C3")
    parser.add_argument("--orders", type=int, nargs="+", default=list(DEFAULT_ORDERS), help="HEALPix orders to bin at")
    parser.add_argument("--null-draws", type=int, default=DEFAULT_NULL_DRAWS, help="uniform samples for the floor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, help="benchmark config.yaml: add the Kent scenarios' population KL")
    parser.add_argument("--expanded-star", type=Path, help="write the expanded directions to this STAR file")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    orders = tuple(args.orders)
    vectors = read_viewing_directions(args.star)
    rows = nonuniformity_by_order(vectors, args.symmetry, orders, args.null_draws, args.seed)
    record: dict[str, Any] = {
        "star": str(args.star),
        "symmetry": args.symmetry,
        "n_particles": len(vectors),
        "n_operators": rotational_operators(args.symmetry).shape[0],
        "null_draws": args.null_draws,
        "orders": rows,
    }
    if args.config:
        record["kent_scenarios_population_kl_bits"] = kent_scenario_kl(args.config, orders)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "nonuniformity.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.expanded_star:
        write_expanded_star(args.expanded_star, expand_directions(vectors, args.symmetry))

    print(f"{args.star}: {len(vectors)} particles, symmetry {args.symmetry} ({record['n_operators']} operators)")
    print("order  KL expanded  uniform floor  above floor  KL raw   half-mass share  empty share")
    for row in rows:
        print(
            f"{row['healpix_order']:5d}  {row['expanded_kl_from_uniform_bits']:11.3f}  "
            f"{row['uniform_mean_kl_from_uniform_bits']:13.3f}  {row['expanded_kl_above_floor_bits']:11.3f}  "
            f"{row['unexpanded_kl_from_uniform_bits']:6.3f}  {row['expanded_sphere_fraction_with_half_of_mass']:15.4f}  "
            f"{row['expanded_directions_without_a_particle']:11.4f}"
        )
    for name, by_order in record.get("kent_scenarios_population_kl_bits", {}).items():
        print(f"{name} (population KL, bits): " + "  ".join(f"order {o}: {v:.3f}" for o, v in by_order.items()))


if __name__ == "__main__":
    main()
