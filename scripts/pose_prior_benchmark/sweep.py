"""The scenarios of the pose-prior sweep, generated from ``config.yaml``'s ``sweep`` section.

The sweep refines mixtures of two simulated pools of particles, a Kent cap and a uniform one, over the mixture's Kent
share, the noise level and the number of particles (``mix_pools.py``). Per noise level it simulates two pools of
``max(n_images)`` particles (the noise seed differs between the pools); a mixture takes the first ``round(share * n)``
particles of the Kent pool and the first ``n - that`` of the uniform pool. An optional ``offsets`` entry adds the
displaced copy of one Kent pool (``add_offsets.py``), refined like a mixture.

``sweep_scenarios`` returns scenario entries in the format of ``config.yaml``'s ``scenarios``: the pools (no variants,
so they only simulate), the mixtures and the offsets scenario. The Snakefile adds them to the scenarios it reads.
"""

from __future__ import annotations

from typing import Any

POOL_KINDS = ("kent", "uniform")
SWEEP_KEYWORD = "sweep"  # in run_scenarios: every mixture and the offsets scenario


def noise_tag(level: float) -> str:
    """A noise level in a scenario name: ``0.5`` -> ``0p5``, ``2.0`` -> ``2``."""
    return f"{level:g}".replace(".", "p")


def weight_tag(share: float) -> str:
    """A Kent share in a scenario name, in percent: ``0.33`` -> ``033``."""
    return f"{round(100 * share):03d}"


def count_tag(n_images: int) -> str:
    """A particle count in a scenario name, in thousands: ``50000`` -> ``50k``."""
    if n_images % 1000:
        raise ValueError(f"particle counts must be multiples of 1000, got {n_images}")
    return f"{n_images // 1000}k"


def pool_name(kind: str, noise_level: float) -> str:
    """The scenario that simulates the pool of ``kind`` at ``noise_level``."""
    return f"pool_{kind}_nz{noise_tag(noise_level)}"


def mixture_name(share: float, n_images: int, noise_level: float) -> str:
    """The scenario that refines a mixture of Kent share ``share`` of ``n_images`` particles."""
    return f"mix_w{weight_tag(share)}_n{count_tag(n_images)}_nz{noise_tag(noise_level)}"


def offsets_name(noise_level: float) -> str:
    """The scenario that refines the displaced copy of the Kent pool at ``noise_level``."""
    return f"offsets_kent_nz{noise_tag(noise_level)}"


def variants_of(share: float, variants: list[str]) -> list[str]:
    """The variants a mixture runs: without ``fixed_true`` for a uniform mixture, whose true prior is the uniform one."""
    return [v for v in variants if not (v == "fixed_true" and share == 0.0)]


def sweep_scenarios(sweep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Scenario entries of the sweep described by ``sweep`` (``config.yaml``'s ``sweep`` section)."""
    shares, noises, counts = sweep["kent_weights"], sweep["noise_levels"], sweep["n_images"]
    if any(not 0.0 <= share <= 1.0 for share in shares):
        raise ValueError(f"kent_weights must lie in [0, 1], got {shares}")
    pool_size = max(counts)
    seeds = sweep["pool_seeds"]
    scenarios: dict[str, dict[str, Any]] = {}
    for noise in noises:
        for kind in POOL_KINDS:
            poses = {"kind": "uniform"} if kind == "uniform" else {"kind": "kent", **sweep["kent"]}
            scenarios[pool_name(kind, noise)] = {
                "poses": poses,
                "dataset": {"noise_level": noise, "n_images": pool_size, "seed": seeds[kind]},
                "variants": [],
            }
    for noise in noises:
        for n_images in counts:
            for share in shares:
                kent = round(share * n_images)
                pools = [
                    {"label": kind, "scenario": pool_name(kind, noise), "count": count}
                    for kind, count in (("kent", kent), ("uniform", n_images - kent))
                    if count > 0
                ]
                scenarios[mixture_name(share, n_images, noise)] = {
                    "mixture_of": pools,
                    "variants": variants_of(share, sweep["variants"]),
                }
    offsets = sweep.get("offsets")
    if offsets:
        scenarios[offsets_name(offsets["noise_level"])] = {
            "shifted_from": pool_name("kent", offsets["noise_level"]),
            "offsets": {"sigma_angstrom": offsets["sigma_angstrom"], "seed": offsets["seed"]},
            "variants": list(sweep["variants"]),
        }
    return scenarios


def sweep_run_scenarios(sweep: dict[str, Any]) -> list[str]:
    """The scenarios a run refines for ``run_scenarios: [sweep]``: every mixture, then the offsets scenario."""
    names = [
        mixture_name(share, n_images, noise)
        for noise in sweep["noise_levels"]
        for n_images in sweep["n_images"]
        for share in sweep["kent_weights"]
    ]
    offsets = sweep.get("offsets")
    return names + ([offsets_name(offsets["noise_level"])] if offsets else [])
