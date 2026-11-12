"""The scenarios of the pose-prior sweep, generated from ``config.yaml``'s ``sweep`` section.

The sweep refines mixtures of two simulated pools of particles, a Kent cap and a uniform one, over the mixture's Kent
share, the noise level and the number of particles (``mix_pools.py``). Per noise level it simulates two pools of
``max(n_images)`` particles (the noise seed differs between the pools); a mixture takes the first ``round(share * n)``
particles of the Kent pool and the first ``n - that`` of the uniform pool. An optional ``offsets`` entry adds a subset of
the cells (noise levels, Kent shares and particle counts of the sweep) with offsets: it displaces every particle of
the pools it needs (``add_offsets.py``, a seed per pool kind) and mixes the displaced pools like the centred ones, so a
cell with offsets holds the same particles as its centred counterpart, each shifted.

The uniform pool is a Kent pool of concentration and ovalness 0 (exactly uniform viewing directions), not the
simulator's ``uniform`` option: that option draws its CTF parameters from another real data set than the Kent option
does (amplitude contrast 0.07 against 0.1, other defoci), and a mixture needs one CTF distribution in both pools.

``sweep_scenarios`` returns scenario entries in the format of ``config.yaml``'s ``scenarios``: the pools (no variants,
so they only simulate), the mixtures, the displaced pools and the mixtures of those. The Snakefile adds them to the
scenarios it reads.
"""

from __future__ import annotations

from typing import Any

POOL_KINDS = ("kent", "uniform")
SWEEP_KEYWORD = "sweep"  # in run_scenarios: every mixture, with and without offsets


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


def offsets_pool_name(kind: str, noise_level: float) -> str:
    """The scenario that displaces the pool of ``kind`` at ``noise_level``."""
    return f"pool_{kind}_off_nz{noise_tag(noise_level)}"


def offsets_mixture_name(share: float, n_images: int, noise_level: float) -> str:
    """The scenario that refines the mixture of Kent share ``share`` of ``n_images`` displaced particles."""
    return f"off_w{weight_tag(share)}_n{count_tag(n_images)}_nz{noise_tag(noise_level)}"


def variants_of(share: float, variants: list[str]) -> list[str]:
    """The variants a mixture runs: without ``fixed_true`` for a uniform mixture, whose true prior is the uniform one."""
    return [v for v in variants if not (v == "fixed_true" and share == 0.0)]


def offsets_cells(sweep: dict[str, Any]) -> list[tuple[float, int, float]]:
    """The (Kent share, particle count, noise level) of every cell refined with offsets, in the order of ``noise``,
    ``count``, ``share``; empty without an ``offsets`` entry. Each cell must be one the sweep has centred."""
    offsets = sweep.get("offsets")
    if not offsets:
        return []
    for key in ("kent_weights", "noise_levels", "n_images"):
        extra = [value for value in offsets[key] if value not in sweep[key]]
        if extra:
            raise ValueError(f"offsets {key} {extra} are not in the sweep's {key} {sweep[key]}")
    return [
        (share, n_images, noise)
        for noise in offsets["noise_levels"]
        for n_images in offsets["n_images"]
        for share in offsets["kent_weights"]
    ]


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
            poses = {"kind": "kent", **sweep["kent"], **({"kappa": 0.0, "beta": 0.0} if kind == "uniform" else {})}
            scenarios[pool_name(kind, noise)] = {
                "poses": poses,
                "dataset": {"noise_level": noise, "n_images": pool_size, "seed": seeds[kind]},
                "variants": [],
            }
    for noise in noises:
        for n_images in counts:
            for share in shares:
                scenarios[mixture_name(share, n_images, noise)] = {
                    "mixture_of": _pools_of(pool_name, share, n_images, noise),
                    "variants": variants_of(share, sweep["variants"]),
                }
    cells = offsets_cells(sweep)
    if cells:
        offsets = sweep["offsets"]
        for noise in offsets["noise_levels"]:
            for kind in POOL_KINDS:
                scenarios[offsets_pool_name(kind, noise)] = {
                    "shifted_from": pool_name(kind, noise),
                    "offsets": {"sigma_angstrom": offsets["sigma_angstrom"], "seed": offsets["pool_seeds"][kind]},
                    "variants": [],
                }
        for share, n_images, noise in cells:
            scenarios[offsets_mixture_name(share, n_images, noise)] = {
                "mixture_of": _pools_of(offsets_pool_name, share, n_images, noise),
                "variants": variants_of(share, sweep["variants"]),
            }
    return scenarios


def _pools_of(name_of: Any, share: float, n_images: int, noise: float) -> list[dict[str, Any]]:
    """The ``mixture_of`` entry of a mixture of Kent share ``share``: the pools ``name_of(kind, noise)`` names, a
    pool that contributes no particle left out."""
    kent = round(share * n_images)
    return [
        {"label": kind, "scenario": name_of(kind, noise), "count": count}
        for kind, count in (("kent", kent), ("uniform", n_images - kent))
        if count > 0
    ]


def sweep_run_scenarios(sweep: dict[str, Any]) -> list[str]:
    """The scenarios a run refines for ``run_scenarios: [sweep]``: every mixture, then every mixture with offsets."""
    names = [
        mixture_name(share, n_images, noise)
        for noise in sweep["noise_levels"]
        for n_images in sweep["n_images"]
        for share in sweep["kent_weights"]
    ]
    return names + [offsets_mixture_name(*cell) for cell in offsets_cells(sweep)]
