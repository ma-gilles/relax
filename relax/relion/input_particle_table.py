"""RELION particle-table ordering, source selection and physical/optics group layouts.

RELION 5.0.1 (source ``f2c1a38``) derives three pieces of per-particle state before its
first expectation step, all from its ``--i`` STAR and ``--random_seed``:

* particle order: ``Experiment::read`` stably sorts the input rows on
  ``rlnMicrographName`` (``exp_model.cpp:900-901``, ``MetaDataTable::newSort`` with a
  ``std::string`` comparator);
* random halves: ``Experiment::divideParticlesInRandomHalves`` keeps an input
  ``rlnRandomSubset`` when every row carries one, otherwise calls ``srand(random_seed)``
  and assigns ``rand() % 2 + 1`` to each particle in that order
  (``exp_model.cpp:261-403``; called from ``ml_optimiser.cpp:2835`` and
  ``ml_optimiser_mpi.cpp:970``);
* scale groups: ``rlnGroupName`` when present, otherwise the part of
  ``rlnMicrographName`` after its pipeline job directory
  (``decomposePipelineFileName``, ``filename.cpp:614``), numbered from one in order of
  first appearance (``exp_model.cpp:926-965``).

The noise-estimate source order (subset 1, then subset 2, each in this order,
``exp_model.cpp:402``) follows from the table. ``build_relion_start_particle_table``
returns rows in RELION's order carrying ``rlnRandomSubset`` and ``rlnGroupNumber``, the
layout of RELION's ``run_it000_data.star``, so it can stand in for that file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from relax.helpers.relion_random import check_init_random_generator_seeds, glibc_first_rand, glibc_rand_sequence

logger = logging.getLogger(__name__)


def relion_class3d_seed_classes(expectation_order, random_seed: int, n_classes: int) -> np.ndarray:
    """Each input row's class in RELION's first Class3D iteration from one reference (0-based).

    ``relion_refine --K K`` with a single ``--ref`` map copies it to every class and scores each
    particle against one random class in the first iteration (``do_generate_seeds``,
    ml_model.cpp:1007-1010): for the particle at sorted position ``j``,
    ``init_random_generator(random_seed + j)`` then ``rand() % K`` (ml_optimiser.cpp:4626-4633,
    :4880-4898). ``expectation_order[j]`` is the input row at sorted position ``j``
    (``relax.helpers.expected_accuracy.relion_class3d_trial_layout``).
    """
    order = np.asarray(expectation_order, dtype=np.int64).reshape(-1)
    seeds = int(random_seed) + np.arange(order.size, dtype=np.int64)
    check_init_random_generator_seeds(seeds)
    classes = np.empty(order.size, dtype=np.int64)
    classes[order] = glibc_first_rand(seeds) % int(n_classes)
    return classes


def relion_particle_order(particles: pd.DataFrame) -> np.ndarray:
    """Return input row indices in RELION's post-read order (stable byte-wise micrograph sort)."""
    n_rows = len(particles)
    if "rlnMicrographName" not in particles.columns:
        return np.arange(n_rows, dtype=np.int64)
    names = particles["rlnMicrographName"].astype(str).tolist()
    return np.asarray(sorted(range(n_rows), key=lambda row: names[row].encode()), dtype=np.int64)


def relion_random_subsets(existing, *, seed: int, n_particles: int) -> np.ndarray:
    """RELION's half-set labels for particles in RELION order (``divideParticlesInRandomHalves``)."""
    if existing is not None:
        existing = np.asarray(existing, dtype=np.int64).reshape(-1)
        if existing.size != n_particles:
            raise ValueError(f"rlnRandomSubset has {existing.size} rows for {n_particles} particles")
        nonzero = existing != 0
        if nonzero.all():
            if not np.isin(existing, (1, 2)).all():
                raise ValueError("rlnRandomSubset values must be 1 or 2")
            subsets = existing
        elif nonzero.any():
            raise ValueError("rlnRandomSubset values must be all zero or all non-zero")
        else:
            existing = None
    if existing is None:
        subsets = glibc_rand_sequence(seed, n_particles) % 2 + 1
    if not ((subsets == 1).any() and (subsets == 2).any()):
        raise ValueError("RELION requires both random halves to be non-empty")
    return np.asarray(subsets, dtype=np.int64)


def relion_split_draw_count(existing, *, n_particles: int) -> int:
    """The ``rand()`` values ``divideParticlesInRandomHalves`` takes after its ``srand(random_seed)``.

    One per particle when it draws the halves (no ``rlnRandomSubset``, or all zero; non-helical,
    exp_model.cpp:378), none when the input carries them. RELION's MPI leader seeded the same stream
    just before (ml_optimiser_mpi.cpp:827), so its later draws start after these
    (relax.reconstruction.solvent_mask).
    """
    if existing is None:
        return int(n_particles)
    return 0 if np.any(np.asarray(existing, dtype=np.int64) != 0) else int(n_particles)


def _post_job_micrograph_name(name: str) -> str:
    """``fn_post`` of RELION's ``decomposePipelineFileName`` (``filename.cpp:614``)."""
    slash = 0
    for _ in range(21):
        found = name.find("/", slash + 1)
        # std::string::npos + 1 wraps to 0, so the last check inspects the name's start.
        start = 0 if found < 0 else found + 1
        if name[start : start + 3] == "job" and len(name) >= start + 6 and name[start + 3 : start + 6].isdigit():
            second = name.find("/", start + 5)
            if second < 0:
                second = len(name) - 1
            return name[second + 1 :]
        if found < 0:
            return name
        slash = found
    raise ValueError(f"more than 20 directories deep in pipeline filename: {name}")


def _relion_group_name_per_particle(particles_in_order: pd.DataFrame) -> list[str]:
    """Each particle's RELION scale-group name (exp_model.cpp:926-955).

    Without rlnGroupName or rlnMicrographName every particle reads an empty
    micrograph name (Experiment::getMicrographName, exp_model.cpp:154-167) and
    so shares one group. Subtomogram particles (rlnTomoName, no micrograph) read
    their tomogram's name there.
    """
    if "rlnGroupName" in particles_in_order.columns:
        group_names = particles_in_order["rlnGroupName"].astype(str).tolist()
    elif "rlnMicrographName" in particles_in_order.columns:
        group_names = [_post_job_micrograph_name(name) for name in particles_in_order["rlnMicrographName"].astype(str)]
    elif "rlnTomoName" in particles_in_order.columns:
        group_names = [_post_job_micrograph_name(name) for name in particles_in_order["rlnTomoName"].astype(str)]
    else:
        group_names = [""] * len(particles_in_order)
    return group_names


def relion_scale_group_numbers(particles_in_order: pd.DataFrame) -> np.ndarray:
    """One-based RELION scale-group numbers for particles already in RELION order, by first appearance of
    their names (:func:`_relion_group_name_per_particle`)."""
    numbers: dict[str, int] = {}
    return np.asarray(
        [numbers.setdefault(name, len(numbers) + 1) for name in _relion_group_name_per_particle(particles_in_order)],
        dtype=np.int64,
    )


def build_relion_start_particle_table(particles: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Return the input particles in RELION's order with its ``rlnRandomSubset`` and ``rlnGroupNumber``."""
    order = relion_particle_order(particles)
    table = particles.iloc[order].reset_index(drop=True).copy()
    existing = table["rlnRandomSubset"].to_numpy() if "rlnRandomSubset" in table.columns else None
    table["rlnRandomSubset"] = relion_random_subsets(existing, seed=seed, n_particles=len(table))
    table["rlnGroupNumber"] = relion_scale_group_numbers(table)
    return table


def _particles_block_bounds(lines: list[str]) -> tuple[int, int, int]:
    """Return (loop header start, first data row, end) of the ``data_particles`` loop."""
    try:
        block = next(i for i, line in enumerate(lines) if line.strip() == "data_particles")
    except StopIteration:
        raise ValueError("STAR file has no data_particles block") from None
    loop = next(i for i in range(block + 1, len(lines)) if lines[i].strip() == "loop_")
    first = loop + 1
    while first < len(lines) and lines[first].strip().startswith("_rln"):
        first += 1
    end = first
    while end < len(lines) and lines[end].strip() and not lines[end].strip().startswith("data_"):
        end += 1
    return loop + 1, first, end


def write_relion_start_particle_star(input_star, output_star, *, seed: int) -> pd.DataFrame:
    """Write the start-up table as STAR text, keeping every input field's text unchanged.

    Rows are reordered and ``rlnRandomSubset``/``rlnGroupNumber`` set; all other tokens,
    and the blocks before ``data_particles`` (e.g. ``data_optics``), are copied verbatim,
    so any reader parses exactly the values it would parse from the input.
    """
    import starfile

    particles = starfile.read(input_star, always_dict=True)["particles"]
    table = build_relion_start_particle_table(particles, seed=seed)
    order = relion_particle_order(particles)
    lines = open(input_star).read().splitlines()
    header_start, first, end = _particles_block_bounds(lines)
    labels = [lines[i].split()[0] for i in range(header_start, first)]
    rows = [lines[i].split() for i in range(first, end)]
    if len(rows) != len(particles) or any(len(row) != len(labels) for row in rows):
        raise ValueError(f"cannot map {input_star} data rows one-to-one onto its particle table")
    for label in ("_rlnRandomSubset", "_rlnGroupNumber"):
        if label not in labels:
            labels.append(label)
            rows = [row + [""] for row in rows]
    subset_col, group_col = labels.index("_rlnRandomSubset"), labels.index("_rlnGroupNumber")
    out = lines[: header_start - 1] + ["loop_"] + [f"{label} #{i + 1}" for i, label in enumerate(labels)]
    for new_row, source_row in enumerate(order):
        row = list(rows[source_row])
        row[subset_col] = str(int(table["rlnRandomSubset"].iat[new_row]))
        row[group_col] = str(int(table["rlnGroupNumber"].iat[new_row]))
        out.append("\t".join(row))
    out += [""] + lines[end:]
    with open(output_star, "w") as handle:
        handle.write("\n".join(out).rstrip("\n") + "\n")
    return table


class RelionHalfsetSource(NamedTuple):
    """Parsed half-set table and its admitted AutoRefine optics geometry.

    ``tables`` is the original STAR payload, consumed for its particle rows.
    It retains the same parsed optics/particle storage as the command's prior
    ``relion_data`` local; no extra retention field or array copy is needed.
    """

    tables: dict
    optics_image_sizes: np.ndarray | None
    optics_pixel_sizes: np.ndarray | None


def read_relion_halfset_source(path, *, n_classes: int, log) -> RelionHalfsetSource:
    """Read half-set STAR input and admit the K1 optics geometry it supplies.

    Particle-row mapping, noise source order, CTF adaptation and dataset halves
    retain their existing owners and execution boundaries.
    """
    import starfile as _starfile

    relion_optics_image_sizes = None
    relion_optics_pixel_sizes = None
    relion_data = _starfile.read(path)
    relion_data["particles"]  # Admit the required particle block before optics metadata.
    if n_classes == 1:
        relion_optics = relion_data.get("optics") if isinstance(relion_data, dict) else None
        if relion_optics is None:
            raise SystemExit("K=1 RELION parity requires an optics table in --relion_half_sets")
        required_optics_columns = {"rlnImageSize", "rlnImagePixelSize"}
        missing_optics_columns = required_optics_columns.difference(relion_optics.columns)
        if missing_optics_columns:
            raise SystemExit(
                "K=1 RELION parity optics table is missing "
                + ", ".join(sorted(missing_optics_columns))
            )
        relion_optics_image_sizes = np.asarray(
            relion_optics["rlnImageSize"],
            dtype=np.int64,
        )
        relion_optics_pixel_sizes = np.asarray(
            relion_optics["rlnImagePixelSize"],
            dtype=np.float64,
        )
        log.info(
            "RELION per-optics image geometry: sizes=%s pixel_sizes=%s",
            relion_optics_image_sizes.tolist(),
            relion_optics_pixel_sizes.tolist(),
        )
    return RelionHalfsetSource(
        tables=relion_data,
        optics_image_sizes=relion_optics_image_sizes,
        optics_pixel_sizes=relion_optics_pixel_sizes,
    )


class GroupParticleSource(NamedTuple):
    """Selected authoritative particle table and its optional STAR provenance."""

    particles: pd.DataFrame | None
    path: Path | None


class ParticleGroupLayout(NamedTuple):
    """RELION group labels in RECOVAR half order plus the full model axis."""

    group_ids_per_half: tuple[np.ndarray, np.ndarray]
    particle_ids_per_half: tuple[np.ndarray, np.ndarray]
    optics_group_ids_per_half: tuple[np.ndarray, np.ndarray]
    n_groups: int
    n_optics_groups: int
    source: str


class PreparedParticleGroups(NamedTuple):
    """Authoritative source paired with group axes in active dataset half order."""

    source: GroupParticleSource
    layout: ParticleGroupLayout
    # Each group's RELION name, group 1 first (model_groups rlnGroupName, ml_model.cpp:780, 1146).
    group_names: tuple[str, ...]


def prepare_particle_group_layout(
    our_particles,
    half1_idx,
    half2_idx,
    *,
    halfset_particles=None,
    halfset_source=None,
    replay_dirs=(),
    init_relion_iteration=0,
) -> PreparedParticleGroups:
    """Select authoritative RELION groups and map them into active half rows.

    The half-set table is authoritative when it carries
    ``rlnGroupNumber``. Otherwise replay directories are searched in caller order.
    Without an authoritative source, groups are
    numbered from the RECOVAR input table as relion_refine does: rlnGroupName,
    else the micrograph name, by first appearance in the micrograph-sorted
    order (exp_model.cpp:900-901, 926-965).  An input ``rlnGroupNumber`` is
    ignored, because relion_refine never reads it (exp_model.cpp:963-965).
    Group numbers remain on RELION's full model axis even when a half-set does
    not contain the highest-numbered group.
    See docs/math/relion_refinement_algorithm.md#particle-group-identity.
    """

    if halfset_particles is not None and "rlnGroupNumber" in halfset_particles.columns:
        group_source = GroupParticleSource(
            halfset_particles, None if halfset_source is None else Path(halfset_source).resolve(),
        )
    else:
        for replay_dir in replay_dirs:
            replay_groups = _read_replay_group_particles(
                replay_dir,
                init_relion_iteration=init_relion_iteration,
            )
            if replay_groups is not None:
                group_source = GroupParticleSource(*replay_groups)
                break
        else:
            group_source = GroupParticleSource(None, None)
    relion_particles = group_source.particles

    from relax.relion import relion_metadata

    if relion_particles is not None:
        group_particles = relion_particles
        source = "supplied RELION data STAR"
    else:
        group_particles = our_particles.iloc[relion_particle_order(our_particles)].reset_index(drop=True)
        group_particles = group_particles.assign(rlnGroupNumber=relion_scale_group_numbers(group_particles))
        source = "RELION scale groups numbered from the RECOVAR input particles STAR"

    our_rows = relion_metadata.particle_identity_rows(our_particles, label="RECOVAR input STAR")
    group_rows = relion_metadata.particle_identity_rows(group_particles, label=source)
    if set(our_rows) != set(group_rows):
        missing = len(set(our_rows) - set(group_rows))
        extra = len(set(group_rows) - set(our_rows))
        raise ValueError(
            f"{source} and RECOVAR input STAR do not contain the same "
            f"rlnImageName/stack identities (missing={missing}, extra={extra})",
        )

    group_numbers_source = np.asarray(group_particles["rlnGroupNumber"], dtype=np.int64).reshape(-1)
    if group_numbers_source.shape != (len(group_particles),):
        raise ValueError(
            f"rlnGroupNumber length {group_numbers_source.size} does not match "
            f"{source} particle count {len(group_particles)}",
        )
    if group_numbers_source.size and int(np.min(group_numbers_source)) < 1:
        raise ValueError("RELION rlnGroupNumber values must be 1-based positive integers")

    group_ids_by_our_row = np.asarray(
        [group_numbers_source[group_rows[identity]] - 1 for identity in our_rows],
        dtype=np.int64,
    )
    particle_ids_by_our_row = np.asarray(
        [group_rows[identity] for identity in our_rows],
        dtype=np.int64,
    )
    if "rlnOpticsGroup" in group_particles.columns:
        optics_numbers_source = np.asarray(group_particles["rlnOpticsGroup"], dtype=np.int64).reshape(-1)
        if optics_numbers_source.shape != (len(group_particles),):
            raise ValueError(
                f"rlnOpticsGroup length {optics_numbers_source.size} does not match "
                f"{source} particle count {len(group_particles)}",
            )
        if optics_numbers_source.size and int(np.min(optics_numbers_source)) < 1:
            raise ValueError("RELION rlnOpticsGroup values must be 1-based positive integers")
    else:
        optics_numbers_source = np.ones(len(group_particles), dtype=np.int64)
    optics_group_ids_by_our_row = np.asarray(
        [optics_numbers_source[group_rows[identity]] - 1 for identity in our_rows],
        dtype=np.int64,
    )
    half_indices = []
    for label, values in (("half1_idx", half1_idx), ("half2_idx", half2_idx)):
        indices = np.asarray(values, dtype=np.int64).reshape(-1)
        if indices.size and (int(np.min(indices)) < 0 or int(np.max(indices)) >= len(our_particles)):
            raise ValueError(f"{label} contains an out-of-bounds RECOVAR particle row")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{label} contains duplicate RECOVAR particle rows")
        half_indices.append(indices)
    if np.intersect1d(half_indices[0], half_indices[1]).size:
        raise ValueError("half1_idx and half2_idx overlap")

    n_groups = int(np.max(group_numbers_source)) if group_numbers_source.size else 0
    n_optics_groups = int(np.max(optics_numbers_source)) if optics_numbers_source.size else 0
    layout = ParticleGroupLayout(
        group_ids_per_half=(
            np.asarray(group_ids_by_our_row[half_indices[0]], dtype=np.int64),
            np.asarray(group_ids_by_our_row[half_indices[1]], dtype=np.int64),
        ),
        particle_ids_per_half=(
            np.asarray(particle_ids_by_our_row[half_indices[0]], dtype=np.int64),
            np.asarray(particle_ids_by_our_row[half_indices[1]], dtype=np.int64),
        ),
        optics_group_ids_per_half=(
            np.asarray(optics_group_ids_by_our_row[half_indices[0]], dtype=np.int64),
            np.asarray(optics_group_ids_by_our_row[half_indices[1]], dtype=np.int64),
        ),
        n_groups=n_groups,
        n_optics_groups=n_optics_groups,
        source=source,
    )

    names_by_number: dict[int, str] = {}
    for number, name in zip(group_numbers_source.tolist(), _relion_group_name_per_particle(group_particles)):
        names_by_number.setdefault(int(number), name)
    group_names = tuple(names_by_number.get(g, "") for g in range(1, n_groups + 1))
    return PreparedParticleGroups(source=group_source, layout=layout, group_names=group_names)


def _read_replay_group_particles(relion_dir, *, init_relion_iteration=0):
    """Load the first authoritative replay data STAR carrying group labels."""

    if relion_dir is None:
        return None
    import starfile as _starfile

    relion_dir = Path(relion_dir).resolve()
    preferred = relion_dir / f"run_it{int(init_relion_iteration):03d}_data.star"
    iter0 = relion_dir / "run_it000_data.star"
    candidates = [preferred]
    if iter0 != preferred:
        candidates.append(iter0)
    candidates.extend(sorted(relion_dir.glob("run_it*_data.star")))
    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        data = _starfile.read(str(path))
        particles = data["particles"] if isinstance(data, dict) else data
        if "rlnGroupNumber" in particles.columns:
            return particles, path
    return None




class ParticleLayout(NamedTuple):
    """Source rows and the paired half-1 accuracy frame of one refinement run.

    Half rows index the loaded input dataset. Accuracy particle IDs index the
    RELION table used by expected-accuracy CTF metadata, or the Class3D source
    ordering. Base order is used for replay; fresh physical order has explicit
    identity trials. See ``docs/math/relion_refinement_algorithm.md#particle-row-layout``.
    """

    half1_rows: np.ndarray
    half2_rows: np.ndarray
    accuracy_base_order_local: np.ndarray | None
    accuracy_trial_order_local: np.ndarray | None
    accuracy_optics_group_ids: np.ndarray | None
    accuracy_particle_ids: np.ndarray


## Once again, this seems to try to match to recovar things that may be hurting more htan helping, would it be easier to clean this up, get away from recovar backend ?
def _map_relion_half_rows(
    our_particles,
    relion_particles,
    *,
    random_seed=None,
    first_iteration=1,
):
    """Map RELION particle rows onto RECOVAR's half-local ordering.

    Supplying ``random_seed`` selects fresh AutoRefine semantics: RELION
    5.0.1's paired mt19937 half shuffle, followed by a stable numeric
    optics-group sort. Continuation and replay callers omit the seed
    and retain their existing row order.
    """
    from relax.relion import relion_metadata

    our_row_by_identity = relion_metadata.particle_identity_rows(
        our_particles,
        label="RECOVAR input STAR",
    )
    relion_row_by_identity = relion_metadata.particle_identity_rows(
        relion_particles,
        label="RELION data STAR",
    )
    if set(our_row_by_identity) != set(relion_row_by_identity):
        raise ValueError(
            "RELION and RECOVAR STAR files do not contain the same "
            "rlnImageName/stack identities"
        )
    our_identities = list(our_row_by_identity)
    relion_identities = list(relion_row_by_identity)

    relion_subsets = np.asarray(relion_particles["rlnRandomSubset"], dtype=np.int64)
    our_relion_rows = np.asarray(
        [relion_row_by_identity[identity] for identity in our_identities],
        dtype=np.int64,
    )
    if random_seed is None:
        our_subsets = relion_subsets[our_relion_rows]
        half1_idx = np.flatnonzero(our_subsets == 1).astype(np.int64)
        half2_idx = np.flatnonzero(our_subsets == 2).astype(np.int64)
        half1_local_by_our_row = {
            int(our_row): local for local, our_row in enumerate(half1_idx)
        }
        half1_base_order_local = np.asarray(
            [
                half1_local_by_our_row[
                    our_row_by_identity[relion_identities[relion_row]]
                ]
                for relion_row in np.flatnonzero(relion_subsets == 1)
            ],
            dtype=np.int64,
        )
    else:
        from relax.helpers.expected_accuracy import relion_auto_refine_half_orders

        relion_optics = (
            np.asarray(relion_particles["rlnOpticsGroup"], dtype=np.int64)
            if "rlnOpticsGroup" in relion_particles.columns
            else None
        )
        ordered_relion_rows = relion_auto_refine_half_orders(
            relion_subsets,
            int(random_seed),
            int(first_iteration),
            optics_group_ids=relion_optics,
        )
        half1_idx, half2_idx = (
            np.asarray(
                [
                    our_row_by_identity[relion_identities[relion_row]]
                    for relion_row in order
                ],
                dtype=np.int64,
            )
            for order in ordered_relion_rows
        )
        # The dataset is already in RELION's physical order. Expected
        # accuracy consumes an explicit identity trial order rather than
        # replaying the shuffle through an inverse permutation.
        half1_base_order_local = None
    half1_particle_ids = our_relion_rows[half1_idx]
    if "rlnOpticsGroup" in relion_particles.columns:
        relion_optics = np.asarray(relion_particles["rlnOpticsGroup"], dtype=np.int64)
        half1_optics_group_ids = relion_optics[half1_particle_ids]
    else:
        half1_optics_group_ids = np.zeros(half1_idx.size, dtype=np.int64)
    return (
        half1_idx,
        half2_idx,
        half1_base_order_local,
        half1_optics_group_ids,
        half1_particle_ids,
    )



def prepare_relion_halfset_layout(
    our_particles: pd.DataFrame,
    relion_particles: pd.DataFrame,
    *,
    random_seed: int | None = None,
    first_iteration: int = 1,
) -> ParticleLayout:
    """Prepare paired-half source rows and their expected-accuracy metadata.

    The mapping helper releases its identity dictionaries before allocating the
    fresh identity trial vector, preserving the original metadata lifetime.
    """
    half1, half2, base_order, optics, particle_ids = _map_relion_half_rows(
        our_particles, relion_particles, random_seed=random_seed, first_iteration=first_iteration,
    )
    return ParticleLayout(
        half1_rows=half1,
        half2_rows=half2,
        accuracy_base_order_local=base_order,
        accuracy_trial_order_local=(np.arange(half1.size, dtype=np.int64) if random_seed is not None else None),
        accuracy_optics_group_ids=optics,
        accuracy_particle_ids=particle_ids,
    )


def prepare_class3d_particle_layout(
    our_particles: pd.DataFrame,
    *,
    n_particles: int,
    random_seed: int,
    init_relion_iteration: int,
) -> ParticleLayout:
    """Select all-data rows and RELION's whole-vector Class3D accuracy trials."""
    # RELION Class3D refines all particles once; the second accumulator stays empty.
    half1_idx = np.arange(n_particles, dtype=np.int64)
    half2_idx = np.empty(0, dtype=np.int64)
    logger.info(
        "Using RELION Class3D all-data split: %d particles + empty second accumulator",
        len(half1_idx),
    )
    # RELION's whole-vector Class3D shuffle picks the expected-accuracy trials.
    from relax.helpers.expected_accuracy import relion_class3d_trial_layout

    (
        expected_accuracy_half1_trial_order_local,
        expected_accuracy_half1_particle_ids,
    ) = relion_class3d_trial_layout(
        relion_particle_order(our_particles),
        int(random_seed),
        first_iteration=max(1, int(init_relion_iteration) + 1),
        optics_group_ids=(
            np.asarray(our_particles["rlnOpticsGroup"], dtype=np.int64)
            if "rlnOpticsGroup" in our_particles.columns
            else None
        ),
    )
    return ParticleLayout(
        half1_rows=half1_idx,
        half2_rows=half2_idx,
        accuracy_base_order_local=None,
        accuracy_trial_order_local=expected_accuracy_half1_trial_order_local,
        accuracy_optics_group_ids=None,
        accuracy_particle_ids=expected_accuracy_half1_particle_ids,
    )
