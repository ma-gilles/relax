"""RELION's start-up particle table rebuilt from the particle STAR that ``relion_refine`` reads.

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

import numpy as np
import pandas as pd

_GLIBC_MODULUS = 2147483647


def glibc_rand_sequence(seed: int, count: int) -> np.ndarray:
    """Return the first ``count`` values of glibc ``rand()`` after ``srand(seed)``.

    glibc's default generator (``random_r.c``, TYPE_3): a 31-word state seeded by the
    Park-Miller recurrence, 310 discarded outputs, then the additive lagged-Fibonacci
    feedback ``r[i] = r[i-31] + r[i-3]`` (mod 2**32) with output ``r[i] >> 1``.
    ``srand(0)`` behaves as ``srand(1)``. The seed is taken as the ``unsigned int`` that
    RELION passes.
    """
    count = int(count)
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    seed = int(seed) & 0xFFFFFFFF
    if seed == 0:
        seed = 1
    word = seed - (1 << 32) if seed >= (1 << 31) else seed  # int32_t in glibc
    state = [word]
    for _ in range(1, 31):
        # C division truncates toward zero; use the magnitude to keep Python's // exact.
        hi = int(word / 127773)
        lo = word - hi * 127773
        word = 16807 * lo - 2836 * hi
        if word < 0:
            word += _GLIBC_MODULUS
        state.append(word)
    state = [value & 0xFFFFFFFF for value in state]
    for i in range(31, 34):
        state.append(state[i - 31])
    out = np.empty(count, dtype=np.int64)
    n_discard = 310
    for i in range(34, 344 + count):
        value = (state[i - 31] + state[i - 3]) & 0xFFFFFFFF
        state.append(value)
        if i >= 34 + n_discard:
            out[i - 344] = value >> 1
    return out


def relion_particle_order(particles: pd.DataFrame) -> np.ndarray:
    """Return input row indices in RELION's post-read order (stable byte-wise micrograph sort)."""
    n_rows = len(particles)
    if "rlnMicrographName" not in particles.columns:
        return np.arange(n_rows, dtype=np.int64)
    names = particles["rlnMicrographName"].astype(str).tolist()
    return np.asarray(sorted(range(n_rows), key=lambda row: names[row].encode()), dtype=np.int64)


def relion_random_subsets(existing, *, seed: int, n_particles: int) -> np.ndarray:
    """RELION's half-set labels for particles in RELION order (``divideParticlesInRandomHalves``)."""
    n_particles = int(n_particles)
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


def relion_scale_group_numbers(particles_in_order: pd.DataFrame) -> np.ndarray:
    """One-based RELION scale-group numbers for particles already in RELION order.

    Without rlnGroupName or rlnMicrographName every particle reads an empty
    micrograph name (Experiment::getMicrographName, exp_model.cpp:154-167) and
    so shares one group.
    """
    if "rlnGroupName" in particles_in_order.columns:
        group_names = particles_in_order["rlnGroupName"].astype(str).tolist()
    elif "rlnMicrographName" in particles_in_order.columns:
        group_names = [_post_job_micrograph_name(name) for name in particles_in_order["rlnMicrographName"].astype(str)]
    else:
        group_names = [""] * len(particles_in_order)
    numbers: dict[str, int] = {}
    return np.asarray([numbers.setdefault(name, len(numbers) + 1) for name in group_names], dtype=np.int64)


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
