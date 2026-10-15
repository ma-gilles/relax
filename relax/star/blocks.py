"""STAR files as blocks of text: RELION's run-file reader, and the ``data_general`` block a job writes back.

:func:`read_star_blocks` keeps every value a string (the run-file and continuation readers convert them);
:func:`initial_model_data_blocks` assembles InitialModel's data STAR, which starts with the input's
``data_general`` block as Refine3D's and Class3D's run files do.
"""

from __future__ import annotations

import shlex


def read_star_blocks(path) -> dict:
    """Parse a STAR file into ``{block: {label: str}}`` (list) or ``{block: {label: [str]}}`` (loop).

    Values stay strings so numbers convert without the last-bit error of pandas' default
    float parser; :func:`_floats` converts with NumPy's correctly rounded ``strtod``.
    """

    blocks: dict = {}
    name = None
    in_loop = False
    labels: list[str] = []
    rows: list[list[str]] = []

    def _close_loop():
        nonlocal in_loop, labels, rows
        if in_loop and name is not None:
            columns = list(zip(*rows)) if rows else [()] * len(labels)
            blocks[name] = {label: list(col) for label, col in zip(labels, columns)}
        in_loop = False
        labels, rows = [], []

    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("data_"):
                _close_loop()
                name = line[5:]
                blocks[name] = {}
                continue
            if line == "loop_":
                in_loop = True
                labels, rows = [], []
                continue
            if line.startswith("_"):
                tokens = shlex.split(line, comments=False)
                label = tokens[0][1:]
                if in_loop and not rows:
                    labels.append(label)
                else:
                    _close_loop()
                    blocks[name][label] = tokens[1] if len(tokens) > 1 else ""
                continue
            if in_loop:
                tokens = shlex.split(line) if '"' in line or "'" in line else line.split()
                if len(tokens) != len(labels):
                    raise ValueError(f"{path}: block {name} row has {len(tokens)} fields, expected {len(labels)}")
                rows.append(tokens)
    _close_loop()
    return blocks


def input_general_block(input_star) -> dict | None:
    """The input STAR's ``data_general`` block as :func:`read_star_blocks` text, or None without one.

    A subtomogram particle STAR says it holds 2D stacks there (``rlnTomoSubTomosAre2DStacks``,
    ``ParticleSet::read``), and relion_refine writes the block back to every ``run_itNNN_data.star``.
    Both data-STAR writers write it back unchanged (:class:`RunFileWriter` for Refine3D and Class3D,
    :func:`initial_model_data_blocks` for InitialModel), so a job's output starts the next job.
    """

    return read_star_blocks(input_star).get("general") or None


def initial_model_data_blocks(input_star, optics, particles) -> dict:
    """The blocks of InitialModel's data STAR for recovar's ``write_star_blocks``, in RELION's order.

    ``optics`` and ``particles`` are the output tables (``optics`` None: the one-table RELION 3.0 form).
    ``input_star`` is a subtomogram run's particle STAR (``TomoDataset.particles_star``; None for single
    particles): its ``data_general`` block (:func:`input_general_block`) goes first.
    """

    if optics is None:
        return {"data_": particles}
    blocks = {"data_optics": optics.copy(), "data_particles": particles}
    general = None if input_star is None else input_general_block(input_star)
    if general:
        import pandas as pd

        columns = {f"_{label}": value for label, value in general_columns(general).items()}
        blocks = {"data_general": pd.DataFrame(columns), **blocks}
    return blocks


def general_columns(general: dict) -> dict:
    """``data_general`` as one-row loop columns, the form relion_refine writes, whichever form the input had."""

    return {label: value if isinstance(value, list) else [value] for label, value in general.items()}
