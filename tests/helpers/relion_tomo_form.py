"""Rewrite a simulated RELION 5 subtomogram project in the form RELION's own tools write.

RECOVAR's simulator (``recovar.simulation.relion_tomo``) writes every STAR block as a ``loop_`` and every
label the reader uses. ``relion_tomo_subtomo`` and RELION's tilt-series jobs write the same information
differently (tests/fixtures/relion5_tomo_tutorial has their text):

- ``data_general`` is a list of label-value pairs, not a loop, under a ``# version 50001`` comment;
- the particle STAR has no ``rlnRandomSubset`` (relion_refine draws the halves);
- a tilt-series STAR leaves out labels at their default: ``rlnTomoXTilt`` (0) and ``rlnCtfScalefactor`` (1).

:func:`write_relion_form` writes a copy of a simulated project with those forms. The images are linked, not
copied, and every value RELION would read is the same as in the simulated project.
"""

from __future__ import annotations

import os
from pathlib import Path

RELION_GENERAL = "\n# version 50001\n\ndata_general\n\n_rlnTomoSubTomosAre2DStacks                       1\n \n"
SIMULATED_GENERAL = "data_general\n\nloop_\n_rlnTomoSubTomosAre2DStacks\n1\n"


def _loop(lines, block):
    start = next(i for i, line in enumerate(lines) if line.strip() == f"data_{block}")
    labels = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "loop_") + 1
    first = labels
    while lines[first].strip().startswith("_rln"):
        first += 1
    end = first
    while end < len(lines) and lines[end].strip() and not lines[end].startswith("data_"):
        end += 1
    return labels, first, end


def drop_loop_column(text: str, block: str, label: str, *, only_if_all: str | None = None) -> str:
    """``text`` without column ``label`` of ``data_<block>``'s loop.

    With ``only_if_all`` the column is dropped only when every row holds that number (RELION's default
    for the absent label), and the text is returned unchanged otherwise.
    """

    lines = text.split("\n")
    labels, first, end = _loop(lines, block)
    names = [lines[i].split()[0] for i in range(labels, first)]
    if label not in names:
        return text
    column = names.index(label)
    rows = [lines[i].split() for i in range(first, end)]
    if only_if_all is not None and any(float(row[column]) != float(only_if_all) for row in rows):
        return text
    kept = [lines[labels + k] for k in range(len(names)) if k != column]
    body = [" ".join(token for k, token in enumerate(row) if k != column) for row in rows]
    return "\n".join(lines[:labels] + kept + body + lines[end:])


def general_as_label_value_pairs(text: str) -> str:
    """The particle STAR ``text`` with ``data_general`` as RELION writes it (a list under ``# version``)."""

    head, found, tail = text.partition(SIMULATED_GENERAL)
    if not found:
        raise ValueError("the simulated particle STAR's data_general loop was not found")
    return RELION_GENERAL + tail


def write_relion_form(project, out, *, drop_random_subset: bool = True) -> Path:
    """Write simulated project ``project`` to ``out`` in RELION's forms and return ``out``.

    ``drop_random_subset=False`` keeps the simulated halves, for a comparison with the simulated project
    in which only the STAR forms differ.
    """

    project, out = Path(project), Path(out)
    (out / "tilt_series").mkdir(parents=True)
    for entry in sorted(project.iterdir()):
        if entry.name not in ("tilt_series", "particles.star", "particles_2d.star"):
            os.symlink(entry, out / entry.name)
    for entry in sorted((project / "tilt_series").iterdir()):
        if entry.suffix != ".star":
            os.symlink(entry, out / "tilt_series" / entry.name)
            continue
        text = entry.read_text()
        for label, default in (("_rlnTomoXTilt", "0"), ("_rlnCtfScalefactor", "1")):
            text = drop_loop_column(text, entry.stem, label, only_if_all=default)
        (out / "tilt_series" / entry.name).write_text(text)
    text = general_as_label_value_pairs((project / "particles.star").read_text())
    if drop_random_subset:
        text = drop_loop_column(text, "particles", "_rlnRandomSubset")
    (out / "particles.star").write_text(text)
    return out


def add_particle_column(text: str, label: str, values) -> str:
    """The particle STAR ``text`` with column ``label`` appended to ``data_particles``, one value per row."""

    lines = text.split("\n")
    _, first, end = _loop(lines, "particles")
    values = [str(value) for value in values]
    if len(values) != end - first:
        raise ValueError(f"{len(values)} values for {end - first} particle rows")
    rows = [f"{line} {value}" for line, value in zip(lines[first:end], values, strict=True)]
    return "\n".join(lines[:first] + [label] + rows + lines[end:])
