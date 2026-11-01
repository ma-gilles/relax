"""A RELION refinement checkpoint (``run_itNNN_optimiser.star`` and the files it names), read as relion_refine
``--continue`` reads it.

A multi-body refinement continues a consensus auto-refine (``relion_refine --continue <consensus optimiser>
--multibody_masks ...``): every per-half model, the particle table and the half maps come from the consensus
iteration. A multi-body checkpoint adds one ``images_body_<b>`` table per body to the data STAR and one map per body
and half. Values stay as RELION wrote them; :mod:`relax.multibody.refine` converts units. The model and optimiser STAR
files of a consensus from before RELION 4 are read as RELION 5.0.1 reads them, into the current labels
(:func:`_half_model`, :func:`read_relion_checkpoint`); their data STAR goes through ``relion_convert_star`` first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import starfile

from relax.star.blocks import read_star_blocks


@dataclass(frozen=True)
class HalfModel:
    """One half's ``run_itNNN_half<h>_model.star``: general values, per-class (or body) shell curves, groups
    and optics-group noise, all as floats."""

    general: dict
    class_shells: list  # per class or body: {label: np.ndarray}, the data_model_class_<k> / body_<k> loop
    classes: dict  # data_model_classes / data_model_bodies loop: {label: np.ndarray}
    group_scale: np.ndarray  # [n_groups] rlnGroupScaleCorrection
    group_particles: np.ndarray  # [n_groups] rlnGroupNrParticles
    sigma2_noise: list  # per optics group: [n_shells] rlnSigma2Noise


@dataclass(frozen=True)
class RelionCheckpoint:
    """The state a ``relion_refine --continue`` starts from."""

    optimiser_star: Path
    optimiser: dict  # data_optimiser_general, strings
    sampling: dict  # data_sampling_general, strings
    models: list  # [half 1, half 2] HalfModel
    particles: object  # pandas DataFrame of data_particles
    optics: object  # pandas DataFrame of data_optics
    body_tables: list  # per body: DataFrame of images_body_<b>; empty for a consensus checkpoint
    half_map_paths: list  # [half][class or body] Path
    model_paths: list  # [half] Path of the half's model STAR

    @property
    def iteration(self) -> int:
        return int(self.optimiser["rlnCurrentIteration"])


def _floats(column) -> np.ndarray:
    return np.asarray([float(v) for v in column], dtype=np.float64)


def _half_model(path: Path, n_optics_in_data: int) -> HalfModel:
    blocks = read_star_blocks(path)
    general = {key: value for key, value in blocks["model_general"].items()}
    # A model STAR from before RELION 4 has no optics groups: MlModel::read takes their number from the data STAR and
    # copies the first group's noise spectrum into each (ml_model.cpp:153-158, :396-400).
    is_pre4 = "rlnNrOpticsGroups" not in general
    if is_pre4:
        general["rlnNrOpticsGroups"] = str(n_optics_in_data)
    nr_bodies = int(general.get("rlnNrBodies", 1))
    if nr_bodies > 1:
        shells = [blocks[f"model_body_{b + 1}"] for b in range(nr_bodies)]
        classes = blocks["model_bodies"]
    else:
        shells = [blocks[f"model_class_{k + 1}"] for k in range(int(general["rlnNrClasses"]))]
        classes = blocks["model_classes"]
    if "rlnAccuracyTranslationsAngst" not in classes:  # pixels until RELION 3.1 (ml_model.cpp:223-228)
        classes = dict(classes)
        classes["rlnAccuracyTranslationsAngst"] = [
            str(float(value) * float(general["rlnPixelSize"])) for value in classes["rlnAccuracyTranslations"]
        ]
    groups = blocks["model_groups"]
    n_optics = int(general["rlnNrOpticsGroups"])
    return HalfModel(
        general=general,
        class_shells=[
            {label: (_floats(values) if label != "rlnReferenceImage" else values) for label, values in block.items()}
            for block in shells
        ],
        classes={
            label: (_floats(values) if label != "rlnReferenceImage" else values) for label, values in classes.items()
        },
        group_scale=_floats(groups["rlnGroupScaleCorrection"]),
        group_particles=_floats(groups["rlnGroupNrParticles"]),
        sigma2_noise=[
            _floats(blocks["model_group_1" if is_pre4 else f"model_optics_group_{g + 1}"]["rlnSigma2Noise"])
            for g in range(n_optics)
        ],
    )


def _resolve(path_text: str, optimiser_star: Path, relion_cwd: Path) -> Path:
    """RELION writes paths relative to its working directory: resolve them there."""

    path = Path(path_text)
    if path.is_absolute():
        return path
    return relion_cwd / path


def read_relion_checkpoint(optimiser_star, *, relion_cwd=None) -> RelionCheckpoint:
    """Read ``optimiser_star`` and the model, data, sampling and map files it names.

    ``relion_cwd`` is the directory relion_refine ran in (the names in the files are relative to it); by default
    the parent of the optimiser's directory, the usual ``<job>/run_itNNN_optimiser.star`` layout.
    """

    optimiser_star = Path(optimiser_star).resolve()
    relion_cwd = optimiser_star.parent.parent if relion_cwd is None else Path(relion_cwd)
    optimiser = read_star_blocks(optimiser_star)["optimiser_general"]
    if int(optimiser["rlnDoSplitRandomHalves"]) != 1:
        raise NotImplementedError("a multi-body refinement continues a split-half (auto-refine) run")
    if "rlnOverallAccuracyTranslationsAngst" not in optimiser:
        # MlOptimiser::read takes the older pixel value as Angstroms, unconverted (ml_optimiser.cpp:1497-1501).
        optimiser = {**optimiser, "rlnOverallAccuracyTranslationsAngst": optimiser["rlnOverallAccuracyTranslations"]}
    model_paths = [
        _resolve(optimiser["rlnModelStarFile"], optimiser_star, relion_cwd),
        _resolve(optimiser["rlnModelStarFile2"], optimiser_star, relion_cwd),
    ]
    data = starfile.read(_resolve(optimiser["rlnExperimentalDataStarFile"], optimiser_star, relion_cwd), always_dict=True)
    models = [_half_model(path, len(data["optics"])) for path in model_paths]
    sampling = read_star_blocks(_resolve(optimiser["rlnOrientSamplingStarFile"], optimiser_star, relion_cwd))[
        "sampling_general"
    ]
    nr_bodies = int(models[0].general.get("rlnNrBodies", 1))
    body_tables = [data[f"images_body_{b + 1}"] for b in range(nr_bodies)] if nr_bodies > 1 else []
    half_map_paths = [
        [_resolve(name, optimiser_star, relion_cwd) for name in model.classes["rlnReferenceImage"]] for model in models
    ]
    return RelionCheckpoint(
        optimiser_star=optimiser_star,
        optimiser=optimiser,
        sampling=sampling,
        models=models,
        particles=data["particles"],
        optics=data["optics"],
        body_tables=body_tables,
        half_map_paths=half_map_paths,
        model_paths=model_paths,
    )
