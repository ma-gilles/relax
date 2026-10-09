"""InitialModel artifact paths, startup metadata and output cadence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from recovar.utils.helpers import recovar_volume_to_relion

from relax.helpers.map_io import write_map
from relax.relion.initial_model_io import _write_data_star, _write_model_star
from relax.vdam.align_symmetry import align_symmetry, select_largest_class
from relax.vdam.state import InitialModelState, NativeParticleState

if TYPE_CHECKING:
    from relax.vdam.ports import NoStageProfile


def _initial_model_mrc_from_prefix(outputname: str) -> str:
    """RELION's job directory plus ``initial_model.mrc``.

    RELION's pipeliner runs ``relion_refine --o <outputname>run`` and writes
    ``<outputname>initial_model.mrc`` (``pipeline_jobs.cpp:3466, 3577``), so the
    ``run`` suffix is removed once, not every trailing r/u/n character.
    """

    return outputname.removesuffix("run") + "initial_model.mrc"


def _class_mrc_paths(output_prefix: str, iteration: int, K: int) -> tuple[str, ...]:
    return tuple(f"{output_prefix}_it{iteration:03d}_class{k + 1:03d}.mrc" for k in range(K))


def write_initial_run_metadata(opts, continuation) -> None:
    """Write startup options and the optional native continuation provenance."""

    Path(opts.outputname).parent.mkdir(parents=True, exist_ok=True)
    config_path = f"{opts.outputname}_native_options.json"
    native_options = asdict(opts)
    del native_options["environment"]  # the saved options keep their format
    native_options["resolved_cuda_allocator"] = os.environ.get(
        "TF_GPU_ALLOCATOR",
        "default",
    )
    native_options["jax_compilation_cache_enabled"] = bool(
        os.environ.get("JAX_COMPILATION_CACHE_DIR")
    )
    native_options["jax_compilation_cache_dir"] = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR"
    )
    native_options["jax_persistent_cache_min_compile_time_secs"] = os.environ.get(
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
    )
    with open(config_path, "w") as f:
        json.dump(native_options, f, indent=2, sort_keys=True)
    if continuation is not None:
        continuation_path = f"{opts.outputname}_diagnostic_continuation.json"
        with open(continuation_path, "w") as f:
            json.dump(
                {
                    "classification": "diagnostic_performance_only",
                    "exactly_one_next_iteration": True,
                    "iteration": int(continuation.iteration),
                    "optimiser_star": str(continuation.optimiser_star),
                    "model_star": str(continuation.model_star),
                    "data_star": str(continuation.data_star),
                    "sampling_star": str(continuation.sampling_star),
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")


def write_iteration_artifacts(
    output_prefix: str,
    state: InitialModelState,
    iteration: int,
    meta: dict,
    *,
    main_star,
    optics_star,
    dataset,
    particle_state: NativeParticleState,
    profile: NoStageProfile,
) -> None:

    out_dir = Path(output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    class_mrcs = _class_mrc_paths(output_prefix, iteration, int(state.K))
    profile.record("setup")
    for k, class_mrc in enumerate(class_mrcs):
        write_map(class_mrc, state.Iref[k], voxel_size=float(state.pixel_size))
    profile.record("class_mrc")
    model_star = f"{output_prefix}_it{iteration:03d}_model.star"
    _write_model_star(model_star, state, class_mrcs)
    profile.record("model_star")
    meta_path = f"{output_prefix}_it{iteration:03d}_recovar_meta.json"
    with open(meta_path, "w") as f:
        json.dump(_json_ready(meta), f, indent=2, sort_keys=True)
    profile.record("meta_json")
    _write_data_star(
        f"{output_prefix}_it{iteration:03d}_data.star",
        main_star,
        optics_star,
        dataset,
        particle_state,
    )
    profile.record("data_star")
    profile.report(f"iteration {iteration} artifact")


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def write_final_outputs(
    output_prefix: str,
    state: InitialModelState,
    *,
    sym_name: str,
    seed: int,
    written_iterations: frozenset[int] = frozenset(),
) -> tuple[str, tuple[str, ...], dict]:
    """Write the last iteration's class maps, model.star and ``initial_model.mrc``.

    RELION's write() always writes the last iteration's model (ml_optimiser.cpp:1361, 3489). The class maps
    and model.star of ``state.iter`` are written here unless this run's iteration sink already wrote them
    (``written_iterations``); a file an earlier run left under the same prefix is never taken as the result.

    ``initial_model.mrc`` is what RELION's InitialModel GUI job writes after relion_refine:
    ``relion_align_symmetry --select_largest_class --apply_sym --sym S`` (:mod:`relax.vdam.align_symmetry`), so for
    a non-C1 ``sym_name`` the largest class is aligned to the symmetry axes and symmetrised.
    """
    iteration = int(state.iter)
    class_mrcs = _class_mrc_paths(output_prefix, iteration, int(state.K))
    model_star = f"{output_prefix}_it{iteration:03d}_model.star"
    out_dir = Path(output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if iteration not in written_iterations:
        for k, class_mrc in enumerate(class_mrcs):
            write_map(class_mrc, state.Iref[k], voxel_size=float(state.pixel_size))
        _write_model_star(model_star, state, class_mrcs)
    final_mrc = _initial_model_mrc_from_prefix(output_prefix)
    best_class = select_largest_class(state.pdf_class)
    # The alignment reads the map as written (RELION's frame); the frame change is its own inverse.
    relion_map = recovar_volume_to_relion(np.asarray(state.Iref[best_class]).real)
    aligned, report = align_symmetry(relion_map, sym_name, seed=seed)
    write_map(final_mrc, recovar_volume_to_relion(aligned).astype(np.float32), voxel_size=float(state.pixel_size))
    report["class"] = best_class + 1
    return final_mrc, class_mrcs, report
