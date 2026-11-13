"""Run relax initial_model and record every M-step's gate statistics (diagnostic of the VDAM gate benchmark).

For each gradient M-step, per Fourier shell inside the current sphere: the voxel count, the shell means of the reference
projector's amplitude and power and of the half first moments' difference (amplitude and power, before any scale), the
FSC floor, the fudge, the scale the run applied and the gate it applied. Each iteration's batch (particle ids and
pseudo-halves) is saved too, so that the scale of every gate can be recomputed offline (analyze_gates.py).

The run is the command's own, with two hooks: the M-step's jitted transaction is called with ``return_intermediates``
(its projector, gradient and updated projector are dropped on the device after the statistics are read), and
``VdamUpdate.maximize`` notes the iteration and its batch. The extra outputs change the compiled program, so a traced
run need not be bitwise equal to an untraced one; compare traced runs with traced runs.

Run:
    python scripts/vdam_gate_benchmark/traced_initial_model.py TRACE_DIR -- <relax initial_model arguments>
Writes TRACE_DIR/gate_trace.jsonl (one line per M-step) and TRACE_DIR/batches.npz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_GEOMETRY: dict = {}


def shell_geometry(n: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """For a centered half volume of side ``n``: the flat indices inside the sphere and their shells (padding 1)."""
    key = (n, radius)
    if key not in _GEOMETRY:
        coord = np.arange(n) - n // 2
        r2 = coord[:, None, None] ** 2 + coord[None, :, None] ** 2 + np.arange(n // 2 + 1)[None, None, :] ** 2
        inside = (r2 < radius**2).reshape(-1)
        shells = np.floor(np.sqrt(r2.reshape(-1)[inside]) + 0.5).astype(np.int64)
        _GEOMETRY[key] = (np.flatnonzero(inside), shells)
    return _GEOMETRY[key]


def shell_means(values: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Shell means of ``values`` (a centered half volume) over shells 0..radius, and the voxel counts."""
    index, shells = shell_geometry(values.shape[0], radius)
    picked = values.reshape(-1)[index]
    counts = np.bincount(shells, minlength=radius + 1)[: radius + 1]
    sums = np.bincount(shells, weights=picked, minlength=radius + 1)[: radius + 1]
    return sums / np.maximum(counts, 1), counts


def install(trace_dir: Path) -> list:
    """Install the two hooks; returns the list the batches are appended to."""
    from relax.vdam import iteration_loop, relion_mstep

    trace_dir.mkdir(parents=True, exist_ok=True)
    handle = open(trace_dir / "gate_trace.jsonl", "w")
    batches: list = []
    context = {"iteration": None}
    device = relion_mstep.relion_vdam_m_step_device
    maximize = iteration_loop.VdamUpdate.maximize

    def traced_maximize(self, current, accumulators, sums, meta):
        context["iteration"] = int(current.iter)
        if current.pseudo_halfsets:
            batches.append(
                (int(current.iter), np.asarray(current.subset_particle_ids, np.int64), np.asarray(current.subset_halfset_ids, np.int8))
            )
        return maximize(self, current, accumulators, sums, meta)

    def traced_device(*args, **kwargs):
        result = device(*args, **{**kwargs, "return_intermediates": True})
        projector = result.pop("projector")
        result.pop("gradient")
        result.pop("updated_projector")
        gate = result.pop("fsc_estimate")
        if not kwargs.get("pseudo_halfsets", True) or kwargs.get("padding_factor", 1) != 1:
            return result
        box_size = kwargs["box_size"]
        radius = int(np.asarray(args[12]))
        radius = radius if radius > 0 else box_size // 2
        difference = np.asarray(result["mom1_h0"]) - np.asarray(result["mom1_h1"])
        projector = np.asarray(projector)
        ref_amp, counts = shell_means(np.abs(projector).astype(np.float64), radius)
        ref_pow, _ = shell_means(np.abs(projector).astype(np.float64) ** 2, radius)
        dis_amp, _ = shell_means(np.abs(difference).astype(np.float64), radius)
        dis_pow, _ = shell_means(np.abs(difference).astype(np.float64) ** 2, radius)
        scale = args[16] if len(args) > 16 else kwargs.get("disagreement_scale")
        record = {
            "iteration": context["iteration"],
            "radius": radius,
            "scale": None if scale is None else float(np.asarray(scale)),
            "fudge": float(np.asarray(args[11])),
            "counts": counts.tolist(),
            "ref_amp": ref_amp.tolist(),
            "ref_pow": ref_pow.tolist(),
            "dis_amp": dis_amp.tolist(),
            "dis_pow": dis_pow.tolist(),
            "floor": np.asarray(args[8], np.float64)[: radius + 1].tolist(),
            "gate": np.asarray(gate, np.float64)[: radius + 1].tolist(),
        }
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        return result

    relion_mstep.relion_vdam_m_step_device = traced_device
    iteration_loop.VdamUpdate.maximize = traced_maximize
    return batches


def save_batches(path: Path, batches: list) -> None:
    """The batches as one npz: iterations, and the concatenated ids and halves with their offsets."""
    iterations = np.asarray([b[0] for b in batches], np.int64)
    sizes = np.asarray([b[1].size for b in batches], np.int64)
    ids = np.concatenate([b[1] for b in batches]) if batches else np.zeros(0, np.int64)
    halves = np.concatenate([b[2] for b in batches]) if batches else np.zeros(0, np.int8)
    np.savez_compressed(path, iterations=iterations, sizes=sizes, ids=ids, halves=halves)


def main() -> int:
    """Parse ``TRACE_DIR -- ARGS``, run the command with the hooks, save the batches."""
    if len(sys.argv) < 3 or sys.argv[2] != "--":
        raise SystemExit(__doc__)
    trace_dir = Path(sys.argv[1])
    batches = install(trace_dir)
    from relax.commands.initial_model import main as initial_model_main

    try:
        return initial_model_main(sys.argv[3:])
    finally:
        save_batches(trace_dir / "batches.npz", batches)


if __name__ == "__main__":
    sys.exit(main())
