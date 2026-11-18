"""Run relax initial_model and record every M-step's gate statistics (diagnostic of the VDAM gate benchmark).

For each gradient M-step, per Fourier shell inside the current sphere: the voxel count, the shell means of the reference
projector's amplitude and power and of the half first moments' difference (amplitude and power, before any scale), the
FSC floor, the fudge, the step, the scale the run applied and the gate it applied. Each iteration's batch (particle ids
and pseudo-halves) is saved too, so that the scale of every gate can be recomputed offline (analyze_gates.py). After each
iteration's resolution update: its average Pmax, current size, batch size and data_vs_prior, and the next resolution.

The run is the command's own, with three hooks: the M-step's jitted transaction is called with ``return_intermediates``
(its projector, gradient and updated projector are dropped on the device after the statistics are read),
``VdamUpdate.maximize`` notes the iteration and its batch, and ``VdamUpdate.update_resolution`` writes the iteration line.
The extra outputs change the compiled program, so a traced run need not be bitwise equal to an untraced one; compare
traced runs with traced runs.

Study options change the next iteration's Fourier radius (RELION: the last shell where data_vs_prior >= 1, plus
incr_size = 10 shells). ``--incr-size N`` sets incr_size from the end of iteration 1. ``--radius-rule moment_window``
reads the crossing from data_vs_prior times the half-0 first moment's effective particles over the half-0 batch,
(1 / n_0) / sum_{i in half 0} a_i^2 with a the per-particle moment coefficients (relax.vdam.gate.advance_noise_coefficients):
the shells the update direction resolves rather than one half-batch. ``--radius-rule signal`` also multiplies by the
reference's signal fraction per shell, 1 - noise / power (clipped to [0, 1]), with the reference's noise power the M-step's
half-moment disagreement power times window_disagreement_scale / 4 (open gate: an upper bound), so that noise the
reference holds does not count as signal. The state keeps the run's own data_vs_prior.

``--switch-gate ITER:GATE`` applies the run's --vdam-gate before iteration ITER and GATE from ITER on. Both gates advance
their state (noise coefficients) every iteration, so the later gate's scale is the one it would have had from the start.

``--relion-floor``: in an M-step that applies a scale, the gate of each shell is at least RELION's gate of the same
M-step (computed by a first call of the transaction without the scale and passed as its FSC floor). A power gate keeps a
shell that enters the radius with near-zero power closed (v = 0 is stable); RELION's amplitude gate opens it when its
data carry signal. The trace's ``floor`` is then that RELION gate.

Run:
    python scripts/vdam_gate_benchmark/traced_initial_model.py TRACE_DIR [--radius-rule {moment_window,signal}] \
        [--incr-size N] [--switch-gate ITER:GATE] [--relion-floor] -- <relax initial_model arguments>
Writes TRACE_DIR/gate_trace.jsonl (one line per M-step), TRACE_DIR/iterations.jsonl (one line per iteration) and
TRACE_DIR/batches.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

_GEOMETRY: dict = {}


@dataclass(frozen=True)
class SwitchedGate:
    """The gate ``early`` before iteration ``switch`` and ``late`` from it on; both advance their state every iteration."""

    early: object
    late: object
    switch: int

    @property
    def name(self) -> str:
        """``early>late@switch``."""
        return f"{self.early.name}>{self.late.name}@{self.switch}"

    def advance(self, state):
        """The state advanced by both gates, and the scale of the gate that applies at ``state.iter``."""
        iteration = int(state.iter)
        state, early_scale = self.early.advance(state)
        state, late_scale = self.late.advance(state)
        return state, (early_scale if iteration < self.switch else late_scale)


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


def grown(values: np.ndarray | None, size: int, fill) -> np.ndarray | None:
    """``values`` padded with ``fill`` to at least ``size`` entries (None stays None)."""
    if values is None or values.size >= size:
        return values
    return np.concatenate([values, np.full(size - values.size, fill, values.dtype)])


def install(
    trace_dir: Path,
    radius_rule: str = "relion",
    incr_size: int | None = None,
    switch: tuple[int, str] | None = None,
    relion_floor: bool = False,
) -> list:
    """Install the hooks (study radius options, gate switch, RELION floor); returns the list the batches are appended to."""
    from relax.vdam import driver, iteration_loop, relion_mstep
    from relax.vdam.gate import advance_noise_coefficients, advance_window_coefficients, window_disagreement_scale

    if switch is not None:
        make_gate = driver.vdam_gate
        driver.vdam_gate = lambda name, n: SwitchedGate(make_gate(name, n), make_gate(switch[1], n), switch[0])
    trace_dir.mkdir(parents=True, exist_ok=True)
    handle = open(trace_dir / "gate_trace.jsonl", "w")
    iterations = open(trace_dir / "iterations.jsonl", "w")
    batches: list = []
    context = {"iteration": None, "record": None}
    window = {"coefficients": None, "halves": np.zeros(0, np.int8), "moment": None, "reference": None}
    device = relion_mstep.relion_vdam_m_step_device
    maximize = iteration_loop.VdamUpdate.maximize
    update_resolution = iteration_loop.VdamUpdate.update_resolution

    def traced_maximize(self, current, accumulators, sums, meta):
        context["iteration"] = int(current.iter)
        if current.pseudo_halfsets:
            batches.append(
                (int(current.iter), np.asarray(current.subset_particle_ids, np.int64), np.asarray(current.subset_halfset_ids, np.int8))
            )
        return maximize(self, current, accumulators, sums, meta)

    def moment_window_factor(current) -> float:
        """Advance the per-particle moment coefficients by this iteration's batch; the half-0 factor."""
        ids = np.asarray(current.subset_particle_ids, np.int64)
        halves = np.asarray(current.subset_halfset_ids, np.int8)
        size = max(int(ids.max()) + 1, window["halves"].size)
        window["halves"] = grown(window["halves"], size, -1)
        window["halves"][ids] = halves
        window["coefficients"] = advance_noise_coefficients(grown(window["coefficients"], size, 0.0), ids, halves, size)
        members = window["halves"] == 0
        return (1.0 / int(np.sum(halves == 0))) / float(np.sum(window["coefficients"][members] ** 2))

    def signal_fraction(current, shells: int) -> np.ndarray:
        """Advance the window coefficients by this iteration's batch and step; the reference's signal fraction per shell."""
        record = context["record"]
        if record is None or record["iteration"] != int(current.iter):
            raise RuntimeError(f"no M-step statistics traced for iteration {int(current.iter)}")
        ids = np.asarray(current.subset_particle_ids, np.int64)
        halves = np.asarray(current.subset_halfset_ids, np.int8)
        size = window["coefficients"].size
        window["moment"], window["reference"] = advance_window_coefficients(
            grown(window["moment"], size, 0.0), grown(window["reference"], size, 0.0), ids, halves, size, record["stepsize"]
        )
        scale = window_disagreement_scale(window["reference"], window["coefficients"])
        power, disagreement = np.asarray(record["ref_pow"]), np.asarray(record["dis_pow"])
        fraction = np.zeros(shells)
        m = min(shells, power.size)
        fraction[:m] = np.clip(1.0 - disagreement[:m] * scale / 4.0 / np.where(power[:m] > 0, power[:m], np.inf), 0.0, 1.0)
        return fraction

    def traced_update_resolution(self, current):
        factor = 1.0
        if radius_rule in ("moment_window", "signal") and current.pseudo_halfsets:
            factor = moment_window_factor(current)
            curve = np.asarray(current.data_vs_prior_class, np.float64) * factor
            if radius_rule == "signal":
                curve = curve * signal_fraction(current, curve.shape[-1])[None, :]
            scaled = replace(current, data_vs_prior_class=curve)
            updated = replace(update_resolution(self, scaled), data_vs_prior_class=current.data_vs_prior_class)
        else:
            updated = update_resolution(self, current)
        if incr_size is not None:
            updated = replace(updated, incr_size=int(incr_size))
        line = {
            "iteration": int(current.iter),
            "ave_pmax": float(current.ave_Pmax),
            "current_size": int(current.current_size),
            "subset_size": int(current.subset_size),
            "stepsize": float(current.grad_current_stepsize),
            "fudge": float(current.tau2_fudge_factor),
            "next_resolution_shell": int(updated.current_resolution_shell),
            "radius_factor": factor,
            "incr_size": int(updated.incr_size),
            "data_vs_prior": np.asarray(current.data_vs_prior_class[0], np.float64).tolist(),
        }
        iterations.write(json.dumps(line) + "\n")
        iterations.flush()
        return updated

    def traced_device(*args, **kwargs):
        if relion_floor and len(args) > 16 and args[16] is not None and kwargs.get("pseudo_halfsets", True):
            relion_gate = device(*args[:16], None, *args[17:], **{**kwargs, "return_intermediates": True})["fsc_estimate"]
            args = (*args[:8], relion_gate, *args[9:])
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
            "stepsize": float(np.asarray(args[10])),
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
        context["record"] = record
        return result

    relion_mstep.relion_vdam_m_step_device = traced_device
    iteration_loop.VdamUpdate.maximize = traced_maximize
    iteration_loop.VdamUpdate.update_resolution = traced_update_resolution
    return batches


def save_batches(path: Path, batches: list) -> None:
    """The batches as one npz: iterations, and the concatenated ids and halves with their offsets."""
    iterations = np.asarray([b[0] for b in batches], np.int64)
    sizes = np.asarray([b[1].size for b in batches], np.int64)
    ids = np.concatenate([b[1] for b in batches]) if batches else np.zeros(0, np.int64)
    halves = np.concatenate([b[2] for b in batches]) if batches else np.zeros(0, np.int8)
    np.savez_compressed(path, iterations=iterations, sizes=sizes, ids=ids, halves=halves)


def main() -> int:
    """Parse ``TRACE_DIR [study options] -- ARGS``, run the command with the hooks, save the batches."""
    if "--" not in sys.argv[2:]:
        raise SystemExit(__doc__)
    split = sys.argv.index("--", 2)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--radius-rule", choices=("relion", "moment_window", "signal"), default="relion")
    parser.add_argument("--incr-size", type=int)
    parser.add_argument("--switch-gate", metavar="ITER:GATE", help="apply GATE from iteration ITER on")
    parser.add_argument("--relion-floor", action="store_true", help="a scaled gate is at least RELION's gate")
    options = parser.parse_args(sys.argv[1:split])
    switch = None
    if options.switch_gate:
        iteration, _, gate = options.switch_gate.partition(":")
        switch = (int(iteration), gate)
    batches = install(options.trace_dir, options.radius_rule, options.incr_size, switch, options.relion_floor)
    trace_dir = options.trace_dir
    from relax.commands.initial_model import main as initial_model_main

    try:
        return initial_model_main(sys.argv[split + 1 :])
    finally:
        save_batches(trace_dir / "batches.npz", batches)


if __name__ == "__main__":
    sys.exit(main())
