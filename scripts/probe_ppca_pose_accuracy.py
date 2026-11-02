"""The expected rotation accuracy along the saved checkpoints of a `relax ppca_initial_model` run.

For each chosen iteration k, loads checkpoint k, scores 100 fixed random particles against it at the HEALPix order and
radius of iteration k + 1 (the final embedding pass for the last iteration; one E-step of the run's own engine), turns each best pose along RELION's rotation schedule
and prints and writes the divergence-based accuracy `acc_rot`, the accuracy of the mean image alone (RELION's
quantity), the shares of the divergence (mean, log det, trace) at the crossing, and the order the accuracy rule
would take next and for the final embedding pass (docs/math/vdam_ppca_algorithm.md sections 4.1 and 18.10).

Run (one GPU, minutes per run):
  python scripts/probe_ppca_pose_accuracy.py RUN_DIR --iterations 10 100 600 1200 -o OUT.json [--final-order-max 5]
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from relax.commands.ppca_initial_model import load_training
from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.iteration_loop import expectation_groups
from relax.ppca_initial_model.order_rule import final_order, next_order
from relax.ppca_initial_model.pose_accuracy import N_TRIALS, expected_rotation_accuracy
from relax.ppca_initial_model.radius_rule import radius_for
from relax.sampling import grids


def run_config(run: dict) -> Config:
    """The `Config` a run recorded in its run.json."""
    options = dict(run["config"])
    options["stages"] = tuple(tuple(stage) for stage in options["stages"])
    if options.get("pc_stages") is not None:
        options["pc_stages"] = tuple(tuple(stage) for stage in options["pc_stages"])
    return Config(**options)


def logged_orders(run_dir: Path) -> dict[int, int]:
    """The HEALPix order of every iteration of the run's last trajectory, from iterations.jsonl."""
    orders = {}
    for line in open(run_dir / "iterations.jsonl"):
        row = json.loads(line)
        orders[row["iteration"]] = row["healpix_order"]
    return orders


def probe(run_dir: Path, iterations: list[int], final_order_max: int) -> list[dict]:
    """One row per iteration: the accuracy of checkpoint k at the order and radius of iteration k + 1."""
    run = json.loads((run_dir / "run.json").read_text())
    config = run_config(run)
    dataset, _, _ = load_training(run["input"]["particles"], "auto")
    diameter = run["input"]["particle_diameter_ang"]
    orders = logged_orders(run_dir)
    ids = np.random.default_rng(0).permutation(dataset.n_images)[:N_TRIALS]
    rows = []
    for k in iterations:
        state = checkpoint.load(run_dir / f"checkpoint_{k:04d}.npz", config, run["identity"])
        target = k + 1 if k + 1 in orders else k  # the final model's embedding pass runs at its own iteration
        order, active = orders[target], config.active_pcs(target)
        probe_config = dataclasses.replace(config, order_rule="accuracy", order_max=max(order, config.stages[0][2]))
        state = dataclasses.replace(state, next_order=order)
        radius = radius_for(probe_config, state, target, dataset.grid_size)
        stats = expectation_groups(dataset, state, probe_config, [ids], target, diameter_ang=diameter)[0]
        trial_ids = np.asarray(stats.original_image_ids)
        eulers = grids.get_relion_hidden_rotation_grid(order, matrices=False)[stats.diagnostics["best_rotation_idx"]]
        accuracy = expected_rotation_accuracy(
            dataset,
            state.theta[:, : 1 + active],
            state.noise,
            ids=trial_ids,
            eulers_deg=eulers,
            radius=radius,
            active_pcs=active,
        )
        rows.append(
            {
                "iteration": k,
                "order": order,
                "radius": radius,
                "acc_rot": accuracy.acc_rot,
                "acc_rot_mean_only": accuracy.acc_rot_mean_only,
                "terms_at_error": accuracy.terms,
                "pmax_mean": stats.diagnostics["pmax_mean"],
                "pose_entropy_mean": stats.diagnostics["pose_entropy_mean"],
                "next_order": next_order(order, accuracy.acc_rot, order_max=order + 1),
                "final_order": final_order(order, accuracy.acc_rot, final_order_max=final_order_max),
            }
        )
        print(
            f"it {k:5d} order {order} radius {radius:2d}  acc_rot {accuracy.acc_rot:6.2f}  "
            f"mean-only {accuracy.acc_rot_mean_only:6.2f}  pmax {rows[-1]['pmax_mean']:.3f}  "
            f"next {rows[-1]['next_order']}  final {rows[-1]['final_order']}",
            flush=True,
        )
    return rows


def main() -> None:
    """Probe the run and write the rows as JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--iterations", type=int, nargs="+", required=True, help="checkpoints to probe")
    parser.add_argument("--final-order-max", type=int, default=5)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    rows = probe(args.run_dir, args.iterations, args.final_order_max)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
