"""Replay a saved K=1 boundary; measure cross-path and same-path GPU differences.

This diagnostic changes no tolerances and makes no scientific acceptance claim.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils as ftu
from recovar.ppca.triangular import unpack_tri_to_full

from relax.commands.ppca_initial_model import load_training
from relax.ppca_initial_class3d import checkpoint as mixture_checkpoint
from relax.ppca_initial_class3d.config import Config as MixtureConfig
from relax.ppca_initial_class3d.expectation import expectation_groups, pose_grid
from relax.ppca_initial_class3d.stream import _mixture_normalization
from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model import iteration_loop as single
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.update import coupled_direction, metric_floor, stochastic_update
from relax.ppca_refinement.full_row_stream import _read_tile, _score_tile


def difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    finite = np.isfinite(a) & np.isfinite(b)
    with np.errstate(invalid="ignore"):
        delta = np.where(finite, np.abs(a - b), 0)
    index = np.unravel_index(np.argmax(delta), delta.shape)
    scale = float(np.max(np.where(np.isfinite(a), np.abs(a), 0)))
    return {"max_abs": float(delta[index]), "scale": scale,
            "over_default_band": int(np.sum(delta > 1e-6 * scale)),
            "max_index": list(map(int, index)),
            "nonfinite_mismatches": int(np.sum(~finite & (a != b)))}


def compare(a, b):
    return {key: difference(a[key], b[key]) for key in a}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    assert jax.default_backend() == "gpu", "This diagnostic requires a real GPU"
    args.output.mkdir(parents=True, exist_ok=False)
    data, _, _ = load_training(args.checkpoint_root / "inputs/training.json", "on")
    with np.load(args.checkpoint_root / "single/checkpoint_0000.npz") as archive:
        meta = json.loads(str(archive["metadata"]))
    config = Config(**meta["config"])
    state = checkpoint.load(args.checkpoint_root / "single/checkpoint_0000.npz", config, meta["identity"])
    mixture_config = MixtureConfig(**dataclasses.asdict(config), n_classes=1)
    mixture_state = mixture_checkpoint.load(args.checkpoint_root / "mixture/checkpoint_0000.npz",
                                           mixture_config, meta["identity"])
    rng = np.random.default_rng()
    rng.bit_generator.state = state.rng_state
    _, groups = single._select_halves(rng, state.order, data.n_images, False)
    grid = pose_grid(data, mixture_config, 1)
    radii = np.asarray(ftu.get_grid_of_radial_distances_real(data.volume_shape, rounded=False)).reshape(-1)
    shells = np.asarray(ftu.get_grid_of_radial_distances_real(data.volume_shape), np.int32).reshape(-1)
    support = radii <= config.stage(1)[0]
    _, step, fudge = config.schedule(1, data.n_images)
    captured = []
    original_prepare = single.prepare_full_row_stream

    def capture(*a, **kw):
        stream = original_prepare(*a, **kw)
        captured.append(stream)
        return stream

    records = {"single": [], "mixture": []}
    for repeat in range(args.repeats):
        for mode in (["single", "mixture"] if repeat % 2 == 0 else ["mixture", "single"]):
            if mode == "single":
                with patch.object(single, "prepare_full_row_stream", capture):
                    stats = single.expectation_groups(data, state, config, groups, 1, diameter_ang=6.)
            else:
                stats = [s.components[0] for s in expectation_groups(data, mixture_state, mixture_config, groups, 1, grid)]
            arrays, directions, coverage = {}, [], []
            for half, stats_half in enumerate(stats):
                direction, _ = coupled_direction(stats_half.lhs_tri, stats_half.residual_gradient, support,
                                                  floor=metric_floor(data.grid_size))
                directions.append(direction)
                coverage.append((jnp.trace(unpack_tri_to_full(stats_half.lhs_tri, config.q + 1), axis1=-2, axis2=-1) > 0)
                                & jnp.asarray(support))
                arrays.update({f"h{half}_lhs": np.asarray(stats_half.lhs_tri),
                               f"h{half}_gradient": np.asarray(stats_half.residual_gradient),
                               f"h{half}_direction": np.asarray(direction)})
                replay, _ = coupled_direction(stats_half.lhs_tri, stats_half.residual_gradient, support,
                                               floor=metric_floor(data.grid_size))
                print(mode, repeat, half, "same-stat eigensolve", difference(direction, replay), flush=True)
            theta, moments, _ = stochastic_update(state.theta, state.moments, jnp.stack(directions),
                                                  jnp.stack(coverage), shells, step=step, fudge=fudge,
                                                  image_size=data.grid_size)
            arrays.update(theta=np.asarray(theta), first=np.asarray(moments.first), second=np.asarray(moments.second))
            np.savez(args.output / f"{mode}_{repeat}.npz", **arrays)
            records[mode].append(arrays)
    stream = captured[0]
    assert stream.static.cuda_kernels, "Native kernels must be active"
    tile, _, layout = _read_tile(stream, groups[0], [None] * len(groups[0]), collect_observation=True)
    kept, posterior = _score_tile(stream, tile, layout["n_blocks"], None)
    first_scores = np.asarray(kept.score).copy()
    kept, repeated = _score_tile(stream, tile, layout["n_blocks"], kept)
    normalized = _mixture_normalization(posterior.center[:, None], posterior.centered_logZ[:, None], jnp.zeros(1))
    report = {
        "device": str(jax.devices()[0]), "groups": [g.tolist() for g in groups],
        "repeated_pass1_scores": difference(first_scores, kept.score),
        "repeated_partition": difference(posterior.centered_logZ, repeated.centered_logZ),
        "singleton_partition": difference(posterior.centered_logZ, normalized[3][:, 0]),
        "cross_path": [compare(records["single"][i], records["mixture"][i]) for i in range(args.repeats)],
        "single_repeat": [compare(records["single"][0], x) for x in records["single"][1:]],
        "mixture_repeat": [compare(records["mixture"][0], x) for x in records["mixture"][1:]],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
