"""Controlled checkpoint continuation probe on the training-only tiny fixture.

The historical smoke assertion remains in validate_vdam_ppca_smoke.py. This
probe starts all continuations from one saved state and records the first
different stage, including repeated execution without serialization.
"""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from relax.commands.ppca_initial_model import load_training, source_identity
from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model import iteration_loop as loop
from relax.ppca_initial_model.config import Config
from relax.ppca_refinement import dense_dataset as dense
from relax.ppca_refinement import residual_statistics as residual


def arrays(value, prefix=""):
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from arrays(getattr(value, field.name), f"{prefix}.{field.name}".strip("."))
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from arrays(value[key], f"{prefix}.{key}".strip("."))
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            yield from arrays(item, f"{prefix}[{i}]")
    elif value is None:
        yield prefix, None
    elif isinstance(value, (np.ndarray, jax.Array)):
        yield prefix, np.asarray(jax.block_until_ready(value))
    else:
        yield prefix, value


def compare(left, right):
    result = {}
    la, ra = dict(arrays(left)), dict(arrays(right))
    for key in sorted(la.keys() | ra.keys()):
        a, b = la.get(key), ra.get(key)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            equal = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
            if equal:
                continue
            record = {"shape": list(a.shape), "dtype": str(a.dtype), "equal": False}
            if a.shape == b.shape and np.issubdtype(a.dtype, np.number):
                delta = np.abs(a - b)
                record["max_abs"] = float(np.max(delta)) if delta.size else 0.0
                if delta.size:
                    numerator = float(np.linalg.norm(np.asarray(delta, dtype=np.float64).ravel()))
                    denominator = float(np.linalg.norm(np.asarray(np.abs(a), dtype=np.float64).ravel()))
                    record["relative_l2"] = numerator / denominator if denominator else 0.0
                record["changed"] = int(np.count_nonzero(a != b))
                if delta.size:
                    idx = np.unravel_index(np.argmax(delta), delta.shape)
                    record["max_index"] = list(idx)
                    record["left_at_max"] = str(a[idx])
                    record["right_at_max"] = str(b[idx])
            result[key] = record
        elif a != b:
            result[key] = {"left": str(a), "right": str(b)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    data, manifest, identity = load_training(args.manifest)
    identity["source"] = source_identity()
    config = Config(
        iterations=2, stages=((1, 2, 0), (2, 3, 0)), oversampling=1,
        shift_range=0, shift_step=2, image_batch_size=4, rotation_block_size=128,
    )
    state = loop.run(data, config, root / "initial", identity, manifest["particle_diameter_ang"], stop_after=1)
    path = root / "initial/checkpoint_0001.npz"
    restored = checkpoint.load(path, config, identity)
    report = {
        "roundtrip": compare(state, restored),
        "config": dataclasses.asdict(config),
        "identity": identity,
        "radius_transition": [state.radius, config.stage(2)[0]],
    }
    rng = np.random.default_rng()
    rng.bit_generator.state = state.rng_state
    selected = rng.permutation(state.order)[:config.schedule(2, data.n_images)[0]]
    report["selected_ids"] = selected.tolist()
    rows = []
    original_coarse = loop.compute_dense_ppca_adaptive_significance
    original_fine = loop.accumulate_dense_ppca_statistics
    original_score = dense.dense_pose_ppca_score_with_moments_blocked
    original_backproject = dense.accumulate_pose_ppca_block_cached
    original_residual = residual.residual_image_statistics

    def capture_op(name, function):
        def wrapped(*a, **kw):
            out = function(*a, **kw)
            slot = rows[-1].setdefault("operations", {}).setdefault(name, [])
            if len(slot) < 2:
                slot.append(out)
            return out
        return wrapped

    def capture_coarse(*a, **kw):
        out = original_coarse(*a, **kw)
        rows[-1]["coarse"] = {
            "candidates": [None if x is None else x.tolist() for x in out.significant_sample_indices],
            "hard_assignment": out.hard_assignment.tolist(),
            "logZ": out.logZ.tolist(),
            "pmax": out.max_posterior_per_image.tolist(),
            "top_scores": out.diagnostics["top_log_score"].tolist(),
        }
        return out

    def capture_fine(*a, **kw):
        out = original_fine(*a, **kw)
        rows[-1].setdefault("fine", []).append({
            "particle_ids": np.asarray(kw["image_indices"]).tolist(),
            "rotation_ids": np.asarray(kw["rotations"]).shape[0],
            "rotation_order_sha256": hashlib.sha256(np.asarray(kw["rotations"]).tobytes()).hexdigest(),
            "rotation_prior_sha256": hashlib.sha256(np.asarray(kw["rotation_log_prior"]).tobytes()).hexdigest(),
            "translation_grid_sha256": hashlib.sha256(np.asarray(kw["translations"]).tobytes()).hexdigest(),
            "mask": None if kw.get("rotation_translation_mask") is None else np.asarray(kw["rotation_translation_mask"]).tolist(),
            "stats": out,
        })
        return out

    loop.compute_dense_ppca_adaptive_significance = capture_coarse
    loop.accumulate_dense_ppca_statistics = capture_fine
    dense.dense_pose_ppca_score_with_moments_blocked = capture_op("fine_score_moments", original_score)
    dense.accumulate_pose_ppca_block_cached = capture_op("fine_backprojection", original_backproject)
    residual.residual_image_statistics = capture_op("fine_residual_images", original_residual)
    try:
        for label, current in (("memory_a", state), ("memory_b", state), ("restored", restored)):
            rows.append({"label": label})
            rows[-1]["result"] = loop.expectation(data, current, config, selected[selected % 2 == 0], 2)
    finally:
        loop.compute_dense_ppca_adaptive_significance = original_coarse
        loop.accumulate_dense_ppca_statistics = original_fine
        dense.dense_pose_ppca_score_with_moments_blocked = original_score
        dense.accumulate_pose_ppca_block_cached = original_backproject
        residual.residual_image_statistics = original_residual
    report["repeated_no_serialization"] = compare(rows[0]["result"], rows[1]["result"])
    report["restored_vs_memory"] = compare(rows[0]["result"], rows[2]["result"])
    report["coarse_repeat"] = compare(rows[0]["coarse"], rows[1]["coarse"])
    report["coarse_restored"] = compare(rows[0]["coarse"], rows[2]["coarse"])
    report["fine_repeat"] = compare(rows[0]["fine"], rows[1]["fine"])
    report["fine_restored"] = compare(rows[0]["fine"], rows[2]["fine"])
    report["operation_repeat"] = compare(rows[0]["operations"], rows[1]["operations"])
    report["operation_restored"] = compare(rows[0]["operations"], rows[2]["operations"])
    original_load = checkpoint.load
    continuations = {}
    for label, start in (("memory_a", state), ("memory_b", state), ("restored", None)):
        if start is not None:
            checkpoint.load = lambda *_args, _state=start, **_kwargs: _state
        try:
            final = loop.run(
                data, config, root / label, identity, manifest["particle_diameter_ang"], resume=path,
            )
        finally:
            checkpoint.load = original_load
        with np.load(root / label / "embeddings.npz", allow_pickle=False) as embedding:
            continuations[label] = {
                "state": final,
                "embedding_ids": embedding["particle_ids"].copy(),
                "embedding_z": embedding["z"].copy(),
            }
    report["continuation_repeat"] = compare(continuations["memory_a"], continuations["memory_b"])
    report["continuation_restored"] = compare(continuations["memory_a"], continuations["restored"])
    (root / "comparison.json").write_text(
        json.dumps(report, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n"
    )
    print(json.dumps({key: list(report[key]) for key in (
        "roundtrip", "repeated_no_serialization", "restored_vs_memory",
        "coarse_repeat", "coarse_restored", "fine_repeat", "fine_restored",
        "operation_repeat", "operation_restored", "continuation_repeat", "continuation_restored"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
