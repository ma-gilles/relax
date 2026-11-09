#!/usr/bin/env python3
"""Contrast-aware inference for the suite scorer: write the scorer's ``inference.npz`` from a run's own E-step.

The suite scorer infers latents and poses with its own pinned code, which has unit contrast. For a run trained with
``--contrast latent`` that would score the model with an inference that ignores what it learned. This script runs the
run's own E-step (the stream, with the contrast latent when the run had one) on the scorer's evaluation images, on
the scorer's protocol grid, and writes ``suite_score_checkpoint_NNNN/inference.npz`` in the layout the scorer's
``--reuse-inference`` reads. The scorer then scores these latents and poses (clustering, FSC, the direct pose fit).

The scorer works on the simulator's sign (protein dark) and a checkpoint copy with its mean negated
(``scripts/ppca_suite_score_star.py``). This script infers on the run's own sign (the RELION-sign STAR the run trained
on, the original checkpoint) and writes ``z`` of the scorer's sign, ``-z``; poses do not depend on the sign.

Latents under the contrast latent: ``u = c z`` keeps the particle's contrast scale; ``u / c`` is the estimate of
``z`` (docs/math/ppca_contrast.md). ``--latent`` picks which one becomes ``z`` (both are saved).

Run on a GPU node (the training environment: this checkout on PYTHONPATH, RELAX_CUDA_LIB set):
    python scripts/ppca_contrast/infer_for_suite.py WORK --run RUN --star particles.star --evaluation DS/evaluation \
        --checkpoints 1000 3000 [--latent unscaled]
where WORK is ``<run>/suite_work/<dataset>__<arm>_s<seed>``, the scorer's work directory.
"""

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Run as python scripts/ppca_contrast/infer_for_suite.py: the repository root must be on the path.
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROTOCOL_SHIFT_RANGE, PROTOCOL_SHIFT_STEP = 6, 2  # the scorer's protocol translation grid, in box-64 pixels
BOX64_CURRENT_SIZE = 62


def file_sha256(path: Path) -> str:
    """The sha256 hex digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_from_metadata(saved: dict, order: int, radius: int, shift_range: float, shift_step: float):
    """The run's :class:`Config` with the inference settings: one stage at ``(radius, order)``, the scorer's
    translation grid, all PCs, no oversampling and exact float32 GEMMs; the model settings (contrast) are the run's."""
    from relax.ppca_initial_model.config import Config

    known = {field.name for field in dataclasses.fields(Config)}
    values = {key: value for key, value in saved.items() if key in known}
    values.update(
        stages=((1, radius, order),), pc_stages=None, radius_rule="stages", order_rule="stages", oversampling=0,
        shift_range=shift_range, shift_step=shift_step, gemm_precision="fp32", contrast_start=1,
        stochastic_batch_size=None, stochastic_all_iterations=False, balanced_stochastic_halves=False,
        skip_final_embeddings=True,
    )
    return Config(**values)


def scorer_order(saved: dict, iteration: int) -> int:
    """The HEALPix order the suite scorer infers at: ``max(3, the checkpoint's stage order)``."""
    stage_order = next(order for start, _, order in reversed(saved["stages"]) if start <= max(iteration, 1))
    return max(3, int(stage_order))


def infer_checkpoint(dataset, run_checkpoint: Path, ids: np.ndarray, args) -> dict:
    """The run's E-step on the images ``ids`` at one checkpoint: ``z`` (both latents), the contrast estimate and poses."""
    import jax.numpy as jnp

    from relax.ppca_initial_model.iteration_loop import contrast_sd_for, expectation
    from relax.ppca_initial_model.state import State
    from relax.ppca_refinement.contrast import latents_over_contrast, split_embeddings

    with np.load(run_checkpoint, allow_pickle=False) as saved:
        theta, noise, prior = saved["theta"], saved["noise"], saved["direction_prior"]
        meta = json.loads(str(saved["metadata"]))
    order = scorer_order(meta["config"], meta["iteration"]) if args.order is None else args.order
    config = config_from_metadata(meta["config"], order, args.radius, args.shift_range, args.shift_step)
    state = State(
        jnp.asarray(theta), None, jnp.asarray(noise), meta["iteration"], np.zeros(0, np.int64), {},
        float(meta["offset_variance"]), meta["radius"], {}, prior if prior.size else None, meta["direction_order"],
        None, None, None, meta.get("contrast_sd"),
    )
    stats = expectation(dataset, state, config, ids, 1, keep_poses=True)
    embeddings = np.asarray(stats.embeddings)
    sd = contrast_sd_for(config, state, 1)
    if sd is None:
        latent, estimate = embeddings, np.ones(len(embeddings), np.float32)
    else:
        latent, estimate = (np.asarray(x) for x in split_embeddings(embeddings, sd))
    unscaled, n_floored = latents_over_contrast(latent, estimate)
    if not np.array_equal(np.asarray(stats.original_image_ids), ids):
        raise ValueError("The E-step returned the images in another order than requested")
    return dict(
        order=order, scaled=latent, unscaled=unscaled, contrast=estimate, contrast_sd=sd, n_below_floor=n_floored,
        rotation_idx=np.asarray(stats.diagnostics["best_rotation_idx"], np.int32),
        translation_idx=np.asarray(stats.diagnostics["best_translation_idx"], np.int32),
    )


def main(argv=None) -> int:
    """Infer every requested checkpoint and write the scorer's inference files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("work", type=Path, help="the scorer's work directory of the run")
    parser.add_argument("--run", type=Path, required=True, help="the run directory (checkpoint_NNNN.npz)")
    parser.add_argument("--star", type=Path, required=True, help="the RELION-sign particle STAR the run trained on")
    parser.add_argument("--evaluation", type=Path, required=True, help="<dataset>/evaluation")
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--latent", choices=("scaled", "unscaled"), default="unscaled")
    parser.add_argument("--radius", type=int, default=BOX64_CURRENT_SIZE // 2, help="Fourier radius of the inference")
    parser.add_argument("--shift-range", type=float, default=PROTOCOL_SHIFT_RANGE)
    parser.add_argument("--shift-step", type=float, default=PROTOCOL_SHIFT_STEP)
    parser.add_argument("--order", type=int, help="HEALPix order (default: the scorer's rule)")
    parser.add_argument("--limit", type=int, help="infer only the first N evaluation images (smoke tests)")
    args = parser.parse_args(argv)

    from relax.commands.ppca_initial_model import load_training
    from relax.sampling import grids
    from scripts.ppca_suite_score_star import write_mean_negated_copy

    ids = np.union1d(
        np.load(args.evaluation / "eval_ids_stratified.npy"), np.load(args.evaluation / "eval_ids_random2000.npy")
    ).astype(np.int64)
    if args.limit:
        ids = ids[: args.limit]
    args.work.mkdir(parents=True, exist_ok=True)
    dataset, _, _ = load_training(args.star, "auto")
    offsets = grids.get_relion_translation_grid(args.shift_range, args.shift_step).astype(np.float32)
    for number in args.checkpoints:
        run_checkpoint = args.run / f"checkpoint_{number:04d}.npz"
        result = infer_checkpoint(dataset, run_checkpoint, ids, args)
        copy = write_mean_negated_copy(run_checkpoint, args.work)
        output = args.work / f"suite_score_checkpoint_{number:04d}"
        output.mkdir(parents=True, exist_ok=True)
        np.savez(
            output / "inference.npz", particle_ids=ids, z=-result[args.latent], rotation_idx=result["rotation_idx"],
            healpix_order=result["order"], checkpoint_sha256=file_sha256(copy), translation_idx=result["translation_idx"],
            translation_grid="protocol", translation_offsets=offsets, current_size=2 * args.radius,
            contrast=result["contrast"], z_scaled=-result["scaled"], z_unscaled=-result["unscaled"],
            latent=args.latent,
        )
        print(f"checkpoint {number}: order {result['order']}, contrast sd {result['contrast_sd']}, "
              f"{len(ids)} images ({result['n_below_floor']} with a contrast below the floor) -> {output / 'inference.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
