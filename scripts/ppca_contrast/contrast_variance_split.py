#!/usr/bin/env python3
"""Split a contrast-latent run's contrast variance between the latent and the loadings that lie along the mean volume.

For each checkpoint of a ``relax ppca_initial_model --contrast latent`` run it prints the prior sd of the contrast latent, the
latent's population mean (``gain``), the mean contrast ``1 + sd gain``, and the variance of the mean-direction amplitude
carried by the latent (``sd^2``) and by the loadings (``sum_k coef_k^2`` with ``coef_k = Re<mu, w_k> / <mu, mu>``, the
inner product of the stored half-space Fourier coefficients). Only the sum is identified by the likelihood
(docs/math/ppca_contrast.md, section 10); the split shows where the loadings took variance from the latent. The last columns
are the contrast spread relative to the mean contrast, from the latent alone and from both, and the mean volume's power.

Run:
    python -I scripts/ppca_contrast/contrast_variance_split.py RUN [RUN ...] [--every 200]
where RUN holds ``iterations.jsonl`` and ``checkpoint_NNNN.npz`` (every 50 iterations).
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parallel_loading_variance(theta: np.ndarray) -> float:
    """Sum over loadings of the squared coefficient of each loading on the mean, in units of the mean.

    ``theta`` is the stored model, the mean in column 0 and the loadings after it (complex, half-space Fourier).
    """
    mean, loadings = theta[:, 0], theta[:, 1:]
    coefficients = np.real(np.conj(mean) @ loadings) / np.sum(np.abs(mean) ** 2)
    return float(np.sum(coefficients**2))


def split_rows(run: Path, every: int) -> list[dict[str, float]]:
    """One row per logged iteration (every ``every``, plus the last checkpoint) of the run's contrast variance split."""
    records = [json.loads(line) for line in (run / "iterations.jsonl").read_text().splitlines()]
    last = max(int(p.stem.split("_")[1]) for p in run.glob("checkpoint_*.npz"))
    rows = []
    for iteration in list(range(0, last, every)) + [last]:
        record = records[min(iteration, len(records) - 1)]
        contrast = record["contrast"]
        sd, gain = contrast["sd"], contrast["gain"]
        with np.load(run / f"checkpoint_{iteration:04d}.npz", allow_pickle=True) as checkpoint:
            parallel = parallel_loading_variance(checkpoint["theta"])
        mean_contrast = 1 + sd * gain
        rows.append(
            {
                "iteration": iteration,
                "sd": sd,
                "gain": gain,
                "mean_contrast": mean_contrast,
                "latent_variance": sd**2,
                "parallel_loading_variance": parallel,
                "relative_sd_latent": sd / mean_contrast,
                "relative_sd_total": float(np.sqrt(sd**2 + parallel)) / mean_contrast,
                "mean_power": record["mean_power"],
            }
        )
    return rows


def main() -> None:
    """Print the variance split of each run given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--every", type=int, default=200, help="iteration spacing of the printed rows")
    args = parser.parse_args()
    header = "iter    sd   gain  mean_c | sd^2  parallel | rel_sd(latent) rel_sd(total) | mean_power"
    for run in args.runs:
        print(run)
        print(header)
        for r in split_rows(run, args.every):
            print(
                f"{r['iteration']:5d} {r['sd']:.3f} {r['gain']:+.2f} {r['mean_contrast']:.3f} | "
                f"{r['latent_variance']:.4f} {r['parallel_loading_variance']:.4f} | "
                f"{r['relative_sd_latent']:.3f}          {r['relative_sd_total']:.3f}        | {r['mean_power']:.3f}"
            )


if __name__ == "__main__":
    main()
