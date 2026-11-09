#!/usr/bin/env python3
"""Score PPCA runs on the cryo-ET contrast fixtures by their latents and, with the contrast latent, their contrast estimate.

The fixture (``em_fixtures/cryoet_ppca_k3conf_contrast30_box64_20261003``) has 399 subtomograms of three conformations and a
per-particle contrast shared by the tilts of a particle; its truth (``project/simulation_info.pkl``: ``particle_volume`` the
state, ``particle_contrast``) is read here for evaluation only. For each run directory (``embeddings.npz`` with ``z``) it
reports the nearest-centroid accuracy of the latents against the states, the ARI of a k-means with as many clusters as
states, the between-state R^2, and, when the run has a ``contrast`` estimate, its correlation with the true contrast, its
mean and spread. A run without the latent is scored the same way on ``z``.

Run:
    python scripts/ppca_contrast/eval_tomo_fixture.py RUN [RUN ...] --truth <project>/simulation_info.pkl [--json out.json]
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def nearest_centroid_accuracy(z: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of points whose nearest class mean (leave-one-out) is their own class."""
    classes = np.unique(labels)
    means = np.stack([z[labels == c].mean(axis=0) for c in classes])
    counts = np.array([(labels == c).sum() for c in classes])
    correct = 0
    for i in range(len(z)):
        adjusted = means.copy()
        k = int(np.searchsorted(classes, labels[i]))
        adjusted[k] = (means[k] * counts[k] - z[i]) / (counts[k] - 1)
        correct += int(np.argmin(np.linalg.norm(adjusted - z[i], axis=1)) == k)
    return correct / len(z)


def kmeans_ari(z: np.ndarray, labels: np.ndarray, seed: int = 0, restarts: int = 20) -> float:
    """Adjusted Rand index of the best of ``restarts`` k-means fits (lowest inertia) with one cluster per class."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    fit = KMeans(n_clusters=len(np.unique(labels)), n_init=restarts, random_state=seed).fit(z)
    return float(adjusted_rand_score(labels, fit.labels_))


def between_state_r2(z: np.ndarray, labels: np.ndarray) -> float:
    """Share of the latents' total variance that lies between the state means."""
    total = ((z - z.mean(axis=0)) ** 2).sum()
    within = sum(((z[labels == c] - z[labels == c].mean(axis=0)) ** 2).sum() for c in np.unique(labels))
    return float(1 - within / total)


def score_run(run: Path, labels: np.ndarray, contrast: np.ndarray) -> dict:
    """The scores of one run directory."""
    with np.load(run / "embeddings.npz", allow_pickle=False) as saved:
        order = np.argsort(saved["particle_ids"])
        z = saved["z"][order].astype(np.float64)
        estimate = saved["contrast"][order] if "contrast" in saved.files else None
        sd = float(saved["contrast_sd"]) if "contrast_sd" in saved.files else None
    row = {
        "nn accuracy": nearest_centroid_accuracy(z, labels),
        "kmeans ARI": kmeans_ari(z, labels),
        "R2 states": between_state_r2(z, labels),
    }
    if estimate is not None:
        row.update({"corr(c)": float(np.corrcoef(estimate, contrast)[0, 1]), "c mean": float(estimate.mean()),
                    "c sd": float(estimate.std()), "sd prior": sd})
    return row


def main() -> None:
    """Score the runs and print a table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--truth", type=Path, required=True, help="the fixture's project/simulation_info.pkl")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    with open(args.truth, "rb") as stream:
        truth = pickle.load(stream)
    labels, contrast = np.asarray(truth["particle_volume"]), np.asarray(truth["particle_contrast"])
    rows = {run.name: score_run(run, labels, contrast) for run in args.runs}
    columns = list(dict.fromkeys(key for row in rows.values() for key in row))
    print("| run | " + " | ".join(columns) + " |")
    print("|" + " --- |" * (len(columns) + 1))
    for name, row in rows.items():
        print(f"| {name} | " + " | ".join("" if row.get(c) is None else f"{row[c]:.3f}" for c in columns) + " |")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
