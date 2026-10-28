"""Score how well a `relax ppca_initial_model` run's latent coordinates separate the true states of a fixture.

Reads RUN_DIR/embeddings.npz (the final model's posterior-mean latents of every particle) and the fixture's
evaluation labels, and writes RUN_DIR/latent_scores.json with, per run:

  between_state_r2   share of the latent variance (summed over coordinates) that the state label explains;
  ari_gmm            adjusted Rand index of a K-component Gaussian mixture fitted to the latents against the
                     labels (K = the number of states; the headline of docs/development/ppca_optimizer_axes.md
                     section 3, which prefers it to the R^2 because pose error inflates the within-state spread);
  spearman_first     |Spearman rank correlation| of the first latent coordinate with the state index, meaningful
                     when the state indices are ordered along one path (a diagnostic, not a gate);
  nn_accuracy        share of particles whose nearest neighbour (in the latents) in a random 5,000-particle
                     subsample carries the same label.

Ground truth is for evaluation only. Run (CPU, seconds):
  python scripts/analyze_ppca_latents.py RUN_DIR --truth FIXTURE/evaluation/truth.npz [--seed 0]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def between_state_r2(z: np.ndarray, labels: np.ndarray) -> float:
    """The share of the total latent variance explained by the state means: ``SS_between / SS_total``."""
    total = np.sum((z - z.mean(axis=0)) ** 2)
    between = sum(
        np.sum(labels == s) * np.sum((z[labels == s].mean(axis=0) - z.mean(axis=0)) ** 2) for s in np.unique(labels)
    )
    return float(between / total)


def gmm_ari(z: np.ndarray, labels: np.ndarray, seed: int) -> float:
    """Adjusted Rand index of a full-covariance mixture with one component per state against ``labels``."""
    mixture = GaussianMixture(n_components=len(np.unique(labels)), covariance_type="full", n_init=5, random_state=seed)
    return float(adjusted_rand_score(labels, mixture.fit_predict(z)))


def nn_accuracy(z: np.ndarray, labels: np.ndarray, seed: int, pool: int = 5000) -> float:
    """Share of particles whose nearest neighbour in a random ``pool``-particle subsample has their label.

    A particle in the pool does not count itself: its second neighbour is taken.
    """
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(z), size=min(pool, len(z)), replace=False)
    neighbours = NearestNeighbors(n_neighbors=2).fit(z[chosen])
    _, index = neighbours.kneighbors(z)
    in_pool = np.zeros(len(z), bool)
    in_pool[chosen] = True
    nearest = np.where(in_pool, index[:, 1], index[:, 0])
    return float(np.mean(labels[chosen[nearest]] == labels))


def score_run(run: Path, truth: Path, seed: int) -> dict:
    """The separation scores of ``run``'s final embeddings against the labels of ``truth``."""
    with np.load(run / "embeddings.npz", allow_pickle=False) as saved:
        ids, z = saved["particle_ids"], saved["z"].astype(np.float64)
    if not np.array_equal(np.sort(ids), np.arange(len(ids))):
        raise ValueError("embeddings.npz must hold every particle exactly once")
    z = z[np.argsort(ids)]
    with np.load(truth, allow_pickle=False) as saved:
        labels = saved["labels"]
    if len(labels) != len(z) or not np.all(np.isfinite(z)):
        raise ValueError(f"{len(z)} finite latents are needed for {len(labels)} labels")
    return {
        "n_particles": int(len(z)),
        "n_states": int(len(np.unique(labels))),
        "q": int(z.shape[1]),
        "between_state_r2": between_state_r2(z, labels),
        "ari_gmm": gmm_ari(z, labels, seed),
        "spearman_first": float(abs(spearmanr(z[:, 0], labels).statistic)),
        "nn_accuracy": nn_accuracy(z, labels, seed),
    }


def main() -> None:
    """Score a run and write ``latent_scores.json`` beside its embeddings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--truth", type=Path, required=True, help="the fixture's evaluation/truth.npz (labels)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the mixture fit and the neighbour subsample")
    args = parser.parse_args()
    scores = score_run(args.run, args.truth, args.seed)
    (args.run / "latent_scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
