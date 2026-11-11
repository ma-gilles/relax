#!/usr/bin/env python3
"""Accuracy of the contrast quadrature of model B: how many nodes per pose, and where to put them.

Images are drawn from the physical model ``y = c (a + B z) + noise`` (white unit noise, ``c ~ N(1, sd^2)``) and scored at the true
pose and at wrong poses whose mean and loadings correlate with the true ones by ``CORRELATIONS`` (neighbouring poses of a grid
down to unrelated ones). The reference is a dense trapezoid grid over ``c`` (``N_REFERENCE`` nodes on ``1 +- 6 sd``, ``z`` in closed
form, float64, independent code). Three node placements are compared with it, each at several node counts:

* ``adaptive``: Gauss-Hermite nodes on model A's Gaussian posterior of the contrast, per pose
  (:func:`relax.ppca_refinement.contrast_b.adaptive_nodes`, the production placement);
* ``prior``: Gauss-Hermite nodes on the prior ``N(1, sd^2)``, the same for every pose;
* ``grid``: equally spaced trapezoid nodes on ``1 +- 4 sd``.

Per setting and scheme: the largest error of the pose log-evidence over the poses within ``RELEVANT_NATS`` of their image's best pose
and the mean error over all poses (nats), the total-variation distance of the posterior
over poses of an image, the relative errors of the moments the M-step reads at the true pose (``E[c]``, ``E[c z]``,
``E[c^2 z z^T]``), and the quantization residual: the distance from the true posterior mean of ``c`` to the nearest node, in units
of the posterior sd. ``nu`` is the integrated signal-to-noise ``<a, a>``; the posterior sd of ``c`` is about ``nu^-1/2``.

Run (CPU, a minute):
    python scripts/ppca_contrast/quadrature_accuracy.py [--images 300] [--sd 0.3] [-o table.md]
"""

import argparse
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relax.ppca_refinement import contrast_b  # noqa: E402

N_PIXELS, N_REFERENCE = 600, 4001
RELEVANT_NATS = 15.0  # a pose within this many nats of its image's best pose counts as carrying posterior mass
CORRELATIONS = (0.99, 0.95, 0.8, 0.5, 0.0)  # of the wrong poses' mean and loadings with the true pose's
SETTINGS = [(q, ratio, nu) for q in (3, 10) for ratio in (0.1, 1.0) for nu in (20.0, 200.0, 2000.0)]
SCHEMES = (
    [("adaptive", k) for k in (1, 2, 3, 4, 5)]
    + [("prior", k) for k in (5, 9, 17, 33)]
    + [("grid", k) for k in (9, 17, 33, 65, 129)]
)


def simulate(rng, q: int, ratio: float, nu: float, sd: float, n_images: int):
    """``(poses, y)``: the true pose and wrong poses as ``(a, B)`` pairs, and images from the true pose."""
    a = rng.normal(size=N_PIXELS)
    a *= np.sqrt(nu) / np.linalg.norm(a)
    loadings = rng.normal(size=(N_PIXELS, q))
    loadings *= np.sqrt(ratio * nu / q) / np.linalg.norm(loadings, axis=0)
    contrast = 1.0 + sd * rng.normal(size=n_images)
    y = contrast[:, None] * (a + rng.normal(size=(n_images, q)) @ loadings.T) + rng.normal(size=(n_images, N_PIXELS))

    def mix(x, rho):
        other = rng.normal(size=x.shape)
        other *= np.linalg.norm(x, axis=0) / np.linalg.norm(other, axis=0)
        return rho * x + np.sqrt(1 - rho**2) * other

    return [(a, loadings)] + [(mix(a, rho), mix(loadings, rho)) for rho in CORRELATIONS], y


def engine_statistics(poses, y):
    """The engine's pass-1 operands of every pose: ``inner`` ``(q + 1, R, B, 1)`` and the packed Gram ``(tri(q + 1), R, B)``."""
    q = poses[0][1].shape[1]
    inner = np.stack([np.stack([y @ a] + [y @ b for b in loadings.T]) for a, loadings in poses], axis=1)[..., None]
    entries = []
    for a, loadings in poses:
        columns = np.concatenate([a[:, None], loadings], axis=1)
        entries.append(columns.T @ columns)
    gram = np.stack([np.stack([m[i, j] for i, j in zip(*np.triu_indices(q + 1))]) for m in entries], axis=1)
    return inner, np.broadcast_to(gram[:, :, None], gram.shape + (y.shape[0],))


def reference(poses, y, sd: float):
    """Dense-grid evidence and moments per pose and image, float64, by direct solves (no eigenbasis, no proposal)."""
    q = poses[0][1].shape[1]
    grid = np.linspace(1 - 6 * sd, 1 + 6 * sd, N_REFERENCE)
    step = grid[1] - grid[0]
    out = {k: [] for k in ("log_z", "e_c", "e_c2", "e_cz", "e_c2zz", "post_sd")}
    for a, loadings in poses:
        t, g = y @ a, y @ loadings
        nu, h, H = a @ a, loadings.T @ a, loadings.T @ loadings
        terms = np.empty((N_REFERENCE, len(y)))
        mean_z = np.empty((N_REFERENCE, len(y), q))
        cov = np.empty((N_REFERENCE, q, q))
        for k, c in enumerate(grid):
            precision = np.eye(q) + c**2 * H
            b = c * (g - c * h)
            solved = np.linalg.solve(precision, b.T).T
            cov[k] = np.linalg.inv(precision)
            mean_z[k] = solved
            terms[k] = (
                -0.5 * (c**2 * nu - 2 * c * t - np.sum(b * solved, axis=1) + np.linalg.slogdet(precision)[1])
                - 0.5 * ((c - 1) / sd) ** 2
                - np.log(sd * np.sqrt(2 * np.pi))
            )
        log_z = logsumexp(terms, axis=0) + np.log(step)
        rho = np.exp(terms - logsumexp(terms, axis=0))  # (grid, images)
        c_col = grid[:, None]
        e_c = np.sum(rho * c_col, axis=0)
        out["log_z"].append(log_z)
        out["e_c"].append(e_c)
        out["e_c2"].append(np.sum(rho * c_col**2, axis=0))
        out["post_sd"].append(np.sqrt(np.sum(rho * c_col**2, axis=0) - e_c**2))
        out["e_cz"].append(np.einsum("kb,kbj->bj", rho * c_col, mean_z))
        second = np.einsum("kb,kbi,kbj->bij", rho * c_col**2, mean_z, mean_z)
        second += np.einsum("kb,kij->bij", rho * c_col**2, cov)
        out["e_c2zz"].append(second)
    return {k: np.stack(v) for k, v in out.items()}


def scheme_nodes(scheme: str, n_nodes: int, inner, factors, sd: float):
    """``(c, log_weight, g_rot)`` of a node placement; the weights include the quadrature measure."""
    if scheme == "adaptive":
        return contrast_b.adaptive_nodes(inner, factors, 1.0, sd, contrast_b.hermite_rule(n_nodes))
    t, g_rot = inner[0], contrast_b.rotate_in(factors, inner[1:])
    if scheme == "prior":
        x, w = contrast_b.hermite_rule(n_nodes)
        c = jnp.asarray(1.0 + sd * x)[:, None, None, None] * jnp.ones_like(t)[None]
        prior = -0.5 * ((c - 1.0) / sd) ** 2 - jnp.log(sd * jnp.sqrt(2 * jnp.pi))
        log_weight = jnp.log(jnp.asarray(w))[:, None, None, None] + contrast_b.node_log_integrand(c, t, g_rot, factors, 1.0, sd) - prior
        return c, log_weight, g_rot
    nodes = np.linspace(1 - 4 * sd, 1 + 4 * sd, n_nodes)
    c = jnp.asarray(nodes)[:, None, None, None] * jnp.ones_like(t)[None]
    return c, np.log(nodes[1] - nodes[0]) + contrast_b.node_log_integrand(c, t, g_rot, factors, 1.0, sd), g_rot


def evaluate(scheme: str, n_nodes: int, inner, gram, ref, sd: float, q: int) -> dict:
    """Errors of one scheme against the reference (arrays ``(poses, images, ...)``)."""
    factors = contrast_b.gram_factors(jnp.asarray(gram), q)
    c, log_weight, g_rot = scheme_nodes(scheme, n_nodes, jnp.asarray(inner), factors, sd)
    log_z = np.asarray(jax.scipy.special.logsumexp(log_weight, axis=0))[..., 0]
    moments = contrast_b.moments_from_nodes(c, log_weight, g_rot, factors, jnp.ones_like(jnp.asarray(inner[0])), q)
    index = contrast_b.packed_index(q + 1)
    weights, sums = np.asarray(moments.weights)[..., 0], np.asarray(moments.sums)
    e_c, e_cz = weights[0], np.moveaxis(weights[1:], 0, -1)
    e_c2zz = np.stack(
        [np.stack([sums[index[(min(i, j) + 1, max(i, j) + 1)]] for j in range(q)], axis=-1) for i in range(q)], axis=-2
    )  # (poses, images, q, q)
    log_post = log_z - logsumexp(log_z, axis=0)
    log_post_ref = ref["log_z"] - logsumexp(ref["log_z"], axis=0)
    tv = 0.5 * np.sum(np.abs(np.exp(log_post) - np.exp(log_post_ref)), axis=0)
    nearest = np.min(np.abs(np.asarray(c)[..., 0] - ref["e_c"][None]), axis=0)  # (poses, images)

    def relative(x, y):
        return np.linalg.norm((x - y).reshape(len(x), -1)) / np.linalg.norm(y.reshape(len(y), -1))

    err = np.abs(log_z - ref["log_z"])
    relevant = log_post_ref > -RELEVANT_NATS  # poses that carry posterior mass for their image
    return {
        "max |dlogZ| (relevant)": float(err[relevant].max()),
        "mean |dlogZ|": float(err.mean()),
        "mean TV": float(tv.mean()),
        "E[c] err / sd": float(np.mean(np.abs(e_c[0] - ref["e_c"][0]) / ref["post_sd"][0])),
        "E[cz] rel": float(relative(e_cz[0], ref["e_cz"][0])),
        "E[c2zz] rel": float(relative(e_c2zz[0], ref["e_c2zz"][0])),
        "residual / sd": float(np.mean(nearest[0] / ref["post_sd"][0])),
    }


def main() -> None:
    """Print one table per setting and the worst case of every scheme."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=300)
    parser.add_argument("--sd", type=float, default=0.3, help="true and model contrast sd")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    lines, worst = [], {}
    for q, ratio, nu in SETTINGS:
        rng = np.random.default_rng(args.seed)
        poses, y = simulate(rng, q, ratio, nu, args.sd, args.images)
        inner, gram = engine_statistics(poses, y)
        ref = reference(poses, y, args.sd)
        lines += [f"\nq = {q}, loading/mean ratio {ratio}, nu = {nu:g}, true and model sd {args.sd}", ""]
        header = None
        for scheme, n_nodes in SCHEMES:
            row = evaluate(scheme, n_nodes, inner, gram, ref, args.sd, q)
            if header is None:
                header = "| scheme | nodes | " + " | ".join(row) + " |"
                lines += [header, "|" + " --- |" * (2 + len(row))]
            lines.append(f"| {scheme} | {n_nodes} | " + " | ".join(f"{v:.2e}" for v in row.values()) + " |")
            previous = worst.setdefault((scheme, n_nodes), dict(row))
            for key, value in row.items():
                previous[key] = max(previous[key], value)
    lines += ["", "Worst case over the settings above", ""]
    keys = list(next(iter(worst.values())))
    lines += ["| scheme | nodes | " + " | ".join(keys) + " |", "|" + " --- |" * (2 + len(keys))]
    lines += [f"| {s} | {k} | " + " | ".join(f"{worst[(s, k)][key]:.2e}" for key in keys) + " |" for s, k in SCHEMES]
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
