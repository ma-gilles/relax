"""Independent reference for ``relax.multibody.pose_neighborhoods``: every weighted 3D point residual built explicitly
and the two depths solved as a least-squares problem, without the covariance reduction."""

from __future__ import annotations

import mpmath
import numpy as np
from scipy.optimize import lsq_linear


def direct_squared_score(points, weights, rotation_i, rotation_j, lift_i, lift_j, normal_i, normal_j, bound=None) -> float:
    """``min sum_l w_l ||(Q_i - Q_j) x_l + u_i - u_j + l_i n_i - l_j n_j||^2`` over the depths, from the ``3M x 2``
    system; ``bound=None`` leaves the depths free. BVLS with ``bound``, the minimum-norm solution without."""

    points = np.asarray(points, dtype=np.float64)
    root = np.sqrt(np.asarray(weights, dtype=np.float64) / np.sum(weights))[:, None]
    constant = root * (points @ (rotation_i - rotation_j).T + (lift_i - lift_j))
    design = np.stack([(root * normal_i).reshape(-1), (root * -normal_j).reshape(-1)], axis=1)
    target = -constant.reshape(-1)
    if bound is None:
        depths = np.linalg.lstsq(design, target, rcond=None)[0]
    else:
        depths = lsq_linear(design, target, bounds=(-bound, bound), method="bvls", tol=1e-14, max_iter=100).x
    residual = design @ depths - target
    return float(residual @ residual)


def direct_free_squared_score_high_precision(points, weights, rotation_i, rotation_j, lift_i, lift_j, normal_i, normal_j) -> float:
    """The free-depth score of :func:`direct_squared_score` in 120-digit arithmetic: the double-precision system loses
    about ``1 / angle**2`` of its accuracy for nearly parallel views, the reference must not."""

    mpmath.mp.dps = 120
    weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    a, b = mpmath.matrix(normal_i.tolist()), mpmath.matrix((-normal_j).tolist())
    rows = [(mpmath.mpf(float(w)), mpmath.matrix((rotation_i - rotation_j).tolist()) * mpmath.matrix(x.tolist())
             + mpmath.matrix((lift_i - lift_j).tolist())) for w, x in zip(weights, np.asarray(points, dtype=np.float64))]
    dot = lambda p, q: sum(p[k] * q[k] for k in range(3))
    mass = sum(w for w, _ in rows)
    aa, bb, ab = mass * dot(a, a), mass * dot(b, b), mass * dot(a, b)
    ar, br = sum(w * dot(a, r) for w, r in rows), sum(w * dot(b, r) for w, r in rows)
    det = aa * bb - ab * ab
    total = sum(w * dot(r, r) for w, r in rows)
    if det == 0:
        return float(total - ar * ar / aa)
    x, y = (-ar * bb + br * ab) / det, (-br * aa + ar * ab) / det
    return float(total + x * ar + y * br)


def true_squared_displacement(points, weights, rotation_i, rotation_j, translation_i, translation_j) -> float:
    """The weighted mean squared displacement of corresponding points of two full 3D poses."""

    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    moved = points @ (rotation_i - rotation_j).T + (translation_i - translation_j)
    return float(weights @ np.sum(moved * moved, axis=1))
