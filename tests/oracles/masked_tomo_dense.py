"""Dense Gaussian reference for the masked-measurement likelihood (tiny sizes only, float64 diagnostic).

Independent of ``relax.masked_tomo``: given dense ``F`` (columns = model components' images in measurement
space), measurement ``V`` and covariance ``C``, it evaluates the PPCA marginal log density on the support of
``C`` with an eigendecomposition.
"""

import numpy as np


def support_pinv(C, rtol=1e-9):
    """Pseudo-inverse and pseudo-log-determinant of a symmetric PSD matrix on its numerical support."""
    w, U = np.linalg.eigh(C)
    keep = w > rtol * w.max()
    Ui = U[:, keep]
    return (Ui / w[keep]) @ Ui.T, float(np.sum(np.log(w[keep]))), Ui


def ppca_logpdf(V, F_mean, F_W, C, rtol=1e-9):
    """``log N(V; F_mean, F_W F_W^T + C)`` restricted to the support of ``C`` (``V`` and the columns of
    ``F_W`` must lie in it). ``F_mean`` ``(m,)``, ``F_W`` ``(m, q)``."""
    _, _, Ui = support_pinv(C, rtol)
    Vs, ms, Ws = Ui.T @ V, Ui.T @ F_mean, Ui.T @ F_W
    Cs = Ui.T @ C @ Ui
    cov = Ws @ Ws.T + Cs
    r = Vs - ms
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    return -0.5 * (r @ np.linalg.solve(cov, r) + logdet + r.size * np.log(2 * np.pi))
