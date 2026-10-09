"""The generic dense coarse scorer, kept as an independent float64 reference for pass 1.

Production pass 1 scores RELION's exact coarse operands with the coarse GEMMs
(``relax.scoring.pass1_program._coarse_pass1_blocks``). Until 2026-10-02 pass 1 also had a
generic dense scorer on the CTF-weighted, noise-whitened half-spectrum operands
(``_e_step_block_scores`` and ``_e_step_block_scores_normalized_cc``, now test helpers in
``tests/helpers/dense_block_scores.py``, on the significance operands). Its arithmetic is restated here in NumPy float64 so the
program can be checked against an implementation that shares none of its operands, layout
or reductions.

Operands, for ``B`` images, ``T`` translations, ``R`` rotations and ``P`` half-spectrum
pixels: ``shifted`` is ``[B, T, P]`` (the translated image, CTF-weighted and divided by the
noise for the Gaussian score; CTF-weighted and scaled by the inverse image power for the
normalized CC), ``ctf2_weight`` is ``[B, P]`` (``CTF^2 / sigma^2``, or ``CTF^2`` times the
inverse image power), ``projected`` is ``[R, P]`` and ``half_weights`` is ``[P]`` (the
half-spectrum multiplicity).
"""

import numpy as np


def _cross_and_model_energy(shifted, ctf2_weight, projected, half_weights):
    shifted = np.asarray(shifted, dtype=np.complex128)
    projected = np.asarray(projected, dtype=np.complex128)
    weights = np.asarray(half_weights, dtype=np.float64)
    cross = -2.0 * np.einsum("btp,rp->brt", np.conj(shifted), projected * weights).real
    model_energy = np.einsum("bp,rp->br", np.asarray(ctf2_weight, dtype=np.float64), np.abs(projected) ** 2 * weights)
    return cross, model_energy


def gaussian_block_scores(shifted, ctf2_weight, projected, half_weights):
    """``[B, R, T]`` generic Gaussian scores ``-(model energy + cross) / 2``.

    They omit the image-energy term, a per-image constant, so they equal the exact
    operands' scores up to a constant per image.
    """

    cross, model_energy = _cross_and_model_energy(shifted, ctf2_weight, projected, half_weights)
    return -0.5 * (cross + model_energy[:, :, None])


def normalized_cc_block_scores(shifted, ctf2_weight, projected, half_weights):
    """``[B, R, T]`` generic normalized cross-correlation ``-cross / 2 / sqrt(model energy)``."""

    cross, model_energy = _cross_and_model_energy(shifted, ctf2_weight, projected, half_weights)
    return (-0.5 * cross) / np.sqrt(np.maximum(model_energy, 1e-30))[:, :, None]
