"""The pass-1 program on RELION's exact coarse operands against the generic dense scorer.

The removed generic scorer (tests/helpers/generic_coarse_reference.py, NumPy float64) scores
the same images and references from the CTF-weighted, noise-whitened operands; the exact
operands divide the translated image by the CTF and carry CTF^2 / sigma^2 in the pixel
weight. The two agree up to a constant per image (the image energy and initial diff2 for the
Gaussian score, the inverse image power for the normalized CC), so the program's scores are
compared after removing that constant.
"""

import jax.numpy as jnp
import numpy as np
from helpers.float_compare import assert_matches
from helpers.generic_coarse_reference import gaussian_block_scores, normalized_cc_block_scores

from relax.scoring import pass1_program
from relax.scoring.pass1_scores import ProgramStatics

N_CLASSES, N_ROT, BLOCK, N_IMAGES, N_TRANS, N_PIXELS = 2, 6, 4, 3, 4, 13


def _operands(seed=20261003):
    rng = np.random.default_rng(seed)

    def complex_normal(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    projected = complex_normal(N_CLASSES, N_ROT, N_PIXELS)
    # A translation is a phase ramp: it keeps every pixel's modulus, hence the image energy.
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(N_TRANS, N_PIXELS)))
    translated = complex_normal(N_IMAGES, 1, N_PIXELS) * phases[None]
    ctf = rng.uniform(0.3, 1.0, size=(N_IMAGES, N_PIXELS)) * rng.choice([-1.0, 1.0], size=(N_IMAGES, N_PIXELS))
    sigma2 = rng.uniform(0.5, 2.0, size=(N_IMAGES, N_PIXELS))
    half_weights = rng.choice([1.0, 2.0], size=N_PIXELS)
    return projected, translated, ctf, sigma2, half_weights


def _program_scores(projected, shifted, pixel_weight, initial_diff2, score_kind):
    blocks = tuple((k, r0, min(BLOCK, N_ROT - r0), BLOCK) for k in range(N_CLASSES) for r0 in range(0, N_ROT, BLOCK))
    zeros = (
        jnp.full(N_IMAGES, -jnp.inf, dtype=jnp.float32),
        jnp.zeros(N_IMAGES, dtype=jnp.float32),
        jnp.zeros(N_IMAGES, dtype=jnp.int32),
    )
    state, values, _ = pass1_program.coarse_pass1_blocks(
        pass1_program.pass1_initial_state(zeros, N_CLASSES),
        jnp.asarray(projected, dtype=jnp.complex64),
        jnp.asarray(shifted, dtype=jnp.complex64),
        jnp.asarray(pixel_weight, dtype=jnp.float32),
        None if initial_diff2 is None else jnp.asarray(initial_diff2, dtype=jnp.float32),
        N_IMAGES,
        tuple((jnp.asarray(0.0, dtype=jnp.float32), None) for _ in blocks),
        None,
        blocks=blocks,
        statics=ProgramStatics(
            n_trans=N_TRANS,
            image_shape=(4, 4),
            volume_shape=(4, 4, 4),
            float64=False,
            score_kind=score_kind,
            exact_weight_order=False,
            return_class_best=True,
        ),
    )
    scores = np.concatenate([np.asarray(v) for v in values], axis=1).reshape(N_IMAGES, N_CLASSES, N_ROT, N_TRANS)
    return state, scores


def test_gaussian_program_scores_are_the_generic_scores_up_to_an_image_constant():
    projected, translated, ctf, sigma2, half_weights = _operands()
    rng = np.random.default_rng(7)
    initial_diff2 = rng.uniform(0.0, 3.0, size=N_IMAGES)
    # Exact operands: the image divided by the CTF, weighted by CTF^2 / sigma^2 and the
    # half-spectrum multiplicity (relax.relion.relion_coarse_operands).
    state, program = _program_scores(
        projected,
        translated / ctf[:, None, :],
        ctf**2 / sigma2 * half_weights,
        initial_diff2,
        "gaussian",
    )
    reference = np.stack(
        [
            gaussian_block_scores(
                ctf[:, None, :] * translated / sigma2[:, None, :], ctf**2 / sigma2, projected[k], half_weights
            )
            for k in range(N_CLASSES)
        ],
        axis=1,
    )
    program_relative = program - program[:, :1, :1, :1]
    reference_relative = reference - reference[:, :1, :1, :1]
    # float32 GEMMs against float64: the relative scores agree to 1.3e-7 of their span here
    # (CPU, 2026-10-02); 1e-5 leaves the float32 accumulation of larger cases room.
    span = np.ptp(reference_relative)
    np.testing.assert_allclose(program_relative, reference_relative, rtol=0, atol=1e-5 * span)
    best = reference.reshape(N_IMAGES, -1).argmax(axis=1)
    best_score, best_pose, best_class = (np.asarray(value) for value in state[2])
    assert_matches(best_class, best // (N_ROT * N_TRANS))
    assert_matches(best_pose, best % (N_ROT * N_TRANS))


def test_normalized_cc_program_scores_are_the_generic_scores_up_to_an_image_scale():
    projected, translated, ctf, _sigma2, half_weights = _operands(seed=11)
    inverse_power = np.random.default_rng(3).uniform(0.2, 2.0, size=N_IMAGES)
    # Exact CC operands: corr_img (CTF^2) times the half weights, the image divided by the CTF.
    state, program = _program_scores(
        projected,
        translated / ctf[:, None, :],
        ctf**2 * half_weights,
        None,
        "normalized_cc",
    )
    reference = np.stack(
        [
            normalized_cc_block_scores(
                ctf[:, None, :] * translated * inverse_power[:, None, None],
                ctf**2 * inverse_power[:, None],
                projected[k],
                half_weights,
            )
            for k in range(N_CLASSES)
        ],
        axis=1,
    )
    # The generic CC carries the inverse image power: sqrt(inverse power) per image.
    # The scaled scores agree to 1.8e-6 of the largest score here (CPU, 2026-10-02).
    scale = np.sqrt(inverse_power)[:, None, None, None]
    np.testing.assert_allclose(program * scale, reference, rtol=0, atol=1e-5 * np.abs(reference).max())
    best = reference.reshape(N_IMAGES, -1).argmax(axis=1)
    _, best_pose, best_class = (np.asarray(value) for value in state[2])
    assert_matches(best_class, best // (N_ROT * N_TRANS))
    assert_matches(best_pose, best % (N_ROT * N_TRANS))
