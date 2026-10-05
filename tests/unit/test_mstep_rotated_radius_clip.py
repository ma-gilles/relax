"""The M-step clip under anisotropic magnification is RELION's rotated-radius rule.

RELION backprojects pixel ``k`` of an image when its weight is nonzero (``ROUND(|k|) <=
current_size / 2``: ``Minvsigma2`` is zero beyond, ml_optimiser.cpp:6894-6901) and when the
rotated radius ``|A^-1 (x, y, 0)|`` is at most ``r_max`` (acc/cuda/cuda_kernels/BP.cuh:322,
BackProjector::backproject2Dto3D), with ``A = inv(M3) Aproj R`` (ObservationModel::applyAnisoMag).
For a scaled rotation the image clip alone is that rule; for an anisotropic ``M`` it is not.
The CPU reference below is written from that rule, independently of relax's mask; the GPU test
compares relax's adjoint with RELION's own BackProjector (the binding).
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from relax.helpers import adjoint

pytestmark = pytest.mark.unit

SYMMETRIC = np.array([[1.015, 0.004], [0.004, 0.99]])
ASYMMETRIC = np.array([[1.015, 0.012], [-0.004, 0.99]])


def _relion_matrices(mag, n, seed=0):
    """RELION's ``A = inv(M3) R`` for ``n`` random rotations (a tilt's Aproj folds into R)."""

    mag3 = np.eye(3)
    mag3[:2, :2] = mag
    rotations = Rotation.random(n, random_state=seed).as_matrix()
    return np.einsum("ij,njk->nik", np.linalg.inv(mag3), rotations)


def _relax_matrices(relion_matrices):
    """relax's M-step matrices are ``inv(A)^T`` (relion.optics_aberrations.relax_projection_magnification)."""

    return np.transpose(np.linalg.inv(relion_matrices), (0, 2, 1))


def _fftw_half_pixels(box):
    """FFTW half-image pixel indices (row-major, ``box // 2 + 1`` columns) and their ``x``, ``y``."""

    half = box // 2 + 1
    rows, x = np.meshgrid(np.arange(box), np.arange(half), indexing="ij")
    y = np.where(rows < half, rows, rows - box)
    return (rows * half + x).reshape(-1), x.reshape(-1), y.reshape(-1)


def _relion_kept(relion_matrices, x, y, r_max):
    """RELION's support, from the rule in the module docstring (float64)."""

    rounded = np.rint(np.sqrt(x * x + y * y)) <= r_max
    k = np.stack([x, y, np.zeros_like(x)], axis=-1).astype(np.float64)
    rotated = np.einsum("nij,pj->npi", np.linalg.inv(relion_matrices), k)
    return rounded[None, :] & (np.sum(rotated * rotated, axis=-1) <= r_max * r_max)


@pytest.mark.parametrize(
    ("matrix", "anisotropic"),
    [
        (np.eye(2), False),
        (1.02 * Rotation.from_euler("z", 17, degrees=True).as_matrix()[:2, :2], False),
        (SYMMETRIC, True),
        (ASYMMETRIC, True),
    ],
)
def test_anisotropy_is_decided_from_the_magnification_matrix(matrix, anisotropic):
    assert adjoint.magnification_is_anisotropic([matrix]) is anisotropic
    assert adjoint.magnification_is_anisotropic([np.eye(2), matrix]) is anisotropic


def test_anisotropic_clip_keeps_the_rounded_image_support_and_the_reference_radius():
    clip = adjoint.mstep_adjoint_max_r(36, None, 2, anisotropic_magnification=True)
    assert clip == adjoint.ReferenceSphereClip(18.5, 2, 18.0)
    assert adjoint._recovar_clip_kwargs(clip) == {"max_r": 18.5, "upsampling": 2}
    # Without magnification nothing changes.
    assert adjoint.mstep_adjoint_max_r(36, None, 2) == 18.0
    with pytest.raises(NotImplementedError, match="another pixel size"):
        adjoint.mstep_adjoint_max_r(36, 20.0, 2, anisotropic_magnification=True)


@pytest.mark.parametrize("mag", [SYMMETRIC, ASYMMETRIC], ids=["symmetric", "asymmetric"])
def test_mask_and_image_clip_are_relions_support(mag):
    box, r_max = 64, 18
    relion = _relion_matrices(mag, 25)
    pixels, x, y = _fftw_half_pixels(box)
    clip = adjoint.mstep_adjoint_max_r(2 * r_max, None, 2, anisotropic_magnification=True)
    image_kept = (x * x + y * y) <= clip.image_radius**2  # recovar's image clip at max_r
    mask = np.asarray(
        adjoint.rotated_radius_mask(pixels, np.asarray(_relax_matrices(relion), np.float32), (box, box), r_max)
    )
    relax_kept = image_kept[None, :] & (mask > 0)
    expected = _relion_kept(relion, x, y, r_max)
    assert np.array_equal(relax_kept, expected)
    # The rule is not the image clip: some pixels beyond r_max are kept and some inside are not.
    exact_image = (x * x + y * y) <= r_max * r_max
    assert np.any(expected & ~exact_image[None, :]) and np.any(exact_image[None, :] & ~expected)


@pytest.mark.gpu
@pytest.mark.requires_relion_bind
@pytest.mark.parametrize("mag", [None, SYMMETRIC, ASYMMETRIC], ids=["none", "symmetric", "asymmetric"])
def test_gpu_adjoint_weight_matches_relion_backprojector(mag):
    """Total BPref weight (one per kept pixel under trilinear splatting) against RELION's BackProjector."""

    import jax.numpy as jnp
    from relax.relion_bind._relion_bind_core import get_backprojector_data

    from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape

    box, current_size, padding = 64, 36, 2
    r_max = current_size // 2
    relion = _relion_matrices(np.eye(2) if mag is None else mag, 40, seed=3)
    pixels, x, y = _fftw_half_pixels(box)
    support = np.rint(np.sqrt(x * x + y * y)) <= r_max  # RELION's nonzero weights
    weights = support.astype(np.float64).reshape(box, box // 2 + 1)
    _data, relion_weight = get_backprojector_data(
        np.broadcast_to(weights, (len(relion),) + weights.shape).astype(np.complex128),
        relion,
        np.broadcast_to(weights, (len(relion),) + weights.shape).copy(),
        ori_size=box,
        padding_factor=padding,
        current_size=current_size,
    )

    window = pixels[support]
    clip = adjoint.mstep_adjoint_max_r(current_size, None, padding, anisotropic_magnification=mag is not None)
    volume_shape = relion_backprojector_volume_shape((box, box, box), padding, current_size=current_size)
    from relax.helpers.half_volume_mstep import half_volume_accumulator_shape

    volume = jnp.zeros(int(np.prod(half_volume_accumulator_shape(volume_shape))), jnp.float32)
    relax_weight = adjoint.adjoint_slice_volume_windowed(
        jnp.ones((len(relion), window.size), jnp.float32),
        jnp.asarray(window, jnp.int32),
        jnp.asarray(_relax_matrices(relion), jnp.float32),
        volume,
        (box, box),
        tuple(volume_shape),
        "linear_interp",
        True,
        True,
        clip,
        True,
    )
    # RELION's CPU BackProjector tests the rotated radius in double, relax's kernel (like RELION's GPU) in
    # float: pixels whose rotated radius is r_max to rounding may go either way.
    k = np.stack([x, y, np.zeros_like(x)], axis=-1)[support].astype(np.float64)
    rotated_r2 = np.sum(np.einsum("nij,pj->npi", np.linalg.inv(relion), k) ** 2, axis=-1)
    boundary = int(np.sum(np.abs(rotated_r2 - r_max**2) < 1e-4 * r_max**2))
    relax_total, relion_total = float(jnp.sum(relax_weight)), float(np.sum(relion_weight))
    assert abs(relax_total - relion_total) <= boundary + 1e-6 * relion_total
    if mag is not None:
        # The image clip alone (the engine before anisotropic magnification was handled) is far off.
        image_clip = np.sum((rotated_r2 <= r_max**2) & (np.sum(k * k, axis=-1) <= r_max**2)[None, :])
        assert abs(float(image_clip) - relion_total) > 4 * max(boundary, 1)
