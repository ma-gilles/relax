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
from helpers.float_compare import assert_matches
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
    assert clip.upsampling == 2
    assert_matches([clip.image_radius, clip.reference_radius], [18.5, 18.0])
    kwargs = adjoint._recovar_clip_kwargs(clip)
    assert kwargs["upsampling"] == 2
    assert_matches(kwargs["max_r"], 18.5)
    # Without magnification nothing changes.
    assert_matches(adjoint.mstep_adjoint_max_r(36, None, 2), 18.0)
    # An optics group on another grid (relax#48, relax#53): scale <= 1 keeps the same clip, scale = 1
    # exactly is the reference grid's clip, and scale > 1 covers its window at r_max * s + 1.
    assert adjoint.mstep_adjoint_max_r(36, 18.0, 2, anisotropic_magnification=True) == clip
    on_finer_grid = adjoint.mstep_adjoint_max_r(36, 18.0 * 0.971, 2, anisotropic_magnification=True)
    assert on_finer_grid == clip
    wider = adjoint.mstep_adjoint_max_r(36, 18.0 * 1.1, 2, anisotropic_magnification=True)
    assert_matches([wider.image_radius, wider.reference_radius, wider.scale], [18.0 * 1.1 + 1.0, 18.0, 1.1])
    # A runtime image radius gives back its reference radius under the same relation.
    assert_matches([clip.runtime_reference_radius(13.5), wider.runtime_reference_radius(12 * 1.1 + 1.0)], [13.0, 12.0])


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


def _group_support(x, y, r_max, scale):
    """The window of an optics group on another grid: its rounded image support at ``r_max * s``."""

    return np.rint(np.sqrt(x * x + y * y)) <= np.floor(r_max * scale + 0.5)


@pytest.mark.parametrize("scale", [1.0, 1.0 + 2e-8, 1.1], ids=["s1", "s1+2e-8", "s1.1"])
@pytest.mark.parametrize("mag", [SYMMETRIC, ASYMMETRIC], ids=["symmetric", "asymmetric"])
def test_group_on_a_wider_grid_keeps_relions_support(mag, scale):
    """relax#53: a group whose field of view is ``s >= 1`` times the reference's, with anisotropic
    magnification. Its window holds its rounded support; the image clip must keep every window pixel
    and the mask must apply ``|A^-1 k| <= r_max`` with ``A = s inv(M3) R`` (applyScaleDifference)."""

    box, r_max = 64, 18
    relion = scale * _relion_matrices(mag, 25)
    pixels, x, y = _fftw_half_pixels(box)
    window = _group_support(x, y, r_max, scale)
    clip = adjoint.mstep_adjoint_max_r(2 * r_max, r_max * scale, 2, anisotropic_magnification=True)
    image_kept = (x * x + y * y) <= clip.image_radius**2
    assert np.all(image_kept[window])  # the image clip never cuts the window
    mask = np.asarray(
        adjoint.rotated_radius_mask(pixels, np.asarray(_relax_matrices(relion), np.float64), (box, box), r_max)
    )
    relax_kept = (window & image_kept)[None, :] & (mask > 0)
    k = np.stack([x, y, np.zeros_like(x)], axis=-1).astype(np.float64)
    rotated = np.einsum("nij,pj->npi", np.linalg.inv(relion), k)
    expected = window[None, :] & (np.sum(rotated * rotated, axis=-1) <= r_max * r_max)
    assert np.array_equal(relax_kept, expected)
    if scale == 1.0:
        # s = 1 exactly is the reference grid's clip.
        assert clip == adjoint.mstep_adjoint_max_r(2 * r_max, None, 2, anisotropic_magnification=True)


# Diamond's 1.36 A against a 1.4 A model (cell C, relax#60), the reference grid, and a coarser grid.
ISOTROPIC_SCALES = [1.36 / 1.4, 1.0, 1.12]


@pytest.mark.parametrize("scale", ISOTROPIC_SCALES, ids=["s0.9714", "s1", "s1.12"])
def test_isotropic_group_clip_keeps_relions_support(scale):
    """relax#60: a group on another grid without anisotropic magnification. recovar's kernel compares
    the image radius ``|k|`` and the rotated radius ``|A^-1 k| = |k| / s`` with one ``max_r``; the clip
    must keep exactly RELION's ``|k| / s <= r_max`` inside the group's window, for ``s`` on both sides of 1."""

    box, r_max = 64, 18
    _pixels, x, y = _fftw_half_pixels(box)
    window = _group_support(x, y, r_max, scale)
    radius2 = (x * x + y * y).astype(np.float64)

    def kernel_kept(clip_radius):
        return window & (radius2 <= clip_radius**2) & (radius2 / scale**2 <= clip_radius**2)

    clip = adjoint.mstep_adjoint_max_r(2 * r_max, r_max * scale, 2)
    expected = window & (radius2 / scale**2 <= r_max * r_max)
    assert np.array_equal(kernel_kept(clip.image_radius), expected)
    if scale < 1.0:
        # The image radius r_max * s (before relax#60) also cut the rotated radius there: the edge ring is lost.
        assert np.sum(expected & ~kernel_kept(r_max * scale)) > 0


@pytest.mark.gpu
@pytest.mark.requires_relion_bind
@pytest.mark.parametrize(
    ("mag", "scale"),
    [
        (None, 1.0),
        (None, 1.36 / 1.4),
        (None, 1.12),
        (SYMMETRIC, 1.0),
        (ASYMMETRIC, 1.0),
        (ASYMMETRIC, 1.0 + 2e-8),
        (ASYMMETRIC, 1.1),
    ],
    ids=["none", "none-s0.9714", "none-s1.12", "symmetric", "asymmetric", "asymmetric-s1+2e-8", "asymmetric-s1.1"],
)
def test_gpu_adjoint_weight_matches_relion_backprojector(mag, scale):
    """Total BPref weight (one per kept pixel under trilinear splatting) against RELION's BackProjector.

    ``scale`` puts the images on a grid ``s`` times the reference's field of view (relax#53, and relax#60
    without magnification): RELION's matrix is ``s A`` and the image weights reach the group's rounded
    support at ``r_max * s``."""

    import jax.numpy as jnp
    from relax.relion_bind._relion_bind_core import get_backprojector_data

    from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape

    box, current_size, padding = 64, 36, 2
    r_max = current_size // 2
    relion = scale * _relion_matrices(np.eye(2) if mag is None else mag, 40, seed=3)
    pixels, x, y = _fftw_half_pixels(box)
    support = _group_support(x, y, r_max, scale)  # RELION's nonzero weights
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
    clip = adjoint.mstep_adjoint_max_r(
        current_size, None if scale == 1.0 else r_max * scale, padding, anisotropic_magnification=mag is not None
    )
    volume_shape = relion_backprojector_volume_shape((box, box, box), padding, current_size=current_size)
    from relax.helpers.half_volume_mstep import half_volume_accumulator_shape

    def adjoint_weight(max_r):
        volume = jnp.zeros(int(np.prod(half_volume_accumulator_shape(volume_shape))), jnp.float32)
        return adjoint.adjoint_slice_volume_windowed(
            jnp.ones((len(relion), window.size), jnp.float32),
            jnp.asarray(window, jnp.int32),
            jnp.asarray(_relax_matrices(relion), jnp.float32),
            volume,
            (box, box),
            tuple(volume_shape),
            "linear_interp",
            True,
            True,
            max_r,
            True,
        )

    relax_weight = adjoint_weight(clip)
    # RELION's CPU BackProjector tests the rotated radius in double, relax's kernel (like RELION's GPU) in
    # float: pixels whose rotated radius is r_max to rounding may go either way.
    k = np.stack([x, y, np.zeros_like(x)], axis=-1)[support].astype(np.float64)
    rotated_r2 = np.sum(np.einsum("nij,pj->npi", np.linalg.inv(relion), k) ** 2, axis=-1)
    boundary = int(np.sum(np.abs(rotated_r2 - r_max**2) < 1e-4 * r_max**2))
    relax_total, relion_total = float(jnp.sum(relax_weight)), float(np.sum(relion_weight))
    assert abs(relax_total - relion_total) <= boundary + 1e-6 * relion_total
    if mag is None and scale < 1.0:
        # The image radius r_max * s (before relax#60) loses the edge ring RELION keeps.
        before = float(jnp.sum(adjoint_weight(adjoint.ReferenceSphereClip(r_max * scale, padding))))
        assert relion_total - before > 4 * max(boundary, 1)
    if mag is not None:
        # The image clip alone (the engine before anisotropic magnification was handled) is far off.
        image_clip = np.sum((rotated_r2 <= r_max**2) & (np.sum(k * k, axis=-1) <= r_max**2)[None, :])
        assert abs(float(image_clip) - relion_total) > 4 * max(boundary, 1)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("scale", "anisotropic"),
    [(1.0, True), (1.1, True), (1.36 / 1.4, False), (1.12, False)],
    ids=["s1", "s1.1", "isotropic-s0.9714", "isotropic-s1.12"],
)
def test_gpu_runtime_radius_reads_back_the_reference_radius(scale, anisotropic):
    """The stable-window adjoint (a capacity clip and a traced logical image radius) keeps the same
    pixels as the static clip of the logical size, on the reference grid and on other ones (relax#53,
    relax#60)."""

    import jax.numpy as jnp

    from relax.helpers.half_volume_mstep import half_volume_accumulator_shape, relion_backprojector_volume_shape

    box, current_size, capacity_size, padding = 64, 36, 48, 2
    r_max = current_size // 2
    relion = scale * _relion_matrices(ASYMMETRIC if anisotropic else np.eye(2), 30, seed=5)
    pixels, x, y = _fftw_half_pixels(box)
    window = pixels[_group_support(x, y, r_max, scale)]
    rows = jnp.ones((len(relion), window.size), jnp.float32)
    matrices = jnp.asarray(_relax_matrices(relion), jnp.float32)
    radius = None if scale == 1.0 else scale

    def weight(size, clip, runtime=None):
        shape = relion_backprojector_volume_shape((box, box, box), padding, current_size=size)
        volume = jnp.zeros(int(np.prod(half_volume_accumulator_shape(shape))), jnp.float32)
        return float(
            jnp.sum(
                adjoint.adjoint_slice_volume_windowed(
                    rows, jnp.asarray(window, jnp.int32), matrices, volume, (box, box), tuple(shape),
                    "linear_interp", True, True, clip, True, runtime_max_r=runtime,
                )
            )
        )

    def clip_at(size):
        return adjoint.mstep_adjoint_max_r(
            size, None if radius is None else (size // 2) * radius, padding, anisotropic_magnification=anisotropic
        )

    static = weight(current_size, clip_at(current_size))
    runtime = weight(capacity_size, clip_at(capacity_size), jnp.float32(clip_at(current_size).image_radius))
    assert runtime == pytest.approx(static, rel=1e-6)
