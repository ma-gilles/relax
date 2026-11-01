"""The pose adapter against noiseless images: an asymmetric off-centre body, written to a real stack and STAR file,
read back and carried through the preprocessing the fixed-pose reconstruction uses."""

from types import SimpleNamespace

import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile
from recovar.core import fourier_transform_utils as ftu
from recovar.core.geometry import translate_images
from recovar.core.slicing import slice_volume
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.utils.helpers import R_to_relion, relion_volume_to_recovar
from scipy.spatial.transform import Rotation

from relax.multibody import geometry
from relax.multibody.bodies import A_ROT90
from relax.multibody.pose_neighborhoods import body_pose_lines
from relax.multibody.subtraction import relion_translate_2d
from relax.relion.macros import relion_round_array
from scripts.multibody_adaptive_subsets import true_offsets

pytestmark = pytest.mark.unit

BOX, PIXEL, WIDTH_A = 64, 2.5, 7.0
PIVOT_PX = np.array([3.0, -2.0, 4.0])
# Blobs relative to the pivot, angstroms, and their masses: asymmetric and off-centre.
BLOBS = np.array([[18.0, 4.0, -6.0], [-9.0, 15.0, 3.0], [4.0, -12.0, 14.0], [26.0, -8.0, 5.0], [-3.0, 2.0, -16.0]])
MASSES = np.array([1.0, 0.7, 1.3, 0.5, 0.9])


def render(centres_px: np.ndarray) -> np.ndarray:
    """Gaussian blobs at ``[blobs, 2]`` image positions (x, y) relative to the box centre, one image."""
    y, x = np.mgrid[:BOX, :BOX] - BOX // 2
    width = WIDTH_A / PIXEL
    image = np.zeros((BOX, BOX))
    for (cx, cy), mass in zip(centres_px, MASSES):
        image += mass * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * width**2))
    return image


class Fixture:
    def __init__(self, tmp_path, n=16, seed=0):
        rng = np.random.default_rng(seed)
        self.A = Rotation.random(n, random_state=rng).as_matrix()
        self.Q = Rotation.from_rotvec(rng.normal(size=(n, 3)) * 0.2).as_matrix()
        self.B = self.A @ self.Q
        self.t = rng.normal(size=(n, 3)) * 6.0  # molecular frame, angstroms
        self.o = rng.uniform(-3.0, 3.0, size=(n, 2)) + 0.37  # fractional consensus origins, pixels
        self.orientation = Rotation.from_rotvec([0.4, -0.3, 0.8]).as_matrix()
        pivot_A = PIVOT_PX * PIXEL
        raw = []
        for A, Q, t, o in zip(self.A, self.Q, self.t, self.o):
            physical = pivot_A + BLOBS @ Q.T + t
            raw.append(render((physical @ A.T)[:, :2] / PIXEL - o))
        self.raw = np.asarray(raw, dtype=np.float32)
        # What a perfect fit returns: the pivot's raw position -r - f equals its true position (A (c0 + t))_xy - o.
        projected = (self.A @ PIVOT_PX)[:, :2]
        self.r = relion_round_array(self.o - projected)
        in_plane = np.einsum("nij,nj->ni", self.A, self.t)[:, :2] / PIXEL
        self.f = self.o - projected - self.r - in_plane
        stack = tmp_path / "particles.mrcs"
        with mrcfile.new(stack) as mrc:
            mrc.set_data(self.raw)
            mrc.set_image_stack()
            mrc.voxel_size = PIXEL
        eulers = np.asarray(R_to_relion(self.A, degrees=True))
        residual = A_ROT90.T @ self.orientation @ np.swapaxes(self.A, 1, 2) @ self.B @ self.orientation.T
        table = pd.DataFrame({
            "rlnImageName": [f"{i + 1:06d}@{stack}" for i in range(n)],
            "rlnAngleRot": eulers[:, 0], "rlnAngleTilt": eulers[:, 1], "rlnAnglePsi": eulers[:, 2],
            "rlnOriginXAngst": self.o[:, 0] * PIXEL, "rlnOriginYAngst": self.o[:, 1] * PIXEL,
            "rlnDefocusU": 10000.0, "rlnDefocusV": 10000.0, "rlnDefocusAngle": 0.0, "rlnPhaseShift": 0.0,
            "rlnCtfBfactor": 0.0, "rlnCtfScalefactor": 1.0, "rlnOpticsGroup": 1,
            "rlnRandomSubset": np.arange(n) % 2 + 1,
        })
        optics = pd.DataFrame({
            "rlnOpticsGroup": [1], "rlnOpticsGroupName": ["opticsGroup1"], "rlnAmplitudeContrast": [0.1],
            "rlnSphericalAberration": [2.7], "rlnVoltage": [300.0], "rlnImagePixelSize": [PIXEL], "rlnImageSize": [BOX],
            "rlnImageDimensionality": [2],
        })
        self.star = tmp_path / "particles.star"
        starfile.write({"optics": optics, "particles": table}, self.star, overwrite=True, float_format="%.12g")
        self.residual_eulers = np.asarray(R_to_relion(residual, degrees=True))

    def read(self):
        """Consensus matrices, composed matrices and origins as the STAR file gives them back."""
        table = starfile.read(self.star, always_dict=True)["particles"]
        consensus = geometry.consensus_matrices(table[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(np.float64))
        composed = geometry.body_matrices(consensus, SimpleNamespace(orientation=self.orientation), self.residual_eulers)
        return consensus, composed, table[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(np.float64) / PIXEL


@pytest.fixture
def fixture(tmp_path):
    return Fixture(tmp_path)


def test_fixture_is_not_trivial(fixture):
    assert np.linalg.norm(((BLOBS * MASSES[:, None]).sum(0) / MASSES.sum())) > 3.0
    projected = (fixture.A @ PIVOT_PX)[:, :2]
    assert np.min(np.abs(projected - np.rint(projected))) >= 0.0 and np.max(np.abs(projected - np.rint(projected))) > 0.2
    assert np.max(np.abs(fixture.f)) > 0.3


def test_lines_contain_the_true_translation_and_rotation(fixture):
    consensus, composed, origins = fixture.read()
    np.testing.assert_allclose(consensus, fixture.A, atol=1e-9)
    np.testing.assert_allclose(composed, fixture.B, atol=1e-9)
    rotations, lifts, normals = body_pose_lines(consensus, composed, PIVOT_PX, origins, fixture.f, PIXEL)
    np.testing.assert_allclose(rotations, fixture.Q, atol=1e-9)
    # The line through u along n holds t, and u has no depth.
    depths = np.sum(fixture.t * normals, axis=1, keepdims=True)
    np.testing.assert_allclose(lifts + depths * normals, fixture.t, atol=1e-8)
    np.testing.assert_allclose(np.sum(lifts * normals, axis=1), 0.0, atol=1e-8)


def corrected_images(fixture, offsets):
    """The raw images read from the stack, through the integer shift, RECOVAR's image transform and its translation
    by ``offsets``: the reconstruction's input."""
    dataset = load_dataset(str(fixture.star), datadir=str(fixture.star.parent), lazy=False, dtype=np.complex64, absent_angles_zero=True)
    raw = np.asarray(dataset.image_source.backend.source.get(np.arange(len(fixture.raw))))
    np.testing.assert_array_equal(raw, fixture.raw)
    shifted = np.stack([relion_translate_2d(image, r.astype(int)) for image, r in zip(raw, fixture.r)])
    half = jnp.asarray(np.asarray(dataset.process_images_half(shifted)))
    return translate_images(half, jnp.asarray(offsets, dtype=jnp.float32), (BOX, BOX), half_image=True), dataset


def expected_images(fixture, dataset):
    centred = np.stack([render((BLOBS @ B.T)[:, :2] / PIXEL) for B in fixture.B]).astype(np.float32)
    return jnp.asarray(np.asarray(dataset.process_images_half(centred)))


def relative_error(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def test_corrected_image_is_the_centred_body_projection(fixture):
    images, dataset = corrected_images(fixture, fixture.f)
    expected = expected_images(fixture, dataset)
    # Blobs near the zero-filled edge of the integer shift are absent here: the body stays inside the box.
    assert relative_error(images, expected) < 1e-3


def centred_volume_ft():
    """The body as a RELION-frame map centred on its pivot, on the grid, in RECOVAR's Fourier frame."""
    k, i, j = np.mgrid[:BOX, :BOX, :BOX] - BOX // 2
    width = WIDTH_A / PIXEL
    relion = np.zeros((BOX, BOX, BOX), dtype=np.float32)
    for centre, mass in zip(BLOBS / PIXEL, MASSES):
        relion += mass * np.exp(-((j - centre[0]) ** 2 + (i - centre[1]) ** 2 + (k - centre[2]) ** 2) / (2.0 * width**2)) / (
            np.sqrt(2.0 * np.pi) * width
        )
    return ftu.get_dft3(jnp.asarray(relion_volume_to_recovar(relion))).reshape(-1)


def slice_error(images, volume, rotations):
    slices = slice_volume(volume, jnp.asarray(rotations, dtype=jnp.float32), (BOX, BOX), (BOX, BOX, BOX), "linear_interp", half_image=True)
    a, b = np.asarray(images).ravel(), np.asarray(slices).ravel()
    scale = np.vdot(b, a) / np.vdot(b, b)
    return float(np.linalg.norm(a - scale * b) / np.linalg.norm(a)), complex(scale)


def test_recovar_slice_at_the_composed_pose_is_the_corrected_image(fixture):
    images, _ = corrected_images(fixture, fixture.f)
    volume = centred_volume_ft()
    # Measured 0.034: linear interpolation on an unpadded grid, which also attenuates the slice by 5%. The sign is
    # RECOVAR's density sign (relion_volume_to_recovar).
    error, scale = slice_error(images, volume, fixture.B)
    assert error < 0.04
    assert -1.1 < scale.real < -0.95 and abs(scale.imag) < 0.01
    # The composed pose is the best one: any small rotation of it, or a moved image, fits worse.
    for rotvec in np.eye(3) * np.deg2rad(1.5):
        assert slice_error(images, volume, fixture.B @ Rotation.from_rotvec(rotvec).as_matrix())[0] > 1.1 * error
    for direction in np.array([[0.25, 0.0], [0.0, 0.25]]):
        assert slice_error(corrected_images(fixture, fixture.f + direction)[0], volume, fixture.B)[0] > 1.1 * error


def test_negative_controls_miss_the_image(fixture):
    wrong = fixture.f - (fixture.o - (fixture.A @ PIVOT_PX)[:, :2] - fixture.r)  # the fractional centring omitted
    images, dataset = corrected_images(fixture, wrong)
    expected = expected_images(fixture, dataset)
    assert relative_error(images, expected) > 0.05
    # Lifting with the body camera B instead of the consensus A misses the line.
    _, lifts, normals = body_pose_lines(fixture.B, fixture.B, PIVOT_PX, fixture.o, fixture.f, PIXEL)
    depths = np.sum(fixture.t * normals, axis=1, keepdims=True)
    assert np.max(np.linalg.norm(lifts + depths * normals - fixture.t, axis=1)) > 0.05


def test_true_offsets_place_the_nuisance_and_align_the_target():
    projected = np.array([[3.3, -1.2], [0.5, 7.7]])
    origins = np.array([[0.4, 0.3], [-1.9, 2.1]])
    position = np.array([[2.5, -1.1], [3.25, 0.5]])
    nuisance, aligned = true_offsets(projected, origins, position)
    np.testing.assert_allclose(projected - origins - nuisance, position)
    np.testing.assert_allclose(-relion_round_array(origins - projected) - aligned, position)
    assert np.max(np.abs(nuisance - aligned)) > 0.1
