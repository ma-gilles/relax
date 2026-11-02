"""A body's images with the other bodies subtracted: the boundary that prepares one body's alignment and
reconstruction inputs.

RELION builds them in ``getFourierTransformsAndCtfs`` (acc_ml_optimiser_impl.h:330-998 at f2c1a38): each image is
norm-corrected and translated by the integer ``r_b`` (zero fill), the CTF-modulated projections of the other bodies
at their current poses are subtracted from both the masked (alignment) and, with
``--reconstruct_subtracted_bodies``, the unmasked (reconstruction) image, and both are shifted by the body's
fractional refined offset ``f_b``. Its soft image mask is linear in the image with the background average of the
same raised-cosine weights for the image (``cosineFilter`` after ``softMaskBackgroundValue``, ``--zero_mask``) and
for the subtracted projections (``softMaskOutsideMap``), so masking the image and the prediction separately and
subtracting equals masking their difference. Relax therefore subtracts in real space, before its RELION image
preprocessing:

``D_ib = translate(x_i, r_ib) - S_ib / c_i``

where ``S_ib`` is the real image whose preprocessed Fourier transform is the summed prediction (band-limited to the
iteration's current size, as RELION's ``Fsum_obody``) and ``c_i = avg_norm / norm_i`` the norm factor the
preprocessing multiplies by. The body's dataset scores ``D_ib`` with no further integer shift, its body mask radius
and the per-image Fourier translation ``f_ib`` (``LocalSearchData.image_translations``).

A future heterogeneous predictor of the other bodies would replace ``S_ib`` here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import CryoEMDataset
from recovar.data_io.image_loader import ImageLoader
from recovar.data_io.image_sources import BackendImageSource

from relax.fourier.preprocessing import half_translation_phase_table
from relax.io.batch_fetch import original_image_indices
from relax.multibody import geometry
from relax.multibody.bodies import Body
from relax.projection.projection import project_relion_projector_half_spectrum_centered_rows
from relax.relion.ctf import relion_exact_ctf_half_from_source_star
from relax.steps.projector_preparation import PreparedProjector


class ArrayImageLoader(ImageLoader):
    """Real particle images held in memory, ``[N, D, D]`` float32, in the dataset's original order."""

    def __init__(self, images: np.ndarray):
        images = np.asarray(images, dtype=np.float32)
        super().__init__(int(images.shape[0]), int(images.shape[1]), dtype=np.float32)
        self._images = images

    def _load(self, indices: np.ndarray) -> np.ndarray:
        return self._images[indices]

    def stack_files(self) -> list:
        return []


def dataset_with_images(dataset: CryoEMDataset, images: np.ndarray, *, mask_diameter_angstrom: float) -> CryoEMDataset:
    """``dataset`` (a whole, unsubset dataset) reading ``images`` instead of its files, with RELION's soft image mask
    of ``mask_diameter_angstrom``. Metadata, CTF and image backend settings are shared."""

    root = dataset.image_source
    if not isinstance(root, BackendImageSource):
        raise TypeError("a body dataset is built from the whole particle dataset, not a subset view")
    if images.shape != (dataset.n_images, *dataset.image_shape):
        raise ValueError(f"body images {images.shape} do not match the dataset {dataset.n_images} x {dataset.image_shape}")
    backend = copy.copy(root.backend)
    backend.source = ArrayImageLoader(images)
    pixel_size, _, width_mask_edge_px = root.backend._relion_image_mask_params
    backend.set_relion_image_mask(
        pixel_size=pixel_size, particle_diameter_ang=float(mask_diameter_angstrom), width_mask_edge_px=width_mask_edge_px
    )
    body = CryoEMDataset(
        image_source=BackendImageSource(backend, info=root.info),
        voxel_size=dataset.voxel_size,
        metadata=dataset._metadata,
        ctf_evaluator=dataset.ctf_evaluator,
        grid_size=dataset.grid_size,
        tilt_series_flag=dataset.tilt_series_flag,
        premultiplied_ctf=dataset.premultiplied_ctf,
        dataset_indices=dataset.dataset_indices,
        dtype=dataset.dtype,
    )
    body.noise = dataset.noise
    body.particles_file, body.poses_file, body.ctf_file, body.datadir = (
        dataset.particles_file, dataset.poses_file, dataset.ctf_file, dataset.datadir
    )
    return body


def translate_zero_fill(volume: np.ndarray, shift) -> np.ndarray:
    """``volume`` moved by the integer ``shift`` per axis, zero where nothing moves in (RELION ``DONT_WRAP``)."""

    out = np.zeros_like(volume)
    src, dst = [], []
    for n, s in zip(volume.shape, (int(v) for v in shift)):
        src.append(slice(max(0, -s), n - max(0, s)))
        dst.append(slice(max(0, s), n - max(0, -s)))
    out[tuple(dst)] = volume[tuple(src)]
    return out


def relion_to_recovar_axes(vector_xyz) -> np.ndarray:
    """A RELION-frame shift ``(x, y, z)`` as a shift of recovar volume axes 0, 1, 2 (``v_rec[a, b, c] =
    -v_rel[c, b, a]``: recovar axis 0 is RELION's x)."""

    return np.asarray(vector_xyz, dtype=np.float64)


def recovar_mask(body: Body) -> np.ndarray:
    """The body's mask in the recovar volume frame."""

    return np.transpose(body.mask, (2, 1, 0))


def body_projector_reference(map_ft, body: Body, volume_shape, mask=None) -> np.ndarray:
    """The real map RELION builds a body's projector from: the stored map times the body mask (or ``mask``, a
    RELION-frame mask of an overlapping body's extra projector), moved so the body's centre of mass is at the origin
    (``setFourierTransformMaps``, ml_model.cpp:1525-1534), in the recovar frame."""

    real = np.asarray(ftu.get_idft3(jnp.asarray(map_ft).reshape(volume_shape)).real, dtype=np.float64)
    weight = recovar_mask(body) if mask is None else np.transpose(mask, (2, 1, 0))
    return translate_zero_fill(real * weight, -relion_to_recovar_axes(body.com))


def stored_body_map(real_body_frame: np.ndarray, body: Body) -> np.ndarray:
    """A reconstruction in the body frame moved back to the stored frame (``selfTranslate(Iref, com, DONT_WRAP)``,
    ml_optimiser_mpi.cpp:2490-2493)."""

    return translate_zero_fill(real_body_frame, relion_to_recovar_axes(body.com))


@dataclass(frozen=True)
class BodyPoses:
    """One body's residual poses for a set of particles: RELION Euler angles (degrees) and offsets (pixels)."""

    eulers_deg: np.ndarray  # [N, 3]
    offsets_px: np.ndarray  # [N, 2]


@dataclass(frozen=True)
class ParticleFrames:
    """The consensus poses the bodies move from: RELION matrices and offsets (pixels), and each image's norm factor
    ``avg_norm / norm`` (the factor RELION's ``TranslateAndNormCorrect`` multiplies by)."""

    consensus: np.ndarray  # [N, 3, 3]
    offsets_px: np.ndarray  # [N, 2]
    norm_factors: np.ndarray  # [N]


def relion_half_to_real(half_spectrum, image_shape) -> jax.Array:
    """The real images ``[N, D, D]`` whose RELION-normalized Fourier transform is ``half_spectrum`` ``[N, P]``.

    The spectrum is in recovar's centred packed layout (``fftshift_y(rfft2(fftshift(image)))``) and RELION's units,
    the forward transform divided by the pixel count (relax's projections; its preprocessed images reach these
    units through ``to_relion_native_fine_units``)."""

    h, w = (int(v) for v in image_shape)
    half = jnp.reshape(half_spectrum, (-1, h, w // 2 + 1))
    real = jnp.fft.irfft2(jnp.fft.ifftshift(half, axes=(-2,)), s=(h, w), axes=(-2, -1))
    return jnp.fft.ifftshift(real, axes=(-2, -1)) * (h * w)


@jax.jit
def _shifted_ctf_prediction(projection, ctf, phases):
    return projection * ctf * phases


def other_body_prediction(
    dataset,
    image_indices,
    target: int,
    bodies: list[Body],
    projectors: list[PreparedProjector],
    poses: list[BodyPoses],
    frames: ParticleFrames,
    *,
    current_size: int,
    padding_factor: int,
) -> jax.Array:
    """The real images ``S`` ``[n, D, D]`` RELION subtracts from body ``target``'s images: the other bodies projected at
    their composed poses, moved to their place in the image translated by ``r_target`` and CTF-modulated.

    ``poses``, ``frames`` and ``projectors`` are per body (the projectors of this half's maps, the bodies' masked
    centred maps at ``current_size``); ``poses[b]`` and ``frames`` index the same particles as ``image_indices``.
    """

    image_shape = tuple(int(v) for v in dataset.image_shape)
    integer = geometry.integer_image_shifts(
        frames.offsets_px, geometry.projected_centers(frames.consensus, bodies[target])
    )
    # RELION multiplies by its own CTF, -sin(gamma) (ctf.h getCTF); relax's exact rows carry recovar's sign, the
    # negative of it, which relax's scoring pairs with its own convention.
    ctf = -relion_exact_ctf_half_from_source_star(dataset, image_indices, image_shape)
    total = None
    for other, body in enumerate(bodies):
        if other == target:
            continue
        rows = geometry.body_matrices(frames.consensus, body, poses[other].eulers_deg)
        slab = jnp.asarray(projectors[other].data)
        projection = project_relion_projector_half_spectrum_centered_rows(
            slab[0] if slab.ndim == 4 else slab,  # one class: [1, pad, pad, pad // 2 + 1]
            jnp.asarray(rows),
            image_shape,
            int(projectors[other].r_max),
            int(padding_factor),
            None,
        )
        shifts = geometry.other_body_shifts(
            geometry.projected_centers(frames.consensus, body), frames.offsets_px, poses[other].offsets_px, integer
        )
        term = _shifted_ctf_prediction(
            projection, ctf.astype(projection.real.dtype), half_translation_phase_table(shifts, image_shape, dtype=jnp.float64)
        )
        total = term if total is None else total + term
    if current_size > image_shape[0]:
        raise ValueError(f"current size {current_size} exceeds the box {image_shape[0]}")
    return relion_half_to_real(total, image_shape)


def relion_soft_mask_outside(images: np.ndarray, radius: float, width: float) -> np.ndarray:
    """RELION ``softMaskOutsideMap`` on images ``[N, D, D]`` in double: a raised cosine from ``radius`` to ``radius +
    width``, and outside it the average of the image under the complementary weights (mask.cpp)."""

    size = images.shape[-1]
    axis = np.arange(size, dtype=np.float64) - size // 2
    r = np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)
    outside = np.where(r < radius, 0.0, np.where(r > radius + width, 1.0, 0.5 + 0.5 * np.cos(np.pi * (radius + width - r) / width)))
    background = (images * outside).sum(axis=(-2, -1), keepdims=True) / outside.sum()
    return images * (1.0 - outside) + background * outside


def native_half_spectrum(real_images) -> jax.Array:
    """:func:`relion_half_to_real`'s inverse in the units of relax's preprocessed images, the plain forward transform,
    in double precision."""

    centred = jnp.fft.fftshift(jnp.asarray(real_images, dtype=jnp.float64), axes=(-2, -1))
    half = jnp.fft.fftshift(jnp.fft.rfft2(centred, axes=(-2, -1)), axes=(-2,))
    return half.reshape(half.shape[0], -1)


@dataclass(frozen=True)
class BodyImagePrediction:
    """The other bodies' prediction of a body's images, subtracted in RELION's order (what a body's ``image_translations`` carries):
    from the preprocessed image in Fourier space in double, the masked prediction from the masked (scored) image and
    the plain one from the unmasked (reconstructed) image (acc_ml_optimiser_impl.h:1388-1510)."""

    translations: np.ndarray  # [n, 2] the refined offsets the images still take (pixels), by dataset image
    prediction: object  # image_indices -> real images S [n, D, D], float64
    mask_radius_px: float
    mask_width_px: float
    current_size: int  # the iteration's image size: RELION's Xi2 starts there in pass 1 as well

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.translations, dtype=dtype)

    def __call__(self, image_indices) -> tuple:
        real = np.asarray(self.prediction(image_indices), dtype=np.float64)
        masked = relion_soft_mask_outside(real, self.mask_radius_px, self.mask_width_px)
        return native_half_spectrum(masked), native_half_spectrum(real)


def translated_body_images(dataset, image_indices, target: int, bodies: list[Body], frames: ParticleFrames) -> np.ndarray:
    """The images ``image_indices`` of ``dataset`` translated by the integer ``r_target`` (float32 as read)."""

    raw = np.asarray(dataset.image_source.backend.source.get(original_image_indices(dataset, image_indices)))
    integer = geometry.integer_image_shifts(
        frames.offsets_px, geometry.projected_centers(frames.consensus, bodies[target])
    ).astype(np.int64)
    return np.stack([relion_translate_2d(image, shift) for image, shift in zip(raw, integer)])


def subtracted_body_images(
    dataset,
    image_indices,
    target: int,
    bodies: list[Body],
    projectors: list[PreparedProjector],
    poses: list[BodyPoses],
    frames: ParticleFrames,
    *,
    current_size: int,
    padding_factor: int,
) -> np.ndarray:
    """``D = translate(x, r_target) - S / c`` for the images ``image_indices`` of ``dataset``: body ``target``'s
    images ready for RELION preprocessing without a further integer shift (see the module docstring)."""

    raw = np.asarray(dataset.image_source.backend.source.get(original_image_indices(dataset, image_indices)))
    integer = geometry.integer_image_shifts(
        frames.offsets_px, geometry.projected_centers(frames.consensus, bodies[target])
    ).astype(np.int64)
    translated = np.stack([relion_translate_2d(image, shift) for image, shift in zip(raw, integer)])
    prediction = np.asarray(other_body_prediction(
        dataset, image_indices, target, bodies, projectors, poses, frames,
        current_size=current_size, padding_factor=padding_factor,
    ))
    return (translated - prediction / frames.norm_factors[:, None, None]).astype(np.float32)


def relion_translate_2d(image: np.ndarray, shift_xy) -> np.ndarray:
    """RELION's ``translate2D`` by an integer ``(dx, dy)``: pixel ``(y, x)`` moves to ``(y + dy, x + dx)``, zero fill."""

    dx, dy = (int(v) for v in shift_xy)
    return translate_zero_fill(image, (dy, dx))
