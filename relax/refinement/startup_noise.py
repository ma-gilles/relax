"""Startup particle ordering, dataset adaptation and scoring-noise preparation.

RELION's host estimator and FFT/mask formulas remain in ``relion.initial_noise``.
The command selects fresh, loaded or resumed noise before invoking this owner.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from recovar import utils

from relax.helpers.shells import shell_of_radius_sq
from relax.refinement.optics_shapes import MultiShapeDataset
from relax.refinement.tomo_half import TomoDataset
from relax.relion import relion_metadata
from relax.relion.initial_noise import compute_avg_unaligned_and_sigma2, radial_power_spectrum


class StartupNoise(NamedTuple):
    """RECOVAR-frame radial noise and the corresponding scoring pixel variance."""

    radial: np.ndarray
    pixel_variance: np.ndarray


def continued_noise(noise_shells, image_shape) -> StartupNoise:
    """A continued run's noise: half 1's pixel rows (the loop installs each half's own spectrum) and the halves'
    mean radial curve."""
    from relax.refinement.noise_updates import noise_pixel_rows

    return StartupNoise(
        radial=np.mean(np.stack(noise_shells, axis=0), axis=0),
        pixel_variance=np.asarray(noise_pixel_rows(noise_shells[0], image_shape)),
    )


def auto_refine_noise_order(our_particles, relion_particles):
    """Map RELION's pre-randomisation ``sorted_idx`` to RECOVAR rows.

    ``divideParticlesInRandomHalves`` stable-sorts the source insertion order
    by random subset before follower rank 1 estimates the startup noise.  For
    datasets with fewer than 1,000 particles in half 1, that estimate therefore
    continues into half 2; it is not a half-1-only calculation.
    """

    our_row_by_identity = relion_metadata.particle_identity_rows(
        our_particles,
        label="RECOVAR input STAR",
    )
    relion_row_by_identity = relion_metadata.particle_identity_rows(
        relion_particles,
        label="RELION data STAR",
    )
    if set(our_row_by_identity) != set(relion_row_by_identity):
        raise ValueError(
            "RELION and RECOVAR STAR files do not contain the same "
            "rlnImageName/stack identities"
        )

    relion_identities = list(relion_row_by_identity)
    relion_subsets = np.asarray(relion_particles["rlnRandomSubset"], dtype=np.int64)
    if not np.all(np.isin(relion_subsets, (1, 2))):
        raise ValueError("RELION random subsets must contain only 1 and 2")
    relion_sorted_rows = np.concatenate(
        [
            np.flatnonzero(relion_subsets == 1),
            np.flatnonzero(relion_subsets == 2),
        ]
    ).astype(np.int64, copy=False)
    source_rows = np.asarray(
        [our_row_by_identity[relion_identities[row]] for row in relion_sorted_rows],
        dtype=np.int64,
    )
    if "rlnOpticsGroup" in relion_particles.columns:
        relion_optics = np.asarray(relion_particles["rlnOpticsGroup"], dtype=np.int64)
        optics_group_ids = relion_optics[relion_sorted_rows]
    else:
        optics_group_ids = np.zeros(source_rows.size, dtype=np.int64)
    return source_rows, optics_group_ids


def class3d_noise_order(our_particles):
    """Return RELION Class3D's startup-noise source rows and optics labels.

    Class3D does not split random halves, so ``sorted_idx`` is the identity
    over the micrograph-sorted particles (exp_model.cpp:900-901) when
    ``calculateSumOfPowerSpectraAndAverageImage`` takes the first particles of
    each optics group (ml_optimiser.cpp:3068-3072).
    """
    from relax.relion.input_particle_table import relion_particle_order

    source_rows = relion_particle_order(our_particles)
    if "rlnOpticsGroup" in our_particles.columns:
        optics_group_ids = np.asarray(our_particles["rlnOpticsGroup"], dtype=np.int64)[source_rows]
    else:
        optics_group_ids = np.ones(source_rows.size, dtype=np.int64)
    return source_rows, optics_group_ids


def whole_transform_power_spectrum(image_real: np.ndarray, n_shells: int) -> np.ndarray:
    """Per-shell mean ``|FFT(image)|²`` over the whole transform: every Hermitian pair counted once.

    RELION's start-up spectrum (``relax.relion.initial_noise.radial_power_spectrum``) averages every
    stored pixel of the FFTW half, which counts the pairs whose two members are both stored (the
    ``kx = 0`` column and an even image's Nyquist column) twice. Same scale: ``1 / (H W)``.
    """
    height, width = image_real.shape[-2:]
    power = np.abs(np.fft.fft2(image_real) / (height * width)) ** 2
    ky = np.fft.fftfreq(height, d=1.0) * height
    kx = np.fft.fftfreq(width, d=1.0) * width
    shell = shell_of_radius_sq(ky[:, None] ** 2 + kx[None, :] ** 2, rule="half_even")
    keep = shell < n_shells
    total = np.bincount(shell[keep], weights=power[keep], minlength=n_shells)
    return total / np.maximum(np.bincount(shell[keep], minlength=n_shells), 1)


_POWER_SPECTRUM = {"relion": radial_power_spectrum, "once": whole_transform_power_spectrum}


def estimate_startup_sigma2(
    dataset,
    *,
    source_rows,
    optics_group_ids,
    image_pixel_size: float,
    particle_diameter_ang: float,
    width_mask_edge_px: int,
    group_pixel_sizes=None,
    pair_counting: str = "relion",
) -> np.ndarray:
    """Reproduce RELION's process-resident fresh AutoRefine noise spectrum.

    MPI follower rank 1 computes the initial spectrum from at most 1,000
    particles per optics group in the stable subset-1-then-subset-2 source
    order, then broadcasts it to the other follower.  RELION writes a rounded
    copy to model STAR, but its first expectation step consumes these unrounded
    values. A ``MultiShapeDataset`` needs ``group_pixel_sizes`` (one per sorted
    optics label): each image is masked with its group's pixel size and brought
    onto the model grid (``image_pixel_size``, the reference box) first.
    ``pair_counting`` is the spectra's Hermitian-pair counting: ``"relion"`` averages
    every stored pixel of the FFTW half, ``"once"`` the whole transform
    (:func:`whole_transform_power_spectrum`).
    """

    source_rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    optics_labels = np.asarray(optics_group_ids, dtype=np.int64).reshape(-1)
    if source_rows.shape != optics_labels.shape:
        raise ValueError(
            "source rows and optics-group labels must have identical shapes",
        )
    if source_rows.size == 0:
        raise ValueError("fresh K=1 live-noise bootstrap requires at least one particle")
    if np.unique(source_rows).size != source_rows.size:
        raise ValueError("fresh K=1 live-noise source rows contain duplicates")
    if int(np.min(source_rows)) < 0 or int(np.max(source_rows)) >= int(dataset.n_units):
        raise ValueError("fresh K=1 live-noise source rows are out of dataset bounds")
    if not np.isfinite(image_pixel_size) or float(image_pixel_size) <= 0.0:
        raise ValueError("RELION image pixel size must be positive and finite")

    unique_optics = sorted(np.unique(optics_labels).tolist())
    optics_to_dense = {int(label): index for index, label in enumerate(unique_optics)}
    optics_by_source_row = {
        int(source_row): optics_to_dense[int(optics_label)]
        for source_row, optics_label in zip(source_rows, optics_labels, strict=True)
    }

    tomo = isinstance(dataset, TomoDataset)

    def refuse_non_finite(row, image):
        # The estimate would end in a non-positive spectrum that names no image (relax#16).
        if not np.all(np.isfinite(image)):
            raise RuntimeError(f"start-up noise estimate: dataset image {int(row)} has a non-finite pixel")

    def image_iter():
        if tomo:
            # RELION's per-image count against minimum_nr_particles_sigma2_noise (10 for subtomograms,
            # ml_optimiser.cpp:2574, :3058): each group's first particle fills it (TomoDataset.startup_noise_images).
            dense_groups = [optics_by_source_row[int(row)] for row in source_rows]
            yield from dataset.startup_noise_images(source_rows, unit_groups=dense_groups, minimum_nr_particles=10)
            return
        if isinstance(dataset, MultiShapeDataset):
            for row, image in dataset.iter_images(source_rows, batch_size=min(256, source_rows.size)):
                refuse_non_finite(row, image)
                yield optics_by_source_row[row], image
            return
        for batch_images, _particle_indices, local_indices in dataset.image_source.iter_batches(
            batch_size=min(256, source_rows.size),
            batch_mode="images",
            subset_indices=source_rows,
        ):
            batch_images = np.asarray(batch_images)
            local_indices = np.asarray(local_indices, dtype=np.int64).reshape(-1)
            if batch_images.shape[0] != local_indices.size:
                raise ValueError("image batch and source-row batch lengths differ")
            for image, source_row in zip(batch_images, local_indices, strict=True):
                row = int(source_row)
                if row not in optics_by_source_row:
                    raise ValueError(f"image source returned an unexpected source row: {row}")
                refuse_non_finite(row, image)
                yield optics_by_source_row[row], image

    _average_image, sigma2_per_group = compute_avg_unaligned_and_sigma2(
        image_iter(),
        box_size=int(dataset.grid_size),
        # RELION masks each source image using its optics-group pixel size.
        # The model/MRC pixel size can differ in the last serialized digits;
        # at this float64 startup-noise boundary that is enough to change the
        # later float32 inverse-noise factor by one ULP.
        pixel_size=float(image_pixel_size),
        particle_diameter_ang=float(particle_diameter_ang),
        width_mask_edge_px=int(width_mask_edge_px),
        do_zero_mask=True,
        nr_optics_groups=len(unique_optics),
        # A subtomogram iterator is already capped per particle; every one of its images counts.
        # Otherwise RELION's minimum_nr_particles_sigma2_noise for single particles, 1,000.
        minimum_nr_particles=int(dataset.unit_image_offsets[-1]) if tomo else 1000,
        power_spectrum=_POWER_SPECTRUM[pair_counting],
        **(
            {}
            if group_pixel_sizes is None
            else {"group_pixel_sizes": group_pixel_sizes, "model_pixel_size": float(image_pixel_size)}
        ),
    )
    sigma2_per_group = np.asarray(sigma2_per_group, dtype=np.float64)
    if not np.all(np.isfinite(sigma2_per_group)) or not np.all(sigma2_per_group > 0.0):
        raise ValueError("fresh K=1 live-noise spectrum must be positive and finite")
    return sigma2_per_group


def prepare_startup_noise(
    dataset, *, source_rows, optics_group_ids, mask_params,
    optics_pixel_sizes, output_dtype, pair_counting,
):
    """Estimate fresh noise on the model grid and expand it for scoring.

    Radial noise stays float64 in RECOVAR units; pixel variance uses output_dtype.
    Rows follow sorted optics labels; a single group returns flat arrays.
    ``pair_counting`` is the image spectra's Hermitian-pair counting.
    See docs/math/relion_refinement_algorithm.md#start-up-noise.
    """
    if (
        source_rows is None or optics_group_ids is None or mask_params is None
        or optics_pixel_sizes is None
    ):
        raise ValueError(
            "RELION start-up noise needs K=1 half sets (Class3D uses the input order), the "
            "particle-diameter mask and the optics pixel size, and no frozen or loaded noise"
        )
    multi_shape = isinstance(dataset, MultiShapeDataset)
    if not multi_shape and np.unique(np.asarray(optics_pixel_sizes)).size != 1:
        raise ValueError("RELION start-up noise with several pixel sizes needs a multi-shape dataset")
    n_optics_groups = int(np.unique(optics_group_ids).size)
    group_pixel_sizes = None
    if multi_shape:
        # Optics labels are 1-based rows of the optics table; spectra follow the sorted labels.
        labels = np.unique(np.asarray(optics_group_ids, dtype=np.int64))
        group_pixel_sizes = np.asarray(optics_pixel_sizes, dtype=np.float64).reshape(-1)[labels - 1]
    sigma2 = estimate_startup_sigma2(
        dataset, source_rows=source_rows, optics_group_ids=optics_group_ids,
        image_pixel_size=(
            float(dataset.voxel_size) if multi_shape else float(np.asarray(optics_pixel_sizes).reshape(-1)[0])
        ),
        particle_diameter_ang=float(mask_params[0]), width_mask_edge_px=int(mask_params[1]),
        group_pixel_sizes=group_pixel_sizes, pair_counting=pair_counting,
    )
    # One spectrum per optics group (MlModel::sigma2_noise[optics_group]); one group
    # keeps the flat layout of the single-optics path.
    radial = sigma2 * float(dataset.grid_size) ** 4
    noise = np.stack([
        scoring_noise_from_sigma2(
            sigma2_group, box_size=int(dataset.grid_size), output_dtype=output_dtype,
        )
        for sigma2_group in sigma2
    ])
    if n_optics_groups == 1:
        return StartupNoise(radial=radial[0], pixel_variance=noise[0])
    return StartupNoise(radial=radial, pixel_variance=noise)


def scoring_noise_from_sigma2(
    sigma2,
    *,
    box_size: int,
    output_dtype=np.float32,
) -> np.ndarray:
    """Expand a RELION-unit radial sigma2 spectrum into RECOVAR FFT units."""

    radial_native = np.asarray(sigma2, dtype=np.float64) * float(box_size) ** 4
    return np.asarray(
        utils.make_radial_image(
            jnp.asarray(radial_native),
            (int(box_size), int(box_size)),
            extend_last_frequency=True,
        ),
        dtype=output_dtype,
    ).reshape(-1)
