"""Model parameters and physical-particle assignments with explicit conventions."""

import json
from pathlib import Path

import mrcfile
import numpy as np
from recovar.core import fourier_transform_utils as ftu
from recovar.reconstruction.relion_functions import griddingCorrect_square

from relax.ppca_initial_class3d.expectation import mixture_tiles, oversampled_update, pose_grid


def export_models(output, state, dataset):
    """Save exact forward parameters and separately labelled corrected display maps.

    Model MRCs are the exact inverse transform of the fitted parameters.
    The stream interpolates theta directly with a separable Fourier tent
    kernel. Its inverse basis multiplies the coefficient map by
    ``prod_d sinc(x_d / N)^2`` (centered voxel coordinates, padding one).
    Dividing instead is the precompensation used BEFORE a forward FFT.

    The separately named gridding-corrected MRCs sample this interpolation
    basis. Finite-grid/support effects remain: these are display maps, not
    guaranteed exact physical densities or gold-standard halfmaps.
    """
    output = Path(output)
    np.savez(output / "models.npz", theta=np.asarray(state.theta), class_prior=state.class_prior,
             noise=np.asarray(state.noise), voxel_size=np.float32(dataset.voxel_size))
    half_shape = ftu.volume_shape_to_half_volume_shape(dataset.volume_shape)
    # Reuse RECOVAR's exact trilinear window; the returned predivided volume
    # is not the operation needed when exporting already fitted coefficients.
    _, interpolation_window = griddingCorrect_square(
        np.ones(dataset.volume_shape, np.float32), dataset.grid_size, 1, order=1
    )
    interpolation_window = interpolation_window.astype(np.float32)
    for k, model in enumerate(state.theta):
        for channel in range(model.shape[1]):
            real = ftu.get_idft3_real(model[:, channel].reshape(half_shape), dataset.volume_shape)
            corrected = real * interpolation_window
            name = "mean" if channel == 0 else f"loading{channel - 1:03d}"
            for filename, volume in ((f"class{k:03d}_model_{name}.mrc", real),
                                     (f"class{k:03d}_{name}_gridding_corrected.mrc", corrected)):
                with mrcfile.new(output / filename, overwrite=True) as mrc:
                    mrc.set_data(np.asarray(volume, np.float32))
                    mrc.voxel_size = dataset.voxel_size


def export_assignments(output, dataset, state, config, grid, *, diameter_ang=None, child_grid=None,
                       candidates=None):
    """Hard label = marginal MAP class; pose = MAP conditional on that class.

    Preserve the distinct joint class/pose MAP decision as well. Class labels
    and indices are zero-based. ET IDs identify physical particles, not tilts.
    ``grid`` is the final update's coarse grid; an oversampled final update
    reports its poses on its child grid (``child_grid``, else derived from the
    run's sampling), the grid pass 2 scored. ``candidates`` restricts each
    tile's poses as the final update's local searches did.
    """
    chunks = {key: [] for key in (
        "particle_ids", "class_probabilities", "class_conditional_z", "joint_map_class",
        "rotation_index_per_class", "translation_index_per_class", "joint_pmax",
    )}
    oversampled = oversampled_update(config, config.iterations)
    for _, _, tile in mixture_tiles(dataset, state, config, [np.arange(dataset.n_images)],
                                     config.iterations, grid, moments=False, diameter_ang=diameter_ang,
                                     candidates=candidates):
        fields = (tile.original_image_ids, tile.class_probabilities, np.asarray(tile.embeddings), tile.best_class,
                  tile.best_rotation_idx, tile.best_translation_idx, tile.max_posterior_per_image)
        for key, value in zip(chunks, fields, strict=True):
            chunks[key].append(value)
    arrays = {key: np.concatenate(value) for key, value in chunks.items()}
    order = np.argsort(arrays["particle_ids"])
    arrays = {key: value[order] for key, value in arrays.items()}
    expected = np.sort(dataset.original_image_indices_from_local(np.arange(dataset.n_images)))
    if len(np.unique(expected)) != dataset.n_images or not np.array_equal(arrays["particle_ids"], expected):
        raise ValueError("Final mixture assignments do not cover every original particle exactly once")
    labels = arrays["class_probabilities"].argmax(axis=1).astype(np.int32)
    if not np.all(np.isfinite(arrays["class_conditional_z"])):
        raise ValueError("Nonfinite final class-conditional embeddings")
    if not np.all(np.isfinite(arrays["class_probabilities"])) or not np.allclose(
            arrays["class_probabilities"].sum(axis=1), 1, rtol=1e-6, atol=1e-6):
        raise ValueError("Final class probabilities must be finite and normalized")
    rows = np.arange(len(labels))
    if oversampled:
        grid = (child_grid if child_grid is not None
                else pose_grid(dataset, config, config.iterations, children=True, sampling_state=state.sampling))
    ri, ti = arrays["rotation_index_per_class"], arrays["translation_index_per_class"]
    if np.any(ri < 0) or np.any(ri >= len(grid.rotations)) or np.any(ti < 0) or np.any(ti >= len(grid.translations)):
        raise ValueError("Invalid pose grid index in final mixture assignments")
    arrays.update(
        class_labels=labels, z=arrays["class_conditional_z"][rows, labels],
        eulers_deg=grid.eulers[ri[rows, labels]], rotations=grid.rotations[ri[rows, labels]],
        translations_px=grid.translations[ti[rows, labels]],
        euler_grid_deg=grid.eulers, translation_grid_px=grid.translations,
    )
    arrays["translations_angstrom"] = arrays["translations_px"] * np.float32(dataset.voxel_size)
    np.savez(Path(output) / "assignments.npz", **arrays)
    counts = np.bincount(labels, minlength=config.n_classes)
    (Path(output) / "classes.json").write_text(json.dumps({
        "class_index_base": 0, "hard_assignment": "marginal MAP class",
        "pose_assignment": "MAP pose conditional on marginal MAP class",
        "latent_coordinates": "conditional posterior mean in each class's independent latent basis",
        "low_probability_latents": "direct conditional-pose expectation, not division by joint class mass",
        "particle_count": len(labels), "hard_counts": counts.tolist(),
        "soft_counts": arrays["class_probabilities"].sum(axis=0, dtype=np.float64).tolist(),
        "class_prior": state.class_prior.tolist(), "translation_units": "pixels of the input images",
        "pose_grid": "order-1 children of the final update's coarse grid (adaptive oversampling)" if oversampled
        else "the final update's coarse grid",
        "coarse_healpix_order": int(grid.order), "coarse_shift_step_px": float(grid.translation_step),
        "local_searches": candidates is not None,
        "sampling": None if state.sampling is None else state.sampling.to_json(),
        "translation_convention": "native PPCA candidate grid; 2D SPA or 3D ET shifts projected as (Aproj*u)[:2]; not a validated RELION origin-column export",
        "map_format": "models.npz and class*_model_*.mrc preserve the fitted Fourier coefficients and their inverse FFT; *_gridding_corrected.mrc multiply that inverse FFT by the separable product_d sinc(x_d/N)^2, with centered voxel coordinates and padding one, to sample the trilinear Fourier interpolation basis. These are display maps, not guaranteed exact physical densities on a finite grid or gold-standard halfmaps",
    }, indent=2) + "\n")
