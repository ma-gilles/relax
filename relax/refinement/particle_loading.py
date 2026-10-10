"""Particle loading, input-mode admission and image preprocessing setup.

RECOVAR owns image reading and Fourier preprocessing. This command boundary
selects the dataset adapter; runtime setup configures its half image backends.
See ``docs/math/relion_refinement_algorithm.md#particle-input-preparation``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from relax.helpers.env_flags import parse_env_true_flag
from relax.helpers.particle_io import ParticleReadPolicy, assert_reads_from_scratch, image_star, prepare_particle_reads
from relax.refinement import command_options
from relax.refinement.optics_shapes import MultiShapeDataset, optics_shape_class_rows
from relax.refinement.tomo_half import TomoDataset, is_relion5_2d_stack_star, load_tomo_dataset
from relax.relion import relion_metadata
from relax.relion.relion_ctf import refuse_generic_ctf_for_optics

if TYPE_CHECKING:
    from recovar.data_io.cryoem_dataset import CryoEMDataset

    from relax.relion.input_particle_table import ParticleLayout

logger = logging.getLogger(__name__)

# RELION's GUI default "Mask diameter (A)" (pipeline_jobs.cpp:4191 Refine3D, 3697 Class3D), the particle
# diameter of a run that gives neither --particle_diameter_ang nor a RELION optimiser STAR.
RELION_GUI_PARTICLE_DIAMETER_ANG = 200.0


@dataclass(frozen=True)
class LoadedParticles:
    """A loaded dataset with the format and preprocessing metadata its consumers need."""

    dataset: CryoEMDataset | MultiShapeDataset | TomoDataset
    tomographic: bool
    shape_class_rows: list[np.ndarray] | None
    mask_parameters: tuple[float, float] | None
    double_preprocessing: bool


class HalfsetParticleInputs(NamedTuple):
    """Selected half rows paired with their startup-noise and accuracy frames."""

    layout: "ParticleLayout"
    noise_source_rows: np.ndarray | None
    noise_optics_group_ids: np.ndarray | None
    accuracy_ctf_params: np.ndarray | None


def validate_particle_optics(optics_table, *, tomographic: bool) -> None:
    """Admit the particle format's qualified optics features."""
    relion_metadata.refuse_unsupported_optics(
        optics_table,
        source="particles.star",
        supported=relion_metadata.TOMO_OPTICS_FEATURES if tomographic else relion_metadata.IMPLEMENTED_OPTICS_FEATURES,
    )


def prepare_relion_halfset_inputs(
    our_particles,
    relion_particles,
    *,
    source_path,
    random_seed: int | None,
    prepare_noise_order: bool,
    tomographic: bool,
    image_grid_size: int | None,
    log,
) -> HalfsetParticleInputs:
    """Adapt a selected half-set source into scoring, noise and accuracy rows.

    Noise uses its pre-shuffle frame. Accuracy CTF rows index the RELION source
    table; tomography instead reads tilt-image CTFs from its dataset half.
    The unused full CTF source table expires after the selected copy is built.
    """
    from relax.relion import input_particle_table

    relion_fresh_initial_noise_source_rows = None
    relion_fresh_initial_noise_optics_group_ids = None
    expected_accuracy_half1_ctf_params = None
    particle_layout = input_particle_table.prepare_relion_halfset_layout(
        our_particles,
        relion_particles,
        random_seed=random_seed,
    )
    if prepare_noise_order:
        from relax.refinement import startup_noise

        (
            relion_fresh_initial_noise_source_rows,
            relion_fresh_initial_noise_optics_group_ids,
        ) = startup_noise.auto_refine_noise_order(our_particles, relion_particles)
    if not tomographic:
        # Subtomogram particles have one CTF per tilt image; their expected accuracy reads
        # those from the tomo half (relax.refinement.tomo_half).
        from recovar.data_io import metadata_readers

        relion_ctf_with_apix = metadata_readers.parse_ctf_from_star(
            source_path,
            image_grid_size,
        )
        expected_accuracy_half1_ctf_params = np.asarray(
            relion_ctf_with_apix[particle_layout.accuracy_particle_ids, 1:],
            dtype=np.float64,
        )
    log.info(
        "Using RELION half-set split: %d (subset=1) + %d (subset=2)",
        len(particle_layout.half1_rows),
        len(particle_layout.half2_rows),
    )
    return HalfsetParticleInputs(
        layout=particle_layout,
        noise_source_rows=relion_fresh_initial_noise_source_rows,
        noise_optics_group_ids=relion_fresh_initial_noise_optics_group_ids,
        accuracy_ctf_params=expected_accuracy_half1_ctf_params,
    )


@dataclass(frozen=True)
class HalfSets:
    """The particles' split into halves, with what the start-up reads from the table it came from.

    ``layout``: the half rows and the expected-accuracy trial order. ``relion_particles``: the RELION half-set
    table (None for a Class3D split of the input STAR). ``optics_image_sizes`` and ``optics_pixel_sizes``: the
    optics geometry every optics group remaps its current size with: the half-set table's for K=1, the input
    STAR's for K>1 (None for a STAR without an optics table). ``noise_source_rows``, ``noise_optics_group_ids`` and
    ``noise_optics_pixel_sizes``: the particles RELION's start-up noise estimate reads, in its order (None when
    the run loads its noise). ``accuracy_ctf_params``: half 1's CTFs for the expected accuracy (None for
    tomography and for Class3D).
    """

    layout: ParticleLayout
    relion_particles: object = None
    optics_image_sizes: np.ndarray | None = None
    optics_pixel_sizes: np.ndarray | None = None
    noise_source_rows: np.ndarray | None = None
    noise_optics_group_ids: np.ndarray | None = None
    noise_optics_pixel_sizes: np.ndarray | None = None
    accuracy_ctf_params: np.ndarray | None = None


def split_half_sets(
    particle_star,
    dataset,
    *,
    halfset_path,
    n_classes: int,
    seed: int,
    init_relion_iteration: int,
    fresh_auto_refine_order: bool,
    noise_order_needed: bool,
    tomographic: bool,
) -> HalfSets:
    """Split the input particles into halves: RELION's half sets from ``halfset_path`` (K=1, required), or
    Class3D's split of the input STAR.

    ``fresh_auto_refine_order`` shuffles the halves in RELION's fresh AutoRefine order (seed + 1).
    ``noise_order_needed``: the run estimates its start-up noise from the images, so the particles that
    estimate reads are listed.
    """
    from relax.relion import input_particle_table

    particles = particle_star["particles"] if isinstance(particle_star, dict) else particle_star
    if halfset_path is not None:
        logger.info("Loading RELION half-set assignments from %s", halfset_path)
        source = input_particle_table.read_relion_halfset_source(halfset_path, n_classes=n_classes, log=logger)
        relion_particles = source.tables["particles"]
        inputs = prepare_relion_halfset_inputs(
            particles,
            relion_particles,
            source_path=halfset_path,
            random_seed=seed if fresh_auto_refine_order else None,
            prepare_noise_order=n_classes == 1 and noise_order_needed,
            tomographic=tomographic,
            image_grid_size=None if tomographic else dataset.grid_size,
            log=logger,
        )
        if fresh_auto_refine_order:
            logger.info(
                "Applied RELION fresh paired AutoRefine particle order (mt19937) with effective seed %d; "
                "BPref will preserve this physical order",
                int(seed) + 1,
            )
        if n_classes == 1:
            return HalfSets(
                layout=inputs.layout,
                relion_particles=relion_particles,
                optics_image_sizes=source.optics_image_sizes,
                optics_pixel_sizes=source.optics_pixel_sizes,
                noise_source_rows=inputs.noise_source_rows,
                noise_optics_group_ids=inputs.noise_optics_group_ids,
                noise_optics_pixel_sizes=source.optics_pixel_sizes,
                accuracy_ctf_params=inputs.accuracy_ctf_params,
            )
        layout = inputs.layout
    elif n_classes == 1:
        raise SystemExit(
            "K=1 auto-refine uses RELION's half sets: a fresh start rebuilds them from the "
            "input STAR (--relion-half-sets-from-input, the default); a RELION-seeded, "
            "replayed or frozen start needs --relion_half_sets"
        )
    else:
        relion_particles = source = inputs = None
        layout = input_particle_table.prepare_class3d_particle_layout(
            particles,
            n_particles=dataset.n_units,
            random_seed=seed,
            init_relion_iteration=init_relion_iteration,
        )
    noise_rows = noise_groups = noise_pixel_sizes = None
    if noise_order_needed:
        from relax.refinement import startup_noise

        noise_rows, noise_groups = startup_noise.class3d_noise_order(particles)
        if not isinstance(particle_star, dict) or "optics" not in particle_star:
            raise SystemExit("Class3D RELION start-up noise needs an optics table in the particle STAR")
        noise_pixel_sizes = np.asarray(particle_star["optics"]["rlnImagePixelSize"], dtype=np.float64)
    # Class3D remaps each optics group's current size as auto-refine does (ml_optimiser.cpp:6917-6923),
    # with the input STAR's optics geometry against the model pixel size (relax#56).
    optics = particle_star.get("optics") if isinstance(particle_star, dict) else None
    return HalfSets(
        layout=layout,
        relion_particles=relion_particles,
        optics_image_sizes=None if optics is None else np.asarray(optics["rlnImageSize"], dtype=np.int64),
        optics_pixel_sizes=None if optics is None else np.asarray(optics["rlnImagePixelSize"], dtype=np.float64),
        noise_source_rows=noise_rows,
        noise_optics_group_ids=noise_groups,
        noise_optics_pixel_sizes=noise_pixel_sizes,
        accuracy_ctf_params=None if inputs is None else inputs.accuracy_ctf_params,
    )


def _apply_relion_image_mask(ds, args, *, relion_half_sets_from_input: bool, sealed_optimiser_star=None):
    """Override the dataset scoring mask with RELION's particle-diameter mask."""
    if sealed_optimiser_star is not None:
        optimiser_star = Path(sealed_optimiser_star).resolve()
        params = relion_metadata.load_relion_mask_params(optimiser_star)
        if params is None:
            raise ValueError(
                f"sealed fixed-arm optimiser lacks RELION mask parameters: {optimiser_star}"
            )
        sealed_particle_diameter, sealed_width_mask_edge = params
        if (
            args.particle_diameter_ang is not None
            and float(args.particle_diameter_ang) != sealed_particle_diameter
        ):
            raise ValueError(
                "fixed diagnostic particle diameter differs from sealed optimiser: "
                f"cli={args.particle_diameter_ang} sealed={sealed_particle_diameter}"
            )
        if float(args.width_mask_edge_px) != sealed_width_mask_edge:
            raise ValueError(
                "fixed diagnostic mask-edge width differs from sealed optimiser: "
                f"cli={args.width_mask_edge_px} sealed={sealed_width_mask_edge}"
            )
    elif args.particle_diameter_ang is not None:
        params = (float(args.particle_diameter_ang), float(args.width_mask_edge_px))
        optimiser_star = "explicit CLI"
    else:
        optimiser_star = command_options.find_relion_optimiser_star(
            args, relion_half_sets_from_input=relion_half_sets_from_input
        )
        params = None if optimiser_star is None else relion_metadata.load_relion_mask_params(optimiser_star)
        if params is None:
            params = (RELION_GUI_PARTICLE_DIAMETER_ANG, float(args.width_mask_edge_px))
            optimiser_star = "RELION GUI default"

    particle_diameter_ang, width_mask_edge_px = params

    if particle_diameter_ang <= 0:
        logger.info("Non-positive RELION particle diameter %.1f A; keeping dataset image mask", particle_diameter_ang)
        return None

    # Use the backend's set_relion_image_mask hook so we get RELION-exact
    # softMaskOutsideMap behavior (geometry + bg-fill mode), not just the
    # mask array overlaid on top of the default "multiply" mode. The
    # multiply mode silently zeros out pixels outside the mask, while
    # RELION blends them with the local background mean — which is what
    # the noise/likelihood downstream expects. See image_backends.py
    # ::set_relion_image_mask for the bit-exact equivalence note.
    backend = ds.image_source.backend
    if hasattr(backend, "set_relion_image_mask"):
        backend.set_relion_image_mask(
            pixel_size=ds.voxel_size,
            particle_diameter_ang=particle_diameter_ang,
            width_mask_edge_px=width_mask_edge_px,
        )
    else:
        from recovar.core import mask as core_mask

        relion_mask = core_mask.relion_soft_image_mask(
            image_size=ds.image_shape[0],
            pixel_size=ds.voxel_size,
            particle_diameter_ang=particle_diameter_ang,
            width_mask_edge_px=width_mask_edge_px,
        )
        backend.image_mask = relion_mask
    if hasattr(ds.image_source, "image_mask"):
        ds.image_source.image_mask = backend.image_mask

    radius_px = particle_diameter_ang / (2.0 * ds.voxel_size)
    logger.info(
        "Applied RELION scoring mask from %s: particle_diameter=%.1f A, width_mask_edge=%.1f px, radius=%.2f px",
        optimiser_star,
        particle_diameter_ang,
        width_mask_edge_px,
        radius_px,
    )
    return params


def load_particle_inputs(
    args,
    *,
    relion_half_sets_from_input: bool,
    image_fourier_backend: str,
    frozen_boundary=None,
    fixed_diagnostic_source_paths=None,
) -> LoadedParticles:
    """Load and configure SPA, multiple-shape or subtomogram command inputs.

    The parsed command owns reading and diagnostic options. Keep mode admission,
    scratch configuration, loading and mask setup in their existing order.
    """
    # ---- Load dataset ----
    logger.info("Loading dataset from %s", args.data_dir)
    from recovar.data_io.cryoem_dataset import load_dataset

    _double_image_preprocessing = parse_env_true_flag("RELAX_USE_FLOAT64_SCORING")
    particle_read_policy = ParticleReadPolicy.from_args(args)
    tomo_run = is_relion5_2d_stack_star(os.path.join(args.data_dir, "particles.star"))
    # A subtomogram particle STAR names no image stacks; load_tomo_dataset stages from its per-tilt STAR.
    particle_scratch = None if tomo_run else prepare_particle_reads(
        os.path.join(args.data_dir, "particles.star"), particle_read_policy
    )
    shape_class_rows = None if tomo_run else optics_shape_class_rows(os.path.join(args.data_dir, "particles.star"))
    if tomo_run:
        # RELION 5 subtomogram 2D stacks: the units are the particles, each over its tilt images.
        command_options.validate_tomo_args(args, frozen_boundary, _double_image_preprocessing)
        flat_star = os.path.join(args.output, "particles_2d.star")
        ds = load_tomo_dataset(
            os.path.join(args.data_dir, "particles.star"),
            os.path.join(args.data_dir, "tomograms.star"),
            flat_star,
            datadir=args.data_dir,
            lazy=not particle_read_policy.preread_images,
            read_policy=particle_read_policy,
        )
        logger.info(
            "Subtomogram particles: %d particles over %d tilt images (%s)",
            ds.n_units, int(ds.unit_image_offsets[-1]), flat_star,
        )
    elif shape_class_rows is None:
        ds = load_dataset(
            image_star(os.path.join(args.data_dir, "particles.star"), particle_scratch),
            lazy=not particle_read_policy.preread_images,
            dtype=np.complex128 if _double_image_preprocessing else np.complex64,
            # relion_refine reads a particle STAR without angles as zero angles.
            absent_angles_zero=True,
        )
        assert_reads_from_scratch(ds, particle_scratch)
        refuse_generic_ctf_for_optics(ds)
    else:
        command_options.validate_multi_shape_args(args, frozen_boundary, _double_image_preprocessing)
        # One dataset per image shape (optics groups sharing box and pixel size).
        ds = MultiShapeDataset(
            [
                load_dataset(
                    image_star(os.path.join(args.data_dir, "particles.star"), particle_scratch),
                    lazy=not particle_read_policy.preread_images,
                    dtype=np.complex64,
                    absent_angles_zero=True,
                    ind=rows,
                )
                for rows in shape_class_rows
            ],
            shape_class_rows,
        )
        for class_dataset in ds.datasets:
            assert_reads_from_scratch(class_dataset, particle_scratch)
            refuse_generic_ctf_for_optics(class_dataset)
        logger.info(
            "Optics groups on %d image shapes: %s",
            len(ds.datasets),
            [(d.image_shape[0], float(d.voxel_size), int(d.n_units)) for d in ds.datasets],
        )
    if _double_image_preprocessing:
        logger.info(
            "Double scoring: loading metadata in float64 and preserving "
            "float64/complex128 through particle masking and FFT"
        )
    # RELION masks every image with its own optics group's pixel size.
    for class_dataset in (
        ds.datasets if shape_class_rows is not None else (ds.images,) if tomo_run else (ds,)
    ):
        relion_mask_params = _apply_relion_image_mask(
            class_dataset,
            args,
            relion_half_sets_from_input=relion_half_sets_from_input,
            sealed_optimiser_star=(
                None
                if frozen_boundary is None or not frozen_boundary.fixed_diagnostic_arm
                else fixed_diagnostic_source_paths["completed_optimiser"]
            ),
        )
    if args.relion_softmask_reduction != "control":
        if image_fourier_backend != "relion_cuda":
            raise ValueError(
                "--relion-softmask-reduction requires --image-fourier-backend relion_cuda"
            )
        backend = getattr(getattr(ds, "image_source", None), "backend", None)
        if backend is None or not hasattr(backend, "set_relion_native_lane_reduction"):
            raise ValueError("Dataset backend does not support RELION soft-mask reduction probes")
        if args.relion_softmask_reduction == "native_lane":
            backend.set_relion_native_lane_reduction(True)
        else:
            os.environ["RECOVAR_RELION_NATIVE_ATOMIC_SOFTMASK_REDUCTION"] = "1"
        logger.warning(
            "Diagnostic RELION soft-mask reduction enabled: %s",
            args.relion_softmask_reduction,
        )
    return LoadedParticles(
        dataset=ds,
        tomographic=tomo_run,
        shape_class_rows=shape_class_rows,
        mask_parameters=relion_mask_params,
        double_preprocessing=_double_image_preprocessing,
    )
