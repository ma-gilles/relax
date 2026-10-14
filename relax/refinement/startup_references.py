"""The start-up references of a refinement: the maps relion_refine reads with --ref, in the loop's layout.

References are RELION-convention maps; ``load_relion_volume`` puts them in the internal frame
(``relax.io.map_io``), and ``get_dft3`` gives the centered Fourier volume. NEVER use raw
``mrcfile.open`` + ``np.fft.fftn(np.fft.ifftshift(...))`` here: that produces a Fourier volume with the
right values but at WRONG array indices (DC at corner instead of center), so ``slice_volume`` reads Nyquist
as if it were DC and projections are off by ~2400x in amplitude at low frequencies.

With a start-up low-pass (``--apply-initial-lowpass``), each map is filtered as RELION's
``initialLowPassFilterReferences`` (ml_optimiser.cpp:3556) filters ``mymodel.Iref`` at start-up, gated only on
``ini_high > 0``. The command chooses the precision (``real_dtype``/``complex_dtype``) and the maps.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from recovar import utils as recovar_utils
from recovar.core import fourier_transform_utils as ftu

from relax.relion.geometry import REFERENCE_FILTER_EDGE_SHELLS
from relax.star import metadata
from relax.steps.half_inputs import HalfPair


@dataclass(frozen=True)
class StartupReferences:
    """The references a run starts from.

    ``fourier``: the loop's start-up volume as a ``HalfPair``: ``(V,)`` for K=1 or ``(K, V)`` for Class3D,
    shared by the halves, or each half's own (a K=1 reference named ``half1``/``half2``,
    :func:`relion_half_reference_path`, or a frozen boundary). ``prior_source``: the Fourier volume the start-up tau2 is bootstrapped from (the
    reference or class 1); None from a frozen boundary, which owns its tau2. ``real_for_projector``: the
    real-space maps handed to the first projector (float64, a ``HalfPair``) when that handoff is on, else None.
    ``reference_real`` (K=1, a ``HalfPair``: each half set's map) and
    ``class_references_real`` (Class3D, a list): the float64 real maps of RELION's start-up tau2 and data-vs-prior.
    ``model_pixel_size``: the reference maps' header pixel size, RELION's model pixel size (None from a frozen
    boundary).
    """

    fourier: HalfPair
    prior_source: np.ndarray | None
    real_for_projector: HalfPair | None = None
    reference_real: HalfPair | np.ndarray | None = None
    class_references_real: list | None = None
    model_pixel_size: float | None = None


class StartupHandoff:
    """The start-up arrays handed to ``refine_single_volume``, which takes them once.

    ``volume``: the loop's start-up Fourier volume, a ``HalfPair`` (``StartupReferences.fourier``).
    ``mean_variance``: the start-up tau2. ``reference_real``: the float64 real maps of the first projector as a
    ``HalfPair`` (of ``real_for_projector``), or None.
    A caller's argument stays referenced until the call returns (relax#26), so at box 800 the three arrays (12.3 GB)
    lived for the whole refinement; :meth:`take` hands them over and drops the holder's references, so they are
    freed once the loop has built its models and first projector.
    """

    def __init__(self, volume, mean_variance, reference_real=None):
        self._arrays = (volume, mean_variance, reference_real)
        self.hands_reference_real = reference_real is not None

    def take(self):
        """``(volume, mean_variance, reference_real)``; the holder keeps none of them. A second take raises."""
        if self._arrays is None:
            raise RuntimeError("the start-up arrays were already taken")
        arrays, self._arrays = self._arrays, None
        return arrays


def _initial_lowpass_real(volume_real, volume_shape, voxel_size, ini_high):
    from relax.relion.reference_initialization import initial_low_pass_filter_references

    filtered = initial_low_pass_filter_references(
        np.asarray(volume_real, dtype=np.float64)[None, ...],
        box_size=int(volume_shape[0]),
        pixel_size=float(voxel_size),
        ini_high_ang=float(ini_high),
        filter_edgewidth=float(REFERENCE_FILTER_EDGE_SHELLS),
    )[0]
    return np.asarray(filtered, dtype=np.float64)


def frozen_boundary_references(frozen_boundary) -> StartupReferences:
    """The per-half Fourier references a frozen boundary owns; its tau2 is its own, so nothing is bootstrapped."""
    return StartupReferences(fourier=HalfPair(*frozen_boundary.means), prior_source=None)


def _startup_map(volume_real, *, volume_shape, pixel_size, ini_high: float | None, real_dtype, complex_dtype):
    """One loaded reference map as the loop starts from it: ``(fourier, real64)``.

    With ``ini_high``, RELION's start-up low-pass at ``pixel_size``, back in ``real_dtype`` for the DFT;
    ``real64`` is the float64 real map (the filtered one when filtered), ``fourier`` the flat centred DFT in
    ``complex_dtype``.
    """
    if ini_high is not None:
        real64 = _initial_lowpass_real(volume_real, volume_shape, pixel_size, ini_high)
        volume_real = real64.astype(real_dtype, copy=False)
    else:
        real64 = np.asarray(volume_real, dtype=np.float64)
    fourier = np.array(ftu.get_dft3(jnp.asarray(volume_real))).astype(complex_dtype).reshape(-1)
    return fourier, real64


def relion_half_reference_path(path, halfset: int) -> str:
    """The reference half set ``halfset`` (1 or 2) of a split-halves refinement reads for ``--ref path``.

    RELION's MPI refinement gives each half-set rank ``fn_ref.getHalf(fn_ref, my_halfset)``
    (ml_optimiser_mpi.cpp:726-731): in the base name (after the last ``/``), every ``half1`` becomes ``half2``
    for half set 2 when the name contains ``half1``, else every ``half2`` becomes ``half1`` for half set 1
    when it contains ``half2``; any other name is read as given (``FileName::getHalf``, filename.cpp:474-489).
    """
    path = str(path)
    directory, slash, base = path.rpartition("/")
    if "half1" in base and halfset == 2:
        base = base.replace("half1", "half2")
    elif "half2" in base and halfset == 1:
        base = base.replace("half2", "half1")
    else:
        return path
    return directory + slash + base


def load_k1_reference(
    path,
    *,
    volume_shape,
    ini_high: float | None,
    real_for_projector: bool,
    real_dtype,
    complex_dtype,
    log: logging.Logger,
    split_halves: bool = True,
) -> StartupReferences:
    """Load the K=1 reference map ``path``, low-pass filtered at ``ini_high`` in the map's own pixel size.

    With ``split_halves`` (a fresh auto-refine; a continued run reads no reference, as RELION's ``--continue``
    does not), each half set reads its own file, :func:`relion_half_reference_path`: ``--ref half1.mrc`` gives
    half set 2 ``half2.mrc``. A missing file stops the run, as RELION's read of it does. ``reference_real`` holds
    each half set's map, from which each half's start-up tau2 and data-vs-prior follow, as each RELION rank's
    ``initialiseDataVersusPrior`` reads its own ``Iref`` (ml_optimiser_mpi.cpp:771); ``prior_source`` (the
    bootstrap of a start that is not RELION's) is half 1's.
    """
    paths = (
        HalfPair(relion_half_reference_path(path, 1), relion_half_reference_path(path, 2))
        if split_halves
        else HalfPair.shared(str(path))
    )
    for half_path in dict.fromkeys(paths):
        if not os.path.exists(half_path):
            raise SystemExit(
                f"Initial reference {half_path} does not exist; RELION's split-halves refinement reads it for one "
                f"half set when --ref is {path} (ml_optimiser_mpi.cpp:730, FileName::getHalf)"
            )
    log.info("Initial reference per half set: half 1 reads %s, half 2 reads %s", paths.half1, paths.half2)
    loaded = {
        half_path: _load_k1_map(
            half_path, volume_shape=volume_shape, ini_high=ini_high, real_dtype=real_dtype,
            complex_dtype=complex_dtype, log=log,
        )
        for half_path in dict.fromkeys(paths)
    }
    pixel_sizes = {loaded[p][2] for p in loaded}
    if len(pixel_sizes) > 1:
        raise SystemExit(
            f"The half-set references {paths.half1} and {paths.half2} have different header pixel sizes "
            f"{sorted(pixel_sizes)}; relax keeps one model pixel size"
        )
    fourier = HalfPair(loaded[paths.half1][0], loaded[paths.half2][0])
    reference_real = HalfPair(loaded[paths.half1][1], loaded[paths.half2][1])
    if paths.half2 == paths.half1:
        fourier, reference_real = HalfPair.shared(fourier.half1), HalfPair.shared(reference_real.half1)
    if ini_high is not None:
        log.info(
            "Applied RELION initialLowPassFilterReferences to init reference: ini_high=%.2f A, fmask_edge=%d shells",
            ini_high, REFERENCE_FILTER_EDGE_SHELLS,
        )
    return StartupReferences(
        fourier=fourier,
        prior_source=fourier.half1,
        real_for_projector=reference_real if real_for_projector else None,
        reference_real=reference_real,
        model_pixel_size=loaded[paths.half1][2],
    )


def _load_k1_map(path, *, volume_shape, ini_high, real_dtype, complex_dtype, log):
    """One K=1 reference map as the loop starts from it: ``(fourier, real64, model_pixel_size)``."""
    from recovar.utils.helpers import load_relion_volume

    init_vol_real = load_relion_volume(path).astype(real_dtype)
    model_pixel_size = metadata.read_relion_mrc_model_pixel_size(path)
    if not np.isfinite(model_pixel_size) or model_pixel_size <= 0.0:
        raise SystemExit(f"Initial RELION reference has invalid voxel size {model_pixel_size}: {path}")
    assert init_vol_real.shape == volume_shape, f"Volume shape mismatch: {init_vol_real.shape} vs {volume_shape}"
    # RELION filters ``mymodel.Iref`` in model coordinates.  The
    # particle STAR optics pixel size can be a rounded serialization
    # (for example 1.416667 versus the MRC header ratio
    # 544.0 / 384 = 1.4166666666666667 A/px),
    # which is enough to flip marginal firstiter-CC winners.
    init_vol_ft, reference_real = _startup_map(
        init_vol_real,
        volume_shape=volume_shape,
        pixel_size=model_pixel_size,
        ini_high=ini_high,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )
    log.info(
        "Initial volume loaded from %s: shape=%s model_pixel_size=%.9g A/px",
        path,
        init_vol_real.shape,
        model_pixel_size,
    )
    return init_vol_ft, reference_real, model_pixel_size


def load_class_references(
    paths,
    *,
    volume_shape,
    ini_high: float | None,
    real_for_projector: bool,
    real_dtype,
    complex_dtype,
    log: logging.Logger,
) -> StartupReferences:
    """Load one map per class, low-pass filtered at ``ini_high`` in the maps' header pixel size.

    RELION's model pixel size is the first reference's header value, and references whose headers differ by
    more than 0.001 A are refused (``MlModel::initialiseFromImages``, ml_model.cpp:899-918); the start-up
    low-pass uses it (``radius = ori_size * mymodel.pixel_size / ini_high``, ml_optimiser.cpp:3563), as
    :func:`load_k1_reference` does.
    """
    from recovar.utils.helpers import load_relion_volume

    model_pixel_size = metadata.read_relion_mrc_model_pixel_size(paths[0])
    if not np.isfinite(model_pixel_size) or model_pixel_size <= 0.0:
        raise SystemExit(f"Initial RELION reference has invalid voxel size {model_pixel_size}: {paths[0]}")
    for p in paths[1:]:
        header_pixel_size = metadata.read_relion_mrc_model_pixel_size(p)
        if abs(header_pixel_size - model_pixel_size) > 0.001:
            raise SystemExit(
                f"Class references have different pixel sizes in their headers: {model_pixel_size} A "
                f"({paths[0]}) and {header_pixel_size} A ({p}); RELION refuses this (ml_model.cpp:912-915)"
            )

    per_class_ft = []
    per_class_real_for_projector = []
    class_references_real = []
    for k, p in enumerate(paths):
        vol_real = np.asarray(load_relion_volume(p)).astype(real_dtype)
        assert vol_real.shape == volume_shape, (
            f"Class {k + 1} volume shape mismatch at {p}: {vol_real.shape} vs {volume_shape}"
        )
        vol_ft, reference_real = _startup_map(
            vol_real,
            volume_shape=volume_shape,
            pixel_size=model_pixel_size,
            ini_high=ini_high,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
        )
        class_references_real.append(reference_real)
        if real_for_projector:
            per_class_real_for_projector.append(reference_real)
        per_class_ft.append(vol_ft)
        log.info("Class %d initial volume loaded from %s", k + 1, p)
    if ini_high is not None:
        log.info(
            "Applied RELION initialLowPassFilterReferences to %d init references: ini_high=%.2f A, fmask_edge=%d shells",
            len(paths), ini_high, REFERENCE_FILTER_EDGE_SHELLS,
        )
    # Stack to (K, V): the halves share the one class stack.
    return StartupReferences(
        fourier=HalfPair.shared(np.stack(per_class_ft, axis=0)),
        # The K-class start-up prior uses class 1 as the representative (a single spectrum).
        prior_source=per_class_ft[0],
        real_for_projector=HalfPair.shared(np.stack(per_class_real_for_projector, axis=0)) if real_for_projector else None,
        class_references_real=class_references_real,
        model_pixel_size=model_pixel_size,
    )


def bootstrap_prior(prior_source, volume_shape):
    """The start-up tau2 bootstrapped from a reference's power spectrum: half of it, floored at 1e-4 of its
    maximum. A RELION start (``relion_start_tau2_and_data_vs_prior``) or a loaded model replaces it."""
    from recovar.reconstruction.regularization import average_over_shells

    power = average_over_shells(jnp.abs(jnp.asarray(prior_source)) ** 2, volume_shape)
    prior = recovar_utils.make_radial_image(power, volume_shape, extend_last_frequency=True)
    return jnp.asarray(prior * 0.5 + jnp.max(prior) * 1e-4)


def _relion_start_shells(
    reference_real,
    initial_noise_radial,
    *,
    box_size: int,
    tau2_fudge: float,
    nr_particles: int,
    pdf_class: float,
    shell_pair_counting: str,
):
    """RELION's start-up tau2 and data_vs_prior shell curves (RELION units) of one class
    (:func:`relion_start_tau2_and_data_vs_prior`)."""

    from recovar.utils.helpers import recovar_volume_to_relion

    from relax.relion.reference_initialization import relion_initial_tau2_and_data_vs_prior

    n4 = float(box_size) ** 4
    sigma2 = np.asarray(initial_noise_radial, dtype=np.float64)
    if sigma2.ndim == 2:
        # The unweighted mean over optics groups that have noise (ml_model.cpp:1560-1573).
        sigma2 = np.mean(sigma2[np.sum(sigma2, axis=1) > 0.0], axis=0)
    sigma2 = sigma2.reshape(-1) / n4
    n_shells = int(box_size) // 2 + 1
    if sigma2.size < n_shells:
        raise ValueError(f"initial noise spectrum has {sigma2.size} shells, need {n_shells}")
    return relion_initial_tau2_and_data_vs_prior(
        recovar_volume_to_relion(np.asarray(reference_real, dtype=np.float64)),
        tau2_fudge=float(tau2_fudge),
        avg_sigma2_noise=sigma2[:n_shells],
        nr_particles=int(nr_particles),
        pdf_class=float(pdf_class),
        shell_pair_counting=shell_pair_counting,
    )


def relion_start_tau2_and_data_vs_prior(
    reference_real,
    initial_noise_radial,
    *,
    box_size: int,
    volume_shape,
    tau2_fudge: float,
    nr_particles: int,
    pdf_class: float = 1.0,
    shell_pair_counting: str = "relion",
):
    """RELION's start-up tau2 (RECOVAR units) and data_vs_prior (RELION units) of one class.

    ``MlModel::initialiseDataVersusPrior`` (ml_model.cpp:1557) on the start-up
    reference after ``initialLowPassFilterReferences``, with the initial noise
    (averaged over optics groups when it has one row per group) and the particle count: each
    auto-refine half model counts its own particles (K=1); a Class3D model counts all of them,
    with the class's start-up ``pdf_class`` (1/K). ``reference_real`` is in RECOVAR's frame and
    ``initial_noise_radial`` is RELION sigma2 times ``box_size**4``.
    """

    tau2, data_vs_prior = _relion_start_shells(
        reference_real,
        initial_noise_radial,
        box_size=box_size,
        tau2_fudge=tau2_fudge,
        nr_particles=nr_particles,
        pdf_class=pdf_class,
        shell_pair_counting=shell_pair_counting,
    )
    mean_variance = jnp.asarray(
        recovar_utils.make_radial_image(tau2 * float(box_size) ** 4, volume_shape, extend_last_frequency=True)
    ).reshape(-1)
    return mean_variance, data_vs_prior


def relion_start_half_sets(
    reference_real: HalfPair, initial_noise_radial, nr_particles, *, box_size, volume_shape, tau2_fudge,
    shell_pair_counting,
):
    """Each auto-refine half set's start-up tau2 and data_vs_prior, as its RELION rank computes them.

    Every rank runs ``initialiseDataVersusPrior`` (ml_optimiser_mpi.cpp:771) on its own ``Iref`` (the getHalf
    reference, :func:`relion_half_reference_path`) with the noise rank 1 estimated and broadcast (:753) and its own
    half set's particle count ``nr_particles[k]``. Returns ``(mean_variance, (data_vs_prior_1, data_vs_prior_2))``:
    the tau2 is ``(V,)`` when the halves share one reference and ``[2, V]`` (each half's own) when they do not.
    An empty half set 2 (``--diagnostic_single_half``, never scored) takes half set 1's.
    """

    def start(reference, count):
        return relion_start_tau2_and_data_vs_prior(
            reference,
            initial_noise_radial,
            box_size=box_size,
            volume_shape=volume_shape,
            tau2_fudge=tau2_fudge,
            nr_particles=int(count),
            shell_pair_counting=shell_pair_counting,
        )

    tau2_1, curve_1 = start(reference_real.half1, nr_particles[0])
    if int(nr_particles[1]) == 0:
        return tau2_1, (curve_1, curve_1)
    tau2_2, curve_2 = start(reference_real.half2, nr_particles[1])
    mean_variance = tau2_1 if reference_real.half2 is reference_real.half1 else jnp.stack([tau2_1, tau2_2])
    return mean_variance, (curve_1, curve_2)


def class_start_data_vs_prior(
    class_references_real, initial_noise_radial, *, box_size, tau2_fudge, nr_particles, shell_pair_counting,
):
    """Class3D's start-up data_vs_prior, ``(K, shells)``: each class over all particles at pdf_class 1/K.

    The first iteration's scale-correction sums take only the shells where it exceeds 3
    (ml_optimiser.cpp:10473). The class tau2 volumes stay the loop's own; only that gate reads this curve,
    so no tau2 volume is built.
    """
    n_classes = len(class_references_real)
    return np.stack(
        [
            _relion_start_shells(
                reference,
                initial_noise_radial,
                box_size=box_size,
                tau2_fudge=tau2_fudge,
                nr_particles=nr_particles,
                pdf_class=1.0 / n_classes,
                shell_pair_counting=shell_pair_counting,
            )[1]
            for reference in class_references_real
        ],
        axis=0,
    )
