#!/usr/bin/env python
"""Derive an optics-feature fixture from a curated synthetic fixture.

Takes the first ``--n-images`` particles of a curated fixture (particles.star, its
stack and its reference/GT maps) and writes them with one RELION optics feature
applied, so relax and RELION can be compared on data that exercises it:

``--premultiply-ctf``
    Each image is multiplied in Fourier space by its own RELION CTF
    (``CTF::getFftwImage`` through relax's RELION binding, the arithmetic
    ``relion_refine`` evaluates for the particle) and the optics group is marked
    ``rlnCtfDataAreCtfPremultiplied 1``. A premultiplied image of real data is
    ``CTF * (CTF * projection + noise)``: exactly what this produces from the
    source image.

``--beam-tilt TX TY`` / ``--odd-zernike "[c0,c1,...]"``
    The optics group gets ``rlnBeamTiltX/Y`` (mrad) and ``rlnOddZernike``, and
    each image is phase-modulated by their odd phase ``exp(i phase)``
    (``ObservationModel::modulatePhase``; the phase of
    :mod:`relax.relion.optics_aberrations`, which matches RELION's
    ``getPhaseCorrection``). White noise modulated by a pure phase is the same
    white noise, so the result is a draw from the aberrated forward model.

``--even-zernike "[c0,c1,...]"``
    The optics group gets ``rlnEvenZernike``, which changes the CTF itself, so the
    signal of every image is re-simulated: the base image ``x`` is fitted per image
    as ``a * idft(CTF * P) + b`` against the recovar simulator's clean projection
    ``P`` of the ground-truth map (the base fixture's forward model), and
    ``a * idft((CTF_even - CTF) * P)`` is added, with both CTFs from RELION's
    ObservationModel (relion_bind optics_ctf_images_batch). The base image's noise is
    kept. Needs a GPU for the projections.

``--mag-matrix M00 M01 M10 M11``
    The optics group gets ``rlnMagMat00..11``. The signal is re-simulated as above,
    now ``a * idft(CTF_mag * P_mag - CTF * P)`` with ``P_mag`` projected with RELION's
    magnified matrix (``applyAnisoMag``: relax's projection matrix times ``M3^T`` on the
    left, :func:`relax.relion.optics_aberrations.projection_rotations`) and ``CTF_mag``
    RELION's CTF at the magnified frequency.

The base fixture's ground truth, initial reference and masks stay valid because
the particles, their poses and the maps are unchanged. The output directory gets a
README.md and GENERATION.json (command, source SHA, input and output sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import starfile

COPIED_MAPS = (
    "reference_gt.mrc",
    "reference_gt_class001.mrc",
    "reference_gt_relion.mrc",
    "reference_init_class001.mrc",
    "reference_init_class001_relion.mrc",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _column(frame, name, optics, default=None):
    """A per-particle value from the particle table, else its optics group's, else ``default``."""

    if name in frame.columns:
        return np.asarray(frame[name], dtype=np.float64)
    groups = np.asarray(frame["rlnOpticsGroup"], dtype=np.int64)
    by_group = {}
    for _, row in optics.iterrows():
        group = int(row["rlnOpticsGroup"])
        if name in optics.columns:
            by_group[group] = float(row[name])
        elif default is not None:
            by_group[group] = float(default)
        else:
            raise KeyError(f"optics group {group} has no {name}")
    return np.asarray([by_group[int(g)] for g in groups], dtype=np.float64)


def relion_ctf_images(particles, optics, box: int, *, n_threads: int) -> np.ndarray:
    """RELION's ``Fctf`` of every particle on the FFTW half grid ``(N, box, box // 2 + 1)``.

    The same ``CTF::setValues`` + ``getFftwImage`` the binding evaluates for
    ``relion_refine``'s CTF operands (no padding, not absolute, with damping).
    """

    from relax.relion_bind import _relion_bind_core as relion_bind

    params = np.stack(
        [
            _column(particles, "rlnDefocusU", optics),
            _column(particles, "rlnDefocusV", optics),
            _column(particles, "rlnDefocusAngle", optics),
            _column(particles, "rlnVoltage", optics),
            _column(particles, "rlnSphericalAberration", optics),
            _column(particles, "rlnAmplitudeContrast", optics),
            _column(particles, "rlnCtfBfactor", optics, 0.0),
            _column(particles, "rlnImagePixelSize", optics),
            _column(particles, "rlnPhaseShift", optics, 0.0),
            _column(particles, "rlnCtfScalefactor", optics, 1.0),
        ],
        axis=1,
    )
    return np.asarray(
        relion_bind.get_ctf_images_batch(np.ascontiguousarray(params), box, box, False, False, True, n_threads),
        dtype=np.float64,
    )


def premultiply(images: np.ndarray, factor_fftw_half: np.ndarray) -> np.ndarray:
    """``irfft2(rfft2(image) * factor)`` in float64, returned as float32.

    ``factor`` is the CTF (real, point-symmetric) and/or the odd-aberration
    modulation ``exp(i phase)`` (Hermitian: the phase is odd). Either commutes with
    the ``(-1)^(x+y)`` factor that moves the transform origin to the image centre, so
    the product does not depend on where the transform puts the image origin.
    """

    box = images.shape[-1]
    spectrum = np.fft.rfft2(np.asarray(images, dtype=np.float64), axes=(-2, -1))
    return np.fft.irfft2(spectrum * factor_fftw_half, s=(box, box), axes=(-2, -1)).astype(np.float32)


def odd_phase(optics_row, box: int) -> np.ndarray:
    """The optics group's odd aberration phase on the FFTW half grid ``(box, box // 2 + 1)``."""

    from relax.relion import optics_aberrations as oa

    labels = set(optics_row.index)
    coefficients = oa.optics_group_odd_coefficients(
        optics_row, has_odd="rlnOddZernike" in labels, has_tilt=bool({"rlnBeamTiltX", "rlnBeamTiltY"} & labels)
    )
    return oa.zernike_phase_fftw_half(
        coefficients, oa.odd_index_to_mn, box, float(optics_row["rlnImagePixelSize"]), int(optics_row["rlnImageSize"])
    )


def relion_half_to_recovar_full(half: np.ndarray) -> np.ndarray:
    """RELION FFTW-half CTFs ``(B, N, N // 2 + 1)`` as RECOVAR's full centered ``(B, N * N)`` operand.

    RECOVAR's frame negates RELION's CTF (relion_ctf._evaluate_exact_ctf_rows); the
    CTF is point-symmetric, so a negative-x pixel takes its mirror's value.
    """

    count, size, _ = half.shape
    k = np.arange(size) - size // 2
    ky, kx = np.meshgrid(k, k, indexing="ij")
    rows = np.where(kx >= 0, ky, -ky) % size
    cols = np.abs(kx)
    return -half[:, rows, cols].reshape(count, size * size)


def optics_signal_change(
    base_dir, particles, star_plain, star_even, images, *, batch: int, threads: int, magnification=None
):
    """``a * idft(CTF_new * P_new - CTF * P)`` per image, fitted to the base images (see the module docstring).

    ``star_even`` holds the new optics table; ``magnification`` is its 2x2 matrix or None.
    """

    import jax.numpy as jnp
    from recovar import utils
    from recovar.core import fourier_transform_utils as ftu
    from recovar.core.configs import ForwardModelConfig
    from recovar.core.slicing import slice_volume
    from recovar.data_io.cryoem_dataset import load_dataset

    from relax.relion.optics_aberrations import projection_rotations, relax_projection_magnification
    from relax.relion_bind import _relion_bind_core as relion_bind

    count = images.shape[0]
    dataset = load_dataset(str(base_dir / "particles.star"), ind=np.arange(count), lazy=True, absent_angles_zero=True)
    config = ForwardModelConfig.from_dataset(dataset, disc_type="cubic")
    volume = jnp.asarray(ftu.get_dft3(utils.load_mrc(str(base_dir / "reference_gt_class001.mrc"))).reshape(-1))
    size = int(images.shape[-1])
    params = np.stack(
        [
            np.asarray(particles["rlnDefocusU"], dtype=np.float64),
            np.asarray(particles["rlnDefocusV"], dtype=np.float64),
            np.asarray(particles["rlnDefocusAngle"], dtype=np.float64),
            np.zeros(count),
            np.ones(count),
            np.asarray(particles.get("rlnPhaseShift", np.zeros(count)), dtype=np.float64),
            np.asarray(particles["rlnOpticsGroup"], dtype=np.float64),
        ],
        axis=1,
    )
    k = np.arange(size) - size // 2
    inside = ((k[:, None] ** 2 + k[None, :] ** 2) < (size // 2 - 1) ** 2).reshape(-1)
    change = np.empty_like(images, dtype=np.float32)
    report = {"fit_correlation_min": 1.0, "recovar_vs_relion_plain_ctf_max_abs": 0.0, "scale_mean": 0.0}
    for start in range(0, count, batch):
        stop = min(count, start + batch)
        rows = np.arange(start, stop)
        plain = relion_half_to_recovar_full(
            np.asarray(relion_bind.optics_ctf_images_batch(str(star_plain), params[rows], size, size, False, threads))
        )
        even = relion_half_to_recovar_full(
            np.asarray(relion_bind.optics_ctf_images_batch(str(star_even), params[rows], size, size, False, threads))
        )
        # The base fixture was simulated with RECOVAR's own CTF; inside the Nyquist
        # circle it must be RELION's (the edge row and column alias differently).
        generic = np.asarray(config.compute_ctf(jnp.asarray(dataset.CTF_params[rows])))
        report["recovar_vs_relion_plain_ctf_max_abs"] = max(
            report["recovar_vs_relion_plain_ctf_max_abs"], float(np.abs(generic - plain)[:, inside].max())
        )
        rotations = np.asarray(dataset.rotation_matrices[rows])
        slices = slice_volume(volume, jnp.asarray(rotations), config.image_shape, config.volume_shape, "cubic")
        clean = np.asarray(ftu.get_idft2((jnp.asarray(plain) * slices).reshape(-1, size, size)).real)
        if magnification is None:
            new_slices = slices
        else:
            magnified = projection_rotations(rotations, 1.0, relax_projection_magnification(magnification))
            new_slices = slice_volume(
                volume, jnp.asarray(magnified, dtype=rotations.dtype), config.image_shape, config.volume_shape, "cubic"
            )
        delta = np.asarray(
            ftu.get_idft2((jnp.asarray(even) * new_slices - jnp.asarray(plain) * slices).reshape(-1, size, size)).real
        )
        base = np.asarray(images[start:stop], dtype=np.float64)
        for local, index in enumerate(rows):
            design = np.stack([clean[local].reshape(-1), np.ones(size * size)], axis=1)
            (scale, offset), *_ = np.linalg.lstsq(design, base[local].reshape(-1), rcond=None)
            fit = scale * clean[local].reshape(-1) + offset
            corr = float(np.corrcoef(fit, base[local].reshape(-1))[0, 1])
            report["fit_correlation_min"] = min(report["fit_correlation_min"], corr)
            report["scale_mean"] += float(scale) / count
            change[index] = (scale * delta[local]).astype(np.float32)
    return change, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path, required=True, help="curated fixture with particles.star")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-images", type=int, required=True)
    parser.add_argument("--premultiply-ctf", action="store_true")
    parser.add_argument("--beam-tilt", type=float, nargs=2, metavar=("TX", "TY"), help="beam tilt in mrad")
    parser.add_argument("--odd-zernike", type=str, help='RELION odd Zernike coefficients, "[c0,c1,...]"')
    parser.add_argument("--even-zernike", type=str, help='RELION even Zernike coefficients, "[c0,c1,...]"')
    parser.add_argument("--mag-matrix", type=float, nargs=4, metavar=("M00", "M01", "M10", "M11"))
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)

    odd = args.beam_tilt is not None or args.odd_zernike is not None
    even = args.even_zernike is not None
    mag = None if args.mag_matrix is None else np.asarray(args.mag_matrix, dtype=np.float64).reshape(2, 2)
    if not (args.premultiply_ctf or odd or even or mag is not None):
        parser.error(
            "choose an optics feature (--premultiply-ctf, --beam-tilt, --odd-zernike, --even-zernike, --mag-matrix)"
        )
    base = args.base_dir.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)

    tables = starfile.read(base / "particles.star", always_dict=True)
    optics = tables["optics"].copy()
    particles = tables["particles"].iloc[: int(args.n_images)].copy()
    if len(particles) != int(args.n_images):
        raise SystemExit(f"{base} has only {len(particles)} particles")
    if optics["rlnOpticsGroup"].nunique() != 1:
        raise SystemExit("the base fixture must have one optics group")

    locations = [str(name).split("@") for name in particles["rlnImageName"]]
    stacks = {stack for _, stack in locations}
    if len(stacks) != 1:
        raise SystemExit(f"the base particles must come from one stack, got {sorted(stacks)}")
    (stack_name,) = stacks
    slices = np.asarray([int(index) - 1 for index, _ in locations], dtype=np.int64)
    box = int(optics["rlnImageSize"].iloc[0])
    out_stack_name = f"particles.{box}.mrcs"
    if args.beam_tilt is not None:
        optics["rlnBeamTiltX"], optics["rlnBeamTiltY"] = float(args.beam_tilt[0]), float(args.beam_tilt[1])
    if args.odd_zernike is not None:
        optics["rlnOddZernike"] = str(args.odd_zernike).replace(" ", "")
    if even:
        optics["rlnEvenZernike"] = str(args.even_zernike).replace(" ", "")
    if mag is not None:
        for (i, j), value in np.ndenumerate(mag):
            optics[f"rlnMagMat{i}{j}"] = float(value)
    source_names = particles["rlnImageName"].copy()
    particles["rlnImageName"] = [f"{row + 1}@{out_stack_name}" for row in range(len(particles))]
    # Written before the images: the even-Zernike CTF is read from it (RELION's ObservationModel).
    starfile.write({"optics": optics, "particles": particles}, out / "particles.star", overwrite=True)
    # 1 when the images keep their phase; exp(i phase) of the odd aberrations otherwise.
    modulation = np.exp(1j * odd_phase(optics.iloc[0], box)) if odd else 1.0

    with mrcfile.mmap(base / stack_name, mode="r", permissive=True) as source:
        data = source.data
        voxel_size = source.voxel_size.copy()
        if data.shape[1:] != (box, box):
            raise SystemExit(f"stack images are {data.shape[1:]}, the optics table says {box}")
        ctf = (
            relion_ctf_images(particles, optics, box, n_threads=int(args.threads))
            if args.premultiply_ctf
            else np.ones((len(slices), 1, 1))
        )
        change = None
        signal_report = None
        if even or mag is not None:
            change, signal_report = optics_signal_change(
                base,
                particles.assign(rlnImageName=source_names),
                base / "particles.star",
                out / "particles.star",
                np.asarray(data[slices]),
                batch=int(args.batch),
                threads=int(args.threads),
                magnification=mag,
            )
            print(json.dumps({"optics_signal_change": signal_report}))
        with mrcfile.new_mmap(
            out / out_stack_name, shape=(len(slices), box, box), mrc_mode=2, overwrite=False
        ) as target:
            for start in range(0, len(slices), int(args.batch)):
                stop = min(start + int(args.batch), len(slices))
                images = np.asarray(data[slices[start:stop]], dtype=np.float64)
                if change is not None:
                    images = images + change[start:stop]
                target.data[start:stop] = premultiply(images, ctf[start:stop] * modulation)
            target.voxel_size = voxel_size

    if args.premultiply_ctf:
        optics["rlnCtfDataAreCtfPremultiplied"] = 1
        starfile.write({"optics": optics, "particles": particles}, out / "particles.star", overwrite=True)

    for name in COPIED_MAPS:
        if (base / name).is_file():
            shutil.copy2(base / name, out / name)
    for name in ("reference_init_classes_relion.star",):
        if (base / name).is_file():
            text = (base / name).read_text().replace(str(base), str(out))
            (out / name).write_text(text)

    here = Path(__file__).resolve()
    repo = here.parents[1]
    record = {
        "command": [sys.executable, str(here), *(argv if argv is not None else sys.argv[1:])],
        "relax_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "relax_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
            )
        ),
        "base_dir": str(base),
        "base_particles_star_sha256": _sha256(base / "particles.star"),
        "n_images": int(args.n_images),
        "features": {
            "ctf_premultiplied": bool(args.premultiply_ctf),
            "beam_tilt_mrad": None if args.beam_tilt is None else [float(v) for v in args.beam_tilt],
            "odd_zernike": args.odd_zernike,
            "even_zernike": args.even_zernike,
            "mag_matrix": None if mag is None else mag.tolist(),
        },
        "optics_signal_change": signal_report,
        "ctf": "relax.relion_bind get_ctf_images_batch (CTF::setValues + getFftwImage, damping on, no padding)",
        "outputs_sha256": {path.name: _sha256(path) for path in sorted(out.iterdir()) if path.is_file()},
    }
    (out / "GENERATION.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
