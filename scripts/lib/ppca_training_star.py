"""RELION particle STARs for single-particle PPCA training data written by study scripts.

``relax ppca_initial_model --i`` reads a particle STAR as ``relax refine`` does: the images as
stored, in RELION's sign (protein white), with the CTF of the STAR's particle and optics tables
(``relax.commands.ppca_initial_model.load_training``). A RECOVAR-simulator stack is the opposite
sign: the simulator projects a positive density through RECOVAR's CTF
(``recovar.core.ctf.evaluate_ctf``, negative at low frequency), so protein is dark. Its RELION-sign
stack is the simulator images negated (:func:`relion_sign_images`); relax then holds the negated
simulator volumes in its internal frame (``relax/io/map_io.py``), which a comparison with simulator
ground truth undoes explicitly.

Script library, not imported by any relax command.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Columns of recovar's CTF parameter rows (``recovar.core.ctf.CTFParamIndex``) the STAR carries.
_DFU, _DFV, _DFANG, _VOLT, _CS, _W, _PHASE_SHIFT = range(7)


def relion_sign_images(simulator_images):
    """RELION-sign (protein white) float32 images of RECOVAR-simulator (protein dark) images."""
    return np.negative(np.asarray(simulator_images, np.float32))


def write_training_star(star_path, stack_name, ctf_params, *, voxel_size, box, header="relax PPCA training particles"):
    """Write a RELION 3.1 particle STAR naming images ``1..N`` of ``stack_name`` with one optics group.

    ``ctf_params`` holds one row per image in recovar's ``CTFParamIndex`` order (defocus U and V in
    Angstrom, defocus angle in degrees, voltage in kV, Cs in mm, amplitude contrast, phase shift in
    degrees; later columns are ignored), as ``ImageMetadata`` and a recovar CTF pickle (after its two
    geometry columns) hold it. Voltage, Cs and amplitude contrast must be the same for every image:
    RELION keeps them in the optics table (``rlnVoltage``, ``rlnSphericalAberration``,
    ``rlnAmplitudeContrast``). No angles or offsets are written. ``stack_name`` is written as given
    (relative names resolve against the STAR's directory). Values are written with ``repr`` and no
    time stamp, so the same inputs give the same bytes (the STAR's sha256 is the checkpoint identity).
    """
    ctf = np.asarray(ctf_params, np.float64)
    if ctf.ndim != 2 or ctf.shape[1] <= _PHASE_SHIFT or ctf.shape[0] == 0:
        raise ValueError(f"Expected one CTF row of at least {_PHASE_SHIFT + 1} columns per image, got {ctf.shape}")
    for column, label in ((_VOLT, "voltage"), (_CS, "Cs"), (_W, "amplitude contrast")):
        if not np.all(ctf[:, column] == ctf[0, column]):
            raise ValueError(f"One optics group needs one {label} for every image")
    if any(char.isspace() for char in str(stack_name)):
        raise ValueError(f"A STAR image name cannot contain white space: {stack_name!r}")
    lines = [
        f"# {header}",
        "",
        "data_optics",
        "",
        "loop_",
        "_rlnOpticsGroup",
        "_rlnOpticsGroupName",
        "_rlnVoltage",
        "_rlnSphericalAberration",
        "_rlnAmplitudeContrast",
        "_rlnImagePixelSize",
        "_rlnImageSize",
        "_rlnImageDimensionality",
        " ".join(
            [
                "1",
                "opticsGroup1",
                repr(float(ctf[0, _VOLT])),
                repr(float(ctf[0, _CS])),
                repr(float(ctf[0, _W])),
                repr(float(voxel_size)),
                str(int(box)),
                "2",
            ]
        ),
        "",
        "",
        "data_particles",
        "",
        "loop_",
        "_rlnImageName",
        "_rlnOpticsGroup",
        "_rlnDefocusU",
        "_rlnDefocusV",
        "_rlnDefocusAngle",
        "_rlnPhaseShift",
    ]
    lines += [
        f"{index}@{stack_name} 1 {row[_DFU]!r} {row[_DFV]!r} {row[_DFANG]!r} {row[_PHASE_SHIFT]!r}"
        for index, row in enumerate((tuple(map(float, row)) for row in ctf), start=1)
    ]
    star_path = Path(star_path)
    star_path.write_text("\n".join(lines) + "\n")
    return star_path
