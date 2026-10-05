"""``relax refine`` / ``relax class3d`` (``full_refinement.main``) on a tiny written data directory, on the CPU.

``write_tiny_data_dir`` writes a RELION particle STAR of 16-pixel images with its stack and RELION-convention
start-up maps (one per class as well). ``controller_inputs`` runs main up to the controller and returns the
keyword arguments it hands ``refine_single_volume`` (the controller does not run). ``run_tiny_main`` runs
main to the end with ``helpers.fake_adaptive_engine`` in place of the global E-step. Main refuses to start
without a GPU device; both stand in a "gpu" device for that check. ``scripts/dev/fingerprint.py`` keeps its
own copy of the data directory writer, because it must also run against trees older than this helper.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PIXEL_SIZE = 4.25
# A --seed whose random halves are both non-empty for twelve particles.
SEED = "42"


def write_tiny_data_dir(root, *, n_images=12, box=16, n_classes=1, seed=5, extra_columns=None):
    """The data directory main reads: particles.star, its stack, reference_init_relion.mrc and
    reference_init_class00K_relion.mrc. ``extra_columns`` adds particle STAR columns."""

    import mrcfile
    import pandas as pd
    import starfile

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with mrcfile.new(root / f"particles.{box}.mrcs") as stack:
        stack.set_data(rng.standard_normal((n_images, box, box)).astype(np.float32))
        stack.voxel_size = PIXEL_SIZE
    maps = ["reference_init_relion.mrc"] + [f"reference_init_class{k + 1:03d}_relion.mrc" for k in range(n_classes)]
    for name in maps:
        with mrcfile.new(root / name) as volume:
            volume.set_data(rng.standard_normal((box, box, box)).astype(np.float32))
            volume.voxel_size = PIXEL_SIZE
    optics = pd.DataFrame({
        "rlnOpticsGroup": [1], "rlnOpticsGroupName": ["opticsGroup1"], "rlnAmplitudeContrast": [0.07],
        "rlnSphericalAberration": [2.7], "rlnVoltage": [300.0], "rlnImagePixelSize": [PIXEL_SIZE],
        "rlnImageSize": [box], "rlnImageDimensionality": [2],
    })
    rows = np.arange(n_images)
    particles = pd.DataFrame({
        "rlnImageName": [f"{i + 1}@particles.{box}.mrcs" for i in rows],
        "rlnMicrographName": [str(i + 1) for i in rows],
        "rlnDefocusU": 15000.0 + 100.0 * rows, "rlnDefocusV": 15100.0 + 100.0 * rows,
        "rlnDefocusAngle": np.full(n_images, 10.0), "rlnPhaseShift": np.zeros(n_images),
        "rlnOpticsGroup": np.ones(n_images, dtype=int),
        "rlnAngleRot": rng.uniform(-180.0, 180.0, n_images), "rlnAngleTilt": rng.uniform(0.0, 180.0, n_images),
        "rlnAnglePsi": rng.uniform(-180.0, 180.0, n_images),
        "rlnOriginXAngst": np.zeros(n_images), "rlnOriginYAngst": np.zeros(n_images),
        **(extra_columns or {}),
    })
    starfile.write({"optics": optics, "particles": particles}, root / "particles.star")
    return root


def main_frame_arrays():
    """Weak references to the arrays the running ``full_refinement.main`` holds in its locals, by identity.

    Reading a frame's locals leaves a snapshot on the frame (Python 3.11) that holds them until they are read
    again: call it once more before checking that a replaced array is gone.
    """

    import weakref

    from relax.refinement import full_refinement

    code = getattr(full_refinement.main, "__wrapped__", full_refinement.main).__code__
    frame = sys._getframe(1)
    while frame is not None and frame.f_code is not code:
        frame = frame.f_back
    if frame is None:
        raise AssertionError("full_refinement.main is not running")
    return [weakref.ref(value) for value in frame.f_locals.values() if hasattr(value, "shape") and hasattr(value, "dtype")]


class ControllerReached(Exception):
    """Raised by the stand-in controller of ``controller_inputs``."""

    def __init__(self, kwargs):
        super().__init__("refine_single_volume reached")
        self.kwargs = kwargs


def _stand_in_device(monkeypatch):
    import jax

    # memory_stats: on a GPU node the XLA reserve check asks the device for its pool limit; none is reported.
    device = SimpleNamespace(platform="gpu", id=0, memory_stats=dict)
    monkeypatch.setattr(jax, "devices", lambda *args, **kwargs: [device])


def _run_main(monkeypatch, command, data, output, arguments, seed=SEED):
    from relax.refinement import full_refinement

    seeded = [] if seed is None else ["--seed", str(seed)]
    argv = ["relax", "--data_dir", str(data), "--output", str(output), *seeded, *map(str, arguments)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "orig_argv", ["python", "-m", f"relax.commands.{command}", *argv[1:]])
    environ = dict(os.environ)
    try:
        full_refinement.run_from_command_line(command)
    finally:  # main sets K=1 environment defaults
        os.environ.clear()
        os.environ.update(environ)


def controller_inputs(monkeypatch, tmp_path, command, *arguments, n_classes=1, data=None, seed=SEED):
    """The keyword arguments ``full_refinement.main`` hands ``refine_single_volume`` for ``arguments``.

    ``data`` is a data directory (default: a new ``write_tiny_data_dir``); ``arguments`` may name files in
    it as ``<DATA>/name``. ``seed=None`` passes no ``--seed``.
    """

    from relax.refinement import iteration_loop

    data = write_tiny_data_dir(tmp_path / "data", n_classes=n_classes) if data is None else Path(data)

    def stand_in(**kwargs):
        raise ControllerReached(kwargs)

    _stand_in_device(monkeypatch)
    monkeypatch.setattr(iteration_loop, "refine_single_volume", stand_in)
    arguments = [str(argument).replace("<DATA>", str(data)) for argument in arguments]
    try:
        _run_main(monkeypatch, command, data, tmp_path / "out", arguments, seed=seed)
    except ControllerReached as reached:
        return reached.kwargs
    raise AssertionError("full_refinement.main returned without calling refine_single_volume")


def run_tiny_main(monkeypatch, tmp_path, command, *arguments, n_classes=1, data=None, output="out"):
    """Run ``relax <command>`` to the end on the stand-in engine; returns the output directory."""

    from helpers.fake_adaptive_engine import install_fake_adaptive_engine

    data = write_tiny_data_dir(tmp_path / "data", n_classes=n_classes) if data is None else Path(data)
    _stand_in_device(monkeypatch)
    install_fake_adaptive_engine(monkeypatch)
    arguments = [str(argument).replace("<DATA>", str(data)) for argument in arguments]
    _run_main(monkeypatch, command, data, tmp_path / output, arguments)
    return tmp_path / output
