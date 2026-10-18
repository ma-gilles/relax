"""Score a PPCA run trained on a converted particle STAR with the curated PPCA test suite's scorer.

The suite (``/scratch/gpfs/GILLES/mg6942/ppca_test_suite_20261006``, read-only) scores a checkpoint with
``tools/suite_score.py``, which infers latents and poses on the dataset's ``training/manifest.json``: RECOVAR-simulator
images, protein dark, read by its pinned relax. A run of ``relax ppca_initial_model --i`` on that dataset's converted
STAR (``scripts/prepare_vdam_ppca_star.py``: the same images negated, RELION's sign) learns the opposite sign: its
internal mean is the negated simulator mean (``relax/io/map_io.py``). This wrapper copies the checkpoint into a
temporary run directory with the mean ``theta[:, 0]`` negated and calls the scorer on the copy; the loadings keep
their sign, and the scorer's own latents absorb it (``mu + W z`` on the dark images is ``-(mu_int) + W (-z_int)``).
Every other array and the metadata are copied unchanged, so the copy of one checkpoint is the same file every time
and ``--reuse-inference`` still recognises it.

    ppca_suite_score_star.py --run <run dir> --eval <dataset>/evaluation [--checkpoint ...] [--output ...] [scorer options]

``--run``, ``--checkpoint`` and ``--output`` mean what they mean to the scorer (the default output is
``<run>/suite_score_<checkpoint stem>``, beside the original run, not in the temporary copy); every other argument is
passed through. The scorer runs under this interpreter (``--python`` to change it) with this environment, so start it
as the suite's README says (its ``cpu_env.sh`` / ``gpu_env.sh``). ``star_sign_copy.json`` in the output records the
original checkpoint and the copy's sha256.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_SUITE_SCORE = Path("/scratch/gpfs/GILLES/mg6942/ppca_test_suite_20261006/tools/suite_score.py")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_checkpoint(run):
    """The scorer's default: the highest-numbered ``checkpoint_*.npz`` of the run."""
    candidates = sorted(Path(run).glob("checkpoint_[0-9]*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_*.npz in {run}")
    return candidates[-1]


def write_mean_negated_copy(checkpoint, directory):
    """Copy ``checkpoint`` into ``directory`` under its own name with ``theta[:, 0]`` negated; return the copy."""
    copy = Path(directory) / Path(checkpoint).name
    with np.load(checkpoint, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    theta = np.array(arrays["theta"], copy=True)
    if theta.ndim != 2 or theta.shape[1] < 2:
        raise ValueError(f"Expected a PPCA theta of one mean and at least one loading, got {theta.shape}")
    theta[:, 0] = -theta[:, 0]
    arrays["theta"] = theta
    with open(copy, "wb") as stream:
        np.savez(stream, **arrays)
    return copy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="relax ppca_initial_model --i output directory")
    parser.add_argument("--checkpoint", type=Path, help="Default: the run's highest checkpoint_*.npz")
    parser.add_argument("--output", type=Path, help="Default: <run>/suite_score_<checkpoint stem>")
    parser.add_argument("--suite-score", type=Path, default=DEFAULT_SUITE_SCORE, help="The suite's suite_score.py")
    parser.add_argument("--python", default=sys.executable, help="Interpreter for the scorer (default: this one)")
    args, passthrough = parser.parse_known_args(argv)
    checkpoint = (args.checkpoint or latest_checkpoint(args.run)).resolve()
    output = (args.output or args.run / f"suite_score_{checkpoint.stem}").resolve()
    with tempfile.TemporaryDirectory(prefix="ppca_suite_score_star_") as temporary:
        copy = write_mean_negated_copy(checkpoint, temporary)
        record = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "mean_negated_copy_sha256": _sha256(copy),
            "reason": "run trained on a RELION-sign STAR; the scorer infers on the RECOVAR-sign manifest images",
            "suite_score": str(args.suite_score.resolve()),
        }
        command = [args.python, str(args.suite_score), "--run", temporary, "--checkpoint", str(copy)]
        completed = subprocess.run([*command, "--output", str(output), *passthrough], check=False)
    if completed.returncode == 0:
        (output / "star_sign_copy.json").write_text(json.dumps(record, indent=2) + "\n")
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
