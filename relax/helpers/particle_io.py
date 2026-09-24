"""Where refinement reads particle images from: RELION ``--preread_images`` / ``--scratch_dir``.

RELION's refinement programs share three reading modes (``ml_optimiser.cpp``
option parsing, ``Experiment::copyParticlesToScratch`` in ``exp_model.cpp``):

- default (the RELION GUI default): read each particle from its original stack
  when it is needed;
- ``--preread_images``: read every particle into memory once at start-up;
- ``--scratch_dir DIR``: copy the stacks to local disk at start-up and read from
  the copy. RELION ignores ``--scratch_dir`` together with ``--preread_images``,
  keeps ``--keep_free_scratch`` GB (default 10) free, and removes the copy when
  the run ends.

relax maps these onto recovar's image loader, which stays the one implementation:
lazy loading (``load_dataset(lazy=True)``) streams batches from disk, eager
loading prereads, and recovar's staging (``RECOVAR_CACHE_DIR``,
:func:`recovar.data_io.staging.stage_stacks`) serves the scratch copy. Two
deliberate differences from RELION: the copy holds whole stack files rather
than only the referenced particles, and a scratch directory too small for all
of them is an error before anything is copied, not a partial copy. Each run
stages into its own ``relax_volatile_*`` subdirectory, removed at exit.

recovar also stages implicitly into a node-local ``$TMPDIR``; relax disables
that unless ``--scratch_dir`` is given, so the reading mode is decided by the
command line alone.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_KEEP_FREE_SCRATCH_GB = 10.0
_RECOVAR_CACHE_DIR_ENV = "RECOVAR_CACHE_DIR"


@dataclass(frozen=True)
class ParticleReadPolicy:
    """RELION's particle reading options, as parsed from the command line."""

    preread_images: bool = False
    scratch_dir: str = ""
    keep_free_scratch_gb: float = DEFAULT_KEEP_FREE_SCRATCH_GB

    @classmethod
    def from_args(cls, args) -> ParticleReadPolicy:
        return cls(
            preread_images=bool(args.preread_images),
            scratch_dir=str(args.scratch_dir or ""),
            keep_free_scratch_gb=float(args.keep_free_scratch_gb),
        )

    @property
    def uses_scratch(self) -> bool:
        return bool(self.scratch_dir) and not self.preread_images


def add_particle_read_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--preread_images``, ``--scratch_dir`` and ``--keep_free_scratch``."""

    parser.add_argument(
        "--preread_images",
        "--preread-images",
        dest="preread_images",
        action="store_true",
        help="Read all particles into host memory at start-up (RELION --preread_images).",
    )
    parser.add_argument(
        "--scratch_dir",
        "--scratch-dir",
        dest="scratch_dir",
        default="",
        help=(
            "Copy the particle stacks to this local directory at start-up and read them from there "
            "(RELION --scratch_dir); the copy is removed at exit. On della use /tmp, the job's "
            "node-local NVMe. Ignored with --preread_images."
        ),
    )
    parser.add_argument(
        "--keep_free_scratch",
        "--keep-free-scratch",
        dest="keep_free_scratch_gb",
        type=float,
        default=DEFAULT_KEEP_FREE_SCRATCH_GB,
        help="GB to keep free on --scratch_dir (RELION --keep_free_scratch).",
    )


class ParticleScratch:
    """One run's staged copy of its particle stacks; removed by :meth:`cleanup` or at exit."""

    def __init__(self, directory: str, staged: dict[str, str]):
        self.directory = directory
        self.staged = dict(staged)
        self._removed = False

    def cleanup(self) -> None:
        if self._removed:
            return
        self._removed = True
        shutil.rmtree(self.directory, ignore_errors=True)
        logger.info("Removed particle scratch copy %s", self.directory)


def _raise_system_exit_on_sigterm() -> None:
    """Let Slurm's SIGTERM (time limit, scancel) run the atexit cleanup."""

    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))


def prepare_particle_reads(
    particles_file: str,
    policy: ParticleReadPolicy,
    *,
    datadir: str | None = None,
    strip_prefix: str | None = None,
) -> ParticleScratch | None:
    """Configure recovar's loaders for *policy*; copy the stacks for ``--scratch_dir``.

    Call before ``load_dataset(..., lazy=not policy.preread_images)``. Returns the
    scratch copy, if one was made; it is removed at interpreter exit.
    """

    previous = os.environ.get(_RECOVAR_CACHE_DIR_ENV)
    if policy.preread_images and policy.scratch_dir:
        logger.warning("--scratch_dir %s is ignored with --preread_images, as in RELION", policy.scratch_dir)
    if not policy.uses_scratch:
        if previous:
            logger.warning(
                "Ignoring %s=%s: relax stages particles only with --scratch_dir", _RECOVAR_CACHE_DIR_ENV, previous
            )
        os.environ[_RECOVAR_CACHE_DIR_ENV] = ""
        logger.info(
            "Particle images: %s",
            "preread into host memory" if policy.preread_images else f"read from {particles_file}'s stacks as needed",
        )
        return None

    from recovar.data_io.image_loader import load_images
    from recovar.data_io.staging import stage_stacks

    if not os.path.isdir(policy.scratch_dir):
        raise FileNotFoundError(f"--scratch_dir {policy.scratch_dir} is not an existing directory")
    source_loader = load_images(
        particles_file,
        lazy=True,
        datadir=datadir or "",
        strip_prefix=strip_prefix,
        skip_staging=True,
    )
    sources = source_loader.stack_files()
    source_loader.close()

    directory = tempfile.mkdtemp(prefix="relax_volatile_", dir=policy.scratch_dir)
    scratch = ParticleScratch(directory, {})
    atexit.register(scratch.cleanup)
    _raise_system_exit_on_sigterm()
    try:
        scratch.staged = stage_stacks(
            sources,
            directory,
            keep_free_bytes=int(policy.keep_free_scratch_gb * 1024**3),
        )
    except BaseException:
        scratch.cleanup()
        raise
    os.environ[_RECOVAR_CACHE_DIR_ENV] = directory
    logger.info("Particle images: %d stack file(s) copied to %s", len(scratch.staged), directory)
    return scratch


def assert_reads_from_scratch(dataset, scratch: ParticleScratch | None) -> None:
    """Fail if a loaded dataset reads any stack outside its scratch copy."""

    if scratch is None:
        return
    loader = dataset.image_source.backend.source
    outside = [path for path in loader.stack_files() if not path.startswith(scratch.directory + os.sep)]
    if outside:
        raise RuntimeError(
            f"--scratch_dir: {len(outside)} stack file(s) are read from outside {scratch.directory}: {outside[:3]}"
        )
