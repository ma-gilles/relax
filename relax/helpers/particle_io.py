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
lazy loading (``load_dataset(lazy=True)``) streams batches from disk and eager
loading prereads. For single particles ``--scratch_dir`` stages as RELION does
(``Experiment::copyParticlesToScratch``, ``getImageNameOnScratch``,
exp_model.cpp:500-530): only the referenced particles, one compact stack per
optics group in STAR order, read through a copy of the particle STAR in the
scratch directory whose ``_rlnImageName`` points into those stacks; every other
column is the input's text. The input STAR stays the only source of image names
for every output. Subtomogram tilt stacks are one file per particle and are
staged whole through recovar's staging (``RECOVAR_CACHE_DIR``,
:func:`recovar.data_io.staging.stage_stacks`). One deliberate difference from
RELION: a scratch directory too small for the copy is an error before anything
is copied, not a partial copy. Each run stages into its own ``relax_volatile_*``
subdirectory, removed at exit.

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

import numpy as np

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
    """One run's staged copy of its particle stacks; removed by :meth:`cleanup` or at exit.

    ``image_star`` is the scratch copy of the particle STAR that reads the compact stacks (single
    particles), or None when the stacks were staged whole (subtomograms, read through
    ``RECOVAR_CACHE_DIR``).
    """

    def __init__(self, directory: str, staged: dict[str, str], image_star: str | None = None):
        self.directory = directory
        self.staged = dict(staged)
        self.image_star = image_star
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
    compact: bool = True,
) -> ParticleScratch | None:
    """Configure recovar's loaders for *policy*; copy the particles for ``--scratch_dir``.

    Call before ``load_dataset(image_star(particles_file, scratch), lazy=not policy.preread_images)``.
    With ``compact`` (single particles) only the referenced particles are copied, one stack per optics
    group; without it (subtomogram tilt stacks) the stack files are copied whole. Returns the scratch
    copy, if one was made; it is removed at interpreter exit.
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
    file_map = source_loader._file_map if compact else None
    source_loader.close()

    directory = tempfile.mkdtemp(prefix="relax_volatile_", dir=policy.scratch_dir)
    scratch = ParticleScratch(directory, {})
    atexit.register(scratch.cleanup)
    _raise_system_exit_on_sigterm()
    keep_free_bytes = int(policy.keep_free_scratch_gb * 1024**3)
    try:
        if compact:
            scratch.image_star, scratch.staged = _stage_compact(
                particles_file, file_map, directory, keep_free_bytes=keep_free_bytes
            )
        else:
            scratch.staged = stage_stacks(sources, directory, keep_free_bytes=keep_free_bytes)
    except BaseException:
        scratch.cleanup()
        raise
    if compact:
        # The compact stacks are named by the scratch STAR: no transparent redirect.
        os.environ[_RECOVAR_CACHE_DIR_ENV] = ""
        logger.info(
            "Particle images: %d particle(s) copied to %d optics-group stack(s) in %s",
            len(file_map),
            len(scratch.staged),
            directory,
        )
    else:
        os.environ[_RECOVAR_CACHE_DIR_ENV] = directory
        logger.info("Particle images: %d stack file(s) copied to %s", len(scratch.staged), directory)
    return scratch


def image_star(particles_file: str, scratch: ParticleScratch | None) -> str:
    """The STAR a dataset reads its images through: the scratch copy for compact stacks.

    The scratch copy names its stacks relative to its own directory, so it is loaded without the
    input's ``datadir`` or ``strip_prefix``. Image names for outputs always come from the input STAR.
    """

    if scratch is None or scratch.image_star is None:
        return particles_file
    return scratch.image_star


def _star_particle_rows(text: str) -> tuple[list[str], int, list[int], dict[str, int]]:
    """Locate the particle loop of a RELION STAR text: (lines, name column, data line numbers, columns)."""

    lines = text.split("\n")
    start = next((i for i, line in enumerate(lines) if line.strip() == "data_particles"), None)
    if start is None:
        start = next((i for i, line in enumerate(lines) if line.strip().startswith("data_")), 0)
    columns: dict[str, int] = {}
    rows: list[int] = []
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("data_") and rows:
            break
        if not stripped or stripped.startswith("#") or stripped == "loop_":
            continue
        if stripped.startswith("_"):
            if rows:
                break
            columns[stripped.split()[0]] = len(columns)
            continue
        rows.append(i)
    if "_rlnImageName" not in columns:
        raise ValueError("the particle STAR has no _rlnImageName column to stage from")
    return lines, columns["_rlnImageName"], rows, columns


# Gaps of up to this many unselected images are read through, so a dense selection becomes a few long
# sequential reads (whole-file speed); a span is capped so the reads still spread over the threads.
_COPY_GAP_IMAGES = 4
_COPY_SPAN_IMAGES = 256


def _copy_threads() -> int:
    """The copy threads: the CPUs this process may use (its Slurm allocation), at most 16."""

    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        cpus = os.cpu_count() or 1
    return max(1, min(16, cpus))


def _copy_images(source: str, layout, shape, file_indices, out, slots) -> None:
    """Copy images ``file_indices`` of an MRC stack into ``out[slots]`` with parallel positional reads.

    The sorted indices are grouped into spans (consecutive images, gaps of at most
    ``_COPY_GAP_IMAGES`` read through, at most ``_COPY_SPAN_IMAGES`` long), one ``os.pread`` per span on
    :func:`_copy_threads` threads: a parallel file system serves many concurrent reads far faster than
    one strided stream (10k images scattered over a 131k-image GPFS stack took 100 s through a memmap),
    and a dense selection reads the file in long sequential spans as a whole-file copy would.
    """

    from concurrent.futures import ThreadPoolExecutor

    first_byte, dtype = layout
    image_bytes = int(np.dtype(dtype).itemsize) * shape[0] * shape[1]
    order = np.argsort(file_indices, kind="stable")
    src = np.asarray(file_indices, dtype=np.int64)[order]
    dst = np.asarray(slots, dtype=np.int64)[order]
    spans, begin = [], 0
    for k in range(1, src.size + 1):
        if k == src.size or src[k] - src[k - 1] > _COPY_GAP_IMAGES or src[k] - src[begin] >= _COPY_SPAN_IMAGES:
            spans.append(np.arange(begin, k))
            begin = k
    fd = os.open(source, os.O_RDONLY)
    try:

        def read(span):
            start = int(src[span[0]])
            count = int(src[span[-1]]) - start + 1
            data = os.pread(fd, count * image_bytes, first_byte + start * image_bytes)
            if len(data) != count * image_bytes:
                raise OSError(f"{source}: short read of images {start + 1}-{start + count}")
            images = np.frombuffer(data, dtype=dtype).reshape(count, *shape)
            out[dst[span]] = images[src[span] - start]

        with ThreadPoolExecutor(max_workers=_copy_threads()) as pool:
            list(pool.map(read, spans))
    finally:
        os.close(fd)


def _stage_compact(particles_file: str, file_map, directory: str, *, keep_free_bytes: int):
    """Copy the referenced particles into one stack per optics group, in STAR order (RELION's
    ``copyParticlesToScratch``), and write the STAR copy that reads them.

    ``file_map`` is the loader's per-row ``mrc_file``/``mrc_index`` (resolved like every read). A stack
    shared by optics groups and a particle named twice each get their own slot. Returns the scratch
    STAR path and ``{stack file name: path}``.
    """

    import time

    import mrcfile
    from recovar.data_io.staging import StagingSpaceError

    with open(particles_file) as handle:
        text = handle.read()
    lines, name_column, row_lines, columns = _star_particle_rows(text)
    if len(row_lines) != len(file_map):
        raise ValueError(f"{particles_file}: {len(row_lines)} particle rows, but the loader has {len(file_map)}")
    group_column = columns.get("_rlnOpticsGroup")
    tokens = [lines[i].split() for i in row_lines]
    if any(len(row) != len(columns) for row in tokens):
        raise ValueError(f"{particles_file}: a particle row does not have {len(columns)} columns")
    groups = [row[group_column] if group_column is not None else "1" for row in tokens]
    files = list(file_map["mrc_file"])
    indices = np.asarray(file_map["mrc_index"], dtype=np.int64)

    plan = {}  # group label -> (rows in STAR order, (ny, nx), mode, voxel size)
    headers = {}
    layout = {}  # path -> (first image byte, image dtype)
    for path in dict.fromkeys(files):
        with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
            headers[path] = (tuple(int(v) for v in mrc.data.shape[-2:]), int(mrc.header.mode), mrc.voxel_size.copy())
            layout[path] = (1024 + int(mrc.header.nsymbt), mrc.data.dtype)
    for row, group in enumerate(groups):
        shape, mode, voxel = headers[files[row]]
        entry = plan.setdefault(group, ([], shape, mode, voxel))
        if (shape, mode) != entry[1:3]:
            raise ValueError(f"optics group {group}: its stacks differ in image shape or MRC mode")
        entry[0].append(row)
    itemsize = {0: 1, 1: 2, 2: 4, 4: 8, 6: 2, 12: 2}
    needed = sum(
        1024 + len(rows) * shape[0] * shape[1] * itemsize.get(mode, 4) for rows, shape, mode, _ in plan.values()
    )
    needed += len(text)
    free = shutil.disk_usage(directory).free
    if free - needed < keep_free_bytes:
        raise StagingSpaceError(
            f"Staging {len(files)} particle(s) needs {needed / 1e9:.2f} GB plus {keep_free_bytes / 1e9:.2f} GB "
            f"kept free, but {directory} has only {free / 1e9:.2f} GB free. Use a larger local directory "
            "or read from the source."
        )

    t0 = time.monotonic()
    staged = {}
    slot = np.empty(len(files), dtype=np.int64)
    names = {}
    for group, (rows, shape, mode, voxel) in plan.items():
        name = f"opticsgroup{group}_particles.mrcs"
        path = os.path.join(directory, name)
        rows = np.asarray(rows, dtype=np.int64)
        slot[rows] = np.arange(rows.size)
        for row in rows:
            names[int(row)] = name
        with mrcfile.new_mmap(path, shape=(rows.size, *shape), mrc_mode=mode, overwrite=False) as out:
            out.voxel_size = voxel
            out.header.ispg = 0  # a stack of images, as relion writes particle stacks
            by_source = {}
            for k, row in enumerate(rows):
                by_source.setdefault(files[row], []).append(k)
            for source, ks in by_source.items():
                ks = np.asarray(ks, dtype=np.int64)
                _copy_images(source, layout[source], shape, indices[rows[ks]], out.data, ks)
        staged[name] = path
    for row, i in enumerate(row_lines):
        tokens[row][name_column] = f"{slot[row] + 1:06d}@{names[row]}"
        lines[i] = "\t".join(tokens[row])
    star = os.path.join(directory, os.path.basename(particles_file))
    with open(star, "w") as handle:
        handle.write("\n".join(lines))
    logger.info(
        "Staged %d particle(s), %.2f GB, into %d optics-group stack(s) in %.1fs",
        len(files),
        needed / 1e9,
        len(staged),
        time.monotonic() - t0,
    )
    return star, staged


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
