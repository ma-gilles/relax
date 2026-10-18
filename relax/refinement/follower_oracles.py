"""RELION's captured follower dispatch schedule.

Expectation ownership is *not* a static equal partition.  RELION's leader
hands each next ``--pool`` chunk to whichever follower requests work next.
Exact replay therefore requires a schedule captured from the same RELION run.

``load_relion_dispatch_schedule`` and ``verify_relion_dispatch_schedule_oracle`` load and validate it
(schema version, sha256 digests, artifact paths); ``make_relion_dispatch_schedule_from_chunks`` builds a
schedule. The definitions moved verbatim from ``relax.refinement.follower_scale``, which reads them to
prepare the follower topology.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RelionDispatchSchedule:
    """Captured dynamic follower ownership in shuffled-position order."""

    relion_iterations: np.ndarray
    owner_by_sorted_position: np.ndarray
    original_particle_id_by_sorted_position: np.ndarray
    n_followers: int
    pool_size: int
    random_seed: int
    oracle_id: str
    oracle_manifest_sha256: str
    oracle_artifact_paths: tuple[str, ...]
    particle_order_sha256: str
    particle_star_relative_path: str
    dispatch_log_relative_path: str
    source: str


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


_RELION_ORACLE_SCHEMA_VERSION = 3


def relion_dispatch_metadata_relative_path(dispatch_log_relative_path: str) -> str:
    """Return the manifest-bound metadata sidecar for a dispatch capture."""

    dispatch = _validate_oracle_artifact_paths([dispatch_log_relative_path])[0]
    return f"{dispatch}.recovar_schedule.json"


def _load_relion_dispatch_rows(path: Path) -> np.ndarray:
    try:
        rows = np.loadtxt(path, dtype=np.int64, comments="#", ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot parse RELION dispatch log {path}: {exc}") from exc
    if rows.ndim != 2 or rows.shape[1] not in (4, 5) or rows.shape[0] < 1:
        raise ValueError(
            f"RELION dispatch log {path} must contain legacy four-column ranges "
            "or v2 five-column identity records; "
            f"observed shape {rows.shape}"
        )
    if rows.shape[1] == 5 and not np.all(rows[:, 0] == 2):
        raise ValueError(f"RELION dispatch log {path} contains a non-v2 schema record")
    return rows


def _relion_dispatch_chunk_columns(rows: np.ndarray, n_particles: int):
    """Return replay chunk columns without conflating sorted and original IDs."""

    if rows.shape[1] == 4:
        raise ValueError(
            "legacy four-column dispatch ranges do not contain the authoritative "
            "sorted-position to original-particle mapping"
        )
    iterations = rows[:, 1]
    ranks = rows[:, 2]
    sorted_positions = rows[:, 3]
    original_ids = rows[:, 4]
    expected = np.arange(int(n_particles), dtype=np.int64)
    for iteration in np.unique(iterations):
        selected = iterations == iteration
        if not np.array_equal(np.sort(sorted_positions[selected]), expected):
            raise ValueError(
                f"v2 dispatch sorted positions are not a bijection at iteration {iteration}"
            )
        if not np.array_equal(np.sort(original_ids[selected]), expected):
            raise ValueError(
                f"v2 dispatch original particle IDs are not a bijection at iteration {iteration}"
            )
    return iterations, ranks, sorted_positions, sorted_positions, original_ids


def _validate_sha256(value: str, *, name: str) -> str:
    value = str(value)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def _validate_oracle_artifact_paths(paths) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in np.asarray(paths).reshape(-1).tolist())
    if not normalized:
        raise ValueError("oracle_artifact_paths must be non-empty")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("oracle_artifact_paths must be sorted and unique")
    for value in normalized:
        path = Path(value)
        if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
            raise ValueError(
                "oracle_artifact_paths must contain normalized relative POSIX paths"
            )
    return normalized


def relion_oracle_manifest_sha256(
    oracle_dir: str | Path,
    artifact_paths,
) -> str:
    """Hash exact RELION state-file names and bytes without binding to a host path."""

    root = Path(oracle_dir).expanduser().resolve()
    paths = _validate_oracle_artifact_paths(artifact_paths)
    digest = hashlib.sha256(b"recovar-relion-oracle-manifest-v1\0")
    for relative in paths:
        artifact = root / relative
        if not artifact.is_file():
            raise ValueError(f"RELION oracle artifact is missing or not a file: {artifact}")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(int(artifact.stat().st_size).to_bytes(8, "little"))
        with artifact.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def relion_ordered_particle_sha256(particles) -> str:
    """Hash RELION particle identities and parity-relevant labels in row order."""

    if "rlnImageName" not in particles.columns:
        raise ValueError("RELION oracle particle STAR is missing rlnImageName")
    columns = [
        name
        for name in (
            "rlnImageName",
            "rlnOpticsGroup",
            "rlnRandomSubset",
            "rlnGroupNumber",
        )
        if name in particles.columns
    ]
    digest = hashlib.sha256(b"recovar-relion-ordered-particles-v1\0")
    for column in columns:
        encoded = column.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    digest.update(int(len(particles)).to_bytes(8, "little"))
    for row in particles[columns].itertuples(index=False, name=None):
        for value in row:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def relion_oracle_id(*, manifest_sha256: str, particle_order_sha256: str) -> str:
    """Derive the portable identity shared by schedule and follower replay."""

    manifest = _validate_sha256(manifest_sha256, name="oracle_manifest_sha256")
    order = _validate_sha256(particle_order_sha256, name="particle_order_sha256")
    digest = hashlib.sha256(b"recovar-relion-oracle-id-v1\0")
    digest.update(manifest.encode("ascii"))
    digest.update(order.encode("ascii"))
    return digest.hexdigest()


def verify_relion_dispatch_schedule_oracle(
    schedule: RelionDispatchSchedule,
    oracle_dir: str | Path,
) -> None:
    """Fail closed unless the supplied RELION directory is the captured oracle."""

    root = Path(oracle_dir).expanduser().resolve()
    observed_manifest = relion_oracle_manifest_sha256(
        root,
        schedule.oracle_artifact_paths,
    )
    if observed_manifest != schedule.oracle_manifest_sha256:
        raise ValueError(
            "RELION oracle state manifest does not match the dispatch schedule "
            f"({observed_manifest} != {schedule.oracle_manifest_sha256})"
        )
    particle_path = root / schedule.particle_star_relative_path
    try:
        import starfile

        table = starfile.read(particle_path)
    except Exception as exc:
        raise ValueError(f"cannot read RELION oracle particle STAR {particle_path}: {exc}") from exc
    particles = table["particles"] if isinstance(table, dict) else table
    observed_order = relion_ordered_particle_sha256(particles)
    if observed_order != schedule.particle_order_sha256:
        raise ValueError(
            "ordered RELION particle identity does not match the dispatch schedule "
            f"({observed_order} != {schedule.particle_order_sha256})"
        )

    metadata_relative = relion_dispatch_metadata_relative_path(
        schedule.dispatch_log_relative_path
    )
    if metadata_relative not in schedule.oracle_artifact_paths:
        raise ValueError(
            "dispatch metadata sidecar is absent from the verified oracle manifest"
        )
    metadata_path = root / metadata_relative
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read dispatch metadata {metadata_path}: {exc}") from exc
    expected_metadata = {
        "schema_version": 2,
        "dispatch_log_schema_version": 2,
        "schedule_schema_version": 3,
        "dispatch_log_relative_path": schedule.dispatch_log_relative_path,
        "n_particles": int(schedule.owner_by_sorted_position.shape[1]),
        "n_followers": int(schedule.n_followers),
        "pool_size": int(schedule.pool_size),
        "random_seed": int(schedule.random_seed),
    }
    if metadata != expected_metadata:
        raise ValueError(
            "manifest-bound dispatch metadata does not match the schedule payload: "
            f"observed={metadata!r} expected={expected_metadata!r}"
        )
    rows = _load_relion_dispatch_rows(root / schedule.dispatch_log_relative_path)
    chunk_iterations, chunk_ranks, chunk_first, chunk_last, chunk_original_ids = _relion_dispatch_chunk_columns(
        rows, expected_metadata["n_particles"]
    )
    original_by_sorted = np.empty(
        (schedule.relion_iterations.size, expected_metadata["n_particles"]), dtype=np.int64
    )
    for row_idx, iteration in enumerate(schedule.relion_iterations):
        selected = chunk_iterations == iteration
        original_by_sorted[row_idx, chunk_first[selected]] = chunk_original_ids[selected]
    reconstructed = make_relion_dispatch_schedule_from_chunks(
        relion_iterations=np.unique(chunk_iterations),
        chunk_iterations=chunk_iterations,
        chunk_ranks=chunk_ranks,
        chunk_first=chunk_first,
        chunk_last=chunk_last,
        n_particles=expected_metadata["n_particles"],
        original_particle_id_by_sorted_position=original_by_sorted,
        n_followers=expected_metadata["n_followers"],
        pool_size=expected_metadata["pool_size"],
        random_seed=expected_metadata["random_seed"],
        oracle_id=schedule.oracle_id,
        oracle_manifest_sha256=schedule.oracle_manifest_sha256,
        oracle_artifact_paths=schedule.oracle_artifact_paths,
        particle_order_sha256=schedule.particle_order_sha256,
        particle_star_relative_path=schedule.particle_star_relative_path,
        dispatch_log_relative_path=schedule.dispatch_log_relative_path,
        source=str(root / schedule.dispatch_log_relative_path),
    )
    if not np.array_equal(reconstructed.relion_iterations, schedule.relion_iterations):
        raise ValueError("schedule iterations were not derived from the bound dispatch log")
    if not np.array_equal(
        reconstructed.owner_by_sorted_position,
        schedule.owner_by_sorted_position,
    ):
        raise ValueError("schedule follower owners were not derived from the bound dispatch log")
    if not np.array_equal(
        reconstructed.original_particle_id_by_sorted_position,
        schedule.original_particle_id_by_sorted_position,
    ):
        raise ValueError("schedule particle identities were not derived from the bound dispatch log")


def load_relion_dispatch_schedule(path: str | Path) -> RelionDispatchSchedule:
    """Load and validate an exact per-iteration RELION dispatch schedule."""

    schedule_path = Path(path).expanduser().resolve()
    with np.load(schedule_path, allow_pickle=False) as values:
        required = {
            "schema_version",
            "relion_iterations",
            "owner_by_sorted_position",
            "original_particle_id_by_sorted_position",
            "n_followers",
            "pool_size",
            "random_seed",
            "oracle_id",
            "oracle_manifest_sha256",
            "oracle_artifact_paths",
            "particle_order_sha256",
            "particle_star_relative_path",
            "dispatch_log_relative_path",
        }
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(
                f"RELION dispatch schedule {schedule_path} is missing keys: {', '.join(missing)}"
            )
        schema_raw = np.asarray(values["schema_version"])
        if not np.issubdtype(schema_raw.dtype, np.integer) or schema_raw.size != 1:
            raise ValueError("schema_version must be an integer scalar")
        schema_version = int(schema_raw.reshape(()))
        if schema_version != _RELION_ORACLE_SCHEMA_VERSION:
            if schema_version == 2:
                raise ValueError(
                    "RELION dispatch schedule schema v2 lacks the authoritative "
                    "original_particle_id_by_sorted_position mapping; rebuild schema v3 "
                    "from a five-column RELION text-log schema v2 capture. Legacy "
                    "four-column logs cannot be migrated exactly."
                )
            raise ValueError(
                "unsupported RELION dispatch schedule schema_version "
                f"{schema_version}; expected {_RELION_ORACLE_SCHEMA_VERSION}"
            )
        iterations_raw = np.asarray(values["relion_iterations"])
        if not np.issubdtype(iterations_raw.dtype, np.integer):
            raise ValueError("relion_iterations must have an integer dtype")
        iterations = np.asarray(iterations_raw, dtype=np.int64).reshape(-1)
        owners_raw = np.asarray(values["owner_by_sorted_position"])
        if not np.issubdtype(owners_raw.dtype, np.integer):
            raise ValueError("owner_by_sorted_position must have an integer dtype")
        owners = np.asarray(owners_raw, dtype=np.int64)
        originals_raw = np.asarray(values["original_particle_id_by_sorted_position"])
        if not np.issubdtype(originals_raw.dtype, np.integer):
            raise ValueError("original_particle_id_by_sorted_position must have an integer dtype")
        originals = np.asarray(originals_raw, dtype=np.int64)
        integer_scalars = {}
        for key in ("n_followers", "pool_size", "random_seed"):
            raw_value = np.asarray(values[key])
            if not np.issubdtype(raw_value.dtype, np.integer):
                raise ValueError(f"{key} must have an integer dtype")
            if raw_value.size != 1:
                raise ValueError(f"{key} must be a scalar")
            integer_scalars[key] = int(raw_value.reshape(()))
        n_followers = integer_scalars["n_followers"]
        pool_size = integer_scalars["pool_size"]
        random_seed = integer_scalars["random_seed"]
        source = (
            str(np.asarray(values["source"]).reshape(()))
            if "source" in values.files
            else str(schedule_path)
        )
        oracle_id_value = str(np.asarray(values["oracle_id"]).reshape(()))
        manifest_sha256 = _validate_sha256(
            str(np.asarray(values["oracle_manifest_sha256"]).reshape(())),
            name="oracle_manifest_sha256",
        )
        particle_order_sha256 = _validate_sha256(
            str(np.asarray(values["particle_order_sha256"]).reshape(())),
            name="particle_order_sha256",
        )
        artifact_paths = _validate_oracle_artifact_paths(values["oracle_artifact_paths"])
        particle_star_relative_path = str(
            np.asarray(values["particle_star_relative_path"]).reshape(())
        )
        _validate_oracle_artifact_paths([particle_star_relative_path])
        if particle_star_relative_path not in artifact_paths:
            raise ValueError(
                "particle_star_relative_path must be included in oracle_artifact_paths"
            )
        dispatch_log_relative_path = str(
            np.asarray(values["dispatch_log_relative_path"]).reshape(())
        )
        _validate_oracle_artifact_paths([dispatch_log_relative_path])
        if dispatch_log_relative_path not in artifact_paths:
            raise ValueError(
                "dispatch_log_relative_path must be included in oracle_artifact_paths"
            )
        dispatch_metadata_relative_path = relion_dispatch_metadata_relative_path(
            dispatch_log_relative_path
        )
        if dispatch_metadata_relative_path not in artifact_paths:
            raise ValueError(
                "dispatch metadata sidecar must be included in oracle_artifact_paths"
            )
        expected_oracle_id = relion_oracle_id(
            manifest_sha256=manifest_sha256,
            particle_order_sha256=particle_order_sha256,
        )
        if oracle_id_value != expected_oracle_id:
            raise ValueError(
                "oracle_id is inconsistent with the captured manifest and particle-order hashes"
            )

    if iterations.size < 1 or np.unique(iterations).size != iterations.size:
        raise ValueError("relion_iterations must be a non-empty unique vector")
    if np.any(iterations < 1) or np.any(np.diff(iterations) <= 0):
        raise ValueError("relion_iterations must be positive and strictly increasing")
    if owners.ndim != 2 or owners.shape[0] != iterations.size or owners.shape[1] < 1:
        raise ValueError(
            "owner_by_sorted_position must have shape (n_iterations, n_particles)"
        )
    if originals.shape != owners.shape:
        raise ValueError(
            "original_particle_id_by_sorted_position must match owner_by_sorted_position shape"
        )
    expected_originals = np.arange(owners.shape[1], dtype=np.int64)
    for row_idx in range(originals.shape[0]):
        if not np.array_equal(np.sort(originals[row_idx]), expected_originals):
            raise ValueError(
                "original_particle_id_by_sorted_position must be a permutation in every iteration"
            )
    if n_followers < 1:
        raise ValueError("n_followers must be positive")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if np.any(owners < 0) or np.any(owners >= n_followers):
        raise ValueError("captured follower owners are out of bounds")
    return RelionDispatchSchedule(
        relion_iterations=iterations,
        owner_by_sorted_position=owners,
        original_particle_id_by_sorted_position=originals,
        n_followers=n_followers,
        pool_size=pool_size,
        random_seed=random_seed,
        oracle_id=oracle_id_value,
        oracle_manifest_sha256=manifest_sha256,
        oracle_artifact_paths=artifact_paths,
        particle_order_sha256=particle_order_sha256,
        particle_star_relative_path=particle_star_relative_path,
        dispatch_log_relative_path=dispatch_log_relative_path,
        source=source,
    )


def make_relion_dispatch_schedule_from_chunks(
    *,
    relion_iterations,
    chunk_iterations,
    chunk_first,
    chunk_last,
    chunk_ranks,
    n_particles: int,
    original_particle_id_by_sorted_position,
    n_followers: int,
    pool_size: int,
    random_seed: int,
    oracle_id: str,
    oracle_manifest_sha256: str,
    oracle_artifact_paths,
    particle_order_sha256: str,
    particle_star_relative_path: str,
    dispatch_log_relative_path: str,
    source: str,
) -> RelionDispatchSchedule:
    """Validate leader dispatch chunks and materialize sorted-position owners."""

    iterations = np.asarray(relion_iterations, dtype=np.int64).reshape(-1)
    chunk_iters = np.asarray(chunk_iterations, dtype=np.int64).reshape(-1)
    first = np.asarray(chunk_first, dtype=np.int64).reshape(-1)
    last = np.asarray(chunk_last, dtype=np.int64).reshape(-1)
    ranks = np.asarray(chunk_ranks, dtype=np.int64).reshape(-1)
    originals = np.asarray(original_particle_id_by_sorted_position)
    if not (chunk_iters.shape == first.shape == last.shape == ranks.shape):
        raise ValueError("dispatch chunk columns must have identical shapes")
    if iterations.size < 1 or np.unique(iterations).size != iterations.size:
        raise ValueError("relion_iterations must be non-empty and unique")
    if n_particles < 1 or n_followers < 1 or pool_size < 1:
        raise ValueError("n_particles, n_followers, and pool_size must be positive")
    if not np.issubdtype(originals.dtype, np.integer):
        raise ValueError("original_particle_id_by_sorted_position must have an integer dtype")
    originals = np.asarray(originals, dtype=np.int64)
    if originals.shape != (iterations.size, n_particles):
        raise ValueError(
            "original_particle_id_by_sorted_position must have shape "
            "(n_iterations, n_particles)"
        )
    expected_originals = np.arange(n_particles, dtype=np.int64)
    for row in originals:
        if not np.array_equal(np.sort(row), expected_originals):
            raise ValueError(
                "original_particle_id_by_sorted_position must be a permutation in every iteration"
            )
    if np.any(first < 0) or np.any(last < first) or np.any(last >= n_particles):
        raise ValueError("dispatch chunks contain invalid sorted-position bounds")
    if np.any(last - first + 1 > pool_size):
        raise ValueError("dispatch chunk exceeds the recorded RELION pool size")
    if np.any(ranks < 1) or np.any(ranks > n_followers):
        raise ValueError("dispatch ranks must be RELION follower ranks 1..n_followers")
    artifact_paths = _validate_oracle_artifact_paths(oracle_artifact_paths)
    particle_star_relative_path = _validate_oracle_artifact_paths(
        [particle_star_relative_path]
    )[0]
    if particle_star_relative_path not in artifact_paths:
        raise ValueError("particle_star_relative_path must be included in oracle_artifact_paths")
    dispatch_log_relative_path = _validate_oracle_artifact_paths(
        [dispatch_log_relative_path]
    )[0]
    if dispatch_log_relative_path not in artifact_paths:
        raise ValueError("dispatch_log_relative_path must be included in oracle_artifact_paths")
    dispatch_metadata_relative_path = relion_dispatch_metadata_relative_path(
        dispatch_log_relative_path
    )
    if dispatch_metadata_relative_path not in artifact_paths:
        raise ValueError("dispatch metadata sidecar must be included in oracle_artifact_paths")
    manifest_sha256 = _validate_sha256(
        oracle_manifest_sha256,
        name="oracle_manifest_sha256",
    )
    order_sha256 = _validate_sha256(
        particle_order_sha256,
        name="particle_order_sha256",
    )
    expected_oracle_id = relion_oracle_id(
        manifest_sha256=manifest_sha256,
        particle_order_sha256=order_sha256,
    )
    if str(oracle_id) != expected_oracle_id:
        raise ValueError("oracle_id is inconsistent with manifest and particle-order hashes")

    owners = np.full((iterations.size, n_particles), -1, dtype=np.int64)
    for row_idx, relion_iteration in enumerate(iterations):
        selected = np.flatnonzero(chunk_iters == relion_iteration)
        if selected.size < 1:
            raise ValueError(f"RELION iteration {int(relion_iteration)} has no dispatch chunks")
        for chunk_idx in selected:
            positions = slice(int(first[chunk_idx]), int(last[chunk_idx]) + 1)
            if np.any(owners[row_idx, positions] >= 0):
                raise ValueError(
                    f"RELION iteration {int(relion_iteration)} has overlapping dispatch chunks"
                )
            owners[row_idx, positions] = int(ranks[chunk_idx]) - 1
        missing = np.flatnonzero(owners[row_idx] < 0)
        if missing.size:
            raise ValueError(
                f"RELION iteration {int(relion_iteration)} leaves {missing.size} particles undispatched"
            )

    return RelionDispatchSchedule(
        relion_iterations=iterations.copy(),
        owner_by_sorted_position=owners,
        original_particle_id_by_sorted_position=originals.copy(),
        n_followers=n_followers,
        pool_size=pool_size,
        random_seed=int(random_seed),
        oracle_id=expected_oracle_id,
        oracle_manifest_sha256=manifest_sha256,
        oracle_artifact_paths=artifact_paths,
        particle_order_sha256=order_sha256,
        particle_star_relative_path=particle_star_relative_path,
        dispatch_log_relative_path=dispatch_log_relative_path,
        source=str(source),
    )
