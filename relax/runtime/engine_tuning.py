"""How the scoring and reconstruction engines execute, read once from the environment (code rule 5).

:class:`EngineTuning` holds the engines' execution settings that come from environment variables: memory
budgets, chunk and batch sizes, host or device placement and warm-up. None of them changes a computed value.
:meth:`EngineTuning.from_environ` reads them through ``relax.runtime.env_flags``; the commands call it once
(``RefinementEnvironment.engine``, ``VdamEnvironment.engine``) and the engines receive the record as an argument.
The default instance is the unset environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from relax.runtime.compile_ahead import (
    COMPILE_AHEAD_ENV,
    COMPILE_AHEAD_MAX_ENV,
    CompileAheadConfig,
    parse_compile_ahead_config,
)
from relax.runtime.env_flags import parse_env_flag, read_env_text

DEFAULT_PREPARE_IMAGE_BATCH = 256
# Two image batches ahead by default: K4 100k/256 Class3D, same node, depth 0 against 2:
# 2115 vs 1933 s (job 14515590), 1925 vs 1756 s (14515190) and 1770 vs 1593 s
# (14523034), with equal GT scores (14522527).
DEFAULT_PREFETCH_BATCHES = 2
DEFAULT_FIRSTITER_CC_RECON_COMPLEX_BUDGET = 268_435_456
DEFAULT_EM_BATCH_PROJECTION_FRACTION = 0.20
DEFAULT_BPREF_SYMMETRY_CHUNK_VOXELS = 16 * 1024 * 1024
# One local resident projector call's transient: the 4 GiB the exact local engine budgeted for its fused projection
# matmul (local_batch_planning.EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB), so the two reserved comparable headroom.
DEFAULT_LOCAL_PROJECTION_CALL_MAX_BYTES = 4 * 1024**3
RESIDENT_DIAGNOSTIC_DIR_ENVS = (
    "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
    "RELAX_BPREF_CONTRIBUTION_DUMP_DIR",
    "RELAX_BPREF_MEMBERSHIP_DUMP_DIR",
    "RELAX_PASS2_DUMP_DIR",
    "RELAX_BPREF_EXECUTION_ORDER_LOCAL_FILE",
    "RELAX_VDAM_KCLASS_STATS_DUMP_DIR",
)
RESIDENT_DIAGNOSTIC_FLAG_ENVS = (
    "RELAX_PASS2_DUMP_NORM_RESIDUAL_INPUTS",
    "RELAX_K1_RELION_TRANSLATED_WAVG_NORM",
    "RELAX_SPARSE_PASS2_LOG_CANDIDATE_DENSITY",
    "RELAX_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION",
    "RELAX_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH",
    "RELAX_RELION_X_HALF_BP_FUSED_ATOMICS",
)


@dataclass(frozen=True)
class LargeGridSwitch:
    """A RELION x-half large-grid route: ``forced`` (its flag, when set) decides, else the route is on from
    ``min_voxels`` full-grid voxels."""

    min_voxels: int
    forced: bool | None = None

    def on(self, full_voxels: int) -> bool:
        if self.forced is not None:
            return self.forced
        return int(full_voxels) >= int(self.min_voxels)


@dataclass(frozen=True, kw_only=True)
class XHalfLargeGrid:
    """Where the RELION x-half M-step puts a large grid (relax.reconstruction.half_volume_mstep): ``to_native_half``
    repacks the public accumulators to the native half layout, ``full_host`` expands them on the host and
    ``host_x0`` enforces the x=0 plane (and the point-group finalize) on the host."""

    to_native_half: LargeGridSwitch = LargeGridSwitch(200_000_000)
    full_host: LargeGridSwitch = LargeGridSwitch(100_000_000)
    host_x0: LargeGridSwitch = LargeGridSwitch(200_000_000)


@dataclass(frozen=True, kw_only=True)
class EngineTuning:
    """The engines' execution settings (no effect on any value).

    ``compile_ahead`` (RELAX_EM_COMPILE_AHEAD, RELAX_EM_COMPILE_AHEAD_MAX_PROGRAMS;
    :func:`relax.runtime.compile_ahead.parse_compile_ahead_config`): the resident pass 2's helper-thread warm-up.
    ``pass2_projection_cache`` (RELAX_SPARSE_PASS2_PROJECTION_CACHE, :func:`parse_projection_cache_request`): the
    resident pass 2's projection cache, None for its default (on, except for an operand dump).
    ``resident_operand_max_bytes`` (RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES, :func:`parse_positive_override`):
    the resident operands' byte budget, None for the allocator-derived one. ``prepare_image_batch``
    (RELAX_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH, :func:`parse_positive_override`; default
    :data:`DEFAULT_PREPARE_IMAGE_BATCH`): images per resident operand preparation call (no value depends on it).
    ``resident_diagnostics``: the diagnostic dumps (:data:`RESIDENT_DIAGNOSTIC_DIR_ENVS`, set to a non-blank value)
    and arms (:data:`RESIDENT_DIAGNOSTIC_FLAG_ENVS`, ``env_flags.parse_env_flag``) that are set, dumps first, each in
    its table's order; the resident engines refuse to run with any of them. ``bpref_symmetry_chunk_voxels``
    (RELAX_RELION_BPREF_SYMMETRY_CHUNK_VOXELS, ``int()`` of the text, default
    :data:`DEFAULT_BPREF_SYMMETRY_CHUNK_VOXELS`): voxels per host chunk of the point-group BPref finalizer (its
    caller refuses a value that is not positive).
    ``mstep_block_rows`` (RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS, :func:`parse_mstep_block_rows`): the resident
    M-step's rows per pixel-axis block, None for the budget-derived one.
    ``local_projection_call_max_bytes`` (RELAX_LOCAL_SEARCH_RESIDENT_PROJECTION_CALL_MAX_BYTES,
    :func:`parse_local_projection_call_max_bytes`): bytes one local resident projector call may hold.
    Three pass-2 memory overrides, each :func:`parse_positive_integer` (None: the device-derived budget):
    ``translation_tile_max_bytes`` (RELAX_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES), ``adjoint_block_max_bytes``
    (RELAX_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES) and ``max_projected_rotations``
    (RELAX_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS, rotations per projection call, also for the projection cache build).
    ``prefetch_depth`` (RELAX_EM_PREFETCH_BATCHES, :func:`parse_prefetch_depth`): image batches read ahead on a worker
    thread by the coarse pass and the resident operand preparation; the batches and their order are unchanged.
    ``x_half_large_grid`` (:class:`XHalfLargeGrid`, :func:`read_x_half_large_grid`): the RELION x-half M-step's
    large-grid routes, RELAX_RELION_X_HALF_TO_NATIVE_HALF, _FULL_HOST and _HOST_X0, each with a ``_MIN_VOXELS``
    variable (:func:`parse_large_grid_switch`); host or device and the accumulator layout, never a value.
    ``firstiter_cc_recon_complex_budget`` (RELAX_RELION_FIRSTITER_RECON_COMPLEX_BUDGET,
    :func:`parse_firstiter_cc_recon_complex_budget`): complex values the dense K-class reconstruction batch may
    materialize (``batch x translations x half pixels``), which caps its image batch.
    ``em_batch_projection_fraction`` (RELAX_RELION_EM_BATCH_PROJECTION_FRACTION,
    :func:`parse_em_batch_projection_fraction`): the share of usable device memory the EM batch planner gives the
    projected poses (``batch_planning.estimate_relion_em_batch_sizes``), within its fixed bounds.
    """

    compile_ahead: CompileAheadConfig = field(default_factory=CompileAheadConfig)
    pass2_projection_cache: bool | None = None
    resident_operand_max_bytes: int | None = None
    prepare_image_batch: int = DEFAULT_PREPARE_IMAGE_BATCH
    resident_diagnostics: tuple[str, ...] = ()
    bpref_symmetry_chunk_voxels: int = DEFAULT_BPREF_SYMMETRY_CHUNK_VOXELS

    mstep_block_rows: int | None = None
    local_projection_call_max_bytes: int = DEFAULT_LOCAL_PROJECTION_CALL_MAX_BYTES
    translation_tile_max_bytes: int | None = None
    adjoint_block_max_bytes: int | None = None
    max_projected_rotations: int | None = None
    prefetch_depth: int = DEFAULT_PREFETCH_BATCHES
    x_half_large_grid: XHalfLargeGrid = XHalfLargeGrid()
    firstiter_cc_recon_complex_budget: int = DEFAULT_FIRSTITER_CC_RECON_COMPLEX_BUDGET
    em_batch_projection_fraction: float = DEFAULT_EM_BATCH_PROJECTION_FRACTION

    @classmethod
    def from_environ(cls) -> EngineTuning:
        return cls(
            compile_ahead=parse_compile_ahead_config(
                read_env_text(COMPILE_AHEAD_ENV, ""), read_env_text(COMPILE_AHEAD_MAX_ENV, ""),
            ),
            pass2_projection_cache=parse_projection_cache_request(read_env_text(PROJECTION_CACHE_ENV)),
            resident_operand_max_bytes=parse_positive_override(
                "RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES",
                read_env_text("RELAX_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES", ""),
            ),
            prepare_image_batch=parse_positive_override(
                "RELAX_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH",
                read_env_text("RELAX_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH", ""),
            ) or DEFAULT_PREPARE_IMAGE_BATCH,
            resident_diagnostics=(
                *(name for name in RESIDENT_DIAGNOSTIC_DIR_ENVS if read_env_text(name, "").strip()),
                *(name for name in RESIDENT_DIAGNOSTIC_FLAG_ENVS if parse_env_flag(name, default=False)),
            ),
            bpref_symmetry_chunk_voxels=int(read_env_text(
                "RELAX_RELION_BPREF_SYMMETRY_CHUNK_VOXELS", str(DEFAULT_BPREF_SYMMETRY_CHUNK_VOXELS),
            )),
            mstep_block_rows=parse_mstep_block_rows(read_env_text(MSTEP_BLOCK_ROWS_ENV, "")),
            local_projection_call_max_bytes=parse_local_projection_call_max_bytes(
                read_env_text(LOCAL_PROJECTION_CALL_MAX_BYTES_ENV, ""),
            ),
            translation_tile_max_bytes=parse_positive_integer(
                TRANSLATION_TILE_MAX_BYTES_ENV, read_env_text(TRANSLATION_TILE_MAX_BYTES_ENV),
            ),
            adjoint_block_max_bytes=parse_positive_integer(
                ADJOINT_BLOCK_MAX_BYTES_ENV, read_env_text(ADJOINT_BLOCK_MAX_BYTES_ENV),
            ),
            max_projected_rotations=parse_positive_integer(
                MAX_PROJECTED_ROTATIONS_ENV, read_env_text(MAX_PROJECTED_ROTATIONS_ENV),
            ),
            prefetch_depth=parse_prefetch_depth(read_env_text(PREFETCH_BATCHES_ENV)),
            x_half_large_grid=read_x_half_large_grid(),
            firstiter_cc_recon_complex_budget=parse_firstiter_cc_recon_complex_budget(
                read_env_text(FIRSTITER_CC_RECON_COMPLEX_BUDGET_ENV),
            ),
            em_batch_projection_fraction=parse_em_batch_projection_fraction(read_env_text(EM_BATCH_PROJECTION_FRACTION_ENV)),
        )


def parse_positive_override(name: str, raw: str) -> int | None:
    """A positive integer override (stripped): blank is None; ``int()`` refuses a non-integer, and a value that is
    not positive is refused naming ``name``."""
    raw = raw.strip()
    if not raw:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


PROJECTION_CACHE_ENV = "RELAX_SPARSE_PASS2_PROJECTION_CACHE"


def parse_projection_cache_request(raw: str | None) -> bool | None:
    """RELAX_SPARSE_PASS2_PROJECTION_CACHE: unset, blank or ``auto`` is None; 1/true/yes/on True; 0/false/no/off
    False (any case, stripped); anything else is refused."""
    mode = "auto" if raw is None or raw.strip() == "" else raw.strip().lower()
    if mode == "auto":
        return None
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{PROJECTION_CACHE_ENV} must be 'auto', 'on', or 'off', got {raw!r}")


MSTEP_BLOCK_ROWS_ENV = "RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS"


def parse_mstep_block_rows(raw: str) -> int | None:
    """RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS (stripped): blank is None; ``int()`` refuses a non-integer, and a
    value that is not a positive power of two is refused."""
    raw = raw.strip()
    if not raw:
        return None
    block = int(raw)
    if block <= 0 or block & (block - 1):
        raise ValueError(f"{MSTEP_BLOCK_ROWS_ENV} must be a positive power of two, got {block}")
    return block


LOCAL_PROJECTION_CALL_MAX_BYTES_ENV = "RELAX_LOCAL_SEARCH_RESIDENT_PROJECTION_CALL_MAX_BYTES"


def parse_local_projection_call_max_bytes(raw: str) -> int:
    """RELAX_LOCAL_SEARCH_RESIDENT_PROJECTION_CALL_MAX_BYTES (stripped): blank is
    :data:`DEFAULT_LOCAL_PROJECTION_CALL_MAX_BYTES`; ``int()`` refuses a non-integer; not positive is refused."""
    raw = raw.strip()
    if not raw:
        return DEFAULT_LOCAL_PROJECTION_CALL_MAX_BYTES
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{LOCAL_PROJECTION_CALL_MAX_BYTES_ENV} must be positive, got {raw!r}")
    return value


TRANSLATION_TILE_MAX_BYTES_ENV = "RELAX_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES"
ADJOINT_BLOCK_MAX_BYTES_ENV = "RELAX_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES"
MAX_PROJECTED_ROTATIONS_ENV = "RELAX_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS"


def parse_positive_integer(name: str, raw: str | None) -> int | None:
    """A positive integer override: unset or empty is None; anything that is not a positive integer is refused,
    naming ``name``."""
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


PREFETCH_BATCHES_ENV = "RELAX_EM_PREFETCH_BATCHES"


def parse_prefetch_depth(raw: str | None) -> int:
    """RELAX_EM_PREFETCH_BATCHES (stripped): unset or blank is :data:`DEFAULT_PREFETCH_BATCHES`; a non-negative
    integer is taken (0 reads synchronously); anything else is refused."""
    raw = (str(DEFAULT_PREFETCH_BATCHES) if raw is None else raw).strip()
    if raw == "":
        return DEFAULT_PREFETCH_BATCHES
    try:
        depth = int(raw)
    except ValueError as exc:
        raise ValueError(f"{PREFETCH_BATCHES_ENV} must be a non-negative integer, got {raw!r}") from exc
    if depth < 0:
        raise ValueError(f"{PREFETCH_BATCHES_ENV} must be a non-negative integer, got {raw!r}")
    return depth


X_HALF_TO_NATIVE_HALF_ENV = "RELAX_RELION_X_HALF_TO_NATIVE_HALF"
X_HALF_FULL_HOST_ENV = "RELAX_RELION_X_HALF_FULL_HOST"
X_HALF_HOST_X0_ENV = "RELAX_RELION_X_HALF_HOST_X0"


def parse_large_grid_switch(flag_raw: str | None, min_voxels_raw: str | None, default_min_voxels: int) -> LargeGridSwitch:
    """A large-grid switch from its two variables: a set flag is on unless it is a false token (0/false/no/off, any
    case, stripped); a minimum that is not an integer silently keeps ``default_min_voxels``."""

    forced = None if flag_raw is None else flag_raw.strip().lower() not in {"0", "false", "no", "off"}
    min_voxels = default_min_voxels
    if min_voxels_raw is not None:
        try:
            min_voxels = int(min_voxels_raw)
        except ValueError:
            min_voxels = default_min_voxels
    return LargeGridSwitch(min_voxels, forced)


def read_x_half_large_grid() -> XHalfLargeGrid:
    """The three switches, each with its ``_MIN_VOXELS`` variable."""

    default = XHalfLargeGrid()
    return XHalfLargeGrid(**{
        field_name: parse_large_grid_switch(
            read_env_text(name), read_env_text(f"{name}_MIN_VOXELS"), getattr(default, field_name).min_voxels,
        )
        for field_name, name in (
            ("to_native_half", X_HALF_TO_NATIVE_HALF_ENV),
            ("full_host", X_HALF_FULL_HOST_ENV),
            ("host_x0", X_HALF_HOST_X0_ENV),
        )
    })


FIRSTITER_CC_RECON_COMPLEX_BUDGET_ENV = "RELAX_RELION_FIRSTITER_RECON_COMPLEX_BUDGET"


def parse_firstiter_cc_recon_complex_budget(raw: str | None) -> int:
    """RELAX_RELION_FIRSTITER_RECON_COMPLEX_BUDGET: unset or blank is
    :data:`DEFAULT_FIRSTITER_CC_RECON_COMPLEX_BUDGET`; anything that is not a positive integer is refused."""
    if raw is None or raw.strip() == "":
        return DEFAULT_FIRSTITER_CC_RECON_COMPLEX_BUDGET
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{FIRSTITER_CC_RECON_COMPLEX_BUDGET_ENV} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{FIRSTITER_CC_RECON_COMPLEX_BUDGET_ENV} must be a positive integer")
    return value


EM_BATCH_PROJECTION_FRACTION_ENV = "RELAX_RELION_EM_BATCH_PROJECTION_FRACTION"


def parse_em_batch_projection_fraction(raw: str | None) -> float:
    """RELAX_RELION_EM_BATCH_PROJECTION_FRACTION: unset or blank is :data:`DEFAULT_EM_BATCH_PROJECTION_FRACTION`;
    anything that is not a positive finite float is refused."""
    if raw is None or raw.strip() == "":
        return float(DEFAULT_EM_BATCH_PROJECTION_FRACTION)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{EM_BATCH_PROJECTION_FRACTION_ENV} must be a positive finite float, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{EM_BATCH_PROJECTION_FRACTION_ENV} must be a positive finite float, got {raw!r}")
    return float(value)
