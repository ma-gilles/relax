"""CLI admission and source binding for sealed diagnostic refinement inputs.

Bundle schemas and array validation remain in :mod:`relax.diagnostics.frozen_boundary`.
Runtime adaptation follows ``docs/math/relion_refinement_algorithm.md``.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from recovar.utils.file_hash import sha256_file as _sha256_file

from relax.diagnostics.frozen_boundary import (
    FROZEN_BOUNDARY_FIXED_DIAGNOSTIC_ARM,
    FROZEN_BOUNDARY_FIXED_MATH_ENVIRONMENT_CONTRACT,
    FROZEN_BOUNDARY_NUMERICAL_CLASSIFICATION_SCOPE,
    FROZEN_BOUNDARY_PROVENANCE_VERIFICATION_SCOPE,
    FrozenRefinementBoundary,
    load_frozen_refinement_boundary,
    validate_fixed_diagnostic_boundary_runtime_config,
    verify_fixed_diagnostic_boundary_sources,
)
from relax.diagnostics.parity_provenance import git_head_or_none, git_worktree_provenance
from relax.diagnostics.relion_projector_capture import build_relion_projector_replay_state

logger = logging.getLogger('relax.refinement.full_refinement')


class CapturedProjector(NamedTuple):
    """Installed replay state paired with its validated source identity.

    ``state`` is the same object installed in the numbered replay slot. Keeping
    that borrowed reference here preserves the command's original retention.
    """

    replay_slot: int
    state: dict
    source_dir: Path
    source_manifest: Path
    source_manifest_sha256: str


class BoundaryInputs(NamedTuple):
    """Loaded diagnostic state and the files bound to its fixed arm."""

    boundary: FrozenRefinementBoundary | None
    source_paths: dict[str, Path] | None


def add_frozen_boundary_arguments(parser) -> None:
    """Register the diagnostic bundle and its source-provenance options."""

    parser.add_argument(
        "--frozen-boundary-dir",
        default=None,
        help=(
            "Diagnostic-only sealed one-iteration restart bundle. Requires K=1, "
            "--max_iter 1, --skip-final-iteration, a matching "
            "--init_relion_iteration, and RELION trajectory replay. The bundle "
            "atomically supplies both half maps, tau2, per-half noise, FSC/Pmax, "
            "poses, identities, schedule, and convergence state."
        ),
    )
    parser.add_argument(
        "--frozen-boundary-manifest",
        default=None,
        help=(
            "SHA-256 manifest for --frozen-boundary-dir. Defaults to "
            "FROZEN_BOUNDARY_SHA256SUMS in that directory."
        ),
    )
    parser.add_argument(
        "--require-fixed-diagnostic-boundary",
        action="store_true",
        help=(
            "Require the fixed real-10076 K=1 physical-it2 reconstructed-projector "
            "diagnostic arm. This does not claim identity to RELION's full in-memory "
            "physical iteration. Historical v2 boundaries remain loadable only when "
            "this fixed-arm gate is absent."
        ),
    )
    parser.add_argument(
        "--frozen-boundary-live-capture-manifest",
        default=None,
        help=(
            "Validated live-state manifest named by the fixed schema-v3 diagnostic arm. "
            "Required with --require-fixed-diagnostic-boundary."
        ),
    )
    parser.add_argument(
        "--frozen-boundary-runtime-environment-manifest",
        default=None,
        help=(
            "Sealed declared command/build and numerical-scope manifest for schema-v3; "
            "hardware/toolchain identity remains explicitly unverified."
        ),
    )
    parser.add_argument(
        "--frozen-boundary-recovar-source-manifest",
        default=None,
        help="Exact RECOVAR source-tree manifest sealed by schema-v3.",
    )
    parser.add_argument(
        "--frozen-boundary-relion-command-line",
        default=None,
        help="Declared source RELION command line recorded by schema-v3.",
    )
    parser.add_argument(
        "--frozen-boundary-relion-git-commit",
        default=None,
        help="Declared base RELION git commit recorded by schema-v3.",
    )
    parser.add_argument(
        "--frozen-boundary-relion-build-id",
        default=None,
        help="Declared source RELION binary/build identifier recorded by schema-v3.",
    )
    parser.add_argument(
        "--frozen-boundary-replay-prefix",
        default=None,
        help="Exact RELION replay filename prefix sealed by schema-v3 (for example, run).",
    )


def load_cli_boundary(args) -> BoundaryInputs:
    """Admit the requested diagnostic arm before loading the experiment."""

    frozen_boundary = None
    fixed_diagnostic_source_paths = None
    if args.frozen_boundary_dir is not None:
        if args.sym != "C1":
            raise SystemExit("frozen-boundary schemas v2/v3 currently support C1 only")
        if args.n_classes != 1:
            raise SystemExit("--frozen-boundary-dir is K=1-only")
        if int(args.max_iter) != 1 or not bool(args.skip_final_iteration):
            raise SystemExit(
                "--frozen-boundary-dir requires --max_iter 1 and --skip-final-iteration"
            )
        if args.perturb_replay_relion_dir is None:
            raise SystemExit(
                "--frozen-boundary-dir requires --perturb_replay_relion_dir"
            )
        if args.init_volume is not None or args.init_noise_from_npz is not None:
            raise SystemExit(
                "--frozen-boundary-dir cannot be combined with --init_volume or "
                "--init_noise_from_npz"
            )
        if args.init_previous_best_poses_npz is not None or args.apply_initial_lowpass:
            raise SystemExit(
                "--frozen-boundary-dir cannot be combined with pose overrides or "
                "--apply-initial-lowpass"
            )
        if args.relion_half_sets is None:
            raise SystemExit(
                "--frozen-boundary-dir requires --relion_half_sets so the sealed "
                "half-set source can be verified"
            )
        if args.relion_current_sizes is not None or args.relion_healpix_orders is not None:
            raise SystemExit(
                "--frozen-boundary-dir cannot be combined with unsealed sampling oracles"
            )
        try:
            frozen_boundary = load_frozen_refinement_boundary(
                args.frozen_boundary_dir,
                manifest_path=args.frozen_boundary_manifest,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid frozen refinement boundary: {exc}") from exc
        if args.require_fixed_diagnostic_boundary and not frozen_boundary.fixed_diagnostic_arm:
            raise SystemExit(
                "--require-fixed-diagnostic-boundary rejects historical schema-v2 state"
            )
        if frozen_boundary.fixed_diagnostic_arm and not args.require_fixed_diagnostic_boundary:
            raise SystemExit(
                "schema-v3 frozen state must be acknowledged with "
                "--require-fixed-diagnostic-boundary"
            )
        if frozen_boundary.fixed_diagnostic_arm:
            missing_provenance_flags = [
                flag
                for flag, value in (
                    ("--frozen-boundary-relion-command-line", args.frozen_boundary_relion_command_line),
                    ("--frozen-boundary-relion-git-commit", args.frozen_boundary_relion_git_commit),
                    ("--frozen-boundary-relion-build-id", args.frozen_boundary_relion_build_id),
                    ("--frozen-boundary-replay-prefix", args.frozen_boundary_replay_prefix),
                )
                if value is None
            ]
            if missing_provenance_flags:
                raise SystemExit(
                    "schema-v3 requires explicit source command/build provenance: "
                    + ", ".join(missing_provenance_flags)
                )
        if int(args.init_relion_iteration) != frozen_boundary.completed_relion_iteration:
            raise SystemExit(
                "--init_relion_iteration does not match the sealed frozen boundary: "
                f"cli={args.init_relion_iteration}, "
                f"boundary={frozen_boundary.completed_relion_iteration}"
            )
        if int(args.healpix_order) != frozen_boundary.healpix_order:
            raise SystemExit(
                "--healpix_order does not match the sealed frozen boundary: "
                f"cli={args.healpix_order}, boundary={frozen_boundary.healpix_order}"
            )
        if int(args.adaptive_oversampling) != 1:
            raise SystemExit(
                "--frozen-boundary-dir currently supports only canonical "
                "--adaptive_oversampling 1"
            )
        try:
            fixed_diagnostic_source_paths = (
                _fixed_diagnostic_source_paths(args, frozen_boundary)
                if frozen_boundary.fixed_diagnostic_arm
                else None
            )
            _verify_frozen_boundary_source_hashes(
                frozen_boundary,
                source_star=Path(args.data_dir) / "particles.star",
                relion_half_star=args.relion_half_sets,
                fixed_diagnostic_source_paths=fixed_diagnostic_source_paths,
            )
            if frozen_boundary.fixed_diagnostic_arm:
                _verify_fixed_diagnostic_provenance_manifests(
                    frozen_boundary,
                    fixed_diagnostic_source_paths,
                )
                _validate_fixed_diagnostic_arm_cli(args)
                _validate_fixed_diagnostic_math_environment()
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Invalid frozen refinement boundary source binding: {exc}") from exc
        logger.info(
            "Diagnostic frozen boundary loaded: dir=%s manifest_sha256=%s "
            "boundary_sha256=%s completed_relion_iteration=%d",
            frozen_boundary.source_dir,
            frozen_boundary.source_manifest_sha256,
            frozen_boundary.boundary_sha256,
            frozen_boundary.completed_relion_iteration,
        )
    elif args.frozen_boundary_manifest is not None:
        raise SystemExit("--frozen-boundary-manifest requires --frozen-boundary-dir")
    elif args.require_fixed_diagnostic_boundary or any(
        value is not None
        for value in (
            args.frozen_boundary_live_capture_manifest,
            args.frozen_boundary_runtime_environment_manifest,
            args.frozen_boundary_recovar_source_manifest,
            args.frozen_boundary_relion_command_line,
            args.frozen_boundary_relion_git_commit,
            args.frozen_boundary_relion_build_id,
            args.frozen_boundary_replay_prefix,
        )
    ):
        raise SystemExit(
            "fixed diagnostic boundary flags require --frozen-boundary-dir"
        )

    return BoundaryInputs(frozen_boundary, fixed_diagnostic_source_paths)


def validate_fixed_boundary_runtime(
    boundary: FrozenRefinementBoundary,
    args,
    *,
    dataset,
    effective_max_healpix_order,
    effective_tau2_fudge,
    effective_perturb_seed,
) -> None:
    """Bind resolved run settings to the sealed arm after experiment setup."""

    if effective_perturb_seed is None:
        raise SystemExit("fixed diagnostic boundary v3 requires a deterministic perturb seed")
    try:
        validate_fixed_diagnostic_boundary_runtime_config(
            boundary,
            _fixed_diagnostic_runtime_config(
                args,
                dataset=dataset,
                effective_max_healpix_order=effective_max_healpix_order,
                effective_tau2_fudge=effective_tau2_fudge,
                effective_perturb_seed=effective_perturb_seed,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Fixed diagnostic boundary runtime config mismatch: {exc}") from exc
    logger.info("Fixed diagnostic boundary v3 runtime config verified")


def validate_particle_half_inputs(
    frozen_boundary,
    *,
    image_names,
    half_rows,
    image_shape,
    log,
) -> None:
    """Admit active half identities and shell geometry against a sealed input.

    Comparisons preserve image/source/subset/half/local index relationships.
    Validation-only names and integer vectors expire at this boundary.
    """
    import numpy as np

    live_names_per_half = (
        np.asarray(image_names[half_rows[0]], dtype=str),
        np.asarray(image_names[half_rows[1]], dtype=str),
    )
    for half, (live_names, frozen_names) in enumerate(
        zip(live_names_per_half, frozen_boundary.image_names_per_half, strict=True),
        start=1,
    ):
        if not np.array_equal(live_names, frozen_names):
            raise SystemExit(
                "Frozen-boundary particle identity/order mismatch for "
                f"half {half}: live_rows={live_names.size}, "
                f"frozen_rows={frozen_names.size}"
            )
        frozen_half = half - 1
        expected_source_rows = np.asarray(
            half_rows[0] if frozen_half == 0 else half_rows[1],
            dtype=np.int64,
        )
        five_field_checks = {
            "source_row": (
                expected_source_rows,
                frozen_boundary.source_rows_per_half[frozen_half],
            ),
            "random_subset": (
                np.full(live_names.shape, half, dtype=np.int8),
                frozen_boundary.random_subsets_per_half[frozen_half],
            ),
            "half_index": (
                np.full(live_names.shape, frozen_half, dtype=np.int8),
                frozen_boundary.half_indices_per_half[frozen_half],
            ),
            "half_local_index": (
                np.arange(live_names.size, dtype=np.int64),
                frozen_boundary.half_local_indices_per_half[frozen_half],
            ),
        }
        for field_name, (expected, frozen) in five_field_checks.items():
            if not np.array_equal(expected, frozen):
                raise SystemExit(
                    "Frozen-boundary five-field identity mismatch for "
                    f"half {half} field {field_name}"
                )
    log.info(
        "Frozen-boundary five-field particle identities match the active half layout "
        "exactly: %d + %d",
        live_names_per_half[0].size,
        live_names_per_half[1].size,
    )
    expected_shells = int(image_shape[0] // 2 + 1)
    shell_shapes = {
        "fsc": frozen_boundary.fsc.shape,
        "half1_noise_radial": frozen_boundary.noise_radial_per_half[0].shape,
        "half2_noise_radial": frozen_boundary.noise_radial_per_half[1].shape,
    }
    unexpected_shell_shapes = {
        name: shape
        for name, shape in shell_shapes.items()
        if shape != (expected_shells,)
    }
    if unexpected_shell_shapes:
        raise SystemExit(
            "Frozen-boundary shell arrays do not match the active dataset: "
            f"expected={(expected_shells,)}, got={unexpected_shell_shapes}"
        )


def projector_only_replay_slots(max_iter: int) -> list[dict]:
    """Return projector-only replay slots for a sealed frozen restart.

    A restarted process must not reinterpret its local slot 0 as RELION's
    process-start iteration 0.  In particular, doing so broadcasts half-1
    sigma2_noise over half 2.  Scoring primitives owned by the boundary are
    instead supplied through its sealed initial state; only a separately
    sealed projector may be attached to these slots later.
    """

    count = int(max_iter) + 1
    if count < 2:
        raise ValueError("frozen replay requires at least one numbered iteration slot")
    return [{} for _ in range(count)]


def validate_projector_only_replay_slots(
    replay_slots: list[dict],
    *,
    projector_slot: int | None = None,
) -> None:
    allowed = {"relion_projector_state"}
    for slot_index, slot in enumerate(replay_slots):
        if slot is None:
            raise ValueError(f"frozen replay slot {slot_index} is missing")
        unexpected = sorted(set(slot) - allowed)
        if unexpected:
            raise ValueError(
                f"frozen replay slot {slot_index} is not projector-only: {unexpected}"
            )
    if projector_slot is not None:
        projector_slot = int(projector_slot)
        if projector_slot < 0 or projector_slot >= len(replay_slots):
            raise ValueError(f"frozen projector slot {projector_slot} is out of range")
        projector_slots = [
            index
            for index, slot in enumerate(replay_slots)
            if "relion_projector_state" in slot
        ]
        if projector_slots != [projector_slot]:
            raise ValueError(
                "frozen replay must contain exactly one projector in numbered "
                f"slot {projector_slot}; got {projector_slots}"
            )
        nonempty_other_slots = [
            index
            for index, slot in enumerate(replay_slots)
            if index != projector_slot and slot
        ]
        if nonempty_other_slots:
            raise ValueError(
                "frozen replay unused/final slots must be empty; got "
                f"{nonempty_other_slots}"
            )


def attach_projector_capture(
    replay_iteration_overrides,
    *,
    capture_dir,
    manifest_path,
    capture_iteration,
    init_relion_iteration,
    relion_replay_dir,
    volume_shape,
    n_classes,
    validated_frozen_boundary_iteration=None,
):
    """Attach one sealed live projector to its exact numbered replay slot."""

    from relax.relion.relion_metadata import read_relion_model_metadata

    capture_iteration = int(capture_iteration)
    init_relion_iteration = int(init_relion_iteration)
    frozen_iteration = (
        None
        if validated_frozen_boundary_iteration is None
        else int(validated_frozen_boundary_iteration)
    )
    if init_relion_iteration != 0 and frozen_iteration != init_relion_iteration:
        raise ValueError(
            "captured RELION projector replay currently requires an uninterrupted "
            "cold-start trajectory (init_relion_iteration=0); a later jump would "
            "reapply MPI process-start noise semantics without a validated frozen boundary"
        )
    if frozen_iteration is not None and capture_iteration != frozen_iteration + 1:
        raise ValueError(
            "frozen-boundary projector capture must represent the immediately following "
            f"numbered iteration: boundary={frozen_iteration}, capture={capture_iteration}"
        )
    replay_slot = capture_iteration - init_relion_iteration - 1
    if replay_iteration_overrides is None:
        raise ValueError("captured RELION projector requires trajectory replay overrides")
    if replay_slot < 0 or replay_slot >= len(replay_iteration_overrides):
        raise ValueError(
            "captured RELION projector iteration is outside the configured replay trajectory: "
            f"capture_iteration={capture_iteration}, init_relion_iteration={init_relion_iteration}, "
            f"replay_slots={len(replay_iteration_overrides)}"
        )
    existing = replay_iteration_overrides[replay_slot]
    if existing is None:
        raise ValueError(f"captured RELION projector replay slot {replay_slot} has no state override")
    if "relion_projector_state" in existing:
        raise ValueError(f"captured RELION projector replay slot {replay_slot} is already populated")

    relion_replay_dir = Path(relion_replay_dir).expanduser().resolve()
    model_candidates = (
        relion_replay_dir / f"run_it{capture_iteration:03d}_half1_model.star",
        relion_replay_dir / f"run_it{capture_iteration:03d}_model.star",
    )
    model_path = next((path for path in model_candidates if path.is_file()), None)
    if model_path is None:
        raise ValueError(
            "captured RELION projector has no matching replay control model: "
            + " or ".join(str(path) for path in model_candidates)
        )
    model_metadata = read_relion_model_metadata(model_path)
    current_size = int(model_metadata["current_image_size"])
    if current_size <= 0:
        raise ValueError(f"invalid captured-projector replay current size: {current_size}")

    capture_dir = Path(capture_dir).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    projector_state = build_relion_projector_replay_state(
        capture_dir,
        manifest_path=manifest_path,
        iteration=capture_iteration,
        current_size=current_size,
        volume_shape=tuple(int(value) for value in volume_shape),
        n_classes=int(n_classes),
    )
    replay_iteration_overrides[replay_slot] = {
        **existing,
        "relion_projector_state": projector_state,
    }
    logger.info(
        "STRICT-PARITY: attached captured RELION Projector::data iteration=%d "
        "replay_slot=%d current_size=%d manifest=%s",
        capture_iteration,
        replay_slot,
        current_size,
        projector_state["source_manifest_sha256"],
    )
    return CapturedProjector(
        replay_slot=replay_slot,
        state=projector_state,
        source_dir=capture_dir,
        source_manifest=manifest_path,
        source_manifest_sha256=projector_state["source_manifest_sha256"],
    )


def expand_boundary_noise(noise_radial_per_half, image_shape):
    """Expand sealed radial noise using the float32 scoring dtype.

    Frozen-boundary radial profiles are stored as float64 so their serialized
    RELION shell values remain lossless.  The captured physical boundary,
    however, scores with float32 full-image noise arrays.  With JAX x64
    enabled, an untyped ``jnp.asarray`` silently promotes the replay arrays to
    float64 and violates the boundary's dtype and byte-level identity.
    """
    from recovar.reconstruction import noise as recon_noise

    return [
        recon_noise.make_radial_noise(jnp.asarray(radial, dtype=jnp.float32), image_shape)
        for radial in noise_radial_per_half
    ]


def _verify_frozen_boundary_source_hashes(
    boundary,
    *,
    source_star,
    relion_half_star,
    fixed_diagnostic_source_paths=None,
) -> None:
    """Bind a frozen diagnostic boundary to the live particle source files."""

    live_sources = {
        "source STAR": (Path(source_star), boundary.source_star_sha256),
        "RELION half-set STAR": (Path(relion_half_star), boundary.relion_half_star_sha256),
    }
    for label, (path, expected_sha256) in live_sources.items():
        if not path.is_file():
            raise ValueError(f"frozen-boundary {label} does not exist: {path}")
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"frozen-boundary {label} SHA-256 mismatch: "
                f"expected {expected_sha256}, got {observed_sha256}"
            )
    if getattr(boundary, "fixed_diagnostic_arm", False):
        if fixed_diagnostic_source_paths is None:
            raise ValueError("frozen-boundary v3 requires the complete live source-path table")
        verify_fixed_diagnostic_boundary_sources(boundary, fixed_diagnostic_source_paths)


## Similarly this should be moved elsewhere, and we should make sure we don't have this repeated
## I can imagine we have some IO subdir e.g. that handles this kind of stuff
def _particle_stack_paths_from_star(source_star: Path) -> tuple[Path, ...]:
    """Resolve every image stack referenced by a fixed v3 fixture."""

    import starfile

    source_star = Path(source_star).expanduser().resolve()
    document = starfile.read(source_star, always_dict=True)
    particles = document.get("particles")
    if particles is None or "rlnImageName" not in particles:
        raise ValueError(f"particle STAR lacks rlnImageName: {source_star}")
    stack_names = {
        str(value).partition("@")[2]
        for value in particles["rlnImageName"]
        if str(value).partition("@")[1] == "@"
    }
    if not stack_names:
        raise ValueError(f"particle STAR has no stack-backed rlnImageName rows: {source_star}")
    resolved = set()
    for stack_name in stack_names:
        stack_path = Path(stack_name)
        if not stack_path.is_absolute():
            stack_path = source_star.parent / stack_path
        resolved.add(stack_path.resolve())
    return tuple(sorted(resolved, key=str))


def _fixed_diagnostic_source_paths(args, boundary) -> dict[str, Path]:
    """Map compact v3 source names to the exact files this run will consume."""

    source_star = (Path(args.data_dir) / "particles.star").resolve()
    replay_dir = Path(args.perturb_replay_relion_dir).expanduser().resolve()
    completed = int(boundary.completed_relion_iteration)
    consumer = int(boundary.consumer_relion_iteration)
    if consumer != int(boundary.sampling_state["consumer_relion_iteration"]):
        raise ValueError("fixed diagnostic boundary consumer iteration ownership is inconsistent")
    prefix = str(args.frozen_boundary_replay_prefix)
    if prefix != str(boundary.runtime_config["replay_prefix"]):
        raise ValueError(
            "fixed diagnostic boundary replay prefix mismatch: "
            f"runtime={prefix!r} sealed={boundary.runtime_config['replay_prefix']!r}"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError(f"invalid sealed RELION replay prefix {prefix!r}")
    if args.frozen_boundary_live_capture_manifest is None:
        raise ValueError(
            "fixed diagnostic boundary v3 requires --frozen-boundary-live-capture-manifest"
        )
    if args.frozen_boundary_runtime_environment_manifest is None:
        raise ValueError(
            "fixed diagnostic boundary v3 requires "
            "--frozen-boundary-runtime-environment-manifest"
        )
    if args.frozen_boundary_recovar_source_manifest is None:
        raise ValueError(
            "fixed diagnostic boundary v3 requires --frozen-boundary-recovar-source-manifest"
        )
    source_paths = {
        "particle_star": source_star,
        "relion_half_star": Path(args.relion_half_sets).expanduser().resolve(),
        "completed_data": replay_dir / f"{prefix}_it{completed:03d}_data.star",
        "completed_optimiser": replay_dir / f"{prefix}_it{completed:03d}_optimiser.star",
        "completed_sampling": replay_dir / f"{prefix}_it{completed:03d}_sampling.star",
        "completed_half1_model": replay_dir / f"{prefix}_it{completed:03d}_half1_model.star",
        "completed_half2_model": replay_dir / f"{prefix}_it{completed:03d}_half2_model.star",
        "consumer_validation_optimiser": replay_dir / f"{prefix}_it{consumer:03d}_optimiser.star",
        "consumer_validation_data": replay_dir / f"{prefix}_it{consumer:03d}_data.star",
        "consumer_validation_sampling": replay_dir / f"{prefix}_it{consumer:03d}_sampling.star",
        "consumer_validation_half1_model": replay_dir / f"{prefix}_it{consumer:03d}_half1_model.star",
        "consumer_validation_half2_model": replay_dir / f"{prefix}_it{consumer:03d}_half2_model.star",
        "live_capture_manifest": Path(
            args.frozen_boundary_live_capture_manifest
        ).expanduser().resolve(),
        "runtime_environment_manifest": Path(
            args.frozen_boundary_runtime_environment_manifest
        ).expanduser().resolve(),
        "recovar_source_manifest": Path(
            args.frozen_boundary_recovar_source_manifest
        ).expanduser().resolve(),
    }
    for index, stack_path in enumerate(_particle_stack_paths_from_star(source_star)):
        source_paths[f"particle_stack:{index}"] = stack_path
    import starfile

    for half in (1, 2):
        model_path = source_paths[f"consumer_validation_half{half}_model"]
        document = starfile.read(model_path, always_dict=True)
        classes = document.get("model_classes")
        if classes is None or "rlnReferenceImage" not in classes or len(classes) != 1:
            raise ValueError(
                f"fixed K=1 diagnostic boundary consumer model lacks one reference map: {model_path}"
            )
        map_path = Path(str(classes["rlnReferenceImage"].iloc[0]))
        if not map_path.is_absolute():
            map_path = model_path.parent / map_path
        source_paths[f"consumer_map:half{half}:class1"] = map_path.resolve()
    return source_paths


def _fixed_diagnostic_runtime_config(
    args,
    *,
    dataset,
    effective_max_healpix_order,
    effective_tau2_fudge,
    effective_perturb_seed,
) -> dict[str, str | float | int | bool]:
    """Materialize every v3 runtime control with stable Python scalar types."""

    mask_edge = 5.0 if args._relion_mask_params is None else float(args._relion_mask_params[1])
    particle_diameter = 0.0 if args._relion_mask_params is None else float(args._relion_mask_params[0])
    return {
        "adaptive_oversampling": int(args.adaptive_oversampling),
        "diagnostic_arm_id": FROZEN_BOUNDARY_FIXED_DIAGNOSTIC_ARM,
        "max_iter": int(args.max_iter),
        "skip_final_iteration": bool(args.skip_final_iteration),
        "init_resolution_angstrom": float(args.init_resolution),
        "offset_range_pixels": float(args.offset_range),
        "offset_step_pixels": float(args.offset_step),
        "perturb_factor": float(args.perturb_factor),
        "fsc_threshold": 1.0 / 7.0,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "provenance_verification_scope": FROZEN_BOUNDARY_PROVENANCE_VERIFICATION_SCOPE,
        "numerical_classification_scope": FROZEN_BOUNDARY_NUMERICAL_CLASSIFICATION_SCOPE,
        "auto_local_healpix_order": int(args.auto_local_healpix_order),
        "max_healpix_order": -1 if effective_max_healpix_order is None else int(effective_max_healpix_order),
        "max_significants": int(args.max_significants),
        "particle_diameter_angstrom": float(particle_diameter),
        "width_mask_edge_px": float(mask_edge),
        "tau2_fudge": float(effective_tau2_fudge),
        "low_resol_join_halves_angstrom": 40.0,
        "image_batch_size": int(args.image_batch_size),
        "rotation_block_size": int(args.rotation_block_size),
        "random_seed": int(args.seed),
        "perturb_seed": int(effective_perturb_seed),
        "n_classes": int(args.n_classes),
        "grid_size": int(dataset.grid_size),
        "voxel_size_angstrom": float(dataset.voxel_size),
        "projection_padding_factor": 2,
        "backprojection_padding_factor": 2,
        "do_ctf_correction": True,
        "firstiter_cc": bool(args.firstiter_cc),
        "do_norm_correction": True,
        "do_scale_correction": False,
        "refs_are_ctf_corrected": True,
        "disc_type": "linear_interp",
        "image_fourier_backend": str(args.image_fourier_backend),
        "local_search_translation_prior_mode": "coarse",
        "declared_relion_command_line": str(args.frozen_boundary_relion_command_line),
        "declared_relion_base_git_commit": str(args.frozen_boundary_relion_git_commit),
        "recovar_git_commit": str(git_head_or_none() or "<unknown>"),
        "declared_relion_build_id": str(args.frozen_boundary_relion_build_id),
        "projector_boundary_kind": "reconstructed-projector boundary",
        "replay_prefix": str(args.frozen_boundary_replay_prefix),
    }


_FIXED_ARM_ALLOWED_RECOVAR_ENV = frozenset(
    {
        "RELAX_COMPACT_CANDIDATE_CAPTURE_DIR",
        "RELAX_COMPACT_CANDIDATE_CAPTURE_ITERATION",
        "RECOVAR_CUDA_LIB",
        "RECOVAR_EXPECTED_REPO_ROOT",
        "RELAX_PARITY_TIMING_DIR",
        "RELAX_PROVENANCE_MODULES",
        "RECOVAR_RELION_BIND_BUILD_DIR",
        "RELAX_RELION_BIND_COPY_TO_PACKAGE",
    }
)


def _validate_fixed_diagnostic_arm_cli(args) -> None:
    """Restrict v3 to its one reviewed real-10076 physical-it2 arm."""

    exact_values = {
        "max_iter": 1,
        "skip_final_iteration": True,
        "init_resolution": 30.0,
        "offset_range": 3.0,
        "offset_step": 1.0,
        "perturb_factor": 0.5,
        "adaptive_oversampling": 1,
        "n_classes": 1,
        "firstiter_cc": True,
        "apply_initial_lowpass": False,
        "image_fourier_backend": "relion_cuda",
    }
    for name, expected in exact_values.items():
        observed = getattr(args, name)
        if observed != expected:
            raise ValueError(
                f"fixed diagnostic arm requires --{name.replace('_', '-')}={expected!r}"
            )

    forbidden_options = {
        "final_replay_relion_dir",
        "relion_projector_capture_dir",
        "relion_projector_capture_manifest",
        "relion_projector_capture_iteration",
        "perturb_replay_restart_provenance",
        "relion_dispatch_schedule",
        "relion_follower_scale_replay",
        "init_class_volumes",
        "init_volume",
        "init_previous_best_poses_npz",
        "init_noise_from_npz",
        "relion_init_dir",
        "relion_optimiser",
        "relion_current_sizes",
        "relion_healpix_orders",
    }
    enabled_forbidden = sorted(
        name for name in forbidden_options if getattr(args, name, None) is not None
    )
    if enabled_forbidden:
        raise ValueError(
            "fixed diagnostic arm rejects alternate state/projector/oracle inputs: "
            + ", ".join(enabled_forbidden)
        )
    forbidden_flags = {
        "stop_after_local_search_profile",
        "stop_after_local_search",
        "stop_after_local_search_score_only",
        "diagnostic_single_half",
    }
    enabled_flags = sorted(name for name in forbidden_flags if bool(getattr(args, name)))
    if enabled_flags:
        raise ValueError(
            "fixed diagnostic arm rejects early-stop/single-half modes: "
            + ", ".join(enabled_flags)
        )
    if str(args.perturb_replay_restart_state_iterations).strip():
        raise ValueError("fixed diagnostic arm rejects sampling-restart substitution")
    if args.replay_relion_normcorr is not None:
        raise ValueError("fixed diagnostic arm rejects external norm-correction replay")
    if args.relion_scale_followers is not None:
        raise ValueError("fixed diagnostic arm rejects follower-scale emulation")


def _validate_fixed_diagnostic_math_environment(environ=None) -> None:
    """Reject unsealed numerical/backend switches for the fixed v3 arm."""

    environment = os.environ if environ is None else environ
    forbidden_recovar = sorted(
        name
        for name, value in environment.items()
        if name.startswith(("RECOVAR_", "RELAX_"))
        and str(value) != ""
        and name not in _FIXED_ARM_ALLOWED_RECOVAR_ENV
    )
    if forbidden_recovar:
        raise ValueError(
            "fixed diagnostic arm has unsealed RECOVAR environment overrides: "
            + ", ".join(forbidden_recovar)
        )
    forbidden_numeric_environment = sorted(
        name
        for name in ("JAX_DEFAULT_MATMUL_PRECISION", "NVIDIA_TF32_OVERRIDE")
        if str(environment.get(name, "")) != ""
    )
    # recovar.jax_config installs this exact deterministic robustness default
    # during package import. Reject every caller-supplied extension or
    # replacement while allowing the internal default that is necessarily
    # present by the time this runtime gate executes.
    xla_flags = str(environment.get("XLA_FLAGS", "")).strip()
    if xla_flags not in {"", "--xla_gpu_enable_triton_gemm=false"}:
        forbidden_numeric_environment.append("XLA_FLAGS")
        forbidden_numeric_environment.sort()
    if forbidden_numeric_environment:
        raise ValueError(
            "fixed diagnostic arm has unsealed compiler/precision environment: "
            + ", ".join(forbidden_numeric_environment)
        )
    if not bool(jax.config.x64_enabled):
        raise ValueError("fixed diagnostic arm requires JAX x64 support enabled")


def _verify_fixed_diagnostic_provenance_manifests(boundary, source_paths) -> None:
    """Semantically verify the sealed source/environment manifests."""

    source_manifest = json.loads(
        Path(source_paths["recovar_source_manifest"]).read_text(encoding="utf-8")
    )
    if source_manifest != {
        "schema": "recovar.em.source_manifest.v1",
        "recovar_git_commit": boundary.runtime_config["recovar_git_commit"],
        "worktree_clean": True,
    }:
        raise ValueError("sealed RECOVAR source manifest content mismatch")
    worktree = git_worktree_provenance()
    if (
        worktree["head"] != boundary.runtime_config["recovar_git_commit"]
        or int(worktree["dirty_count"]) != 0
    ):
        raise ValueError(
            "runtime RECOVAR source tree differs from the sealed clean commit: "
            f"head={worktree['head']} dirty_count={worktree['dirty_count']}"
        )

    environment_manifest = json.loads(
        Path(source_paths["runtime_environment_manifest"]).read_text(encoding="utf-8")
    )
    expected_environment = {
        "schema": "recovar.em.runtime_environment.v1",
        "diagnostic_arm_id": boundary.runtime_config["diagnostic_arm_id"],
        "math_environment_contract": FROZEN_BOUNDARY_FIXED_MATH_ENVIRONMENT_CONTRACT,
        "jax_enable_x64": boundary.runtime_config["jax_enable_x64"],
        "provenance_verification_scope": boundary.runtime_config[
            "provenance_verification_scope"
        ],
        "numerical_classification_scope": boundary.runtime_config[
            "numerical_classification_scope"
        ],
        "declared_relion_command_line": boundary.runtime_config["declared_relion_command_line"],
        "declared_relion_base_git_commit": boundary.runtime_config["declared_relion_base_git_commit"],
        "declared_relion_build_id": boundary.runtime_config["declared_relion_build_id"],
        "recovar_git_commit": boundary.runtime_config["recovar_git_commit"],
        "projector_boundary_kind": boundary.runtime_config["projector_boundary_kind"],
    }
    if environment_manifest != expected_environment:
        raise ValueError("sealed runtime command/build/environment manifest content mismatch")


def attach_cli_projector_capture(args, replay_slots, *, volume_shape, frozen_boundary):
    """Attach the captured RELION projector the command names (``--relion-projector-capture-dir``) to its
    replay slot and return it (None without one). A frozen boundary's slots must stay projector-only."""
    captured_projector = None
    if args.relion_projector_capture_dir is not None:
        if args.perturb_replay_relion_dir is None:
            raise SystemExit(
                "--relion-projector-capture-dir requires --perturb_replay_relion_dir"
            )
        if args.relion_projector_capture_iteration is None:
            raise SystemExit(
                "--relion-projector-capture-dir requires "
                "--relion-projector-capture-iteration"
            )
        capture_dir = Path(args.relion_projector_capture_dir).expanduser().resolve()
        capture_manifest = (
            Path(args.relion_projector_capture_manifest).expanduser().resolve()
            if args.relion_projector_capture_manifest is not None
            else capture_dir
            / f"iter{int(args.relion_projector_capture_iteration)}_VALIDATED_SHA256SUMS"
        )
        try:
            captured_projector = attach_projector_capture(
                replay_slots,
                capture_dir=capture_dir,
                manifest_path=capture_manifest,
                capture_iteration=args.relion_projector_capture_iteration,
                init_relion_iteration=args.init_relion_iteration,
                relion_replay_dir=args.perturb_replay_relion_dir,
                volume_shape=volume_shape,
                n_classes=args.n_classes,
                validated_frozen_boundary_iteration=(
                    None
                    if frozen_boundary is None
                    else frozen_boundary.completed_relion_iteration
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid captured RELION projector replay: {exc}") from exc
    elif (
        args.relion_projector_capture_manifest is not None
        or args.relion_projector_capture_iteration is not None
    ):
        raise SystemExit(
            "--relion-projector-capture-manifest/iteration require "
            "--relion-projector-capture-dir"
        )
    if frozen_boundary is not None:
        validate_projector_only_replay_slots(
            replay_slots,
            projector_slot=None if captured_projector is None else captured_projector.replay_slot,
        )
    return captured_projector
