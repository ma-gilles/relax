#!/usr/bin/env python3
"""Frozen solvent masks and masked FSC reporting for RELION-vs-relax scoring.

Every scored dataset has exactly one frozen mask, generated once with RELION's
own ``relion_mask_create`` and recorded in a ``MASK.json`` beside it. The
registry ``docs/benchmarks/frozen_masks.json`` maps a dataset key to that
record; scoring refuses a dataset without one instead of falling back to an
unmasked number. Masked metrics are reported, never gated.

Masked half-map resolution follows relion_postprocess: the gold-standard FSC
of the masked half maps corrected by phase randomisation, read from
``relion_postprocess`` run with the frozen mask on each arm's half maps.
Cross-engine and ground-truth masked FSCs multiply both maps by the same mask.

Subcommands::

    make-mask    generate a mask from the fixed recipe (RECIPE below)
    adopt-mask   freeze a mask a RELION reference already used, with its command
    verify-mask  regenerate a mask from its MASK.json and compare hashes
    score        masked metrics of one RELION arm and one relax arm

Frames: masks and RELION maps are in RELION's MRC file frame. relax writes
``final_*.mrc`` with ``write_mrc``, whose file array is the negated RELION file
array (``relion_volume_to_recovar`` is ``-transpose`` and ``write_mrc``
transposes back), so a relax map is negated before postprocessing. The two
file frames share one voxel layout, so a mask applies to both; a mask source
is chosen with positive molecular density (RELION's map for real data, the
relax-frame ground truth for the synthetic fixtures, whose RELION-frame maps
carry negative density). The score checks the sign and overlap of the two
merged maps before any cross-engine comparison.

See docs/benchmarks/masked_fsc.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.fsc_metrics import shell_fsc  # noqa: E402

DEFAULT_REGISTRY = REPO / "docs" / "benchmarks" / "frozen_masks.json"
DEFAULT_SCORES = REPO / "docs" / "benchmarks" / "masked_fsc_scores.json"
DEFAULT_SCORES_MARKDOWN = REPO / "docs" / "benchmarks" / "masked_fsc.md"
DEFAULT_SCORECARD = REPO / "docs" / "math" / "em_k1_realdata_science_equivalence_scorecard_v1.json"
RELION_BIN = Path("/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin")
MASK_SCHEMA = "relax-frozen-mask-v1"
SCORE_SCHEMA = "relax-masked-fsc-v1"

# The fixed recipe. The threshold rule and the 15 A low-pass are those of the
# 2026-08-31 RELION masked-FSC evidence for 10073/10345/10097; widths are fixed
# in Angstrom and rounded to pixels, which gives that evidence's 5 px extension
# and 8 px soft edge at 1.31-1.40 A/px.
RECIPE = {
    "lowpass_angstrom": 15.0,
    "threshold_rule": "median + 0.05 * (99.99th percentile - median) of the low-passed source map",
    "robust_positive_percentile": 99.99,
    "threshold_fraction": 0.05,
    "extend_angstrom": 7.0,
    "soft_edge_angstrom": 11.0,
}

# One postprocess protocol for every arm. --force_mask keeps RELION from
# silently reporting an unmasked resolution; --random_seed fixes the phase
# randomisation.
POSTPROCESS_FLAGS = (
    "--force_mask",
    "--skip_fsc_weighting",
    "--low_pass",
    "0",
    "--randomize_at_fsc",
    "0.8",
    "--random_seed",
    "42",
)

# MRC header bytes 224-303 hold label 0, where RELION writes "Relion <version>
# <date time>". The timestamp is the only non-reproducible byte of a RELION
# mask, so generated masks carry a fixed label instead.
LABEL_OFFSET = 224
LABEL_LENGTH = 80
MRC_HEADER_BYTES = 1024
CANONICAL_LABEL = b"relax frozen mask (RELION timestamp label replaced)"

# Cross-engine comparisons require the two merged maps to overlap inside the
# mask; below this they are not in the same frame (e.g. a standalone run that
# converged to another pose basin) and the comparison is reported as null.
FRAME_CORRELATION_MIN = 0.5
FSC_THRESHOLD = 1.0 / 7.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 23):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(path: Path) -> str:
    """SHA-256 of an MRC file after its 1024-byte header (no extended header)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        handle.seek(MRC_HEADER_BYTES)
        while chunk := handle.read(1 << 23):
            digest.update(chunk)
    return digest.hexdigest()


def header_without_label_sha256(path: Path) -> str:
    header = bytearray(Path(path).read_bytes()[:MRC_HEADER_BYTES])
    header[LABEL_OFFSET : LABEL_OFFSET + LABEL_LENGTH] = bytes(LABEL_LENGTH)
    return hashlib.sha256(bytes(header)).hexdigest()


def canonicalize_label(path: Path) -> str:
    """Replace RELION's timestamped label 0 with a fixed one; return the old label."""
    with Path(path).open("r+b") as handle:
        handle.seek(LABEL_OFFSET)
        old = handle.read(LABEL_LENGTH)
        handle.seek(LABEL_OFFSET)
        handle.write(CANONICAL_LABEL.ljust(LABEL_LENGTH, b"\0"))
    return old.rstrip(b"\0").decode("ascii", "replace")


def binary_record(name: str, relion_bin: Path) -> dict[str, str]:
    path = relion_bin / name
    version = subprocess.run([str(path), "--version"], check=True, capture_output=True, text=True).stdout
    return {"path": str(path), "sha256": sha256_file(path), "version": version.strip().splitlines()[0]}


def run_logged(command: list[str], log: Path) -> None:
    with log.open("w") as handle:
        handle.write(f"COMMAND: {shlex.join(command)}\n")
        handle.flush()
        done = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"exit {done.returncode}: {shlex.join(command)}; see {log}")


def read_mrc(path: Path) -> tuple[np.ndarray, float]:
    import mrcfile

    with mrcfile.open(path, permissive=False) as handle:
        data = np.asarray(handle.data, dtype=np.float32).copy()
        voxel = float(handle.voxel_size.x)
    if data.ndim != 3 or len(set(data.shape)) != 1:
        raise ValueError(f"{path}: expected a cubic volume, got {data.shape}")
    return data, voxel


def write_mrc_raw(path: Path, data: np.ndarray, voxel: float) -> None:
    """Write a file-frame array unchanged (no axis transpose)."""
    import mrcfile

    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.asarray(data, dtype=np.float32))
        handle.voxel_size = voxel


def recipe_threshold(lowpass: np.ndarray) -> dict[str, float]:
    """The written threshold rule, evaluated exactly as the 2026-08-31 evidence did."""
    median = float(np.median(lowpass))
    level = float(np.percentile(lowpass, RECIPE["robust_positive_percentile"]))
    if not level > median:
        raise ValueError(f"no positive density: percentile {level} <= median {median}")
    threshold = float(median + RECIPE["threshold_fraction"] * (level - median))
    return {"global_median": median, "robust_positive_level": level, "initial_threshold": threshold}


def recipe_pixels(pixel_size: float) -> tuple[int, int]:
    extend = int(round(RECIPE["extend_angstrom"] / pixel_size))
    soft = int(round(RECIPE["soft_edge_angstrom"] / pixel_size))
    if extend < 1 or soft < 1:
        raise ValueError(f"pixel size {pixel_size} gives a zero-width mask edge")
    return extend, soft


def mask_sanity(mask: np.ndarray, source: np.ndarray) -> dict[str, Any]:
    """Coverage, clipping and edge checks; ``passed`` summarises them."""
    from scipy import ndimage

    if not np.all(np.isfinite(mask)) or mask.min() < -1e-6 or mask.max() > 1.000001:
        raise ValueError("mask values are non-finite or outside [0, 1]")
    n = mask.shape[0]
    support = np.argwhere(mask > 1e-6)
    touches_edge = bool(support.size and (support.min() == 0 or support.max() == n - 1))
    hard = mask >= 0.5
    labels, count = ndimage.label(hard)
    sizes = np.bincount(labels.ravel())[1:]
    positive = np.clip(source.astype(np.float64), 0.0, None)
    # Reported only: unfiltered solvent noise is positive half the time, so
    # this fraction is well below one even for a mask that covers the particle.
    capture = float((positive * mask).sum() / positive.sum()) if positive.sum() > 0 else float("nan")
    # Density above 3 sigma of the source map is molecular; a mask that leaves
    # any of it at weight < 0.5 clips the particle.
    strong = source > float(source.mean() + 3.0 * source.std())
    clipped = float(np.count_nonzero(strong & ~hard) / max(np.count_nonzero(strong), 1))
    edge = (mask > 0.01) & (mask < 0.99)
    stats = {
        "weighted_box_fraction": float(mask.mean(dtype=np.float64)),
        "voxels_ge_0p5": int(np.count_nonzero(hard)),
        "soft_edge_voxels": int(np.count_nonzero(edge)),
        "hard_components": int(count),
        "largest_component_fraction": float(sizes.max() / sizes.sum()) if sizes.size else 0.0,
        "touches_box_edge": touches_edge,
        "positive_density_captured": capture,
        "strong_density_outside_hard_mask": clipped,
    }
    checks = {
        "does_not_touch_box_edge": not touches_edge,
        "box_fraction_le_0p5": stats["weighted_box_fraction"] <= 0.5,
        "has_soft_edge": stats["soft_edge_voxels"] > 0,
        "no_strong_density_clipped": clipped <= 0.001,
    }
    stats["checks"] = checks
    stats["passed"] = all(checks.values())
    return stats


def source_volume(paths: list[Path]) -> tuple[np.ndarray, float]:
    """One map, or the mean of several (multi-class ground truth)."""
    volumes = [read_mrc(path) for path in paths]
    voxels = {round(voxel, 5) for _, voxel in volumes}
    if len(voxels) != 1 or len({v.shape for v, _ in volumes}) != 1:
        raise ValueError("source maps differ in shape or voxel size")
    data = np.mean([v for v, _ in volumes], axis=0, dtype=np.float64).astype(np.float32)
    return data, volumes[0][1]


def generate_mask(record: Mapping[str, Any], out_dir: Path, relion_bin: Path, threads: int) -> Path:
    """Run the recorded generation and return the canonical-label mask path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / record["mask"]["file"]
    generation = record["generation"]
    if generation["kind"] == "recipe":
        sources = [Path(s["path"]) for s in record["source_maps"]]
        pixel = float(record["pixel_size_A"])
        if len(sources) == 1:
            source_path = sources[0]
        else:
            data, voxel = source_volume(sources)
            source_path = out_dir / "source_mean.mrc"
            write_mrc_raw(source_path, data, voxel)
        lowpass_path = out_dir / "source_lowpass.mrc"
        run_logged(
            [
                str(relion_bin / "relion_image_handler"),
                "--i",
                str(source_path),
                "--o",
                str(lowpass_path),
                "--lowpass",
                format(RECIPE["lowpass_angstrom"], "g"),
                "--angpix",
                format(pixel, ".9g"),
            ],
            out_dir / "relion_image_handler.log",
        )
        lowpass, _ = read_mrc(lowpass_path)
        threshold = recipe_threshold(lowpass)["initial_threshold"]
        extend, soft = recipe_pixels(pixel)
        command = [
            str(relion_bin / "relion_mask_create"),
            "--i",
            str(lowpass_path),
            "--o",
            str(mask_path),
            "--ini_threshold",
            format(threshold, ".17g"),
            "--extend_inimask",
            str(extend),
            "--width_soft_edge",
            str(soft),
            "--j",
            str(threads),
        ]
    else:
        command = [str(relion_bin / "relion_mask_create"), *generation["arguments"], "--o", str(mask_path)]
    run_logged(command, out_dir / "relion_mask_create.log")
    if record["mask"]["label"] == "canonical":
        canonicalize_label(mask_path)
    return mask_path


def build_record(args: argparse.Namespace, kind: str) -> dict[str, Any]:
    sources = [Path(p).resolve() for p in args.source_map]
    pixel = float(args.pixel_size)
    record: dict[str, Any] = {
        "schema": MASK_SCHEMA,
        "dataset": args.dataset,
        "symmetry": args.symmetry,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_maps": [{"path": str(p), "sha256": sha256_file(p)} for p in sources],
        "source_role": args.source_role,
        "source_frame": args.source_frame,
        "pixel_size_A": pixel,
        "relion_binaries": {
            name: binary_record(name, args.relion_bin) for name in ("relion_image_handler", "relion_mask_create")
        },
    }
    if kind == "recipe":
        extend, soft = recipe_pixels(pixel)
        record["generation"] = {
            "kind": "recipe",
            "recipe": RECIPE,
            "extend_pixels": extend,
            "soft_edge_pixels": soft,
            "threads": args.threads,
        }
        record["mask"] = {"file": args.mask_file, "label": "canonical"}
    else:
        record["generation"] = {
            "kind": "adopted",
            "arguments": list(args.arguments),
            "adopted_from": {"path": str(Path(args.adopt).resolve()), "sha256": sha256_file(Path(args.adopt))},
            "note": args.note,
        }
        record["mask"] = {"file": args.mask_file, "label": "as_adopted"}
    return record


def finalize_record(record: dict[str, Any], mask_path: Path, out_dir: Path) -> dict[str, Any]:
    mask, voxel = read_mrc(mask_path)
    sources = [Path(s["path"]) for s in record["source_maps"]]
    source, _ = source_volume(sources)
    if mask.shape != source.shape:
        raise ValueError(f"mask {mask.shape} and source {source.shape} differ")
    record["box"] = int(mask.shape[0])
    record["mask"].update(
        {
            "sha256": sha256_file(mask_path),
            "payload_sha256": payload_sha256(mask_path),
            "header_without_label_sha256": header_without_label_sha256(mask_path),
            "header_voxel_size_A": voxel,
        }
    )
    lowpass = out_dir / "source_lowpass.mrc"
    if record["generation"]["kind"] == "recipe":
        record["generation"]["threshold"] = recipe_threshold(read_mrc(lowpass)[0])
    record["sanity"] = mask_sanity(mask, source)
    s = record["sanity"]
    record["sanity"]["summary"] = (
        f"{'PASS' if s['passed'] else 'FAIL'}: box fraction {s['weighted_box_fraction']:.3f}, "
        f"{s['hard_components']} component(s), captures {s['positive_density_captured']:.3f} of positive density, "
        f"strong density outside hard mask {s['strong_density_outside_hard_mask']:.4f}, "
        f"edge-clear {not s['touches_box_edge']}"
    )
    return record


def cmd_make_or_adopt(args: argparse.Namespace, kind: str) -> int:
    out_dir = Path(args.out_dir).resolve()
    if (out_dir / "MASK.json").exists():
        raise SystemExit(f"{out_dir}/MASK.json exists; masks are frozen, never regenerated in place")
    record = build_record(args, kind)
    if kind == "recipe":
        mask_path = generate_mask(record, out_dir, args.relion_bin, args.threads)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / args.mask_file
        mask_path.write_bytes(Path(args.adopt).read_bytes())
    record = finalize_record(record, mask_path, out_dir)
    (out_dir / "MASK.json").write_text(json.dumps(record, indent=1) + "\n")
    print(record["sanity"]["summary"])
    return 0 if record["sanity"]["passed"] else 1


def verify_mask(mask_json: Path, relion_bin: Path, threads: int) -> dict[str, Any]:
    """Regenerate a mask in a scratch directory and compare it to the frozen file."""
    record = json.loads(Path(mask_json).read_text())
    frozen = Path(mask_json).parent / record["mask"]["file"]
    if sha256_file(frozen) != record["mask"]["sha256"]:
        raise ValueError(f"{frozen} no longer matches MASK.json")
    for source in record["source_maps"]:
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"source map changed: {source['path']}")
    with tempfile.TemporaryDirectory(prefix="relax_mask_verify_") as tmp:
        regenerated = generate_mask(record, Path(tmp), relion_bin, threads)
        result = {
            "file_sha256_match": sha256_file(regenerated) == record["mask"]["sha256"],
            "payload_sha256_match": payload_sha256(regenerated) == record["mask"]["payload_sha256"],
            "header_without_label_match": (
                header_without_label_sha256(regenerated) == record["mask"]["header_without_label_sha256"]
            ),
        }
    # An adopted mask keeps RELION's timestamp label byte-for-byte, so only its
    # payload and the rest of its header can reproduce.
    required = ("payload_sha256_match", "header_without_label_match")
    if record["mask"]["label"] == "canonical":
        required += ("file_sha256_match",)
    result["reproduced"] = all(result[key] for key in required)
    result["verified_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.mask_json).resolve()
    result = verify_mask(path, args.relion_bin, args.threads)
    print(json.dumps(result))
    if args.record:
        record = json.loads(path.read_text())
        record["reproducibility"] = result
        path.write_text(json.dumps(record, indent=1) + "\n")
    return 0 if result["reproduced"] else 1


def cmd_register(args: argparse.Namespace) -> int:
    """Add verified masks to the registry; a registered dataset never changes mask."""
    registry = json.loads(args.registry.read_text()) if args.registry.exists() else {"schema": "relax-frozen-mask-registry-v1", "masks": {}}
    for mask_json in args.mask_json:
        path = Path(mask_json).resolve()
        record = json.loads(path.read_text())
        if not record.get("reproducibility", {}).get("reproduced"):
            raise SystemExit(f"{path}: run verify-mask --record first")
        if not record["sanity"]["passed"]:
            raise SystemExit(f"{path}: mask failed its sanity checks")
        entry = {
            "mask_json": str(path),
            "mask_sha256": record["mask"]["sha256"],
            "symmetry": record["symmetry"],
            "box": record["box"],
            "pixel_size_A": record["pixel_size_A"],
            "source_role": record["source_role"],
            "generation": record["generation"]["kind"],
            "sanity": record["sanity"]["summary"],
        }
        old = registry["masks"].get(record["dataset"])
        if old is not None and old["mask_sha256"] != entry["mask_sha256"]:
            raise SystemExit(f"{record['dataset']} already has a different frozen mask")
        registry["masks"][record["dataset"]] = entry
    registry["masks"] = dict(sorted(registry["masks"].items()))
    args.registry.write_text(json.dumps(registry, indent=1) + "\n")
    return 0


def load_frozen_mask(dataset: str, registry: Path = DEFAULT_REGISTRY) -> tuple[Path, dict[str, Any]]:
    """Return the frozen mask of a dataset, failing when none is registered."""
    entries = json.loads(Path(registry).read_text())["masks"]
    if dataset not in entries:
        raise KeyError(f"dataset {dataset!r} has no frozen mask in {registry}; masked scoring refuses to fall back")
    entry = entries[dataset]
    record = json.loads(Path(entry["mask_json"]).read_text())
    mask_path = Path(entry["mask_json"]).parent / record["mask"]["file"]
    if record["mask"]["sha256"] != entry["mask_sha256"] or sha256_file(mask_path) != entry["mask_sha256"]:
        raise ValueError(f"frozen mask of {dataset} does not match the registry sha256")
    return mask_path, record


def read_star_table(path: Path, block: str) -> dict[str, np.ndarray] | dict[str, str]:
    """Minimal STAR reader for relion_postprocess output: one block, loop or pairs."""
    lines = Path(path).read_text().splitlines()
    start = lines.index(f"data_{block}")
    columns: list[str] = []
    rows: list[list[str]] = []
    pairs: dict[str, str] = {}
    for line in lines[start + 1 :]:
        text = line.strip()
        if text.startswith("data_"):
            break
        if not text or text.startswith("#") or text == "loop_":
            continue
        if text.startswith("_"):
            parts = text.split()
            if len(parts) > 1 and not parts[1].startswith("#"):
                pairs[parts[0][1:]] = parts[1]
            else:
                columns.append(parts[0][1:])
        elif columns:
            rows.append(text.split())
    if columns:
        array = np.asarray(rows, dtype=np.float64)
        return {name: array[:, i] for i, name in enumerate(columns)}
    return pairs


def first_sustained_below(curve: np.ndarray, threshold: float, consecutive: int) -> int | None:
    """First non-DC shell starting ``consecutive`` values below threshold (the scorecard rule)."""
    values = np.asarray(curve, dtype=np.float64)
    for shell in range(1, values.size - consecutive + 1):
        if np.all(values[shell : shell + consecutive] < threshold):
            return shell
    return None


def band_auc(curve: np.ndarray, first: int, last: int) -> float:
    band = np.asarray(curve, dtype=np.float64)[first : last + 1]
    if band.size < 2 or not np.all(np.isfinite(band)):
        raise ValueError("invalid FSC band")
    return float(np.trapezoid(band) / (band.size - 1))


def postprocess(
    half1: np.ndarray, half2: np.ndarray, voxel: float, mask_path: Path, out_dir: Path, relion_bin: Path
) -> dict[str, Any]:
    """Run relion_postprocess on two RELION-frame half maps and read its FSC curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, data in (("half1", half1), ("half2", half2)):
        path = out_dir / f"{name}_relion_frame.mrc"
        write_mrc_raw(path, data, voxel)
        paths.append(path)
    command = [
        str(relion_bin / "relion_postprocess"),
        "--i",
        str(paths[0]),
        "--i2",
        str(paths[1]),
        "--o",
        str(out_dir / "postprocess"),
        "--angpix",
        format(voxel, ".9g"),
        "--mask",
        str(mask_path),
        *POSTPROCESS_FLAGS,
    ]
    run_logged(command, out_dir / "relion_postprocess.log")
    star = out_dir / "postprocess.star"
    general = read_star_table(star, "general")
    fsc = read_star_table(star, "fsc")
    for path in paths:
        path.unlink()
    for extra in ("postprocess.mrc", "postprocess_masked.mrc"):
        (out_dir / extra).unlink(missing_ok=True)
    return {
        "command": command,
        "star": str(star),
        "star_sha256": sha256_file(star),
        "final_resolution_A": float(general["rlnFinalResolution"]),
        "curves": {
            "corrected": fsc["rlnFourierShellCorrelationCorrected"],
            "masked": fsc["rlnFourierShellCorrelationMaskedMaps"],
            "unmasked": fsc["rlnFourierShellCorrelationUnmaskedMaps"],
            "phase_randomized_masked": fsc["rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps"],
        },
    }


def unmasked_band(relax_halves, relion_halves, scorecard: Path) -> dict[str, int]:
    """The real-data scorecard's jointly resolved band from unmasked half-map FSCs."""
    from scripts.summarize_em_k1_realdata_science_equivalence import jointly_resolved_band

    thresholds = json.loads(Path(scorecard).read_text())["thresholds"]
    band = jointly_resolved_band(
        shell_fsc(*relax_halves),
        shell_fsc(*relion_halves),
        threshold=float(thresholds["fsc_threshold"]),
        consecutive=int(thresholds["crossing_consecutive_shells"]),
    )
    return {
        "first_shell": int(band["first_shell"]),
        "last_shell": int(band["last_shell"]),
        "definition": "scorecard jointly resolved band (unmasked half-map FSC, three-shell 1/7 crossing)",
    }


def masked_cross(a: np.ndarray, b: np.ndarray, mask: np.ndarray, band: Mapping[str, int]) -> dict[str, Any]:
    curve = shell_fsc(a * mask, b * mask)
    return {"band_auc": band_auc(curve, band["first_shell"], band["last_shell"]), "curve": curve.tolist()}


def frame_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    weight = mask > 0.5
    x = a[weight].astype(np.float64)
    y = b[weight].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    return float(np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y)))


def load_arm(merged: Path | None, half1: Path | None, half2: Path | None, frame: str):
    """Load an arm's maps as RELION file-frame arrays (relax maps are negated)."""
    sign = {"relion": 1.0, "relax": -1.0}[frame]
    out = {}
    for name, path in (("merged", merged), ("half1", half1), ("half2", half2)):
        if path is None:
            out[name] = None
            continue
        data, voxel = read_mrc(path)
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{path}: non-finite values")
        out[name] = sign * data
        out["voxel"] = voxel
    return out


def score(args: argparse.Namespace) -> dict[str, Any]:
    mask_path, record = load_frozen_mask(args.dataset, args.registry)
    mask, _ = read_mrc(mask_path)
    pixel = float(record["pixel_size_A"])
    arms = {
        "relion": load_arm(*(Path(p) if p else None for p in args.relion_maps), frame="relion"),
        "relax": load_arm(*(Path(p) if p else None for p in args.relax_maps), frame="relax"),
    }
    for engine, arm in arms.items():
        for name in ("merged", "half1", "half2"):
            if arm[name] is not None and arm[name].shape != mask.shape:
                raise ValueError(f"{engine} {name} shape {arm[name].shape} != mask {mask.shape}")
    out_dir = Path(args.out_dir).resolve()
    result: dict[str, Any] = {
        "schema": SCORE_SCHEMA,
        "label": args.label,
        "dataset": args.dataset,
        "mask": {"path": str(mask_path), "sha256": record["mask"]["sha256"]},
        "pixel_size_A": pixel,
        "box": record["box"],
        "postprocess_flags": list(POSTPROCESS_FLAGS),
        "maps": {
            engine: {
                name: {"path": str(p), "sha256": sha256_file(Path(p))} if p else None
                for name, p in zip(("merged", "half1", "half2"), paths)
            }
            for engine, paths in (("relion", args.relion_maps), ("relax", args.relax_maps))
        },
        "null_reasons": {},
    }
    halves_present = {e: arms[e]["half1"] is not None and arms[e]["half2"] is not None for e in arms}
    band = None
    if all(halves_present.values()):
        band = unmasked_band(
            (arms["relax"]["half1"], arms["relax"]["half2"]),
            (arms["relion"]["half1"], arms["relion"]["half2"]),
            args.scorecard,
        )
    result["band"] = band
    for engine, arm in arms.items():
        if not halves_present[engine]:
            result[engine] = None
            result["null_reasons"][engine] = "half maps not available"
            continue
        post = postprocess(arm["half1"], arm["half2"], pixel, mask_path, out_dir / engine, args.relion_bin)
        curves = post.pop("curves")
        # rlnFinalResolution is RELION's convention (last shell before the
        # first corrected-FSC value below 0.143); the scorecard's sustained
        # three-shell crossing is recorded beside it.
        sustained = first_sustained_below(curves["corrected"], FSC_THRESHOLD, 3)
        entry = {
            **post,
            "masked_resolution_A": post["final_resolution_A"],
            "masked_sustained_crossing_shell": sustained,
            "masked_sustained_resolution_A": record["box"] * pixel / sustained if sustained else None,
            "curves": {k: v.tolist() for k, v in curves.items()},
        }
        if band is not None:
            entry["masked_corrected_band_auc"] = band_auc(curves["corrected"], band["first_shell"], band["last_shell"])
            entry["unmasked_band_auc"] = band_auc(curves["unmasked"], band["first_shell"], band["last_shell"])
        result[engine] = entry
    if arms["relax"]["merged"] is not None and arms["relion"]["merged"] is not None:
        corr = frame_correlation(arms["relax"]["merged"], arms["relion"]["merged"], mask)
        result["frame_correlation_in_mask"] = corr
        if corr < FRAME_CORRELATION_MIN:
            result["cross_engine"] = None
            result["null_reasons"]["cross_engine"] = (
                f"merged maps correlate {corr:.3f} < {FRAME_CORRELATION_MIN} inside the mask: not in the same frame"
            )
        elif band is None:
            result["cross_engine"] = None
            result["null_reasons"]["cross_engine"] = "no scorecard band without both arms' half maps"
        else:
            result["cross_engine"] = {
                name: masked_cross(arms["relax"][name], arms["relion"][name], mask, band)
                for name in ("merged", "half1", "half2")
            }
    else:
        result["cross_engine"] = None
        result["null_reasons"]["cross_engine"] = "a merged map is missing"
    if args.gt_map:
        gt, _ = read_mrc(Path(args.gt_map))
        result["gt"] = {"path": args.gt_map, "sha256": sha256_file(Path(args.gt_map))}
        for engine, arm in arms.items():
            if arm["merged"] is None or band is None:
                result["gt"][engine] = None
                result["null_reasons"][f"gt.{engine}"] = "merged map or band missing"
                continue
            corr = frame_correlation(arm["merged"], gt, mask)
            if corr < FRAME_CORRELATION_MIN:
                result["gt"][engine] = None
                result["null_reasons"][f"gt.{engine}"] = f"map-GT correlation {corr:.3f} < {FRAME_CORRELATION_MIN} inside the mask: common frame not established"
                continue
            result["gt"][engine] = {"frame_correlation_in_mask": corr, **masked_cross(arm["merged"], gt, mask, band)}
    return result


def cmd_score(args: argparse.Namespace) -> int:
    result = score(args)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir) / "masked_fsc.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    for engine in ("relion", "relax"):
        entry = result[engine]
        print(
            engine,
            "masked" if entry else "null",
            f"{entry['masked_resolution_A']:.3f} A" if entry else result["null_reasons"][engine],
        )
    print("wrote", out)
    return 0


def summarize_score(path: Path) -> dict[str, Any]:
    """Compact scorecard row of one ``masked_fsc.json``."""
    result = json.loads(Path(path).read_text())
    row: dict[str, Any] = {
        "label": result["label"],
        "dataset": result["dataset"],
        "section": "real" if result["dataset"].startswith("empiar") else "synthetic",
        "mask_sha256": result["mask"]["sha256"],
        "band": None if result["band"] is None else [result["band"]["first_shell"], result["band"]["last_shell"]],
        "evidence": {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))},
        "null_reasons": dict(result["null_reasons"]),
    }
    for engine in ("relion", "relax"):
        entry = result[engine]
        row[engine] = None if entry is None else {
            "masked_resolution_A": entry["masked_resolution_A"],
            "masked_sustained_resolution_A": entry["masked_sustained_resolution_A"],
            "masked_corrected_band_auc": entry.get("masked_corrected_band_auc"),
            "unmasked_band_auc": entry.get("unmasked_band_auc"),
        }
    cross = result.get("cross_engine")
    row["cross_engine_masked_band_auc"] = None if cross is None else {k: v["band_auc"] for k, v in cross.items()}
    row["frame_correlation_in_mask"] = result.get("frame_correlation_in_mask")
    gt = result.get("gt")
    row["gt_masked_band_auc"] = None if gt is None else {
        engine: None if gt.get(engine) is None else gt[engine]["band_auc"] for engine in ("relion", "relax")
    }
    return row


def cmd_collect(args: argparse.Namespace) -> int:
    """Add or replace rows (by label) in the masked-score record."""
    table = json.loads(args.scores.read_text()) if args.scores.exists() else {"schema": "relax-masked-fsc-scores-v1", "rows": []}
    rows = {row["label"]: row for row in table["rows"]}
    registry = json.loads(args.registry.read_text())["masks"]
    for path in args.score_json:
        row = summarize_score(Path(path))
        if registry.get(row["dataset"], {}).get("mask_sha256") != row["mask_sha256"]:
            raise SystemExit(f"{path}: mask is not the registered frozen mask of {row['dataset']}")
        rows[row["label"]] = row
    table["rows"] = sorted(rows.values(), key=lambda row: (row["section"] != "real", row["dataset"], row["label"]))
    args.scores.write_text(json.dumps(table, indent=1) + "\n")
    return 0


def _f(value: Any, digits: int) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _masked_cell(entry: Mapping[str, Any]) -> str:
    if not entry:
        return "—"
    sustained = entry["masked_sustained_resolution_A"]
    # No sustained crossing: the corrected masked FSC stays above 1/7 to Nyquist.
    return f"{entry['masked_resolution_A']:.2f} ({'Nyquist' if sustained is None else f'{sustained:.2f}'})"


def render_scores_table(table: Mapping[str, Any], section: str) -> list[str]:
    """Markdown rows of one section; shared by masked_fsc.md and the scorecards."""
    lines = [
        "| Run | Mask | Band | RELION masked (Å) | relax masked (Å) | RELION masked AUC | relax masked AUC "
        "| Unmasked AUC RELION / relax | Cross-engine masked AUC merged / h1 / h2 | GT masked AUC RELION / relax |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (r for r in table["rows"] if r["section"] == section):
        engines = {e: row[e] or {} for e in ("relion", "relax")}
        cross = row["cross_engine_masked_band_auc"]
        gt = row["gt_masked_band_auc"]
        cells = [
            f"`{row['label']}`",
            f"`{row['dataset']}` `{row['mask_sha256'][:12]}`",
            "—" if row["band"] is None else f"{row['band'][0]}-{row['band'][1]}",
            *(
                _masked_cell(engines[e]) for e in ("relion", "relax")
            ),
            _f(engines["relion"].get("masked_corrected_band_auc"), 4),
            _f(engines["relax"].get("masked_corrected_band_auc"), 4),
            f"{_f(engines['relion'].get('unmasked_band_auc'), 4)} / {_f(engines['relax'].get('unmasked_band_auc'), 4)}",
            "—" if cross is None else " / ".join(_f(cross[k], 4) for k in ("merged", "half1", "half2")),
            "—" if gt is None else f"{_f(gt['relion'], 4)} / {_f(gt['relax'], 4)}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_scores_markdown(table: Mapping[str, Any], registry: Mapping[str, Any]) -> str:
    lines = [
        "# Masked FSC with frozen masks",
        "",
        "<!-- Generated by scripts/masked_fsc.py render; do not edit by hand. -->",
        "",
        "Reporting only: no gate reads these values. Every dataset is scored with its one frozen mask",
        "(registry [`frozen_masks.json`](frozen_masks.json)); `scripts/masked_fsc.py score` refuses a dataset without one.",
        "Method, mask recipe and reproduction: [masked FSC method](masked_fsc_method.md).",
        "",
        "Masked resolution is relion_postprocess `rlnFinalResolution` of the phase-randomisation-corrected masked",
        "half-map FSC (RELION's convention); the value in parentheses is the scorecard's sustained three-shell 1/7",
        "crossing of the same curve ('Nyquist' when the curve never falls below it). AUCs are normalised trapezoids",
        "over the band, the real-data scorecard's jointly",
        "resolved band (shells) from the unmasked half-map FSCs of both arms. Cross-engine and GT columns multiply both",
        "maps by the frozen mask; they are null when the maps do not share a frame (masked correlation below "
        f"{FRAME_CORRELATION_MIN}).",
        "",
        "## Frozen masks",
        "",
        "| Dataset | Symmetry | Box | Pixel (Å) | Source | Generation | SHA-256 | Sanity |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for key, entry in registry["masks"].items():
        lines.append(
            f"| `{key}` | {entry['symmetry']} | {entry['box']} | {entry['pixel_size_A']:g} | {entry['source_role']} "
            f"| {entry['generation']} | `{entry['mask_sha256'][:16]}` | {entry['sanity']} |"
        )
    for section, title in (("real", "Real data"), ("synthetic", "Synthetic data")):
        lines += ["", f"## {title}", "", *render_scores_table(table, section)]
    nulls = [(row["label"], field, reason) for row in table["rows"] for field, reason in row["null_reasons"].items()]
    if nulls:
        lines += ["", "## Null values", ""]
        lines += [f"- `{label}` {field}: {reason}." for label, field, reason in nulls]
    return "\n".join(lines) + "\n"


def cmd_render(args: argparse.Namespace) -> int:
    rendered = render_scores_markdown(json.loads(args.scores.read_text()), json.loads(args.registry.read_text()))
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            raise SystemExit(f"{args.output} is stale; run python scripts/masked_fsc.py render")
        return 0
    args.output.write_text(rendered)
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--relion-bin", type=Path, default=RELION_BIN)
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("make-mask", "adopt-mask"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--symmetry", required=True)
        p.add_argument("--source-map", nargs="+", required=True, help="RELION-frame map(s); several are averaged")
        p.add_argument(
            "--source-role", required=True, help="e.g. 'RELION reference final merged map' or 'ground truth'"
        )
        p.add_argument(
            "--source-frame",
            choices=("relion", "relax"),
            required=True,
            help="file frame of the source map(s); recorded only, the voxel layouts coincide (see module docstring)",
        )
        p.add_argument("--pixel-size", type=float, required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--mask-file", default="mask.mrc")
        if name == "adopt-mask":
            p.add_argument("--adopt", required=True, help="mask file the RELION reference used")
            p.add_argument(
                "--arguments",
                nargs=argparse.REMAINDER,
                required=True,
                help="relion_mask_create arguments (without --o) that generated it",
            )
            p.add_argument("--note", default="")
    p = sub.add_parser("register")
    p.add_argument("mask_json", nargs="+")
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p = sub.add_parser("verify-mask")
    p.add_argument("mask_json")
    p.add_argument("--record", action="store_true", help="write the result into MASK.json")
    p = sub.add_parser("collect")
    p.add_argument("score_json", nargs="+")
    p.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p = sub.add_parser("render")
    p.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--output", type=Path, default=DEFAULT_SCORES_MARKDOWN)
    p.add_argument("--check", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("--dataset", required=True)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--relion-maps",
        nargs=3,
        required=True,
        metavar=("MERGED", "HALF1", "HALF2"),
        help="RELION-frame maps; pass '' for a missing one",
    )
    p.add_argument(
        "--relax-maps",
        nargs=3,
        required=True,
        metavar=("MERGED", "HALF1", "HALF2"),
        help="relax final_*.mrc maps; pass '' for a missing one",
    )
    p.add_argument("--gt-map", default=None, help="RELION-frame ground-truth map (synthetic data)")
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    p.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.command == "make-mask":
        return cmd_make_or_adopt(args, "recipe")
    if args.command == "adopt-mask":
        return cmd_make_or_adopt(args, "adopted")
    if args.command == "collect":
        return cmd_collect(args)
    if args.command == "render":
        return cmd_render(args)
    if args.command == "register":
        return cmd_register(args)
    if args.command == "verify-mask":
        return cmd_verify(args)
    return cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
