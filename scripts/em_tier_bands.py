#!/usr/bin/env python
"""FSC of the long-tier runs against RELION repeat bands, unmasked and inside the frozen mask.

For each long-tier case found under a run root (K1 50k/256 standalone, K4 50k/256, and the
K1 and K4 100k/256 completions) this compares the final relax map(s) with every RELION run of
the band (the stored reference plus its curated same-command repeats) and with the ground
truth, and the RELION runs with each other:

* GT FSC-AUC of relax and of every RELION run (the band is the RELION spread);
* cross-engine FSC-AUC of relax against every RELION run, and RELION against RELION;
* the same two, masked: both maps multiplied by the dataset's frozen mask
  (``docs/benchmarks/frozen_masks.json``) before the FSC.

K4 classes are matched by the Hungarian assignment on unmasked FSC-AUC (relax and each RELION
run against the RELION reference; each against the GT classes); every class is reported, none is
averaged away. Maps are compared in the internal frame: relax maps through ``load_relax_map``
(RELION's convention; an unlabeled map of an older commit holds the negated array and is read as
such), RELION maps through ``load_relion_volume``, and the GT maps and the mask, which are written
in RECOVAR's ``write_mrc`` convention, through ``load_mrc``.

    python scripts/em_tier_bands.py --run-root <long tier run root> --output bands.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.fsc_metrics import normalized_fsc_auc, shell_fsc  # noqa: E402


@dataclass(frozen=True)
class BandCase:
    output_glob: str  # relax output directory under the run root
    classes: int  # 0 for K1 (merged map), K for classes
    data_set: str
    relion_set: str
    repeat_sets: tuple[str, ...]
    mask_key: str
    relion_final: str  # K1: merged map name; K4: class-map prefix


CASES = {
    "k1_50k256": BandCase(
        "**/k1_long_standalone",
        0,
        "k1_50k256_data",
        "k1_50k256_relion_os0",
        ("k1_50k256_relion_repeats",),
        "noise1_k1_50k256_c1",
        "run_class001.mrc",
    ),
    "k4_50k256": BandCase(
        "**/kclass_long",
        4,
        "k4_50k256_data",
        "k4_50k256_relion_os0",
        ("k4_50k256_relion_repeats",),
        "pdb_k4_50k256_c1",
        "run_it015_class",
    ),
    "k1_100k256": BandCase(
        "**/k1_100k256_recovar",
        0,
        "k1_100k256_data",
        "k1_100k256_relion",
        ("k1_100k256_relion_repeats",),
        "pdb_k1_100k256_c1",
        "run_class001.mrc",
    ),
    "k4_100k256": BandCase(
        "**/k4_100k256_recovar",
        4,
        "k4_100k256_data",
        "k4_100k256_relion",
        ("k4_100k256_dispatch_oracle",),
        "ribosembly_k4_100k256_c1",
        "run_it015_class",
    ),
}


def _load(path: Path, *, relion: bool) -> np.ndarray:
    """A RELION map (``relion``) or a RECOVAR-convention GT map or mask, in the internal frame."""
    from recovar.utils import helpers

    return np.asarray((helpers.load_relion_volume if relion else helpers.load_mrc)(str(path)), dtype=np.float64)


def _load_relax(path: Path) -> np.ndarray:
    from relax.helpers.map_io import load_relax_map

    return np.asarray(load_relax_map(path, legacy_recovar_sign=True), dtype=np.float64)


def _auc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a, b = a * mask, b * mask
    return float(normalized_fsc_auc(shell_fsc(a, b)))


def relion_runs(case: BandCase) -> dict[str, Path]:
    """The reference and every repeat directory holding the final RELION maps."""
    from helpers.em_fixtures import fixture_root

    probe = case.relion_final if not case.classes else f"{case.relion_final}001.mrc"
    runs = {"reference": fixture_root(case.relion_set)}
    for name in case.repeat_sets:
        root = fixture_root(name)
        for d in sorted({p.parent for p in root.rglob(probe)}):
            runs[f"{name}/{d.relative_to(root)}"] = d
    return runs


def _maps(case: BandCase, relax_dir: Path, runs: dict[str, Path]) -> tuple[list, dict[str, list], list]:
    from helpers.em_fixtures import fixture_root

    data = fixture_root(case.data_set)
    if case.classes:
        relax = [_load_relax(relax_dir / f"final_class{c:03d}.mrc") for c in range(1, case.classes + 1)]
        relion = {
            k: [_load(d / f"{case.relion_final}{c:03d}.mrc", relion=True) for c in range(1, case.classes + 1)]
            for k, d in runs.items()
        }
        gt = [_load(data / f"reference_gt_class{c:03d}.mrc", relion=False) for c in range(1, case.classes + 1)]
    else:
        relax = [_load_relax(relax_dir / "final_merged.mrc")]
        relion = {k: [_load(d / case.relion_final, relion=True)] for k, d in runs.items()}
        gt = [_load(data / "reference_gt.mrc", relion=False)]
    return relax, relion, gt


def _match(maps: list, targets: list) -> list[int]:
    if len(maps) == 1:
        return [0]
    scores = np.array([[_auc(m, t) for t in targets] for m in maps])
    _, cols = linear_sum_assignment(-scores)
    return [int(c) for c in cols]


def _per_map(maps: list, targets: list, mask: np.ndarray) -> dict:
    order = _match(maps, targets)
    unmasked = [_auc(m, targets[j]) for m, j in zip(maps, order)]
    masked = [_auc(m, targets[j], mask) for m, j in zip(maps, order)]
    return {
        "assignment": [j + 1 for j in order],
        "fsc_auc": unmasked,
        "masked_fsc_auc": masked,
        "min_fsc_auc": min(unmasked),
        "min_masked_fsc_auc": min(masked),
    }


def score_case(name: str, relax_dir: Path) -> dict:
    from scripts.masked_fsc import load_frozen_mask

    case = CASES[name]
    runs = relion_runs(case)
    relax, relion, gt = _maps(case, relax_dir, runs)
    mask_path, _ = load_frozen_mask(case.mask_key)
    mask = _load(mask_path, relion=False)
    ref = relion["reference"]
    out = {
        "relax_dir": str(relax_dir),
        "relion_runs": {k: str(v) for k, v in runs.items()},
        "mask": str(mask_path),
        "relax_vs_gt": _per_map(relax, gt, mask),
        "relion_vs_gt": {k: _per_map(v, gt, mask) for k, v in relion.items()},
        "relax_vs_relion": {k: _per_map(relax, v, mask) for k, v in relion.items()},
        "relion_vs_relion": {
            f"{a}|{b}": _per_map(relion[a], relion[b], mask) for a, b in itertools.combinations(relion, 2)
        },
    }
    band = [v["min_fsc_auc"] for v in out["relion_vs_gt"].values()]
    band_masked = [v["min_masked_fsc_auc"] for v in out["relion_vs_gt"].values()]
    out["band"] = {
        "relion_gt_fsc_auc": [min(band), max(band)],
        "relax_gt_fsc_auc": out["relax_vs_gt"]["min_fsc_auc"],
        "relion_gt_masked_fsc_auc": [min(band_masked), max(band_masked)],
        "relax_gt_masked_fsc_auc": out["relax_vs_gt"]["min_masked_fsc_auc"],
        "relax_vs_reference_fsc_auc": out["relax_vs_relion"]["reference"]["min_fsc_auc"],
        "relion_vs_relion_fsc_auc": [v["min_fsc_auc"] for v in out["relion_vs_relion"].values()],
        "n_relion_runs": len(runs),
        "reference": ref is not None,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    results, missing = {}, []
    for name, case in CASES.items():
        hits = sorted(
            {p.resolve() for p in args.run_root.glob(case.output_glob) if p.is_dir() and any(p.glob("final_*.mrc"))}
        )
        if len(hits) != 1:
            missing.append(f"{name}: {len(hits)} output directories")
            continue
        results[name] = score_case(name, hits[0])
        b = results[name]["band"]
        print(
            f"{name:11s} GT FSC-AUC relax {b['relax_gt_fsc_auc']:.4f} RELION [{b['relion_gt_fsc_auc'][0]:.4f}, "
            f"{b['relion_gt_fsc_auc'][1]:.4f}]  masked relax {b['relax_gt_masked_fsc_auc']:.4f} RELION "
            f"[{b['relion_gt_masked_fsc_auc'][0]:.4f}, {b['relion_gt_masked_fsc_auc'][1]:.4f}]  "
            f"relax-vs-RELION {b['relax_vs_reference_fsc_auc']:.4f}",
            flush=True,
        )
    args.output.write_text(json.dumps({"cases": results, "missing": missing}, indent=1) + "\n")
    for line in missing:
        print("MISSING " + line, flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
