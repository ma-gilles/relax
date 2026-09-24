#!/usr/bin/env python
"""FSC summaries of the fast parity cases against their RELION oracles.

For each case of ``tests/integration/test_em_parity_fast.py`` this compares the relax output
maps with the RELION oracle maps of the same iteration (K1: half 1 with half 1, half 2 with
half 2; K>1: classes matched by the Hungarian assignment on FSC-AUC) and reports, per map
pair, the FSC curve, the FSC-AUC (``scripts/fsc_metrics.normalized_fsc_auc``) and the minimum
shell FSC over shells 1..current_size/2 (the band the iteration refined). It also summarizes
per-particle Pmax (quantiles, and the matched RELION-minus-relax difference).

The same comparison between two RELION runs of a case (a repeat, or a continue from the
same stored state) measures RELION's own spread for that case:

    python scripts/em_tier_fsc.py relax --run-root <pytest basetemp root> --output fsc.json
    python scripts/em_tier_fsc.py relion --case k1_replay --other <RELION dir> --output band.json

Maps are compared in the RECOVAR frame: relax MRCs through ``load_mrc`` and RELION MRCs
through ``load_relion_volume``. FSC is the quality measure; correlation is not reported.
"""

from __future__ import annotations

import argparse
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
class Case:
    output: str  # output directory name the test writes under its tmp_path
    relion_set: str  # manifest fixture set of the oracle
    iteration: int  # RELION iteration the relax output is compared with
    relax_maps: tuple[str, ...]
    classes: int  # 0 for K1 half maps, K for class maps
    relax_relion_frame: bool = False  # run_k_class_parity.py writes its maps in the RELION frame


def _k1(output, relion_set, iteration, prefix=""):
    return Case(output, relion_set, iteration, (f"{prefix}final_half1.mrc", f"{prefix}final_half2.mrc"), 0)


def _k4(output, relion_set):
    return Case(output, relion_set, 3, tuple(f"final_class{c:03d}.mrc" for c in range(1, 5)), 4)


CASES: dict[str, Case] = {
    "k1_replay": _k1("k1_replay", "k1_5k128_relion_os0", 4, "recovar_"),
    "k1_local_replay": _k1("k1_local_replay", "k1_5k128_relion_os0", 7, "recovar_"),
    "k1_adaptive_replay": _k1("k1_adaptive_replay", "k1_5k128_relion_os1", 4, "recovar_"),
    "kclass_replay": Case(
        "kclass_replay", "k2_5k128_relion_os0", 1, ("recovar_class001.mrc", "recovar_class002.mrc"), 2, True
    ),
    "k1_coldstart_standalone": _k1("k1_coldstart_standalone", "k1_5k128_relion_os0", 3),
    "k1_coldstart_relion_seeded_debug": _k1("k1_coldstart_relion_seeded_debug", "k1_5k128_relion_os0", 3),
    "k1_perturbreplay": _k1("k1_perturbreplay", "k1_5k128_relion_os0", 3),
    # Fresh standalone cold start at oversampling 1 (reaches the adaptive / resident pass 2).
    "k1_os1_coldstart_standalone": _k1("k1_os1_coldstart_standalone", "k1_5k128_relion_os1", 3),
    "kclass_coldstart": _k4("kclass_coldstart", "k4_5k128_oracle_h2_os1"),
    "kclass_nonadaptive_replay": _k4("kclass_strict", "k4_5k128_oracle_h1_os1"),
    "kclass_strict_oversample_coldstart": _k4("kclass_strict_os1", "k4_5k128_oracle_h1_os1"),
}


def relion_map_names(case: Case) -> list[str]:
    it = f"run_it{case.iteration:03d}"
    if case.classes:
        return [f"{it}_class{c:03d}.mrc" for c in range(1, case.classes + 1)]
    return [f"{it}_half1_class001.mrc", f"{it}_half2_class001.mrc"]


def _relion_star(relion_dir: Path, case: Case, kind: str) -> Path:
    it = f"run_it{case.iteration:03d}"
    return relion_dir / (f"{it}_half1_{kind}.star" if not case.classes and kind == "model" else f"{it}_{kind}.star")


def relion_current_size(relion_dir: Path, case: Case) -> int:
    """RELION's current image size at the compared iteration (model_general rlnCurrentImageSize)."""
    import starfile

    general = starfile.read(_relion_star(relion_dir, case, "model"), always_dict=True)["model_general"]
    return int(general["rlnCurrentImageSize"])


def _load(path: Path, *, relion: bool) -> np.ndarray:
    from recovar.utils import helpers

    loader = helpers.load_relion_volume if relion else helpers.load_mrc
    return np.asarray(loader(str(path)), dtype=np.float64)


def pair_metrics(lhs: np.ndarray, rhs: np.ndarray, band_shells: int) -> dict:
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)
    if curve.size < 2 or not np.isfinite(curve[1:]).any():
        raise ValueError("FSC has no finite non-DC shell")
    last = min(curve.size, band_shells)
    return {
        "fsc_auc": float(normalized_fsc_auc(curve)),
        "min_shell_fsc_in_band": float(np.nanmin(curve[1:last])),
        "band_shells": int(last),
        "fsc": [round(float(v), 6) for v in curve],
    }


def compare_maps(lhs: list[np.ndarray], rhs: list[np.ndarray], case: Case, band_shells: int) -> dict:
    if not case.classes:
        pairs = {f"half{h + 1}": pair_metrics(lhs[h], rhs[h], band_shells) for h in range(2)}
        return {"pairs": pairs, "assignment": None}
    k = case.classes
    auc = np.array([[normalized_fsc_auc(shell_fsc(lhs[i], rhs[j])) for j in range(k)] for i in range(k)])
    rows, cols = linear_sum_assignment(-auc)
    pairs = {f"class{i + 1}->relion{j + 1}": pair_metrics(lhs[i], rhs[j], band_shells) for i, j in zip(rows, cols)}
    return {"pairs": pairs, "assignment": [int(c) + 1 for c in cols]}


def _summary(result: dict) -> dict:
    aucs = [p["fsc_auc"] for p in result["pairs"].values()]
    mins = [p["min_shell_fsc_in_band"] for p in result["pairs"].values()]
    return {
        "min_fsc_auc": float(min(aucs)),
        "mean_fsc_auc": float(np.mean(aucs)),
        "min_shell_fsc_in_band": float(min(mins)),
    }


def _quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    q = np.quantile(values, [0.1, 0.5, 0.9]) if values.size else [np.nan] * 3
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "p10": float(q[0]),
        "p50": float(q[1]),
        "p90": float(q[2]),
    }


def relion_pmax(relion_dir: Path, case: Case) -> dict[str, float]:
    import starfile

    table = starfile.read(_relion_star(relion_dir, case, "data"), always_dict=True)["particles"]
    return dict(zip(table["rlnImageName"].astype(str), table["rlnMaxValueProbDistribution"].astype(float)))


def relax_pmax(out_dir: Path) -> np.ndarray | None:
    """Per-image Pmax of the last relax iteration in input-image order.

    Runs that record ``pmax_per_image_by_image_iter_*`` store input order directly. The
    replay script stores ``pmax_per_image_iter_*`` in half order (half 1's images, then half
    2's, as listed in ``half1_indices`` / ``half2_indices``); it is scattered back here.
    """
    npz_path = out_dir / "refinement_results.npz"
    if not npz_path.exists():
        return None
    z = np.load(npz_path, allow_pickle=True)
    keys = sorted(k for k in z.files if k.startswith("pmax_per_image_by_image_iter_"))
    if keys:
        return np.asarray(z[keys[-1]], dtype=np.float64)
    keys = sorted(k for k in z.files if k.startswith("pmax_per_image_iter_"))
    if not keys or "half1_indices" not in z.files:
        return None
    half_order = np.concatenate([np.asarray(z["half1_indices"]), np.asarray(z["half2_indices"])]).astype(np.int64)
    values = np.asarray(z[keys[-1]], dtype=np.float64)
    if values.size != half_order.size or np.unique(half_order).size != half_order.size:
        raise ValueError(f"{npz_path}: Pmax and half-set indices do not describe one image order")
    by_image = np.empty_like(values)
    by_image[half_order] = values
    return by_image


def pmax_summary(relax: np.ndarray | None, relion: dict[str, float], data_star: Path | None) -> dict:
    out = {"relion": _quantiles(np.fromiter(relion.values(), dtype=np.float64))}
    if relax is None:
        return out
    out["relax"] = _quantiles(relax)
    if data_star is not None and relax.size == len(relion):
        import starfile

        names = starfile.read(data_star, always_dict=True)["particles"]["rlnImageName"].astype(str)
        diff = relax - np.asarray([relion[n] for n in names])
        out["relax_minus_relion"] = _quantiles(diff) | {"mean_abs": float(np.mean(np.abs(diff)))}
    return out


def _data_star(case: Case) -> Path:
    from helpers.em_fixtures import fixture_root

    return fixture_root(case.relion_set.split("_relion")[0].split("_oracle")[0] + "_data") / "particles.star"


def score_relax_case(name: str, out_dir: Path) -> dict:
    from helpers.em_fixtures import fixture_dir

    case = CASES[name]
    relion_dir = fixture_dir(case.relion_set)
    band = relion_current_size(relion_dir, case) // 2
    lhs = [_load(out_dir / m, relion=case.relax_relion_frame) for m in case.relax_maps]
    rhs = [_load(relion_dir / m, relion=True) for m in relion_map_names(case)]
    result = compare_maps(lhs, rhs, case, band)
    result["summary"] = _summary(result)
    arrays = out_dir / "k_class_parity_arrays.npz"
    if arrays.exists():  # the K-class replay stores both engines' Pmax in one particle order
        z = np.load(arrays)
        diff = np.asarray(z["recovar_pmax"], dtype=np.float64) - np.asarray(z["relion_pmax"], dtype=np.float64)
        result["pmax"] = {"relax": _quantiles(z["recovar_pmax"]), "relion": _quantiles(z["relion_pmax"]),
                          "relax_minus_relion": _quantiles(diff) | {"mean_abs": float(np.mean(np.abs(diff)))}}
    else:
        result["pmax"] = pmax_summary(relax_pmax(out_dir), relion_pmax(relion_dir, case), _data_star(case))
    result["relax_dir"] = str(out_dir)
    result["relion_dir"] = str(relion_dir)
    return result


def score_relion_pair(name: str, other: Path) -> dict:
    """The same case metrics between the stored oracle and another RELION run of it."""
    from helpers.em_fixtures import fixture_dir

    case = CASES[name]
    relion_dir = fixture_dir(case.relion_set)
    band = relion_current_size(relion_dir, case) // 2
    names = relion_map_names(case)
    lhs = [_load(other / m, relion=True) for m in names]
    rhs = [_load(relion_dir / m, relion=True) for m in names]
    result = compare_maps(lhs, rhs, case, band)
    result["summary"] = _summary(result)
    other_pmax = relion_pmax(other, case)
    oracle_pmax = relion_pmax(relion_dir, case)
    common = sorted(set(other_pmax) & set(oracle_pmax))
    diff = np.asarray([other_pmax[n] - oracle_pmax[n] for n in common])
    result["pmax"] = {
        "other": _quantiles(np.fromiter(other_pmax.values(), dtype=np.float64)),
        "relion": _quantiles(np.fromiter(oracle_pmax.values(), dtype=np.float64)),
        "other_minus_relion": _quantiles(diff) | {"mean_abs": float(np.mean(np.abs(diff)))},
    }
    result["other_dir"] = str(other)
    result["relion_dir"] = str(relion_dir)
    return result


def find_case_dirs(run_root: Path) -> dict[str, Path]:
    """Map each case to its output directory under a pytest basetemp; duplicates are an error."""
    found: dict[str, Path] = {}
    for name, case in CASES.items():
        hits = {p.resolve() for p in run_root.rglob(case.output) if p.is_dir() and (p / case.relax_maps[0]).exists()}
        if len(hits) > 1:
            raise ValueError(f"{name}: several output directories under {run_root}: {sorted(hits)}")
        if hits:
            found[name] = hits.pop()
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    relax = sub.add_parser("relax", help="score every case found under a fast-tier run root")
    relax.add_argument("--run-root", type=Path, required=True)
    relax.add_argument("--require", nargs="*", default=[], help="cases that must be present")
    relax.add_argument("--output", type=Path, required=True)
    relion = sub.add_parser("relion", help="score another RELION run of one case against the oracle")
    relion.add_argument("--case", choices=sorted(CASES), required=True)
    relion.add_argument("--other", type=Path, required=True)
    relion.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "relax":
        dirs = find_case_dirs(args.run_root)
        missing = sorted(set(args.require) - set(dirs))
        results = {name: score_relax_case(name, path) for name, path in sorted(dirs.items())}
        payload = {"run_root": str(args.run_root), "missing_required": missing, "cases": results}
    else:
        results = {args.case: score_relion_pair(args.case, args.other)}
        payload = {"cases": results}
        missing = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    for name, res in results.items():
        s = res["summary"]
        print(
            f"{name:38s} min FSC-AUC {s['min_fsc_auc']:.6f}  mean {s['mean_fsc_auc']:.6f}  "
            f"min shell FSC {s['min_shell_fsc_in_band']:.6f}",
            flush=True,
        )
    if missing:
        print(f"MISSING required cases: {missing}", flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
