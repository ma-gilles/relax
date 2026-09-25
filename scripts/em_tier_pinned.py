#!/usr/bin/env python
"""Pinned relax outputs for the smoke and medium tiers.

``tests/tiers/pinned_fast_cases.json`` stores, per fast parity case, the relax-vs-RELION FSC
summary of a reference run at a named relax commit: FSC-AUC and minimum in-band shell FSC per
map pair, the FSC curve (4 decimals) and the per-particle Pmax quantiles, one entry per GPU
model. A tier run's ``fsc.json`` (``scripts/em_tier_fsc.py``) is compared case by case only with
the entry pinned on its own GPU model; control and candidate always share the model.

    python scripts/em_tier_pinned.py compare --tier medium --fsc <run>/fsc.json --gpu-model <model> --output <run>/pinned_compare.json
    python scripts/em_tier_pinned.py regenerate --fsc <run>/fsc.json --receipt <run>/RECEIPT.json   # baseline regeneration

The comparison uses the tolerances in ``tests/tiers/fsc_thresholds.json``. It is reported
only, never enforced, until that file records the user's approval (``"approved": true``).
Regeneration rewrites the pinned file and is run only on the user's explicit request.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED = REPO_ROOT / "tests" / "tiers" / "pinned_fast_cases.json"
THRESHOLDS = REPO_ROOT / "tests" / "tiers" / "fsc_thresholds.json"
TIER_CASES = {
    "smoke": ("k1_local_replay", "k1_adaptive_replay", "kclass_replay"),
    "medium": (
        "k1_replay",
        "k1_local_replay",
        "k1_adaptive_replay",
        "kclass_replay",
        "k1_coldstart_standalone",
        "k1_coldstart_relion_seeded_debug",
        "k1_os1_coldstart_standalone",
        "k1_gui60_coldstart_standalone",
        "k1_multioptics_coldstart",
        "k1_perturbreplay",
        "kclass_coldstart",
        "kclass_nonadaptive_replay",
        "kclass_strict_oversample_coldstart",
    ),
}


def _pin(case: dict) -> dict:
    return {
        "pairs": {
            k: {
                "fsc_auc": round(v["fsc_auc"], 7),
                "min_shell_fsc_in_band": round(v["min_shell_fsc_in_band"], 7),
                "fsc": [round(x, 4) for x in v["fsc"]],
            }
            for k, v in case["pairs"].items()
        },
        "assignment": case.get("assignment"),
        "summary": {k: round(v, 7) for k, v in case["summary"].items()},
        "pmax": case.get("pmax", {}),
    }


def compare(tier: str, fsc: dict, pinned: dict, thresholds: dict) -> dict:
    tol = thresholds["pinned_tolerance"]
    rows, failures = {}, []
    for name in TIER_CASES[tier]:
        now = fsc["cases"].get(name)
        ref = pinned["cases"].get(name)
        if now is None:
            failures.append(f"{name}: no current result")
            rows[name] = {"status": "missing"}
            continue
        if ref is None:
            # A newly added case until its first passing run is pinned (regenerate --cases).
            rows[name] = {"status": "not_pinned", "min_fsc_auc": now["summary"]["min_fsc_auc"]}
            continue
        d_auc = now["summary"]["min_fsc_auc"] - ref["summary"]["min_fsc_auc"]
        d_shell = now["summary"]["min_shell_fsc_in_band"] - ref["summary"]["min_shell_fsc_in_band"]
        ok = d_auc >= -tol["min_fsc_auc_drop"] and d_shell >= -tol["min_shell_fsc_drop"]
        rows[name] = {
            "status": "ok" if ok else "fail",
            "min_fsc_auc": now["summary"]["min_fsc_auc"],
            "pinned_min_fsc_auc": ref["summary"]["min_fsc_auc"],
            "delta_min_fsc_auc": d_auc,
            "delta_min_shell_fsc": d_shell,
        }
        if not ok:
            failures.append(f"{name}: min FSC-AUC {d_auc:+.2e}, min shell FSC {d_shell:+.2e} against the pinned run")
    enforced = bool(thresholds.get("approved"))
    return {
        "cases": rows,
        "verdict": {
            "status": "fail" if failures else "pass",
            "enforced": enforced,
            "failures": failures,
            "pinned_sources": sorted({json.dumps(c.get("source"), sort_keys=True) for c in pinned["cases"].values()}),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--tier", choices=sorted(TIER_CASES), required=True)
    c.add_argument("--fsc", type=Path, required=True)
    c.add_argument("--gpu-model", required=True, help="GPU model of the run (the receipt's gpu_model)")
    c.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("regenerate", help="baseline regeneration: pin a reference run (explicit request only)")
    r.add_argument("--fsc", type=Path, required=True)
    r.add_argument("--receipt", type=Path, required=True, help="RECEIPT.json of the run that produced --fsc")
    r.add_argument("--tier", choices=sorted(TIER_CASES), default="medium", help="the cases to pin (a smoke run pins its four)")
    r.add_argument("--cases", nargs="+", help="pin only these cases of the tier (a newly added case); others keep their pins")
    r.add_argument("--item", help="the tier item that produced the single --cases case (default: the case name)")
    r.add_argument("--note", help="JSON object added to each pinned case's source (e.g. pin_mode and observed modes)")
    args = parser.parse_args(argv)
    fsc = json.loads(args.fsc.read_text())
    stored = json.loads(PINNED.read_text()) if PINNED.exists() else {}
    if args.cmd == "compare":
        thresholds = json.loads(THRESHOLDS.read_text())
        pinned = stored.get("models", {}).get(args.gpu_model)
        if pinned is None or None in thresholds["pinned_tolerance"].values():
            reason = f"no run pinned on GPU model {args.gpu_model!r}" if pinned is None else "no tolerance yet"
            result = {"cases": {}, "verdict": {"status": "not_configured", "enforced": False, "failures": [],
                                               "reason": reason, "gpu_model": args.gpu_model}}
            args.output.write_text(json.dumps(result, indent=1) + "\n")
            print(f"pinned comparison: not configured ({reason})")
            return 0
        result = compare(args.tier, fsc, pinned, thresholds)
        result["verdict"]["gpu_model"] = args.gpu_model
        args.output.write_text(json.dumps(result, indent=1) + "\n")
        v = result["verdict"]
        print(f"pinned comparison on {args.gpu_model}: {v['status']} ({'enforced' if v['enforced'] else 'reported only'})")
        for line in v["failures"]:
            print("  " + line)
        return 0
    receipt = json.loads(args.receipt.read_text())
    model = receipt.get("gpu_model") or ""
    names = list(args.cases or TIER_CASES[args.tier])
    unknown = sorted(set(names) - set(TIER_CASES[args.tier]))
    if unknown:
        raise SystemExit(f"refusing to pin: {unknown} are not {args.tier} cases")
    missing = sorted(set(names) - set(fsc["cases"]))
    # The pinned cases' own tier items must pass; a failure elsewhere in the tier (a unit file,
    # say) does not change these outputs, so it is recorded with the pin rather than blocking it.
    items = {i["name"]: i["status"] for i in json.loads(Path(receipt["summary"]).read_text())["items"]}
    if args.item and len(names) != 1:
        raise SystemExit("refusing to pin: --item names the item of exactly one --cases case")
    item_of = {names[0]: args.item} if args.item else {}
    not_passed = sorted(n for n in names if items.get(item_of.get(n, n)) != "pass")
    if missing or not_passed or receipt.get("dirty") or not model or "," in model:
        raise SystemExit(
            f"refusing to pin: missing {missing}, case items not passed {not_passed}, dirty {receipt.get('dirty')}, "
            f"GPU model {model!r} (exactly one model required)"
        )
    other_failures = sorted(n for n, status in items.items() if status != "pass")
    stored["note"] = (
        "Pinned relax-vs-RELION FSC summaries of the fast parity cases, one entry per GPU model; a run is compared "
        "only with the entry of its own model. Regenerate only on explicit request (scripts/em_tier_pinned.py regenerate)."
    )
    source = {k: receipt.get(k) for k in ("sha", "job", "gpu_model", "run_root", "written_utc")}
    source["tier_status"] = receipt["status"]
    source["other_failed_items"] = other_failures
    if args.item:
        source["item"] = args.item
    if args.note:
        source.update(json.loads(args.note))
    entry = stored.setdefault("models", {}).setdefault(model, {"cases": {}})
    for name in names:
        entry["cases"][name] = _pin(fsc["cases"][name]) | {"source": source}
    PINNED.write_text(json.dumps(stored, indent=1, sort_keys=True) + "\n")
    print(f"pinned {len(names)} cases on {model} from {receipt.get('sha')} (job {receipt.get('job')})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
