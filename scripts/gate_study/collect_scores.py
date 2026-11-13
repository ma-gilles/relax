"""Paired tables of suite v3.2 headline metrics for the gate study (one table per dataset, arms by seeds).

Rows are named gs_<TAG>_<ARM>_s<SEED> in <work>/scores/<row>/report.json. For each dataset and metric: per seed the
value of each arm, mean and range over seeds, and for each arm against the reference arm the paired differences and the
predeclared verdict (better in every seed pair and mean gain above the reference arm's seed range). D sets use the
distinctive protein's line, as SHARED_RULES asks. Mean-only and true-model references are printed beside the map scores.

Run: python scripts/gate_study/collect_scores.py WORK --arms A O [--reference A] [--seeds 11 12 13] [--markdown OUT.md]
"""

import argparse
import json
from pathlib import Path

import numpy as np

TAGS = {
    "C_n1": "C__realistic__noise1__b128crop64",
    "C_n3": "C__realistic__noise3__b128crop64",
    "Cshift_n1": "C__realistic_shift__noise1__b128crop64",
    "T10_n1": "T10__realistic__noise1__b128crop64",
    "D10_n1": "D10__realistic__noise1__b128crop64",
    "D20_n1": "D20__realistic__noise1__b128crop64",
    "D40_n1": "D40__realistic__noise1__b128crop64",
    "H1_n3": "H1__realistic__noise3__b128crop64",
}
LEAD = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026")
TRUE = {  # true-model rows (true poses and model), as tools/report_tools/row_table.py
    "C__realistic__noise1__b128crop64": LEAD / "metrics_suite_v2_round17_20261103/scores/true_C__realistic__noise1__b128crop64",
    "C__realistic__noise3__b128crop64": LEAD / "metrics_suite_v2_round17_20261103/scores/true_C__realistic__noise3__b128crop64",
    "D10__realistic__noise1__b128crop64": LEAD / "metrics_suite_v2_D10_20261103/scores/true_D10__realistic__noise1__b128crop64",
    "D20__realistic__noise1__b128crop64": LEAD / "metrics_suite_v2_round17_20261103/scores/true_D20__realistic__noise1__b128crop64",
    "D40__realistic__noise1__b128crop64": LEAD / "metrics_suite_v2_round17_20261103/scores/true_D40__realistic__noise1__b128crop64",
    "T10__realistic__noise1__b128crop64": LEAD / "metrics_suite_v2_round17_20261103/scores/true_T10__realistic__noise1__b128crop64",
}
# (name, higher is better)
METRICS = [
    ("decoded", True),
    ("per_image", True),
    ("bp", True),
    ("gain", True),
    ("pose_med", False),
    ("pose_p90", False),
    ("n_struct", True),
    ("ii_z_to_e", True),
    ("states_solved", True),
]


def headline(report: dict) -> dict:
    m1, m2, m3, m4 = report["metric1"], report["metric2"], report["metric3"], report["metric4"]
    distinctive = "distinctive" in m1["decoded_class"]
    key = "distinctive" if distinctive else "macro"
    pose = m2["dec"]["distinctive"] if distinctive else m2["dec"]["overall"]
    out = {
        "decoded": m1["decoded_class"][key],
        "per_image": m1["per_image"].get(key),
        "bp": m1["bp_class"][key],
        "gain": m1["gain"][key],
        "mean_only": m1["references"]["mean_only"][key],
        "pose_med": pose["median_deg"],
        "pose_p90": pose["p90_deg"],
        "n_struct": m3.get("mean"),
        "ii_z_to_e": m4["table"]["macro"]["z_to_E"]["value"] if m4.get("table") else None,
        "per_class_decoded": m1["decoded_class"].get("per_class"),
        "per_class_pose": m2["dec"].get("per_class_median_deg"),
    }
    if report["dataset"].startswith("T10"):
        out["states_solved"] = int(np.sum(np.asarray(out["per_class_pose"]) < 10))  # per-state median pose under 10 deg
    return out


def load_rows(work: Path, tag: str, arms, seeds):
    rows = {}
    for arm in arms:
        for seed in seeds:
            path = work / "scores" / f"gs_{tag}_{arm}_s{seed}" / "report.json"
            if path.is_file():
                rows[(arm, seed)] = headline(json.loads(path.read_text()))
    return rows


def true_reference(dataset):
    path = TRUE.get(dataset)
    if path is None or not (path / "report.json").is_file():
        return None
    report = json.loads((path / "report.json").read_text())
    key = "distinctive" if "distinctive" in report["metric1"]["decoded_class"] else "macro"
    return report["metric1"]["decoded_class"][key], report["metric1"]["bp_class"][key]


def table(work, tag, arms, seeds, reference):
    rows = load_rows(work, tag, arms, seeds)
    if not rows:
        return f"### {tag}: no scores yet\n", {}
    lines = [f"### {tag}", ""]
    any_row = next(iter(rows.values()))
    truth = true_reference(TAGS[tag])
    lines.append(
        f"References: mean only {any_row['mean_only']:.3f}"
        + (f"; true model decoded {truth[0]:.3f}, BP {truth[1]:.3f}" if truth else "; true model: n/a")
        + ("  (D: distinctive protein)" if tag.startswith("D") else "")
    )
    lines.append("")
    lines.append("| metric | " + " | ".join(f"{a} mean [min, max]" for a in arms) + " | " + " | ".join(
        f"{a}-{reference} by seed | verdict" for a in arms if a != reference) + " |")
    lines.append("|---" * (1 + len(arms) + 2 * (len(arms) - 1)) + "|")
    verdicts = {}
    for name, higher in METRICS:
        values = {a: [rows.get((a, s), {}).get(name) for s in seeds] for a in arms}
        if all(v is None for vs in values.values() for v in vs):
            continue
        cells = []
        for a in arms:
            v = np.asarray([x for x in values[a] if x is not None], float)
            cells.append(f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}] (n={len(v)})" if len(v) else "n/a")
        for a in arms:
            if a == reference:
                continue
            pairs = [(x, y) for x, y in zip(values[a], values[reference]) if x is not None and y is not None]
            if not pairs:
                cells += ["n/a", "n/a"]
                continue
            diff = np.asarray([x - y for x, y in pairs])
            ref = np.asarray([y for _, y in pairs])
            sign = 1 if higher else -1
            spread = ref.max() - ref.min()
            if len(pairs) == len(seeds) and np.all(sign * diff > 0) and sign * diff.mean() > spread:
                verdict = "better beyond spread"
            elif len(pairs) == len(seeds) and np.all(sign * diff < 0) and -sign * diff.mean() > spread:
                verdict = "worse beyond spread"
            else:
                verdict = "within spread"
            verdicts[(a, name)] = verdict
            cells += [" / ".join(f"{d:+.3f}" for d in diff), verdict]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines), verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("work", type=Path)
    parser.add_argument("--arms", nargs="+", default=["A", "O"])
    parser.add_argument("--reference", default="A")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--tags", nargs="+", default=list(TAGS))
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    text = []
    for tag in args.tags:
        block, _ = table(args.work, tag, args.arms, args.seeds, args.reference)
        text.append(block)
    out = "\n".join(text)
    print(out)
    if args.markdown:
        args.markdown.write_text(out + "\n")


if __name__ == "__main__":
    main()
