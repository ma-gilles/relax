"""Gate-versus-oracle mechanism table of the gate study from gate traces (scripts/gate_study/gate_trace.py outputs).

For each dataset and arm, mean over seeds of the shell-averaged (shells 1..radius) gap between the arm's applied gate and
a reference table, per iteration window, for the mean and the loading channel; the references are the code-matched oracle
(oracle/<TAG>_cm.npz, oracle_cm; the validated one) and the task's oracle (oracle/<TAG>.npz, oracle). Also the effective
gate's gap, the median sqrt(u), and late per-shell values of the loading gate.

Run: python scripts/gate_study/mechanism_summary.py GS_DIR [--arms A O Ocm] [--markdown OUT.md]
"""

import argparse
from pathlib import Path

import numpy as np

WINDOWS = ((1, 100), (101, 500), (501, 1500), (1501, 2500), (2501, 3000))
TAGS = ("C_n1", "C_n3", "Cshift_n1", "T10_n1", "D10_n1", "D20_n1", "D40_n1", "H1_n3")


def window_gaps(trace, table, key="gates"):
    """[window, channel] mean over the window's iterations of mean over shells 1..radius of trace[key] - table."""
    values = trace[key]
    it, radius = trace["iteration"], trace["radius"].astype(int)
    shells = np.arange(values.shape[1])
    inside = (shells[None, :] >= 1) & (shells[None, :] <= radius[:, None])  # [iteration, shell]
    gap = np.where(inside[..., None], values - table[None, : values.shape[1]], np.nan)
    per_iteration = np.nanmean(gap, axis=1)  # shells without updated voxels (NaN summaries) are left out
    return np.stack([np.nanmean(per_iteration[(it >= lo) & (it <= hi)], axis=0) for lo, hi in WINDOWS])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gs", type=Path)
    parser.add_argument("--arms", nargs="+", default=["A", "O", "Ocm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    lines = ["| dataset | arm | reference | " + " | ".join(f"{lo}-{hi} mean/load" for lo, hi in WINDOWS) + " |",
             "|---" * (3 + len(WINDOWS)) + "|"]
    late = ["", "Late loading gate per shell (iteration 3000, seed mean): arm / Ocm / task oracle", "",
            "| dataset | arm | " + " | ".join(f"sh {s}" for s in (8, 16, 20, 24, 28, 30, 31)) + " | median sqrt(u) late mean/load |",
            "|---" * 10 + "|"]
    for tag in TAGS:
        cm = np.load(args.gs / "oracle" / f"{tag}_cm.npz")["oracle_cm"]
        task = np.load(args.gs / "oracle" / f"{tag}.npz")["oracle"]
        for arm in args.arms:
            traces = [args.gs / "analysis" / "p0" / f"{tag}_{arm}_s{s}.npz" for s in args.seeds]
            traces = [dict(np.load(p)) for p in traces if p.is_file()]
            if not traces:
                continue
            for name, table, key in (("Ocm", cm, "gates"), ("task O", task, "gates"), ("Ocm (effective gate)", cm, "effective_gate_mean")):
                if key not in traces[0]:
                    continue
                g = np.mean([window_gaps(t, table, key) for t in traces], axis=0)
                lines.append(f"| {tag} | {arm} (n={len(traces)}) | {name} | " + " | ".join(f"{a:+.3f}/{b:+.3f}" for a, b in g) + " |")
            final = np.mean([t["gates"][-1] for t in traces], axis=0)
            su = np.mean([np.nanmean(t["sqrt_u_median"][t["iteration"] > 2500][:, 1:31], axis=(0, 1)) for t in traces], axis=0)
            late.append(f"| {tag} | {arm} | " + " | ".join(f"{final[s, 1]:.2f}/{cm[s, 1]:.2f}/{task[s, 1]:.2f}" for s in (8, 16, 20, 24, 28, 30, 31))
                        + f" | {su[0]:.2f}/{su[1]:.2f} |")
    text = "\n".join(lines + late)
    print(text)
    if args.markdown:
        args.markdown.write_text(text + "\n")


if __name__ == "__main__":
    main()
