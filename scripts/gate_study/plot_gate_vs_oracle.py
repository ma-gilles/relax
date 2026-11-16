"""Gate-versus-oracle figure of the gate study: the baseline's applied gate at iteration 3000 per shell against the two
oracle tables, one panel per dataset, for one channel (mean or shared loadings).

Lines: A (current gate; seed mean, the band spans the three seeds), Ocm (code-matched oracle, validated on known truth),
O (the task's oracle, F/(F+1/P)). Line style as well as colour separates the series.

Run: python scripts/gate_study/plot_gate_vs_oracle.py GS_DIR --channel loading -o figure.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TAGS = ("C_n1", "C_n3", "Cshift_n1", "T10_n1", "D10_n1", "D20_n1", "D40_n1", "H1_n3")
COLORS = {"A": "#2a78d6", "Ocm": "#eb6834", "O": "#1baf7a"}  # reference categorical slots 1-3 (validated, light mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gs", type=Path)
    parser.add_argument("--channel", choices=("mean", "loading"), default="loading")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    c = 0 if args.channel == "mean" else 1
    shells = np.arange(1, 32)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    for ax, tag in zip(axes.ravel(), TAGS):
        cm = np.load(args.gs / "oracle" / f"{tag}_cm.npz")["oracle_cm"][:, c]
        task = np.load(args.gs / "oracle" / f"{tag}.npz")["oracle"][:, c]
        final = np.stack([np.load(args.gs / "analysis" / "p0" / f"{tag}_A_s{s}.npz")["gates"][-1, :, c] for s in args.seeds])
        ax.fill_between(shells, final.min(0)[shells], final.max(0)[shells], color=COLORS["A"], alpha=0.18, linewidth=0)
        ax.plot(shells, final.mean(0)[shells], color=COLORS["A"], lw=2, marker="o", ms=3, label="A: current gate (it. 3000)")
        ax.plot(shells, cm[shells], color=COLORS["Ocm"], lw=2, ls="--", label="Ocm: code-matched oracle")
        ax.plot(shells, task[shells], color=COLORS["O"], lw=2, ls=":", label="O: task's oracle")
        ax.set_title(tag.replace("_", " "), fontsize=10)
        ax.grid(color="#e4e3df", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(-0.03, 1.03)
    for ax in axes[1]:
        ax.set_xlabel("Fourier shell (box 64)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{args.channel} gate")
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle(f"VDAM gate per shell at iteration 3000 against the oracles ({args.channel} channel)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)


if __name__ == "__main__":
    main()
