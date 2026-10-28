"""Stack the per-run plots of plot_ppca_run.py (one folder per arm and seed) into one labeled image per seed.

For each seed and each plot name the arms are stacked top to bottom under a heading with the arm's label and
between-state R^2 (from the run's latent_scores.json). An arm is ROOT:NAME:LABEL, ROOT the Snakemake output root that
holds NAME/seed<seed>/latent_scores.json; PLOTS_DIR holds NAME_seed<seed>/ with the plots. An arm without that seed's
folder is left out of its montage.

Run (CPU, seconds):
  python scripts/montage_ppca_run_plots.py PLOTS_DIR \
      --arm ROOT/round3:long_q1_control:"stages table" --arm ROOT/round4:dvp_1200:"data_vs_prior radius" \
      --plots volume_projections pc_histogram --seeds 11 12 13 --prefix montage_q1
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def heading(text: str, width: int, height: int = 70) -> Image.Image:
    """A white bar of ``width`` x ``height`` pixels carrying ``text``."""
    bar = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(bar).text((20, height // 2), text, fill="#0b0b0b", font=ImageFont.load_default(size=38), anchor="lm")
    return bar


def stack(images: list[Image.Image]) -> Image.Image:
    """The images one above the other on a white canvas as wide as the widest."""
    width = max(im.width for im in images)
    canvas = Image.new("RGB", (width, sum(im.height for im in images)), "white")
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height
    return canvas


def main() -> None:
    """Write ``<prefix>_<plot>_seed<seed>.png`` into PLOTS_DIR for every plot and seed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plots", type=Path)
    parser.add_argument("--arm", action="append", required=True, metavar="ROOT:NAME:LABEL")
    parser.add_argument("--plots", dest="names", nargs="+", default=["volume_projections", "pc_histogram"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--prefix", default="montage")
    args = parser.parse_args()
    arms = [arm.split(":", 2) for arm in args.arm]
    for seed in args.seeds:
        for plot in args.names:
            parts = []
            for root, name, label in arms:
                path = args.plots / f"{name}_seed{seed}" / f"{plot}.png"
                if not path.exists():
                    continue
                r2 = json.loads((Path(root) / name / f"seed{seed}" / "latent_scores.json").read_text())[
                    "between_state_r2"
                ]
                image = Image.open(path).convert("RGB")
                parts += [heading(f"{label}   (seed {seed}, between-state R2 {r2:.3f})", image.width), image]
            if parts:
                out = args.plots / f"{args.prefix}_{plot}_seed{seed}.png"
                stack(parts).save(out)
                print(out)


if __name__ == "__main__":
    main()
