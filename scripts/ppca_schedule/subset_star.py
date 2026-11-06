"""Write the particle STAR of the images of one true state of a suite dataset.

A suite dataset's training STAR names images 1 to N of one stack, in the order of the labels of its truth file
(``truth.npz``, array ``labels``). The subset keeps the STAR's optics table and the rows of the images whose label is the
state, names them as before, and links the stack beside it, so ``relax ppca_initial_model --i`` trains on that state alone.

Run:
  python scripts/ppca_schedule/subset_star.py TRAINING/particles.star DATASET/evaluation/truth.npz --state 5 -o TRAINING_state5
"""

import argparse
from pathlib import Path

import numpy as np


def subset_star(star: Path, labels: np.ndarray, state: int, output: Path) -> int:
    """Write ``output``/particles.star with the rows of ``star`` whose image has label ``state``; return their count.

    ``labels[i]`` is the label of image ``i + 1`` (the number before the ``@`` of ``_rlnImageName``). The stack the
    rows name is linked into ``output`` under its own name.
    """
    lines = star.read_text().splitlines()
    start = lines.index("data_particles")
    first = next(i for i in range(start, len(lines)) if lines[i].startswith("_rln"))
    names = [line.split()[0] for line in lines[first:] if line.startswith("_rln")]
    body = first + len(names)
    column = names.index("_rlnImageName")
    rows = [line for line in lines[body:] if line.strip()]
    kept = [row for row in rows if labels[int(row.split()[column].split("@")[0]) - 1] == state]
    if not kept:
        raise ValueError(f"No image of {star} has label {state}")
    stacks = {row.split()[column].split("@", 1)[1] for row in kept}
    output.mkdir(parents=True, exist_ok=True)
    for stack in stacks:
        link = output / stack
        link.unlink(missing_ok=True)
        link.symlink_to((star.parent / stack).resolve())
    (output / "particles.star").write_text("\n".join([*lines[:body], *kept]) + "\n")
    return len(kept)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("star", type=Path, help="the training particles.star of the suite dataset")
    parser.add_argument("truth", type=Path, help="the dataset's evaluation/truth.npz (array labels)")
    parser.add_argument("--state", type=int, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True, help="directory for the subset particles.star")
    args = parser.parse_args(argv)
    with np.load(args.truth) as truth:
        labels = truth["labels"]
    print(f"{subset_star(args.star, labels, args.state, args.output)} images of state {args.state} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
