# RELAX! It's [RELION](https://github.com/3dem/relion) ... in JAX.

RELAX is meant to be a playground for experimenting with cryo-EM algorithms. It reproduces RELION's
algorithms in Python/JAX, with CUDA backends, and is aimed at method developers. Anyone is encouraged
to copy, fork, or modify the code for their own purposes. Just keep the
[GPL-2.0-or-later license](LICENSE) inherited from RELION.

As of now, RELAX has no (reliable) features that RELION doesn't have, so practitioners should stick
to RELION.

The code was written almost entirely using AI coding tools, with a lot of human supervision.

## Current capabilities

| What | Command |
| --- | --- |
| Ab initio reconstruction / 3D initial models (VDAM), with one or multiple classes | `relax initial_model --K K` |
| 3D auto-refinement (Refine3D) | `relax refine` |
| 3D classification (Class3D) | `relax class3d --n_classes K` |
| Cryo-ET initial models, refinement, and classification | The same commands, with RELION 5 tilt-series inputs |

RELAX reads RELION STAR files and particle stacks and runs on NVIDIA GPUs. Refinement and
classification support one or two GPUs; subtomogram auto-refinement supports up to four on one node.

See the [user guide](docs/user_guide.md) for inputs and runnable examples, and the
[benchmarks](docs/benchmarks/relion_vs_relax.md) for comparisons with RELION.

## How to experiment

Clone the repo and `cd` into it **before** launching Claude Code or Codex:

```bash
git clone https://github.com/ma-gilles/relax.git
cd relax
```

Then launch your coding agent from there. In my experience, Claude Opus 5.5 is the best at this.

Describe as precisely as you can the modification you would like to make to RELION, and ask the
agent to implement it in RELAX. It may help to first have a conversation with the agent. Literally ask:

> Here is my plan: [...]. Ask any clarifying questions before you start.

Use [RECOVAR](https://ma-gilles.github.io/recovar/reference/api/simulation/) to generate synthetic but
realistic datasets, and compare your modified algorithm with the original on the same data. Start
small, check against the ground truth, and then try real datasets.

## Installation

RELAX uses [pixi](https://pixi.sh) and currently needs Linux, a CUDA 12 GPU and toolkit, and a RELION 5
source tree for the native RELION binding. From inside the cloned repo:

```bash
pixi install
pixi run build-cuda
RELION_SRC_DIR=/path/to/relion/src pixi run build-relion-bind
```

RECOVAR is installed automatically at the commit pinned in `pixi.toml`. Run RELAX commands inside
`pixi shell`, or prefix them with `pixi run`. The [user guide](docs/user_guide.md) has the details.

## Looking for something to work on?

Here is a [list of computational problems from the 2026 3DEM GRC](https://ma-gilles.github.io/notes/GRC_problems.pdf)
that Eric Verbeke and I put together.

Want to contribute? Send me an email at [gilles@princeton.edu](mailto:gilles@princeton.edu).
Want a feature imported from RELION? Open an [issue](https://github.com/ma-gilles/relax/issues).

License: [GPL-2.0-or-later](LICENSE).

Enjoy!
