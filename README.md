# RELAX! It's [RELION](https://github.com/3dem/relion) ... in JAX.

## What exists

| RELION job | relax entry point |
| --- | --- |
| 3D initial model (VDAM), K=1 and K>1 | `relax initial_model --K K` |
| 3D auto-refine (Refine3D), K=1 | `python -m scripts.run_full_refinement` |
| 3D classification (Class3D), K>1 | `python -m scripts.run_full_refinement --n_classes K` |
| Subtomogram auto-refine (RELION 5 tilt series), K=1 | `python -m scripts.run_full_refinement` on a RELION 5 particles.star |

- Single GPU (CUDA). Handles both single-particle (SPA) and tilt-series data.
- It reproduces RELION's results on a number of datasets; see the
  [benchmarks](docs/benchmarks/relion_vs_relax.md).
- Refine3D takes several optics groups, including other pixel sizes and boxes, but the default
  command still refuses them while they are qualified against RELION. InitialModel and Class3D
  take one optics group.
- Not yet: CTF-premultiplied particles, beam tilt, higher-order aberrations and magnification.
  More to come.

relax is a reproduction of RELION's code in JAX/Python, with CUDA backends. It was written nearly
entirely by AI coding tools, with a lot of human supervision.

## Install (development)

relax depends on [RECOVAR](https://github.com/ma-gilles/recovar) and pins one
RECOVAR commit in `pixi.toml`.

```bash
pixi install
pixi run build-cuda
RELION_SRC_DIR=/path/to/relion/src pixi run build-relion-bind
```

## Commands

```bash
relax initial_model ...
relax build_cuda
relax build_relion_bind
```

License: GPL-2.0-or-later

## Contribute?

Want to contribute? Send me an email `gilles@princeton.edu` 

Want a feature imported from RELION? Open an issue.
