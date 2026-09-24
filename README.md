# relax

RELION in JAX.

## What exists

| RELION job | relax entry point |
| --- | --- |
| 3D initial model (VDAM), K=1 and K>1 | `relax initial_model` (`--K`) |
| 3D auto-refine (Refine3D), K=1 | `python -m scripts.run_full_refinement` |
| 3D classification (Class3D), K>1 | `python -m scripts.run_full_refinement --n_classes K` |

Single GPU and a single optics group only. More to come.

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

## Benchmarks

RELION vs relax resolution and wall time for every synthetic and real dataset run: [docs/benchmarks/relion_vs_relax.md](docs/benchmarks/relion_vs_relax.md).

License: GPL-2.0-or-later
