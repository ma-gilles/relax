# relax

RELION in JAX.

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
