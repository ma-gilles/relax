# relax

relax (RELION in JAX) holds the RECOVAR EM and VDAM refinement code: standard
3D auto-refinement, K-class classification, InitialModel/VDAM ab initio, the
pose-marginal PPCA refinement, the EM CUDA library and the RELION C++ binding
used as a numerical reference. It imports RECOVAR for the shared numerical
core (Fourier utilities, CTF, slicing, masks, data I/O and the pipeline CUDA
library); RECOVAR never imports relax.

Status: scaffold. Not yet a working release.

## Install (development)

relax pins one RECOVAR commit (`pyproject.toml`, `pixi.toml`). For
co-development, point the `dev` pixi feature at a local RECOVAR checkout.

```bash
pixi install
pixi run build-cuda          # librelax_cuda.so (needs recovar's CUDA headers)
RELION_SRC_DIR=/path/to/relion/src pixi run build-relion-bind
```

## Commands

```bash
relax initial_model ...      # InitialModel / VDAM ab initio
relax build_cuda             # build the EM CUDA library
relax build_relion_bind      # build the RELION pybind11 binding
```

## Licence

GPL-2.0-or-later; see `LICENSE`.
