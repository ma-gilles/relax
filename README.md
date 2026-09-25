# relax

RELION in JAX.

## What exists

| RELION job | relax entry point |
| --- | --- |
| 3D initial model (VDAM), K=1 and K>1 | `relax initial_model` (`--K`) |
| 3D auto-refine (Refine3D), K=1 | `python -m scripts.run_full_refinement` |
| 3D classification (Class3D), K>1 | `python -m scripts.run_full_refinement --n_classes K` |

Single GPU. Refine3D (K=1) accepts several optics groups, including groups on other pixel sizes
and boxes, but the default command refuses them for now: they run only on the device-resident
pass 2 and without the first-iteration cross-correlation, i.e. with `--no-firstiter_cc` and
`RELAX_SPARSE_PASS2_RESIDENT=1 RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF=1
RELAX_K1_RELION_POWERCLASS_SPECTRUM_NORM=1 RELAX_K1_RELION_EXACT_BPREF_OPERANDS=1`
(end-to-end GPU qualification against RELION in progress). OPEN: the M-step of a group on another
grid at its full box differs from RELION at the reference-sphere edge from iteration 3 (S3b); under
investigation.
InitialModel and Class3D take one optics group. Not yet: cryo-ET subtomograms,
CTF-premultiplied particles, beam tilt, higher-order aberrations and magnification. More to come.

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
