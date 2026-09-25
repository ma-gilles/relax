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
pass 2 (the K=1 default) and without the first-iteration cross-correlation, i.e. with
`--no-firstiter_cc` and `RELAX_K1_RELION_POWERCLASS_SPECTRUM_NORM=1 RELAX_K1_RELION_EXACT_BPREF_OPERANDS=1`
(end-to-end GPU qualification against RELION in progress).

Refine3D (K=1) runs its E-step and M-step on the device-resident engine by default (global pass 2,
local search and device significance). `RELAX_SPARSE_PASS2_RESIDENT=0` and
`RELAX_LOCAL_SEARCH_RESIDENT=0` select the earlier compact and exact-local engines for A/B
checks. Class3D and the passes the resident drivers do not implement (subset replays, and any
configuration their checks refuse) use the earlier engines automatically; each run's
`refinement_results.npz` records the engine of every pass (`pass2_engine_trajectory`).
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
