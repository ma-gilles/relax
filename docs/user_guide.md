# relax user guide

relax runs RELION's 3D initial model, 3D auto-refine and 3D classification on
one CUDA GPU. It reads RELION's own inputs: a particle STAR file with an optics
block, its `.mrcs` stacks and, for refinement, a reference map.

## Install

relax needs [pixi](https://pixi.sh), a Linux machine with a CUDA 12 GPU and a
RELION 5 source tree (for the native RELION binding). The environment pins one
RECOVAR commit, so no separate RECOVAR install is needed.

```bash
git clone git@github.com:ma-gilles/relax.git
cd relax
pixi install
```

## Build the native libraries

Build the CUDA kernels and the RELION binding once per checkout, on a machine
with the CUDA toolkit:

```bash
pixi run build-cuda
RELION_SRC_DIR=/path/to/relion/src pixi run build-relion-bind
```

Run every command below inside the environment (`pixi shell`, or prefix each
command with `pixi run`). Choose the GPU with `CUDA_VISIBLE_DEVICES`.

## Supported GPUs

relax needs an NVIDIA GPU of compute capability 6.0 (Pascal) or newer and at
least 16 GB of memory. It has been run end to end on P100 16 GB (6.0), A100
80 GB (8.0) and H100 80 GB (9.0), and on 80 GB cards limited to 16 to 40 GB; the
[GPU compatibility matrix](development/gpu_compatibility.md) lists every
workflow per card. Batch sizes and caches are sized from the card's free memory,
so smaller cards run the same commands, more slowly.

Host memory grows with the box: a box-800 refinement (EMPIAR-10202, 30,515 particles) peaks at
about 434 GB of host memory, in the final reconstruction's padded inverse FFT; RELION 5 used about
449 to 475 GB over its MPI ranks for the same run. Request at least 480 GB for a box of 800.
With large particle stacks on a network file system, pass `--scratch_dir` (node-local disk) or
`--preread_images`: reading the images over the network can dominate the first iterations.

Leave `XLA_PYTHON_CLIENT_PREALLOCATE` unset. relax runs with JAX's default, one
preallocated memory pool, and that is the supported mode. `=false` lets another
process share the GPU, but the pool then grows in separate regions and an array
must fit inside one of them: a long refinement of a large box (above about 550
pixels on an 80 GB card, 320 on 40 GB) can stop with `RESOURCE_EXHAUSTED` while
memory is free. An out-of-memory error carries a note with the pool's limit, its peak
use and the request, which says whether the pool was fragmented, and names the setting
when it is off.

The default build compiles the kernels for compute capability 7.0 to 9.0 (and
later cards through PTX). For a Pascal card (P100, GTX 10-series), add its
architecture, with a CUDA 12 toolkit (CUDA 13 cannot build for cards older than
compute capability 7.5):

```bash
export CUDA_ARCH="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80"
pixi run build-cuda                                    # relax's kernels
pixi run python -m recovar.commands.build_custom_cuda --force  # RECOVAR's kernels
```

A library built without your card's architecture is refused when relax first
loads it, with a message naming the card, the library's targets and the rebuild
command.

## Inputs

- **Particles**: a RELION 3.1+ particle STAR file with a `data_optics` block
  (pixel size, voltage, Cs, amplitude contrast, image size) and a
  `data_particles` block with `rlnImageName` (`index@stack.mrcs`) and the CTF
  columns. Several optics groups, including groups with different image sizes,
  are supported.
- **Reference map** (refine and class3d): an MRC in RELION's map convention,
  the file `relion_refine --ref` would read. relax accepts a map written by
  RELION or relax, or an unlabelled map whose file name contains `_relion`;
  it refuses any other map, because a map with the opposite sign convention
  would start the run from the negated reference.
- `relax refine` and `relax class3d` read the particles from
  `<data_dir>/particles.star` and resolve relative image paths against
  `<data_dir>`. RELION writes them relative to its project directory, so copy
  the STAR file into `<data_dir>` and link the project directories it names
  (usually `Extract`) next to it, as in the examples below.

## 3D initial model (VDAM)

```bash
relax initial_model \
  --i Extract/job005/particles.star \
  --o InitialModel/relax/run \
  --K 1 \
  --sym C1 \
  --particle-diameter 200
```

`--K 3` builds three classes. The output prefix gets RELION-style
`run_itNNN_*` files; the last iteration's map is the initial model.

## 3D auto-refine (Refine3D, K=1)

```bash
mkdir -p Refine3D/relax_input
cp Extract/job005/particles.star Refine3D/relax_input/particles.star
ln -s "$PWD/Extract" Refine3D/relax_input/Extract
relax refine \
  --data_dir Refine3D/relax_input \
  --output Refine3D/relax \
  --init_volume InitialModel/relax/run_it200_class001.mrc \
  --sym C1 \
  --particle_diameter_ang 200 \
  --init_resolution 60 \
  --seed 1
```

The run stops at convergence, as RELION's auto-refine does. Without
`--init_volume` it reads `<data_dir>/reference_init_relion.mrc`. The defaults are
the RELION GUI's job defaults ([audit](development/relion_defaults.md)); pass
`--no-firstiter_cc` when the reference is on the images' greyscale.

## 3D classification (Class3D, K>1)

```bash
mkdir -p Class3D/relax_input
cp Extract/job005/particles.star Class3D/relax_input/particles.star
ln -s "$PWD/Extract" Class3D/relax_input/Extract
relax class3d \
  --data_dir Class3D/relax_input \
  --output Class3D/relax \
  --n_classes 4 \
  --ref_star Class3D/references.star \
  --particle_diameter_ang 200 \
  --init_resolution 60 \
  --seed 1
```

`--ref_star` is the STAR file `relion_refine --ref` reads for Class3D, with one
`rlnReferenceImage` map per class. `--init_class_volumes a.mrc,b.mrc,...` gives
the maps directly instead. `--init_volume ref.mrc` is RELION's usual start from one map
(`relion_refine --ref ref.mrc --K K`): every class starts from it and each particle is
assigned a random class in the first iteration (in the second with `--firstiter_cc`), from
`--seed` as RELION does. The run does 25 iterations (`--max_iter`), as the
RELION GUI does.

`--skip_align` is RELION's classification without alignment (GUI "Perform image
alignment: No"): each particle is scored against every class at the angles and offsets
of the input STAR, which the run writes back unchanged. With the default
`--firstiter_cc` and one `--init_volume`, the first iteration scores each particle
by cross-correlation at its given pose, as RELION does. Not implemented yet with
it: `--firstiter_cc` from several references, `--continue` and subtomogram particles.

### Continuing a run

`relax refine` and `relax class3d` write RELION's `run_itNNN_*` files every iteration.
`--continue <output>/run_itNNN_optimiser.star` resumes after iteration NNN with the same
command, as `relion_refine --continue` does, also for optics groups of several image
sizes or pixel sizes. `--max_iter` stays the last iteration of the whole run. A Class3D
run from one `--init_volume` with `--firstiter_cc` gives the particles their random
classes in iteration 2, so a continuation from iteration 1 scores classes that are still
copies of one map and never seeds them, as RELION's continuation does; the run warns.
Continue from iteration 2 or later.

## Subtomogram auto-refine (RELION 5 tilt series)

```bash
mkdir -p Tomo/relax_input
cp Extract/job012/particles.star Tomo/relax_input/particles.star
cp Tomograms/job008/tomograms.star Tomo/relax_input/tomograms.star
for d in Extract Tomograms; do ln -s "$PWD/$d" "Tomo/relax_input/$d"; done
relax refine \
  --data_dir Tomo/relax_input \
  --output Tomo/relax_refine \
  --init_volume Tomo/reference_relion.mrc \
  --particle_diameter_ang 250 \
  --init_resolution 40 \
  --seed 1
```

`<data_dir>` holds the RELION 5 particle STAR file of pseudo-subtomograms
written as 2D stacks (its `data_general` block sets
`rlnTomoSubTomosAre2DStacks 1`), named `particles.star`, and the matching
`tomograms.star`. The file names in the example are placeholders for your
Extract and tomogram jobs; link every project directory the two STAR files
name. relax detects the 2D-stack format and scores every particle over its tilt images.
As for single particles, `--firstiter_cc` (the default) starts with RELION's cross-correlation
iteration; pass `--no-firstiter_cc` when the reference is on the images' greyscale.

## Subtomogram 3D initial model (VDAM)

```bash
relax initial_model \
  --ios Tomo/relax_input/optimisation_set.star \
  --o Tomo/relax_initial_model/run \
  --K 1 \
  --sym C1 \
  --particle-diameter 250
```

`--ios` takes the RELION 5 optimisation set (its `rlnTomoParticlesFile` and
`rlnTomoTomogramsFile`, relative to the set's directory), as `relion_refine --ios` does. One
optics group, as for single particles. With one optics group RELION seeds only the first class
(the start-up reads the first particle's tilt images alone), and an empty class stays empty:
`--K 2` then gives one map and an empty class, in RELION and relax alike.

## Subtomogram 3D classification

```bash
relax class3d \
  --data_dir Tomo/relax_input \
  --output Tomo/relax_class3d \
  --n_classes 2 \
  --init_volume Tomo/reference_relion.mrc \
  --particle_diameter_ang 250 \
  --init_resolution 40 \
  --seed 1
```

The same inputs as the subtomogram auto-refine.

## Useful options

- `relax refine --help` lists every option; `docs/development/em_parity_runbook.md`
  maps each `relion_refine` option to its relax name.
- `--preread_images` reads all particles into memory at start-up;
  `--scratch_dir DIR` copies the referenced particles to local disk first (RELION's
  `--preread_images` and `--scratch_dir`).
- `--max_iter N` limits the number of iterations.
- `--mode relax` sets every opt-in correction of RELION's own inconsistencies (`--gridding_kernel separable`
  and five others, each also a flag of its own); they are described in
  [RELION-consistency options](math/relion_consistency_options.md). The default, `--mode relion`, keeps
  RELION's rule everywhere.
