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
  --no-firstiter_cc \
  --seed 1
```

`<data_dir>` holds the RELION 5 particle STAR file of pseudo-subtomograms
written as 2D stacks (its `data_general` block sets
`rlnTomoSubTomosAre2DStacks 1`), named `particles.star`, and the matching
`tomograms.star`. The file names in the example are placeholders for your
Extract and tomogram jobs; link every project directory the two STAR files
name. relax detects the 2D-stack format and scores every particle over its tilt images.
Subtomograms have no first-iteration cross-correlation yet, so the reference must be on
the images' greyscale and `--no-firstiter_cc` is required.

## Subtomogram 3D classification

```bash
relax class3d \
  --data_dir Tomo/relax_input \
  --output Tomo/relax_class3d \
  --n_classes 2 \
  --init_volume Tomo/reference_relion.mrc \
  --particle_diameter_ang 250 \
  --init_resolution 40 \
  --no-firstiter_cc \
  --seed 1
```

The same inputs as the subtomogram auto-refine.

## Useful options

- `relax refine --help` lists every option; `docs/development/em_parity_runbook.md`
  maps each `relion_refine` option to its relax name.
- `--preread_images` reads all particles into memory at start-up;
  `--scratch_dir DIR` copies the stacks to local disk first (RELION's
  `--preread_images` and `--scratch_dir`).
- `--max_iter N` limits the number of iterations.
