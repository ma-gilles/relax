#!/bin/bash -l
# Masked-tomo arm-B optimisation job: snapshots recovar-masked and relax-masked-opt (working trees), writes the
# reference likelihoods/pilot from MT_REF_COMMIT (default cc40162f) into ref/, and runs the given command there.
# GPU: sbatch --partition=gpu --constraint=h100 --gpus=1 ...; CPU: sbatch --partition=genx ...
#SBATCH --job-name=mto
#SBATCH --account=ccm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/home/mgilles/projects/dogic-recovar-et/masked-tomo/logs/mto-%j.log
set -euo pipefail
W=/mnt/home/mgilles/projects/dogic-recovar-et
REF=${MT_REF_COMMIT:-cc40162f}
WT=${MT_WORKTREE:-relax-masked-opt}  # the RELAX worktree to snapshot (under $W)
GPU=${SLURM_GPUS:-${SLURM_GPUS_ON_NODE:-0}}
if [ "$GPU" != "0" ]; then
  module load modules/2.4-20250724 gcc/13.3.0 python/3.11.11 cuda/12.8.0
else
  module load modules/2.4-20250724 gcc/13.3.0 python/3.11.11
fi
source $W/venv/bin/activate
export TMPDIR; TMPDIR=$(mktemp -d "/tmp/mgilles-mto-${SLURM_JOB_ID}.XXXXXX"); trap 'rm -rf -- "$TMPDIR"' EXIT
mkdir -p $TMPDIR/recovar $TMPDIR/relax $TMPDIR/ref
(cd $W/recovar-masked && git ls-files -co --exclude-standard recovar | tar -cf - -T -) | tar -xf - -C $TMPDIR/recovar
if [ -n "${MT_RECOVAR_FIX:-}" ]; then  # a second RECOVAR snapshot (e.g. recovar-projfix) at $TMPDIR/recovar_fix
  mkdir -p $TMPDIR/recovar_fix
  (cd $W/$MT_RECOVAR_FIX && git ls-files -co --exclude-standard recovar | tar -cf - -T -) | tar -xf - -C $TMPDIR/recovar_fix
  echo "$MT_RECOVAR_FIX $(git -C $W/$MT_RECOVAR_FIX rev-parse HEAD) diff-sha $(git -C $W/$MT_RECOVAR_FIX diff HEAD | sha256sum | cut -c1-16)"
fi
(cd $W/$WT && git ls-files -co --exclude-standard relax tests scripts pyproject.toml | tar -cf - -T -) | tar -xf - -C $TMPDIR/relax
git -C $W/$WT show $REF:relax/masked_tomo/likelihoods.py > $TMPDIR/ref/likelihoods_ref.py
git -C $W/$WT show $REF:relax/masked_tomo/pilot.py | sed 's/^from relax.masked_tomo import likelihoods as L$/import likelihoods_ref as L/' > $TMPDIR/ref/pilot_ref.py
grep -q "^import likelihoods_ref as L" $TMPDIR/ref/pilot_ref.py
echo "recovar-masked $(git -C $W/recovar-masked rev-parse HEAD) diff-sha $(git -C $W/recovar-masked diff HEAD | sha256sum | cut -c1-16)"
echo "$WT $(git -C $W/$WT rev-parse HEAD) diff-sha $(git -C $W/$WT diff HEAD | sha256sum | cut -c1-16) untracked: $(git -C $W/$WT ls-files -o --exclude-standard relax scripts | tr '\n' ' ')"
echo "reference $REF $(git -C $W/$WT rev-parse $REF)"
export PYTHONPATH=$TMPDIR/recovar:$TMPDIR/relax RELAX_ALLOW_SHADOWED_RECOVAR=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export XDG_CACHE_HOME="$W/cache" MPLCONFIGDIR="$W/cache/matplotlib"
if [ "$GPU" != "0" ]; then
  export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_MEM_FRACTION=.85 JAX_COMPILATION_CACHE_DIR="$W/cache/jax"
  export RECOVAR_CUDA_CACHE_DIR="$W/cache/recovar-cuda-recovar-masked" CUDA_CACHE_PATH="$W/cache/cuda-driver"
  export NVCC; NVCC=$(command -v nvcc)
  PIPLIBS=$(python -c "import site,pathlib;print(':'.join(sorted({str(p) for r in site.getsitepackages() for p in (pathlib.Path(r)/'nvidia').glob('*/lib') if p.is_dir()})))")
  export LD_LIBRARY_PATH="$PIPLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  nvidia-smi -L
else
  export JAX_PLATFORMS=cpu
fi
hostname; date
python -c "import jax, recovar, relax; print(jax.devices(), recovar.__file__, relax.__file__)"
cd $TMPDIR
/usr/bin/time -v "$@"
