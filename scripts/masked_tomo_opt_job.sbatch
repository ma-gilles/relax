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
(cd $W/relax-masked-opt && git ls-files -co --exclude-standard relax tests scripts pyproject.toml | tar -cf - -T -) | tar -xf - -C $TMPDIR/relax
git -C $W/relax-masked-opt show $REF:relax/masked_tomo/likelihoods.py > $TMPDIR/ref/likelihoods_ref.py
git -C $W/relax-masked-opt show $REF:relax/masked_tomo/pilot.py | sed 's/^from relax.masked_tomo import likelihoods as L$/import likelihoods_ref as L/' > $TMPDIR/ref/pilot_ref.py
grep -q "^import likelihoods_ref as L" $TMPDIR/ref/pilot_ref.py
echo "recovar-masked $(git -C $W/recovar-masked rev-parse HEAD) diff-sha $(git -C $W/recovar-masked diff HEAD | sha256sum | cut -c1-16)"
echo "relax-masked-opt $(git -C $W/relax-masked-opt rev-parse HEAD) diff-sha $(git -C $W/relax-masked-opt diff HEAD | sha256sum | cut -c1-16) untracked: $(git -C $W/relax-masked-opt ls-files -o --exclude-standard relax scripts | tr '\n' ' ')"
echo "reference $REF $(git -C $W/relax-masked-opt rev-parse $REF)"
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
