#!/bin/bash
# Round-16 copy of round 2's queue_step.sh: same environment (round 2 env.sh), round 14's coupled_queue.py, which
# runs round 14's driver copy (the φ / fudge knobs). queue_step.sh OUT ARMS.json SHARD N_SHARDS SLOTS
source /scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/round2_c_sgd_20261027/tools/env.sh
export TMPDIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/r17tmp  # short path for multiprocessing sockets
mkdir -p $JAX_COMPILATION_CACHE_DIR $RECOVAR_CUDA_CACHE_DIR $TMPDIR
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
git -c safe.directory='*' -C $SRC rev-parse HEAD; git -c safe.directory='*' -C $SRC status --short | head
exec $PY /scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/round17_simple_sgd_20261102/tools/coupled_queue.py "$@"
