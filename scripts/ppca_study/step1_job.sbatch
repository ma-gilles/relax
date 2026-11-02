#!/bin/bash
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=r17_step1
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/round17_simple_sgd_20261102/logs/%x-%j.log
# Round 17 step 1, box 64, 18 arms, up to 2 per GPU, then the suite scorer on all finished runs. Resumable: arms restart from checkpoints, scored checkpoints are skipped; a chained job continues if this one dies.
# Host memory: 4 arms at about 8 GB + 4 scorers at about 10 GB: 96G.
T=/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/round17_simple_sgd_20261102
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader
export OMP_NUM_THREADS=4
$T/tools/queue_step.sh $T/runs $T/${ARMS:-arms_step1_b64.json} 0 1 2; rc=$?
echo "queue rc=$rc $(date +%T)"
ls -d $T/runs/*_s11/DONE | xargs -n1 dirname | awk '{print NR%4, $0}' | while read i r; do echo "$((i%2)) $r"; done | xargs -P 4 -L1 bash -c 'SCORE_GPU=$0 bash '$T'/tools/post12.sh $1'
echo "scoring done $(date +%T)"
exit $rc
