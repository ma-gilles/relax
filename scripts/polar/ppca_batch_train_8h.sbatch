#!/usr/bin/env bash
#SBATCH --job-name=ppca-batch-sweep
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=08:00:00
source "$POLAR_SOURCE_ROOT/scripts/polar/ppca_batch_train_body.sh"
