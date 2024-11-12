#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=4:00:00
#SBATCH --mem=24GB
#SBATCH --job-name=eval_iaunet
#SBATCH --exclude=falcon3
#SBATCH --output=./outputs/eval/eval_job_%j.out
#SBATCH -A revvity

if [ -z "$1" ]; then
  echo "Usage: sbatch eval.sh <experiment_path>"
  exit 1
fi

EXPERIMENT_PATH=$1

echo "Running evaluation for experiment path: $EXPERIMENT_PATH"
nvidia-smi

python eval.py "$EXPERIMENT_PATH"
