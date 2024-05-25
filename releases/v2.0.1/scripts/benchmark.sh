#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=benchmark_sparseunet
#SBATCH --output=./outputs/eval/eval_job_%j.out

nvidia-smi

python tools/benchmark.py --experiment_name base