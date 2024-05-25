#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=analysis_iaunet
#SBATCH --output=./tools/analysis/performance/logs/log_%j.out

nvidia-smi

python eval.py

