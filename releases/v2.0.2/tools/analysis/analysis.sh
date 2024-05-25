#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=analysis_iaunet
#SBATCH --output=./outputs/analysis/job_%j.out

nvidia-smi

python tools/analysis/iam_offset.py

