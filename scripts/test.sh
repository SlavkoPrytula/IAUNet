#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=eval_iaunet
#SBATCH --output=./outputs/eval/eval_job_%j.out
#SBATCH -A revvity

nvidia-smi

python eval.py 