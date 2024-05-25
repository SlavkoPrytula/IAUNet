#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=64:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=iaunet
#SBATCH --output=./outputs/train/job_%j.out

#SBATCH -A revvity

nvidia-smi

python main.py --job-id $SLURM_JOB_ID