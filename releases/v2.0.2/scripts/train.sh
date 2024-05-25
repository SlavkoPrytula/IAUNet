#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=64:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --job-name=iaunet
#SBATCH --output=./outputs/train/job_%j.out
#SBATCH -A revvity

nvidia-smi

python main.py --job_id $SLURM_JOB_ID