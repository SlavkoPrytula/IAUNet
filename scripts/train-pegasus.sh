#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g
#SBATCH --time=64:00:00
#SBATCH --mem=64GB
#SBATCH --exclude=falcon3
#SBATCH --cpus-per-task=8

#SBATCH --job-name=iaunet
#SBATCH --output=./outputs/train/job_%j.log

#SBATCH -A revvity

nvidia-smi

python main.py job_id=$SLURM_JOB_ID \
               dataset=evican2_easy \
               trainer=gpu
               