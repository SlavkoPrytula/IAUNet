#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:2
#SBATCH --time=64:00:00
#SBATCH --mem=96GB
#SBATCH --exclude=falcon3
#SBATCH --cpus-per-task=4

#SBATCH --job-name=iaunet
#SBATCH --output=./outputs/train/job_%j.log

#SBATCH -A revvity

nvidia-smi

python main.py job_id=$SLURM_JOB_ID dataset=livecell_crop
# srun --partition=gpu --gres=gpu:tesla:1 --time=60 --exclude=falcon3 --pty /bin/bash