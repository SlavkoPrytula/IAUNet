#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=64:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --job-name=iaunet
#SBATCH --array=1-5%5 
#SBATCH --output=./outputs/train/job_%A_%a.out
#SBATCH -A revvity

echo "Running job $SLURM_JOB_ID"
echo "Running group $SLURM_ARRAY_JOB_ID"
nvidia-smi

python main.py --job-id $SLURM_ARRAY_JOB_ID --run-id $SLURM_ARRAY_TASK_ID
