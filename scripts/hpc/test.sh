#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=4:00:00
#SBATCH --mem=24GB
#SBATCH --job-name=test_iaunet
#SBATCH --exclude=falcon3
#SBATCH --output=./outputs/test/job_%j.log
#SBATCH -A revvity

module load any/python/3.8.3-conda
conda activate iaunet

nvidia-smi

python eval.py