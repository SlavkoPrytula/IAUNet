#!/bin/bash
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=40G
#SBATCH --time=01:00:00

#SBATCH --job-name=test_iaunet
#SBATCH --output=test-log-%J.txt

#SBATCH -A project_465001327


singularity exec /project/project_465001327/lumi_setup.sif python test_gpu.py