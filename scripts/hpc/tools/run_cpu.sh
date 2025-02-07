#!/bin/bash
#SBATCH --partition=amd
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=coco_splits
#SBATCH --output=./outputs/train/cpu_job_%j.log
#SBATCH -A revvity

python dataset/datasets/tools/prepare_cellpainting_gallery.py