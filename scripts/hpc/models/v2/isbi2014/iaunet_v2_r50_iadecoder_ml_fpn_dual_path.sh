#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=96:00:00
#SBATCH --mem=64GB
#SBATCH --exclude=falcon3
#SBATCH --cpus-per-task=8

#SBATCH --job-name=iaunet
#SBATCH --output=./outputs/train/job_%j.log

#SBATCH -A revvity


DATASET="isbi2014"
echo "running on $DATASET with:"
echo "model=model/iaunet/v2/iaunet-r50"
echo "model.decoder.type=iadecoder_ml_fpn_dual_path"

python main.py model=model/iaunet/v2/iaunet-r50 \
               model.decoder.type=iadecoder_ml_fpn_dual_path \
               model.decoder.num_classes=2 \
               model.decoder.dec_layers=3 \
               dataset=$DATASET \
               job_id=$SLURM_JOB_ID
