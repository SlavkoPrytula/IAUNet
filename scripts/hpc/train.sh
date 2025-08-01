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

module load any/python/3.8.3-conda
conda activate iaunet

nvidia-smi

DATASET="revvity_25"

python main.py model=v2/iaunet-r50 \
               model.decoder.type=iadecoder_ml_fpn/experimental/deep_supervision \
               model.decoder.num_classes=1 \
               model.decoder.dec_layers=3 \
               model.decoder.num_queries=100 \
               model.decoder.dim_feedforward=1024 \
               dataset=$DATASET \
               job_id=$SLURM_JOB_ID

            #    logger=group_logger \
            #    callbacks.progress_bar.refresh_rate=0 \