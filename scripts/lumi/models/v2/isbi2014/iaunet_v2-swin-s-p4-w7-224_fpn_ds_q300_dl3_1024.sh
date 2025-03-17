#!/bin/bash
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=54:00:00

#SBATCH --job-name=iaunet
#SBATCH --output=./outputs/train/job_%j.log

#SBATCH -A project_465001327



DATASET="isbi2014"
echo "running on $DATASET with:"
echo "model=v2/iaunet-swin-s-p4-w7-224"
echo "model.decoder.type=iadecoder_ml_fpn/experimental/deep_supervision"

singularity exec /project/project_465001327/lumi_setup.sif \
               python main.py model=v2/iaunet-swin-s-p4-w7-224 \
               model.decoder.type=iadecoder_ml_fpn/experimental/deep_supervision \
               model.decoder.num_classes=2 \
               model.decoder.dec_layers=3 \
               model.decoder.num_queries=300 \
               model.decoder.dim_feedforward=1024 \
               dataset=$DATASET \
               job_id=$SLURM_JOB_ID
