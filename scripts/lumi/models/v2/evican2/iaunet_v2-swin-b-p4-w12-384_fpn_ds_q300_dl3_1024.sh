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



DATASET="evican2_easy"
echo "running on $DATASET with:"
echo "model=v2/iaunet-swin-b-p4-w12-384"
echo "model.decoder.type=iadecoder_ml_fpn/experimental/deep_supervision"

singularity exec /project/project_465001327/lumi_setup.sif \
               python main.py model=v2/iaunet-swin-b-p4-w12-384 \
               model.decoder.type=iadecoder_ml_fpn/experimental/deep_supervision \
               model.decoder.num_classes=1 \
               model.decoder.dec_layers=3 \
               model.decoder.num_queries=300 \
               model.decoder.dim_feedforward=1024 \
               dataset=$DATASET \
               job_id=$SLURM_JOB_ID
