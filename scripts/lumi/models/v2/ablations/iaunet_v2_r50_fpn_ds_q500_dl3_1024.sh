#!/bin/bash
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=54:00:00

#SBATCH --job-name=iaunet
#SBATCH --output=./outputs/train/ablations_v2/job_%j.log

#SBATCH -A project_465001327



DATASET="livecell_crop"
echo "running on $DATASET with:"
echo "model=v2/iaunet-r50"
echo "model.decoder.type=iadecoder_ml_fpn/experimental/deep_supervision"

singularity exec /project/project_465001327/lumi_setup.sif \
               python main.py model=v2/iaunet-r50 \
               model.decoder.type=iadecoder_ml_fpn/experimental/deep_supervision \
               model.decoder.num_classes=1 \
               model.decoder.dec_layers=3 \
               model.decoder.num_queries=500 \
               model.decoder.dim_feedforward=1024 \
               dataset=$DATASET \
               dataset.train_dataset.batch_size=4 \
               dataset.valid_dataset.batch_size=4 \
               run.experiment_name=ablations_v2 \
               job_id=$SLURM_JOB_ID
