#!/bin/bash
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=48:00:00

#SBATCH --job-name=iaunet
#SBATCH --output=./outputs/train/job_%j.log

#SBATCH -A project_465001327



DATASET="livecell_crop"
echo "running on $DATASET with:"
echo "model=model/iaunet/v2/iaunet-r50"
echo "model.decoder.type=iadecoder_ml_fpn"

singularity exec /project/project_465001327/lumi_setup.sif \
               python main.py model=model/iaunet/v2/iaunet-r50 \
               model.decoder.type=iadecoder_ml_fpn/experimental/mask_decoupling \
               model.decoder.num_classes=1 \
               model.decoder.dec_layers=3 \
               dataset=$DATASET \
               job_id=$SLURM_JOB_ID
