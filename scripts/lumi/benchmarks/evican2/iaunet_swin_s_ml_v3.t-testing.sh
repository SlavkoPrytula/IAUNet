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


singularity exec /project/project_465001327/lumi_setup.sif \
            python main.py \
            model=model/iaunet/iaunet-swin-s \
            model.decoder.instance_head.type=InstanceHead-v3.t-testing \
            model.decoder.type=iadecoder_ml_fpn \
            dataset=evican2_medium \
            job_id=$SLURM_JOB_ID