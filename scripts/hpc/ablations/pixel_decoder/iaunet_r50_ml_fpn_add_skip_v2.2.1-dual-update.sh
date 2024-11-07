#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=64:00:00
#SBATCH --mem=64GB
#SBATCH --exclude=falcon3
#SBATCH --cpus-per-task=8

#SBATCH --job-name=iaunet
#SBATCH --output=./outputs/benchmarks/job_%j.log

#SBATCH -A revvity

nvidia-smi


python main.py model=model/iaunet/iaunet-r50 \
               model.decoder.instance_head.type=InstanceHead-v2.2.1-dual-update \
               model.decoder.type=iadecoder_ml_fpn_add_skip \
               dataset=livecell_crop \
               job_id=$SLURM_JOB_ID