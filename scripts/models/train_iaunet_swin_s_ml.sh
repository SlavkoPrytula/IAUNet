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


# iadecoder_ml + InstanceHead-v2.2.1-dual-update
python main.py model=model/iaunet/iaunet-swin-s \
               model.decoder.instance_head.type=InstanceHead-v2.2.1-dual-update \
               model.decoder.type=iadecoder_ml \
               dataset=neurlps22_cellseg \
               job_id=$SLURM_JOB_ID