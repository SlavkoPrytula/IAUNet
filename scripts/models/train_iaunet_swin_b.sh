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


# iadecoder + InstanceHead-v1.1
python main.py model=model/iaunet/iaunet-swin-b \
               model.decoder.instance_head.type=InstanceHead-v1.1 \
               model.decoder.type=iadecoder \
               dataset=revvity_25 \
               job_id=$SLURM_JOB_ID
