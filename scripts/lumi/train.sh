#!/bin/bash
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=40G
#SBATCH --time=64:00:00

#SBATCH --job-name=iaunet
#SBATCH --output=./outputs/train/job_%j.log

#SBATCH -A project_465001327


singularity exec /project/project_465001327/lumi_setup.sif \
            python main.py \
            model=model/iaunet/iaunet-r50 \
            model.decoder.instance_head.type=InstanceHead-v1.1 \
            model.decoder.type=iadecoder \
            dataset=revvity_25 \
            job_id=$SLURM_JOB_ID
