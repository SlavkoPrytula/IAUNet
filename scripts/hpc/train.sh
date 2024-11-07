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

nvidia-smi

python main.py model=model/iaunet/iaunet-r50 \
               model.decoder.instance_head.type=InstanceHead-v2.2.1-dual-update \
               model.decoder.type=iadecoder_ml \
               dataset=livecell_crop \
               job_id=$SLURM_JOB_ID
# srun --partition=gpu --gres=gpu:tesla:1 --time=60 --exclude=falcon3 --pty /bin/bash
# srun --partition=small-g --time=60 -A project_465001327 --pty singularity exec --bind /project/project_465001327 --bind /scratch/project_465001327 --bind /usr --bind /etc --bind /run/munge --bind /opt/cray/ --bind /etc/passwd --bind /etc/group --bind /run/netconfig/resolv.conf --bind /var /project/project_465001327/lumi_setup.sif /bin/bash


