#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=eval_sparseunet
#SBATCH --output=./outputs/eval/eval_job_%j.out

nvidia-smi

# python test_net_v2.py
# python test.py

# python eval_v2.py --experiment_name base
# python eval_v1.py --experiment_name base
python eval.py --experiment_name base
# python eval.py --experiment_name fusing_iams_preds
# python eval.py --experiment_name fusing_iams_logits