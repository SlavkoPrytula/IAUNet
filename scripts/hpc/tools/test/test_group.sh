#!/bin/bash

experiment_paths=(
    "runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml]/[InstanceHead-v2.2.1-dual-update]/[job=52560796]-[2024-11-06 12:18:01]"
    # "runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2.1-dual-update]/[job=52560797]-[2024-11-06 12:18:01]"
    # "runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn_add_skip]/[InstanceHead-v2.2.1-dual-update]/[job=52560798]-[2024-11-06 12:18:01]"
    
    # "runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn_no_mask_branch]/[InstanceHead-v2.2.1-dual-update]/[job=52577561]-[2024-11-10 01:10:08]"
    # "runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn_no_inst_branch]/[InstanceHead-v2.2.1-dual-update]/[job=52577560]-[2024-11-10 01:11:42]"
    
    
    # "runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2.a-removed-mask-feats]/[job=52560800]-[2024-11-06 12:17:46]"
    # "runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2.a-removed-inst-feats]/[job=52560799]-[2024-11-06 12:18:01]"
    
    # "runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2.a-no-guided-query]/[job=52577565]-[2024-11-10 01:31:45]"
    # "runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2.a-no-support-query]/[job=52577562]-[2024-11-10 01:10:08]"
    
    # "runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2.3-dual-update]/[job=52577564]-[2024-11-10 01:10:08]"
)

for experiment_path in "${experiment_paths[@]}"; do
    echo "Submitting job for experiment path: $experiment_path"
    sbatch scripts/hpc/tools/test/test.sh "$experiment_path"
done
