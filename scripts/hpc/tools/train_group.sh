# sbatch scripts/models/train_iaunet_r50.sh
# sbatch scripts/models/train_iaunet_r50_ml.sh

# sbatch scripts/models/train_iaunet_r101.sh
# sbatch scripts/models/train_iaunet_r101_ml.sh

# sbatch scripts/models/train_iaunet_swin_s.sh
# sbatch scripts/models/train_iaunet_swin_s_ml.sh

# sbatch scripts/models/train_iaunet_swin_b.sh
# sbatch scripts/models/train_iaunet_swin_b_ml.sh



# ablations. 
# ------------------
# pixel-decoder. +++
# sbatch scripts/hpc/ablations/pixel_decoder/iaunet_r50_ml_v2.2.1-dual-update.sh
# sbatch scripts/hpc/ablations/pixel_decoder/iaunet_r50_ml_fpn_v2.2.1-dual-update.sh
# sbatch scripts/hpc/ablations/pixel_decoder/iaunet_r50_ml_fpn_add_skip_v2.2.1-dual-update.sh
# sbatch scripts/hpc/ablations/pixel_decoder/iaunet_r50_ml_fpn_no_inst_branch_v2.2.1-dual-update.sh
# sbatch scripts/hpc/ablations/pixel_decoder/iaunet_r50_ml_fpn_no_mask_branch_v2.2.1-dual-update.sh

# ------------------
# transformer-decoder. [architecture] ++
# sbatch scripts/hpc/ablations/transformer_decoder/architecture/iaunet_r50_ml_fpn_v2.2.a-removed-inst-feats.sh
# sbatch scripts/hpc/ablations/transformer_decoder/architecture/iaunet_r50_ml_fpn_v2.2.a-removed-mask-feats.sh
# sbatch scripts/hpc/ablations/transformer_decoder/architecture/iaunet_r50_ml_fpn_v2.2.a-no-support-query.sh
# sbatch scripts/hpc/ablations/transformer_decoder/architecture/iaunet_r50_ml_fpn_v2.2.a-no-guided-query.sh

# sbatch scripts/hpc/ablations/transformer_decoder/architecture/iaunet_r50_ml_fpn_v2.2.3-swin-dual-update.sh
