# python main.py job_id=11111 \
#                dataset=revvity_25 \
#                model.evaluator.metric=[segm,boundary] \
#                trainer=gpu


# python main.py model=model/iaunet/iaunet-r50 \
#                model.decoder.instance_head.type=InstanceHead-v2.2.1-dual-update \
#                model.decoder.type=iadecoder_ml_fpn \
#                dataset=revvity_25 \
#                job_id=11111



DATASET="isbi2014"
echo "running on $DATASET with:"
echo "model=model/iaunet/v2/iaunet-r50"
echo "model.decoder.type=iadecoder_ml_fpn_dual_path_staged"


python main.py model=model/iaunet/v2/iaunet-r50 \
               model.decoder.type=iadecoder_ml_fpn_dual_path_staged \
               model.decoder.num_classes=2 \
               model.decoder.dec_layers=3 \
               dataset=$DATASET \
               job_id=11111
