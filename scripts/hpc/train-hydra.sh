# python main.py job_id=11111 \
#                dataset=revvity_25 \
#                model.evaluator.metric=[segm,boundary] \
#                trainer=gpu


python main.py model=model/iaunet/iaunet-r50 \
               model.decoder.instance_head.type=InstanceHead-v2.2.1-dual-update \
               model.decoder.type=iadecoder_ml_fpn \
               dataset=revvity_25 \
               job_id=11111