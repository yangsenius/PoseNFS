python train.py \
--cfg exp_config/coco/sync/256x256/resnet.yaml \
--exp_name o/coco-ours-3/ \
--gpu 0,1 \
--param_flop \
--batchsize 60
--load_ckpt
