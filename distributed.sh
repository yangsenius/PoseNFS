# 
python -m torch.distributed.launch --nproc_per_node=3 train_dist.py --exp_name o/distributed_parts8_novector/ --cfg configs/coco_3.yaml --gpu 1,2,3 --distributed --batchsize 20