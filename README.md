# Pose-Neural-Fabric-Search

We integrate NAS with the task of human pose estimation. To make full use of the structure of human body and NAS' ability of learning structures of networks, we model body pose into multiple parts,each of which is predicted by a cell-based neural fabric.

# Steps

1. create the `o` directory to reserve each experiment's output
```
mkdir o  `
```
2. run the code
```
python train.py --cfg configs/our3.yaml --exp_name o/your_experiment_name --gpu 0,1 --param_flop
```
or
```
sh train_coco.sh
```



 
