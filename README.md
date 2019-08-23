# Introduction to PNFS

This is the repository of `Pose Neural Fabrics Search (PNFS)`. We tightly integrate NAS with the task of human pose estimation. 
More information see the paper [Pose Neural Fabrics Search]

# Steps

## Create the `o` directory to reserve each experiment's output
```
mkdir o  
```
## Train the model
```
python train.py \
--cfg configs/our3.yaml \
--exp_name o/your_train_experiment_name \
--gpu 0,1 
```
other training optional commands

```
--batchsize 32  // change the default batchsize
--param_flop   // report the parameters and FLOPs
--search search_method_name   // options: ['None','random','sync','first_order_gradient','second_order_gradient']
--debug   // output the augmented input data for check
--visualize // visualize the predicted heatmaps for an image with the training process (per 5 epcohes)
--show_arch_value   // print the parameters of architecture in the training process
```
## Test the model
```
python test.py \
--cfg configs/ours3.yaml \
--exp_name o/your_test_experiment_name \
--gpu 0,1 \
--test_model o/path_to_your_saved_model \
--flip_test 
```
other testing optional commands
```
--visualize   // visualize the predicted heatmaps
--param_flop
---margin 1.15  // [1.0,1.5] margin between bbox border and input size when testing 
--flip_test   // horizontal flip test
--use_dt   // use the detection results of COCO val set or test-dev set
```

## Detailed Settings

All detailed settings of the model is recorded in the [`configs/*.yaml`](configs/).

#### Configuration for Fabric-Subnetwork

A snippet of the `*.yaml` for the hyperparameters of subnetworks :
```yaml
subnetwork_config:

  dataset_name: 'coco'
  parts_num : 3
  cell_config:
  
      vector_in_pixel : True
      vector_dim: 8
      convolution_mode: '2D'
      
      search_alpha: true
      search_beta: true
      operators: ["skip_connect", "Sep_Conv_3x3","Atr_Conv_3x3","max_pool_3x3"] # 

      depth: 7
      cut_layers_num: 4  # first several layers
      size_types: [4,8,16,32] # scales is [1/4, 1/8, 1/16, 1/32]
      hidden_states_num: 1
      factor: 16
      input_nodes_num: 1 # default
```

More potential cutomized computing units can be defined as candidate operations in [`src/architecture/operators.py`](src/architecture/operators.py).

#### Body Parts Mode
The body keypoints assignment for different parts is defined in [`src/network_factory/body_parts.py`](src/network_factory/body_parts.py)
The partion type of body parts can have more possibilities.

#### Exploration

About the `vector in pixel` method, we provide two convolutional mode `Conv2d` and `Conv3d` to implement the idea of how to construct the vector representation (`5D-Tensor`) of keypoint in [`src/network_factory/subnetwork.py`](src/network_factory/subnetwork.py). We use the `Conv2d` mode (reshape `5D-Tensor` to `4D-Tensor`) to get the data in paper. We indicate that the way of construting the vector represntation can be further explored in other fashions as long as the norm of the vector is under supervision.

# Experiments Results

Coming soon.




 
