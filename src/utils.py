import torch
import torch.nn as nn
import os
import logging
import torchvision
import math
import numpy as np
import cv2
from collections import namedtuple
import matplotlib.pyplot as plt

def save_model(epoch, best, current_result, model, optimizer, scheduler, output_dir, logger):

    logger.info('==> mAP is: {:.3f}'.format(current_result))
    
    save_best = best if best>current_result else current_result
    ckeckpoint = {
        'best_result':save_best,
        'epoch':epoch + 1,
        'model': model.state_dict(), #module for nn.DataParalle
        'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict()
    }

    ckpt_path =  os.path.join(output_dir,'previous_ckpt.tar')
    torch.save( ckeckpoint ,ckpt_path)
    logger.info("epoch : {}, save ckeckpoint in {} ".format(epoch,ckpt_path ))

    if  current_result > best :
        best = current_result
        best_path = os.path.join(output_dir,'best_ckpt.tar')
        torch.save(model.state_dict(),best_path)
        logger.info('!(^ 0 ^)! New Best Result = {} and save model in {}\n'
                                        .format(best,best_path))
    
    return best

def load_ckpt(model , optimizer, scheduler, output_dir, logger):
    ckpt_path =  os.path.join(output_dir,'previous_ckpt.tar')
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

        begin = ckpt['epoch']
        best = ckpt['best_result']
        logger.info("load checkpoint from path:{}".format(ckpt_path))
        logger.info("training begin from previous checkpoint, history best_result = {}".format(best))
    else:
        logger.info("no ckeckpoint find in {}, training from epoch 0".format(ckpt_path))
        begin = 0

    return begin,best

import matplotlib.pyplot as plt

def visualize_heatamp(input,output,file_name):

    std=torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    mean=torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
    
    input = input.detach().cpu()
    image = input * std + mean
    image = image.permute(0,2,3,1).numpy()
    output = output.detach().cpu().permute(0,2,3,1).numpy()
    keypoint_channels = output.shape[-1]
   
    print(image.shape,output.shape)
    keypoint_name = {
        0: "nose",1: "L_eye",2: "R_eye",3: "L_ear",4: "R_ear",
        5: "L_shoulder",6: "R_shoulder",7: "L_elbow",8: "R_elbow",9: "L_wrist",10: "R_wrist",
        11: "L_hip",12: "R_hip",13: "L_knee",14: "R_knee",15: "L_ankle",16: "R_ankle" }

    fig=plt.figure(figsize=(16,8))#,constrained_layout=True)
    fig.subplots_adjust(top=0.93,bottom=0.075,left=0.0,right=0.985,hspace=0.485,wspace=0.0) #(left=0.03, right=0.97, hspace=0.3, wspace=0.05)
    
    fig_lines = 3#len(output) 
    fig_rows =  6#keypoint_channels+1

    
    #############    show intergral coordinate and argmax coordinate   ############
    for num in range(len(output)):
        fig.add_subplot( fig_lines, fig_rows , 1 + num*fig_rows)
        plt.imshow(image[num],cmap='gray') 

        for i in range(keypoint_channels):

            fig.add_subplot( fig_lines, fig_rows , i+2+num*fig_rows,)
            plt.imshow(output[num,:,:,i], cmap= 'seismic' ,interpolation='lanczos')
            plt.colorbar()
            plt.title(keypoint_name[i])
    plt.show()
    plt.savefig("debug_detection_heatmap_{}.png".format(file_name))


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name+'_debug.jpg', ndarr)

import  shutil
import glob
#glob.glob('*.py')

scripts_path = glob.glob('*.py')+glob.glob('*.yaml')+\
                glob.glob('exp_config/*.yaml')+\
                    glob.glob('src/*.yaml')+glob.glob('src/*.py')+\
                        glob.glob('src/architecture/*.py')+glob.glob('src/task_dataset/*.py')+glob.glob('src/network_factory/*.py')


def save_scripts_in_exp_dir(path, scripts_to_save= scripts_path):

    print("save scripts in {}".format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def filter_arch_parameters(model):
        for name,param in model.named_parameters():
            if'alpha' in name or 'beta' in name:
                continue
            yield param

def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]


            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size = [0,0], #output_size=list(output.size()), 
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of structures" + os.linesep
    for layer in layer_instances:
        details += "{} : {}   ".format(layer, layer_instances[layer])

    return details


    # class visualization_heatmaps(object):


    # def __init__(self,model,img):

    #     """
    #     Arg:    model - nn.Module
    #             img - (3,H,W) torch.tensor.float32
    #     """

    #     self.model = model.cuda()
    #     self.img = img.unsqueeze(0).cuda()
    #     self.inv_transform_std=torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    #     self.inv_transform_mean=torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)

    # def output(self,):

    #     return self.model(self.img)  #[1,k,H/4,W/4]
    
    # def vi(self):
        
    #     heatmaps = self.output()
    #     heatmaps = heatmaps.squeeze().cpu().numpy() #[k,H/4,W/4]

    #     img = self.img.cpu() * self.inv_transform_std + self.inv_transform_mean
    #     img = img.permute(1,2,0).numpy()


    #     fig=plt.figure(figsize=(16,8),constrained_layout=False)
    #     fig.subplots_adjust(top=0.93,bottom=0.075,left=0.0,right=0.985,hspace=0.485,wspace=0.0) #(left=0.03, right=0.97, hspace=0.3, wspace=0.05)
        
    #     fig_lines = 3
    #     fig_rows = 6

    #     fig.add_subplot( fig_lines, fig_rows , 1)
        
    #     plt.imshow(img,cmap='gray')   

    #     plt.title('image')
        
    #     keypoint_channels = len(heatmaps)
    
    #     #############   show heatmaps   ############
    #     for i in range(keypoint_channels):
    #         fig.add_subplot( fig_lines, fig_rows , i+2,)
    #         plt.imshow(heatmaps[i,:,:], cmap= 'seismic' ,interpolation='lanczos')#,vmin=0,vmax=1) # coolwarm seismic magma viridis tab20b  interpolation=' lanczos'
    #         plt.colorbar()

    #     plt.show() 
          