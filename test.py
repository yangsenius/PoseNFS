# Written by Sen Yang (yangsenius@seu.edu.cn)

import torch
import torchvision
import os
import numpy as np
import pprint 

import yaml
from easydict import EasyDict as edict
import logging
import argparse
from timeit import default_timer as timer
import datetime
#from thop import profile

from src.build_your_net import bulid_up_network

from src.evaluate import evaluate
#from src.search_arch import Search_Arch
from src.loss import MSELoss
from src.task_dataset.dataset import dataset_
from src.utils import   save_batch_image_with_joints,\
                    save_model,\
                    save_scripts_in_exp_dir,\
                    AverageMeter, \
                    load_ckpt,\
                    visualize_heatamp


def args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    parser.add_argument('--cfg',            help='experiment configure file name',  required=True,   default='config.yaml', type=str)
    parser.add_argument('--exp_name',       help='experiment name',        default='test'     , type=str)
    parser.add_argument('--use_dt',       help='if use detection results or',  action='store_true' ,default= False )
    parser.add_argument('--flip_test',       help='',  action='store_true' ,default= False )
    parser.add_argument('--test_model',       help='test model',  type=str  )
    parser.add_argument('--param_flop',     help=' ', action='store_true', default=False)
    
    parser.add_argument('--gpu',       help='gpu ids',  type=str  )
    parser.add_argument('--margin',       help='margin_to_border',  type=float ,default= 1.15 )
    parser.add_argument('--visualize',       help='visualize',  action='store_true' ,default= False )
    parser.add_argument('--dataset', help='run test.py on which dataset. options: test or val', default='val')
    
    args = parser.parse_args()
    return args

def logging_set(output_dir):

    logging.basicConfig(filename = os.path.join(output_dir,'test_{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),
                    format = '%(asctime)s - %(name)s: L%(lineno)d - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger

def main():

    arg = args()
    
    if not os.path.exists(arg.exp_name):
        os.makedirs(arg.exp_name)
    print(arg.exp_name.split('/')[0])
    assert arg.exp_name.split('/')[0]=='o',"'o' is the directory of experiment, --exp_name o/..."
    

    output_dir = arg.exp_name

    logger = logging_set(output_dir)
    logger.info('\n================ experient name:[{}] ===================\n'.format(arg.exp_name))
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    torch.backends.cudnn.enabled = True
 
    config = edict( yaml.load( open(arg.cfg,'r')))

    config.test.flip_test = arg.flip_test
    config.test.batchsize = 128
    config.model.margin_to_border = arg.margin

    logger.info('------------------------------ configuration ---------------------------')
    logger.info('\n==> available {} GPUs , numbers are {}\n'.format(torch.cuda.device_count(),os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info(pprint.pformat(config))
    logger.info('------------------------------- -------- ----------------------------')
    
    criterion = MSELoss()

    Arch = bulid_up_network(config, criterion)

    if arg.param_flop:
        Arch._print_info()
    
    logger.info("=========>current architecture's values before evaluate")

    if hasattr(Arch.backbone,"alphas"):

        Arch.backbone._show_alpha()
        Arch.backbone._show_beta()

    for id,group in enumerate(Arch.groups):

        group._show_alpha()
        group._show_beta()
   
    if arg.test_model:
        logger.info('\n===> load ckpt in : {}'.format(arg.test_model))
        Arch.load_state_dict(torch.load(arg.test_model))
    elif config.test.ckpt !='':
        logger.info('\n===> load ckpt in : '+ config.test.ckpt +'...')
        Arch.load_state_dict(torch.load(config.test.ckpt))
    elif os.path.exists(os.path.join(output_dir,'best_ckpt.tar')):
        logger.info('\n===> load ckpt in : '+ os.path.join(output_dir,'best_ckpt.tar'))
        Arch.load_state_dict(torch.load(os.path.join(output_dir,'best_ckpt.tar')))
    else:
        logger.info('\n===>no ckpt is found, use the initial model ...')
        #raise ValueError
    #logger.info(Arch.backbone.alphas)

    logger.info("=========>Architecture's parameters")
    if hasattr(Arch,"backbone"):
        if hasattr(Arch.backbone,"alphas"):
            Arch.backbone._show_alpha(original_value=False)
            Arch.backbone._show_beta(original_value=False)
        for g in Arch.groups:
            g._show_alpha(original_value=False)
            g._show_beta(original_value=False)
    
    Arch = torch.nn.DataParallel(Arch).cuda()
    
    valid_dataset = dataset_(config,config.images_root_dir,
                            config.annotation_root_dir,
                            mode='val',
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]))
    #test_img(valid_dataset,output_dir)
    valid_dt_dataset =dataset_(config,config.images_root_dir,
                            config.person_detection_results_path,
                            mode='dt',
                            dataset = config.test.dataset_name,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]))

    if arg.use_dt:

        logger.info("\n >>> use detection results ")
        valid_dataloader = torch.utils.data.DataLoader(valid_dt_dataset, batch_size = config.test.batchsize, shuffle = False , num_workers = 4 , pin_memory=True )
    else:
        logger.info("\n >>> use groundtruth bbox ")
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = config.test.batchsize, shuffle = False , num_workers = 4 , pin_memory=True )
    
    if arg.visualize:
        for i in range(len(valid_dataset)):
            imageid = 185250
            
            if valid_dataset[i][1]!=imageid: # choose an image_id
                continue
            print(valid_dataset[i][1])
            sample = valid_dataset[i]
            logger.info("visualize the predicted heatmap of image id {} ".format(imageid))
            img = sample[0].unsqueeze(0)
            #samples = next(iter(valid_dataloader))
            #img = samples[0]
            output = Arch(img)
            print(img.size(),output.size())
            visualize_heatamp(img,output,'heatmaps',show_img=False)
            break

    results = evaluate( Arch, valid_dataloader , config, output_dir)
    logger.info('map = {}'.format(results))


def test_img(dataset,output_dir):

    id = np.random.randint(len(dataset))
    sample = dataset[id]
    std=torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    mean=torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)

    inputdata = sample[0]
    input_flip = inputdata.flip([2])

    image = inputdata * std + mean
    image_flip = input_flip * std + mean
    image = image.permute(1,2,0).numpy()
    image_flip = image_flip.permute(1,2,0).numpy()
    
    import cv2
    path = os.path.join(output_dir,'test{}.jpg'.format(id))
    path_flip = os.path.join(output_dir,'test{}_flip.jpg'.format(id))
    print(path)
    cv2.imwrite(path,image*255)
    cv2.imwrite(path_flip,image_flip*255)

if __name__=='__main__':
    main()
