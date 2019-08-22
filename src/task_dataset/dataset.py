# -*- coding: UTF-8 -*-
#!/usr/bin/python
# Written by Sen Yang (yangsenius@seu.edu.cn)

import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
import random

import cv2
import logging
import json

from pycocotools.coco import COCO
from pycocotools import mask 
from skimage.filters import gaussian

from torch.utils.data import Dataset

from task_dataset.occlusion_augmentation import Random_Occlusion_Augmentation
from task_dataset.preprocess import make_affine_matrix,\
                                    mpii_to_coco_format,\
                                    symmetric_exchange_after_flip,\
                                    bbox_rectify

logger = logging.getLogger(__name__)


class dataset_(Dataset):

    def __init__(self, config, images_dir, annotions_path, mode='train', 
                                                            transform = None, 
                                                            dataset = None,
                                                            augment=False,
                                                            **kwargs):
        super(dataset_,self).__init__()

        name = config.model.dataset_name
        self.name = name
        # basic setting  
        logger.info("\n\t DATASET: {} \t Mode: {}\n".format(name, mode))
        assert name == 'coco' or name =='mpii'

        self.mode = mode
        self.score_threshold = config.test.bbox_score_threshold
        self.num_joints = config.model.keypoints_num
        self.input_size =  (config.model.input_size.w ,config.model.input_size.h) # w,h
        self.heatmap_size = (config.model.heatmap_size.w ,config.model.heatmap_size.h)
        self.transform = transform
        self.margin_to_border = config.model.margin_to_border  # input border/bbox border

        # data augmentation setting
        self.augment = augment
        self.aug_scale = config.train.aug_scale
        self.aug_rotation = config.train.aug_rotation
        self.flip = config.train.aug_flip
        self.random_occlusion = config.train.aug_occlusion
        if self.random_occlusion:
            self.occlusion_prob = config.train.occlusion_prob
            self.occlusion_size = config.train.occlusion_size
            self.occlusion_nums = config.train.occlusion_nums

        # use mask information or not
        self.use_mask = config.model.use_mask
        self.sigma_factor = config.train.heatmap_peak_sigma_factor
                # From COCO statistics to determine the sigma for each keypoint's heatmap gaussian
        self.sigmas = self.sigma_factor *np.array(
                    #[0.9,0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0]
                    [1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    )                                                                                     
        
             # load train or val groundtruth data
        if self.mode == 'train' or self.mode == 'val':
            self.data = self.dataset_groundtruth_setting(self.name, images_dir, annotions_path, mode, config)

            # load detection results
        if self.mode == 'dt':
            self.data = self.dataset_detection_setting(self.name, images_dir, annotions_path, mode, dataset)


        self.print_info()
        

    def dataset_groundtruth_setting(self, dataset_name, images_dir, annotions_path, mode, config):

        if dataset_name == 'coco':
            #self.kpts_weight = np.array([ 1., 1., 1., 1., 1., 1., 1., 
            #                            1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5],dtype=np.float32  )
            self.kpts_weight = np.array([ 1., 1., 1., 1., 1., 1., 1., 
                                        1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],dtype=np.float32  )
            self.images_root = os.path.join(images_dir ,mode +'2017')
            self.coco=COCO( os.path.join(annotions_path,'person_keypoints_{}2017.json'.format(mode)))   # train or val
            self.index_to_image_id = self.coco.getImgIds()    
            cats = [cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())]
            self.classes = ['__background__'] + cats
            logger.info("==> classes:{}".format(self.classes))
            logger.info("dataset:{} , total images:{}".format(mode,len(self.index_to_image_id)))
        
            return self.coco_get_gt_db()
        
        if dataset_name == 'mpii':
            # 0 - r ankle, 
            # 1 - r knee, 
            # 2 - r hip, 
            # 3 - l hip, 
            # 4 - l knee, 
            # 5 - l ankle,
            # 6 - pelvis, 
            # 7 - thorax, 
            # 8 - upper neck, 
            # 9 - head top, 
            # 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
            #self.kpts_weight = np.array([ 1.5, 1.3, 1.2, 1.2, 1.3, 1.5, 1.2, 
            #                            1., 1., 1., 1.3, 1.2, 1., 1., 1.2, 1.3],dtype=np.float32  )
            self.kpts_weight = np.array([ 1., 1., 1., 1., 1., 1., 1., 
                                        1., 1., 1., 1., 1., 1., 1., 1., 1.],dtype=np.float32  )
            self.images_root = images_dir
            self.annotations = json.load(open(os.path.join(annotions_path,'{}.json'.format(mode)),'r'))
            return self.mpii_get_db(mode)




    def dataset_detection_setting(self, dataset_name, images_dir, annotions_path, mode, dataset):

        if dataset_name == 'coco':
            self.images_root = os.path.join(images_dir,dataset +'2017')
            self.annotations = json.load( open(annotions_path,'r')) # dt.json
            return self.coco_get_dt_db()

        if dataset_name == 'mpii':
            self.images_root = images_dir
            #try:
                #self.annotations = json.load(os.path.join(annotions_path,'{}.json'.format(mode)))
            #except IOError :
            mode = dataset
            self.annotations = json.load(open(os.path.join(annotions_path,'{}.json'.format(mode)),'r'))
                #pass
    
            return self.mpii_get_db(mode)
        
        

    def print_info(self):

        logger.info('need the grountruth mask information == {}'.format(self.use_mask))

        if self.augment:
            logger.info( "augmentation is used for training" )
            logger.info("setting: scale=1±{},rotation=0±{},flip(p=0.5)={},random_occlusion={}"
                            .format(self.aug_scale,self.aug_rotation,self.flip,self.random_occlusion))
            if self.random_occlusion:
                logger.info("occlusion block: probability = {},size = {},block_nums= {}"
                            .format(self.occlusion_prob,self.occlusion_size,self.occlusion_nums))
            
        else:
            logger.info('augmentation is not used ')

        logger.info("dataset:{} , total samples: {}".format(self.mode,len(self.data)))
        logger.info('Initial:Standard deviation of gaussian kernel for different keypoints heatmaps is:\n==> {}'
                                                .format(self.sigmas))


    def get_image_path(self,file_name):
        image_path = os.path.join(self.images_root,file_name)
        return image_path

    def mpii_get_db(self,mode):
        "get mpii dataset"
        gt_db = []
        index = 0
        for a in self.annotations:
            
            file_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            left   = c[0]-0.5*s[0]*200/1.25
            top    = c[1]-0.5*s[1]*200/1.25
            right  = c[0]+0.5*s[0]*200/1.25 
            bottom = c[1]+0.5*s[1]*200/1.25
            bbox = [left,top,right-left,bottom-top]

            image_id = int(file_name.split('.jpg')[0])

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            keypoints = np.zeros((self.num_joints,  3), dtype=np.float)
            if mode != 'test':
                joints = np.array(a['joints'])

                ###########   mpii to coco order #######
                joints = mpii_to_coco_format(joints)
                ###################################
                
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]
                keypoints[:, 0:2] = joints[:, 0:2]
                keypoints[:, 2] = joints_vis[:]

            gt_db.append({
                'file_name': file_name,
                'center': c,
                'scale': s,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'keypoints': keypoints,
                'image_id': image_id,
                'index': index,
                'bbox':bbox
                })
            index +=1
        
        #container = select_data(gt_db)
        return gt_db
            
            

    def coco_get_dt_db(self,):
        "get coco detection results"

        score_threshold = self.score_threshold
        container = []
        index = 0
        logger.info("=> total bbox: {}".format(len(self.annotations)))
        for ann in self.annotations:

            image_id = ann['image_id']
            category_id = ann['category_id']
            bbox = ann['bbox']
            bbox[0],bbox[1],bbox[2],bbox[3] = int(bbox[0]),int(bbox[1]),int(bbox[2])+1,int(bbox[3])+1
            score = ann['score']
            if score < score_threshold or category_id != 1:
                continue
            file_name = '%012d.jpg' % image_id

            container.append({
                    'bbox':bbox,
                    'score':score,
                    'index':index,
                    'image_id':image_id,
                    'file_name':file_name,

                })
            index = index + 1

        return container


    def coco_get_gt_db(self,):
        "get coco groundtruth database"

        container = []
        index = 0
       
        for image_id in self.index_to_image_id:

            img_info = self.coco.loadImgs(image_id)[0]
            width = img_info['width']
            height = img_info['height']
            file_name = img_info['file_name']

            annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id,iscrowd=None))

            bbox_index = 0
            for ann in annotations:

                keypoints = ann['keypoints']
                if ann['area'] <=0 or  max(keypoints)==0 or ann['iscrowd']==1:
                    continue
                bbox = ann['bbox']
                
                if self.use_mask:
                    rle = mask.frPyObjects(ann['segmentation'], height, width)
                    seg_mask = mask.decode(rle)
                else:
                    seg_mask = ''

                bbox =bbox_rectify(width,height,bbox,keypoints)

                container.append({
                    'bbox':bbox,
                    'keypoints':keypoints,
                    'index':index,
                    'bbox_index':bbox_index,
                    'image_id':image_id,
                    'file_name':file_name,
                    'mask':seg_mask,

                })

                index = index + 1
                bbox_index = bbox_index + 1
        
        return container

    
    def __getitem__(self,id):

        data = self.data[id]
        keypoints = data['keypoints'] if 'keypoints' in data else ''
        bbox =      data['bbox']
        score =     data['score'] if 'score' in data else 1
        index =     data['index']
        file_name = data['file_name']
        image_id =  data['image_id']
        mask     =  data['mask'] if 'mask' in data else ''

        image_path = self.get_image_path(file_name)
        input_data = cv2.imread( image_path ) #(h,w,3)

        info = {}
        info['index'] = index

        # rotation and rescale augmentation 
        if self.mode == 'train' and self.augment == True:
            s = self.aug_scale 
            r = self.aug_rotation 
            aug_scale = np.clip(np.random.randn()*s + 1, 1-s, 1 + s) # 1-s,1+s
            aug_rotation = np.clip(np.random.randn()*r,  -2*r, 2*r)  if random.random() <= 0.6 else 0
        else:
            aug_scale = 1
            aug_rotation = 0
        
        # transform the image to adapt input size
        affine_matrix = make_affine_matrix(bbox,self.input_size,
                                                margin = self.margin_to_border,
                                                aug_scale=aug_scale,
                                                aug_rotation=aug_rotation,  )

        input_data = cv2.warpAffine(input_data,affine_matrix[[0,1],:], self.input_size,)

        if self.transform is not None:#  torchvision.transforms.ToTensor() and normalize()
                #  (H,W,3) range [0,255] numpy.ndarray  ==> (c,h,w) [0.0,1.0] =>[-1.0,1.0] torch.FloatTensor
            input_data = self.transform(input_data)

        # add mask information to dataset 
        if mask is not '':
            mask = cv2.warpAffine(mask      ,affine_matrix[[0,1],:], self.input_size) #note! mask:(h_,w_,1)->(h,w)
            mask = cv2.resize(mask,self.heatmap_size)
            mask = mask.astype(np.float32)
            mask = mask[np.newaxis,:,:]     #(1,h,w,x) or(1,h,w)

            # mask may be divided into several parts( ndim =4) we make them into a singel heatmap
            if mask.ndim == 4:
                mask = np.amax(mask,axis=3)

        # train mode  
        if self.mode == 'train':

            keypoints = self.kpt_affine(keypoints,affine_matrix)
            # flip with 0.5 probability
            if self.augment and self.flip and np.random.random() <= 0.5 and self.transform is not None :

                input_data = input_data.flip([2]) # dataloader does not support negative numpy index
                keypoints[:,0] = self.input_size[0] - 1 - keypoints[:,0] 
                keypoints = symmetric_exchange_after_flip(keypoints, self.name) 

                if mask is not '':
                    #mask = np.flip(mask,2) # dataloader does not support negative numpy index
                    mask = torch.from_numpy(mask).flip([2]).numpy()
            
            # Random Occlusion Augmentation strategy
            if self.random_occlusion:
                input_data,keypoints = Random_Occlusion_Augmentation(   input_data,  keypoints, self.name,
                                                                        probability = self.occlusion_prob,
                                                                        size = self.occlusion_size,
                                                                        block_nums=self.occlusion_nums,
                                                                        mode="specified_occlusion")

            heatmap_gt, kpt_visible = self.make_gt_heatmaps(keypoints)
            
            info['keypoints'] = keypoints
            info['prior_mask'] = mask
            
            return input_data , heatmap_gt, kpt_visible, info

        # valid or test mode
        if self.mode == 'val' or self.mode =='dt':
            info['prior_mask'] = mask
            #info['area'] = area
            return input_data , image_id  , score, np.linalg.inv(affine_matrix), np.array(bbox), info 


    def kpt_affine(self,keypoints,affine_matrix):
        '[17*3] ==affine==> [17,3]'

        keypoints = np.array(keypoints).reshape(-1,3)
        for id,points in enumerate(keypoints):
            if points[2]==0:
                continue
            vis = points[2] # prevent python value bug
            points[2] = 1
            keypoints[id][0:2] = np.dot(affine_matrix, points)[0:2]
            keypoints[id][2] = vis 

            if keypoints[id][0]<=0 or (keypoints[id][0]+1)>=self.input_size[0] or \
                    keypoints[id][1]<=0 or (keypoints[id][1]+1)>=self.input_size[1]:
                keypoints[id][0]=0
                keypoints[id][1]=0
                keypoints[id][2]=0

        return keypoints

    def make_gt_heatmaps(self,keypoints):
        """
        Generate `gt heatmaps` from keypoints coordinates 

        We can generate adaptive `kernel size` and `peak value` of `gaussian distribution`

        """
        
        heatmap_gt = np.zeros((len(keypoints),self.heatmap_size[1],self.heatmap_size[0]),dtype=np.float32)
        kpt_visible = np.array(keypoints[:,2])

        downsample = self.input_size[1] / self.heatmap_size[1]

        for id,kpt in enumerate(keypoints):
            if kpt_visible[id]==0:
                continue
            if kpt_visible[id]==1 or kpt_visible[id]==2:  # 1: label but invisible 2: label visible

                gt_x = min(int((kpt[0]/downsample+0.5)), self.heatmap_size[0]-1)
                gt_y = min(int((kpt[1]/downsample+0.5)), self.heatmap_size[1]-1)

                #sigma_loose = (2/kpt_visible[id])  # loose punishment for invisible label keypoints: sigma *2
                heatmap_gt[id,gt_y,gt_x] = 1
                heatmap_gt[id,:,:] = gaussian(heatmap_gt[id,:,:],sigma=self.sigmas[id])#*sigma_loose)
                amx = np.amax(heatmap_gt[id])
                heatmap_gt[id] /= amx  # make the max value of heatmap equal 1

                ### Note : label information should not changed! 
                #if self.random_occlusion:  
                    # reducing the max-value to represent low-confidence
                #    loose = 2/kpt_visible[id] # loose = 2: loose punishment for invisible label keypoints
                #    heatmap_gt[id] /= loose 

        kpt_visible = kpt_visible > 0
        kpt_visible = kpt_visible.astype(np.float32)
        #kpt_visible = self.kpts_weight * kpt_visible
        
        return heatmap_gt, kpt_visible
        

    def __len__(self,):

        return len(self.data)
    
    def augmentation_reset(self,aug_flip=None,
                                aug_scale=None,
                                aug_rotation=None,
                                aug_occlusion=None):
        
        if aug_flip is not None:
            self.flip = aug_flip  
        if aug_scale is not None:
            self.aug_scale = aug_scale
        if aug_rotation is not None:
            self.aug_rotation =  aug_rotation 
        if aug_occlusion is not None:
            self.random_occlusion = aug_occlusion 

        return None
    
    def update_sigma(self,sigma_factor):

        self.sigmas = sigma_factor *np.array(
                    #[0.9,0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0]
                    [1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    )                                                                                      
        logger.info('\nUpdate @ Standard deviation of gaussian kernel for different keypoints heatmaps is:\n==> {} '
                                                .format(self.sigmas))
        return self

    
