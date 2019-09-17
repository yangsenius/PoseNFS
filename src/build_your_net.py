#from thop import profile
import torch

from .architecture.meta_arch import Meta_Arch
from .network_factory.backbone_arch import Backbone_Arch
from .network_factory.subnetwork import Sub_Arch
from .network_factory.resnet_feature import BackBone_ResNet
from .network_factory.mobilenet_v2_feature import  BackBone_MobileNet
from .network_factory.part_representation import Body_Part_Representation
# from .network_factory.meta_pose_capsule_embedding import Meta_Pose_Capsule_Embedding
# from .network_factory.meta_pose_squashing import Meta_Pose_Squash

from utils import  get_model_summary

import logging
logger = logging.getLogger(__name__)

def bulid_up_network(config,criterion):

    # if config.model.use_backbone:
    #     logger.info("backbone of architecture is {}".format(config.model.backbone_net_name))

    if config.model.backbone_net_name=="resnet":
        backbone = BackBone_ResNet(config,is_train=True)

    if config.model.backbone_net_name=="mobilenet_v2":
        backbone = BackBone_MobileNet(config,is_train=True)

    if config.model.backbone_net_name=="meta_arch":
        logger.info("backbone:{}".format(config.model.backbone))
        backbone = Backbone_Arch(criterion,**config.model.backbone)

       
    Arch = Body_Part_Representation(config.model.keypoints_num,  criterion, backbone, **config.model.subnetwork_config)


    if config.model.use_pretrained:
        Arch.load_pretrained(config.model.pretrained)

    
        
    logger.info("\n\nbackbone: params and flops")
    logger.info(get_model_summary(backbone,torch.randn(1, 3, config.model.input_size.h,config.model.input_size.w)))

    logger.info("\n\nwhole architecture: params and flops")
    logger.info(get_model_summary(Arch,torch.randn(1, 3, config.model.input_size.h,config.model.input_size.w)))

    # flops, params = profile( backbone, input_size=(1, 3, config.model.input_size.h,config.model.input_size.w),  )
    # logger.info(">>> total params of BackBone: {:.2f}M\n>>> total FLOPS of Backbone: {:.3f} G\n".format(
    #                 (params / 1000000.0),(flops / 1000000000.0)))
    # flops, params = profile(Arch, input_size=(1, 3, config.model.input_size.h,config.model.input_size.w),  )
    # logger.info(">>> total params of Whole Model: {:.2f}M\n>>> total FLOPS of Model: {:.3f} G\n".format(
    #                     (params / 1000000.0),(flops / 1000000000.0)))

    return Arch
