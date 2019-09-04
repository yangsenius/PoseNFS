
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers ,heatmap_channels,feature_num=4,frozen_resnet=False,**kwargs):
        self.inplanes = 64
       # extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = False
        
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        assert feature_num>=0 and feature_num<=4
        self.feature_num = feature_num
        self.frozen_resnet = frozen_resnet

        if feature_num>=1:
            self.layer1 = self._make_layer(block, 64, layers[0])
        if feature_num>=2:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if feature_num>=3: 
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if feature_num==4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.frozen_resnet:
            for param in self.parameters():
                param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.feature_num >=1:
            f_4 = self.layer1(x)
        if self.feature_num >=2:
            f_8 = self.layer2(f_4 )
        if self.feature_num >=3:
            f_16 = self.layer3(f_8)
        if self.feature_num ==4:
            f_32 = self.layer4(f_16 )

        if self.feature_num==0:
            return x
        if self.feature_num==1:
            return f_4
        if self.feature_num==2:
            return f_4, f_8
        if self.feature_num==3:
            return f_4, f_8, f_16
        if self.feature_num==4:
            return f_4, f_8, f_16,f_32

    def init_weights(self, use_pretrained=True, pretrained=''):
        logger.info('=> init deconv weights from normal distribution')
        if use_pretrained == True:
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading resnet pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> no resnet imagenet pretrained model!')
            

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def BackBone_ResNet(config, is_train=True,**kwargs):
    
    layers = config.model.backbone_layers
    block_class, layers = resnet_spec[layers]

    resnet = ResNet(block_class, layers,config.model.keypoints_num,  frozen_resnet=config.model.frozen_resnet,   
                                feature_num=config.model.backbone_feature_num,**kwargs)

    if is_train and config.model.init_weights:
        resnet.init_weights(pretrained=config.model.backbone_pretrained_path)
    
    return  resnet

def test_BackBone(is_train=True,**kwargs):
    
    layers = 50
    block_class, layers = resnet_spec[layers]

    resnet = ResNet(block_class, layers,17, feature_num=4,**kwargs)
    
    return  resnet
