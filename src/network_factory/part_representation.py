import torch
import torch.nn as nn
import os
from network_factory.subnetwork import Sub_Arch
from network_factory.body_parts import parts_mode

import logging

logger = logging.getLogger(__name__)

class Body_Part_Representation(nn.Module):
    """
    associate each keypoint with specified human body part
    we make the whole body into P compositions (see: body_parts.py)
    each part associated with corresponding keypoints is regressed by a networks `Sub_Arch` seperately
    so each keypoint may be predicted by several sub-networks such as shoulder,elbow,hip,etc

    Using this method ,we can prior knowledge of constraint relationships abont human body skeleton
    """

    def __init__(self, out_dim, criterion, backbone, **subnetwork_config):

        super(Body_Part_Representation,self).__init__()

        self.subnetwork_config = subnetwork_config
        self.backbone = backbone
        self.criterion = criterion
        self.out_dim = out_dim

        backbone_feature_num = self.backbone.feature_num
        
        parts_num = subnetwork_config['parts_num']
        self.parts = parts_mode(subnetwork_config['dataset_name'],parts_num)

        logger.info("\nbody parts is {}".format(self.parts))

        self.groups = [] #nn.ModuleList()

        for part_name in self.parts:
            keypoints_num = len(self.parts[part_name])

            # make different networks to different body parts
            if not hasattr(self,'{}'.format(part_name)): 

                setattr(self,'{}'.format(part_name),

                        
                        Sub_Arch(   keypoints_num,
                                    criterion,
                                    name = part_name,
                                    **subnetwork_config['cell_config']))

                self.groups.append(eval('self.'+'{}'.format(part_name)))
                
        
        # consider for the backbone's feature map channels are too large
        with torch.no_grad():
            test = self.backbone
            test = test.cpu()
            feature = test(torch.randn(1,3,64,64))

        self.reduce_flag=[False]*len(self.parts)
        self.reduce = nn.ModuleList()
        
        # connect the backbone with different group in channel number
        # use conv_1x1 LEN
        for i in range(len(self.groups)):

            # put the feature map of backbone to repalce the specified position of Sub_Arch
            input_position = self.groups[i].cut_layers_num - 1

            # indicate the group how many feature maps are send to the group 
            self.groups[i].Num[ input_position ]  = backbone_feature_num

            # take Channels[0] to compare
            cell_channel = self.groups[i].Channels

            if feature[0].size(1)!= cell_channel[0]:
                self.reduce_flag[i]=True
                group_reduce_conv = nn.ModuleList()
                for id,f in enumerate(feature):
                    group_reduce_conv.append(
                        nn.Conv2d(f.size(1), cell_channel[id],1,1,0))
                self.reduce.append(group_reduce_conv)
            else:
                self.reduce.append(None)
    
    def forward(self,x):

        
        all_part_outputs = torch.zeros(
            size=(len(self.parts),x.size(0),self.out_dim, x.size(2)//4, x.size(3)//4)
            ).to(x.device)

        shared_feature = self.backbone(x)

        for id,part_name in enumerate(self.parts):
            
            if self.reduce_flag[id]==True:
                f = [self.reduce[id][f_id](ff) for f_id,ff in enumerate(shared_feature)]
                
            else:
                f = shared_feature

            
            all_part_outputs[id,:,self.parts[part_name],:,:] = eval('self.'+'{}'.format(part_name))(f)
        
        # sum some part represenatations 
        # for some keypoints sharing between parts, their prediction are summed (or averaged)
        output = torch.sum(all_part_outputs,dim=0)

        return output

    def new(self):
        """
        create a new model and initialize it with current arch parameters.
        However, its weights are left untouched.
        :return:
        """
        subnetwork_config = self.subnetwork_config
        model_new = Body_Part_Representation(self.out_dim, self.criterion,self.backbone, **subnetwork_config)

        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new


    def arch_parameters(self):

        self.all_group_arch_parameters = []

        for group in self.groups:

            if group.search_alpha:
                self.all_group_arch_parameters.append(group.alphas)
            if group.search_beta:
                self.all_group_arch_parameters.append(group.betas)

        if hasattr(self.backbone, "alphas"):
            if self.backbone.search_alpha:
                self.all_group_arch_parameters.append(self.backbone.alphas)
        if hasattr(self.backbone, "betas"):
            if self.backbone.search_beta:
                self.all_group_arch_parameters.append(self.backbone.betas)


        return self.all_group_arch_parameters

    def arch_parameters_random_search(self):

        for group in self.groups:

            # beta control the fabrics outside the cell

            group._arch_parameters =[]
            if group.search_alpha:
                group.alphas = nn.Parameter(torch.randn( group.k, group.num_ops))
                group._arch_parameters.append(group.alphas)
            if group.search_beta:
                group.betas  = nn.Parameter(torch.randn( group.cells_num, group.types_c))
                group._arch_parameters.append(group.betas)

        if hasattr(self.backbone, "alphas"):
            if self.backbone.search_alpha:
                self.all_group_arch_parameters.append(nn.Parameter(torch.randn(self.backbone.k, self.backbone.num_ops)))
        if hasattr(self.backbone, "betas"):
            if self.backbone.search_beta:
                self.all_group_arch_parameters.append(nn.Parameter(torch.randn(self.backbone.cells_num, self.backbone.types_c)))


    def loss(self,x,target,target_weight,info=None):

        kpts = self(x)
        loss = self.criterion(kpts, target ,target_weight)

        return loss

    def load_pretrained(self, pretrained=''):

        if os.path.exists(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_state_dict.items():
                if k in state_dict:
                    if 'final_layer' in k: # final_layer is excluded
                        continue
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.load_state_dict(state_dict)

            logger.info('=> loading pretrained model in {}'.format(pretrained))

        else:
            logger.info('=> no pretrained found in {}!'.format(pretrained))


    def _print_info(self):

        if hasattr(self.backbone,"_print_info"):
            self.backbone._print_info()
        logger.info("channels reduction is {}".format(self.reduce_flag))
        for g in self.groups:
            g._print_info()