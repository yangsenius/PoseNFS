import torch
import torch.nn as nn
import os
from architecture.meta_arch import Meta_Arch
import logging

logger = logging.getLogger(__name__)

##########################   Backbone_Arch  ##############################
########### for the backbone of different human body parts ########
# such as #
#                           1   2   3
#    1/1       *
#    1/2            *
#    1/4                *   O   O   O
#    1/8                    O   O   O
#    1/16                       O   O
#    1/32                           O

class Backbone_Arch(Meta_Arch):
    """
    reserve_layers_num: first several layers , default  3 just like the above

    """
    def __init__(self,criterion, is_train=True,**backbone_config):

        super(Backbone_Arch,self).__init__(1,criterion,name="backbone",**backbone_config)
        # only use the first three layers as backbone

        self.reserve_layers_num = backbone_config['reserve_layers_num']
        self.feature_num = self.Num[self.reserve_layers_num-1]

        for i,layer in enumerate(self.cell_fabrics):

            for j,cell in enumerate(layer):
                if i >  self.reserve_layers_num-1:
                    # delete cell in the range of (reserve_layers_num,len(layer))
                    # delele the attribute : self.cell_i_j (cell(nn.Module))
                    self.cell_fabrics[i][j]=None
                    delattr(self,"cell_{}_{}".format(i,j))
                    self.Num[i] =0


        self.cells_num = sum(self.Num)
        self.alphas = nn.Parameter(10*torch.randn(self.k, self.num_ops))
        # beta control the fabrics outside the cell
        self.betas  = nn.Parameter(10*torch.randn(self.cells_num,  self.types_c))

        with torch.no_grad():
            # initialize to 0 value,and then use softmax to normalize to the same
            self.alphas.mul_(0.0)
            self.betas.mul_(0.0)

        self._arch_parameters =[]
        if self.search_alpha:
            self._arch_parameters .append(self.alphas)
        if self.search_beta:
            self._arch_parameters .append(self.betas)

        self.final_layer = None

        if backbone_config['frozen_backbone']:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self,x):

        # Num indicate cell numbers in each layer
        # we use it to judge  prev_paral,prev_below,prev_prev
        Num = self.Num

        x = self.stem(x)


        prev_prev_layer = []

        prev_layer = []
        # cell_fabrics = [layer1 , layer2, layer3,...]
        cell_id = 0
        for j in range(self.reserve_layers_num):

            layer = self.cell_fabrics[j]
            OUTPUTS = []

            for i, cell in enumerate( layer):

                if j==0: # for first layer all input come from x
                    output = cell(x, x, x , x,self.alphas, self.betas[cell_id]) # cancel prev_prev!!
                    cell_id += 1
                    OUTPUTS.append(output)
                    prev_prev_layer.append(output)
                    prev_layer.append(output)

                else: # cells in layer 1,2 are special
                                     # consider for last several layer
                    if i<Num[j-1]:
                    # if prev_cell exist
                    #if i<Num[j-1]:

                        prev_paral = prev_layer[i]
                    else :
                        prev_paral = prev_layer[i-1] # else for bottom cell

                    prev_above = prev_layer[i-1] if i!=0 else prev_paral # else for top cell
                    prev_below = prev_layer[i+1] if i<Num[j-1]-1 else prev_paral # if for cell above the diagnoal
                    prev_prev = prev_prev_layer[i] if j>2 and j>i else prev_paral # if for 2 cell !!!

                    #prev_prev = prev_prev_layer[i] if i<Num[j-2] else prev_paral
                    #prev_prev = torch.zeros_like(prev_prev) # cancel prev_prev!!
                    output = cell(prev_paral, prev_above, prev_below , prev_prev,
                                    self.alphas, self.betas[cell_id])
                    cell_id += 1

                    OUTPUTS.append(output)
                    #prev_layer.append(output)

            prev_prev_layer = prev_layer
            prev_layer = OUTPUTS

        return OUTPUTS