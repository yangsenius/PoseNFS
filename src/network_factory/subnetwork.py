import torch
import torch.nn as nn
import os
from architecture.meta_arch import Meta_Arch
import logging

logger = logging.getLogger(__name__)

######################  Sub_Arch  #############################
### for differen branch of different body parts
## cut the head and reserve the tail of the meta-arch

#                          4   5   6   7
#    1/1
#    1/2
#    1/4                   O   O   O   O
#    1/8                   O   O   O
#    1/16                  O   O
#    1/32                  O


class Sub_Arch(Meta_Arch):

    def __init__(self,group_out_dim,criterion,**cell_config):

        super(Sub_Arch,self).__init__(group_out_dim,criterion,**cell_config)

        self.cut_layers_num = cell_config['cut_layers_num']

        self.vector_in_pixel = cell_config['vector_in_pixel']

        self.vector_dim = cell_config['vector_dim']
        self.vector_conv_mode = cell_config['convolution_mode']

        # use backbone to replace the 3 layer cells ,so make cell_fabrics[0:cut_layers_num] == []
        for i,layer in enumerate(self.cell_fabrics):

            for j,cell in enumerate(layer):
                if i < cell_config['cut_layers_num']:
                    # delete cell in the range of (0,cut_layers_num)
                    # delele the attribute : self.cell_i_j (cell(nn.Module))

                    # self.cell_fabrics[i][j]=None  multi-gpus bugs
                    delattr(self,"cell_{}_{}".format(i,j))
                    self.Num[i] =0

        self.cells_num = sum(self.Num)
        self.alphas = nn.Parameter(torch.randn(self.k, self.num_ops))
        # beta control the fabrics outside the cell
        self.betas  = nn.Parameter(torch.randn(self.cells_num,  self.types_c))

        with torch.no_grad():
            # initialize to 0 value,and then use softmax to normalize to the same
            self.alphas.mul_(0.0)
            self.betas.mul_(0.0)

        self._arch_parameters =[]
        
        if self.search_alpha:
            self._arch_parameters .append(self.alphas)
        if self.search_beta:
            self._arch_parameters .append(self.betas)


       
       
        #self.final_layer_reduce = [self.Channels[0],self.out_dim]
        if self.vector_in_pixel :

            if self.vector_conv_mode=='2D':

                self.final_layer = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.Channels[0],self.out_dim*self.vector_dim,1,1,0))

            #if self.vector_conv_mode=='3D':

        else:
            
            self.final_layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(self.Channels[0],self.out_dim,1,1,0))
        

    ### put the feature map as the input of different body part module (or add extra_information from head or upper_limb part)
    def forward(self,f,extra_information=None):
        # Num indicate cell numbers in each layer
        # we use it to judge  prev_paral,prev_below,prev_prev
        Num = self.Num

        prev_prev_layer = [c for c in f]
        prev_layer = [c for c in f]

        # cell_fabrics = [layer1 , layer2, layer3,...]
        cell_id = 0
        for j in range(self.cut_layers_num,self.arch_depth): # begin from cut_layers_num

            # layer = self.cell_fabrics[j]

            OUTPUTS = []

            for i in range(self.Num[j]): # the number of cells in j-th layer of cell-fabric
                
                cell = eval('self.cell_{}_{}'.format(j,i))  # the j-th layer i-th scale of cell-fabric
                
                 # cells in layer 1,2 ,3 are replaced by resnet
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

             # final OUPUTS are upsampling to the 1/4 spatial size and concatenated
                            # if self.use_extra_information :
                            #     assert extra_information is not None
                            #     info = torch.cat([OUTPUTS[0],extra_information],dim=1)
                            # else:
            info = OUTPUTS[0]

        OUT  = self.final_layer(info)

        if self.vector_in_pixel:

            part_vector = OUT.permute(0,2,3,1) #[N,H,w,kpt_num*vetor_dim]
            
            part_vector = part_vector.view(part_vector.size(0),part_vector.size(1),part_vector.size(2),-1,self.vector_dim)

                # squash
            norm_in_vector = torch.norm(part_vector,dim=-1)  # [N,H,W,kpt_num]
            squash_prob= norm_in_vector**2/(norm_in_vector**2+1)
            OUT = squash_prob.permute(0,3,1,2)

        return OUT
