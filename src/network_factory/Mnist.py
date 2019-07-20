import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.operators import OPS , Connections
from architecture.cells_fabrics import Constrcut_Cells_Fabrics
from architecture.meta_cell import prev_prev_skip

import logging
logger = logging.getLogger(__name__)  

class mnist_arch(nn.Module):

    def __init__(self,out_dim,name="mnist",**Cell_Config):

        super(mnist_arch, self).__init__()


        self.arch_name = name
        self.out_dim = out_dim
        self.cell_config = Cell_Config
        self.arch_depth = Cell_Config['depth']
        self.cell_size_types = Cell_Config['size_types']

        self.N = Cell_Config['hidden_states_num']
        self.F = Cell_Config['factor']
        self.I = Cell_Config['input_nodes_num']

        self.search_alpha = Cell_Config['search_alpha']
        self.search_beta = Cell_Config['search_beta']
        self.operators_used = Cell_Config['operators']

        # channel number varys in different cell size types (not in depth)
        self.Channels = [i*self.F for i in self.cell_size_types]

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.Channels[0], kernel_size=1, stride=1, padding=0,bias=False))

        cell_fabrics = Constrcut_Cells_Fabrics( depth=self.arch_depth,  tpyes_=self.cell_size_types, 
                                                Channels=self.Channels, hidden_num=self.N, operators_used=self.operators_used )


        self.cell_fabrics =[]
        cell_id = 1
        Num = [] # Num is a list store the number of cells in each layer j
        # we use this method to make the cell(nn.Module) registered to the model.parameters()
        for i,layer in enumerate(cell_fabrics):
            l = []
            Num.append(len(layer))
            for j,cell in enumerate(layer):
                if not hasattr(self,'cell_{}_{}'.format(i,j)):  # self.cell_0_1,self.cell_2,...
                    setattr(self,'cell_{}_{}'.format(i,j) ,cell) # self.cell_1 =cell(,,)
                l.append(eval('self.'+'cell_{}_{}'.format(i,j)))
                cell_id += 1

            self.cell_fabrics.append(l)

        ############
        self.Num = Num # N is a list store the number of cells in each layer j
        self.cells_num = sum(Num)
        self.k = sum(1 for i in range(self.N) for j in range(self.I + i)) #  2  input nodes
        self.num_ops = len(self.operators_used) # 8
        self.types_c = len(Connections)

        # alpha control the fabrics inside the cell
        self.alphas = nn.Parameter(torch.zeros(self.k, self.num_ops))
        # beta control the fabrics outside the cell
        self.betas  = nn.Parameter(torch.zeros(self.cells_num,  self.types_c))
        # initialize all to 0 value,and then use softmax to normalize
        # note: alphas and betass are registered to the model.parameters()

        self._arch_parameters =[]
        if self.search_alpha:
            self._arch_parameters .append(self.alphas)
        if self.search_beta:
            self._arch_parameters .append(self.betas)

        # the output channel of last cell layer
        self.final_layer_reduce = [self.Channels[0],self.out_dim]

        c  = self.Channels[0]

        self.conv_resize = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c*2, 3, 2, 1),
                nn.Conv2d(c*2, c*4, 3, 2, 1)
            )

        self.fc1 = nn.Linear(c*4*7*7, 128)     # input size = 28x28  here:7x7
        self.fc2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(128,self.out_dim)
            )


    def forward(self,x):

        # Num indicate cell numbers in each layer
        # we use it to judge  prev_paral,prev_below,prev_prev
        Num = self.Num
        x = self.stem(x)

      
        prev_prev_layer = []
        prev_layer = []

        # cell_fabrics = [layer1 , layer2, layer3,...]
        cell_id = 0
        for j in range(self.arch_depth):

            layer = self.cell_fabrics[j]
            OUTPUTS = []

            for i, cell in enumerate( layer):

                if j==0: # for first layer all input come from x
                    output = cell(x, x, x , x,self.alphas, self.betas[cell_id]) # cancel prev_prev!!
                    cell_id += 1
                    OUTPUTS.append(output)
                    prev_prev_layer.append(output)
                    prev_layer.append(output)

                else: 
                    if i<Num[j-1]:
                        prev_paral = prev_layer[i]
                    else :
                        prev_paral = prev_layer[i-1] # else for bottom cell

                    prev_above = prev_layer[i-1] if i!=0 else prev_paral # else for top cell
                    prev_below = prev_layer[i+1] if i<Num[j-1]-1 else prev_paral # if for cell above the diagnoal
                    prev_prev = prev_prev_layer[i] if j>2 and j>i else prev_paral # if for 2 cell !!!

                    output = cell(prev_paral, prev_above, prev_below , prev_prev,
                                    self.alphas, self.betas[cell_id])
                    cell_id += 1

                    OUTPUTS.append(output)

            prev_prev_layer = prev_layer
            prev_layer = OUTPUTS

        info = OUTPUTS[0]
        OUT = self.conv_resize(info)

        OUT = OUT.view(OUT.size(0), -1)

        OUT = self.fc1(OUT)
        OUT = self.fc2(OUT)
        #OUT = F.log_softmax(OUT, dim=1)

        return OUT
# class mnist_arch(nn.Module):

#     def __init__(self,out_dim,name="mnist",**Cell_Config):

    
#         super(mnist_arch, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#         # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )

#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 3 * 3, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )

#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

#     # Spatial transformer network forward function
#     def stn(self, x):
#         xs = self.localization(x)
#         xs = xs.view(-1, 10 * 3 * 3)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)

#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)

#         return x

#     def forward(self, x):
#         # transform the input
#         x = self.stn(x)

#         # Perform the usual forward pass
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
        
#         return F.log_softmax(x, dim=1)

    def new(self):
        """
        create a new model and initialize it with current arch parameters.
        However, its weights are left untouched.
        :return:
        """
        cell_config = self.cell_config
        model_new = mnist_arch(self.out_dim, self.criterion, **cell_config)

        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new


    def arch_parameters(self):
        return self._arch_parameters

    def arch_parameters_random_search(self):

        self._arch_parameters =[]

        if self.search_alpha:
            self.alphas = nn.Parameter(torch.randn( self.k, self.num_ops))
            # beta control the fabrics outside the cel
            self._arch_parameters .append(self.alphas)
        if self.search_beta:
            self.betas  = nn.Parameter(torch.randn( self.cells_num, self.types_c))

            self._arch_parameters .append(self.betas)

    #################  show information function #######################################
    def _print_info(self):
        logger.info("\n========================== {} Architecture Configuration ======================".format(self.arch_name))
        logger.info("Macrostructure:: Channels for cell in different size {}={} ".format(self.cell_size_types, self.Channels))
        logger.info("Macrostructure:: total number of cells is {}".format( self.cells_num))
        logger.info("Macrostructure:: the number of cells in each layer of cell_fabrics(after cut) are {}, total={} layers".format( self.Num,self.arch_depth))
        logger.info("Macrostructure:: channel reduction of the final layer architecture is  {}".format(self.final_layer_reduce))
        logger.info("Macrostructure:: Search Space of BETA is {}, optimization is {}\n".format(self.betas.shape,self.search_beta))

        logger.info("inside the cell:: Search Space of ALPHA is {}, optimization is {}".format(self.alphas.shape,self.search_alpha))
        logger.info("inside the cell:: operators used  in each  node are {}".format( self.operators_used))
        logger.info("inside the cell:: the number of hidden states is {}".format( self.N))
        logger.info("inside the cell:: the factor of channel numbers is {}".format( self.F))
        logger.info("inside the cell:: the number of input nodes  is {}".format( self.I))
        logger.info("inside the cell:: the prev——prev——skip connect is {}\n".format( prev_prev_skip))
        logger.info(">>> total params of Model: {:.2f}M".format(sum(p.numel() for p in self.parameters()) / 1000000.0))
        self._show_alpha(original_value=True)
        self._show_beta(original_value=True)
        logger.info("=========================================================================++++++++++++")

    def _show_alpha(self,original_value=False):

        # normalize or not
        if original_value:
            value = self.alphas.data.tolist()
        else:
            value = F.softmax(self.alphas,dim=-1).data.tolist()

        logger.info("[==={}==>]alpha values(normalize:{}) in a cell: operators' coefficient are:"
                    .format(self.arch_name,not original_value))

        if self.alphas.size(0)==1:
            operators = [(x,round(y,3)) for (x,y) in zip(self.operators_used,value[0])] #squeeze
            logger.info("=>the single edge is mixed in:{}" .format(operators))
        if self.alphas.size(0)>1:

            for id,alpha in enumerate(value):
                operators =[(x,round(y,3)) for (x,y) in zip(self.operators_used,alpha)]
                logger.info("=>the {} edge is mixed in:{}" .format(id+1,operators))

    def _show_beta(self,original_value=False):

        if original_value:
            value = self.betas.data.tolist()
        else:
            value = F.softmax(self.betas,dim=-1).data.tolist()

        logger.info("[==={}==>]beta values(normalize:{}): multi-scales  coefficient are:" 
                    .format(self.arch_name,not original_value))
        for id,betas in enumerate(value):
            l=  [round(x,3) for x in  betas ]
            logger.info("=>the number {} cell's multi-scales input is mixed with:{}" .format(id+1,l))