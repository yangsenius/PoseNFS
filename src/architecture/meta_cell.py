## author : yangsen (yangsenius@seu.edu.cn)

import torch
import torch.nn as nn 
import torch.nn.functional as F
from architecture.operators import OPS , Connections , ReLUConvBN

# Note that !
# In paper, we don't take the previous previous cell's output to the current cell
# Therefore, we set `prev_prev_skip = False`
prev_prev_skip = False



# Ops_Combination is inside the cell

class Ops_Combination(nn.Module):
    "combine all specified operators by `alpha` weights to represent a layer "

    def __init__(self,channel,ops_used):
        super(Ops_Combination,self).__init__()

        self.ops = nn.ModuleList()
        self.ops_used = ops_used
        for name in ops_used:
            op = OPS[name](channel)  # only consideration the input channel in a cell
            if 'pool' in name:
                op = nn.Sequential(op, nn .BatchNorm2d(channel,affine = False))

            self.ops.append(op)
    
    def forward(self, input ,alphas):
        # output = sum(all of alpha *operator(x))
        
        # normalize
        alphas = F.softmax(alphas,dim=-1)
       
        output = len(self.ops_used) *sum([alpha * op(input) for alpha, op in zip(alphas, self.ops)]) #len_use
        return output




# Connection_Combination is outside the cell

class Connection_Combination(nn.Module):
    "combine 3 types of connection method by 'beta' weights to become an input node "

    def __init__(self,):
        super(Connection_Combination,self).__init__()
            
    def forward(self,  prev_parallel, prev_above, prev_below ,betas):

        betas = F.softmax(betas,dim=-1)
        mix = 3*betas[0] * prev_parallel + 3*betas[1] *  prev_above + 3*betas[2] *  prev_below   # *3

        mix = F.relu(mix)

        return mix


class Cell(nn.Module):

    """
    args:
    
        pos_i   :   determine the spatial size of the cell
        pos_j   :   determine the layer depth of the cell
        c       :   the channel of hidden states in current cell
        c_p     :   the channel of hidden states in previous cell
        c_pp    :   the channel of hidden states in previous previous cell

        input_nodes_num     :   the number of input_nodes = 2
        hidden_states_num   :   the number of hidden_states = 2
       
    
    input :  c_prev_parallel, c_prev_down, c_prev_up , c_prev_prev :[N,c_,h_,w_]   

            ★ we set c_*h_ = constant, so we can compute the (h_,w_) by c_ to determine which method to use to 
            keep the channel of input equal to the channel inside the cell
            

    alpha,          [k, ops_num]   k = total opertation edge in the cell
    beta,           [connection_num]

            

        hidden states : [N, c, h, w ]

    return :  output tensor [N, c , h, w]

    """

    def __init__(self,pos_i,pos_j, c,  c_prev_parallel, c_prev_above, c_prev_below , c_prev_prev ,hidden_states_num ,
                                                        input_nodes_num = 2, 
                                                        skip = prev_prev_skip,
                                                        operators_used = ["zero"]):
        
        super(Cell,self).__init__()
        self.pos = (pos_i,pos_j)
        self.cell_inner_channel = c
        self.steps = hidden_states_num
        self.input_num = input_nodes_num 

        # the architecture struture and parameters inside the cell
        self.cell_arch = nn.ModuleList()
        
        self.type_c = ['reduce_connect','direct_connect','upsampling_connect']

        self.cell_connections = Connection_Combination()

        # process the input from other cells ,the order is important
        self.skip = skip
        if skip: # if use prev_prev_cell or not
            input_source = [c_prev_parallel, c_prev_above, c_prev_below , c_prev_prev]
        else:
            input_source = [c_prev_parallel, c_prev_above, c_prev_below ]

        self.propress_list = nn.ModuleList()
        for id, i_c in enumerate(input_source):
            if i_c == c:
                self.propress_list.append( Connections['direct_connect'](c))
            elif i_c//c==2: # spatial size is half
                self.propress_list.append( Connections['upsampling_connect'](c))
            else:
                assert c//i_c==2 # spatial size is double
                self.propress_list.append( Connections['reduce_connect'](c))

        # 1x1 Conv to keep the output channel equal the cell channel after concate
        self.concat_flag = False
        if self.steps > 1:
            self.concat_flag = True
            
        self.channel_keep = nn.Conv2d(c*self.steps,c,1,1,0)

        for i in range(self.steps):
            # for each hidden states, 
            # it connect with all input nodes and all previous hidden states (if have)
            for j in range(self.input_num + i ):
                
                mix_ops = Ops_Combination(c,operators_used)
                self.cell_arch.append(mix_ops)

    def forward(self, prev_paral, prev_above, prev_below , prev_prev, alphas,  betas, other_input=None):
        
        # beta control
        
        prev_paral = self.propress_list[0](prev_paral)
        prev_above = self.propress_list[1](prev_above)
        prev_below = self.propress_list[2](prev_below)
        
        states = []
        input_node_1 = self.cell_connections(prev_paral, prev_above, prev_below ,betas  )
        states.append(input_node_1)

        if self.skip : # prev_prev_cell's input 
            prev_prev  = self.propress_list[3](prev_prev)
            states.append(prev_prev)
        else:
            if self.input_num >1 and other_input is not None:
                assert other_input.size()==prev_prev.size() , \
                            "if not skip and has two or more input, the extra input should keep the same size with prev_prev "
                other_information = self.propress_list[3](other_input)
                states.append(other_information)

        start = 0

        for i in range(self.steps):

            hidden_state = sum(

                [self.cell_arch[start + j](node, alphas[start + j]) for j , node in enumerate(states)] )

            start += len(states)
            states.append(hidden_state)

        if self.concat_flag:
            # concatenate all hidden states
            output_node = torch.cat(states[-self.steps:], dim = 1)
        else:
            output_node = states[-1]

            # c*hidden_states_num  -> c 
        output_node = self.channel_keep(output_node)
        

        return output_node
