# Does human brain prune the useless neural cells?

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

class Empty_Cell(nn.Module):
    r"""
    This class is used to replace the useless `Cell` object whose output has no computing contributions
    for the next layer's cells (i.e. all the beta parameters associated with it are zero values).

    Args: like the `__init__` of the `Cell` Module
    
        pos_i   :   determine the spatial size of the cell
        pos_j   :   determine the layer depth of the cell
        c       :   the channel of hidden states in the current cell

    Input: imitate the input of the `forward` function of the `Cell` Module

    Return: [N, c, H, W]
    """
    def __init__(self,pos_i,pos_j,c):
        super(Empty_Cell, self).__init__()
        self.output_channel = c
        self.pos=(pos_i, pos_j)

    def forward(self, prev_paral, prev_above, prev_below , prev_prev, alphas,  betas, other_input=None):
        N, H, W = prev_paral.size(0), prev_paral.size(2), prev_paral.size(3)
        output = torch.zeros(
            size=(N, self.output_channel, H, W), 
            device=prev_paral.device,
            dtype=betas.dtype)
        return output

def associated_cell_is_useful(a_i, a_j, type_c, pos_i, pos_j, cell_id, useful_cell_positions, useful_cell_ids, betas):
    r"""
    `Input`:
        `a_i`: the i-position of the associated cell of the current cell(pos_i, pos_j)
        `a_j`: the j-position of the associated cell of the current cell(pos_i, pos_j)
        `type_c`: the type-id of the connected association [0,1,2] (direct_connect, reduce, upsampling)

        `pos_i, pos_j, cell_id --> the curret cell`

        `useful_cell_positions`: a list recording the position (i,j) of cells which have been regarded as useful.
        `useful_cell_ids`: a list recording the cell-id (betas index) of cells which have been regarded as useful.

        `betas`: `[cell_num, 3]`

    `Output`:
        if the associated cell is existing and useful:
            return `True`
        else:
            return `False`
    """
    if (a_i, a_j) in useful_cell_positions:
        _c_id = useful_cell_ids[ useful_cell_positions.index((a_i, a_j)) ]
    else:
        return False
    associated_beta = betas[_c_id][type_c] 
    if associated_beta > 0:
        useful_cell_positions.append((pos_i, pos_j))
        useful_cell_ids.append(cell_id)
        return True
    else:
        return False

from architecture.meta_arch import Meta_Arch

def Prune_the_Useless_Cells(arch):

    for name, cnf in arch.named_modules():
        # hasattr(cnf, 'betas'): # prune all the useless cells in the `Sub_Arch` module (subnetworks, CNFs)
        if not isinstance(cnf, Meta_Arch): 
            continue
        depth = cnf.arch_depth
        betas = F.softmax(cnf.betas,dim=-1) # [cells_num, 3]
        Num = cnf.Num.copy() # not change the original num
        cells_num = cnf.cells_num
        if sum(Num) != cells_num:
            # in part_representation, we add the number of backbone feature pyrmiads in the cut layer position
            Num[cnf.cut_layers_num - 1] = 0 
        useful_cell_positions = []
        useful_cell_ids = []
        id = 0
        logger.info(betas)
        # if a cell is useful for comptuing, 
        # we must find one path from it to the final cell with non-zero associated beta values.
        # so we need to make judgement from the final layer to the previous layers by the reverse 'cell_id' order
        # if the cell is connected by a useful cell in the next layer, it will be useful
        for pos_j_, num in enumerate(Num[::-1]): # Reverse order
            # num==0: the layer has no cells
            if num ==0: 
                continue

            pos_j = (depth-1) - pos_j_ # absolute position of layer
            for pos_i_ in range(num): 
                pos_i = (num - 1) - pos_i_ # Reverse order
                id +=1
                cell_id = cells_num - id # cell_id: index in `betas`

                if pos_i ==0 and pos_j == (depth-1): # num==1: the cell in the final layer will always be preserved.
                    useful_cell_positions.append((pos_i, pos_j))
                    useful_cell_ids.append(cell_id)
                else:
                    # associated cells array: (pos_i, pos_j, beta_type_id)
                    associations = [(pos_i-1, pos_j+1, 2), (pos_i, pos_j+1, 0), (pos_i+1, pos_j+1, 1)] 
                    # (pos_i, pos_j) is the prev_below  the (pos_i-1, pos_j+1) tpec_c = 2
                    # (pos_i, pos_j) is the prev_paral  the (pos_i, pos_j+1)  type_c = 0
                    # (pos_i, pos_j) is the prev_above  the (pos_i+1, pos_j+1) type_c =1
                    
                    prune_the_cell = True 
                    
                    for (x,y,t) in associations:
                        if associated_cell_is_useful(x, y, t, 
                                                    pos_i, pos_j, cell_id, 
                                                    useful_cell_positions, useful_cell_ids, betas):
                            # once the cell associated it is useful, then this cell is regarded as useful
                            # so do not prune this cell
                            prune_the_cell = False
                            break
                        
                    if prune_the_cell:
                        if hasattr(cnf, 'cell_{}_{}'.format(pos_j, pos_i)):
                            output_channel = eval('cnf.'+'cell_{}_{}'.format(pos_j, pos_i)).cell_inner_channel
                            setattr(cnf, 'cell_{}_{}'.format(pos_j, pos_i), Empty_Cell(pos_i, pos_j, output_channel))
                            logger.info('Cell Pruning... xxx===>000: replace the [{}-CNF]-[cell_{}_{}] by a empty cell'.format(name, pos_j, pos_i))

    return arch

from architecture.operators import *
from architecture.meta_cell import Cell

def Prune_the_Useless_Operations(Arch):
    for n, m in Arch.named_modules():
        # find all the 'Sub_Arch'
        #if hasattr(m, 'alphas'):
        if isinstance(m, Meta_Arch):
            # one-shot-search, 
            # alphas = [h(h+1)/2, the candidate operation numbers]; 
            # h: the number of hidden nodes
            alphas = F.softmax(m.alphas, dim=-1)
            
            for nn, mm in m.named_modules():
                # find all the existing useful `Cell` modules in the `Sub_Arch`
                if hasattr(mm, 'cell_arch'):
                    for e, alpha_operations in enumerate(alphas):
                        for o, alpha in enumerate(alpha_operations):
                            if not isinstance(mm.cell_arch[e].ops[o], Zero) and alpha==0:
                                mm.cell_arch[e].ops[o] = Zero()
                                logger.info(
                                    "Operation Pruning... &&&-->000: replace the [{}-CNF]-[{}]-[edge{}]-[{}]-th operation by a Zero operation"
                                                .format(n,nn,e,o))
    
    return Arch

def Prune(Arch, prune_cells=True, prune_operations=True):
    if prune_cells:
        Arch = Prune_the_Useless_Cells(Arch)
    if prune_operations:
        Arch = Prune_the_Useless_Operations(Arch)
    return Arch

            

        


