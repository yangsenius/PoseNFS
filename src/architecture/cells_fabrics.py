from architecture.meta_cell import Cell




#we design cells fabrics to represent multi-scales and multi-layers like below

#                           1   2   3   4   5   6   7
#    1/1       *
#    1/2            *
#    1/4                *   O   O   O   O   O   O   O
#    1/8                    O   O   O   O   O   O
#    1/16                       O   O   O   O
#    1/32                           O   O

#each "O" means a "Cell" , which has its position (i,j) denoted as Cell(i,j)  ,
#i denotes spatial size,
#j denotes layer depth.



def Constrcut_Cells_Fabrics(depth=8,
                            tpyes_=[4,8,16,32],
                            Channels=10*[4,8,16,32],
                            hidden_num=1, 
                            operators_used=['zero','skip_connect'], 
                            concat_factor = 1, 
                            skip = False ):
    """
    depth : layer nums
    tpyes_: the number of types of cell size  =[4,8,16,32]
    Channels : channel types for different cell size  =[4*N*F]
    hidden_num : the number of hidden states in the cell
    operators_used = ['zero','skip_connect',....]

    concat_factor: the factor of concatenating all hidden states inside the cell

    (★ ★ if the cell output channel has been reduced to be same as channel inside the cell, concat_factor=1 )
    """
    N = concat_factor
    Cell_Fabrics = []

    for j in range(depth):

        Cell_Layer = []
        # first layer has 2 cells, and add cells number in deeper layer within num_bound
        # the size is also controlled by depth//2
        # (depth-j):the layer has less cells in deeper depth and the final layer
        num_bound = min(j+2, len(tpyes_),depth-j)#,depth//2,(depth-j))
        for i in range( num_bound ):

            # for first layer
            if j == 0:
                # c means channel inside the cell
                c = Channels[i]

                c_prev_parallel  = c if i==0 else Channels[i-1]
                c_prev_above = c if i==0 else Channels[i-1]
                c_prev_below = c if i==0 else Channels[i-1]
                c_prev_prev = c if i==0 else Channels[i-1]

                Cell_Layer.append(Cell( i,j, c, c_prev_parallel, c_prev_above, c_prev_below , c_prev_prev,hidden_num,
                                            operators_used = operators_used))

            else:
                c = Channels[i]

                ################# for cell in prev_parallel #################
                # means channel from the output of previous-parallel cell
                # spacial consideration for bottom cell(num_bound-1,j)
                if i<len(Cell_Fabrics[j-1]):
                    c_prev_parallel = Channels[i]*N
                else:
                    c_prev_parallel = Channels[i-1]*N


                ################# for cell in previous-above #################
                # means channel from the output of previous-above cell
                # spacial consideration for top cell(0,j)
                c_prev_above = Channels[i-1]*N       if i!=0 else c_prev_parallel


                ################ for cell in previous-below #################
                if i<len(Cell_Fabrics[j-1])-1:# for cell in the tail
                    c_prev_below = Channels[i+1]*N
                else:
                    #means channel from the output of previous-below cell
                    # only when i<j and i!=num_bound-1(under diagonal not bottom),
                    # cell(i,j) has cell(j+1,i+1) from previous-below
                    c_prev_below = Channels[i+1]*N     if i<j and i!=num_bound-1 else c_prev_parallel


                ################ for cell in previous-previous-parallel  #################
                if num_bound == depth-j and num_bound<len(tpyes_):# for cell in the tail
                    c_prev_prev = Channels[i]*N
                else:
                    # means channel from the output of previous-previous-parallel cell
                    # only when j>i and j>1
                    # cell(i,j) has cell(i,j-2) otherwise c_prev_parallel
                    c_prev_prev = Channels[i]*N     if j>i and j>1  else c_prev_parallel


                Cell_Layer.append(Cell( i,j, c, c_prev_parallel, c_prev_above, c_prev_below , c_prev_prev,hidden_num,
                                            operators_used = operators_used))

        Cell_Fabrics.append(Cell_Layer)

    return Cell_Fabrics
