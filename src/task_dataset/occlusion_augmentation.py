import numpy as np
import torch

def Random_Occlusion_Augmentation(input,keypoints,dataset_name,size=(21,21),probability=0.5,block_nums = 2,mode="black_occlusion"):
    """
    We adopt a `Random Occlusion Augmentation` strategy to make the model robust to  feature Ambiguity caused by 
    occlusion . For human pose estimation , the most important objective of learning is to learn the information or 
    knowledge of the constraint between skeleton joints and try to omit the image feature interference from other 
    objects . 
    
    In another word, the structure feature is more crucial than local image feature for human body keypoint detection

    We put a specified size `black block` to occlude the input image with certain probability in any region

    and if the groundtruth keypoint fall into the `black block` region, we make the heatmap peak value  of  
    corresponding keypoint equal to half the original value ,such as `reduce 1  to 0.5`  

    This `reducing max-value` method will be implemented in the function: `self.make_gt_heatmaps()`

                                __________
                            |    *     |
                            |   * [*]  |  
                            | *      * |
                            |  *[ ]*   |
                            |  *   *   |
                            |__________|   
    #############################
    **2019.3.31:
    add mode `exchange_occlusion`,`specified_occlusion`:

    consideration for factors like light condition variant, we modify the basic occlusion method:
    we aslo pick up two block region randomly, and judge 

        if (one contains some keypoints and another does not contain any kepypoints):
            exchange two block region image pixels, to make the region not containing kepypoints to occlude the choosen keypoints
        else :
            do nothinig
    
    add mode `specified_occlusion`:

    ##############################


    [ ] : means random `black block`
        *  : means keypoints
    Arg:    

        `input`:        (3,H,W)   
        `keypoints`:    (C,3)       c * [x , y, visibility]
        `size`:         a tuple of odd integer number such as (5,5) (3,5) 
        `probability`:  the probability of using `Random_Occlusion_Augmentation` for each sample
        `block_nums`:   how many blocks will be used in occlusion
    
    Return:

        Augmented `input`:(3,H,W) and `keypoints`: (C,3)
    
    """

    if not isinstance(input, np.ndarray):
        input = input.numpy()

    assert(size[0]<50 and size[1]<50, "block can not bigger than 50x50")
    assert(mode=="black_occlusion" or mode=="exchange_occlusion" or mode=="specified_occlusion")
    assert(dataset_name=='coco' or dataset_name=='mpii')
    if np.random.random() <= probability:
        w,   h = input.shape[2], input.shape[1]
        rx, ry = (size[0]-1)/2 , (size[1]-1)/2 # block radius
        
        if mode =="black_occlusion":

            margin = 30 # block center in margin of image border

            for num in range(block_nums):

            
                x,  y  = np.random.randint(margin,w-margin), np.random.randint(margin,h-margin) # block center position
                left ,top ,right, bottom = int(max(0,x-rx)), int(max(0,y-ry)), int(min(x+rx,w)), int(min(y+ry,h)) #block region
                
                input[:,top:bottom,left:right] = 0  # zero value

                # judge the keypoint's visibility  ; `*` here means `and` operation for bool array
                keypoints[:,2]= np.where((left <= keypoints[:,0])* (keypoints[:,0] <= right) *     
                                        (top <= keypoints[:,1] ) * (keypoints[:,1] <= bottom)*
                                        (keypoints[:,1]!=0) * (keypoints[:,0]!=0),  # consideration for keypoint [0,0,0]
                                        1 ,keypoints[:,2] )  # True: = 1 invisible keypoints    False: keep original 
        
        if mode == 'specified_occlusion':
            # for mpii  # 7 l wrist    8 r wrist  11 l knee     12-1  r knee    13 - l ankle    14  - r ankle 
            if dataset_name =='mpii':
                keypoints_id = [7,8,12,13,14]
            if dataset_name =='coco':
                keypoints_id = [9,10,13,14,15,16]

            for id in keypoints_id:

                if dataset_name=='mpii':
                    if keypoints[id,2]==0: # means this keypoint is not labled already
                        continue
                if dataset_name=='coco':
                    if keypoints[id,2]==0 or keypoints[id,2]==1: # means this keypoint is invisible already
                        continue

                margin = rx

                # choose a random region block from image to occlude the keypoint
                x,  y  = np.random.randint(margin,w-margin), np.random.randint(margin,h-margin) # block center position
                left ,top ,right, bottom = int(max(0,x-rx)), int(max(0,y-ry)), int(min(x+rx,w)), int(min(y+ry,h)) #block region
            
                # if the random region block from image contain anyone of kepypoints, do nothing and continue
                skip = False
                for p in keypoints:
                    if left<p[0]<right and top<p[1]<bottom:
                        skip = True
                        break
                if skip:
                    continue
                #if any(left< keypoints[:,0]) and any(keypoints[:,0]< right) and   any(top<keypoints[:,1]) and any(keypoints[:,1]<bottom) :
                    #continue     
                
                point_x, point_y = keypoints[id,0],keypoints[id,1]
                p_left ,p_top ,p_right, p_bottom = int(max(0,point_x-rx)), int(max(0,point_y-ry)), int(min(point_x+rx,w)), int(min(point_y+ry,h)) #blocked region
                #print("\n",point_x, point_y)
                #print(keypoints)
                # input[:,p_top:p_bottom,p_left:p_right] =  input[:,top:bottom,left:right] # occlude the keypoint by random region from image

                # two regions' areas may not be equal,so 
                # may  (top+p_bottom-p_top)>w or left+p_right-p_left)>h
                #if input[:,p_top:p_bottom,p_left:p_right].shape !=input[:,top:(top+p_bottom-p_top),left:(left+p_right-p_left)].shape:
                #    continue
                input[:,p_top:p_bottom,p_left:p_right] =  input[:,top:(top+p_bottom-p_top),left:(left+p_right-p_left)]



    
    input = torch.from_numpy(input)
    return input, keypoints