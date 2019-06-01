import numpy as np 

############################################################################################
############################################################################################
###########################  Utilization Functions    ######################################
############################################################################################

def make_affine_matrix(bbox, target_size, margin=1.2, aug_rotation= 0, aug_scale=1):
    """
    transform bbox ROI to adapat the net-input size

    `margin`: determine the distance between the bbox and input border . default: 1.2
    `aug_rotation`: the rotation angle of augmentation, range from [0,180]. default: 0
    `aug_scale`: the rescale size of augmentation . default: 1

                                target_size
                                __________
                            |  ______  |
                            |-|      |-|
                            |-| bbox |-|
                            |-|      |-|
                            |-|______|-|
                            |__________|

    t: 3x3 matrix, means transform BBOX-ROI to input center-roi

    rs: 3x3 matrix , means augmentation of rotation and scale   

    """

    (w,h)=target_size

    #choose small-proportion side to make scaling
    scale = min((w/margin) /bbox[2],
                (h/margin) /bbox[3])
    # 4:3 bbox area for NMS computing later
    #area = (w*h)/(scale*scale*margin*margin)

    # transform 
    t = np.zeros((3, 3))
    offset_X= w/2 - scale*(bbox[0]+bbox[2]/2)
    offset_Y= h/2 - scale*(bbox[1]+bbox[3]/2)
    t[0, 0] = scale
    t[1, 1] = scale
    t[0, 2] = offset_X
    t[1, 2] = offset_Y
    t[2, 2] = 1

    # augmentation
    theta = aug_rotation*np.pi/180
    alpha = np.cos(theta)*aug_scale
    beta = np.sin(theta)*aug_scale
    rs = np.zeros((3,3))
    rs[0, 0] = alpha
    rs[0, 1] = beta
    rs[0, 2] = (1-alpha)*(w/2)-beta*(h/2)
    rs[1, 0] = -beta
    rs[1, 1] = alpha
    rs[1, 2] = beta *(w/2) + (1-alpha)*(h/2)
    rs[2, 2] = 1
    
    # matrix multiply
    # first: t , orignal-transform
    # second: rs, augment scale and augment rotation
    final_matrix = np.dot(rs,t)
    return final_matrix  #, area


def mpii_to_coco_format(keypoints_original):
    #mpii:  # 0 - r ankle, 
            # 1 - r knee, 
            # 2 - r hip, 
            # 3 - l hip, 
            # 4 - l knee, 
            # 5 - l ankle,
            # 6 - pelvis, 
            # 7 - thorax, 
            # 8 - upper neck, 
            # 9 - head top, 
            # 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
    # simliar to coco:
    # 0-9head top,    1-8 upper neck ,     2-7thorax
    # 3-13 l shoulder    4-12 r shoulder     5-14 l elbow     6 -11 r elbow      7 -15 l wrist    8 -10 r wrist
    # 9-3 l hip    10-2 - r hip      11-4 - l knee     12-1  r knee    13 5 - l ankle    14 0 - r ankle  15 6 - pelvis
    parts = [[0,9],[1,8],[2,7],
            [3,13],[4,12],[5,14],[6,11],[7,15],[8,10],
            [9,3],[10,2],[11,4],[12,1],[13,5],[14,0],[15,6]]

    keypoints = keypoints_original.copy()

    for part in parts:
  
        keypoints[part[0],:] = keypoints_original[part[1],:].copy()

    return keypoints

def symmetric_exchange_after_flip(keypoints_flip, name):
    "flip will make the left-right body parts exchange"

    if name == 'mpii':
        # for original mpii format 
        #parts = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        # for mpii to coco like
        parts = [[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]

    elif name == 'coco':
        parts = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
    else:
        raise ValueError

    keypoints = keypoints_flip.copy()
    for part in parts:

        tmp = keypoints[part[1],:].copy()
        keypoints[part[1],:] = keypoints[part[0],:].copy()
        keypoints[part[0],:] = tmp

    return keypoints

def bbox_rectify(width,height,bbox,keypoints,margin=5):
        """
        use bbox_rectify() function to let the bbox cover all visible keypoints
        to reduce the label noise resulting from some visible (or invisible) keypoints not in bbox
        """
        kps = np.array(keypoints).reshape(-1, 3)   #array([[x1,y1,1],[],[x17,y17,1]]]
        # for label ：      kps[kps[:,2]>0]
        # for visibel label：  kps[kps[:,2]==2]
        border = kps[kps[:,2] >=1 ]
        if sum(kps[:,2] >=1) > 0:
            a, b = min(border[:,0].min(),bbox[0]), min(border[:,1].min(), bbox[1])
            c, d = max(border[:,0].max(),bbox[0]+bbox[2]), max(border[:,1].max(),bbox[1]+bbox[3])
            assert abs(margin)<20 ,"margin is too large"
            a,b,c,d=max(0,int(a-margin)),max(0,int(b-margin)),min(width,int(c+margin)),min(height,int(d+margin))

            return [a,b,c-a,d-b]
        else:
        	return bbox

# for mpii 
def select_data(db):

    pixel_std = 200
    db_selected = []
    for rec in db:
        num_vis = 0
        joints_x = 0.0
        joints_y = 0.0
        for joint, joint_vis in zip(
                rec['joints_3d'], rec['joints_3d_vis']):
            if joint_vis[0] <= 0:
                continue
            num_vis += 1

            joints_x += joint[0]
            joints_y += joint[1]
        if num_vis == 0:
            continue

        joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

        area = rec['scale'][0] * rec['scale'][1] * (pixel_std**2)
        joints_center = np.array([joints_x, joints_y])
        bbox_center = np.array(rec['center'])
        diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
        ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

        metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
        if ks > metric:
            db_selected.append(rec)

    logger.info('=> num db: {}'.format(len(db)))
    logger.info('=> num selected db: {}'.format(len(db_selected)))
    return db_selected
