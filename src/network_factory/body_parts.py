# change the order to be simliar to coco:
# 0-9 head top     1-8 upper neck ,     2-7thorax
# 3-13 l shoulder    4-12 r shoulder     5-14 l elbow     6 -11 r elbow      7 -15 l wrist    8 -10 r wrist
# 9-3 l hip    10-2 - r hip      11-4 - l knee     12-1  r knee    13 5 - l ankle    14 0 - r ankle  15 6 - pelvis

mpii_keypoints_id = {}

mpii_keypoints_id['head_top'] = 0
mpii_keypoints_id['upper_neck']= 1
mpii_keypoints_id['throax'] = 2
mpii_keypoints_id['l_shoulder'] = 3
mpii_keypoints_id['r_shoulder'] = 4
mpii_keypoints_id['l_elbow'] = 5
mpii_keypoints_id['r_elbow'] = 6
mpii_keypoints_id['l_wrist'] = 7
mpii_keypoints_id['r_wrist'] = 8
mpii_keypoints_id['l_hip'] = 9
mpii_keypoints_id['r_hip'] = 10
mpii_keypoints_id['l_knee'] = 11
mpii_keypoints_id['r_knee'] = 12
mpii_keypoints_id['l_ankle'] = 13
mpii_keypoints_id['r_ankle'] = 14
mpii_keypoints_id['pelvis'] = 15

coco_keypoints_id = {}
coco_keypoints_id['nose'] = 0
coco_keypoints_id['l_eye']= 1
coco_keypoints_id['r_eye'] = 2
coco_keypoints_id['l_ear'] = 3
coco_keypoints_id['r_ear'] = 4
coco_keypoints_id['l_shoulder'] = 5
coco_keypoints_id['r_shoulder'] = 6
coco_keypoints_id['l_elbow'] = 7
coco_keypoints_id['r_elbow'] = 8
coco_keypoints_id['l_wrist'] = 9
coco_keypoints_id['r_wrist'] = 10
coco_keypoints_id['l_hip'] =11
coco_keypoints_id['r_hip'] = 12
coco_keypoints_id['l_knee'] = 13
coco_keypoints_id['r_knee'] = 14
coco_keypoints_id['l_ankle'] = 15
coco_keypoints_id['r_ankle'] = 16


def parts_mode(dataset_name,parts_num):

    if dataset_name=='coco':
        k =  lambda name : coco_keypoints_id[name]
        parts_names = coco_keypoints_id
    elif dataset_name=='mpii':
        k =  lambda name : mpii_keypoints_id[name]
        parts_names = mpii_keypoints_id
    else:
        print("wrong dataset name")
        raise ValueError

    parts = {}
    assert parts_num == 1 or parts_num == 3 or parts_num == 8 or parts_num ==5

    if parts_num ==1:
        parts['whole_body'] = [k(name) for name in parts_names]
    
    ######################
    if parts_num == 3:

        parts['upper_limb_part'] = [k('l_shoulder'),k('r_shoulder'),k('l_elbow'),k('r_elbow'),k('l_wrist'),k('r_wrist')]
        
        if dataset_name == 'mpii':
            parts['head_part'] = [k('head_top'), k('upper_neck'),  k('throax')]                  
            parts['lower_limb_part'] = [k('l_hip'), k('r_hip'), k('l_knee'), k('r_knee'), k('l_ankle'), k('r_ankle'), k('pelvis')]
        if dataset_name == 'coco':
            parts['head_part'] =  [k('nose'), k('l_eye'),k('r_eye'),k('l_ear'),k('r_ear'),]
            parts['lower_limb_part'] = [k('l_hip'),k('r_hip'), k('l_knee'), k('r_knee'),k('l_ankle'),k('r_ankle'),] 

    if parts_num == 5:

        parts['left_arm_part'] = [k('l_elbow') ,k('l_wrist')]
        parts['right_arm_part'] = [k('r_elbow'), k('r_wrist')]
        parts['legs_part'] = [k('l_knee'),k('r_knee'),k('l_ankle'),k('r_ankle')]

        if dataset_name == 'mpii':
            parts['head_shoulder_part'] = [k('head_top'),k('upper_neck'),k('throax'),k('l_shoulder'),k('r_shoulder') ]
            parts['hips_part'] = [k('pelvis'), k('l_hip'), k('r_hip')]   
        
        if dataset_name == 'coco':
            parts['head_shoulder_part'] = [k('nose'), k('l_eye'),k('r_eye'),k('l_ear'),k('r_ear'), k('l_shoulder'),k('r_shoulder') ]
            parts['hips_part'] = [k('l_hip'), k('r_hip')] 

    if parts_num == 8:
        
        parts['left_upper_arm_part'] = [k('l_shoulder'),k('l_elbow')]
        parts['right_upper_arm_part'] = [k('r_shoulder'),k('r_elbow') ]
        parts['left_lower_arm_part'] = [k('l_elbow'),k('l_wrist')]
        parts['right_lower_arm_part'] = [k('r_elbow'),k('r_wrist')]
        parts['left_lower_leg_part'] = [k('l_knee'), k('l_ankle')]
        parts['right_lower_leg_part'] = [k('r_knee'),k('r_ankle')]
        
        if dataset_name == 'mpii':
            parts['head_shoulder_part'] = [k('head_top'),k('upper_neck'),k('throax'),k('l_shoulder'),k('r_shoulder') ]
            parts['upper_legs_part'] = [k('l_hip'),k('pelvis'),k('r_hip'),k('l_knee'),k('r_knee')]
        if dataset_name =='coco':      
            parts['head_shoulder_part'] = [ k('nose'),k('l_eye'),k('r_eye'),k('l_ear'),k('r_ear'),k('l_shoulder'),k('r_shoulder') ]
            parts['upper_legs_part'] = [k('l_hip'),k('r_hip'),k('l_knee'),k('r_knee')]   

    return parts