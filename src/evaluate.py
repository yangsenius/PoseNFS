# Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)

import torch
import numpy as np
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from collections import defaultdict
from collections import OrderedDict

from timeit import default_timer as timer
import tqdm
import json
import logging
logger = logging.getLogger(__name__)


def inference(model ,input, affine_matrix_inv, info , upsample, flip_test=False ,
                                                                with_mask = False, 
                                                                use_refine = False ,
                                                                boosting = None,
                                                                dataset_name ='coco'):
    input = input.cuda()
    
    heatmap_dt = model(input)
    z = heatmap_dt.clone()
    # flip-test
    if flip_test:
        input_filp = input.flip([3])
        heatmap_dt_flip = model(input_filp)
        # need to change the heatmap order
        heatmap_dt_flip = symmetric_exchange_after_flip(heatmap_dt_flip, dataset_name)
        heatmap_dt = ( z  + heatmap_dt_flip.flip([3]) ) * 0.5


    # from heatmap compute keypoints coordinate
    heatmap_dt = heatmap_dt.cpu()
    coordinate_argmax, maxval =  get_final_coord( heatmap_dt,) #post_processing = True)

    # [N,kpts_num,3]
    pred_kpt = compute_orignal_coordinate(  affine_matrix=affine_matrix_inv, 
                                            heatmap_coord=coordinate_argmax,
                                            up = upsample ,)
                                            #bounding = bbox) this is not good !
    return pred_kpt, maxval

##################### evaluate #####################
def evaluate( model , dataloader , config, output_dir, with_mask = False, use_refine = False ,boosting = None):
    model.eval()

    flip_test = config.test.flip_test
    logger.info("flip test is {}".format(flip_test))
    dataset_name = 'coco' #default
    dataset_name = config.model.dataset_name
    

    upsample = config.model.input_size.w/config.model.heatmap_size.w

    # for coco
    predict_results =[]
    imgs_ids = []
    # for mpii
    all_preds = np.zeros((len(dataloader.dataset), config.model.keypoints_num, 3), dtype=np.float32)
    idx = 0

    
    #####   output results in val-or-test dataset  #########
    with torch.no_grad():

        for id ,(input,image_id ,score ,affine_matrix_inv, bbox, info) in enumerate(dataloader):

            start = timer()
            num_images = input.size(0)
            pred_kpt, maxval = inference(model,input,affine_matrix_inv,info,upsample,flip_test=flip_test,
                                                                                    with_mask = False, 
                                                                                    use_refine = False ,
                                                                                    boosting = None,
                                                                                    dataset_name = dataset_name)
            time = timer() - start

            # for mpii
            imgs_ids.extend(image_id.tolist())
            all_preds[idx:idx + num_images, :, 0:2] = pred_kpt[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2] = maxval[:,:]
            idx += num_images
            # for coco
            pred_kpt[:,:,2] = maxval[:,:] # regard max-value as score
            pred_kpt = pred_kpt.numpy().round(3).reshape(len(pred_kpt),-1)
            
            index = info['index']
            #area = info['area']

            # for coco
            for i in range(len(input)):

                predict_results.append({
                    'image_id': image_id[i].item(),
                    'keypoints':pred_kpt[i].tolist(),
                    'score': score[i].item(),
                    'index':index[i].item(),
                    'bbox':bbox[i].tolist(),
                    #'area':area[i].item()
                })

            
            if id % 100 ==0:
                logger.info("evaluate: iters[{}/{}], inference-speed: {:.2f} samples/s"
                                        .format(id,len(dataloader),len(input)/time))

    if dataset_name =='mpii':

        name_values, results = mpii_eval(config, all_preds, output_dir)

        model_name = 'pose_net'
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        return results
        
    if dataset_name =='coco':

        logger.info('\n==> the number of predict_results samples is {} before OKS NMS'.format(len(predict_results)))

        ##########  analysize the oks metric
        image_kpts = defaultdict(list)

        # 1 image_id <==> n person keypoints results
        for sample in predict_results:
            image_kpts[sample['image_id']].append(sample)
        logger.info('==> oks_nms ...')
        logger.info('==> images num: {}'.format(len(image_kpts)))

        ##### oks nms ####
        begin = timer()
        dt_list = []
        for img in image_kpts.keys():

            kpts = image_kpts[img]

            # score = bbox_score * kpts_mean_score
            for sample in kpts:
                bbox_score = sample['score']

                kpt_scores = sample['keypoints'][2::3]
                high_score_kpt = [s for s in kpt_scores if s > config.test.confidence_threshold]
                kpts_mean_score = sum(high_score_kpt)/(len(high_score_kpt)+1e-5)
                sample['score'] = bbox_score * kpts_mean_score

            # oks nms
            kpts = oks_nms(kpts, config.test.oks_nms_threshold)
            #kpts = oks_nms_sb(kpts, config.test.oks_nms_threshold)

            #if all(a['score']<= 0.2 for a in kpts):
            #    continue  #skip peroson bbox with low-score!

            #score_list = [a['score'] for a in kpts]
            #if sum(score_list)/(len(score_list)+1e-8) <=0.10:
            #    continue

            for sample in kpts:

                image_id = sample['image_id']
                keypoints = sample['keypoints']
                score =sample['score']
                tmp = {'image_id':image_id, 'keypoints':keypoints,'category_id':1,'score':score }

                dt_list.append(tmp)

        logger.info('\n==>  the number of predict_results samples is {} after OKS NMS , consuming time = {:.3f} \n'.format(len(dt_list),timer()-begin))

        name_values, AP = coco_eval(config,dt_list,output_dir)

        model_name = 'pose_net_for'+ dataset_name
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        return AP





def coco_eval(config, dt, output_dir):
    """
    Evaluate the result with COCO API
    """

    gt_path = os.path.join( config.annotation_root_dir,'person_keypoints_val2017.json')
    dt_path = os.path.join(output_dir,'dt.json')

    with open(dt_path, 'w') as f:
        json.dump(dt, f)
        
    logger.info('==>dt.json has been written in '+ os.path.join(output_dir,'dt.json'))

    coco = COCO(gt_path)

    coco_dets = coco.loadRes(dt_path)
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
   
    #coco_eval.params.imgIds  =  image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

    info_str = []
    for ind, name in enumerate(stats_names):
        info_str.append((name, coco_eval.stats[ind]))

    name_value = OrderedDict(info_str)

    return  name_value,  name_value['AP']




from scipy.io import loadmat, savemat

def mpii_eval(config, dt, output_dir):

    # convert 0-based index to 1-based index
        preds = dt[:, :, 0:2] + 1.0

        ## change coco-like order to mpii order

        for id,item in enumerate(preds):
            preds[id] = coco_like_to_mpii_format(item.copy())

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'pred': preds})

        #if 'test' in config.DATASET.TEST_SET:
        #    return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(config.annotation_root_dir, 'gt_val.mat')
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

###########################################################################################
###########################  Post Process Utilization Functions    ########################
###########################################################################################

def get_max_coord(target):

    '''target = [N,C,H,W]
    heatmap max position to coordinate '''
    N = target.size()[0]
    C = target.size()[1]
    W = target.size()[-1]

    target_index = target.view( N ,C ,-1)
    # max_index = torch.argmax(target_index,1,keepdim =True)  #如果值全为0，会返回最后一个索引，那么要对其置零
    max_value ,  max_index = torch.max (target_index,2) #元组1

    max_index = max_index * torch.tensor(torch.gt(max_value,torch.zeros(size = max_index.size())).numpy(),dtype = torch.long) #
    #如果值全为0，会返回最后一个索引，那么要判断是否大于0对其置零 类型转换

    max_index_view = max_index.view(N,C)

    keypoint_x_y = torch.zeros((N,C,2),dtype = torch.long)

    keypoint_x_y[:,:,0] = max_index_view[:,:] % W # index/48 = y....x 商为行数，余为列数 target.size()[-1]=48
    keypoint_x_y[:,:,1] = max_index_view[:,:] // W # index/48 = y....x 商为行数，余为列数
    #max_value = max_value.view(N,C,1)
    return keypoint_x_y, max_value


def get_final_coord(batch_heatmaps,post_processing = True):
    '''simple baseline coordinates offset'''

    coords, maxvals = get_max_coord(batch_heatmaps)
    heatmap_height = batch_heatmaps.size()[2]
    heatmap_width = batch_heatmaps.size()[3]

    if post_processing:
        for n in range(coords.size()[0]):
            for p in range(coords.size()[1]):
                hm = batch_heatmaps[n][p]
                px = int(np.floor(coords[n][p][0] + 0.5))
                py = int(np.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = torch.tensor([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords = coords.float()
                    coords[n][p] += torch.sign(diff) * .25

    preds = coords.clone()

    return preds, maxvals

def compute_orignal_coordinate(affine_matrix, heatmap_coord, up = 4, bounding= None):
    '''
    A = [j,n,m]  B = [j,m,p]
    [j,n,p]= [j,n,m] * [j,m,p]

    C = torch.matmul(A,B) = [j,n,p]

    A : `affine_matrix` :[N,3,3]
    B : `orig_coord`:    [N,17,3] -> [N,3,17]
    C : `affine_coord`:      [N,3,3]*[N,3,17] = [N,3,17]  =>[N,17,3]

    return C

    Note: `bounding` is prevent the final coordinates falling outside the image border

    `bounding` = [left_x,up_y, w, h] like `bbox`

    '''

    N = heatmap_coord.size()[0]
    C = heatmap_coord.size()[1]

    heatmap_coord = up * heatmap_coord + 0.5
    orig_coord = torch.ones(N,C,3)

    orig_coord[:,:,0:2] = heatmap_coord[:,:,0:2]
    orig_coord = orig_coord.permute(0,2,1)

    #[N,3,17]    =              [N,3,3]   matmul   [N,3,17]
    affine_coord = torch.matmul(affine_matrix.float(), orig_coord.float())

    affine_coord = affine_coord.permute(0,2,1)
    
    if bounding is not None: # this is not good
        # restrict the keypoints falling into bbox
        for i in range(len(affine_coord)):
            affine_coord_x = affine_coord[i,:,0].clamp(bounding[i,0],bounding[i,0]+bounding[i,2])
            affine_coord_y = affine_coord[i,:,1].clamp(bounding[i,1],bounding[i,1]+bounding[i,3])
            affine_coord[i,:,0] = affine_coord_x
            affine_coord[i,:,1] = affine_coord_y

    return affine_coord



def oks_nms(kpts_candidates, threshold):
    """
    keypoints : data-format: [17,3]

    `NMS algorithm`:

    index = 0

    while(1):

        step: `1.` take highest score keypoints(=index) as groudtruth 'gt', judege if break or continue

        step: `2.` supression all other keypoints 'kpt' and pop it from the list, judege if computeOKS(gt,kpt) > threshold  

            (note: supression is a inner-while)

                id = index + 1
                while(1):

                    if break or continue
                    if computeOKS(gt,kpt(id)) > threshold
                    list.pop()
                    id +=1

                index +=1

        step: `3.` goto step `1`

    """
    # data format [51] ->[num,3]
    if len(np.array(kpts_candidates[0]['keypoints']).shape) !=2:
        for q,k in enumerate(kpts_candidates):
            kpts_candidates[q]['keypoints'] = np.array(k['keypoints']).reshape(-1,3)

    # sort the list by keypoints scores, from bigger to smaller
    kpts_order = sorted(kpts_candidates, key=lambda tmp:tmp['score'], reverse=True)

    # nms #
    index = 0
    while True:

        if index >=len(kpts_order):
            break

        gt = kpts_order[index]['keypoints']

        bbox = kpts_order[index]['bbox']
        gt_area = bbox[2]*bbox[3]
        #gt_area = kpts_order[index]['area'] #see COCO_Dataset.make_affine_matrix

        id = index + 1
        while True:

            if id >=len(kpts_order):
                break
            kpt = kpts_order[id]['keypoints']

            if ComputeOKS(gt,kpt,gt_area) > threshold:
                kpts_order.pop(id)

            id += 1

        index += 1

    # numpy  [:,17,3] --> list [:,51]
    for q,k in enumerate(kpts_order):
        kpts_order[q]['keypoints'] = np.array(k['keypoints']).reshape(-1).round(3).tolist()

    return kpts_order


def ComputeOKS(dt_keypoints,gt_keypoints,gt_area,invisible_threshold = 0):
    """
    Args:   dt_keypoints = [17,3]
            gt_keypoints = [17,3]

    Return:
            oks = [17]
            sum_oks = sum(oks)/sum(label_gt)

    """
    #print(dt_keypoints,gt_keypoints,gt_area)
    dt_keypoints = np.array(dt_keypoints)
    gt_keypoints = np.array(gt_keypoints)
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
    oks = np.zeros(k)
        # compute oks between each detection and ground truth object
    label_num_gt = gt_keypoints[:,2]
    dx =  dt_keypoints[:,0]-gt_keypoints[:,0]
    dy =  dt_keypoints[:,1]-gt_keypoints[:,1]

    oks = (dx**2 + dy**2) / vars / (gt_area+np.spacing(1)) / 2

    #print(oks)
    oks = np.exp(-oks)
    sum_oks = np.sum((label_num_gt>0)*oks)/(sum(label_num_gt>invisible_threshold)+1e-5)
    #print(label_num_gt>0)
    label_num = sum(label_num_gt>0)
    return  np.round(sum_oks,4)


############  simple baseline contrast ######
def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious

def oks_nms_sb(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'] for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['bbox'][2]*kpts_db[i]['bbox'][3] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]

        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]
    
    kpts_left=[]

    for ke in keep:
        kpts_left.append(kpts_db[ke])

    return kpts_left

def coco_like_to_mpii_format(keypoints_original):
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
  
        keypoints[part[1],:] = keypoints_original[part[0],:].copy()

    return keypoints

def symmetric_exchange_after_flip(heatmap_flip,name):

    if name == 'mpii':
        # for original mpii format 
        #parts = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        # for mpii to coco like
        parts = [[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]
    elif name == 'coco':
        parts = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
    else:
        raise ValueError

    heatmap = heatmap_flip.clone()
    for part in parts:

        tmp = heatmap[:,part[1],:,:].clone()
        heatmap[:,part[1],:,:] = heatmap[:,part[0],:,:].clone()
        heatmap[:,part[0],:,:] = tmp

    return heatmap

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info("results on dataset ...\n")
    logger.info(
        '| Name ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |\n'  )
