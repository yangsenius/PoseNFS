import torch
from utils import AverageMeter,save_batch_image_with_joints
from timeit import default_timer as timer
import logging
import os

logger = logging.getLogger(__name__)

def train(epoch, train_queue, arch_queue ,model,search_arch,criterion,optimizer,lr,search_strategy,
            output_dir, logger, config, args):
    # when the search_strategy is `None` or `sync `, the arch_queue is None

    loss = AverageMeter()
    model = model.train()
    if arch_queue is not None:
        valid_iter = iter(arch_queue)

    # only update W for several epoches before update the alpha
    NAS_begin = config.train.arch_search_epoch

    current_search_strategy = search_strategy if epoch >= NAS_begin else 'None'

    logger.info("Current Architecture Search strategy is {} . *{} Search begin in epoch: {}"
                    .format(current_search_strategy,   config.train.arch_search_strategy,NAS_begin))

    for iters, (x, train_heatmap_gt, train_kpt_visible,  train_info) in enumerate(train_queue):


        start = timer()
        optimizer.zero_grad()

        x = x.cuda(non_blocking=True)
        train_heatmap_gt = train_heatmap_gt.cuda(non_blocking=True)
        train_kpt_visible = train_kpt_visible.float().cuda(non_blocking=True)

        if  search_strategy=='first_order_gradient' or search_strategy=='second_order_gradient':

            x_valid , valid_heatmap_gt, valid_kpt_visible, valid_info = next(valid_iter)
            x_valid = x_valid.cuda(non_blocking=True)
            valid_heatmap_gt = valid_heatmap_gt.cuda(non_blocking=True)
            valid_kpt_visible = valid_kpt_visible.cuda(non_blocking=True)

            search_arch.step(x, train_heatmap_gt, train_kpt_visible, x_valid, valid_heatmap_gt, valid_kpt_visible, lr, optimizer,
                             search_strategy= current_search_strategy,
                             # Note : NOT `search_strategy`, because `current_search_strategy` can be none in early epoches
                             weight_optimization_flag = config.train.arch_search_weight_optimization_flag)

        
        kpts = model(x)
        backward_loss = criterion(kpts,train_heatmap_gt.to(kpts.device), train_kpt_visible.to(kpts.device))

        #backward_loss = model.loss(x, train_heatmap_gt, train_kpt_visible,info=train_info)
        
        backward_loss.backward()

        loss.update(backward_loss.item(), x.size(0))

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        optimizer.step()
        time = timer() - start

        
       
        if iters % 100 == 0:

            # torch.cuda.empty_cache()

            if args.debug:

                save_batch_image_with_joints(   x,
                                                train_info['keypoints'],
                                                train_kpt_visible.unsqueeze(-1).cpu(),
                                                os.path.join(output_dir,'debug_image_'+str(iters)))
                #logger.info(train_heatmap_dt[0],train_heatmap_dt[3])
                #visualize_heatamp(x,train_heatmap_dt, os.path.join(output_dir,'debug_'+str(iters)))

            #embedding_loss = model.embedding_loss if hasattr(model,'embedding_loss') else 0

            logger.info('epoch: {}   \titers:[{}|{}]   \tloss:{:.6f}({:.5f})  \tfeed-speed:{:.2f} samples/s' #  \tembedloss:{:.8f}'
                        .format(epoch,iters,len(train_queue),loss.val, loss.avg ,len(x)/time))#,embedding_loss))

        if iters % 1000 == 0 and args.show_arch_value and hasattr(model,"groups"):
            model.groups[0]._show_alpha()
            model.groups[1]._show_alpha()
            model.groups[2]._show_alpha()

    if args.show_arch_value or epoch % 10==0: # alpha and beta will be constant when nas is `none`
        logger.info("=========>current architecture's values before evaluate")
        if hasattr(model,"backbone"):
            if hasattr(model.backbone,"alphas"):
                model.backbone._show_alpha(original_value=True)
                model.backbone._show_beta(original_value=True)
            for g in model.groups:
                g._show_alpha(original_value=False)
                g._show_beta(original_value=False)
        # #model.groups[0]._show_alpha(original_value=True)
        # model.groups[1]._show_alpha()
        # model.groups[1]._show_beta()
        # #model.groups[1]._show_alpha(original_value=True)
        # model.groups[2]._show_alpha()
        # model.groups[2]._show_beta()
        # #model.groups[2]._show_alpha(original_value=True)
