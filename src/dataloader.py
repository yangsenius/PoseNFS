import torch
import torchvision
from task_dataset.dataset import dataset_
from torch.utils.data import DistributedSampler
import logging
logger = logging.getLogger(__name__)

data_normalize = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

    
def Dataloaders(search_strategy,config,arg):

    train_dataset = dataset_( config, config.images_root_dir,
                                config.annotation_root_dir,
                                mode ='train',
                                augment = config.train.augmentation,
                                transform = data_normalize )

    valid_dataset = dataset_(config,config.images_root_dir,
                                config.annotation_root_dir,
                                mode ='val',
                                transform = data_normalize)




    if search_strategy in ['None','sync','random'] :
        return normal_dataloader(train_dataset,valid_dataset,config,arg)
    else:
        return split_for_nas(train_dataset,valid_dataset,config,
                            split_for_train = config.train.split_for_train,
                            split_for_valid = config.train.split_for_archvalid)
        
    #return train_queue,arch_queue,valid_queue
    
        





def split_for_nas(train_dataset,valid_dataset,config,split_for_train=2,split_for_valid=1):
    assert type(split_for_train)==type(split_for_valid)==int
    assert split_for_train>=split_for_valid

     # keep the same iterations
    factor = int(split_for_train/split_for_valid)
    
    data_nums = list(range(len(train_dataset)))

    train_split = int(len(train_dataset)*split_for_train/(split_for_train+split_for_valid))
    valid_split = len(train_dataset)-train_split
    
    logger.info("\nfor weight optimization and architecture search")
    logger.info("split trainset into: train/val(train for arch_parameters) is [{}/{}],number = [{}/{}] "
        .format(split_for_train,split_for_valid,train_split,valid_split))
    
    train_batch = factor*config.train.batchsize
    valid_batch = config.train.batchsize
    train_iters = round(train_split/train_batch)
    valid_iters = round(valid_split/valid_batch)
    print(valid_iters,train_iters)
    assert train_iters==valid_iters
    
    logger.info("batchsize  for train={} val={}(train for arch_parameters)".format(train_batch,valid_batch))
    logger.info("iterations for train={} val={}(train for arch_parameters)".format(train_iters,valid_iters))


    num_workers = config.num_workers
    pin_memory = True
    logger.info("\n num_workers of dataloader is {}".format(num_workers))

    train_queue = torch.utils.data.DataLoader(train_dataset, 
                batch_size = train_batch, num_workers = num_workers,   pin_memory=pin_memory ,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(data_nums[:train_split]), )
    
    arch_queue = torch.utils.data.DataLoader(train_dataset, 
                batch_size = valid_batch,  num_workers = num_workers ,   pin_memory=pin_memory ,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(data_nums[train_split:]), )

    valid_queue = torch.utils.data.DataLoader(valid_dataset, 
                batch_size = config.test.batchsize, shuffle = False , num_workers = num_workers , pin_memory=pin_memory )
    
    return train_queue, arch_queue , valid_queue  


def normal_dataloader(train_dataset,valid_dataset,config,arg):
    
    num_workers = config.num_workers
    pin_memory = True
    logger.info("\n num_workers of dataloader is {}".format(num_workers))


    if arg.distributed:
        train_dist_sampler =  DistributedSampler(train_dataset)
        #valid_sampler_dist =  DistributedSampler(valid_dataset)
        
    else:
        train_dist_sampler = None

    train_queue = torch.utils.data.DataLoader(train_dataset, 
                    batch_size = config.train.batchsize, 
                    num_workers = num_workers ,   
                    pin_memory=pin_memory , 
                    shuffle = (train_dist_sampler is None), 
                    sampler= train_dist_sampler
                    )

    valid_queue = torch.utils.data.DataLoader(valid_dataset, 
                    batch_size = config.train.batchsize, 
                    num_workers = num_workers ,   
                    pin_memory=pin_memory , 
                    shuffle = False, )

    

    if arg.distributed:
        return train_queue ,None, valid_queue ,train_dist_sampler
    else:
        return train_queue ,None, valid_queue