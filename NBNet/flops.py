# models
import logging
import math
import os
import time
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import argparse
import math
from DWT_IDWT_layer import DWT_2D, IDWT_2D
from utils import *

from atten import GFEB
from edge import CannyDetector
from einops import rearrange
# compute_flops_thop.py
import torch
from thop import profile, clever_format
from models import define_G  # 根据你仓库的导出方式调整

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-opt', type=str, default="train_IRN_x4.yml")

    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args(args=[])
    opt = parse(args.opt, is_train=True)
    
    opt = "train_IRN_x4.yml"

    opt['train']['manual_seed'] = 4366

    opt['gpu_ids'] = [0]
    opt['datasets']['train']['n_workers'] = 32


    opt['use_Norm_Layer'] = True

    # Loss Options
    opt['train']['pixel_criterion_kl'] = 'KL'
    opt['train']['lambda_rec_kl'] = 1

    # batch size to 16
    #opt['datasets']['train']['batch_size'] = 1 # 16  # TODO: 000000 BatchSize

    opt['train']['save_pic'] = True

    # Rectify for debug
    # opt['train']['niter'] = 50000  # per 5000 iters saves a model and test model
    opt['logger']['print_freq'] = 100  # print train info
    opt['logger']['save_checkpoint_freq'] = 100  # print train info
    opt['train']['val_freq'] = 100  # print validation info

    # for swi transformer windows size
    opt['network_G']['window_size'] = 8

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists
            mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                    and 'pretrain_model' not in key and 'resume' not in key))

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)
    # 如果你的 define_G 依赖完整 opt 结构，请用实际 opt
    netG = define_G(opt)
    netG.eval()
    input = torch.randn(1, 3, 144, 144)  # 根据你模型的输入分辨率
    flops, params = profile(netG, inputs=(input,), verbose=False)
    print('Raw FLOPs (thop):', flops)
    print('Params:', params)
    print('Nice format:', clever_format([flops, params], '%.3f'))

if __name__ == '__main__':
    main()