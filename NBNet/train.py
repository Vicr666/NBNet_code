# train
import argparse
import math
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data import create_dataloader, create_dataset, DistIterSampler
from models import create_model
from utils import *
from options import *


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:

    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    if os.environ.get('RANK') is None:
        os.environ['RANK'] = '0'
    # # config port and server
    os.environ['MASTER_ADDR'] = 'region-3.seetacloud.com:13731'
    os.environ['MASTER_PORT'] = '5678'
    os.environ['WORLD_SIZE'] = '1'
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main(step=None, path=None):
    if path is None:
        opt_path = 'train_IRN_x4.yml'
    else:
        opt_path = path

    #### options
    parser = argparse.ArgumentParser()

    parser.add_argument('-opt', type=str, default="train_IRN_x4.yml")

    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args(args=[])
    opt = parse(args.opt, is_train=True)

    opt['train']['manual_seed'] = 4366
    if step is not None:
        opt['train']['lr_steps'] = step


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
    opt['logger']['save_checkpoint_freq'] = 1000  # print train info
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

        # config loggers. Before it, the log will not work
        setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
        setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base{}'.format(loggerIdx.log_idx))
        logger.info(dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                # from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base{}'.format(loggerIdx.log_idx))

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.6f}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.6f} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                torch.cuda.empty_cache()
                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_psnr_y = 0.0
                avg_ssim_y = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = tensor2img(visuals['SR'])  # uint8
                    gt_img = tensor2img(visuals['GT'])  # uint8

                    if not opt['train']['save_pic']:
                        lr_img = tensor2img(visuals['LR'])
                        gtl_img = tensor2img(visuals['LR_ref'])
                        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        mkdir(img_dir)

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                        save_img(sr_img, save_img_path)

                        # Save LR images
                        save_img_path_L = os.path.join(img_dir, '{:s}_forwLR_{:d}.png'.format(img_name, current_step))
                        save_img(lr_img, save_img_path_L)

                        # Save ground truth  first time
                        if current_step == opt['train']['val_freq']:
                            save_img_path_gt = os.path.join(img_dir, '{:s}_GT_{:d}.png'.format(img_name, current_step))
                            save_img(gt_img, save_img_path_gt)
                            save_img_path_gtl = os.path.join(img_dir,
                                                             '{:s}_LR_ref_{:d}.png'.format(img_name, current_step))
                            save_img(gtl_img, save_img_path_gtl)

                    # calculate PSNR
                    crop_size = opt['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    avg_ssim += calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)

                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    cropped_sr_img_y = sr_img_y[crop_size:-crop_size, crop_size:-crop_size]
                    cropped_gt_img_y = gt_img_y[crop_size:-crop_size, crop_size:-crop_size]
                    avg_psnr_y += calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    avg_ssim_y += calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx

                avg_psnr_y = avg_psnr_y / idx
                avg_ssim_y = avg_ssim_y / idx

                # log
                logger.info('# Validation # PSNR: {:.6f}. SSIM: {:.6f}. PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.format(avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y))
                logger_val = logging.getLogger('val{}'.format(loggerIdx.log_idx))  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.6f}. ssim: {:.6f}. PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.format(
                    epoch, current_step, avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)

                    tb_logger.add_scalar('psnr_y', avg_psnr_y, current_step)
                    tb_logger.add_scalar('ssim_y', avg_ssim_y, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    loggerIdx.log_idx += 1
    main()  #9