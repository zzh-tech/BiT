import yaml
import torch
import random
import time
import os
import importlib
import torchmetrics
import os.path as osp
import torch.distributed as dist
import numpy as np
from einops import rearrange
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model.utils import Logger, AverageMeter, init_seeds
from datetime import datetime

if __name__ == '__main__':
    """
    cmd:
    python -m torch.distributed.launch --nproc_per_node=4 train_bit.py --config ./configs/bit_adobe240.yaml
    python -m torch.distributed.launch --nproc_per_node=4 train_bit.py --config ./configs/bit++_adobe240.yaml
    python -m torch.distributed.launch --nproc_per_node=4 train_bit.py --config  ./configs/bit_rbi.yaml
    python -m torch.distributed.launch --nproc_per_node=4 train_bit.py --config  ./configs/bit++_rbi.yaml
    """
    parser = ArgumentParser(description='Blur Interpolation Transformer')
    parser.add_argument('--config', default='./configs/bit_adobe240.yaml', help='path of config')
    parser.add_argument('--port', type=str, default=None, help='port number')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args = parser.parse_args()

    # parse cfgs
    with open(args.config) as f:
        cfgs = yaml.full_load(f)
    train_cfgs = cfgs['train_args']

    # ddp initialization
    torch.backends.cudnn.benchmark = True
    if args.port is not None:
        os.environ['MASTER_PORT'] = args.port
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f'local_rank: {local_rank}')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_seeds(seed=local_rank)
    dist.init_process_group(backend="nccl")

    # create model
    model_cls = getattr(importlib.import_module('model'), cfgs['model_args']['name'])
    resume_from = None
    if 'resume_from' in train_cfgs:
        resume_from = train_cfgs['resume_from']
    model = model_cls(**cfgs['model_args']['args'],
                      optimizer_args=cfgs['optimizer_args'],
                      scheduler_args=cfgs['scheduler_args'],
                      loss_args=cfgs['loss_args'],
                      local_rank=local_rank,
                      load_from=train_cfgs['load_from'],
                      resume_from=resume_from)

    # create dataloaders
    # create training dataloader
    train_dataset_cfgs = cfgs['train_dataset_args']
    train_dataset = getattr(importlib.import_module('data'), train_dataset_cfgs['name'])(**train_dataset_cfgs['args'])
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=train_cfgs['train_batch_size'],
                              num_workers=train_cfgs['num_workers'],
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler)
    # create validation dataloader
    valid_dataset_cfgs = cfgs['valid_dataset_args']
    valid_dataset = getattr(importlib.import_module('data'), valid_dataset_cfgs['name'])(**valid_dataset_cfgs['args'])
    valid_loader = DataLoader(valid_dataset,
                              batch_size=train_cfgs['valid_batch_size'],
                              num_workers=train_cfgs['num_workers'],
                              pin_memory=True)

    # create loggers
    if local_rank == 0:
        logger = Logger(
            file_path=osp.join(train_cfgs['save_to'], 'log_{}.txt'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        )
        logger(model.get_num_params(), timestamp=False)
        writer = SummaryWriter(train_cfgs['save_to'])
        cfgs_bp = osp.join(train_cfgs['save_to'], 'cfg.yaml')
        with open(cfgs_bp, 'w') as f:
            yaml.dump(cfgs, f)
    else:
        logger = None
        writer = None

    # loop
    step = 0
    step_per_epoch = len(train_loader)
    start_epoch = 0
    end_epoch = train_cfgs['epoch']
    if 'start_epoch' in train_cfgs:
        start_epoch = train_cfgs['start_epoch']
    for epoch in range(start_epoch, end_epoch):
        train_sampler.set_epoch(epoch)

        # training
        time_stamp = time.time()
        for i, tensor in enumerate(train_loader):
            # record time after loading data
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tensor['lq_imgs'] = tensor['lq_imgs'].to(device, non_blocking=True)
            tensor['gt_imgs'] = tensor['gt_imgs'].to(device, non_blocking=True)
            results = model.update(inputs=tensor, training=True)
            # record time after updating model
            train_time_interval = time.time() - time_stamp
            # print training info
            if ((step + 1) % train_cfgs['print_steps'] == 0) and (local_rank == 0):
                msg = 'epoch: {:>3}, lr: {:.7f}, batch: [{:>5}/{:>5}], time: {:.2f} + {:.2f} sec, loss: {:.5f}'
                msg = msg.format(epoch + 1,
                                 model.get_lr(),
                                 i + 1,
                                 step_per_epoch,
                                 data_time_interval,
                                 train_time_interval,
                                 results['loss'].item())
                logger(msg, prefix='[train]')
                writer.add_scalar('learning_rate', model.get_lr(), step + 1)
                writer.add_scalar('train/loss', results['loss'].item(), step + 1)
            # record image results
            if ((step + 1) % train_cfgs['save_results_steps'] == 0) and (local_rank == 0):
                lq_img = results['lq_img']
                gt_img = results['gt_img']
                pred_img = results['pred_img'].clamp(0, 1)
                lq_img = rearrange(lq_img * 255., 'b c h w -> b h w c').cpu().detach().numpy().astype(np.uint8)
                gt_img = rearrange(gt_img * 255., 'b c h w -> b h w c').cpu().detach().numpy().astype(np.uint8)
                pred_img = rearrange(pred_img * 255., 'b c h w -> b h w c').cpu().detach().numpy().astype(np.uint8)
                b = pred_img.shape[0]
                for j in range(b):
                    all_imgs = np.concatenate([lq_img[j], pred_img[j], gt_img[j]], axis=1)[:, :, ::-1]
                    writer.add_image('train/img_results_{}'.format(j), all_imgs, step + 1, dataformats='HWC')
            step += 1
            time_stamp = time.time()
        # save model
        model.scheduler_step()
        if local_rank == 0:
            model.save_model(osp.join(train_cfgs['save_to'], 'latest.ckpt'))
            if (epoch + 1) % train_cfgs['save_model_epoches'] == 0:
                model.save_model(osp.join(train_cfgs['save_to'], '{}.ckpt'.format(epoch + 1)))

        # evaluation
        if (epoch + 1) % train_cfgs['eval_epochs'] != 0:
            dist.barrier()
            continue
        loss_meter = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        random_idx = random.randint(0, len(valid_loader))
        time_stamp = time.time()
        for i, tensor in enumerate(valid_loader):
            tensor['lq_imgs'] = tensor['lq_imgs'].to(device, non_blocking=True)
            tensor['gt_imgs'] = tensor['gt_imgs'].to(device, non_blocking=True)
            results = model.update(inputs=tensor, training=False)
            lq_img = results['lq_img'].detach()
            gt_img = results['gt_img'].detach()
            pred_img = results['pred_img'].detach().clamp(0, 1)
            b = pred_img.shape[0]
            psnr_val = torchmetrics.functional.psnr(pred_img, gt_img, data_range=1)
            ssim_val = torchmetrics.functional.ssim(pred_img, gt_img, data_range=1)
            psnr_meter.update(psnr_val, b)
            ssim_meter.update(ssim_val, b)
            loss_meter.update(results['loss'].item(), b)
            # record image results
            if (i == random_idx) and (local_rank == 0):
                lq_img = rearrange(lq_img * 255., 'b c h w -> b h w c').cpu().numpy().astype(np.uint8)
                gt_img = rearrange(gt_img * 255., 'b c h w -> b h w c').cpu().numpy().astype(np.uint8)
                pred_img = rearrange(pred_img * 255., 'b c h w -> b h w c').cpu().numpy().astype(np.uint8)
                b = pred_img.shape[0]
                for j in range(b):
                    all_imgs = np.concatenate([lq_img[j], pred_img[j], gt_img[j]], axis=1)[:, :, ::-1]
                    writer.add_image('valid/img_results_{}'.format(j), all_imgs, step + 1, dataformats='HWC')
        eval_time_interval = time.time() - time_stamp
        if local_rank == 0:
            msg = 'eval time: {:.2f} sec, loss: {:.5f}, psnr: {:.5f}, ssim: {:.5f}'.format(
                eval_time_interval, loss_meter.avg, psnr_meter.avg, ssim_meter.avg
            )
            logger(msg, prefix='[valid]')
            writer.add_scalar('valid/loss', loss_meter.avg, epoch + 1)
            writer.add_scalar('valid/psnr', psnr_meter.avg, epoch + 1)
            writer.add_scalar('valid/ssim', ssim_meter.avg, epoch + 1)
        dist.barrier()
    dist.destroy_process_group()
