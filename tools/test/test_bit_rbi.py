import yaml
import os
import torch
import time
import importlib
import torchmetrics
import cv2
import os.path as osp
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model.utils import init_seeds
from einops import rearrange
from model.utils import Logger

if __name__ == '__main__':
    """
    cmd:        
    CUDA_VISIBLE_DEVICES=0 ./tools/test/test_bit_rbi.sh ./checkpoints/bit_rbi/cfg.yaml ./checkpoints/bit_rbi/latest.ckpt ./results/bit_rbi/
    CUDA_VISIBLE_DEVICES=0 ./tools/test/test_bit_rbi.sh ./checkpoints/bit++_rbi/cfg.yaml ./checkpoints/bit++_rbi/latest.ckpt ./results/bit++_rbi/    
    """
    parser = ArgumentParser(description='blur interpolation transformer')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--config', default='./configs/bit++_rbi.yaml',
                        help='path of config')
    parser.add_argument('--checkpoint', type=str, default=None, help='path of checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='where to save logs and image results')
    parser.add_argument('--save_img', type=bool, default=False, help='whether to save image results')
    parser.add_argument('--num_gts', type=int, default=9, help='number of gt images')
    args = parser.parse_args()

    # parse cfgs
    with open(args.config) as f:
        cfgs = yaml.full_load(f)

    # ddp initialization
    torch.backends.cudnn.benchmark = True
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f'local_rank: {local_rank}')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_seeds(seed=local_rank)
    dist.init_process_group(backend="nccl")

    # create model
    model_cls = getattr(importlib.import_module('model'), cfgs['model_args']['name'])
    model = model_cls(**cfgs['model_args']['args'],
                      optimizer_args=cfgs['optimizer_args'],
                      scheduler_args=cfgs['scheduler_args'],
                      loss_args=cfgs['loss_args'],
                      local_rank=local_rank,
                      load_from=args.checkpoint)

    # create test dataloader
    test_dataset_cfgs = cfgs['test_dataset_args']
    test_dataset = getattr(importlib.import_module('data'), test_dataset_cfgs['name'])(**test_dataset_cfgs['args'])
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=4,
                             pin_memory=True)

    # test
    save_dir = args.save_dir
    save_img = args.save_img
    if save_dir is None:
        save_dir = './'
    logger = Logger(osp.join(save_dir, 'test_bit_log.txt'), verbose=True)
    num_gts = 3
    psnr_vals, ssim_vals = [0, ] * num_gts, [0, ] * num_gts
    time_stamp = time.time()
    total_time = 0.
    ts = torch.linspace(start=0, end=1, steps=num_gts, device=device)
    for i, tensor in tqdm(enumerate(test_loader), total=len(test_loader)):
        lq_imgs = tensor['lq_imgs'].to(device, non_blocking=True)
        gt_imgs = tensor['gt_imgs'].to(device, non_blocking=True)
        # record the testing time
        torch.cuda.synchronize()
        time_start = time.time()
        pred_imgs = model.inference(lq_imgs, ts).detach()
        torch.cuda.synchronize()
        total_time += time.time() - time_start
        pred_imgs = pred_imgs.clip(0, 1)
        # calculate scores
        for j in range(num_gts):
            psnr_val = torchmetrics.functional.psnr(pred_imgs[:, j], gt_imgs[:, j], data_range=1)
            ssim_val = torchmetrics.functional.ssim(pred_imgs[:, j], gt_imgs[:, j], data_range=1)
            psnr_vals[j] += psnr_val
            ssim_vals[j] += ssim_val
        # save images
        if save_img:
            pred_imgs = rearrange(pred_imgs * 255, 'n c h w -> h (n w) c').cpu().detach().numpy().astype(np.uint8)
            gt_imgs = rearrange(gt_imgs * 255, 'n c h w -> h (n w) c').cpu().detach().numpy().astype(np.uint8)
            save_imgs = np.concatenate([pred_imgs, gt_imgs], axis=0)
            save_name = osp.join(save_dir, '{:8d}.png'.format(i))
            cv2.imwrite(save_name, save_imgs)
    avg_time = total_time / len(test_loader)
    print("avg_time: {:3f} s".format(avg_time))
    print("total time: {:3f} s".format(total_time))
    print("total samples: ", len(test_loader))
    psnr_vals = [psnr_val.cpu() / len(test_loader) for psnr_val in psnr_vals]
    ssim_vals = [ssim_val.cpu() / len(test_loader) for ssim_val in ssim_vals]
    eval_time_interval = time.time() - time_stamp
    logger('cost time: {:.3f}'.format(eval_time_interval))
    for idx, t in enumerate(ts):
        logger('t: {:.3f}, psnr: {:.3f}, ssim: {:.3f}'.format(t, psnr_vals[idx], ssim_vals[idx]))
    logger('average -> psnr: {:.3f}, ssim: {:.3f}'.format(np.mean(psnr_vals), np.mean(ssim_vals)))
    dist.destroy_process_group()
