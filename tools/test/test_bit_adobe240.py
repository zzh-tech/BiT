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
from glob import glob
from model.utils import init_seeds
from einops import rearrange
from model.utils import Logger

if __name__ == '__main__':
    """
    cmd:
    CUDA_VISIBLE_DEVICES=0 ./tools/test/test_bit_adobe240.sh ./checkpoints/bit_adobe240/cfg.yaml ./checkpoints/bit_adobe240/latest.ckpt ./results/bit_adobe240/ /home/zhong/Dataset/Adobe_240fps_dataset/Adobe_240fps_blur/    
    CUDA_VISIBLE_DEVICES=1 ./tools/test/test_bit_adobe240.sh ./checkpoints/bit++_adobe240/cfg.yaml ./checkpoints/bit++_adobe240/latest.ckpt ./results/bit++_adobe240/ /home/zhong/Dataset/Adobe_240fps_dataset/Adobe_240fps_blur/
    """
    parser = ArgumentParser(description='blur interpolation transformer')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--config', default='./configs/bit++_adobe240.yaml',
                        help='path of config')
    parser.add_argument('--checkpoint', type=str, default=None, help='path of checkpoint')
    parser.add_argument('--data_dir', type=str, default=None, help='directory of input images')
    parser.add_argument('--save_dir', type=str, default=None, help='where to save image logs and results')
    parser.add_argument('--save_img', type=bool, default=False, help='whether to save image results')
    parser.add_argument('--num_future', type=int, default=1, help='number of future frames')
    parser.add_argument('--num_past', type=int, default=1, help='number of past frames')
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

    # test
    save_dir = args.save_dir
    save_img = args.save_img
    if save_dir is None:
        save_dir = './'
    logger = Logger(osp.join(save_dir, 'test_bit_like_bin_log.txt'), verbose=True)
    time_stamp = time.time()
    num_future = args.num_future
    num_past = args.num_past
    step = 4
    count = 0
    name_fmt = '{:05d}.png'
    num_gts = 11
    psnr_vals, ssim_vals = 0, 0
    psnr_seq_vals, ssim_seq_vals = 0, 0
    idx_func = lambda x: int(osp.basename(x).split('.')[0])
    ts = torch.linspace(start=0, end=1, steps=num_gts, device=device)
    ts = ts[[1, num_gts // 2]]

    lq_dir_root = osp.join(args.data_dir, 'test_blur')
    lq_dirs = os.listdir(lq_dir_root)
    for lq_dir in lq_dirs:
        logger('start inference for {} ...'.format(lq_dir))
        psnr_seq_val, ssim_seq_val = 0, 0
        seq_count = 0
        lq_dir_path = osp.join(lq_dir_root, lq_dir)
        if not osp.isdir(lq_dir_path):
            continue
        save_dir_path = osp.join(save_dir, lq_dir)
        lq_img_paths = glob(osp.join(lq_dir_path, '*.png'))
        lq_img_paths = sorted(lq_img_paths, key=idx_func)
        num_lq_imgs = len(lq_img_paths)
        for i in tqdm(range(num_past, num_lq_imgs - num_future)):
            sub_lq_img_paths = [lq_img_paths[j] for j in range(i - num_past, i + num_future + 1)]
            lq_imgs = [cv2.imread(img_path) for img_path in sub_lq_img_paths]
            lq_imgs = np.stack(lq_imgs, axis=0)[np.newaxis, ...]  # B, H, W, C
            lq_imgs = rearrange(lq_imgs, 'b n h w c -> b n c h w') / 255.
            lq_imgs = torch.from_numpy(lq_imgs).float().to(device, non_blocking=True)

            gt_img_path = lq_img_paths[i].replace('test_blur', 'test')
            sub_gt_img_paths = [osp.join(osp.dirname(gt_img_path),
                                         name_fmt.format(idx_func(osp.basename(gt_img_path)) - step)),
                                gt_img_path]
            gt_imgs = [cv2.imread(img_path) for img_path in sub_gt_img_paths]

            pred_imgs = []
            for j, t in enumerate(ts):
                pred_img = model.inference(lq_imgs, [t]).detach().clamp(0, 1).squeeze(dim=1)  # C, H, W
                tmp_gt_img = gt_imgs[j]
                tmp_gt_img = torch.from_numpy(tmp_gt_img) / 255.
                tmp_gt_img = rearrange(tmp_gt_img, 'h w c -> c h w').to(device, non_blocking=True)[None]
                img_psnr = torchmetrics.functional.psnr(pred_img, tmp_gt_img, data_range=1)
                img_ssim = torchmetrics.functional.ssim(pred_img, tmp_gt_img, data_range=1)
                psnr_vals += img_psnr
                ssim_vals += img_ssim
                count += 1
                psnr_seq_val += img_psnr
                ssim_seq_val += img_ssim
                seq_count += 1
                pred_img = rearrange(pred_img[0] * 255, 'c h w -> h w c').cpu().detach().numpy().astype(np.uint8)
                pred_imgs.append(pred_img)

            assert len(pred_imgs) == len(gt_imgs)
            if save_img:
                os.makedirs(save_dir_path, exist_ok=True)
                for j, gt_img_path in enumerate(sub_gt_img_paths):
                    img_name = osp.basename(gt_img_path)
                    save_img_path = osp.join(save_dir_path, img_name)
                    cv2.imwrite(save_img_path, pred_imgs[j])

        psnr_seq_val_avg = psnr_seq_val / seq_count
        ssim_seq_val_avg = ssim_seq_val / seq_count
        psnr_seq_vals += psnr_seq_val_avg
        ssim_seq_vals += ssim_seq_val_avg
        logger('[seq {}]: psnr: {:.3f}, ssim: {:.3f}'.format(lq_dir, psnr_seq_val_avg, ssim_seq_val_avg))

    psnr_vals_avg = psnr_vals / count
    ssim_vals_avg = ssim_vals / count
    psnr_seq_vals_avg = psnr_seq_vals / len(lq_dirs)
    ssim_seq_vals_avg = ssim_seq_vals / len(lq_dirs)
    eval_time_interval = time.time() - time_stamp
    logger('cost time: {:.3f}'.format(eval_time_interval))
    logger('[seq average] psnr: {:.3f}, ssim: {:.3f}'.format(psnr_seq_vals_avg, ssim_seq_vals_avg))
    logger('[image average] psnr: {:.3f}, ssim: {:.3f}'.format(psnr_vals_avg, ssim_vals_avg))
    dist.destroy_process_group()
