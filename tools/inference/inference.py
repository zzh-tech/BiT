import yaml
import os
import torch
import time
import importlib
import cv2
import os.path as osp
import torch.distributed as dist
import numpy as np
import imageio as iio
from tqdm import tqdm
from argparse import ArgumentParser
from model.utils import init_seeds
from einops import rearrange
import pyshine as ps


def add_header(img, header, add_space=True):
    temp_path = 'temp.png'
    iio.imsave(temp_path, img)
    img = cv2.imread(temp_path)
    os.remove(temp_path)
    if add_space:
        space = np.ones_like(img) * 108
        space = space[:space.shape[0] // 8, ]
        img = np.vstack([space, img])
    return ps.putBText(img, header, text_offset_x=10, text_offset_y=10, vspace=10, hspace=10,
                       font_scale=2.5, background_RGB=(108, 108, 108), text_RGB=(255, 250, 250),
                       font=cv2.FONT_HERSHEY_PLAIN)[:, :, ::-1]


if __name__ == '__main__':
    """
    cmd:
    sh ./tools/inference/inference.sh ./checkpoints/bit++_adobe240/cfg.yaml ./checkpoints/bit++_adobe240/latest.ckpt ./demo/00777.png ./demo/00785.png ./demo/00793.png ./demo/bit++_results/ 30
    sh ./tools/inference/inference.sh ./checkpoints/bit_adobe240/cfg.yaml ./checkpoints/bit_adobe240/latest.ckpt ./demo/00777.png ./demo/00785.png ./demo/00793.png ./demo/bit_results/ 30
    """
    parser = ArgumentParser(description='blur interpolation transformer')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--config', default='./configs/cfg.yaml', help='path of config')
    parser.add_argument('--checkpoint', type=str, default=None, help='path of checkpoint')
    parser.add_argument('--img_pre', type=str, required=True, help='path of previous blurry image')
    parser.add_argument('--img_cur', type=str, required=True, help='path of current blurry image')
    parser.add_argument('--img_nxt', type=str, default=None, help='path of next blurry image')
    parser.add_argument('--save_dir', type=str, required=True, help='where to save image results')
    parser.add_argument('--num', type=int, default=11, help='number of extracted images')
    parser.add_argument('--gif', type=bool, default=True, help='whether to generate the corresponding gif')
    parser.add_argument('--multi_infer', type=bool, default=True, help='multiple inferences with shared features')
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

    save_dir = args.save_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # inference
    eval_time_interval = 0
    ts = torch.linspace(start=0, end=1, steps=args.num, device=device)
    img_pre = cv2.imread(args.img_pre)
    img_cur = cv2.imread(args.img_cur)
    lq_imgs = [img_pre, img_cur]
    if args.img_nxt is not None:
        img_nxt = cv2.imread(args.img_nxt)
        lq_imgs.append(img_nxt)

    save_name = osp.join(save_dir, 'blur.png')
    cv2.imwrite(save_name, img_cur)
    gif_imgs = 15 * [img_cur, ]
    lq_imgs = np.stack(lq_imgs, axis=0)[np.newaxis]  # 1, 2, H, W, C
    lq_imgs = torch.from_numpy(lq_imgs).to(device, non_blocking=True).float() / 255.
    lq_imgs = rearrange(lq_imgs, 'b n h w c -> b n c h w')

    if not args.multi_infer:
        for i, t in tqdm(enumerate(ts), total=args.num):
            torch.cuda.synchronize()
            time_stamp = time.time()
            pred_img = model.inference(lq_imgs, [t])
            torch.cuda.synchronize()
            eval_time_interval += time.time() - time_stamp
            pred_img = pred_img.detach().clamp(0, 1)[0][0]
            pred_img = rearrange(pred_img * 255., 'c h w -> h w c').cpu().detach().numpy().astype(np.uint8)
            save_name = osp.join(save_dir, '{:08d}.png'.format(i))
            cv2.imwrite(save_name, pred_img)
            gif_imgs.append(pred_img)
    else:
        torch.cuda.synchronize()
        time_stamp = time.time()
        pred_imgs = model.inference(lq_imgs, ts)
        torch.cuda.synchronize()
        eval_time_interval += time.time() - time_stamp
        pred_imgs = pred_imgs.detach().clamp(0, 1)[0]
        for i in range(args.num):
            pred_img = pred_imgs[i]
            pred_img = rearrange(pred_img * 255., 'c h w -> h w c').cpu().detach().numpy().astype(np.uint8)
            save_name = osp.join(save_dir, '{:08d}.png'.format(i))
            cv2.imwrite(save_name, pred_img)
            gif_imgs.append(pred_img)
    if args.gif:
        gif_path = osp.join(args.save_dir, 'demo.gif')
        with iio.get_writer(gif_path, mode='I') as writer:
            for img in gif_imgs:
                writer.append_data(img[:, :, ::-1])
    print('total runtime: {:.3f}'.format(eval_time_interval))
    print('average runtime: {:.3f}'.format(eval_time_interval / args.num))
    dist.destroy_process_group()
