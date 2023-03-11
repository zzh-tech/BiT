import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange
from itertools import chain

try:
    import model.loss as loss
    from model.utils import pair, ckpt_convert, calc_model_params
    from model.arches import default_conv
    from model.bit import BiT
except:
    import loss
    from utils import pair, ckpt_convert, calc_model_params
    from arches import default_conv
    from bit import BiT


class Model:
    def __init__(self, image_size, in_chs, out_chs, embed_dim, depths, num_heads,
                 window_size, mlp_ratio, num_mid, optimizer_args, scheduler_args, loss_args,
                 num_gts=3, use_mlp=True, use_conv=False, norm_layer=nn.LayerNorm,
                 in_num=2, skip=False, anchor_num_layers=3, mid_ratio=1., accum_iters=2,
                 local_rank=0, load_from=None, resume_from=None, ddp=True):
        super(Model, self).__init__()
        self.num_gts = num_gts
        self.num_mid = num_mid
        self.mid_ratio = mid_ratio
        self.iters = 0
        self.accum_iters = accum_iters
        self.model = BiT(img_size=image_size, in_chs=in_chs, out_chs=out_chs, embed_dim=embed_dim, depths=depths,
                         num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                         norm_layer=norm_layer, use_mlp=use_mlp, use_conv=use_conv, num_mid=num_mid,
                         in_num=in_num, skip=skip, anchor_num_layers=anchor_num_layers)
        if ddp:
            # replace BN as SyncBN
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        try:
            # load checkpoint
            if (local_rank == 0) and (load_from is not None):
                self.load_model(load_from)
            # modify the model
            del self.model.last_shuffle_up
            self.model.last_shuffle_up = nn.Identity()
            self.last_shuffle_up = nn.Sequential(
                default_conv(2 * embed_dim, out_chs * (4 ** 2), kernel_size=3, stride=1), nn.PixelShuffle(4)
            )
        except:
            # modify the model
            del self.model.last_shuffle_up
            self.model.last_shuffle_up = nn.Identity()
            self.last_shuffle_up = nn.Sequential(
                default_conv(2 * embed_dim, out_chs * (4 ** 2), kernel_size=3, stride=1), nn.PixelShuffle(4)
            )
            # load checkpoint
            if (local_rank == 0) and (load_from is not None):
                self.load_model(load_from)
        # resume checkpoint
        if (local_rank == 0) and (resume_from is not None):
            self.load_model(resume_from)
        self.device = torch.device("cuda", local_rank)
        # move model to GPU
        self.model = self.model.to(self.device)
        self.last_shuffle_up = self.last_shuffle_up.to(self.device)
        # nulti-gpus wrapper
        if ddp:
            self.model = DDP(self.model,
                             device_ids=[local_rank],
                             output_device=local_rank,
                             find_unused_parameters=True,
                             broadcast_buffers=False)
            self.last_shuffle_up = DDP(self.last_shuffle_up,
                                       device_ids=[local_rank],
                                       output_device=local_rank,
                                       find_unused_parameters=False,
                                       broadcast_buffers=False)
        else:
            self.model = DP(self.model)
            self.last_shuffle_up = DP(self.last_shuffle_up)

        # Initialize optimizer, learning rate scheduler, and loss function
        self.optimizer = getattr(optim, optimizer_args['name'])(
            chain(self.model.parameters(), self.last_shuffle_up.parameters()), **optimizer_args['args']
        )
        self.scheduler = getattr(optim.lr_scheduler, scheduler_args['name'])(self.optimizer,
                                                                             **scheduler_args['args'])
        if (local_rank == 0) and (resume_from is not None):
            self.load_others(resume_from)
        try:
            self.loss = getattr(nn, loss_args['name'])(**loss_args['args'])
        except:
            self.loss = getattr(loss, loss_args['name'])(**loss_args['args'])

    def load_model(self, load_from):
        model_dict = torch.load(load_from)
        self.model.load_state_dict(ckpt_convert(model_dict['model']))
        if 'last_shuffle_up' in model_dict:
            self.last_shuffle_up.load_state_dict(ckpt_convert(model_dict['last_shuffle_up']))

    def load_others(self, load_from):
        model_dict = torch.load(load_from)
        self.optimizer.load_state_dict(model_dict['optimizer'])
        self.scheduler.load_state_dict(model_dict['scheduler'])

    def save_model(self, save_to):
        model_dict = {
            'model': self.model.state_dict(),
            'last_shuffle_up': self.last_shuffle_up.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(model_dict, save_to)

    def scheduler_step(self):
        self.scheduler.step()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def get_num_params(self):
        return calc_model_params([self.model, self.last_shuffle_up])

    def update(self, inputs, training=True):
        results = inputs
        lq_imgs, gt_imgs = results['lq_imgs'], results['gt_imgs']  # B, N', C, H, W
        lq_img = lq_imgs[:, 1]
        gt_img = gt_imgs[:, self.num_gts // 2]
        mid_ts = torch.linspace(start=0, end=self.num_gts - 1, steps=self.num_mid, device=self.device).long()
        mid_gt_imgs = gt_imgs[:, mid_ts]  # B, num_mid, C, H, W
        mid_gt_imgs_inv = torch.flip(mid_gt_imgs, dims=[1])

        if training:
            self.model.train()
            t = inputs['t'].to(self.device)
            pred_feat, mid_pred_imgs = self.model(lq_imgs, t)
            pred_feat_inv, mid_pred_imgs_inv = self.model(lq_imgs.flip(dims=[1, ]), 1 - t)
            pred_img = self.last_shuffle_up(torch.cat([pred_feat, pred_feat_inv], dim=1))
            mid_pred_imgs = rearrange(mid_pred_imgs, 'B N C H W -> (B N) C H W')
            mid_pred_imgs_inv = rearrange(mid_pred_imgs_inv, 'B N C H W -> (B N) C H W')
            mid_gt_imgs = rearrange(mid_gt_imgs, 'B N C H W -> (B N) C H W')
            mid_gt_imgs_inv = rearrange(mid_gt_imgs_inv, 'B N C H W -> (B N) C H W')

            # Gradient accumulation
            loss_val = self.mid_ratio * (self.loss(mid_pred_imgs, mid_gt_imgs) / self.num_mid)
            loss_val = loss_val + self.mid_ratio * (self.loss(mid_pred_imgs_inv, mid_gt_imgs_inv) / self.num_mid)
            loss_val = loss_val / 2.
            loss_val = loss_val + self.loss(pred_img, gt_img)
            loss_val = loss_val / self.accum_iters
            loss_val.backward()
            self.iters += 1
            if self.iters % self.accum_iters == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.model.eval()
            with torch.no_grad():
                t = torch.tensor([0.5]).to(self.device)

                pred_feat, mid_pred_imgs = self.model(lq_imgs, t)
                pred_feat_inv, mid_pred_imgs_inv = self.model(lq_imgs.flip(dims=[1, ]), 1 - t)
                pred_img = self.last_shuffle_up(torch.cat([pred_feat, pred_feat_inv], dim=1))
                mid_pred_imgs = rearrange(mid_pred_imgs, 'B N C H W -> (B N) C H W')
                mid_pred_imgs_inv = rearrange(mid_pred_imgs_inv, 'B N C H W -> (B N) C H W')
                mid_gt_imgs = rearrange(mid_gt_imgs, 'B N C H W -> (B N) C H W')
                mid_gt_imgs_inv = rearrange(mid_gt_imgs_inv, 'B N C H W -> (B N) C H W')

                loss_val = self.mid_ratio * (self.loss(mid_pred_imgs, mid_gt_imgs) / self.num_mid)
                loss_val = loss_val + self.mid_ratio * (self.loss(mid_pred_imgs_inv, mid_gt_imgs_inv) / self.num_mid)
                loss_val = loss_val / 2.
                loss_val = loss_val + self.loss(pred_img, gt_img)
                loss_val = loss_val / self.accum_iters
        results['pred_img'] = pred_img
        results['gt_img'] = gt_img
        results['lq_img'] = lq_img
        results['loss'] = loss_val
        return results

    @torch.no_grad()
    def inference(self, lq_imgs, ts):
        if isinstance(ts, list):
            ts = torch.tensor([ts])
        ts = ts.to(self.device)
        self.model.eval()
        pred_feats = self.model.module.inference(lq_imgs, ts)
        pred_feats_inv = self.model.module.inference(lq_imgs.flip(dims=[1, ]), 1 - ts)
        pred_imgs = []
        for i in range(len(ts)):
            pred_imgs.append(
                self.last_shuffle_up(torch.cat([pred_feats[:, i], pred_feats_inv[:, i]], dim=1))
            )
        pred_imgs = torch.stack(pred_imgs, dim=1)
        return pred_imgs
