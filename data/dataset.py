import cv2
import random
import os
import torch
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from torch.utils.data import Dataset

try:
    from data.data_aug import Compose
except:
    from data_aug import Compose


class Adobe240(Dataset):
    """
    used for blur decomposition (synthetic)
    all single images are under lq_dir and gt_dir
    the images share the same names under lq_dir and gt_dir
    """

    def __init__(self, lq_dir, gt_dir, aug_args, num_future=1, num_past=1, num_gts=11, file_fmt='{:05d}', train=True,
                 suffix='png', idx_fun=lambda x: int(osp.basename(x).split('.')[0]), full_gts=False):
        self.num_gt = num_gts
        self.num_future = num_future
        self.num_past = num_past
        self.file_fmt = file_fmt
        self.suffix = suffix
        self.idx_fun = idx_fun
        self.train = train
        self.full_gts = full_gts
        self.lq_img_paths, self.gt_img_paths = self.get_paths(lq_dir, gt_dir)
        assert len(self.lq_img_paths) == len(self.gt_img_paths), \
            f'length of lq_img_paths {len(self.lq_img_paths)}, length of gt_img_paths: {len(self.gt_img_paths)}'
        self.pipeline = Compose(aug_args)
        self.temporal_inverse = False
        if 'temporal_inverse' in aug_args:
            self.temporal_inverse = aug_args['temporal_inverse']

    def get_paths(self, lq_dir, gt_dir):
        lq_img_paths, gt_img_paths = [], []
        half_num = self.num_gt // 2
        idx_fun = self.idx_fun
        seqs = os.listdir(lq_dir)
        for seq in seqs:
            seq_lq_img_paths = glob(osp.join(lq_dir, seq, '*.{}'.format(self.suffix)))
            seq_lq_img_paths = sorted(seq_lq_img_paths, key=idx_fun)
            start = self.num_past
            end = len(seq_lq_img_paths) - self.num_future
            if self.num_past + 1 + self.num_future > len(seq_lq_img_paths):
                continue
            lq_img_paths += [
                [seq_lq_img_paths[i + j] for j in range(-self.num_past, self.num_future + 1)]
                for i in range(start, end)
            ]
            gt_img_paths += [
                [osp.join(gt_dir, seq, self.file_fmt.format(idx_fun(seq_lq_img_paths[i]) + j) + '.' + self.suffix)
                 for j in range(-half_num, half_num + 1)]
                for i in range(start, end)
            ]
        return lq_img_paths, gt_img_paths

    def __getitem__(self, idx):
        tensor = {}
        tensor['lq_imgs'] = [cv2.imread(path) for path in self.lq_img_paths[idx]]
        num_gts = len(self.gt_img_paths[idx])
        if self.train:
            i = random.randint(0, num_gts - 1)
        else:
            i = num_gts // 2
        tensor['t'] = torch.linspace(start=0, end=1, steps=num_gts)[i]
        tmp_gt_img_paths = self.gt_img_paths[idx]
        if not self.full_gts:
            tmp_gt_img_paths = [self.gt_img_paths[idx][0], self.gt_img_paths[idx][i], self.gt_img_paths[idx][-1]]
        tensor['gt_imgs'] = [cv2.imread(path) for path in tmp_gt_img_paths]
        tensor = self.augment(tensor)
        tensor['lq_imgs'] = torch.stack(tensor['lq_imgs'], dim=0)
        tensor['gt_imgs'] = torch.stack(tensor['gt_imgs'], dim=0)
        tensor["lq_imgs_path"] = self.lq_img_paths[idx]
        tensor["gt_imgs_path"] = self.gt_img_paths[idx]
        return tensor

    def __len__(self):
        return len(self.lq_img_paths)

    def augment(self, tensor):
        num_lq_imgs = len(tensor['lq_imgs'])
        out = self.pipeline(image=tensor['lq_imgs'][0])
        tensor['lq_imgs'][0], replay_args = out['image'], out['replay_args']
        for i in range(1, num_lq_imgs):
            out = self.pipeline(image=tensor['lq_imgs'][i], replay_args=replay_args)
            tensor['lq_imgs'][i] = out['image']

        num_gt_imgs = len(tensor['gt_imgs'])
        for i in range(num_gt_imgs):
            out = self.pipeline(image=tensor['gt_imgs'][i], replay_args=replay_args)
            tensor['gt_imgs'][i] = out['image']

        if self.temporal_inverse:
            if random.randint(0, 1):
                tensor['lq_imgs'].reverse()
                tensor['gt_imgs'].reverse()
        return tensor

    def visualize(self, idx, aug=False, title=None):
        """
        Visualize the images to check the correctness of implementation
        """
        lq_img_paths, gt_img_paths = self.lq_img_paths[idx], self.gt_img_paths[idx]
        print("lq_img_paths:\n", lq_img_paths)
        print("gt_img_paths:\n", gt_img_paths)
        lq_imgs = [cv2.imread(path) for path in lq_img_paths]
        gt_imgs = [cv2.imread(path) for path in gt_img_paths]
        if aug:
            tensor = {}
            tensor['lq_imgs'], tensor['gt_imgs'] = lq_imgs, gt_imgs
            tensor = self.augment(tensor)
            lq_imgs, gt_imgs = tensor['lq_imgs'], tensor['gt_imgs']
            lq_imgs = [(lq_img * 255).numpy().astype(np.uint8).transpose((1, 2, 0)) for lq_img in lq_imgs]
            gt_imgs = [(gt_img * 255).numpy().astype(np.uint8).transpose((1, 2, 0)) for gt_img in gt_imgs]
        # imgs = np.concatenate(lq_imgs + gt_imgs, axis=1)[:, :, ::-1]
        imgs = np.concatenate(lq_imgs, axis=1)[:, :, ::-1]
        plt.figure(figsize=(4 * (imgs.shape[1] // imgs.shape[0]), 4))
        plt.title(label=title)
        plt.axis('off')
        plt.imshow(imgs)
        imgs = np.concatenate(gt_imgs, axis=1)[:, :, ::-1]
        plt.figure(figsize=(4 * (imgs.shape[1] // imgs.shape[0]), 4))
        plt.title(label=title)
        plt.axis('off')
        plt.imshow(imgs)
        plt.show()


class RBI(Dataset):
    """
    real-world blur interpolation dataset
    all single images are under lq_dir and gt_dir
    the images share the same names under lq_dir and gt_dir
    """

    def __init__(self, lq_dir, gt_dir, aug_args, num_future=1, num_past=1, num_gts=9, num_steps=20,
                 file_fmt='{}', suffix='png', idx_fun=lambda x: int(osp.basename(x).split('.')[0]), train=True,
                 full_gts=False):
        self.num_gt = num_gts
        self.num_future = num_future
        self.num_past = num_past
        self.file_fmt = file_fmt
        self.suffix = suffix
        self.idx_fun = idx_fun
        self.num_steps = num_steps
        self.train = train
        self.full_gts = full_gts
        self.val_seqs = ['_Scene{}'.format(i) for i in range(51, 57)]
        self.val_seqs.remove('_Scene52')

        self.lq_img_paths, self.gt_img_paths = self.get_paths(lq_dir, gt_dir)
        assert len(self.lq_img_paths) == len(self.gt_img_paths), \
            f'length of lq_img_paths {len(self.lq_img_paths)}, length of gt_img_paths: {len(self.gt_img_paths)}'
        self.pipeline = Compose(aug_args)
        self.temporal_inverse = False
        if 'temporal_inverse' in aug_args:
            self.temporal_inverse = aug_args['temporal_inverse']

    def get_paths(self, lq_dir, gt_dir):
        lq_img_paths, gt_img_paths = [], []
        idx_fun = self.idx_fun
        seqs = os.listdir(lq_dir)
        for seq in seqs:
            if self.train:
                if seq in self.val_seqs:
                    continue
            else:
                if seq not in self.val_seqs:
                    continue
            seq_lq_img_paths = glob(osp.join(lq_dir, seq, '**', '*.{}'.format(self.suffix)), recursive=True)
            seq_lq_img_paths = sorted(seq_lq_img_paths, key=idx_fun)
            start = self.num_past
            end = len(seq_lq_img_paths) - self.num_future
            if self.num_past + 1 + self.num_future > len(seq_lq_img_paths):
                continue
            lq_img_paths += [
                [seq_lq_img_paths[i + j] for j in range(-self.num_past, self.num_future + 1)]
                for i in range(start, end)
            ]
            gt_img_paths += [
                [osp.join(gt_dir, seq, 'RGB', self.file_fmt.format(
                    self.num_steps * (idx_fun(seq_lq_img_paths[i]) - 1) + j + 1) + '.' + self.suffix)
                 for j in range(self.num_gt)]
                for i in range(start, end)
            ]
        return lq_img_paths, gt_img_paths

    def __getitem__(self, idx):
        tensor = {}
        tensor['lq_imgs'] = [cv2.imread(path) for path in self.lq_img_paths[idx]]
        num_gts = len(self.gt_img_paths[idx])
        if self.train:
            i = random.randint(0, num_gts - 1)
        else:
            i = num_gts // 2
        tensor['t'] = torch.linspace(start=0, end=1, steps=num_gts)[i]
        tmp_gt_img_paths = self.gt_img_paths[idx]
        if not self.full_gts:
            tmp_gt_img_paths = [self.gt_img_paths[idx][0], self.gt_img_paths[idx][i], self.gt_img_paths[idx][-1]]
        tensor['gt_imgs'] = [cv2.imread(path) for path in tmp_gt_img_paths]
        tensor = self.augment(tensor)
        tensor['lq_imgs'] = torch.stack(tensor['lq_imgs'], dim=0)
        tensor['gt_imgs'] = torch.stack(tensor['gt_imgs'], dim=0)
        tensor["lq_imgs_path"] = self.lq_img_paths[idx]
        tensor["gt_imgs_path"] = self.gt_img_paths[idx]
        return tensor

    def __len__(self):
        return len(self.lq_img_paths)

    def augment(self, tensor):
        num_lq_imgs = len(tensor['lq_imgs'])
        out = self.pipeline(image=tensor['lq_imgs'][0])
        tensor['lq_imgs'][0], replay_args = out['image'], out['replay_args']
        for i in range(1, num_lq_imgs):
            out = self.pipeline(image=tensor['lq_imgs'][i], replay_args=replay_args)
            tensor['lq_imgs'][i] = out['image']

        num_gt_imgs = len(tensor['gt_imgs'])
        for i in range(num_gt_imgs):
            out = self.pipeline(image=tensor['gt_imgs'][i], replay_args=replay_args)
            tensor['gt_imgs'][i] = out['image']

        if self.temporal_inverse:
            if random.randint(0, 1):
                tensor['lq_imgs'].reverse()
                tensor['gt_imgs'].reverse()
        return tensor

    def visualize(self, idx, aug=False, title=None):
        """
        Visualize the images to check the correctness of implementation
        """
        lq_img_paths, gt_img_paths = self.lq_img_paths[idx], self.gt_img_paths[idx]
        print("lq_img_paths:\n", lq_img_paths)
        print("gt_img_paths:\n", gt_img_paths)
        lq_imgs = [cv2.imread(path) for path in lq_img_paths]
        gt_imgs = [cv2.imread(path) for path in gt_img_paths]
        if aug:
            tensor = {}
            tensor['lq_imgs'], tensor['gt_imgs'] = lq_imgs, gt_imgs
            tensor = self.augment(tensor)
            lq_imgs, gt_imgs = tensor['lq_imgs'], tensor['gt_imgs']
            lq_imgs = [(lq_img * 255).numpy().astype(np.uint8).transpose((1, 2, 0)) for lq_img in lq_imgs]
            gt_imgs = [(gt_img * 255).numpy().astype(np.uint8).transpose((1, 2, 0)) for gt_img in gt_imgs]
        imgs = np.concatenate(lq_imgs, axis=1)[:, :, ::-1]
        plt.figure(figsize=(4 * (imgs.shape[1] // imgs.shape[0]), 4))
        plt.title(label=title)
        plt.axis('off')
        plt.imshow(imgs)
        imgs = np.concatenate(gt_imgs, axis=1)[:, :, ::-1]
        plt.figure(figsize=(4 * (imgs.shape[1] // imgs.shape[0]), 4))
        plt.title(label=title)
        plt.axis('off')
        plt.imshow(imgs)
        plt.show()


if __name__ == '__main__':
    # validation code for Adobe240 dataset
    lq_dir = '/home/zhong/Dataset/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/'
    gt_dir = '/home/zhong/Dataset/Adobe_240fps_dataset/Adobe_240fps_blur/train/'
    aug_args = {
        'image': {
            'RandomCrop': {
                'height': 256,
                'width': 256,
            },
            'Rot90': {},
            'Flip': {},
            'ToTensor': {}
        }
    }
    dataset = Adobe240(lq_dir, gt_dir, aug_args)
    idx = 530
    print('{} images'.format(len(dataset)))
    dataset.visualize(idx, title='w/o aug')
    dataset.visualize(idx, aug=True, title='w/ aug')

    # # validation code for RBI dataset
    # lq_dir = '/home/zhong/Dataset/RBI/Blur/'
    # gt_dir = '/home/zhong/Dataset/RBI/Sharp/'
    # aug_args = {
    #     'image': {
    #         'RandomCrop': {
    #             'height': 256,
    #             'width': 256,
    #         },
    #         'Rot90': {},
    #         'Flip': {},
    #         'ToTensor': {}
    #     }
    # }
    # dataset = RBI(lq_dir, gt_dir, aug_args, train=False)
    # idx = 39
    # print('{} images'.format(len(dataset)))
    # dataset.visualize(idx, title='w/o aug')
    # dataset.visualize(idx, aug=True, title='w/ aug')
