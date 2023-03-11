import cv2
import random
import torch
import importlib
import numpy as np
import albumentations as A


class Compose:
    """
    Compose a series of image augmentations
    """

    def __init__(self, aug_args):
        transforms = []
        for key, val in aug_args['image'].items():
            try:
                transforms.append(getattr(importlib.import_module('data.data_aug'), key.split('_')[0])(**val))
            except:
                transforms.append(getattr(importlib.import_module('data_aug'), key.split('_')[0])(**val))
        self.tranforms = transforms

    def __call__(self, image, bbox=None, ratio=1., flow=False, replay_args=None):
        # Copy the list
        replay_args = list(replay_args) if isinstance(replay_args, list) else None
        out = {'image': image, 'bbox': bbox}
        replay_args_record = []
        for transform in self.tranforms:
            args = replay_args.pop(0) if isinstance(replay_args, list) else None
            out, args = transform(**out, ratio=ratio, flow=flow, args=args)
            replay_args_record.append(args)

        out['replay_args'] = replay_args_record
        return out


class ToTensor:
    """
    transfer numpy.array to torch tensor
    """

    def __init__(self, val_range=255.):
        self.val_range = val_range

    def __call__(self, image, bbox=None, flow=False, args=None, **kwargs):
        image = torch.from_numpy(image).permute(2, 0, 1)
        if flow:
            return {'image': image, 'bbox': bbox}, args
        image = image / self.val_range
        # args: arguments for replaying this augmentation
        return {'image': image, 'bbox': None}, args


class NearBBoxResizedSafeCrop:
    """
    Crop near the outside of the bbox and resize the cropped image
    max_ratio means the max valid crop ratio from boundary of the bbox to boundary of the image, (0, 1)
    bbox format is pascal_voc, a bounding box looks like [x_min, y_min, x_max, y_max], e.g. [97, 12, 247, 212]
    """

    def __init__(self, height, width, max_ratio=0.1):
        self.height = height
        self.width = width
        self.max_ratio = max_ratio

    def __call__(self, image, bbox, flow=False, trend=False, args=None, **kwargs):
        if len(bbox) == 5:
            x_min, y_min, x_max, y_max, _ = bbox
        elif len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
        else:
            raise ValueError
        img_h, img_w, _ = image.shape

        # Prepare args
        if args is None:
            args = {}
            args['ratio'] = self.max_ratio * random.uniform(0, 1)

        # Crop image
        ratio = args['ratio']
        x_min = int((1 - ratio) * x_min)
        y_min = int((1 - ratio) * y_min)
        x_max = int(x_max + ratio * (img_w - x_max))
        y_max = int(y_max + ratio * (img_h - y_max))
        image = image[y_min:y_max, x_min:x_max]
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # Adjust the values based on the size if it is optical flow
        if flow:
            image[:, :, 0] *= self.width / float(x_max - x_min)
            image[:, :, 1] *= self.height / float(y_max - y_min)

        # args: arguments for replaying this augmentation
        return {'image': image, 'bbox': None}, args


class RandomCrop:
    """
    Randomly crop the image based on the given cropping size
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, ratio=1., bbox=None, flow=False, args=None, **kwargs):
        img_h, img_w, _ = image.shape

        # Prepare args
        if args is None:
            args = {}
            args['x_min'] = random.randint(0, img_w - self.width)
            args['y_min'] = random.randint(0, img_h - self.height)

        # Crop image
        x_min = int(args['x_min'] * ratio)
        y_min = int(args['y_min'] * ratio)
        x_max = int(x_min + self.width * ratio)
        y_max = int(y_min + self.height * ratio)
        image = image[y_min:y_max, x_min:x_max]

        # args: arguments for replaying this augmentation
        return {'image': image, 'bbox': None}, args


class Resize:
    """
    resize the image based on the given size
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, bbox=None, flow=False, args=None, **kwargs):
        img_h, img_w, _ = image.shape

        # Resize image
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if flow:
            image[:, :, 0] *= self.width / img_w
            image[:, :, 1] *= self.height / img_h

        # args: arguments for replaying this augmentation
        return {'image': image, 'bbox': None}, args


class Rot90:
    '''
    Rotate n times 90 degree for the input tensor (counter-wise)
    '''

    def __call__(self, image, bbox, flow=False, args=None, **kwargs):
        # Prepare args
        if args is None:
            args = {}
            args['rot_num'] = random.randint(-2, 2)

        # Rotate image
        n = args['rot_num']
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0)  # (1, c, h, w)
        if flow:
            image = flow_rot90(image, n=n)
        else:
            image = torch.rot90(image, k=n, dims=[-2, -1])
        image = image.squeeze(dim=0).permute(1, 2, 0).numpy().astype(np.float)

        return {'image': image, 'bbox': bbox}, args


class Flip:
    '''
    Flip the input tensor
    flip_flag == 0: keep original
    flip_flag == 1: left-right flipping
    flip_flog == 2: up-down flipping
    '''

    def __call__(self, image, bbox, flow=False, args=None, **kwargs):
        # Prepare args
        if args is None:
            args = {}
            args['flip_flag'] = random.randint(0, 2)

        # Flip image
        flip_flag = args['flip_flag']
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0)  # (1, c, h, w)
        if flow:
            if flip_flag == 0:
                pass
            elif flip_flag == 1:
                image = flow_lr_flip(image)
                # image = trend_lr_flip(image)
            elif flip_flag == 2:
                image = flow_ud_flip(image)
                # image = trend_ud_flip(image)
            else:
                raise NotImplementedError('flip_flag: {}'.format(flip_flag))
        else:
            if flip_flag == 0:
                pass
            elif flip_flag == 1:
                image = torch.flip(image, dims=[-1, ])
            elif flip_flag == 2:
                image = torch.flip(image, dims=[-2, ])
            else:
                raise NotImplementedError('flip_flag: {}'.format(flip_flag))
        image = image.squeeze(dim=0).permute(1, 2, 0).numpy().astype(np.float32)

        return {'image': image, 'bbox': bbox}, args


class ColorJitter:
    def __init__(self, **kwargs):
        self.transform = A.ReplayCompose([A.ColorJitter(**kwargs), ])

    def __call__(self, image, bbox, flow=False, args=None, **kwargs):
        if flow:
            return {'image': image, 'bbox': bbox}, args

        # Prepare args
        if args is None:
            tsf_image = self.transform(image=image.astype(np.uint8))
            args = tsf_image['replay']
            image = tsf_image['image'].astype(np.float)
        else:
            image = A.ReplayCompose.replay(args, image=image.astype(np.uint8))['image'].astype(np.float)

        return {'image': image, 'bbox': bbox}, args


def _flow_rot_minus90(flow):
    rot_flow = flow.clone()
    # spatial rotation (-90 degree)
    rot_flow = torch.rot90(rot_flow, k=-1, dims=[-2, -1])
    rot_flow = rot_flow[:, [1, 0]]
    rot_flow[:, 0] = -1 * rot_flow[:, 0]
    return rot_flow


def _flow_rot_plus90(flow):
    rot_flow = flow.clone()
    # spatial rotation (+90 degree)
    rot_flow = torch.rot90(rot_flow, k=1, dims=[-2, -1])
    rot_flow = rot_flow[:, [1, 0]]
    rot_flow[:, 1] = -1 * rot_flow[:, 1]
    return rot_flow


def flow_rot90(flow, n=1):
    assert len(flow.shape) == 4
    if n == 0:
        return flow
    rot_func = _flow_rot_plus90 if n > 0 else _flow_rot_minus90
    for _ in range(abs(n)):
        flow = rot_func(flow)
    return flow


def flow_lr_flip(flow):
    assert len(flow.shape) == 4
    flip_flow = flow.clone()
    flip_flow = torch.flip(flip_flow, dims=[-1, ])
    flip_flow[:, 0] = -1 * flip_flow[:, 0]
    return flip_flow


def flow_ud_flip(flow):
    flip_flow = flow.clone()
    flip_flow = torch.flip(flip_flow, dims=[-2, ])
    flip_flow[:, 1] = -1 * flip_flow[:, 1]
    return flip_flow


def flow_diagonal_reverse(flow):
    reverse_flow = flow.clone()
    reverse_flow = -1 * reverse_flow
    return reverse_flow
