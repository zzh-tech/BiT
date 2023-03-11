import os
import random
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
from datetime import datetime
from itertools import chain


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ckpt_convert(param):
    return {
        k.replace('module.', ''): v
        for k, v in param.items()
        if 'module.' in k
    }


class Logger:
    """
    Logger class to record training log
    """

    def __init__(self, file_path, verbose=True):
        self.verbose = verbose
        self.create_dir(file_path)
        self.logger = open(file_path, 'a+')

    def create_dir(self, file_path):
        dir = osp.dirname(file_path)
        os.makedirs(dir, exist_ok=True)

    def __call__(self, *args, prefix='', timestamp=True):
        if timestamp:
            now = datetime.now()
            now = now.strftime("%Y/%m/%d, %H:%M:%S - ")
        else:
            now = ''
        if prefix == '':
            info = prefix + now
        else:
            info = prefix + ' ' + now
        for msg in args:
            if not isinstance(msg, str):
                msg = str(msg)
            info += msg + '\n'
        self.logger.write(info)
        if self.verbose:
            print(info, end='')
        self.logger.flush()

    def __del__(self):
        self.logger.close()


class AverageMeter(object):
    """
    Compute and store the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_model_params(model):
    """
    default unit of measurement is Million (M.)
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    if isinstance(model, list):
        params = chain(*[submodel.parameters() for submodel in model])
    else:
        params = model.parameters()
    for param in params:
        mul_value = np.prod(param.size())
        total_params += mul_value
        if param.requires_grad:
            trainable_params += mul_value
        else:
            non_trainable_params += mul_value
    total_params /= 1e6
    trainable_params /= 1e6
    non_trainable_params /= 1e6
    msg = 'total_params: {:.3f}\ntrainable_params: {:.3f}\nnon_trainable_params: {:.3f}'.format(
        total_params, trainable_params, non_trainable_params
    )
    return msg
