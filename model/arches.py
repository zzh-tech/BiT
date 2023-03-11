import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=(stride, stride),
                     padding=(kernel_size // 2), bias=bias, groups=groups)


def default_norm(n_feats):
    return nn.BatchNorm2d(n_feats)


def default_act():
    return nn.ReLU(True)


class ResBlock(nn.Module):
    def __init__(
            self, n_feats, kernel_size, bias=True,
            conv=default_conv, norm=False, act=default_act):

        super(ResBlock, self).__init__()

        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if norm:
                modules.append(norm(n_feats))
            if act and i == 0:
                modules.append(act())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res
