import torch
from torch.nn.modules.loss import _Loss


class CharbonnierLoss(_Loss):
    """
    Charbonnier loss (color mean)
    """

    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        diff_sq = diff * diff
        diff_sq_color = torch.mean(diff_sq, 1, True)
        error = torch.sqrt(diff_sq_color + self.eps * self.eps)
        loss = torch.mean(error)
        return loss
