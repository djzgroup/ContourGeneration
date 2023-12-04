import torch
from torch import nn
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

#空间注意力机制
class SpatialAttentionModule(nn.Module):
    """ this function is used to achieve the spatial attention module in CBAM paper"""
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(1, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, ctx, x):
        out1 = torch.mean(x,dim=1,keepdim=True) #B,1,N

        out2, _ = torch.max(x, dim=1,keepdim=True)#B,1,N

        out = torch.cat([out2, out1], dim=1) #B,2,N

        out = self.conv1(out) #B,1,N
        out = self.bn(out) #B,1,N
        out =self.relu(out) #B,1,N

        out = self.sigmoid(out) #b, c, n
        return out * x
class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)

def lr_func(epoch):
    if epoch <= start_epoch:
        return 1.0
    elif epoch <= end_epoch:
        total = end_epoch - start_epoch
        delta = epoch - start_epoch
        frac = delta / total
        return (1-frac) * 1.0 + frac * (end_lr / start_lr)
    else:
        return end_lr / start_lr
