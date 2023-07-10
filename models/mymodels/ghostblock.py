import torch
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU
import math

def ConvBnAct(in_channel, out_channel, kernel_size, stride, padding, group=1, has_bn=True, is_linear=False):
    return Sequential(
        Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding, groups=group, bias=False),
        BatchNorm2d(out_channel) if has_bn else Sequential(),
        ReLU(inplace=True) if not is_linear else Sequential()
    )

class GhostModule(Module):
    def __init__(self, in_channel, out_channel, is_linear=False):
        super(GhostModule, self).__init__()
        self.out_channel = out_channel
        init_channel = math.ceil(out_channel / 2)
        new_channel = init_channel

        self.primary_conv = ConvBnAct(in_channel, init_channel, 1, 1, 0, is_linear=is_linear)
        self.cheap_operation = ConvBnAct(init_channel, new_channel, 3, 1, 1, group=init_channel, is_linear=is_linear)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channel, :, :]


