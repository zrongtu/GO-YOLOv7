from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

class MobileOneBlock(nn.Module):
    def __init__(self,in_channels: int,out_channels: int,kernel_size: int,stride: int = 1,padding: int = 0,dilation: int = 1,groups: int = 1,inference_mode: bool = False,use_se: bool = False,num_conv_branches: int = 1,is_linear: bool = False) -> None:
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        if is_linear:
            self.activation = nn.Identity()
        else:
            self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=True)
        else:
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self.ConvBn(kernel_size=kernel_size,padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self.ConvBn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

  



