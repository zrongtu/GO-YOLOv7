import math

import torch
from torch.nn import Module

from models.mymodels.mobileoneblock import MobileOneBlock

class GMConv(Module):
    def __init__(self, in_channel, out_channel, is_linear=False, inference_mode=False, num_conv_branches=1):
        super(GMConv, self).__init__()
        self.out_channel = out_channel
        half_outchannel = math.ceil(out_channel / 2)

        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches

        self.primary_conv = MobileOneBlock(in_channels=in_channel,out_channels=half_outchannel,kernel_size=1,stride=1,padding=0,groups=1,inference_mode=self.inference_mode,use_se=False,num_conv_branches=self.num_conv_branches,is_linear=is_linear)
        self.cheap_operation = MobileOneBlock(in_channels=half_outchannel,out_channels=half_outchannel,kernel_size=3,stride=1,padding=1,groups=half_outchannel,inference_mode=self.inference_mode,use_se=False,num_conv_branches=self.num_conv_branches,is_linear=is_linear)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out









