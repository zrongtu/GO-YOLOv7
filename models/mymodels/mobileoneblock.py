from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,out_channels=int(in_channels * rd_ratio),kernel_size=1,stride=1,bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),out_channels=in_channels,kernel_size=1,stride=1,bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


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

    def ConvBn(self,kernel_size: int, padding: int) -> nn.Sequential:
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=kernel_size,stride=self.stride,padding=padding,groups=self.groups,bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list

    def BnFuse(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def reparameterize(self):
        if self.inference_mode:
            return
        kernel, bias = self.KernelBias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,out_channels=self.rbr_conv[0].conv.out_channels,kernel_size=self.rbr_conv[0].conv.kernel_size,stride=self.rbr_conv[0].conv.stride,padding=self.rbr_conv[0].conv.padding,dilation=self.rbr_conv[0].conv.dilation,groups=self.rbr_conv[0].conv.groups,bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def KernelBias(self) -> Tuple[torch.Tensor, torch.Tensor]:

        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self.BnFuse(self.rbr_scale)
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self.BnFuse(self.rbr_skip)

        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self.BnFuse(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final



