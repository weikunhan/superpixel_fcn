"""ASPP module for SpixelNet

Using ASPP for increasing the receptive field of individual neurons in SpixelNet

Author: Weikun Han <weikunhan@gmail.com>

Reference: 
- https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
"""

import torch
import torch.nn as nn


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
            # nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        
        for mod in self:
            x = mod(x)
        
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rates = tuple(atrous_rates)
        
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        
        for conv in self.convs:
            res.append(conv(x))
        
        res = torch.cat(res, dim=1)
        
        return self.project(res)