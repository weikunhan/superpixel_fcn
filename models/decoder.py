"""Decoder module for SpixelNet

Using decoder for gradually recovering the spatial information for SpixelNet

Author: Weikun Han <weikunhan@gmail.com>

Reference: 
- https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/decoder.py
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.convs = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1))
        self.conv2 = nn.Conv2d(256, 9, kernel_size=3, stride=1, padding=1)

    def forward(self, x, out):
        size = out.shape[-2:]
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x = torch.cat((x, out), dim=1)
        x = self.convs(x)
        x = self.conv2(x)
        
        return x