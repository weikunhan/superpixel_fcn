"""Modified SpixelFCN models 

Author: Weikun Han <weikunhan@gmail.com>

Reference: 
- https://github.com/weikunhan/superpixel_fcn/blob/master/models/Spixel_single_layer.py
- https://github.com/weikunhan/superpixel_fcn/blob/master/models/model_util.py
"""

import os
import torch
import torch.nn as nn

__all__ = ['SpixelNet', 'spixelnet_bn', 'spixelnet']

model_paths = {
    'spixelnet_bn': os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                                 'pretrain_models/spixelnet_bn_model_best.pth.tar')
}

def predict_mask(in_planes, out_planes):
    return  nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, batch_norm_flag):
        if batch_norm_flag:
            modules = [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)]
        else:
            modules = [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2),
                nn.LeakyReLU(0.1)]
        super(Conv, self).__init__(*modules)


class Deconv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        modules = [
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1)]
        super(Conv, self).__init__(*modules)


class SpixelNet(nn.Module):
    def __init__(self, batch_norm_flag=True):
        super(SpixelNet,self).__init__()
        self.conv1a = Conv(3, 16, 3, 1, batch_norm_flag)
        self.conv1b = Conv(16, 16, 3, 1, batch_norm_flag)
        self.conv2a = Conv(16, 32, 3, 2, batch_norm_flag)
        self.conv2b = Conv(32, 32, 3, 1, batch_norm_flag)
        self.conv3a = Conv(32, 64, 3, 2, batch_norm_flag)
        self.conv3b = Conv(64, 64, 3, 1, batch_norm_flag)
        self.conv4a = Conv(64, 128, 3, 2, batch_norm_flag)
        self.conv4b = Conv(128, 128, 3, 1, batch_norm_flag)
        self.conv5a = Conv(128, 256, 3, 2, batch_norm_flag)
        self.conv5b = Conv(256, 256, 3, 1, batch_norm_flag)
        self.deconv4 = Deconv(256, 128)
        self.conv4 = Conv(256, 128, 3, 1, batch_norm_flag)
        self.deconv3 = Deconv(128, 64)
        self.conv3 = Conv(128, 64, 3, 1, batch_norm_flag)
        self.deconv2 = Deconv(64, 32)
        self.conv2 = Conv(64, 32, 3, 1, batch_norm_flag)
        self.deconv1 = Deconv(32, 16)
        self.conv1 = Conv(32 , 16, 3, 1, batch_norm_flag)
        self.pred_mask = predict_mask(16, 9)
        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, 0.1)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1a(x)
        out1 = self.conv1b(x)
        x = self.conv2a(out1)
        out2 = self.conv2b(x)
        x = self.conv3a(out2)
        out3 = self.conv3b(x)
        x = self.conv4a(out3)
        out4 = self.conv4b(x)
        x = self.conv5a(out4)
        out5 = self.conv5b(x)
        x = self.deconv4(out5)
        x = torch.cat((out4, x), dim=1)
        x = self.conv4(x)
        x = self.deconv3(x)
        x = torch.cat((out3, x), dim=1)
        x = self.conv3(x)
        x = self.deconv2(x)
        x = torch.cat((out2, x), dim=1)
        x = self.conv2(x)
        x = self.deconv1(x)
        x = torch.cat((out1, x), dim=1)
        x = self.conv1(x)
        x = self.pred_mask(x)
        x = self.softmax(x)

        return x

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def _spixelnet(arch, data, pretrained, **kwargs):
    model = SpixelNet(**kwargs)
    
    if data:
        model.load_state_dict(data['state_dict'])
    elif pretrained:
        if torch.cuda.is_available():
            data = torch.load(model_paths[arch])
        else: 
            data = torch.load(model_paths[arch], map_location=torch.device('cpu'))
        
        model.load_state_dict(data['state_dict'])
    else:
        pass

    return model

def spixelnet_bn(data, pretrained=False, **kwargs):
    return _spixelnet('spixelnet_bn', data, pretrained, batch_norm_flag=True, **kwargs)

def spixelnet(data, pretrained=False, **kwargs):
    return _spixelnet('spixelnet', data, pretrained, batch_norm_flag=False, **kwargs)