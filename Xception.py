"""
Creates an Xception and U-Net hybrid Model

Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

__all__ = ['xception']

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class SeparableConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False):
        super(SeparableConvTranspose2d, self).__init__()

        self.conv_transpose1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride, padding, output_padding
                               , groups=in_channels, bias=bias)
        self.pointwise = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, output_padding=0, groups=1, bias=bias)

    def forward(self, x):
        x = self.conv_transpose1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class DBlock(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(DBlock, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.ConvTranspose2d(in_filters, out_filters, kernel_size=1, stride=strides, bias=False, output_padding=1)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConvTranspose2d(in_filters, out_filters, 3, stride=1, output_padding=0, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConvTranspose2d(filters, filters, 3, stride=1, output_padding=0, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConvTranspose2d(in_filters, out_filters, 3, stride=1, output_padding=0, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.Upsample(scale_factor=2))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception and U-Net inspired Neural network
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self):
        super(Xception, self).__init__()
        self.input = nn.Conv2d(288, 32, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1  = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2  = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3  = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.dblock3 = DBlock(728, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.dblock2 = DBlock(256, 128, 2, 2, start_with_relu=True, grow_first=True)
        self.dblock1 = DBlock(128, 3, 2, 2, start_with_relu=True, grow_first=True)


    def forward(self, x):
        x = self.input(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        x = self.dblock3(x)
        x = self.dblock2(x)
        x = self.dblock1(x)

        x = torch.sigmoid(x)

        return x


def xception(pretrained=False, **kwargs):

    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model

