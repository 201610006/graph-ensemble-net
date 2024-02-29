import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
nonlinearity = partial(F.relu, inplace=True)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = torch.cat([x, out], 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(BasicBlock(in_channels, growth_rate))
        for i in range(1, num_layers):
            self.layers.append(BasicBlock(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate=32, block_config=(6, 12, 24, 16)):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        in_channels = 2 * growth_rate
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(in_channels, num_layers, growth_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            in_channels += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionBlock(in_channels, in_channels // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                in_channels = in_channels // 2
        self.features.add_module('norm5', nn.BatchNorm2d(in_channels))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        self.gru = MRGblock_res(num_classes, 1024, 256)

    def forward(self, x):
        features = self.features(x)

        out = self.gru(features)

        return out

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! ")

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

class MRGblock_res(nn.Module):
    def __init__(self, num_classes, in_channel, channel):
        super(MRGblock_res, self).__init__()
        # self.conv_u = nn.Conv2d(in_channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv_u = Conv_Bn_Activation(in_channel, channel, kernel_size=1, stride=1, activation='leaky')
        self.dilate1 = nn.Conv2d(in_channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.gru1 = GRU_block(num_classes, channel, channel)
        self.gru2 = GRU_block(num_classes, channel, channel)
        self.gru3 = GRU_block(num_classes, channel, channel)
        self.gru4 = GRU_block(num_classes, channel, channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv_u(x)
        dilate1_out = nonlinearity(self.dilate1(x))

        h_t = self.avgpool(x1)
        h_t = torch.flatten(h_t, 1)
        x_t = self.avgpool(dilate1_out)
        x_t = torch.flatten(x_t, 1)
        y1, x_out = self.gru1(x_t, h_t)

        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x1)))
        x_t2 = self.avgpool(dilate2_out)
        x_t2 = torch.flatten(x_t2, 1)
        y2, x_out = self.gru2(x_t2, x_out)

        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        x_t3 = self.avgpool(dilate3_out)
        x_t3 = torch.flatten(x_t3, 1)
        y3, x_out = self.gru3(x_t3, x_out)

        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        x_t4 = self.avgpool(dilate4_out)
        x_t4 = torch.flatten(x_t4, 1)
        y4, x_out = self.gru4(x_t4, x_out)

        return y1, y2, y3, y4

class GRU_block(nn.Module):
    def __init__(self, num_classes=10, in_channels=256, out_channels=256):
        super().__init__()
        self.linear_x_z = nn.Linear(in_channels, out_channels)  # (feature_size, hidden_size)
        self.linear_h_z = nn.Linear(in_channels, out_channels)  # (hidden_size, hidden_size)
        self.linear_x_r = nn.Linear(in_channels, out_channels)
        self.linear_h_r = nn.Linear(in_channels, out_channels)

        self.linear_t = nn.Linear(in_channels, out_channels)
        self.linear_u = nn.Linear(in_channels, out_channels)

        self.linear_out = nn.Linear(out_channels, num_classes)

    def forward(self, x, h_t_1):
        z_t = torch.sigmoid(self.linear_x_z(x) + self.linear_h_z(h_t_1))
        r_t = torch.sigmoid((self.linear_x_r(x) + self.linear_h_r(h_t_1)))

        h_hat_t = torch.tanh(self.linear_t(x) + self.linear_u(torch.mul(r_t, h_t_1)))

        h_t = torch.mul((1 - z_t), h_t_1) + torch.mul(z_t, h_hat_t)

        y = self.linear_out(h_t)
        return y, h_t

