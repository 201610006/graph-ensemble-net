import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


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
    def __init__(self, block, num_blocks, growth_rate=32, num_classes=1000):
        super(DenseNet, self).__init__()
        self.in_channels = 64  # 设置初始输入的通道数

        self.features = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, growth_rate, num_blocks[0], 0)
        self.layer2 = self._make_layer(block, growth_rate, num_blocks[1], 1)
        self.layer3 = self._make_layer(block, growth_rate, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, growth_rate, num_blocks[3], 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _make_layer(self, block, growth_rate, num_blocks, layer):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_channels, growth_rate))
            self.in_channels += growth_rate
        if layer>0:
            trans = TransitionBlock(self.in_channels, self.in_channels//2)
            layers.append(trans)
            self.in_channels = self.in_channels // 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)

        layer4 = self.layer4(layer3)

        return layer2, layer3, layer4


# 创建DenseNet模型
def densenet(num_classes=1000):
    return DenseNet(BasicBlock, [6, 12, 24, 16], num_classes=num_classes)


# 获取DenseNet预训练模型
def densenet_pretrained(pretrained=False, **kwargs):
    model = densenet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet121-a639ec97.pth'),
                              strict=False)
    return model

