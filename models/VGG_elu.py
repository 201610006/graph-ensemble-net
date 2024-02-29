import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from MobileNet_elu import ELU_block


# 定义VGG16的基本卷积块
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv, batch_norm=True):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# 定义VGG16模型
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),
            VGGBlock(64, 128, 2),
            VGGBlock(128, 256, 3),
            VGGBlock(256, 512, 3),
            VGGBlock(512, 512, 3)
        )

        self.elu = ELU_block(num_classes, 256, 512, 512)

    def forward(self, x):
        features = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i in {2, 3, 4}:
                features.append(x)

        out = self.elu(features)

        return out

# 创建VGG16模型
def vgg16(num_classes=10):
    return VGG16(num_classes=num_classes)

# 获取VGG16预训练模型
def vgg16_elu_pretrained(pretrained=False, **kwargs):
    model = vgg16(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'), strict=False)
    return model
