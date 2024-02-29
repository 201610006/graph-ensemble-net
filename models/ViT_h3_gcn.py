import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from functools import partial
import torch.nn.functional as F
from einops import rearrange, repeat
from backbones.VIT import Transformer

nonlinearity = partial(F.relu, inplace=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out * x
        return out

class ViT_block(nn.Module):
    def __init__(self, *, num_patches=49, num_classes=10, dim=768, depth=4, heads=9, mlp_dim=1024, pool='cls', channels=768, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.emb_x = nn.Linear(channels, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.pool = pool
        self.layer_nor = nn.LayerNorm(dim)


    def forward(self, img):
        img = rearrange(img, 'b c h w -> b (h w) c')
        x = self.emb_x(img)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # (b, dim)

        x = self.to_latent(x)  # Identity (b, dim)

        return self.layer_nor(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, dim=768, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.layer2_c = 128# 512
        self.layer3_c = 256#1024
        self.layer4_c = 512#2048

        self.mm_mrg4 = MRGblock(self.layer4_c, self.layer3_c)
        self.mm_mrg3 = MRGblock(self.layer3_c, self.layer3_c)
        self.mm_mrg2 = MRGblock(self.layer2_c, self.layer3_c)

        self.ff1_1x1 = nn.Conv2d(self.layer3_c * 3, self.layer3_c * 3, kernel_size=4, stride=4, padding=1)

        self.vit_b = ViT_block(dim=dim, depth=6)

        self.linear_m1 = nn.Linear(dim, num_classes)
        self.linear_s1 = nn.Linear(dim, 1)

        self.linear_m2 = nn.Linear(self.layer3_c * 3, num_classes)
        self.linear_s2 = nn.Linear(self.layer3_c * 3, 1)

        self.linear_m3 = nn.Linear(self.layer4_c, num_classes)
        self.linear_s3 = nn.Linear(self.layer4_c, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_out = nn.Linear(num_classes, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out1 = self.layer1(out)
        out2 = self.layer2(out1)

        out3 = self.layer3(out2)
        out4 = self.layer4(out3)


        m4 = self.mm_mrg4(out4)
        # m3 = torch.cat([self.upsample(m4), out3], dim=1)
        m3 = self.mm_mrg3(out3)
        # m2 = torch.cat([self.upsample(m3), out2], dim=1)
        m2 = self.mm_mrg2(out2)

        fusion_f = torch.cat([self.upsample(m4), m3, self.maxpool(m2)], dim=1)

        fusion_ff = self.ff1_1x1(fusion_f)

        head1 = self.vit_b(fusion_ff)

        class1 = self.linear_m1(head1)
        score1 = torch.sigmoid(self.linear_s1(head1))

        head2 = self.avgpool(fusion_f)
        head2 = head2.view(head2.size(0), -1)
        class2 = self.linear_m2(head2)
        score2 = torch.sigmoid(self.linear_s2(head2))

        head3 = self.avgpool(out4)
        head3 = head3.view(head3.size(0), -1)
        class3 = self.linear_m3(head3)
        score3 = torch.sigmoid(self.linear_s3(head3))

        out = (score1 * F.normalize(class1, p=2, dim=1) + score2 * F.normalize(class2, p=2, dim=1) + score3 * F.normalize(class3, p=2, dim=1) )/(score2 + score3 + score1 + 1e-6)
        out = self.linear_out(out)

        return out, (class1, score1), (class2, score2), (class3, score3)


def resnet18(num_classes=1000, pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    if pretrained:
        load_pretrained_params(model, 'https://download.pytorch.org/models/resnet18-5c106cde.pth')

    return model


def resnet34(num_classes=1000, pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

    if pretrained:
        load_pretrained_params(model, 'https://download.pytorch.org/models/resnet34-333f7ec4.pth')

    return model


def resnet50(num_classes=1000, pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

    if pretrained:
        load_pretrained_params(model, 'https://download.pytorch.org/models/resnet50-19c8e357.pth')

    return model


def resnet101(num_classes=1000, pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

    if pretrained:
        load_pretrained_params(model, 'https://download.pytorch.org/models/resnet50-19c8e357.pth')

    return model


def load_pretrained_params(model, url):
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

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

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ConvGRU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.ga1 = SpatialAttention()
        self.ga2 = SpatialAttention()

        self.conv_a1 = Conv_Bn_Activation(in_channels, in_channels, kernel_size=1, stride=1, activation='leaky')
        self.conv_a2 = Conv_Bn_Activation(in_channels, in_channels, kernel_size=1, stride=1, activation='leaky')

        self.conv_a3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x_t):

        ga1 = self.ga1(x_t)
        x_c = ga1 * x_t

        x_b = self.conv_a1(x_c)

        x_b = x_b * (1-ga1)

        x_t = self.conv_a2(x_b+x_c) + x_t

        out = self.conv_a3(x_t)

        return out

class MRGblock(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(MRGblock, self).__init__()

        self.conv_u = Conv_Bn_Activation(in_channel, mid_channel, kernel_size=1, stride=1, activation='leaky')

        self.conv_d0 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, dilation=1, padding=1)#, groups=mid_channel//8)
        self.conv_d1 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, dilation=3, padding=3)#, groups=mid_channel//4)
        self.conv_d3 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, dilation=5, padding=5)#, groups=mid_channel//2)
        self.conv_1x1 = nn.Conv2d(mid_channel, mid_channel, kernel_size=1, padding=0)

        # self.bnm = nn.BatchNorm2d(mid_channel)

        self.conv_n1 = Conv_Bn_Activation(mid_channel, mid_channel, kernel_size=1, stride=1, activation='leaky')
        self.conv_n2 = Conv_Bn_Activation(mid_channel, mid_channel, kernel_size=1, stride=1, activation='leaky')
        self.conv_n3 = Conv_Bn_Activation(mid_channel, mid_channel, kernel_size=1, stride=1, activation='leaky')
        self.conv_n4 = Conv_Bn_Activation(mid_channel, mid_channel, kernel_size=1, stride=1, activation='leaky')

        self.out_norm = nn.BatchNorm2d(mid_channel)

        self.gru1 = ConvGRU(mid_channel, mid_channel)
        self.gru2 = ConvGRU(mid_channel, mid_channel)
        self.gru3 = ConvGRU(mid_channel, mid_channel)
        self.gru4 = ConvGRU(mid_channel, mid_channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_u(x)
        x_t1 = nonlinearity(self.conv_d0(x))
        x_t1 = self.conv_n1(x_t1)
        y1 = self.gru1(x_t1)

        x_t2 = nonlinearity(self.conv_1x1(self.conv_d1(x)))
        x_t2 = self.conv_n2(x_t2)
        y2 = self.gru2(x_t2)

        x_t3 = nonlinearity(self.conv_1x1(self.conv_d1(self.conv_d0(x))))
        x_t3 = self.conv_n3(x_t3)
        y3 = self.gru3(x_t3)

        x_t4 = nonlinearity(self.conv_1x1(self.conv_d3(self.conv_d1(self.conv_d0(x)))))
        x_t4 = self.conv_n4(x_t4)
        y4 = self.gru4(x_t4)

        return self.out_norm(x + y1 + y2 + y3 + y4)


def lce(predict, onehot_target, conf_score, gamma=0.2):
    b,c = predict.shape
    onehot_target = torch.eye(c)[onehot_target]
    l_p = -(torch.log(predict.softmax(dim=1) * conf_score + (1-conf_score) * onehot_target)*onehot_target).sum(dim=1).mean()
    l_r = -gamma * torch.log(conf_score).sum(dim=1).mean()
    return l_p+l_r

