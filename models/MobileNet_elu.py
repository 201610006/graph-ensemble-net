import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from einops import rearrange
from functools import partial
from GCN_vertex import gcn_backbone
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)

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
        if activation == "relu":
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

class ELU_block(nn.Module):
    def __init__(self, num_classes=10, c2=256, c3=512, c4=1024):
        super(ELU_block, self).__init__()
        self.layer2_c = c2  # 512
        self.layer3_c = c3  # 1024
        self.layer4_c = c4  # 2048
        self.fix = 64  #
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.mm_mrg4 = MRGblock(self.layer4_c, self.fix * 4)
        self.mm_mrg3 = MRGblock(self.layer3_c, self.fix * 4)
        self.mm_mrg2 = MRGblock(self.layer2_c, self.fix * 4)
        self.ff1_1x1 = nn.Conv2d(self.fix * 4 * 3, self.fix, kernel_size=1, padding=2)
        self.ff2_1x1 = nn.Conv2d(self.fix * 4 * 3, self.fix * 2, kernel_size=1, padding=2)
        self.ff3_1x1 = nn.Conv2d(self.fix * 4 * 3, self.fix * 4, kernel_size=1, padding=2)

        self.cv1_1x1 = nn.Conv2d(self.fix * 25, self.fix * 4, kernel_size=1, groups=self.fix, padding=0)
        self.cv2_1x1 = nn.Conv2d(self.fix * 2 * 9, self.fix * 4, kernel_size=1, groups=self.fix * 2, padding=0)

        self.mm_gcn4 = gcn_backbone(n_filters=self.fix * 4)
        self.mm_gcn3 = gcn_backbone(n_filters=self.fix * 4)
        self.mm_gcn2 = gcn_backbone(n_filters=self.fix * 4)

        self.linear_m1 = nn.Linear(self.fix * 12, num_classes)
        self.linear_s1 = nn.Linear(self.fix * 12, 1)

        self.linear_m2 = nn.Linear(self.fix * 12, num_classes)
        self.linear_s2 = nn.Linear(self.fix * 12, 1)

        self.linear_m3 = nn.Linear(self.layer4_c, num_classes)
        self.linear_s3 = nn.Linear(self.layer4_c, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_out = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        # torch.Size([2, 512, 13, 13]) torch.Size([2, 256, 26, 26]) torch.Size([2, 128, 52, 52])
        out2 = x[0]
        out3 = x[1]
        out4 = x[2]
        #print(out4.shape, out3.shape, out2.shape)
        m4 = self.mm_mrg4(out4)
        # m3 = torch.cat([self.upsample(m4), out3], dim=1)
        m3 = self.mm_mrg3(out3)
        # m2 = torch.cat([self.upsample(m3), out2], dim=1)
        m2 = self.mm_mrg2(out2)

        #print(m4.shape, m3.shape, m2.shape)
        fusion_f = torch.cat([self.upsample(m4), m3, self.maxpool(m2)], dim=1)
        fusion_ff1 = self.ff1_1x1(fusion_f)
        fusion_ff2 = self.ff2_1x1(fusion_f)
        fusion_ff3 = self.ff3_1x1(fusion_f)

        m2 = rearrange(fusion_ff1, 'b n (k w) (l h) -> b (n k l) w h', h=6, w=6).contiguous()
        m2 = self.cv1_1x1(m2)
        m2 = self.mm_gcn2(m2)
        m2 = self.avgpool(m2)
        m2 = m2.view(m2.size(0), -1)

        m3 = rearrange(fusion_ff2, 'b n (k w) (l h) -> b (n k l) w h', h=10, w=10).contiguous()
        m3 = self.cv2_1x1(m3)
        m3 = self.mm_gcn3(m3)
        m3 = self.avgpool(m3)
        m3 = m3.view(m3.size(0), -1)

        m4 = self.mm_gcn4(fusion_ff3)
        m4 = self.avgpool(m4)
        m4 = m4.view(m4.size(0), -1)

        head1 = torch.cat([m2, m3, m4], dim=1)
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

        out = (score1 * F.normalize(class1, p=2, dim=1) + score2 * F.normalize(class2, p=2,
                                                                               dim=1) + score3 * F.normalize(class3,
                                                                                                             p=2,
                                                                                                             dim=1)) / (
                          score2 + score3 + score1 + 1e-6)
        out = self.linear_out(out)

        return out, (class1, score1), (class2, score2), (class3, score3)


# 定义MobileNet的基本模块
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

# 定义Depthwise Separable Convolution模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# 定义MobileNet模型
class MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1)
        )
        self.elu = ELU_block(num_classes)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i in {5, 10, 11}:
                features.append(x)

        out = self.elu(features)

        return out

# 创建MobileNet模型
def mobilenet(num_classes=1000):
    return MobileNet(num_classes=num_classes)

# 获取MobileNet预训练模型
def mobilenet_elu_pretrained(pretrained=False, **kwargs):
    model = mobilenet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'), strict=False)
    return model
