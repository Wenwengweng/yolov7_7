import torch
import torch.nn as nn
import math


class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class new_Block(nn.Module):
    def __init__(self, channel):
        super(new_Block, self).__init__()

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, c, h, w = x.size()

        # c
        x_w = torch.mean(x, dim=3, keepdim=True)
        x_w_h = torch.mean(x_w, dim=2, keepdim=True)
        # print(x_w_h.shape)
        # w
        x_h = torch.mean(x, dim=2, keepdim=True)
        x_h_c = torch.mean(x_h, dim=1, keepdim=True)
        # print(x_h_c.shape)
        # h
        x_c = torch.mean(x, dim=1, keepdim=True)
        x_c_w = torch.mean(x_c, dim=3, keepdim=True)
        # print(x_c_w.shape)

        s_c = self.sigmoid(x_w_h)
        # print(s_c.shape)
        x_h_c = s_c * x_h_c
        # print(x_h_c.shape)
        x_c_w = s_c * x_c_w
        # print(x_c_w.shape)

        x_h_c_conv_relu = self.relu(self.bn(x_h_c))
        x_c_w_conv_relu = self.relu(self.bn(x_c_w))

        s_w = self.sigmoid(x_h_c_conv_relu)
        s_h = self.sigmoid(x_c_w_conv_relu)

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


if __name__ == "__main__":
    # 创建模拟输入
    batch_size = 1
    channels = 512
    height, width = 640, 640
    x = torch.randn(batch_size, channels, height, width)

    # 创建模型实例
    ca_block = CA_Block(channel=channels)
    new_block = new_Block(channel=channels)
    se_block = se_block(channel=channels)
    cbam_block = cbam_block(channel=channels)
    eca_block = eca_block(channel=channels)

    # 统计模型参数量
    ca_block_params = sum(p.numel() for p in ca_block.parameters())
    new_block_params = sum(p.numel() for p in new_block.parameters())
    se_block_params = sum(p.numel() for p in se_block.parameters())
    cbam_block_params = sum(p.numel() for p in cbam_block.parameters())
    eca_block_params = sum(p.numel() for p in eca_block.parameters())

    print("CA_Block 参数量:", ca_block_params)
    print("new_Block 参数量:", new_block_params)
    print("se_Block 参数量:", se_block_params)
    print("cbam_Block 参数量:", cbam_block_params)
    print("eca_Block 参数量:", eca_block_params)
