"""
cbam:convolutional block attention module
channel and spatial attention
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel attention:
    """

    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial attention:
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Channel + Spatial attention: == CBAM
    """
    """
    B C H W ==> B C H*W ==> B C H W ==>B C H*W ==>Cov operate  (B C H*W 1) * B C H W = B C H W
    input : B C H W
    return: B C H W
    """

    def __init__(self, in_planes, reduction):
        super(CBAM, self).__init__()
        self.Channel = ChannelAttention(in_planes, reduction)
        self.Spatial = SpatialAttention()

    def forward(self, x):
        channel = self.Channel(x)
        x = x * channel
        spatial = self.Spatial(x)
        x = x * spatial
        return x
