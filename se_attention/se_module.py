"""
SENet: channel attention module
"""
from torch import nn


class SELayer(nn.Module):
    """
    SELayer: FC Channel Attention
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayerWithCBH(nn.Module):
    """
    LightWeight Channel Attention
    """
    def __init__(self, channel, reduction=16):
        super(SELayerWithCBH, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CBH = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.CBH(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
