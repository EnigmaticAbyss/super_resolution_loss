#Enhanced Super-Resolution Generative Adversarial Network

import torch.nn as nn

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)

class ESRGANGenerator(nn.Module):
    def __init__(self, in_channels=3, num_blocks=16):
        super(ESRGANGenerator, self).__init__()
        self.initial = nn.Conv2d(in_channels, 64, 9, 1, 4)
        self.blocks = nn.Sequential(
            *[ResidualInResidualDenseBlock(64) for _ in range(num_blocks)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.output = nn.Conv2d(64, in_channels, 9, 1, 4)

    def forward(self, x):
        x = self.initial(x)
        res = x
        x = self.blocks(x)
        x = self.upsample(x + res)
        return self.output(x)
