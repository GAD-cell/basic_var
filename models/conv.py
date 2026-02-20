"""
Implement the module that will be applied after every "unpatchify" step in the decoder.
This will help images look smooth and just a raw concatenation of patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConv(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        # x: [B, C, H, W]
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return x + out