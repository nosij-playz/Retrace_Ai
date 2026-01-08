import torch
from torch import nn
import torch.nn.functional as F

"""
Windows-safe fallback for StyleGAN2 fused ops.
Keeps GPU enabled, disables CUDA JIT compilation.
"""


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        if self.bias is not None:
            if input.ndim == 2:
                input = input + self.bias
            else:
                input = input + self.bias.view(1, -1, 1, 1)
        return F.leaky_relu(input, self.negative_slope) * self.scale


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        if input.ndim == 2:
            input = input + bias
        else:
            input = input + bias.view(1, -1, 1, 1)
    return F.leaky_relu(input, negative_slope) * scale
