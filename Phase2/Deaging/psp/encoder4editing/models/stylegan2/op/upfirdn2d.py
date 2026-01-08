import torch
import torch.nn.functional as F

"""
Windows-safe, shape-correct upfirdn2d implementation
Uses native PyTorch ops only
Keeps GPU acceleration
No CUDA / C++ extensions
"""


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(
        input,
        kernel,
        up, up,
        down, down,
        pad[0], pad[1], pad[0], pad[1]
    )


def upfirdn2d_native(
    input, kernel,
    up_x, up_y,
    down_x, down_y,
    pad_x0, pad_x1,
    pad_y0, pad_y1
):
    batch, channel, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    # NHWC
    input = input.permute(0, 2, 3, 1)

    # Upsample
    out = input.view(batch, in_h, 1, in_w, 1, channel)
    out = F.pad(out, (0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1))
    out = out.view(batch, in_h * up_y, in_w * up_x, channel)

    # Padding
    out = F.pad(
        out,
        (
            0, 0,
            max(pad_x0, 0), max(pad_x1, 0),
            max(pad_y0, 0), max(pad_y1, 0),
        ),
    )

    out = out[
        :,
        max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    # Convolution
    out = out.permute(0, 3, 1, 2)
    out = out.reshape(-1, 1, out.shape[2], out.shape[3])

    kernel = torch.flip(kernel, [0, 1]).to(out.device, out.dtype)
    kernel = kernel.view(1, 1, kernel_h, kernel_w)

    out = F.conv2d(out, kernel)

    out = out.reshape(
        batch,
        channel,
        out.shape[2],
        out.shape[3]
    )

    # Downsample
    return out[:, :, ::down_y, ::down_x]
