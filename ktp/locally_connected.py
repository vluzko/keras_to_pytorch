# coding=utf-8
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from torch.nn.functional import unfold


def conv2d_local(input, weight, bias=None, padding=0, stride=1, dilation=1):
    """Calculate the local convolution.

    Args:
        input:
        weight:
        bias:
        padding:
        stride:
        dilation:

    Returns:

    """
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))

    out_height, out_width, out_channels, in_channels, kernel_height, kernel_width = weight.size()
    kernel_size = (kernel_height, kernel_width)

    # N x [inC * kH * kW] x [outH * outW]
    cols = unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
    cols2 = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)

    output_size = out_height * out_width
    input_size = in_channels * kernel_height * kernel_width
    weights_view = weight.view(output_size, out_channels, input_size)
    permuted_weights = weights_view.permute(0, 2, 1)
    out = torch.matmul(cols2, permuted_weights)
    out = out.view(cols2.size(0), out_height, out_width, out_channels).permute(0, 3, 1, 2)

    if bias is not None:
        out = out + bias.expand_as(out)
    return out


class Conv2dLocal(Module):
    """A 2D locally connected layer.

    Attributes:
        weight (torch.Tensor): The weights. out_height x out_width x out_channels x in_channels x kernel_height x kernel_width
    """

    def __init__(self, in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2dLocal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))

        self.weight = Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input):
        return conv2d_local(
            input, self.weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation)
