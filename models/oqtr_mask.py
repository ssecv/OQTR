import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, with_relu=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.GroupNorm(8, out_planes)
        self.with_relu = with_relu
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x


class RefineModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.conv_cat = BasicConv2d(2 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        x_cat = self.conv_cat(torch.cat((x0, x1), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class MaskDecoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.relu = nn.ReLU(True)

        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv3 = nn.Conv2d(channel, channel, 1)
        self.conv5 = nn.Conv2d(2 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1

        x2_1 = self.conv_upsample2(F.interpolate(x1, size=x3.shape[-2:], mode='bilinear')) * \
            self.conv_upsample2(F.interpolate(x2, size=x3.shape[-2:], mode='bilinear')) * x3
        x2_1 = self.conv3(x2_1)

        x2_2 = torch.cat((x2_1, self.conv_upsample2(F.interpolate(x1_1, size=x3.shape[-2:], mode='bilinear'))), 1)

        x3_1 = self.conv_concat2(x2_2)
        x3_1 = self.conv_concat2(x3_1)
        x = self.conv5(x3_1)

        return x


class CrossFusionModule(nn.Module):

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, channel=64):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5
        self.rm5 = RefineModule(512, channel * 2)

    def forward(self, x, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        # (bs, num_queries, num_heads, hidden_dim//num_heads)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        # (bs, num_heads, hidden_dim//num_heads, w/32, h/32)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim2 // self.num_heads, k.shape[-2], k.shape[-1])
        # (bs, num_queries, num_heads, w/32, h/32)
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)

        x = torch.einsum("bqwh,bnmwh->bnqmwh", x, weights)
        x = torch.sum(x, 3).flatten(0, 1)
        x = self.rm5(x)

        return x


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)
