#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/11/15 11:08 PM
# @Author : Zhengchang
# @Email : 819192552@qq.com
# @File : pooling.py
# @WhatToDO :
import torch
from torch import nn


class Pooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)     # [80, 64, 120]
    >>> asp_layer = AttentiveStatisticsPooling(64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-12

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [B, T, D].
        """

        # 计算mean和std
        def _compute_statistics(x, dim=2, eps=self.eps):
            mean = x.sum(dim)
            std = torch.sqrt(
                (x - mean.unsqueeze(dim)).pow(2).sum(dim).clamp(eps)
            )
            return mean, std

        mean, std = _compute_statistics(x)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats
