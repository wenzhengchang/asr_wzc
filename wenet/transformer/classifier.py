#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/11/16 5:48 PM
# @Author : Zhengchang
# @Email : 819192552@qq.com
# @File : classifier.py
# @WhatToDO :
import torch
from torch import nn


# 分类器，把512的维度分到9个地方口音
class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.

    Example
    -------
    >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outpupts = outputs.unsqueeze(1)
    >>> cos = classify(outputs)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
    """

    def __init__(
            self,
            input_size,
            output_size,
    ):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.
        """

        y = self.linear(x)
        # print("classifier出来的维度",y.shape)
        return y
