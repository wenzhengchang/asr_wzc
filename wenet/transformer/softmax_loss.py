#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/11/16 11:10 PM
# @Author : Zhengchang
# @Email : 819192552@qq.com
# @File : softmax_loss.py
# @WhatToDO :
import torch
from torch import nn
import torch.nn.functional as F


class SoftmaxLoss(nn.Module):

    def __init__(self):
        self.criterion = torch.nn.Softmax

    def forward(self, x:torch.Tensor, targets:torch.Tensor)->torch.Tensor:
        """
        Arguments
        ---------
        x : torch.Tensor
            Network output tensor, of shape
            [batch, 1, outdim].
        targets : torch.Tensor
            Target tensor, of shape [batch, 1].

        Returns
        -------
        loss: torch.Tensor
            Loss for current examples.
        """
        # outputs = outputs.squeeze(1)
        # targets = targets.squeeze(1)

        targets = F.one_hot(targets.long(), x.shape[1]).float()
        loss = self.criterion(x, targets)
        return loss
