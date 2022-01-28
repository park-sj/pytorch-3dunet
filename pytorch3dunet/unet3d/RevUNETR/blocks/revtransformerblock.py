# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:21:30 2021

@author: user
"""

from typing import Optional, Sequence, Tuple, Union
import revtorch.revtorch as rv
import torch.nn as nn
import numpy as np
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.selfattention import SABlock
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm


class ReversibleTransformerBlock(nn.Module):

    def __init__(self, hidden_size: int,
                 mlp_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.0) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        fBlock = nn.Sequential(
            nn.LayerNorm(hidden_size),
            MLPBlock(hidden_size, mlp_dim, dropout_rate)
            )
        gBlock = nn.Sequential(
            nn.LayerNorm(hidden_size),
            SABlock(hidden_size, num_heads, dropout_rate)
            )
        self.block = rv.ReversibleBlock(fBlock, gBlock)

    def forward(self, x):
        x = self.block(x)
        return x

def makeReversibleTransformerBlock(hidden_size: int,
                                   mlp_dim: int,
                                   num_heads: int,
                                   dropout_rate: float = 0.0):
    if not (0 <= dropout_rate <= 1):
        raise ValueError("dropout_rate should be between 0 and 1.")

    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size should be divisible by num_heads.")
        
    fBlock = nn.Sequential(
            nn.LayerNorm(hidden_size),
            MLPBlock(hidden_size, mlp_dim, dropout_rate)
            )
    gBlock = nn.Sequential(
            nn.LayerNorm(hidden_size),
            SABlock(hidden_size, num_heads, dropout_rate)
            )
    return rv.ReversibleBlock(fBlock, gBlock)