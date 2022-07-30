"""
1) Init Transformer
    a) Embed action into larger action space
2) Function for Position Encoding
3) Function for Split embedding heads for Spatial Attention and Temporal Attention
4) Function for Spatial Attention
5) Function for Temporal Attention
6) Function for MLP
7) Function for GRU Gate from GTrXL 
"""

import os
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class PositionEncoding(nn.Module):
    def __init__():
        pass
    def forward():
        pass

class Split():
    def time_split(self):
        pass
    def space_split(self):
        pass

class GTrXL_Time_Block(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class GTrXL_Space_Block(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass


class STT(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass


class GRUGate(nn.Module):
    """GRU Gating from GTrXL (https://arxiv.org/pdf/1910.06764.pdf)"""

    def __init__(self, **kwargs):
        super().__init__()

        embed_size = kwargs["embed_size"]

        self.w_r = nn.Linear(embed_size, embed_size, bias=False)
        self.u_r = nn.Linear(embed_size, embed_size, bias=False)
        self.w_z = nn.Linear(embed_size, embed_size)
        self.u_z = nn.Linear(embed_size, embed_size, bias=False)
        self.w_g = nn.Linear(embed_size, embed_size, bias=False)
        self.u_g = nn.Linear(embed_size, embed_size, bias=False)

        self.init_bias()

    def init_bias(self):
        with torch.no_grad():
            self.w_z.bias.fill_(-2)  # This is the value set by GTrXL paper

    def forward(self, x, y):
        z = torch.sigmoid(self.w_z(y) + self.u_z(x))
        r = torch.sigmoid(self.w_r(y) + self.u_r(x))
        h = torch.tanh(self.w_g(y) + self.u_g(r * x))

        return (1.0 - z) * x + z * h