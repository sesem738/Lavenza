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

class GTrXL_TS_Layer(nn.Module):
    """Create a single transformer block. STT may stack multiple blocks.

    Args:
        num_heads: Number of heads to use for MultiHeadAttention.
        embed_size: The dimensionality of the layer.
        history_len: The maximum number of observations to take in.
        dropout: Dropout percentage.
    """

    def __init__(
        self,
        num_heads: int,
        embed_size: int,
        history_len: int,
        dropout: float
    ):
        super().__init__()
        self.layernorm1T = nn.LayerNorm(embed_size)
        self.layernorm2T = nn.LayerNorm(embed_size)

        self.attentionT = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffnT = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )
        self.attn_gateT = GRUGate(embed_size)
        self.mlp_gateT = GRUGate(embed_size)

        self.transformT2S = nn.Linear(configT.block_size, configS.n_embd)
        self.transformT2S_q = nn.Linear(configT.block_size_state, configS.n_embd)
        self.transformS2T = nn.Linear(configS.n_embd, configT.block_size_state)

        self.layernorm1S = nn.LayerNorm(embed_size)
        self.layernorm2S = nn.LayerNorm(embed_size)

        self.attentionS = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffnS = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )
        self.attn_gateS = GRUGate(embed_size)
        self.mlp_gateS = GRUGate(embed_size)
        
        # Just storage for attention weights for visualization
        self.alphaT = None
        self.alphaS = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attentionT, self.alphaT = self.attention(
            x,
            x,
            x,
            attn_mask=self.attn_mask[: x.size(1), : x.size(1)],
            average_attn_weights=True,  # Only affects self.alpha for visualizations
        )
        # Skip connection then LayerNorm
        x = self.attn_gate(x, F.relu(attention))
        x = self.layernorm1(x)
        ffn = self.ffn(x)
        # Skip connection then LayerNorm
        x = self.mlp_gate(x, F.relu(ffn))
        x = self.layernorm2(x)
        return x

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

class STT():
    def __init__(self):
        pass
    
    def forward(self):
        pass