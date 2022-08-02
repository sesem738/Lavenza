import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # add positional embedding directly to the state/state+action embedding
        # note: dim is a bit different from NLP embeddings
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
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

class Split():
	
	def time_split(self):
		pass

	def space_split(self):
		pass