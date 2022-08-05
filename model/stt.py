import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import PositionalEncoding

class STT(nn.Module):
	def __init__(
		self,
		obs_dim: int, 
		num_actions: int,
		dmodel: int,
		heads_time: int,
		heads_space: int,
		num_layers: int,
		embed_size: int,
		his_len: int,
		dropout: float = 0.2,
		):

		self.obs_dim = obs_dim
		self.action_dim = num_actions
		# Embed Action Space if Critic
		self.pos_encoder = PositionalEncoding(dmodel, dropout)
		self.dropout = nn.Dropout(dropout)
		self.laynorm_ffn = nn.LayerNorm(embed_size)
		self.space_att_bias = nn.Linear(embed_size,1)
		self.time_att_bias = nn.Linear(embed_size,1)
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
			if isinstance(module, nn.Linear) and module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, (nn.MultiheadAttention)):
			module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
			module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
			module.in_proj_bias.data.zero_()
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
	
	def forward(self):
		pass

