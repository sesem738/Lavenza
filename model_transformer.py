import os
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from einops import repeat,rearrange
from copy import deepcopy

'''Attention Block'''
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1   #不同层dropout概率不同  嵌入层
    resid_pdrop = 0.1  #跳层
    attn_pdrop = 0.1   #注意力层
    mask = True

    def __init__(self,block_size, **kwargs):
        self.block_size = block_size    #输入块大小 128
        for k,v in kwargs.items():
            setattr(self, k, v)   #setattr() 函数对应函数 getattr()，用于设置属性值，
class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768
    init_gru_gate_bias: float = 2.0

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.if_mask = config.mask
        if self.if_mask:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))   #torch.tril切割成下三角矩阵
                                        .view(1, 1, config.block_size, config.block_size))             #[1,1,128,128]的下三角矩阵；  register_buffer不作为模型参数，就是在内存中定义一个常量，同时，模型保存和加载的时候可以写入和读出
        self.n_head = config.n_head

    def forward(self, q,k,v, attn_bias=None):
        B, T_k, C = k.size()   #torch.Size([batch_size, time_seq, n_embd])  batch_size一直放在最前面   128个字符 每个都映射成512向量
        B, T_q, C = q.size()
        B, T_v, C = v.size()  
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(k).view(B, T_k, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T_k, hs)
        q = self.query(q).view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T_q, hs)
        v = self.value(v).view(B, T_v, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T_v, hs)

        # causal self-attention; Self-attend: (B, nh, T_q, hs) x (B, nh, hs, T_k) -> (B, nh, T_q, T_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))    # transpose(-2, -1)转置最后两位
        if self.if_mask:
            att_scores = att.masked_fill(self.mask[:, :, -T_q:, :T_k] == 0, float('-inf'))    # b[;2]=b[0],b[1]     att下三角有值 上三角全为-inf
        else:
            att_scores = att
        '''
        a=torch.tensor([1,0,2,3])
        a.masked_fill(mask = torch.ByteTensor([1,1,0,0]), value=torch.tensor(-1e9))
        tensor([-1.0000e+09, -1.0000e+09,  2.0000e+00,  3.0000e+00])
        '''
        if attn_bias is not None:
            # attn_bias = attn_bias.view(B, T_q, T_v, self.n_head, C // self.n_head)
            # attn_bias = rearrange(attn_bias,'b q v h s -> b h q v s').mean(-1)
            att_scores = att_scores + attn_bias
        att_probs = F.softmax(att_scores, dim=-1)
        att_probs = self.attn_drop(att_probs)
        y = att_probs @ v # (B, nh, T_q, T_k) x (B, nh, T_v, hs) -> (B, nh, T_q, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side
                                                         # 调用view之前最好先contiguous,x.contiguous().view()  view需要tensor的内存是整块的

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att_scores

from ray.rllib.utils.typing import ModelConfigDict, TensorType, List

class GRUGate(nn.Module):
    """Implements a gated recurrent unit for use in AttentionNet"""

    def __init__(self, dim: int, init_bias: int = 0., **kwargs):
        """
        input_shape (torch.Tensor): dimension of the input
        init_bias (int): Bias added to every input to stabilize training
        """
        super().__init__(**kwargs)
        # Xavier initialization of torch tensors
        self._w_r = nn.Parameter(torch.zeros(dim, dim))
        self._w_z = nn.Parameter(torch.zeros(dim, dim))
        self._w_h = nn.Parameter(torch.zeros(dim, dim))
        nn.init.xavier_uniform_(self._w_r)
        nn.init.xavier_uniform_(self._w_z)
        nn.init.xavier_uniform_(self._w_h)
        self.register_parameter("_w_r", self._w_r)
        self.register_parameter("_w_z", self._w_z)
        self.register_parameter("_w_h", self._w_h)

        self._u_r = nn.Parameter(torch.zeros(dim, dim))
        self._u_z = nn.Parameter(torch.zeros(dim, dim))
        self._u_h = nn.Parameter(torch.zeros(dim, dim))
        nn.init.xavier_uniform_(self._u_r)
        nn.init.xavier_uniform_(self._u_z)
        nn.init.xavier_uniform_(self._u_h)
        self.register_parameter("_u_r", self._u_r)
        self.register_parameter("_u_z", self._u_z)
        self.register_parameter("_u_h", self._u_h)

        self._bias_z = nn.Parameter(torch.zeros(dim, ).fill_(init_bias))
        self.register_parameter("_bias_z", self._bias_z)

    def forward(self, inputs: TensorType, **kwargs) -> TensorType:
        # Pass in internal state first.
        h, X = inputs

        r = torch.tensordot(X, self._w_r, dims=1) + \
            torch.tensordot(h, self._u_r, dims=1)
        r = torch.sigmoid(r)

        z = torch.tensordot(X, self._w_z, dims=1) + \
            torch.tensordot(h, self._u_z, dims=1) - self._bias_z
        z = torch.sigmoid(z)

        h_next = torch.tensordot(X, self._w_h, dims=1) + \
            torch.tensordot((h * r), self._u_h, dims=1)
        h_next = torch.tanh(h_next)

        return (1 - z) * h + z * h_next


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, q,k,v, attn_bias=None):
        
        x = self.mlp(self.ln2(q))
        return x, None

class BlockGTrXL(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 2 * config.n_embd),
            nn.GELU(),
            nn.Linear(2 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.gate1 = GRUGate(config.n_embd, config.init_gru_gate_bias)
        self.gate2 = GRUGate(config.n_embd, config.init_gru_gate_bias)

    def forward(self, q,k,v, attn_bias=None):
        att_output, layer_att = self.attn(self.ln1(q),self.ln1(k),self.ln1(v),attn_bias)
        x = self.gate1((q,att_output))
        x = self.gate2((x,self.mlp(self.ln2(x))))
        return x, layer_att

class BlockGTrXLTS(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        configS = deepcopy(config)
        configT = deepcopy(config)
        configS.n_embd = config.n_embdS
        # configS.block_size = configT.n_embd

        self.ln1T = nn.LayerNorm(configT.n_embd)
        self.ln2T = nn.LayerNorm(configT.n_embd)
        self.attnT = CausalSelfAttention(configT)
        self.mlpT = nn.Sequential(
            nn.Linear(configT.n_embd, 2 * configT.n_embd),
            nn.GELU(),
            nn.Linear(2 * configT.n_embd, configT.n_embd),
            nn.Dropout(configT.resid_pdrop),
        )
        self.gate1T = GRUGate(configT.n_embd, configT.init_gru_gate_bias)
        self.gate2T = GRUGate(configT.n_embd, configT.init_gru_gate_bias)

        self.transformT2S = nn.Linear(configT.block_size, configS.n_embd)
        self.transformT2S_q = nn.Linear(configT.block_size_state, configS.n_embd)
        self.transformS2T = nn.Linear(configS.n_embd, configT.block_size_state)

        self.ln1S = nn.LayerNorm(configS.n_embd)
        self.ln2S = nn.LayerNorm(configS.n_embd)
        self.attnS = CausalSelfAttention(configS)
        self.mlpS = nn.Sequential(
            nn.Linear(configS.n_embd, 2 * configS.n_embd),
            nn.GELU(),
            nn.Linear(2 * configS.n_embd, configS.n_embd),
            nn.Dropout(configS.resid_pdrop),
        )
        self.gate1S = GRUGate(configS.n_embd, configS.init_gru_gate_bias)
        self.gate2S = GRUGate(configS.n_embd, configS.init_gru_gate_bias)

    def forward(self, q,k,v, attn_bias=None):
        att_outputT, layer_attT = self.attnT(self.ln1T(q),self.ln1T(k),self.ln1T(v),attn_bias[0])
        q = self.gate1T((q,att_outputT))
        q = self.gate2T((q,self.mlpT(self.ln2T(q))))

        q = self.transformT2S_q(q.transpose(1, 2))
        k = self.transformT2S(k.transpose(1, 2))
        v = self.transformT2S(v.transpose(1, 2))
        att_biasT = attn_bias[1]
        att_biasT = self.transformT2S_q(att_biasT.transpose(-1, -2)).transpose(-1, -2)

        att_outputS, layer_attS = self.attnS(self.ln1S(q),self.ln1S(k),self.ln1S(v),att_biasT)
        q = self.gate1S((q,att_outputS))
        q = self.gate2S((q,self.mlpS(self.ln2S(q))))
        q = self.transformS2T(q)
        return q.transpose(1, 2), layer_attT

from ray.rllib.models.torch.modules import GRUGate, \
    RelativeMultiHeadAttention, SkipConnection

class BlockSeq(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.use_TS = config.use_TS
        if config.use_GTrXL is True:
            if self.use_TS == True:
                self.layer = nn.Sequential(*[BlockGTrXLTS(config) for _ in range(config.n_layer)])
            else:
                self.layer = nn.Sequential(*[BlockGTrXL(config) for _ in range(2*config.n_layer)])
        else:
            self.layer = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

    def forward(self,  q,k,v, attn_bias=None):
        if self.use_TS == False:
            attn_bias = attn_bias[0]

        for _, layer_module in enumerate(self.layer):
            q, layer_att = layer_module( q,k,v, attn_bias)
            del layer_att

        return q


class STT(nn.Module):
    """  the Spatiotemporal Transformer """   

    def __init__(self, config):
        super().__init__()

        self.embd = config.n_embd
        self.n_head = config.n_head
        self.tok_emb = nn.Linear(config.state_dim, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.att_bias_encoderS = nn.Linear(config.state_dim,1)
        self.att_bias_encoderT = nn.Linear(config.block_size,1)
        # transformer
        self.blocks = BlockSeq(config)
        self.block_size = config.block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, q,k,att_bias=None):
        '''
        BatchSize, TimeSequence, StateEmbedding = input.size()
        Output with the same size
        BatchSize,  T_q, T_k, 1 = att_bias.size()
        '''
        v = k
        b, t_q, c = q.size()
        b, t_k, c = k.size()
        assert t_k <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings_q = self.pos_emb[:, :t_q, :] # each position maps to a (learnable) vector
        position_embeddings_kv = self.pos_emb[:, :t_k, :] # each position maps to a (learnable) vector
        if att_bias is not None:
            att_bias = repeat(att_bias,'q k c -> b q k c',b=b)
            att_bias = repeat(att_bias,'b q k c -> b h q k c',h=self.n_head)
            att_biasS = self.att_bias_encoderS(att_bias.to(q.device)).squeeze(-1)
            att_biasT = self.att_bias_encoderT(rearrange(att_bias,'b h q k c -> b h q c k').to(q.device)).squeeze(-1)
            att_biasT = self.tok_emb(att_biasT)
        else:
            att_biasS = None
            att_biasT = None
        q = self.drop(self.tok_emb(q) + position_embeddings_q)
        k = self.drop(self.tok_emb(k) + position_embeddings_kv)
        v = self.drop(self.tok_emb(v) + position_embeddings_kv)
        x = self.blocks(q,k,v,[att_biasS,att_biasT])
        x = self.ln_f(x)

        return x.mean(1).squeeze(1)


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
