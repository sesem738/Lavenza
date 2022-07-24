import torch
import torch.nn as nn
import torch.nn.functional as F
from model_transformer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	'''
	Input and output size: (BatchSize, TimeSequence, StateEmbedding)
	InitDict={
		'state_dim': int
		'action_dim' : int
		'attembedding' : int
		'atthead' : int
		'block_size': int
		'if_load_GRD' : bool
	}
	'''
	def __init__(self, InitDict):
		super().__init__()
		state_dim = InitDict['state_dim']
		mid_dim = InitDict['embeddingT']
		action_dim = InitDict['action_dim']
		config = GPTConfig(block_size=InitDict['block_size'], 
							block_size_state=InitDict['block_size_state'], 
							state_dim=state_dim,
							n_layer=InitDict['attlayer'],
							n_head=InitDict['atthead'],
							mask=False,
							n_embd=InitDict['embeddingT'],
							n_embdS=InitDict['embeddingS'],
							init_gru_gate_bias=InitDict['init_gru_gate_bias'],
							use_TS=InitDict['use_TS'],
							use_GTrXL=InitDict['use_GTrXL'])

		self.netDynamic = STT(config)
		self.netaction = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.RELU(),
									nn.Linear(mid_dim, action_dim),nn.Tanh())
		

	def forward(self, state,past_state,att_bias):
		if state.ndim == 1:
			state = state.unsqueeze(0)
		state = state.reshape(state.shape[0],1,-1)
		return self.netaction(self.netDynamic(state,past_state,att_bias))  # action







class CriticAdvAtt(nn.Module):
    def __init__(self, InitDict):
        '''
        Input and output size: (BatchSize, TimeSequence, StateEmbedding)
        InitDict={
            'state_dim': int
            'action_dim' : int
            'attembedding' : int
            'atthead' : int
            'block_size': int
            'if_load_GRD' : bool
        }
        '''
        super().__init__()
        state_dim = InitDict['state_dim']
        mid_dim = InitDict['embeddingT']
        action_dim = InitDict['action_dim']
        config = GPTConfig(block_size=InitDict['block_size'], 
                            block_size_state=InitDict['block_size_state'], 
                            state_dim=state_dim,
                            n_layer=InitDict['attlayer'],
                            n_head=InitDict['atthead'],
                            mask=False,
                            n_embd=InitDict['embeddingT'],
                            n_embdS=InitDict['embeddingS'],
                            init_gru_gate_bias=InitDict['init_gru_gate_bias'],
                            use_TS=InitDict['use_TS'],
                            use_GTrXL=InitDict['use_GTrXL'])
        self.netDynamic = STT(config)
        self.net = nn.Sequential(
                                    nn.Linear(mid_dim, mid_dim), nn.RELU(),
                                    nn.Linear(mid_dim, 1), 
                                )


        layer_norm(self.net[-1], std=0.5)  # output layer for Q value

    def forward(self, state,past_state,att_bias):
        return self.net(self.netDynamic(state,past_state,att_bias))