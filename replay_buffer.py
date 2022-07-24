import numpy as np
import torch
import time

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=self.device)
		self.action = torch.zeros((max_size, action_dim), dtype=torch.float32, device=self.device)
		self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=self.device)
		self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)
		self.not_done = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)



	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
		self.action[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
		self.next_state[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
		self.reward[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
		self.not_done[self.ptr] = 1. - torch.as_tensor(done, dtype=torch.float32, device=self.device)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind]
		)
	
	def prior_samples(self, batch_size, his_len):
		ind = np.random.randint(his_len, len(self), size=batch_size)
		# History
		obs = torch.zeros([batch_size, his_len, self.state_dim], dtype=torch.float32, device=self.device)
		actions = torch.zeros([batch_size, his_len, self.action_dim], dtype=torch.float32, device=self.device)
		next_obs = torch.zeros([batch_size, his_len, self.state_dim], dtype=torch.float32, device=self.device)
		rewards = torch.zeros([batch_size, his_len, 1], dtype=torch.float32, device=self.device)
		not_done = torch.zeros([batch_size, his_len, 1], dtype=torch.float32, device=self.device)
		# his_obs_len = his_len * np.ones(batch_size)

		for i, id in enumerate(ind):
			
			start_id, id = self._get_valid_sequence(id, his_len)
			obs[i] = self.state[start_id:id]
			actions[i] = self.action[start_id:id]
			next_obs[i] = self.next_state[start_id:id]
			rewards[i] = self.reward[start_id:id]
			not_done[i] = self.not_done[start_id:id]

		return (obs, actions, next_obs, rewards, not_done)

	def _get_valid_sequence(self, id:int, his_len:int):
		"""Checks if the sequence of transitions is valid, i.e., has and terminal transitions.
		Returns id's for a valid sequence"""
		start_id = id-his_len
		
		not_valid = True
		while not_valid:

			# Check sequence for termination
			if len(np.where(self.not_done[start_id:id] == 1)[0]) != 0:
				temp = start_id + (np.where(self.not_done[start_id:id] == 1)[0][-1]) + 1
				if (temp + his_len) >= len(self):
					start_id = np.random.randint(0, len(self)-his_len,1)[0]
				else:
					start_id = temp
				id = start_id + his_len
			
			if (len(np.where(self.not_done[start_id:id] == 1)[0]) == 0):
				not_valid = False
		
		return start_id, id
		
	
	def __len__(self):
		return self.size


if __name__=="__main__":
	replay = ReplayBuffer(288, 1)
	print('------------')

	print(replay.state.element_size()*replay.state.nelement())