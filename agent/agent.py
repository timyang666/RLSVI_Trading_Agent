import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque
from utils import MLP, DQNWithPrior
import torch
import torch.multiprocessing
import torch.nn as nn
import typing

has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if has_gpu else "cpu")
device = torch.device("cpu") # comment this if you want to use a GPU

class Agent:
	def __init__(self,state_size,prior_variance,noise_variance,num_ensemble,is_eval=False,model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

##############################################################
		self.discount = 0.99
		self.running_loss = 1.
		self.learning_rate = 5e-4
		self.feature_dim = state_size
		self.prior_variance = prior_variance #
		self.noise_variance = noise_variance #
		self.num_ensemble = num_ensemble # number of models in ensemble
		self.index = np.random.randint(self.num_ensemble) #
		hidden_dims = [50,50]
		self.dims = [self.feature_dim] + hidden_dims + [self.action_size]

		
		self.models = []
		if model_name=="":
			for i in range(self.num_ensemble):
				self.models.append(DQNWithPrior(self.dims,scale = np.sqrt(self.prior_variance)).to(device))
			self.models[i].initialize()
			self.target_nets = []
			for i in range(self.num_ensemble):
				self.target_nets.append(DQNWithPrior(self.dims,scale = np.sqrt(self.prior_variance)).to(device))
			self.target_freq = 10 #   target nn updated every target_freq episodes
			self.num_episodes = 0

			self.optimizer = []
			for i in range(self.num_ensemble):
			    self.optimizer.append(torch.optim.Adam(self.models[i].parameters(),lr=self.learning_rate))
		else:
			self.models =[]
			self.test_mode = True
			if self.prior_network:
				self.models.append(DQNWithPrior(self.dims))
			else:
				self.models.append(MLP(self.dims))
			self.models[0].load_state_dict(torch.load(test_model_path))
			self.models[0].eval()
			self.index = 0

		###############################################################

		# self.model = load_model("models/" + model_name) if is_eval else self._model()

	# def _model(self):
	# 	model = Sequential()
	# 	model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
	# 	model.add(Dense(units=32, activation="relu"))
	# 	model.add(Dense(units=8, activation="relu"))
	# 	model.add(Dense(self.action_size, activation="linear"))
	# 	model.compile(loss="mse", optimizer=Adam(lr=0.001))
	# 	return model

	def act(self, state):
		# if not self.is_eval and np.random.rand() <= self.epsilon:
		# 	return random.randrange(self.action_size)

		# options = self.model.predict(state)
		# return self.epsilon_boltzmann_action(options,self.epsilon)
		feature = state
		with torch.no_grad():
			if str(device)=="cpu":
				action_values = (self.models[self.index](torch.tensor(feature).float())).numpy()
				
			else:
				out = (self.models[self.index](torch.tensor(feature).float()).to(device))
				action_values = (out.to("cpu")).numpy()
			action = self.epsilon_boltzmann_action(action_values,self.epsilon)
		return action




	def expReplay(self, batch_size):
		loss_ensemble = 0

		for sample_num in range(self.num_ensemble):
			minibatch = random.sample(self.memory, batch_size)
			# minibatch = self.memory.sample(batch_size=batch_size)

			feature_batch = torch.zeros(batch_size, self.feature_dim, device=device)
			action_batch = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
			reward_batch = torch.zeros(batch_size, 1, device=device)
			perturb_batch = torch.zeros(batch_size,self.num_ensemble, device=device)
			non_terminal_idxs = []
			next_feature_batch = []

			for i, d in enumerate(minibatch):
				s, a, r, s_next, perturb, _ = d
				feature_batch[i] = torch.from_numpy(s)
				action_batch[i] = torch.tensor(a, dtype=torch.long)
				reward_batch[i] = r
				perturb_batch[i] = torch.from_numpy(perturb)
				if s_next is True:
					non_terminal_idxs.append(i)
					next_feature_batch.append(s_next)
			model_estimates = ( self.models[sample_num](feature_batch)).gather(1, action_batch).float()

			future_values = torch.zeros(batch_size, device=device)
			if non_terminal_idxs != []:
				next_feature_batch = torch.tensor(next_feature_batch,dtype=torch.float, device=device)
				future_values[non_terminal_idxs] = (self.target_nets[sample_num](next_feature_batch)).max(1)[0].detach()
			future_values = future_values.unsqueeze(1)
			temp = perturb_batch[:,sample_num].unsqueeze(1)
			target_values = reward_batch + self.discount * future_values + perturb_batch[:,sample_num].unsqueeze(1)

			assert(model_estimates.shape==target_values.shape)

			loss = nn.functional.mse_loss(model_estimates, target_values)

			self.optimizer[sample_num].zero_grad()
			loss.backward()
			self.optimizer[sample_num].step()
			loss_ensemble += loss.item()
		self.running_loss = 0.99 * self.running_loss + 0.01 * loss_ensemble

		self.num_episodes += 1

		self.index = np.random.randint(self.num_ensemble)

		# mini_batch = []
		# l = len(self.memory)
		# for i in range(l - batch_size + 1, l):
		# 	mini_batch.append(self.memory[i])

		# for state, action, reward, next_state, done in mini_batch:
		# 	target = reward
		# 	if not done:
		# 		target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

		# 	target_f = self.model.predict(state)
		# 	target_f[0][action] = target
		# 	self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 

	def epsilon_boltzmann_action(self, action_values, epsilon):
		action_values = action_values - max(action_values)
		action_probabilities = np.exp(action_values / (np.exp(1)*epsilon))
		action_probabilities /= np.sum(action_probabilities)
		return np.random.choice(self.action_size, 1, p=action_probabilities[0])
    
	def save(self,e,path=None):
		if path is None:
			path = "models/model_ep" + str(e)+'.pt'
		torch.save(self.models[self.index].state_dict(), path)
