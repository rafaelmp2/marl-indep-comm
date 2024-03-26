import torch
import numpy as np
from torch.distributions import Categorical


class Agents:
	def __init__(self, args):
		self.n_actions = args.n_actions
		self.n_agents = args.n_agents
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape

		# currently jsut support idql without param sharing and with or without comm
		if args.alg == 'idql':
			from algos.idql import IDQL
			self.policy = IDQL(args)
			print("IDQL initialized")
		else:
			raise Exception("No such algorithm!")

		self.args = args



	def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False, msg_all=None, msg_i=None):
		
		inputs = obs.copy()
		avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

		if self.args.last_action:
		    inputs = np.hstack((inputs, last_action))  # concatenates arrays column wise (horizontally)

		hidden_state = self.policy.eval_hidden

	
		inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
		avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

		# cuda
		if self.args.cuda:
			inputs = inputs.cuda(device=self.args.cuda_device)
			hidden_state = hidden_state.cuda(device=self.args.cuda_device)
			if self.args.with_comm:
				msg_all = msg_all.cuda(device=self.args.cuda_device)
				msg_i = msg_i.cuda(device=self.args.cuda_device)


		if self.args.with_comm:
			# reshape to make them [n_ep, n_ag, dim] to be comp with the training
			msg_all = msg_all.reshape(1, self.args.n_agents-1, -1)
			msg_i = msg_i.reshape(1, 1, -1)

			
		if self.args.with_comm:
			# if comm
			q_value, self.policy.eval_hidden = self.policy.eval_rnn(inputs, hidden_state, msg_all, agent_num, msg_i)
		else:
			q_value, self.policy.eval_hidden = self.policy.eval_rnn(inputs, hidden_state)

		
		if self.args.alg == 'coma':
			raise Exception("Not implemented")
			action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
		else:
			q_value[avail_actions == 0.0] = - float("inf")
			# epsilon greedy
			if np.random.uniform() < epsilon:
				action = np.random.choice(avail_actions_ind)  # picks an action from the available actions array
			else:
				action = torch.argmax(q_value)

		return action


	# this is to be used during execution; given obs of agent i gives messages for this agent; shouold be done for each agent
	def get_all_messages(self, obs, last_action):
		obs = torch.tensor(obs, dtype=torch.float32)
		last_action = torch.tensor(last_action, dtype=torch.float32)
		inputs = obs.reshape(1, -1)

		if self.args.cuda:
			inputs = inputs.cuda(device=self.args.cuda_device)

		msgs_agents = self.policy.commtest(inputs)

		return msgs_agents


	def _get_max_episode_len(self, batch):
		terminated = batch['terminated']
		episode_num = terminated.shape[0]  # number of episode batches inside this batch
		max_episode_len = 0
		for episode_idx in range(episode_num):
		    for transition_idx in range(self.args.episode_limit):
		        if terminated[episode_idx, transition_idx, 0] == 1:  
		            if transition_idx + 1 >= max_episode_len:
		                max_episode_len = transition_idx + 1
		            break
		return max_episode_len


	# this is to give the messages for this agent for training from the experience batch
	def get_msgs_for_train(self, batch, train_step, agent_id, epsilon=None):
		max_episode_len = self._get_max_episode_len(batch)  # inside batch there are several episode batches; as they may have different sizes, gets the bigger
		for key in batch.keys():
		    batch[key] = batch[key][:, :max_episode_len]  

		msgs_agent_i, msgs_agent_i_next = self.policy.generate_msgs_agent_i(batch, max_episode_len, agent_id)

		return msgs_agent_i, msgs_agent_i_next


	def train(self, batch, train_step, agent_id, all_msgs=None, all_msgs_next=None, epsilon=None): 

		max_episode_len = self._get_max_episode_len(batch)  # inside batch there are several episode batches; as they may have different sizes, gets the bigger
		for key in batch.keys():
		    batch[key] = batch[key][:, :max_episode_len]  

		self.policy.learn(batch, max_episode_len, train_step, agent_id, epsilon, all_msgs=all_msgs, all_msgs_next=all_msgs_next)

		# savind model
		if train_step > 0 and train_step % self.args.save_cycle == 0:
		    self.policy.save_model(train_step, agent_id)


	def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
		# inputs refers to q_value of all actions
		action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # sum of avail_actions
		# converts output of actor network into a prob dist with softmax
		prob = torch.nn.functional.softmax(inputs, dim=-1)

		# noise of epsilon
		prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
		prob[avail_actions == 0] = 0.0  # unavailable actions get 0 prob

		# note that after setting the unavaible actions prob to 0, the sum in prob is not 1, but no need to regularize because torch.distributions.categorical will be regularized
		# categorical is not used during training so the probability of the action performed during training needs to be regularized again

		if epsilon == 0 and evaluate:
			action = torch.argmax(prob)
		else:
			action = Categorical(prob).sample().long()

		return action
