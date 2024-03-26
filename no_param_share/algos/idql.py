import torch
import os
from network.base_net import RNN
import numpy as np


from network.simple_comm_net import Comm_net 


class IDQL:
	def __init__(self, args):
		self.n_actions = args.n_actions
		self.n_agents = args.n_agents
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape
		input_shape = self.obs_shape

		if args.with_comm:
			input_comm_shape = self.obs_shape

		print("obs shape: ", self.obs_shape)

		# input dimension for rnn according to the params
		if args.last_action:
		    input_shape += self.n_actions

		if args.with_comm:
			input_shape += (args.n_agents) * args.final_msg_dim #(args.n_agents - 1) * args.final_msg_dim
			print("obs shape with comm: ", input_shape)
		
		self.eval_rnn = RNN(input_shape, args)  # each agent picks a net of actions
		self.target_rnn = RNN(input_shape, args)

		self.args = args

		if args.with_comm:
			self.commtest = Comm_net(input_comm_shape, args)
			self.target_commtest = Comm_net(input_comm_shape, args)

		# cuda
		if self.args.cuda:
			self.eval_rnn.cuda(device=self.args.cuda_device)
			self.target_rnn.cuda(device=self.args.cuda_device)
			if args.with_comm:
				self.commtest.cuda(device=self.args.cuda_device)
				self.target_commtest.cuda(device=self.args.cuda_device)

		self.model_dir = args.model_dir + '/' + args.alg

		if self.args.load_model:
		    if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
		        path_rnn = self.model_dir + '/rnn_net_params.pkl'
		        path_vdn = self.model_dir + '/vdn_net_params.pkl'
		        print('Successfully load the model: {} and {}'.format(path_rnn, path_vdn))
		    else:
		    	raise Exception("No such model!")

		# make parameters of target and eval the same
		self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
		if self.args.with_comm:
			self.target_commtest.load_state_dict(self.commtest.state_dict())

		if args.with_comm:
			self.eval_parameters = list(self.eval_rnn.parameters()) + list(self.commtest.parameters())
		else:
			self.eval_parameters = list(self.eval_rnn.parameters()) 	

		if args.optimizer == "RMS":
		    self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)


		# during learning one should keep an eval_hidden and a target_hidden for each agent of each episode
		self.eval_hidden = None
		self.target_hidden = None

		print("IDQL algorithm initialized")


	def learn(self, batch, max_episode_len, train_step, agent_id, epsilon=None, all_msgs=None, all_msgs_next=None):  

		'''
			batch: batch with episode batches from before training the model
			max_episode_len: len of the longest episode batch in batch
			train_step: it is used to control and update the params of the target network
			agent_id: id of agent i

			------------------------------------------------------------------------------

			the extracted data is 4D, with meanings 1-> n_episodes, 2-> n_transitions in the episode, 
			3-> data of multiple agents, 4-> obs dimensions
			hidden_state is related to the previous experience so one cant randomly extract
			experience to learn, so multiple episodes are extracted at a time and then given to the
			nn one at a time   
		'''

		episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch
		self.init_hidden(episode_num)

		#convert data in batch to tensor
		for key in batch.keys():  
		    if key == 'actions':
		        batch[key] = torch.tensor(batch[key], dtype=torch.long)
		    else:
		        batch[key] = torch.tensor(batch[key], dtype=torch.float32)

		obs, obs_next, actions, reward, avail_actions, avail_actions_next, terminated = batch['obs'][:, :, agent_id], \
		                          	batch['obs_next'][:, :, agent_id], batch['actions'][:, :, agent_id], batch['reward'], \
		                            batch['avail_actions'][:, :, agent_id], batch['avail_actions_next'][:, :, agent_id], batch['terminated']

		# used to set the td error of the filled experiments to 0, not to affect learning
		mask = 1 - batch["padded"].float()  

		state = batch['state']

		# cuda
		if self.args.cuda:
			obs = obs.cuda(device=self.args.cuda_device)
			obs_next = obs_next.cuda(device=self.args.cuda_device)
			actions = actions.cuda(device=self.args.cuda_device)
			reward = reward.cuda(device=self.args.cuda_device)
			mask = mask.cuda(device=self.args.cuda_device)
			terminated = terminated.cuda(device=self.args.cuda_device)

		# gets q value corresponding to each agent, dimensions are (episode_number, max_episode_len, n_agents, n_actions)
		q_evals, q_targets = self.get_q_values(batch, max_episode_len, agent_id, all_msgs, all_msgs_next)

		# get q value corresponding to each agents action and remove last dim
		q_evals = torch.gather(q_evals, dim=2, index=actions)

		q_targets[avail_actions_next == 0.0] = - 9999999
		q_targets = q_targets.max(dim=2, keepdim=True)[0]

		targets = reward + self.args.gamma * q_targets * (1 - terminated)

		td_error = targets.detach() - q_evals
		masked_td_error = mask * td_error  

		# there are still useless experiments, so the avg is according the number of real experiments
		loss = (masked_td_error ** 2).sum() / mask.sum()

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
		self.optimizer.step()

		# update target networks
		if train_step > 0 and train_step % self.args.target_update_cycle == 0:
			self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
			if self.args.with_comm:
				self.target_commtest.load_state_dict(self.commtest.state_dict())


	def generate_msgs_agent_i(self, batch, max_episode_len, agent_id):
		#convert data in batch to tensor
		for key in batch.keys():  
		    if key == 'actions':
		        batch[key] = torch.tensor(batch[key], dtype=torch.long)
		    else:
		        batch[key] = torch.tensor(batch[key], dtype=torch.float32)

		episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch
		q_evals, q_targets = [], []
		all_msgs_list, all_msgs_next_list = [], []
		for transition_idx in range(max_episode_len):
			all_msgs, all_msgs_next = self.get_msgs(batch, transition_idx, agent_id)  # add last action and agent_id to the obs

			# cuda
			if self.args.cuda:
				if self.args.with_comm:
					all_msgs = all_msgs.cuda(device=self.args.cuda_device)
					all_msgs_next = all_msgs_next.cuda(device=self.args.cuda_device)

			all_msgs_list.append(all_msgs)
			all_msgs_next_list.append(all_msgs_next)

		'''
		q_eval and q_target are lists containing max_episode_len arrays with dimensions (episode_number, n_agents, n_actions)
		convert the lists into arrays of (episode_number, max_episode_len, n_agents, n_actions)
		'''

		all_msgs_list = torch.stack(all_msgs_list, dim=1)
		all_msgs_next_list = torch.stack(all_msgs_next_list, dim=1)


		return all_msgs_list, all_msgs_next_list


	def get_msgs(self, batch, transition_idx, agent_id):
		obs, obs_next, actions_onehot = batch['obs'][:, transition_idx, agent_id], \
		                          batch['obs_next'][:, transition_idx, agent_id], batch['actions_onehot'][:]
		episode_num = obs.shape[0]
		inputs, inputs_next = [], []
		inputs.append(obs)
		inputs_next.append(obs_next)

		all_msgs, all_msgs_next = None, None

		# NOTE NOT USING AGENT ID NOW
		if self.args.with_comm:
			### comm messages
			inputs_msg = torch.cat([x for x in inputs], dim=-1)
			inputs_msg_next = torch.cat([x for x in inputs_next], dim=-1)
			if self.args.cuda:
				inputs_msg = inputs_msg.cuda(device=self.args.cuda_device)
				inputs_msg_next = inputs_msg_next.cuda(device=self.args.cuda_device)
				
			all_msgs = self.commtest(inputs_msg)
			all_msgs_next = self.target_commtest(inputs_msg_next)

		return all_msgs, all_msgs_next


	def get_q_values(self, batch, max_episode_len, agent_id, all_msgs, all_msgs_next):
		episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch
		q_evals, q_targets = [], []

		# if doing communication get the messages
		if self.args.with_comm:
			all_msgs_i = all_msgs[agent_id] 
			all_msgs_i_next = all_msgs_next[agent_id] 
			
			others_msgs = [all_msgs[i]for i in range(self.n_agents)if i!=agent_id]
			others_msgs_next = [all_msgs_next[i]for i in range(self.n_agents)if i!=agent_id]

			others_msgs = torch.stack(others_msgs, dim=1)
			others_msgs_next = torch.stack(others_msgs_next, dim=1)
		

		for transition_idx in range(max_episode_len):
			inputs, inputs_next = self._get_inputs(batch, transition_idx, agent_id)  # add last action and agent_id to the obs
			
			if self.args.with_comm:
				# msgs from other agents
				others_msgs_tidx = others_msgs[:, :, transition_idx, :]  # select time step, all_msgs contains from all of them  [n_ep, n_a, ts, dim]
				others_msgs_next_tidx = others_msgs_next[:, :, transition_idx, :]  # select time step, all_msgs contains from all of them  [n_ep, n_a, ts, dim]
				
				# msgs from this agent
				msg_agent_tidx = all_msgs_i[:, transition_idx, :]
				msg_agent_next_tidx = all_msgs_i_next[:, transition_idx, :]

			# cuda
			if self.args.cuda:
				inputs = inputs.cuda(device=self.args.cuda_device)
				inputs_next = inputs_next.cuda(device=self.args.cuda_device)
				self.eval_hidden = self.eval_hidden.cuda(device=self.args.cuda_device)
				self.target_hidden = self.target_hidden.cuda(device=self.args.cuda_device)
				if self.args.with_comm:
					others_msgs_tidx = others_msgs_tidx.cuda(device=self.args.cuda_device)
					others_msgs_next_tidx = others_msgs_next_tidx.cuda(device=self.args.cuda_device)
					msg_agent_tidx = msg_agent_tidx.cuda(device=self.args.cuda_device)
					msg_agent_next_tidx = msg_agent_next_tidx.cuda(device=self.args.cuda_device)


			if self.args.with_comm:
				# messages from the others too must go here all_msgs
				q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden, msgs=others_msgs_tidx, agent_num=agent_id, msg_i=msg_agent_tidx)  
				q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden, msgs=others_msgs_next_tidx, agent_num=agent_id, msg_i=msg_agent_next_tidx)
			else:
				q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  
				q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

			# Change the q_eval dimension back to (8, 5(n_agents), n_actions)
			q_eval = q_eval.view(episode_num, -1)
			q_target = q_target.view(episode_num, -1)
			q_evals.append(q_eval)
			q_targets.append(q_target)

		'''
		q_eval and q_target are lists containing max_episode_len arrays with dimensions (episode_number, n_agents, n_actions)
		convert the lists into arrays of (episode_number, max_episode_len, n_agents, n_actions)
		'''

		q_evals = torch.stack(q_evals, dim=1)
		q_targets = torch.stack(q_targets, dim=1)
		return q_evals, q_targets


	def _get_inputs(self, batch, transition_idx, agent_id):
		obs, obs_next, actions_onehot = batch['obs'][:, transition_idx, agent_id], \
		                          batch['obs_next'][:, transition_idx, agent_id], batch['actions_onehot'][:]
		episode_num = obs.shape[0]
		inputs, inputs_next = [], []
		inputs.append(obs)
		inputs_next.append(obs_next)

		# adds last action and agent number to obs
		if self.args.last_action:
		    if transition_idx == 0:  # if it is the first transition, let the previous action be a 0 vector
		        inputs.append(torch.zeros_like(actions_onehot[:, transition_idx, agent_id]))
		    else:
		        inputs.append(actions_onehot[:, transition_idx - 1, agent_id])
		    inputs_next.append(actions_onehot[:, transition_idx, agent_id])

		inputs = torch.cat([x.reshape(episode_num, -1) for x in inputs], dim=1)
		inputs_next = torch.cat([x.reshape(episode_num, -1) for x in inputs_next], dim=1)

		return inputs, inputs_next



	def init_hidden(self, episode_num):
		# initializes eval_hidden and target_hidden for each agent of each episode, as in DQN there is a net and a target net to stabilize learning

		self.eval_hidden = torch.zeros((episode_num, self.args.rnn_hidden_dim))
		self.target_hidden = torch.zeros((episode_num, self.args.rnn_hidden_dim))


	def save_model(self, train_step, agent_id, end_training=False):
		# save final policies at the end of training
		if end_training:
			torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + f'final_rnn_net_params_{agent_id}.pkl')
		else:
			num = str(train_step // self.args.save_cycle)
			if not os.path.exists(self.model_dir):
				os.makedirs(self.model_dir)
			torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + f'_rnn_net_params_{agent_id}.pkl')

