from agent.agent import Agents
from common.worker import RolloutWorker, RolloutWorker_SMAC
from common.buffer import ReplayBuffer
import os
import matplotlib.pyplot as plt
import numpy as np
import torch


class Runner:
	def __init__(self, env, args):
		self.env = env

		self.agents = Agents(args)

		if args.env.find('3m') > -1 or args.env.find('3s_vs_5z') > -1:
			self.rolloutWorker = RolloutWorker_SMAC(env, self.agents, args)
		else:
			# for ma_gym or others not smac (tbd)
			self.rolloutWorker = RolloutWorker(env, self.agents, args)

		self.buffer = ReplayBuffer(args)

		self.args = args

		self.save_path = self.args.result_dir + '/' + args.alg
		if not os.path.exists(self.save_path):
		    os.makedirs(self.save_path)


	def run(self, num):
		plt.figure()
		plt.axis([0, self.args.n_steps, 0, 100])
		win_rates = []
		episode_rewards = []
		train_steps = 0
		time_steps = 0
		evaluate_steps = -1

		while time_steps < self.args.n_steps:
			if time_steps // self.args.evaluate_cycle > evaluate_steps:
				print('Run {}, train step {}/{}'.format(num, time_steps, self.args.n_steps))
				win_rate, episode_reward = self.evaluate(epoch_num=time_steps)
				episode_rewards.append(episode_reward)
				win_rates.append(win_rate)

				plt.cla()
				plt.subplot(2, 1, 1)
				plt.plot(range(len(win_rates)), win_rates)
				plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
				plt.ylabel('win_rate')

				plt.subplot(2, 1, 2)
				plt.plot(range(len(episode_rewards)), episode_rewards)
				plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
				plt.ylabel('episode_rewards')

				plt.savefig(self.save_path + '/plt_{}_{}_{}ts.png'.format(num, self.args.env, self.args.n_steps), format='png')
				np.save(self.save_path + '/episode_rewards_{}_{}_{}ts'.format(num, self.args.env, self.args.n_steps), episode_rewards)
				np.save(self.save_path + '/win_rates_{}_{}_{}ts'.format(num, self.args.env, self.args.n_steps), win_rates)

				evaluate_steps += 1

			episodes = []

			for episode_idx in range(self.args.n_episodes):
				episode, _, _, info = self.rolloutWorker.generate_episode(episode_idx)  
				episodes.append(episode)
				time_steps += info['steps_taken']


			episode_batch = episodes[0]
			episodes.pop(0)

			# put observations of all the generated epsiodes together
			for episode in episodes:
			    for key in episode_batch.keys():
			        episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
			
			# again, coma doesnt need buffer, so wont store the episodes sampled
			if self.args.alg.find('coma') > -1:
				episode_batch['terminated'] = episode_batch['terminated'].astype(float)
				
				self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
				train_steps += 1
			else:
			    self.buffer.store_episode(episode_batch)
			    for train_step in range(self.args.train_steps):
			        mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
			        self.agents.train(mini_batch, train_steps)
			        train_steps += 1

		self.agents.policy.save_model(train_step, end_training=True)

		plt.cla()
		plt.subplot(2, 1, 1)
		plt.plot(range(len(win_rates)), win_rates)
		plt.xlabel('steps*{}'.format(self.args.evaluate_cycle))
		plt.ylabel('win_rate')

		plt.subplot(2, 1, 2)
		plt.plot(range(len(episode_rewards)), episode_rewards)
		plt.xlabel('steps*{}'.format(self.args.evaluate_cycle))
		plt.ylabel('episode_rewards')

		plt.savefig(self.save_path + '/plt_{}_{}_{}ts.png'.format(num, self.args.env, self.args.n_steps), format='png')
		np.save(self.save_path + '/episode_rewards_{}_{}_{}ts'.format(num, self.args.env, self.args.n_steps), episode_rewards)
		np.save(self.save_path + '/win_rates_{}_{}_{}ts'.format(num, self.args.env, self.args.n_steps), win_rates)

	def evaluate(self, epoch_num=None):
		win_counter = 0
		episode_rewards = 0
		steps_avrg = 0
		for epoch in range(self.args.evaluate_epoch):
			_, episode_reward, won, info = self.rolloutWorker.generate_episode(epoch, evaluate=True, epoch_num=epoch_num)
			episode_rewards += episode_reward
			steps_avrg += info['steps_taken']
			if won:  # if env ended in winning state 
				win_counter += 1

		return win_counter / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

