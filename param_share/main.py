
import gym
import ma_gym
from common.arguments import common_args, config_args
from runner import Runner
from smac.env import StarCraft2Env

import numpy as np
import torch
import random

#import warnings
#warnings.filterwarnings("ignore", category=UserWarning) 

N_EXPERIMENTS = 1

if __name__ == '__main__':
	args = common_args()

	seed = random.randrange(0, 2**32 - 1)
	print("Using seed: ", seed)
	torch.random.manual_seed(seed)
	np.random.seed(seed)
	

	if args.env in ["3m", "5m_vs_6m", "3s_vs_5z", "MMM2"]:
		env = StarCraft2Env(map_name=args.env)
		env_info = env.get_env_info()
		args.n_actions = env_info["n_actions"]
		args.n_agents = env_info["n_agents"]
		args.state_shape = env_info["state_shape"]
		args.obs_shape = env_info["obs_shape"]
		args.episode_limit = env_info["episode_limit"]
	elif args.env in ["PredatorPrey"]:
		# to avoid registering a whole new environment just do it here for now
		env = gym.make('PredatorPrey7x7-v0', grid_shape=(7, 7), n_agents=4, n_preys=2, penalty=-0.75)
		args.n_actions = env.action_space[0].n
		args.n_agents = env.n_agents
		args.state_shape = 28 * args.n_agents 
		args.obs_shape = 28
		args.episode_limit = env._max_steps
		print("Env PP with penalty ", env._penalty)
	else:
		raise Exception('Invalid environment: environment not supported!')
		

	print("Environment {} initialized, for {} time steps and evaluating every {} time steps".format(args.env, \
																							args.n_steps, args.evaluate_cycle))

	# load args
	if args.alg == "idql":
		args = config_args(args)
	else:
		raise Exception('No such algorithm!')


	print("CUDA set to", args.cuda)
	print("Communication set to", args.with_comm)
	print("With args:\n", args)

	runner = Runner(env, args)

	# parameterize run according to the number of independent experiments to run, i.e., independent sets of n_epochs over the model; default is 1
	if args.learn:
		runner.run(N_EXPERIMENTS)
