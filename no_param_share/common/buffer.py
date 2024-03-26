import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0


        # create buffer to store info: arrays with shape [a,b,c]  TODO: change some parts of the buffer to work with > than 1D shapes as tuples (in shape) will trigger errors
        # buffer is a dic where the values for each key are arrays that store several episode_batches info linked to each other for each key
        self.buffers = {'obs': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'actions': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        'state': np.empty([self.size, self.episode_limit, self.state_shape]),  # state is global regardless the agent; this is the diff between state and observation
                        'reward': np.empty([self.size, self.episode_limit, 1]),
                        'obs_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'state_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'avail_actions': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_actions_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'actions_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),  # note: padded is to pad the episode buffer if the episode generated length is less than episode_limit
                        'terminated': np.empty([self.size, self.episode_limit, 1])
                        }

        self.lock = threading.Lock()



    def store_episode(self, episode_batch):
        batch_size = episode_batch['obs'].shape[0]  # get number of state episodes inside this batch
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)  # get buffer index to store inside 
            # store the informations
            self.buffers['obs'][idxs] = episode_batch['obs']
            self.buffers['actions'][idxs] = episode_batch['actions']
            self.buffers['state'][idxs] = episode_batch['state']
            self.buffers['reward'][idxs] = episode_batch['reward']
            self.buffers['obs_next'][idxs] = episode_batch['obs_next']
            self.buffers['state_next'][idxs] = episode_batch['state_next']
            self.buffers['avail_actions'][idxs] = episode_batch['avail_actions']
            self.buffers['avail_actions_next'][idxs] = episode_batch['avail_actions_next']
            self.buffers['actions_onehot'][idxs] = episode_batch['actions_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']


    def sample(self, batch_size):
        # samples from buffer ´batch_size´ episode batches
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer


    def _get_storage_idx(self, inc=None):
        # gets index to store self.size (buffer_size) in the buffer and not exceding 
        inc = inc or 1  # if 0 equals 1

        # checks if buffer has space to store the requested episode batch
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx