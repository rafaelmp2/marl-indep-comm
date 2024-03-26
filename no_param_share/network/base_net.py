import torch.nn as nn
import torch.nn.functional as F
from torch_dct import idct
import torch

'''
Because the RNN is used here, the last hidden_state is required each time. For an episode of data, each obs needs the last hidden_state to select the action.
Therefore, it is not possible to directly and randomly extract a batch of experience input to the neural network, so a batch of episodes is needed here, and the transition of the same position of this batch of episodes is passed in each time.
In this case, the hidden_state can be saved, and the next experience is the next experience
'''

class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # rnn gating mechanism; gated recurrent unit
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.input_shape = input_shape
        self.msg_dim = args.final_msg_dim

    def forward(self, obs, hidden_state, msgs=None, agent_num=None, msg_i=None):
        # if communicating
        if self.args.with_comm:
            ep_num = 1
            if agent_num == None:
                ep_num = obs.shape[0] // self.args.n_agents
            
            if agent_num != None:
                # if batch size = 1 (during execution): NOTE in the independent case we always have agent id, just on the centralised we dont
                # [1, input_dim] -> [1, input_dim + msg_dim]
                msgs = msgs.clone().detach()
                obs = torch.cat((obs, msgs.reshape(obs.shape[0], -1)), dim=-1)
                obs = torch.cat((obs, msg_i.reshape(obs.shape[0], -1)), dim=-1)

        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        return q, h


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
