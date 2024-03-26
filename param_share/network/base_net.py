import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, obs, hidden_state, msgs=None, agent_num=None):
        # if communicating        
        if self.args.with_comm:

            ep_num = 1
            if agent_num == None:
                ep_num = obs.shape[0] // self.args.n_agents
            
            msgs_rec = msgs

            # select the messages only from the other agetns, i.e., remove the ones of agent_num: [n_agents - 1, obs_dim]
            if agent_num != None:
                # if batch size = 1 (during execution)
                # TO RECEIVE ALL MSGS
                idxs = torch.tensor([i for i in range(self.args.n_agents) if i!=agent_num]).to("cuda" if self.args.cuda else "cpu")
                #idxs = torch.tensor([i for i in range(self.args.n_agents)]).to("cuda" if self.args.cuda else "cpu")
                msgs_rec = torch.index_select(msgs_rec, dim=1, index=idxs)
                # [1, input_dim] -> [1, input_dim + msg_dim]
                obs = torch.cat((obs, msgs_rec.reshape(obs.shape[0], -1)), dim=-1)
            else:
                # during training everything comes together (bs >= 1), so need another way to cat the respective messages to the right indices
                # i.e., all agents should only receive the messages from the others and not themselves (m_-i)
                a_mask = torch.eye(self.args.n_agents).reshape(self.args.n_agents, self.args.n_agents, 1)
                a_mask = torch.abs(a_mask - 1)
                if self.args.cuda:
                    a_mask = a_mask.cuda()
                
                # msg_rec: [bs, n_agents, msg_dim]
                msgs_rec_rep = msgs_rec.repeat(1, self.args.n_agents, 1).reshape(ep_num, self.args.n_agents, self.args.n_agents, -1)
                # NOTE CHANGED TO RECEIVE ALL MSGS
                msgs_repective_idxs = msgs_rec_rep * a_mask #* a_mask
                # [bs, n_a, (n_a - 1)*msg_dim] - m_-i: each agent receives the messages from the others
                msgs_repective_idxs_no_0 = msgs_repective_idxs[msgs_repective_idxs.count_nonzero(dim=-1) != 0].reshape(ep_num, self.args.n_agents, -1)
                msgs_rec = msgs_repective_idxs_no_0

                # cat messages to the inputs to the policy network
                # obs here is in shape [bs * n_a, input_dim]; need to change to [bs, n_a, input_dim]
                obs_aux = obs.reshape(ep_num, self.args.n_agents, -1)
                # now concat with msgs_rec and change to previous shape: [bs, n_a, input_dim+msg_dim] -> [bs*n_a, input_dim+msg_dim]
                obs = torch.cat((obs_aux, msgs_rec), dim=-1).reshape(ep_num * self.args.n_agents, -1)


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
