import torch
import torch.nn as nn
import torch.nn.functional as F
#import sparse
#from numcompress import compress, decompress
import sys
import numpy as np
from torch_dct import dct, idct

def get_tensor_size(t: torch.tensor):
	return t.nelement() * t.element_size()

# input obs of all agents，output encoded message for each one of the agents
class Comm_net(nn.Module):
    def __init__(self, input_shape, args):
        super(Comm_net, self).__init__()
        self.fc1 = nn.Linear(input_shape, args.comm_net_dim)
        self.fc2 = nn.Linear(args.comm_net_dim, args.comm_net_dim)
        self.fc3 = nn.Linear(args.comm_net_dim, args.final_msg_dim)

        self.args = args
        self.input_shape = input_shape

        
    def forward(self, inputs):
        # inputs for now are the observations + last action of all agents that will be here used to generate the messages and then 
        # messages should be generated from a msg net, then they are cut and sent to the others rnn, where they are decoded to the original size
        # [b, e, n_a, o_dim] -> [b, e, n_a, m_dim]
        # note that the inputs come per transition

        ep_num = inputs.shape[0] // self.args.n_agents

        # simple fc net
        x1 = F.relu(self.fc1(inputs))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)

        m = x3

        final_msg = m.reshape(-1, self.args.n_agents, self.args.final_msg_dim) 

        return final_msg


