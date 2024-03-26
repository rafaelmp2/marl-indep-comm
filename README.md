# Fully Independent Communication in Multi-Agent Reinforcement Learning

This repository contains the source code used in the paper [Fully Independent Communication in Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2401.15059). <br>
The base codes were adapted from [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms) and the environments used are from [SMAC](https://github.com/oxwhirl/smac) and [ma-gym](https://github.com/koulanurag/ma-gym). <br>

### Directories
The folders contain the codes for the MARL framework both when sharing and not sharing parameters. The "no_param_share" folder contains the MARL framework when agents do not share parameters and the "param_share" folder contains the framework for when the agents share learning parameters. Inside each folder, it is implemented independent learning both with and without communication, under the respective parameter configuration. For details please refer to the paper.

### Running examples
* Running PredatorPrey with communication
```
python main.py --env PredatorPrey --n_steps 1000000 --alg idql --cuda True --with_comm True --rnn_hidden_dim 64
```
* Running a SMAC environment (3s_vs_5z in this case) without communication
```
python main.py --env 3s_vs_5z --n_steps 3000000 --alg idql --cuda True --cuda_device 0 --rnn_hidden_dim 32
```