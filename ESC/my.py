import torch
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import os
import argparse
from scripts.arguments import add_arguments
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def build_net(layer_shape, activation, output_activation):
    '''build net with for loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)
class Q_Net(nn.Module):
    def __init__(self, args,action_dim,hid_shape):
        super(Q_Net, self).__init__()
        self.args=args
        self.s_net_hidden1 = nn.Sequential(
            nn.Linear(in_features=self.args.car_dim, out_features=hid_shape, bias=True),
            nn.ReLU())
        self.s_net_hidden2 = nn.Sequential(
            nn.Linear(in_features=hid_shape, out_features=1+self.args.car_dim*self.args.ESC_max_agent, bias=True),
            nn.ReLU())

        if self.args.use_ESC:
            layers = [1+self.args.car_dim+self.args.car_dim*self.args.ESC_max_agent] + [hid_shape] + [action_dim]
        else:
            layers = [(1+self.args.K)*self.args.car_dim] + [hid_shape] + [action_dim]
        self.Q = build_net(layers, nn.ReLU, nn.Identity)
    def forward(self, s):
        if self.args.use_ESC:
            self_state_length = self.args.car_dim
            num_states = (s.shape[1] - self_state_length) // self.args.car_dim
            self_states = s[:, :self_state_length]  # 提取自身车辆状态
            surrounding_states = s[:, self_state_length:].view(-1, num_states, self.args.car_dim)  # 提取周围车辆状态

            batch_size = s.shape[0]
            sum_outputs = torch.zeros(batch_size, 1+self.args.car_dim*self.args.ESC_max_agent)  # 初始化输出和的张量
            for i in range(batch_size):
                for j in range(num_states):
                    state = surrounding_states[i, j]
                    if -1 not in state:  # 检查状态是否有效（不包含 -1）
                        output = self.s_net_hidden1(state.unsqueeze(0))  # 应用 s_net_hidden1
                        output = self.s_net_hidden2(output)  # 应用 s_net_hidden2
                        sum_outputs[i] += output.squeeze(0)  # 累加输出

            # 拼接自身车辆状态与周围车辆状态的加和
            s = torch.cat((self_states, sum_outputs), dim=1)
        # s = s.reshape((-1, self.args.car_dim))
        o=self.Q(s)
        return o
argparser = argparse.ArgumentParser(description='CARLA CILQR')
add_arguments(argparser)
args = argparser.parse_args()
Q=Q_Net(args,2,256)
a=torch.tensor([[1,2,3,4,6,7,8,9,-1,-1,-1,-1],
               [1,2,3,4,6,7,8,9,-1,-1,-1,-1]], dtype=torch.float)
a1=torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,-1,-1,-1,-1], dtype=torch.float)
a2=torch.tensor([1,2,3,4], dtype=torch.float)
Q(a)

# a=a[a!=-1]
# print(a)
# a = a.reshape((a.shape[0],-1, 4))
# print(a)
# def process(s,args):
#     obs=torch.zeros((s.shape[0],args.car_dim*args.obs_num+1+args.car_dim), dtype=torch.float)
#     for index in range(s.shape[0]):
#         s_per=s[index]
#         s_per=s_per[s_per!=-1]
#         s_per=s_per.reshape((-1, 4))
#         s_per_ego=s_per[0]
#         s_per_sur=s_per[1:]
#         s_per_sur=Q(s_per_sur).sum(axis=0)
#         s_per=torch.hstack((s_per_ego,s_per_sur))
#         obs[index]=s_per
#     print(obs.shape)
#     return obs
#
# process(a,args)